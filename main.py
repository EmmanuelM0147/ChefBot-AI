"""
ChefBot FastAPI service: retrieve recipe context, stream a Gemini-authored dish,
and append tool-calculated macro estimates (never model-invented nutrition).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, AsyncIterator, Iterator, Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from monitor import (
    init_monitoring_table,
    new_transaction_id,
    save_interaction_safe,
    update_feedback,
)
from retrieval import RecipeResult, search_recipes

load_dotenv()

GENERATION_MODEL = "gemini-2.5-flash"
APP_TITLE = "ChefBot AI"

# Streamlit Cloud frontend + local dev defaults.
DEFAULT_CORS_ORIGINS = (
    "https://chefbot-ai-9v272ty2jahksxappfpvhqg.streamlit.app,"
    "http://localhost:8501,"
    "http://127.0.0.1:8501"
)

SYSTEM_PROMPT = """
You are ChefBot - a Michelin-star chef who is obsessively allergen-safe and
inventory-disciplined.

Hard rules:
1. Use ONLY the recipes and details provided in the DATABASE CONTEXT block.
   Do not invent dishes, ingredients, steps, brands, or techniques that are not
   supported by that context.
2. Prefer adapting the highest-scoring / most relevant retrieved recipe to the
   user's inventory and dietary choices. If nothing fits safely, say so clearly
   and suggest the closest safe option from the context only.
3. Treat dietary choices and allergies as non-negotiable constraints. Call out
   allergen risks present in the chosen context recipe.
4. Structure the answer as:
   - Dish title
   - Why this fits the inventory / diet (brief)
   - Ingredients (quantities from context; note substitutions only if the
     substitute appears in the user's inventory or context)
   - Step-by-step method
   - Allergen & safety notes
5. Nutrition numbers are FORBIDDEN in your prose. You must call the
   `estimate_macros` tool exactly once with the final ingredient list used in
   the recipe. Do not invent, guess, or restate calorie/macro figures - the
   server appends the tool's calculated estimate after your text.
6. If the database context is empty, apologize and ask for different ingredients.
   Do not improvise a recipe from general knowledge.
""".strip()

# Deterministic per-item estimates (typical cooked edible portion / common unit).
# This table is the only nutrition source - never the LLM.
_MACRO_TABLE: dict[str, dict[str, float]] = {
    "chicken": {"calories": 165, "protein_g": 31.0, "carbs_g": 0.0, "fat_g": 3.6},
    "beef": {"calories": 250, "protein_g": 26.0, "carbs_g": 0.0, "fat_g": 17.0},
    "pork": {"calories": 242, "protein_g": 27.0, "carbs_g": 0.0, "fat_g": 14.0},
    "salmon": {"calories": 208, "protein_g": 20.0, "carbs_g": 0.0, "fat_g": 13.0},
    "tuna": {"calories": 132, "protein_g": 28.0, "carbs_g": 0.0, "fat_g": 1.3},
    "egg": {"calories": 78, "protein_g": 6.3, "carbs_g": 0.6, "fat_g": 5.3},
    "eggs": {"calories": 78, "protein_g": 6.3, "carbs_g": 0.6, "fat_g": 5.3},
    "milk": {"calories": 61, "protein_g": 3.2, "carbs_g": 4.8, "fat_g": 3.3},
    "butter": {"calories": 102, "protein_g": 0.1, "carbs_g": 0.0, "fat_g": 11.5},
    "cheese": {"calories": 113, "protein_g": 7.0, "carbs_g": 0.4, "fat_g": 9.0},
    "yogurt": {"calories": 59, "protein_g": 10.0, "carbs_g": 3.6, "fat_g": 0.4},
    "rice": {"calories": 206, "protein_g": 4.3, "carbs_g": 45.0, "fat_g": 0.4},
    "pasta": {"calories": 221, "protein_g": 8.1, "carbs_g": 43.0, "fat_g": 1.3},
    "bread": {"calories": 79, "protein_g": 2.7, "carbs_g": 15.0, "fat_g": 1.0},
    "potato": {"calories": 161, "protein_g": 4.3, "carbs_g": 37.0, "fat_g": 0.2},
    "potatoes": {"calories": 161, "protein_g": 4.3, "carbs_g": 37.0, "fat_g": 0.2},
    "tomato": {"calories": 22, "protein_g": 1.1, "carbs_g": 4.8, "fat_g": 0.2},
    "tomatoes": {"calories": 22, "protein_g": 1.1, "carbs_g": 4.8, "fat_g": 0.2},
    "onion": {"calories": 44, "protein_g": 1.2, "carbs_g": 10.0, "fat_g": 0.1},
    "garlic": {"calories": 4, "protein_g": 0.2, "carbs_g": 1.0, "fat_g": 0.0},
    "olive oil": {"calories": 119, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 13.5},
    "oil": {"calories": 120, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 13.6},
    "flour": {"calories": 110, "protein_g": 3.0, "carbs_g": 23.0, "fat_g": 0.3},
    "sugar": {"calories": 48, "protein_g": 0.0, "carbs_g": 12.5, "fat_g": 0.0},
    "salt": {"calories": 0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0},
    "pepper": {"calories": 6, "protein_g": 0.2, "carbs_g": 1.5, "fat_g": 0.1},
    "spinach": {"calories": 7, "protein_g": 0.9, "carbs_g": 1.1, "fat_g": 0.1},
    "broccoli": {"calories": 55, "protein_g": 3.7, "carbs_g": 11.0, "fat_g": 0.6},
    "carrot": {"calories": 25, "protein_g": 0.6, "carbs_g": 6.0, "fat_g": 0.1},
    "carrots": {"calories": 25, "protein_g": 0.6, "carbs_g": 6.0, "fat_g": 0.1},
    "beans": {"calories": 127, "protein_g": 8.7, "carbs_g": 22.8, "fat_g": 0.5},
    "lentils": {"calories": 116, "protein_g": 9.0, "carbs_g": 20.0, "fat_g": 0.4},
    "tofu": {"calories": 76, "protein_g": 8.0, "carbs_g": 1.9, "fat_g": 4.8},
    "avocado": {"calories": 160, "protein_g": 2.0, "carbs_g": 8.5, "fat_g": 14.7},
    "lemon": {"calories": 17, "protein_g": 0.6, "carbs_g": 5.4, "fat_g": 0.2},
    "lime": {"calories": 20, "protein_g": 0.5, "carbs_g": 7.0, "fat_g": 0.1},
    "ginger": {"calories": 5, "protein_g": 0.1, "carbs_g": 1.1, "fat_g": 0.0},
    "soy sauce": {"calories": 8, "protein_g": 1.3, "carbs_g": 0.8, "fat_g": 0.0},
    "cream": {"calories": 52, "protein_g": 0.4, "carbs_g": 0.4, "fat_g": 5.5},
    "mushroom": {"calories": 22, "protein_g": 3.1, "carbs_g": 3.3, "fat_g": 0.3},
    "mushrooms": {"calories": 22, "protein_g": 3.1, "carbs_g": 3.3, "fat_g": 0.3},
}


def estimate_macros(ingredients: list[str]) -> dict[str, Any]:
    """
    Calculate an estimated macro total from a recipe ingredient list.

    Uses a hardcoded lookup table only. Unmatched lines are reported, never
    fabricated with guessed nutrition values.
    """
    totals = {
        "calories": 0.0,
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fat_g": 0.0,
    }
    matched: list[dict[str, Any]] = []
    unmatched: list[str] = []

    for raw in ingredients or []:
        line = str(raw).strip()
        if not line:
            continue
        lowered = line.lower()
        hit_key: str | None = None
        # Prefer longer keys so "olive oil" wins over "oil".
        for key in sorted(_MACRO_TABLE.keys(), key=len, reverse=True):
            if key in lowered:
                hit_key = key
                break
        if hit_key is None:
            unmatched.append(line)
            continue
        macros = _MACRO_TABLE[hit_key]
        for metric, value in macros.items():
            totals[metric] += value
        matched.append({"ingredient": line, "matched_as": hit_key, **macros})

    return {
        "source": "hardcoded_estimate_macros_tool",
        "disclaimer": (
            "Approximate totals from a fixed nutrient table. "
            "Unmatched ingredients contribute 0 (not guessed)."
        ),
        "totals": {
            "calories": round(totals["calories"], 1),
            "protein_g": round(totals["protein_g"], 1),
            "carbs_g": round(totals["carbs_g"], 1),
            "fat_g": round(totals["fat_g"], 1),
        },
        "matched_items": matched,
        "unmatched_items": unmatched,
    }


def format_macro_appendix(result: dict[str, Any]) -> str:
    """Render tool output as response text - never LLM-authored nutrition."""
    totals = result.get("totals", {})
    unmatched = result.get("unmatched_items") or []
    lines = [
        "",
        "",
        "---",
        "## Estimated Macros (tool-calculated)",
        f"- Calories: {totals.get('calories', 0)} kcal",
        f"- Protein: {totals.get('protein_g', 0)} g",
        f"- Carbs: {totals.get('carbs_g', 0)} g",
        f"- Fat: {totals.get('fat_g', 0)} g",
        f"_Source: {result.get('source')} - {result.get('disclaimer')}_",
    ]
    if unmatched:
        preview = ", ".join(unmatched[:8])
        suffix = " …" if len(unmatched) > 8 else ""
        lines.append(f"_Unmatched (excluded from totals): {preview}{suffix}_")
    lines.append("")
    return "\n".join(lines)


class GenerateRecipeRequest(BaseModel):
    inventory: list[str] = Field(
        ...,
        min_length=1,
        description="Ingredients currently available to the user.",
    )
    dietary_choices: str = Field(
        default="",
        description="Dietary preferences, restrictions, or allergens.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Max retrieved recipe context documents.",
    )


class FeedbackRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1)
    feedback: Literal["thumbs_up", "thumbs_down"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_monitoring_table()
    yield


app = FastAPI(title=APP_TITLE, version="1.0.0", lifespan=lifespan)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip().strip('"')


def _cors_origins() -> list[str]:
    raw = _env("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for origin in origins:
        if origin not in seen:
            seen.add(origin)
            unique.append(origin)
    return unique


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transaction-Id"],
)

@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    api_key = _env("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        raise RuntimeError("GEMINI_API_KEY is not configured in .env")
    return genai.Client(api_key=api_key)


def format_database_context(recipes: list[RecipeResult]) -> str:
    if not recipes:
        return "DATABASE CONTEXT:\n(no matching recipes found)\n"

    blocks: list[str] = ["DATABASE CONTEXT:"]
    for index, recipe in enumerate(recipes, start=1):
        ingredients = recipe.get("ingredients") or []
        instructions = recipe.get("instructions") or []
        score = recipe.get("score")
        blocks.append(
            "\n".join(
                [
                    f"[Recipe {index}] title: {recipe.get('title')}",
                    f"relevance_score: {score}",
                    "ingredients:",
                    *[f"  - {item}" for item in ingredients],
                    "instructions:",
                    *[
                        f"  {step_i}. {step}"
                        for step_i, step in enumerate(instructions, start=1)
                    ],
                ]
            )
        )
    return "\n\n".join(blocks)


def build_user_prompt(
    inventory: list[str],
    dietary_choices: str,
    recipes: list[RecipeResult],
) -> str:
    inventory_text = ", ".join(item.strip() for item in inventory if item.strip())
    diet_text = dietary_choices.strip() or "none specified"
    return (
        f"USER INVENTORY:\n{inventory_text}\n\n"
        f"DIETARY CHOICES:\n{diet_text}\n\n"
        f"{format_database_context(recipes)}\n\n"
        "Create one finished recipe for the user following the system rules. "
        "When ready, call estimate_macros with the exact ingredient strings used."
    )


def _extract_function_calls(chunk: types.GenerateContentResponse) -> list[types.FunctionCall]:
    calls: list[types.FunctionCall] = []
    if not chunk.candidates:
        return calls
    for candidate in chunk.candidates:
        content = candidate.content
        if not content or not content.parts:
            continue
        for part in content.parts:
            if part.function_call and part.function_call.name:
                calls.append(part.function_call)
    return calls


def _ingredient_list_for_macros(
    function_calls: list[types.FunctionCall],
    recipes: list[RecipeResult],
    inventory: list[str],
) -> list[str]:
    for call in function_calls:
        if call.name != "estimate_macros":
            continue
        args = dict(call.args or {})
        ingredients = args.get("ingredients")
        if isinstance(ingredients, list) and ingredients:
            return [str(item) for item in ingredients]

    if recipes and recipes[0].get("ingredients"):
        return [str(item) for item in recipes[0]["ingredients"]]
    return [str(item) for item in inventory]


def stream_recipe_response(
    inventory: list[str],
    dietary_choices: str,
    recipes: list[RecipeResult],
) -> Iterator[str]:
    """
    Sync generator using client.models.generate_content_stream as required.

    Passes the hardcoded `estimate_macros` Python tool into Gemini function
    calling, then appends the tool's calculated macros to the streamed text so
    nutrition figures never come from model hallucination.
    """
    client = get_genai_client()
    prompt = build_user_prompt(inventory, dietary_choices, recipes)
    collected_calls: list[types.FunctionCall] = []
    yielded_text = False

    stream = client.models.generate_content_stream(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[estimate_macros],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                # Disable SDK auto-loop so we append macros ourselves and the
                # model cannot paraphrase/hallucinate nutrition after the tool.
                disable=True,
            ),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.AUTO,
                )
            ),
            temperature=0.4,
        ),
    )

    for chunk in stream:
        collected_calls.extend(_extract_function_calls(chunk))
        text = chunk.text
        if text:
            yielded_text = True
            yield text

    # If the model only emitted a function call, stream a tool-free narrative
    # pass so the client still receives a real-time recipe.
    if not yielded_text:
        narrative_stream = client.models.generate_content_stream(
            model=GENERATION_MODEL,
            contents=(
                f"{prompt}\n\n"
                "Write the full recipe now. Do not call tools and do not include "
                "any nutrition numbers."
            ),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
            ),
        )
        for chunk in narrative_stream:
            text = chunk.text
            if text:
                yield text

    tool_invoked = any(call.name == "estimate_macros" for call in collected_calls)
    if not tool_invoked:
        # Force Gemini function calling against the hardcoded Python tool.
        forced = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[
                types.Content(role="user", parts=[types.Part(text=prompt)]),
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "Call estimate_macros now with the final recipe "
                                "ingredient list. Do not write nutrition numbers."
                            )
                        )
                    ],
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[estimate_macros],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=["estimate_macros"],
                    )
                ),
                temperature=0.0,
            ),
        )
        collected_calls.extend(_extract_function_calls(forced))

    macro_ingredients = _ingredient_list_for_macros(
        collected_calls, recipes, inventory
    )
    # Execute locally and append - nutrition never comes from model prose.
    macro_result = estimate_macros(macro_ingredients)
    yield format_macro_appendix(macro_result)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": APP_TITLE}


@app.post("/api/generate-recipe")
async def generate_recipe(
    body: GenerateRecipeRequest,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    """
    Retrieve grounded recipe context, then stream a Gemini recipe response.
    Persists monitoring fields in PostgreSQL after the stream completes.
    """
    inventory = [item.strip() for item in body.inventory if item and item.strip()]
    if not inventory:
        raise HTTPException(status_code=422, detail="inventory must not be empty")

    user_query = ", ".join(inventory)
    transaction_id = new_transaction_id()
    started = time.perf_counter()

    try:
        recipes = await search_recipes(
            user_ingredients=inventory,
            dietary_preferences=body.dietary_choices,
            limit=body.limit,
        )
    except Exception as exc:  # noqa: BLE001 - surface retrieval failures cleanly
        background_tasks.add_task(
            save_interaction_safe,
            transaction_id=transaction_id,
            user_query=user_query,
            dietary_choices=body.dietary_choices,
            best_recipe_id=None,
            best_recipe_title=None,
            llm_output="",
            response_latency_ms=(time.perf_counter() - started) * 1000.0,
            model_name=GENERATION_MODEL,
            status="retrieval_error",
        )
        raise HTTPException(
            status_code=502,
            detail=f"Recipe retrieval failed: {exc}",
        ) from exc

    best = recipes[0] if recipes else None
    # Filled by the stream; BackgroundTasks runs after the response finishes.
    monitor_state: dict[str, Any] = {
        "llm_output": "",
        "latency_ms": None,
        "status": "ok",
    }

    async def event_stream() -> AsyncIterator[str]:
        chunks: list[str] = []
        try:
            iterator = stream_recipe_response(
                inventory=inventory,
                dietary_choices=body.dietary_choices,
                recipes=recipes,
            )
            for piece in iterator:
                chunks.append(piece)
                yield piece
            monitor_state["status"] = "ok"
        except Exception:
            monitor_state["status"] = "generation_error"
            raise
        finally:
            monitor_state["llm_output"] = "".join(chunks)
            monitor_state["latency_ms"] = (time.perf_counter() - started) * 1000.0

    def persist_interaction() -> None:
        save_interaction_safe(
            transaction_id=transaction_id,
            user_query=user_query,
            dietary_choices=body.dietary_choices,
            best_recipe_id=(best or {}).get("recipe_id"),
            best_recipe_title=(best or {}).get("title"),
            llm_output=str(monitor_state.get("llm_output") or ""),
            response_latency_ms=monitor_state.get("latency_ms"),
            model_name=GENERATION_MODEL,
            status=str(monitor_state.get("status") or "ok"),
        )

    background_tasks.add_task(persist_interaction)

    return StreamingResponse(
        event_stream(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Transaction-Id": transaction_id,
            "Access-Control-Expose-Headers": "X-Transaction-Id",
        },
    )


@app.post("/api/feedback")
async def submit_feedback(body: FeedbackRequest) -> dict[str, Any]:
    """Attach thumbs-up / thumbs-down to a previously logged interaction."""
    updated = update_feedback(body.transaction_id, body.feedback)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found or monitoring database unavailable.",
        )
    return {
        "status": "ok",
        "transaction_id": body.transaction_id,
        "feedback": body.feedback,
    }


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": APP_TITLE,
        "docs": "/docs",
        "generate_recipe": {
            "method": "POST",
            "path": "/api/generate-recipe",
            "body": {
                "inventory": ["chicken", "garlic", "tomato"],
                "dietary_choices": "high protein, dairy-free",
                "limit": 5,
            },
        },
        "feedback": {
            "method": "POST",
            "path": "/api/feedback",
            "body": {
                "transaction_id": "uuid",
                "feedback": "thumbs_up",
            },
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
