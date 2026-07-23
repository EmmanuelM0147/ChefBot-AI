"""
Async recipe retrieval over the ChefBot Qdrant collection.

Embeds a *rewritten* inventory + diet query with Gemini, runs cosine search
with an inventory text filter (hybrid), then *re-ranks* candidates by
ingredient overlap before returning top-k.

When Gemini embedding quota is exhausted, falls back to ingredient-filter-only
retrieval so the app remains usable on free tier.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchText,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
)

COLLECTION_NAME = "chefbot_recipes"
EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_SIZE = 768
MAX_EMBED_RETRIES = 3
QUERY_VECTOR_CACHE_MAX = 512
# Over-fetch factor before ingredient-overlap re-ranking.
RERANK_CANDIDATE_MULTIPLIER = 3

# Production default (see evaluate_retrieval.py / evals/retrieval_results.json).
RetrievalMode = Literal["hybrid", "vector_only", "filter_only"]
DEFAULT_RETRIEVAL_MODE: RetrievalMode = "hybrid"

# Light culinary expansions used by query rewriting (no extra LLM call).
_INGREDIENT_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "chicken": ("poultry", "roast chicken", "chicken breast"),
    "beef": ("ground beef", "steak", "braised beef"),
    "pork": ("pork chop", "pulled pork"),
    "salmon": ("fish", "seafood", "baked salmon"),
    "shrimp": ("prawns", "seafood", "garlic shrimp"),
    "tofu": ("soy", "plant protein", "stir fry"),
    "eggs": ("omelet", "frittata", "breakfast"),
    "egg": ("omelet", "frittata"),
    "pasta": ("noodles", "spaghetti", "italian"),
    "rice": ("fried rice", "grain bowl", "pilaf"),
    "tomato": ("tomatoes", "marinara", "tomato sauce"),
    "tomatoes": ("tomato", "marinara"),
    "garlic": ("aromatics", "garlic butter"),
    "lemon": ("citrus", "lemon zest"),
    "spinach": ("greens", "leafy vegetables"),
    "mushroom": ("mushrooms", "umami"),
    "mushrooms": ("mushroom", "umami"),
    "chickpeas": ("hummus", "legumes"),
    "lentils": ("legumes", "soup"),
    "potato": ("potatoes", "roast potatoes"),
    "potatoes": ("potato", "mash"),
}

# Bounded LRU: identical inventory/diet queries skip another embed call.
_QUERY_VECTOR_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_CACHE_LOCK = asyncio.Lock()
_INDEX_ENSURED = False


class RecipeResult(TypedDict):
    recipe_id: str | None
    title: str
    ingredients: list[Any]
    instructions: list[Any]
    score: float | None


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip().strip('"')


@lru_cache(maxsize=1)
def _genai_client() -> genai.Client:
    load_dotenv()
    api_key = _env("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        raise RuntimeError("Set GEMINI_API_KEY in .env before searching recipes.")
    return genai.Client(api_key=api_key)


@lru_cache(maxsize=1)
def get_qdrant_client() -> AsyncQdrantClient:
    """Process-wide AsyncQdrantClient (reuse across requests)."""
    load_dotenv()
    url = _env("QDRANT_URL", "http://localhost:6333")
    api_key = _env("QDRANT_API_KEY") or None
    return AsyncQdrantClient(url=url, api_key=api_key)


async def close_qdrant_client() -> None:
    """Close the shared client (call from app lifespan shutdown)."""
    if get_qdrant_client.cache_info().currsize == 0:
        return
    client = get_qdrant_client()
    await client.close()
    get_qdrant_client.cache_clear()


def clean_ingredients(user_ingredients: list) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in user_ingredients or []:
        if not isinstance(raw, str):
            continue
        item = raw.strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def build_query_context(
    user_ingredients: list[str],
    dietary_preferences: str,
) -> str:
    """Baseline (non-rewritten) inventory + diet query string."""
    cleaned_ingredients = clean_ingredients(user_ingredients)
    inventory = ", ".join(cleaned_ingredients) if cleaned_ingredients else "none listed"
    diet = (dietary_preferences or "").strip() or "none specified"
    return (
        f"Available ingredients: {inventory}\n"
        f"Dietary preferences: {diet}\n"
        f"Find recipes that can be cooked with these ingredients."
    )


def rewrite_query_for_retrieval(
    user_ingredients: list[str],
    dietary_preferences: str,
) -> str:
    """
    Rewrite a sparse fridge list into a retrieval-oriented natural-language query.

    Expands ingredients with light culinary synonyms and turns diet flags into
    explicit constraints so the embedding query carries clearer cooking intent.
    """
    ingredients = clean_ingredients(user_ingredients)
    diet = (dietary_preferences or "").strip()

    expansions: list[str] = []
    for item in ingredients:
        for alias in _INGREDIENT_EXPANSIONS.get(item.lower(), ()):
            if alias not in expansions and alias.lower() not in {
                i.lower() for i in ingredients
            }:
                expansions.append(alias)

    inventory = ", ".join(ingredients) if ingredients else "common pantry staples"
    expansion_text = ", ".join(expansions[:8]) if expansions else inventory
    diet_clause = (
        f"Hard dietary constraints: {diet}. Exclude recipes that violate these."
        if diet
        else "No special dietary constraints."
    )
    return (
        f"Find a practical home-cooked recipe that uses these fridge ingredients: "
        f"{inventory}. "
        f"Related culinary search terms: {expansion_text}. "
        f"{diet_clause} "
        f"Prefer recipes where many inventory ingredients appear together."
    )


def ingredient_overlap_score(
    user_ingredients: list[str],
    recipe: RecipeResult,
) -> float:
    """Fraction of inventory tokens found in the recipe ingredients text."""
    ingredients = clean_ingredients(user_ingredients)
    if not ingredients:
        return 0.0
    blob = " ".join(str(x).lower() for x in (recipe.get("ingredients") or []))
    hits = sum(1 for item in ingredients if item.lower() in blob)
    return hits / len(ingredients)


def rerank_by_ingredient_overlap(
    recipes: list[RecipeResult],
    user_ingredients: list[str],
    *,
    limit: int,
    vector_weight: float = 0.35,
    overlap_weight: float = 0.65,
) -> list[RecipeResult]:
    """
    Second-stage re-ranking: blend vector similarity with inventory overlap.

    Overlap is weighted higher so recipes covering more fridge items beat
    semantically close but poorly matched candidates.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if not recipes:
        return []

    vector_scores = [
        float(recipe["score"])
        for recipe in recipes
        if recipe.get("score") is not None
    ]
    if vector_scores:
        min_v = min(vector_scores)
        max_v = max(vector_scores)
        span = (max_v - min_v) or 1.0
    else:
        min_v, span = 0.0, 1.0

    ranked: list[tuple[float, RecipeResult]] = []
    for recipe in recipes:
        raw_vector = recipe.get("score")
        if raw_vector is None:
            vector_norm = 0.0
        else:
            vector_norm = (float(raw_vector) - min_v) / span
        overlap = ingredient_overlap_score(user_ingredients, recipe)
        combined = (vector_weight * vector_norm) + (overlap_weight * overlap)
        reranked = dict(recipe)
        reranked["score"] = round(combined, 6)
        ranked.append((combined, reranked))  # type: ignore[arg-type]

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [recipe for _, recipe in ranked[:limit]]


def build_ingredient_filter(user_ingredients: list[str]) -> Filter | None:
    """
    Match recipes whose `ingredients` payload array mentions ANY inventory item.

    Recipe lines look like "2 cups diced tomatoes", so we use MatchText (OR'd via
    `should`) instead of exact MatchAny equality.
    """
    conditions: list[FieldCondition] = []
    seen: set[str] = set()
    for raw in user_ingredients:
        if not isinstance(raw, str):
            continue
        ingredient = raw.strip().lower()
        if not ingredient or ingredient in seen:
            continue
        seen.add(ingredient)
        conditions.append(
            FieldCondition(
                key="ingredients",
                match=MatchText(text=ingredient),
            )
        )

    if not conditions:
        return None
    return Filter(should=conditions)


async def _ensure_ingredients_text_index(qdrant: AsyncQdrantClient) -> None:
    """Create a text index on ingredients so MatchText payload filters work."""
    global _INDEX_ENSURED
    if _INDEX_ENSURED:
        return
    try:
        await qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="ingredients",
            field_schema=TextIndexParams(
                type=PayloadSchemaType.TEXT,
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=40,
                lowercase=True,
            ),
            wait=True,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "already exists" not in message and "duplicate" not in message:
            raise
    _INDEX_ENSURED = True


def is_quota_error(exc: BaseException) -> bool:
    text = str(exc)
    return (
        isinstance(exc, genai_errors.ClientError)
        and (getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in text)
    ) or "RESOURCE_EXHAUSTED" in text or "429" in text


def is_daily_quota(exc: BaseException) -> bool:
    text = str(exc)
    if "PerDay" not in text and "RequestsPerDay" not in text:
        return False
    # Some free-tier errors label the metric *PerDay* but still return a short
    # retryDelay (tens of seconds). Treat those as transient rate limits.
    match = re.search(r"Please retry in\s+([0-9.]+)\s*s", text, flags=re.IGNORECASE)
    if match:
        try:
            if float(match.group(1)) <= 120:
                return False
        except ValueError:
            pass
    return True


def quota_retry_wait_seconds(exc: BaseException, attempt: int) -> float:
    """Exponential backoff, preferring Gemini's 'Please retry in Xs' hint."""
    wait_seconds = float(2**attempt)
    message = str(exc)
    match = re.search(r"Please retry in\s+([0-9.]+)\s*s", message, flags=re.IGNORECASE)
    if match:
        try:
            wait_seconds = max(wait_seconds, float(match.group(1)) + 1.0)
        except ValueError:
            pass
    return min(wait_seconds, 60.0)


def _cache_key(query_text: str) -> str:
    return hashlib.sha256(query_text.encode("utf-8")).hexdigest()


async def embed_query(query_text: str) -> list[float]:
    key = _cache_key(query_text)
    async with _CACHE_LOCK:
        cached = _QUERY_VECTOR_CACHE.get(key)
        if cached is not None:
            _QUERY_VECTOR_CACHE.move_to_end(key)
            return cached

    client = _genai_client()
    last_error: BaseException | None = None

    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        try:
            response = await client.aio.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=query_text,
                config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE),
            )
            if not response.embeddings or response.embeddings[0].values is None:
                raise RuntimeError("Embedding API returned no vector for the search query.")
            vector = list(response.embeddings[0].values)
            async with _CACHE_LOCK:
                _QUERY_VECTOR_CACHE[key] = vector
                _QUERY_VECTOR_CACHE.move_to_end(key)
                while len(_QUERY_VECTOR_CACHE) > QUERY_VECTOR_CACHE_MAX:
                    _QUERY_VECTOR_CACHE.popitem(last=False)
            return vector
        except Exception as exc:  # noqa: BLE001 - classify quota vs hard failures
            last_error = exc
            if not is_quota_error(exc):
                raise
            if is_daily_quota(exc):
                raise
            await asyncio.sleep(quota_retry_wait_seconds(exc, attempt))

    assert last_error is not None
    raise last_error


async def _filter_only_search(
    qdrant: AsyncQdrantClient,
    query_filter: Filter | None,
    limit: int,
) -> list[RecipeResult]:
    """Fallback when embeddings are unavailable: inventory text filter only."""
    points, _ = await qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    results: list[RecipeResult] = []
    for point in points:
        hit = _format_hit(point)
        hit["score"] = None
        results.append(hit)
    return results


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _format_hit(point: Any) -> RecipeResult:
    payload = point.payload or {}
    point_id = getattr(point, "id", None)
    return {
        "recipe_id": str(point_id) if point_id is not None else None,
        "title": str(payload.get("recipe_title") or ""),
        "ingredients": _as_list(payload.get("ingredients")),
        "instructions": _as_list(payload.get("directions")),
        "score": float(point.score) if getattr(point, "score", None) is not None else None,
    }


async def search_recipes_with_mode(
    user_ingredients: list,
    dietary_preferences: str,
    *,
    limit: int = 5,
    mode: RetrievalMode = DEFAULT_RETRIEVAL_MODE,
    allow_embed_fallback: bool = False,
    rewrite_query: bool = True,
    rerank: bool = True,
) -> list[RecipeResult]:
    """
    Run one explicit retrieval strategy (used by production + offline eval).

    Modes:
    - hybrid: cosine rank over recipes that mention ≥1 inventory ingredient
    - vector_only: cosine rank with no inventory filter
    - filter_only: inventory MatchText filter, no vector ranking

    Production extras (on by default):
    - rewrite_query: expand inventory/diet into a richer retrieval query
    - rerank: over-fetch candidates, then re-rank by inventory overlap
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if mode not in ("hybrid", "vector_only", "filter_only"):
        raise ValueError(f"Unknown retrieval mode: {mode}")

    ingredients = clean_ingredients(list(user_ingredients or []))
    if rewrite_query:
        query_text = rewrite_query_for_retrieval(ingredients, dietary_preferences)
    else:
        query_text = build_query_context(ingredients, dietary_preferences)
    query_filter = build_ingredient_filter(ingredients)

    fetch_limit = limit
    if rerank:
        fetch_limit = max(limit, limit * RERANK_CANDIDATE_MULTIPLIER)

    qdrant = get_qdrant_client()
    if query_filter is not None and mode in ("hybrid", "filter_only"):
        await _ensure_ingredients_text_index(qdrant)

    if mode == "filter_only":
        results = await _filter_only_search(qdrant, query_filter, fetch_limit)
        if rerank:
            return rerank_by_ingredient_overlap(results, ingredients, limit=limit)
        return results[:limit]

    try:
        query_vector = await embed_query(query_text)
    except Exception as exc:
        if allow_embed_fallback and is_quota_error(exc):
            results = await _filter_only_search(qdrant, query_filter, fetch_limit)
            if rerank:
                return rerank_by_ingredient_overlap(results, ingredients, limit=limit)
            return results[:limit]
        raise

    response = await qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter if mode == "hybrid" else None,
        limit=fetch_limit,
        with_payload=True,
    )
    results = [_format_hit(point) for point in response.points]
    if rerank:
        return rerank_by_ingredient_overlap(results, ingredients, limit=limit)
    return results[:limit]


async def search_recipes(
    user_ingredients: list,
    dietary_preferences: str,
    limit: int = 5,
) -> list[RecipeResult]:
    """
    Production search: hybrid retrieval + query rewrite + overlap re-rank.

    If Gemini embedding quota is exhausted, degrade gracefully to filter-only
    so generation can still proceed from inventory matches.
    """
    return await search_recipes_with_mode(
        user_ingredients,
        dietary_preferences,
        limit=limit,
        mode=DEFAULT_RETRIEVAL_MODE,
        allow_embed_fallback=True,
        rewrite_query=True,
        rerank=True,
    )


async def main() -> None:
    """Smoke-test helper: python -m retrieval  (or: python retrieval.py)"""
    import json

    sample_inventory = ["chicken", "garlic", "tomato"]
    sample_diet = "high protein, low carb"
    try:
        matches = await search_recipes(sample_inventory, sample_diet, limit=5)
        print(json.dumps(matches, indent=2, ensure_ascii=False))
    finally:
        await close_qdrant_client()


if __name__ == "__main__":
    asyncio.run(main())
