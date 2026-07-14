"""
Async recipe retrieval over the ChefBot Qdrant collection.

Embeds the caller's inventory + dietary preferences with Gemini, then runs a
cosine similarity search with a payload filter so results must mention at least
one inventory ingredient.

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
from typing import Any, TypedDict

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


def build_query_context(
    user_ingredients: list[str],
    dietary_preferences: str,
) -> str:
    """Combine inventory + diet into one embedding-friendly query string."""
    cleaned_ingredients = [
        ingredient.strip()
        for ingredient in user_ingredients
        if isinstance(ingredient, str) and ingredient.strip()
    ]
    inventory = ", ".join(cleaned_ingredients) if cleaned_ingredients else "none listed"
    diet = (dietary_preferences or "").strip() or "none specified"
    return (
        f"Available ingredients: {inventory}\n"
        f"Dietary preferences: {diet}\n"
        f"Find recipes that can be cooked with these ingredients."
    )


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
    return "PerDay" in text or "RequestsPerDay" in text


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


async def search_recipes(
    user_ingredients: list,
    dietary_preferences: str,
    limit: int = 5,
) -> list[RecipeResult]:
    """
    Semantic recipe search constrained to the caller's inventory.

    1. Embed ingredients + dietary preferences as one query context.
    2. Filter Qdrant payloads so at least one inventory ingredient appears in
       the recipe `ingredients` array.
    3. Rank remaining candidates by cosine similarity and return top `limit`.

    If Gemini embedding quota is exhausted, degrade gracefully to filter-only
    retrieval so generation can still proceed from inventory matches.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")

    ingredients = list(user_ingredients or [])
    query_text = build_query_context(ingredients, dietary_preferences)
    query_filter = build_ingredient_filter(ingredients)

    qdrant = get_qdrant_client()
    if query_filter is not None:
        await _ensure_ingredients_text_index(qdrant)

    try:
        query_vector = await embed_query(query_text)
    except Exception as exc:
        if not is_quota_error(exc):
            raise
        # Free-tier embed quota exhausted - still return usable context.
        return await _filter_only_search(qdrant, query_filter, limit)

    results = await qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return [_format_hit(point) for point in results.points]


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
