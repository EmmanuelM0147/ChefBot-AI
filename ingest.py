"""
Ingest ChefBot recipes into Qdrant with Gemini embedding vectors.

Reads the first 1,000 recipes (test limit), embeds in batches of 32, and
upserts PointStruct records into the `chefbot_recipes` collection.

Note: text-embedding-004 is no longer available on the Gemini API; we use
gemini-embedding-001 with output_dimensionality=768 (same vector size).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "chefbot_recipes"
EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_SIZE = 768
BATCH_SIZE = 32
MAX_RECIPES = 1_000
# Free-tier embed quota is ~100 requests/min (each text in a batch counts).
# Pace batches so we stay under the limit across the full 1,000-recipe run.
BATCH_PAUSE_SECONDS = 20
MAX_EMBED_RETRIES = 6

DATA_CANDIDATES = (
    Path("dataset/2_Recipe_json.json"),
    Path("dataset/recipes.json"),
)


def resolve_dataset_path() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    searched = ", ".join(str(p) for p in DATA_CANDIDATES)
    raise FileNotFoundError(f"No recipe dataset found. Looked for: {searched}")


def load_recipes(path: Path, limit: int = MAX_RECIPES) -> list[dict[str, Any]]:
    """Load recipes from JSONL (one object per line) or a JSON array."""
    print(f"Reading recipes from {path} (limit={limit})...")
    recipes: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        first_char = handle.read(1)
        while first_char and first_char.isspace():
            first_char = handle.read(1)
        if not first_char:
            raise ValueError(f"Dataset file is empty: {path}")

        handle.seek(0)
        if first_char == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError("Expected a JSON array of recipe objects.")
            recipes = payload[:limit]
        else:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    recipes.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {path}"
                    ) from exc
                if len(recipes) >= limit:
                    break

    print(f"Loaded {len(recipes)} recipes.")
    return recipes


def format_ingredients(ingredients: Any) -> str:
    if ingredients is None:
        return ""
    if isinstance(ingredients, list):
        return "; ".join(str(item).strip() for item in ingredients if str(item).strip())
    return str(ingredients).strip()


def build_context(recipe: dict[str, Any]) -> str:
    title = str(recipe.get("recipe_title") or "").strip()
    category = str(recipe.get("category") or "").strip()
    ingredients = format_ingredients(recipe.get("ingredients"))
    return (
        f"Title: {title}\n"
        f"Category: {category}\n"
        f"Ingredients: {ingredients}"
    )


def chunked(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def init_qdrant(url: str, api_key: str | None, *, recreate: bool = False) -> QdrantClient:
    client = QdrantClient(url=url, api_key=api_key or None)
    print(f"Connected to Qdrant at {url}")

    exists = client.collection_exists(COLLECTION_NAME)
    if exists and recreate:
        print(f"Collection '{COLLECTION_NAME}' already exists - recreating...")
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(
            f"Initialized collection '{COLLECTION_NAME}' "
            f"({VECTOR_SIZE} dims, Cosine similarity)."
        )
    else:
        count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        print(
            f"Reusing collection '{COLLECTION_NAME}' "
            f"({count} points already indexed - resume mode)."
        )
    return client


def existing_point_count(qdrant: QdrantClient) -> int:
    if not qdrant.collection_exists(COLLECTION_NAME):
        return 0
    return int(qdrant.count(collection_name=COLLECTION_NAME, exact=True).count)


def is_daily_quota_error(exc: Exception) -> bool:
    message = str(exc)
    return "PerDay" in message or "RequestsPerDay" in message


def embed_batch(genai_client: genai.Client, texts: list[str]) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        try:
            response = genai_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE),
            )
            break
        except genai_errors.ClientError as exc:
            last_error = exc
            if getattr(exc, "code", None) != 429 and "RESOURCE_EXHAUSTED" not in str(exc):
                raise
            if is_daily_quota_error(exc):
                print(
                    "  Daily embedding quota exhausted. "
                    "Progress is saved - re-run ingest.py later to resume."
                )
                raise
            # Prefer the API's retry hint when present; otherwise back off.
            wait_seconds = BATCH_PAUSE_SECONDS * attempt
            message = str(exc)
            marker = "Please retry in "
            if marker in message:
                try:
                    wait_seconds = max(
                        wait_seconds,
                        float(message.split(marker, 1)[1].split("s", 1)[0]) + 1,
                    )
                except ValueError:
                    pass
            print(
                f"  Rate limited (attempt {attempt}/{MAX_EMBED_RETRIES}). "
                f"Sleeping {wait_seconds:.0f}s..."
            )
            time.sleep(wait_seconds)
    else:
        assert last_error is not None
        raise last_error

    if not response.embeddings:
        raise RuntimeError("Embedding API returned no embeddings.")

    vectors: list[list[float]] = []
    for embedding in response.embeddings:
        if embedding.values is None:
            raise RuntimeError("Embedding API returned an empty vector.")
        if len(embedding.values) != VECTOR_SIZE:
            raise RuntimeError(
                f"Unexpected embedding size {len(embedding.values)}; "
                f"expected {VECTOR_SIZE}."
            )
        vectors.append(list(embedding.values))
    return vectors


def upsert_batch(
    qdrant: QdrantClient,
    recipes: list[dict[str, Any]],
    vectors: list[list[float]],
    start_id: int,
) -> None:
    points = [
        PointStruct(
            id=start_id + offset,
            vector=vector,
            payload={
                "recipe_title": recipe.get("recipe_title"),
                "category": recipe.get("category"),
                "subcategory": recipe.get("subcategory"),
                "description": recipe.get("description"),
                "ingredients": recipe.get("ingredients"),
                "directions": recipe.get("directions"),
                "num_ingredients": recipe.get("num_ingredients"),
                "num_steps": recipe.get("num_steps"),
            },
        )
        for offset, (recipe, vector) in enumerate(zip(recipes, vectors))
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def main() -> int:
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"')
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333").strip().strip('"')
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip().strip('"') or None

    if not gemini_api_key or gemini_api_key.startswith("your-"):
        print(
            "ERROR: Set a real GEMINI_API_KEY in .env before running ingestion.",
            file=sys.stderr,
        )
        return 1

    dataset_path = resolve_dataset_path()
    recipes = load_recipes(dataset_path, limit=MAX_RECIPES)
    if not recipes:
        print("ERROR: No recipes loaded.", file=sys.stderr)
        return 1

    print(f"Initializing Gemini client (model={EMBEDDING_MODEL})...")
    genai_client = genai.Client(api_key=gemini_api_key)
    recreate = "--recreate" in sys.argv
    qdrant = init_qdrant(qdrant_url, qdrant_api_key, recreate=recreate)

    already_indexed = 0 if recreate else existing_point_count(qdrant)
    if already_indexed >= len(recipes):
        print(
            f"Nothing to do - collection already has {already_indexed} points "
            f"(target {len(recipes)})."
        )
        return 0
    if already_indexed:
        print(f"Resuming after {already_indexed} previously indexed recipes...")
        recipes = recipes[already_indexed:]

    total_target = already_indexed + len(recipes)
    total_batches = (len(recipes) + BATCH_SIZE - 1) // BATCH_SIZE
    indexed = already_indexed

    print(
        f"Starting ingestion: {len(recipes)} remaining recipes "
        f"(IDs {already_indexed}→{total_target - 1}) in batches of {BATCH_SIZE} "
        f"({total_batches} batches)..."
    )

    try:
        for batch_index, batch in enumerate(chunked(recipes, BATCH_SIZE), start=1):
            contexts = [build_context(recipe) for recipe in batch]
            print(
                f"[{batch_index}/{total_batches}] Embedding recipes "
                f"{indexed + 1}-{indexed + len(batch)}..."
            )
            vectors = embed_batch(genai_client, contexts)

            print(
                f"[{batch_index}/{total_batches}] Upserting "
                f"{len(batch)} points into '{COLLECTION_NAME}'..."
            )
            upsert_batch(qdrant, batch, vectors, start_id=indexed)
            indexed += len(batch)
            print(
                f"[{batch_index}/{total_batches}] Progress: "
                f"{indexed}/{total_target} recipes indexed "
                f"({indexed / total_target:.0%})."
            )
            if batch_index < total_batches:
                print(
                    f"  Pausing {BATCH_PAUSE_SECONDS}s to respect free-tier rate limits..."
                )
                time.sleep(BATCH_PAUSE_SECONDS)
    except genai_errors.ClientError as exc:
        if is_daily_quota_error(exc):
            print(
                f"Stopped early with {indexed}/{total_target} points saved. "
                "Re-run tomorrow (or after quota reset) to finish."
            )
            return 2
        raise

    print(f"Done. Indexed {indexed} recipes into '{COLLECTION_NAME}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
