"""
Offline retrieval evaluation for ChefBot (Zoomcamp-style).

Compares multiple retrieval approaches on a fixed inventory query set and
reports Hit@k, MRR, Precision@k, and mean ingredient coverage.

Relevance proxy (no hand-labeled doc IDs required):
  A retrieved recipe is relevant if at least `min_ingredient_hits` inventory
  tokens appear in its ingredients text (case-insensitive substring match).

Production uses the winning mode from this script (hybrid) — see README.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from retrieval import (
    RetrievalMode,
    RecipeResult,
    close_qdrant_client,
    search_recipes_with_mode,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("chefbot.evaluate_retrieval")

MODES: tuple[RetrievalMode, ...] = ("hybrid", "vector_only", "filter_only")
DEFAULT_QUERIES = Path("evals/retrieval_queries.json")
DEFAULT_OUT = Path("evals/retrieval_results.json")
DEFAULT_K = 5
DEFAULT_MIN_HITS = 2


@dataclass(frozen=True)
class EvalQuery:
    id: str
    ingredients: list[str]
    diet: str
    notes: str = ""


def load_queries(path: Path) -> list[EvalQuery]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected a non-empty JSON array in {path}")
    queries: list[EvalQuery] = []
    for item in raw:
        ingredients = [
            str(x).strip()
            for x in (item.get("ingredients") or [])
            if str(x).strip()
        ]
        if not ingredients:
            raise ValueError(f"Query {item.get('id')!r} has no ingredients")
        queries.append(
            EvalQuery(
                id=str(item.get("id") or f"q{len(queries)+1}"),
                ingredients=ingredients,
                diet=str(item.get("diet") or "").strip(),
                notes=str(item.get("notes") or "").strip(),
            )
        )
    return queries


def _ingredients_blob(recipe: RecipeResult) -> str:
    parts = [str(x).lower() for x in (recipe.get("ingredients") or [])]
    return " ".join(parts)


def inventory_hits(ingredients: list[str], recipe: RecipeResult) -> int:
    blob = _ingredients_blob(recipe)
    hits = 0
    for token in ingredients:
        needle = token.strip().lower()
        if needle and needle in blob:
            hits += 1
    return hits


def is_relevant(
    ingredients: list[str],
    recipe: RecipeResult,
    *,
    min_hits: int,
) -> bool:
    needed = min(min_hits, len(ingredients))
    return inventory_hits(ingredients, recipe) >= needed


def hit_at_k(relevances: list[bool]) -> float:
    return 1.0 if any(relevances) else 0.0


def mrr(relevances: list[bool]) -> float:
    for index, relevant in enumerate(relevances, start=1):
        if relevant:
            return 1.0 / index
    return 0.0


def precision_at_k(relevances: list[bool]) -> float:
    if not relevances:
        return 0.0
    return sum(1 for r in relevances if r) / len(relevances)


def coverage_at_k(ingredients: list[str], recipes: list[RecipeResult]) -> float:
    """Fraction of inventory tokens found in ANY of the top-k recipes."""
    if not ingredients:
        return 0.0
    found = 0
    for token in ingredients:
        needle = token.strip().lower()
        if not needle:
            continue
        if any(needle in _ingredients_blob(recipe) for recipe in recipes):
            found += 1
    return found / len(ingredients)


def summarize_mode(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "hit_rate": round(mean(r["hit"] for r in rows), 4),
        "mrr": round(mean(r["mrr"] for r in rows), 4),
        "precision_at_k": round(mean(r["precision"] for r in rows), 4),
        "avg_coverage": round(mean(r["coverage"] for r in rows), 4),
        "avg_retrieved": round(mean(r["retrieved"] for r in rows), 2),
    }


def pick_winner(summaries: dict[str, dict[str, float]]) -> str:
    """
    Prefer higher Hit@k, then MRR, then Precision@k, then coverage.
    Ties break toward hybrid (production-safe inventory constraint).
    """
    ranked = sorted(
        summaries.items(),
        key=lambda item: (
            item[1]["hit_rate"],
            item[1]["mrr"],
            item[1]["precision_at_k"],
            item[1]["avg_coverage"],
            1 if item[0] == "hybrid" else 0,
        ),
        reverse=True,
    )
    return ranked[0][0]


async def evaluate_mode(
    mode: RetrievalMode,
    queries: list[EvalQuery],
    *,
    k: int,
    min_hits: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, query in enumerate(queries, start=1):
        print(
            f"  [{mode}] {index}/{len(queries)} {query.id} "
            f"({', '.join(query.ingredients)})"
        )
        try:
            recipes = await search_recipes_with_mode(
                query.ingredients,
                query.diet,
                limit=k,
                mode=mode,
                allow_embed_fallback=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("  %s failed on %s: %s", mode, query.id, exc)
            rows.append(
                {
                    "query_id": query.id,
                    "mode": mode,
                    "retrieved": 0,
                    "hit": 0.0,
                    "mrr": 0.0,
                    "precision": 0.0,
                    "coverage": 0.0,
                    "error": str(exc),
                    "titles": [],
                }
            )
            continue

        relevances = [
            is_relevant(query.ingredients, recipe, min_hits=min_hits)
            for recipe in recipes
        ]
        rows.append(
            {
                "query_id": query.id,
                "mode": mode,
                "retrieved": len(recipes),
                "hit": hit_at_k(relevances),
                "mrr": mrr(relevances),
                "precision": precision_at_k(relevances),
                "coverage": coverage_at_k(query.ingredients, recipes),
                "titles": [r.get("title") or "" for r in recipes],
                "relevance_flags": relevances,
            }
        )
    return rows


async def run_eval(
    queries: list[EvalQuery],
    *,
    k: int,
    min_hits: int,
    modes: tuple[RetrievalMode, ...] = MODES,
) -> dict[str, Any]:
    per_mode_rows: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, float]] = {}

    for mode in modes:
        print(f"\n=== Mode: {mode} ===")
        rows = await evaluate_mode(mode, queries, k=k, min_hits=min_hits)
        per_mode_rows[mode] = rows
        summaries[mode] = summarize_mode(rows)

    winner = pick_winner(summaries)
    return {
        "k": k,
        "min_ingredient_hits": min_hits,
        "n_queries": len(queries),
        "relevance_definition": (
            f"Recipe is relevant if >= {min_hits} inventory ingredient "
            "token(s) appear in its ingredients text "
            "(case-insensitive substring)."
        ),
        "summaries": summaries,
        "winner": winner,
        "production_mode": "hybrid",
        "details": per_mode_rows,
    }


def print_report(report: dict[str, Any]) -> None:
    print("\n=== Retrieval evaluation summary ===")
    print(f"Queries: {report['n_queries']}  |  k={report['k']}  |  "
          f"min_hits={report['min_ingredient_hits']}")
    print(f"Relevance: {report['relevance_definition']}")
    print()
    header = f"{'mode':<14} {'Hit@k':>8} {'MRR':>8} {'P@k':>8} {'Cover':>8}"
    print(header)
    print("-" * len(header))
    for mode, stats in report["summaries"].items():
        marker = " *" if mode == report["winner"] else "  "
        print(
            f"{mode:<14}{marker}"
            f"{stats['hit_rate']:>7.3f} "
            f"{stats['mrr']:>8.3f} "
            f"{stats['precision_at_k']:>8.3f} "
            f"{stats['avg_coverage']:>8.3f}"
        )
    print()
    print(f"Winner: {report['winner']}")
    print(
        f"Production uses: {report['production_mode']} "
        "(inventory-constrained semantic rank; filter-only remains embed fallback)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare ChefBot retrieval approaches (Hit@k / MRR / P@k)"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
        help=f"Gold query JSON (default: {DEFAULT_QUERIES})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Write full JSON report (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Top-k cutoff (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=DEFAULT_MIN_HITS,
        help=(
            "Min inventory ingredients that must appear in a recipe "
            f"for it to count as relevant (default: {DEFAULT_MIN_HITS})"
        ),
    )
    args = parser.parse_args(argv)

    if args.k < 1:
        logger.error("--k must be >= 1")
        return 1
    if args.min_hits < 1:
        logger.error("--min-hits must be >= 1")
        return 1
    if not args.queries.exists():
        logger.error("Query file not found: %s", args.queries)
        return 1

    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} retrieval queries from {args.queries}")

    try:
        report = asyncio.run(
            run_eval(queries, k=args.k, min_hits=args.min_hits)
        )
    finally:
        try:
            asyncio.run(close_qdrant_client())
        except Exception:  # noqa: BLE001
            pass

    print_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Drop per-query title lists from a slim copy? Keep full details for reviewers.
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
