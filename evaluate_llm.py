"""
Offline LLM approach A/B evaluation for ChefBot (Zoomcamp-style).

For each inventory query:
  1. Retrieve hybrid context once (shared across approaches).
  2. Generate a recipe with each prompt approach (no tools — prose only).
  3. Score each answer with the same Gemini LLM-as-judge.

Writes evals/llm_results.json and prints the winning approach.
Production uses prompts.DEFAULT_APPROACH (keep in sync with the winner).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from evaluate import judge_interaction
from evaluate_retrieval import EvalQuery, load_queries
from prompts import APPROACHES, DEFAULT_APPROACH, get_approach
from retrieval import (
    RecipeResult,
    close_qdrant_client,
    is_daily_quota,
    is_quota_error,
    quota_retry_wait_seconds,
    search_recipes,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("chefbot.evaluate_llm")

GENERATION_MODEL = "gemini-2.5-flash"
DEFAULT_QUERIES = Path("evals/retrieval_queries.json")
DEFAULT_OUT = Path("evals/llm_results.json")
DEFAULT_LIMIT = 4
DEFAULT_PAUSE = 2.0
MAX_GEN_RETRIES = 4


def _parse_inventory(query: EvalQuery) -> tuple[list[str], str]:
    return list(query.ingredients), query.diet


def _format_database_context(recipes: list[RecipeResult]) -> str:
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


def build_eval_user_prompt(
    inventory: list[str],
    dietary_choices: str,
    recipes: list[RecipeResult],
) -> str:
    inventory_text = ", ".join(item.strip() for item in inventory if item.strip())
    diet_text = dietary_choices.strip() or "none specified"
    return (
        f"USER INVENTORY:\n{inventory_text}\n\n"
        f"DIETARY CHOICES:\n{diet_text}\n\n"
        f"{_format_database_context(recipes)}\n\n"
        "Create one finished recipe for the user following the system rules. "
        "Do not include nutrition numbers."
    )

def generate_answer(
    client: genai.Client,
    *,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Single-shot generation without tools (fair prompt A/B on prose quality)."""
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.4,
    )
    last_error: BaseException | None = None
    for attempt in range(1, MAX_GEN_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=user_prompt,
                config=config,
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("Model returned an empty recipe.")
            return text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if not is_quota_error(exc) or is_daily_quota(exc):
                raise
            wait = quota_retry_wait_seconds(exc, attempt)
            logger.warning(
                "Generation rate-limited (attempt %s/%s); sleeping %.1fs",
                attempt,
                MAX_GEN_RETRIES,
                wait,
            )
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def summarize_approach(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("overall_score") is not None]

    def avg(key: str) -> float | None:
        values = [r[key] for r in scored if r.get(key) is not None]
        return round(mean(values), 3) if values else None

    verdicts: dict[str, int] = {}
    for row in scored:
        v = str(row.get("verdict") or "unknown")
        verdicts[v] = verdicts.get(v, 0) + 1

    return {
        "n_scored": len(scored),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "avg_relevance": avg("relevance_score"),
        "avg_groundedness": avg("groundedness_score"),
        "avg_safety": avg("safety_score"),
        "avg_overall": avg("overall_score"),
        "verdicts": verdicts,
    }


def pick_winner(summaries: dict[str, dict[str, Any]]) -> str:
    """Higher overall, then groundedness, then relevance; tie → production default."""

    def key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, int]:
        name, stats = item
        return (
            float(stats.get("avg_overall") or 0.0),
            float(stats.get("avg_groundedness") or 0.0),
            float(stats.get("avg_relevance") or 0.0),
            1 if name == DEFAULT_APPROACH else 0,
        )

    ranked = sorted(summaries.items(), key=key, reverse=True)
    return ranked[0][0]


async def run_ab(
    client: genai.Client,
    queries: list[EvalQuery],
    approach_names: list[str],
    *,
    pause: float,
    top_k: int,
) -> dict[str, Any]:
    details: dict[str, list[dict[str, Any]]] = {name: [] for name in approach_names}

    for q_index, query in enumerate(queries, start=1):
        inventory, diet = _parse_inventory(query)
        print(
            f"\n[{q_index}/{len(queries)}] {query.id}: "
            f"{', '.join(inventory)} | diet={diet or 'none'}"
        )

        try:
            recipes = await search_recipes(inventory, diet, limit=top_k)
        except Exception as exc:  # noqa: BLE001
            logger.error("Retrieval failed for %s: %s", query.id, exc)
            for name in approach_names:
                details[name].append(
                    {
                        "query_id": query.id,
                        "approach": name,
                        "error": f"retrieval: {exc}",
                    }
                )
            continue

        user_prompt = build_eval_user_prompt(inventory, diet, recipes)

        for a_index, name in enumerate(approach_names):
            approach = get_approach(name)
            print(f"  Generating with approach={name} ...")
            try:
                # Offline A/B has no tool loop; neutralize macros-tool instructions.
                system_prompt = (
                    approach["system_prompt"]
                    + "\n\nNOTE: Tools are unavailable in this offline evaluation. "
                    "Omit nutrition numbers entirely."
                )
                answer = generate_answer(
                    client,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                judge_row = {
                    "user_query": ", ".join(inventory),
                    "dietary_choices": diet,
                    "retrieved_titles": [
                        r.get("title") for r in recipes if r.get("title")
                    ],
                    "assistant_answer": answer,
                }
                print(f"  Judging approach={name} ...")
                scores = judge_interaction(client, judge_row)
                details[name].append(
                    {
                        "query_id": query.id,
                        "approach": name,
                        "answer_preview": answer[:400],
                        "answer_chars": len(answer),
                        "relevance_score": scores["relevance_score"],
                        "groundedness_score": scores["groundedness_score"],
                        "safety_score": scores["safety_score"],
                        "overall_score": scores["overall_score"],
                        "verdict": scores["verdict"],
                        "rationale": scores["rationale"],
                    }
                )
                print(
                    f"    -> overall={scores['overall_score']} "
                    f"verdict={scores['verdict']}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Approach %s failed on %s: %s", name, query.id, exc)
                details[name].append(
                    {
                        "query_id": query.id,
                        "approach": name,
                        "error": str(exc),
                    }
                )

            # Pause between Gemini calls (generate + judge already two calls).
            if pause > 0 and not (
                q_index == len(queries) and a_index == len(approach_names) - 1
            ):
                time.sleep(pause)

    summaries = {
        name: summarize_approach(rows) for name, rows in details.items()
    }
    winner = pick_winner(summaries)
    return {
        "model": GENERATION_MODEL,
        "n_queries": len(queries),
        "approaches": {
            name: {
                "description": APPROACHES[name]["description"],
            }
            for name in approach_names
        },
        "summaries": summaries,
        "winner": winner,
        "production_approach": DEFAULT_APPROACH,
        "details": details,
    }


def print_report(report: dict[str, Any]) -> None:
    print("\n=== LLM approach A/B summary ===")
    print(f"Queries: {report['n_queries']}  |  model={report['model']}")
    print()
    header = (
        f"{'approach':<22} {'overall':>8} {'ground':>8} "
        f"{'relev':>8} {'safety':>8} {'n':>4}"
    )
    print(header)
    print("-" * len(header))
    for name, stats in report["summaries"].items():
        marker = " *" if name == report["winner"] else "  "
        print(
            f"{name:<22}{marker}"
            f"{(stats.get('avg_overall') or 0):>7.3f} "
            f"{(stats.get('avg_groundedness') or 0):>8.3f} "
            f"{(stats.get('avg_relevance') or 0):>8.3f} "
            f"{(stats.get('avg_safety') or 0):>8.3f} "
            f"{stats.get('n_scored', 0):>4}"
        )
    print()
    print(f"Winner: {report['winner']}")
    print(f"Production DEFAULT_APPROACH: {report['production_approach']}")
    if report["winner"] != report["production_approach"]:
        print(
            "NOTE: Update prompts.DEFAULT_APPROACH to the winner "
            "before deploying."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A/B evaluate ChefBot prompt approaches with LLM-as-judge"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
        help=f"Query JSON (default: {DEFAULT_QUERIES})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Write JSON report (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max queries to run (default: {DEFAULT_LIMIT}; keep small on free tier)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=DEFAULT_PAUSE,
        help=f"Seconds between Gemini calls (default: {DEFAULT_PAUSE})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retrieved recipes shared across approaches (default: 5)",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=list(APPROACHES.keys()),
        help="Approach names to compare (default: all in prompts.py)",
    )
    args = parser.parse_args(argv)

    api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"')
    if not api_key or api_key.startswith("your-"):
        logger.error("Set GEMINI_API_KEY in .env before running LLM A/B eval.")
        return 1
    if not args.queries.exists():
        logger.error("Query file not found: %s", args.queries)
        return 1
    if args.limit < 1:
        logger.error("--limit must be >= 1")
        return 1

    for name in args.approaches:
        get_approach(name)

    queries = load_queries(args.queries)[: args.limit]
    print(
        f"Loaded {len(queries)} queries; approaches={args.approaches}; "
        f"pause={args.pause}s"
    )

    client = genai.Client(api_key=api_key)
    try:
        report = asyncio.run(
            run_ab(
                client,
                queries,
                list(args.approaches),
                pause=args.pause,
                top_k=args.top_k,
            )
        )
    finally:
        try:
            asyncio.run(close_qdrant_client())
        except Exception:  # noqa: BLE001
            pass

    print_report(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
