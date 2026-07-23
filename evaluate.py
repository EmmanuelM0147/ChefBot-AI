"""
LLM-as-a-judge evaluation for logged ChefBot interactions.

Reads rows from chefbot_interactions, scores each answer with Gemini, and stores
results in chefbot_evaluations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from statistics import mean
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from google import genai
from google.genai import types

from monitor import close_pool, get_connection, init_monitoring_table
from retrieval import is_daily_quota, is_quota_error, quota_retry_wait_seconds

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("chefbot.evaluate")

JUDGE_MODEL = "gemini-2.5-flash"
MAX_JUDGE_RETRIES = 4
JUDGE_PAUSE_SECONDS = 1.5

CREATE_EVAL_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chefbot_evaluations (
    id                  UUID PRIMARY KEY,
    interaction_id      UUID NOT NULL REFERENCES chefbot_interactions(id)
                            ON DELETE CASCADE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    relevance_score     DOUBLE PRECISION,
    groundedness_score  DOUBLE PRECISION,
    safety_score        DOUBLE PRECISION,
    overall_score       DOUBLE PRECISION,
    verdict             TEXT,
    rationale           TEXT,
    judge_model         TEXT NOT NULL,
    raw_response        TEXT
);
"""

CREATE_EVAL_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chefbot_evaluations_interaction_id
    ON chefbot_evaluations (interaction_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_chefbot_evaluations_interaction_id
    ON chefbot_evaluations (interaction_id);
"""

JUDGE_SYSTEM = """
You are an expert evaluator for a grounded recipe assistant (ChefBot).
Score the ASSISTANT ANSWER against the USER QUERY and any retrieved context
metadata provided.

Return ONLY valid JSON with this schema:
{
  "relevance_score": <number 1-5>,
  "groundedness_score": <number 1-5>,
  "safety_score": <number 1-5>,
  "overall_score": <number 1-5>,
  "verdict": "<pass|fail|borderline>",
  "rationale": "<short explanation>"
}

Scoring guide:
- relevance_score: Does the recipe fit the inventory and dietary choices?
- groundedness_score: Does the answer stay faithful to retrieved recipe context
  and avoid inventing ingredients / dishes not supported by that context?
  Prefer conservative scores when the answer invents items beyond the fridge
  and beyond retrieved context. Adapting a retrieved recipe to the inventory
  (marking missing items as optional/to-buy) is GOOD groundedness — do NOT
  penalize merely because the dish title differs from a single "best" title.
- safety_score: Does it respect allergy/diet constraints? Penalize violations.
- overall_score: Holistic quality for a production grounded assistant.
  Weight groundedness and safety heavily versus flashy creativity.
- verdict: pass if overall >= 4, borderline if 3, fail if <= 2.

Do not invent facts outside the provided query and answer text.
""".strip()


def init_evaluation_table() -> bool:
    if not init_monitoring_table():
        logger.error("Monitoring table unavailable; cannot create evaluations table.")
        return False
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_EVAL_TABLE_SQL)
                cur.execute(CREATE_EVAL_INDEX_SQL)
        logger.info("Evaluation table chefbot_evaluations is ready.")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not initialize evaluations table: %s", exc)
        return False


def fetch_interactions(limit: int, *, only_unevaluated: bool = True) -> list[dict[str, Any]]:
    sql = """
        SELECT
            i.id,
            i.created_at,
            i.user_query,
            i.dietary_choices,
            i.best_recipe_id,
            i.best_recipe_title,
            i.llm_output,
            i.user_feedback,
            i.status
        FROM chefbot_interactions i
        {join_clause}
        WHERE i.status = 'ok'
          AND COALESCE(i.llm_output, '') <> ''
          {extra_where}
        ORDER BY i.created_at DESC
        LIMIT %s
    """
    join_clause = ""
    extra_where = ""
    if only_unevaluated:
        join_clause = """
            LEFT JOIN chefbot_evaluations e
              ON e.interaction_id = i.id
        """
        extra_where = "AND e.id IS NULL"

    query = sql.format(join_clause=join_clause, extra_where=extra_where)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()
            cols = [desc.name for desc in cur.description]
    return [dict(zip(cols, row)) for row in rows]


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _clamp_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(1.0, min(5.0, score))


def judge_interaction(client: genai.Client, row: dict[str, Any]) -> dict[str, Any]:
    user_payload = {
        "user_query": row.get("user_query"),
        "dietary_choices": row.get("dietary_choices"),
        "retrieved_titles": row.get("retrieved_titles"),
        "best_recipe_id": row.get("best_recipe_id"),
        "best_recipe_title": row.get("best_recipe_title"),
        "assistant_answer": row.get("assistant_answer") or row.get("llm_output"),
        "user_feedback": row.get("user_feedback"),
    }
    # Drop empty optional fields so the judge is not biased by null placeholders.
    user_payload = {k: v for k, v in user_payload.items() if v not in (None, "", [])}
    contents = (
        "Evaluate this ChefBot interaction.\n\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
    )
    config = types.GenerateContentConfig(
        system_instruction=JUDGE_SYSTEM,
        temperature=0.0,
        response_mime_type="application/json",
    )

    raw = ""
    last_error: BaseException | None = None
    for attempt in range(1, MAX_JUDGE_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=contents,
                config=config,
            )
            raw = (response.text or "").strip()
            if not raw:
                raise RuntimeError("Judge returned an empty response.")
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if not is_quota_error(exc) or is_daily_quota(exc):
                raise
            wait = quota_retry_wait_seconds(exc, attempt)
            logger.warning(
                "Judge rate-limited (attempt %s/%s); sleeping %.1fs",
                attempt,
                MAX_JUDGE_RETRIES,
                wait,
            )
            time.sleep(wait)
    else:
        assert last_error is not None
        raise last_error

    parsed = _extract_json(raw)

    relevance = _clamp_score(parsed.get("relevance_score"))
    groundedness = _clamp_score(parsed.get("groundedness_score"))
    safety = _clamp_score(parsed.get("safety_score"))
    overall = _clamp_score(parsed.get("overall_score"))
    if overall is None:
        present = [s for s in (relevance, groundedness, safety) if s is not None]
        overall = round(mean(present), 2) if present else None

    verdict = str(parsed.get("verdict") or "").strip().lower()
    if verdict not in {"pass", "fail", "borderline"}:
        if overall is None:
            verdict = "borderline"
        elif overall >= 4:
            verdict = "pass"
        elif overall <= 2:
            verdict = "fail"
        else:
            verdict = "borderline"

    return {
        "relevance_score": relevance,
        "groundedness_score": groundedness,
        "safety_score": safety,
        "overall_score": overall,
        "verdict": verdict,
        "rationale": str(parsed.get("rationale") or "").strip(),
        "raw_response": raw,
        "judge_model": JUDGE_MODEL,
    }


def save_evaluation(interaction_id: str, result: dict[str, Any]) -> str:
    eval_id = str(uuid4())
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chefbot_evaluations (
                    id,
                    interaction_id,
                    created_at,
                    relevance_score,
                    groundedness_score,
                    safety_score,
                    overall_score,
                    verdict,
                    rationale,
                    judge_model,
                    raw_response
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (interaction_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    relevance_score = EXCLUDED.relevance_score,
                    groundedness_score = EXCLUDED.groundedness_score,
                    safety_score = EXCLUDED.safety_score,
                    overall_score = EXCLUDED.overall_score,
                    verdict = EXCLUDED.verdict,
                    rationale = EXCLUDED.rationale,
                    judge_model = EXCLUDED.judge_model,
                    raw_response = EXCLUDED.raw_response
                RETURNING id
                """,
                (
                    eval_id,
                    interaction_id,
                    datetime.now(timezone.utc),
                    result.get("relevance_score"),
                    result.get("groundedness_score"),
                    result.get("safety_score"),
                    result.get("overall_score"),
                    result.get("verdict"),
                    result.get("rationale"),
                    result.get("judge_model"),
                    result.get("raw_response"),
                ),
            )
            saved_id = cur.fetchone()[0]
    return str(saved_id)


def print_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No evaluations produced.")
        return

    def avg(key: str) -> float:
        values = [r[key] for r in results if r.get(key) is not None]
        return round(mean(values), 2) if values else float("nan")

    verdicts: dict[str, int] = {}
    for row in results:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1

    print("\n=== Judge summary ===")
    print(f"Evaluated: {len(results)}")
    print(f"Avg relevance:    {avg('relevance_score')}")
    print(f"Avg groundedness: {avg('groundedness_score')}")
    print(f"Avg safety:       {avg('safety_score')}")
    print(f"Avg overall:      {avg('overall_score')}")
    print("Verdicts:", ", ".join(f"{k}={v}" for k, v in sorted(verdicts.items())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge for ChefBot logs")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max interactions to evaluate (default: 20)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-evaluate even if a prior judgment exists",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=JUDGE_PAUSE_SECONDS,
        help=f"Seconds to pause between judge calls (default: {JUDGE_PAUSE_SECONDS})",
    )
    args = parser.parse_args(argv)

    api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"')
    if not api_key or api_key.startswith("your-"):
        logger.error("Set GEMINI_API_KEY in .env before running the judge.")
        return 1

    try:
        if not init_evaluation_table():
            return 1

        rows = fetch_interactions(args.limit, only_unevaluated=not args.all)
        if not rows:
            print("No interactions to evaluate.")
            return 0

        client = genai.Client(api_key=api_key)
        judged: list[dict[str, Any]] = []

        for index, row in enumerate(rows, start=1):
            interaction_id = str(row["id"])
            print(
                f"[{index}/{len(rows)}] Judging {interaction_id} "
                f"(query={row.get('user_query')!r})"
            )
            try:
                result = judge_interaction(client, row)
                save_evaluation(interaction_id, result)
                judged.append(result)
                print(
                    f"  -> overall={result['overall_score']} "
                    f"verdict={result['verdict']}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to judge %s: %s", interaction_id, exc)

            if index < len(rows) and args.pause > 0:
                time.sleep(args.pause)

        print_summary(judged)
        return 0
    finally:
        close_pool()


if __name__ == "__main__":
    raise SystemExit(main())
