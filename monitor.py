"""
ChefBot monitoring - LLM Zoomcamp-style PostgreSQL interaction logging.

Tracks each generation: user query, best matched recipe id, LLM output,
response latency, and optional thumbs feedback.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Literal
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("chefbot.monitor")

FeedbackValue = Literal["thumbs_up", "thumbs_down"] | None

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chefbot_interactions (
    id              UUID PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_query      TEXT NOT NULL,
    dietary_choices TEXT,
    best_recipe_id  TEXT,
    best_recipe_title TEXT,
    llm_output      TEXT,
    response_latency_ms DOUBLE PRECISION,
    user_feedback   TEXT
        CHECK (user_feedback IS NULL
               OR user_feedback IN ('thumbs_up', 'thumbs_down')),
    model_name      TEXT,
    status          TEXT NOT NULL DEFAULT 'ok'
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chefbot_interactions_created_at
    ON chefbot_interactions (created_at DESC);
"""


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip().strip('"')


@contextmanager
def get_connection() -> Generator[Any, None, None]:
    """Open a short-lived psycopg connection (Zoomcamp-style per-request connect)."""
    import psycopg

    database_url = get_database_url()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured in .env")

    conn = psycopg.connect(database_url, connect_timeout=5)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_monitoring_table() -> bool:
    """
    Create the local PostgreSQL tracking table if it does not exist.

    Returns True on success, False if Postgres is unreachable (non-fatal).
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                cur.execute(CREATE_INDEX_SQL)
        logger.info("Monitoring table chefbot_interactions is ready.")
        return True
    except Exception as exc:  # noqa: BLE001 - monitoring must not block the app
        logger.warning("Could not initialize monitoring table: %s", exc)
        return False


def new_transaction_id() -> str:
    return str(uuid4())


def log_interaction(
    *,
    transaction_id: str | None = None,
    user_query: str,
    dietary_choices: str = "",
    best_recipe_id: str | None = None,
    best_recipe_title: str | None = None,
    llm_output: str = "",
    response_latency_ms: float | None = None,
    user_feedback: FeedbackValue = None,
    model_name: str = "gemini-2.5-flash",
    status: str = "ok",
) -> str | None:
    """
    Insert one generation transaction into PostgreSQL.

    Returns the transaction id on success, or None if logging failed.
    """
    tx_id = transaction_id or new_transaction_id()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chefbot_interactions (
                        id,
                        created_at,
                        user_query,
                        dietary_choices,
                        best_recipe_id,
                        best_recipe_title,
                        llm_output,
                        response_latency_ms,
                        user_feedback,
                        model_name,
                        status
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        user_query = EXCLUDED.user_query,
                        dietary_choices = EXCLUDED.dietary_choices,
                        best_recipe_id = EXCLUDED.best_recipe_id,
                        best_recipe_title = EXCLUDED.best_recipe_title,
                        llm_output = EXCLUDED.llm_output,
                        response_latency_ms = EXCLUDED.response_latency_ms,
                        model_name = EXCLUDED.model_name,
                        status = EXCLUDED.status
                    """,
                    (
                        tx_id,
                        datetime.now(timezone.utc),
                        user_query,
                        dietary_choices or None,
                        best_recipe_id,
                        best_recipe_title,
                        llm_output,
                        response_latency_ms,
                        user_feedback,
                        model_name,
                        status,
                    ),
                )
        logger.info(
            "Logged interaction %s (latency_ms=%s, recipe_id=%s)",
            tx_id,
            response_latency_ms,
            best_recipe_id,
        )
        return tx_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to log interaction %s: %s", tx_id, exc)
        return None


def update_feedback(transaction_id: str, user_feedback: FeedbackValue) -> bool:
    """Attach thumbs-up / thumbs-down feedback to an existing transaction."""
    if user_feedback not in ("thumbs_up", "thumbs_down", None):
        raise ValueError("user_feedback must be 'thumbs_up', 'thumbs_down', or None")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chefbot_interactions
                    SET user_feedback = %s
                    WHERE id = %s
                    """,
                    (user_feedback, transaction_id),
                )
                updated = cur.rowcount > 0
        if updated:
            logger.info("Updated feedback for %s → %s", transaction_id, user_feedback)
        else:
            logger.warning("No interaction found for feedback update: %s", transaction_id)
        return updated
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to update feedback for %s: %s", transaction_id, exc)
        return False


def save_interaction_safe(**kwargs: Any) -> None:
    """Background-task wrapper that never raises into the request lifecycle."""
    try:
        log_interaction(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Background monitoring save failed: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = init_monitoring_table()
    print("init_monitoring_table:", ok)
    if ok:
        tx = log_interaction(
            user_query="chicken, garlic, tomato",
            dietary_choices="high protein",
            best_recipe_id="0",
            best_recipe_title="Smoke Test Recipe",
            llm_output="Test output",
            response_latency_ms=12.3,
        )
        print("logged:", tx)
        if tx:
            print("feedback:", update_feedback(tx, "thumbs_up"))
