"""
ChefBot monitoring - LLM Zoomcamp-style PostgreSQL interaction logging.

Tracks each generation: user query, best matched recipe id, LLM output,
response latency, and optional thumbs feedback.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Literal
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("chefbot.monitor")

FeedbackValue = Literal["thumbs_up", "thumbs_down"] | None

# Small pool: monitoring is write-light and must not starve the API workers.
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 5
POOL_TIMEOUT_SECONDS = 10.0
CONNECT_TIMEOUT_SECONDS = 5
MAX_CONNECT_RETRIES = 2

_POOL: Any | None = None

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


def normalize_database_url(database_url: str) -> str:
    """
    Prepare a Postgres URL for local Docker or hosted providers (Supabase/Neon).

    Remote hosts get sslmode=require when unset. Localhost / docker stay plain.
    Not for Render — use Supabase, Neon, or Vercel Postgres in production.
    """
    url = database_url.strip()
    if not url:
        return url

    lowered = url.lower()
    is_local = any(
        host in lowered
        for host in (
            "localhost",
            "127.0.0.1",
            "@postgres:",  # docker compose service hostname
            "@db:",
        )
    )
    if is_local:
        return url

    if "sslmode=" not in lowered:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"
    return url


def _pool_connect_kwargs(database_url: str) -> dict[str, Any]:
    """
    Connection kwargs for psycopg.

    Supabase transaction poolers (port 6543 / pooler hosts) do not support
    prepared statements well, so disable them there.
    """
    kwargs: dict[str, Any] = {"connect_timeout": CONNECT_TIMEOUT_SECONDS}
    lowered = database_url.lower()
    uses_transaction_pooler = (
        ":6543" in lowered
        or "pooler.supabase.com" in lowered
        or "pgbouncer=true" in lowered
    )
    if uses_transaction_pooler:
        kwargs["prepare_threshold"] = None
    return kwargs


def get_pool() -> Any:
    """Lazy process-wide psycopg ConnectionPool."""
    global _POOL
    if _POOL is not None:
        return _POOL

    from psycopg_pool import ConnectionPool

    database_url = normalize_database_url(get_database_url())
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured in .env")

    _POOL = ConnectionPool(
        conninfo=database_url,
        min_size=POOL_MIN_SIZE,
        max_size=POOL_MAX_SIZE,
        timeout=POOL_TIMEOUT_SECONDS,
        kwargs=_pool_connect_kwargs(database_url),
        open=True,
        name="chefbot-monitoring",
    )
    logger.info(
        "Monitoring DB pool ready (min=%s max=%s)",
        POOL_MIN_SIZE,
        POOL_MAX_SIZE,
    )
    return _POOL


def close_pool() -> None:
    """Close the shared pool (call from app lifespan shutdown)."""
    global _POOL
    if _POOL is None:
        return
    try:
        _POOL.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error closing monitoring DB pool: %s", exc)
    finally:
        _POOL = None


def _reset_pool() -> None:
    """Drop a broken pool so the next acquire recreates it."""
    close_pool()


@contextmanager
def get_connection() -> Generator[Any, None, None]:
    """
    Borrow a pooled connection with one reconnect retry on transient failure.
    """
    last_error: BaseException | None = None
    conn_cm: Any | None = None
    conn: Any | None = None

    for attempt in range(1, MAX_CONNECT_RETRIES + 1):
        try:
            pool = get_pool()
            conn_cm = pool.connection()
            conn = conn_cm.__enter__()
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Monitoring DB acquire failed (attempt %s/%s): %s",
                attempt,
                MAX_CONNECT_RETRIES,
                exc,
            )
            _reset_pool()
            if attempt >= MAX_CONNECT_RETRIES:
                raise
            time.sleep(0.4 * attempt)

    assert conn is not None and conn_cm is not None
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn_cm.__exit__(None, None, None)


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


def get_monitoring_summary() -> dict[str, Any]:
    """
    Lightweight Zoomcamp-style metrics from Postgres.

    This is the intended monitoring surface (Supabase + SQL/API), not Grafana.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*)::int AS interactions,
                    COUNT(*) FILTER (WHERE status = 'ok')::int AS ok_count,
                    COUNT(*) FILTER (WHERE status <> 'ok')::int AS error_count,
                    COUNT(*) FILTER (WHERE user_feedback = 'thumbs_up')::int AS thumbs_up,
                    COUNT(*) FILTER (WHERE user_feedback = 'thumbs_down')::int AS thumbs_down,
                    ROUND(AVG(response_latency_ms)::numeric, 1) AS avg_latency_ms
                FROM chefbot_interactions
                """
            )
            inter = cur.fetchone()
            inter_cols = [d.name for d in cur.description]
            interaction_stats = dict(zip(inter_cols, inter))

            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_name = 'chefbot_evaluations'
                )
                """
            )
            has_evals = bool(cur.fetchone()[0])
            evaluation_stats: dict[str, Any] = {
                "evaluations": 0,
                "avg_overall": None,
                "pass_count": 0,
                "fail_count": 0,
                "borderline_count": 0,
            }
            if has_evals:
                cur.execute(
                    """
                    SELECT
                        COUNT(*)::int AS evaluations,
                        ROUND(AVG(overall_score)::numeric, 2) AS avg_overall,
                        COUNT(*) FILTER (WHERE verdict = 'pass')::int AS pass_count,
                        COUNT(*) FILTER (WHERE verdict = 'fail')::int AS fail_count,
                        COUNT(*) FILTER (WHERE verdict = 'borderline')::int AS borderline_count
                    FROM chefbot_evaluations
                    """
                )
                ev = cur.fetchone()
                ev_cols = [d.name for d in cur.description]
                evaluation_stats = dict(zip(ev_cols, ev))

    return {
        "source": "supabase_postgres",
        "interactions": {
            key: (float(value) if hasattr(value, "as_tuple") else value)
            for key, value in interaction_stats.items()
        },
        "evaluations": {
            key: (float(value) if hasattr(value, "as_tuple") else value)
            for key, value in evaluation_stats.items()
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
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
    finally:
        close_pool()
