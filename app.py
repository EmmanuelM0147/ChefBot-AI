"""
ChefBot Streamlit UI - inventory in, streamed recipe out, thumbs feedback.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("CHEFBOT_API_URL", "http://localhost:8000").strip().strip('"')
GENERATE_URL = f"{API_BASE_URL.rstrip('/')}/api/generate-recipe"
FEEDBACK_URL = f"{API_BASE_URL.rstrip('/')}/api/feedback"

DIET_OPTIONS = [
    "Vegetarian",
    "Vegan",
    "Gluten-free",
    "Dairy-free",
    "Nut-free",
    "Shellfish allergy",
    "Egg allergy",
    "Soy allergy",
    "Low carb",
    "High protein",
    "Keto",
    "Halal",
    "Kosher",
]


st.set_page_config(
    page_title="ChefBot AI",
    page_icon="🥗",
    layout="centered",
)

st.title("ChefBot AI")
st.caption("Tell us what’s in your fridge - get a grounded, streamed recipe.")

with st.form("recipe_form", clear_on_submit=False):
    inventory_raw = st.text_area(
        "Fridge inventory",
        placeholder="e.g. chicken, garlic, tomatoes, rice, olive oil",
        help="Comma-separated list of ingredients you have on hand.",
        height=120,
    )
    dietary_choices = st.multiselect(
        "Allergies & diets",
        options=DIET_OPTIONS,
        help="Select any allergies or dietary preferences to enforce.",
    )
    submitted = st.form_submit_button("Generate Recipe", type="primary")


def parse_inventory(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


async def stream_recipe_async(payload: dict[str, Any]) -> AsyncIterator[str]:
    """Async POST to FastAPI and yield response text chunk-by-chunk."""
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", GENERATE_URL, json=payload) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise httpx.HTTPStatusError(
                    f"API error {response.status_code}: {body}",
                    request=response.request,
                    response=response,
                )
            transaction_id = response.headers.get("X-Transaction-Id")
            if transaction_id:
                st.session_state["last_transaction_id"] = transaction_id
            async for chunk in response.aiter_text():
                if chunk:
                    yield chunk


def stream_recipe(payload: dict[str, Any]) -> Iterator[str]:
    """Sync bridge for st.write_stream with transaction-id capture."""
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", GENERATE_URL, json=payload) as response:
            if response.status_code >= 400:
                body = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"API error {response.status_code}: {body}")
            transaction_id = response.headers.get("X-Transaction-Id")
            if transaction_id:
                st.session_state["last_transaction_id"] = transaction_id
            for chunk in response.iter_text():
                if chunk:
                    yield chunk


def post_feedback(transaction_id: str, feedback: str) -> None:
    response = httpx.post(
        FEEDBACK_URL,
        json={"transaction_id": transaction_id, "feedback": feedback},
        timeout=10.0,
    )
    response.raise_for_status()


if submitted:
    inventory = parse_inventory(inventory_raw)
    if not inventory:
        st.error("Add at least one inventory item (comma-separated).")
        st.stop()

    payload = {
        "inventory": inventory,
        "dietary_choices": ", ".join(dietary_choices) if dietary_choices else "",
        "limit": 5,
    }

    st.subheader("Your recipe")
    status = st.status(f"Contacting ChefBot at `{GENERATE_URL}`…", expanded=False)

    try:
        try:
            recipe_text = st.write_stream(stream_recipe_async(payload))
        except TypeError:
            recipe_text = st.write_stream(stream_recipe(payload))
        status.update(label="Recipe ready", state="complete")
        st.session_state["last_recipe"] = recipe_text or ""
        st.session_state["last_payload"] = payload
        st.session_state["show_feedback"] = True
    except Exception as exc:  # noqa: BLE001 - show friendly UI errors
        status.update(label="Generation failed", state="error")
        message = str(exc)
        if "429" in message or "RESOURCE_EXHAUSTED" in message or "quota" in message.lower():
            st.error(
                "Gemini free-tier embedding quota is exhausted. "
                "Wait for the quota reset (often ~1 day for the daily limit), "
                "or upgrade billing. The API will automatically fall back to "
                "ingredient-only search after a backend reload."
            )
            st.code(message, language="text")
        elif "connect" in message.lower() or "refused" in message.lower():
            st.error(
                f"Could not reach the recipe API at `{GENERATE_URL}`. "
                "Is FastAPI running?\n\n"
                f"`{message}`"
            )
        else:
            st.error(f"Recipe generation failed.\n\n`{message}`")
        st.session_state["show_feedback"] = False

if st.session_state.get("show_feedback") and st.session_state.get("last_recipe"):
    st.divider()
    st.markdown("### How was this recipe?")
    feedback = st.feedback("thumbs", key="recipe_feedback")
    if feedback is not None:
        label = "thumbs_up" if feedback == 1 else "thumbs_down"
        st.session_state["last_feedback"] = label
        transaction_id = st.session_state.get("last_transaction_id")
        if transaction_id:
            try:
                post_feedback(transaction_id, label)
                st.success(f"Thanks - recorded **{label.replace('_', ' ')}** in monitoring.")
            except Exception as exc:  # noqa: BLE001
                st.warning(
                    f"Feedback saved locally, but monitoring API update failed: `{exc}`"
                )
        else:
            st.success(f"Thanks - recorded **{label.replace('_', ' ')}**.")
            st.caption("No transaction id returned by the API; DB feedback was skipped.")
