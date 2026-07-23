"""
ChefBot Streamlit UI - inventory in, streamed recipe out, thumbs feedback.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import httpx
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()


def resolve_api_base_url() -> str:
    """Prefer Streamlit secrets, then env, then deployed Vercel API."""
    try:
        secret_url = st.secrets.get("CHEFBOT_API_URL")  # type: ignore[attr-defined]
        if secret_url:
            return str(secret_url).strip().strip('"').rstrip("/")
    except Exception:
        pass

    return (
        os.getenv("CHEFBOT_API_URL", "https://chef-bot-ai-one.vercel.app")
        .strip()
        .strip('"')
        .rstrip("/")
    )


API_BASE_URL = resolve_api_base_url()
GENERATE_URL = f"{API_BASE_URL}/api/generate-recipe"
FEEDBACK_URL = f"{API_BASE_URL}/api/feedback"
MONITORING_SUMMARY_URL = f"{API_BASE_URL}/api/monitoring/summary"
MONITORING_DASHBOARD_URL = f"{API_BASE_URL}/api/monitoring/dashboard"

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

QUICK_STARTS = {
    "Weeknight protein": "chicken, garlic, lemon, olive oil, rice",
    "Pantry pasta": "pasta, canned tomatoes, garlic, olive oil, basil",
    "Veggie skillet": "eggs, spinach, onion, potato, cheese",
    "Soup night": "carrots, celery, onion, lentils, broth",
}

MACRO_SPLIT = re.compile(
    r"\n---\n## Estimated Macros \(tool-calculated\)",
    flags=re.IGNORECASE,
)

SHARED_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560;9..144,700&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--app-bg) !important;
  color: var(--text);
  font-family: "Outfit", sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
div[data-testid="stDecoration"], [data-testid="stDeployButton"] {
  display: none !important;
}

.block-container {
  max-width: 860px !important;
  padding-top: 1.4rem !important;
  padding-bottom: 3.2rem !important;
}
.stApp { overflow-x: hidden; }


@keyframes rise {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
  0% { background-position: 0% 50%; }
  100% { background-position: 100% 50%; }
}
@keyframes pulse-dot {
  0%, 100% { opacity: 0.35; transform: scale(0.9); }
  50% { opacity: 1; transform: scale(1); }
}

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.75rem;
  animation: rise 0.45s ease both;
}
.topbar-meta {
  font-size: 0.8rem;
  color: var(--muted);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.hero { animation: rise 0.7s ease both; margin-bottom: 1.25rem; }
.hero-kicker {
  display: inline-flex; align-items: center; gap: 0.45rem;
  font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--accent); font-weight: 500; margin-bottom: 0.75rem;
}
.hero-kicker span.dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
  box-shadow: 0 0 0 4px var(--accent-glow);
}
.hero-brand {
  font-family: "Fraunces", Georgia, serif;
  font-size: clamp(2.6rem, 6.5vw, 3.8rem);
  font-weight: 700; letter-spacing: -0.035em; line-height: 0.98;
  margin: 0 0 0.75rem 0; color: var(--text);
}
.hero-sub {
  font-size: 1.05rem; font-weight: 300; color: var(--muted);
  max-width: 36rem; margin: 0; line-height: 1.55;
}

div[data-testid="stForm"] {
  animation: rise 0.75s ease 0.05s both;
  border: 1px solid var(--line) !important;
  background: var(--panel) !important;
  border-radius: 22px !important;
  padding: 1.1rem 1.15rem 0.95rem 1.15rem !important;
  box-shadow: var(--shadow);
}

.quick-label {
  font-size: 0.8rem; color: var(--muted); margin: 0.2rem 0 0.45rem 0;
}

.stButton > button[kind="secondary"] {
  background: var(--chip-bg) !important;
  border: 1px solid var(--line) !important;
  color: var(--text) !important;
  border-radius: 999px !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
}
.stButton > button[kind="secondary"]:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}

.stFormSubmitButton > button {
  width: 100%;
  background: var(--cta-bg) !important;
  background-size: 180% 180% !important;
  animation: shimmer 4s ease infinite;
  color: var(--cta-text) !important;
  border: none !important;
  border-radius: 14px !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  padding: 0.72rem 1.2rem !important;
  box-shadow: var(--cta-shadow);
}
.stFormSubmitButton > button:hover {
  filter: brightness(1.04);
}
.stTextArea textarea:focus, .stMultiSelect [data-baseweb="select"] > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

.stTextArea textarea, .stMultiSelect [data-baseweb="select"] > div {
  background: var(--input-bg) !important;
  color: var(--text) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
}
.stTextArea textarea { min-height: 108px !important; font-size: 1rem !important; }

label p, [data-testid="stWidgetLabel"] p {
  font-size: 0.92rem !important; font-weight: 500 !important; color: var(--text) !important;
}

.section-head {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 1rem; margin: 1.6rem 0 0.8rem 0; animation: rise 0.5s ease both;
}
.section-label {
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.55rem; letter-spacing: -0.02em; margin: 0; color: var(--text);
}
.live-pill {
  display: inline-flex; align-items: center; gap: 0.4rem;
  font-size: 0.75rem; color: var(--accent-2);
  border: 1px solid var(--accent-2-line); border-radius: 999px; padding: 0.22rem 0.65rem;
}
.live-pill i {
  width: 7px; height: 7px; border-radius: 50%; background: var(--accent-2);
  animation: pulse-dot 1.2s ease infinite; display: inline-block;
}

.dish-title {
  font-family: "Fraunces", Georgia, serif;
  font-size: clamp(1.5rem, 3.4vw, 2rem);
  letter-spacing: -0.02em; line-height: 1.15;
  margin: 0 0 0.85rem 0; color: var(--text);
}
.recipe-shell {
  animation: rise 0.55s ease both;
  border: 1px solid var(--line);
  background: var(--recipe-bg);
  border-radius: 20px;
  padding: 1.3rem 1.35rem 1.15rem 1.35rem;
  box-shadow: var(--shadow);
  line-height: 1.7;
  color: var(--text);
}
.recipe-shell h1, .recipe-shell h2, .recipe-shell h3 {
  font-family: "Fraunces", Georgia, serif; letter-spacing: -0.02em;
}
.macro-card {
  margin-top: 1rem; border: 1px solid var(--accent-2-line);
  background: var(--macro-bg); border-radius: 16px;
  padding: 1rem 1.15rem; color: var(--muted); font-size: 0.95rem;
}
.macro-card h2, .macro-card h3 {
  font-family: "Fraunces", Georgia, serif;
  color: var(--accent-2) !important; font-size: 1.1rem !important; margin-top: 0 !important;
}

.empty-stage {
  animation: rise 0.65s ease 0.1s both;
  margin-top: 1.35rem; border: 1px solid var(--line); border-radius: 20px;
  padding: 1.4rem 1.3rem; background: var(--empty-bg);
}
.empty-stage h3 {
  font-family: "Fraunces", Georgia, serif; font-size: 1.3rem;
  margin: 0 0 0.4rem 0; color: var(--text);
}
.empty-stage p { margin: 0; color: var(--muted); line-height: 1.55; }

.feedback-wrap {
  margin-top: 1.25rem; padding: 1rem 1.1rem; border-radius: 16px;
  border: 1px solid var(--line); background: var(--panel);
}
.history-card {
  border: 1px solid var(--line); border-radius: 14px; padding: 0.75rem 0.9rem;
  background: var(--chip-bg); margin-bottom: 0.55rem;
}
.history-card strong { color: var(--text); }
.foot {
  margin-top: 2.2rem; color: var(--muted); font-size: 0.8rem; text-align: center;
}
[data-testid="stCaption"] { color: var(--muted) !important; }
div[data-testid="stAlert"] { border-radius: 14px !important; }
[data-testid="stSidebar"] {
  /* Solid fill so main content never shows through the drawer */
  background: #101612 !important;
  background-color: #101612 !important;
  border-right: 1px solid var(--line) !important;
  z-index: 1000002 !important;
}
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
  background-color: transparent !important;
}
[data-testid="stSidebar"] * { color: var(--text); }
.pref-hint {
  font-size: 0.78rem; color: var(--muted); margin: 0.15rem 0 0.85rem 0; line-height: 1.4;
}
/* Hide the swipe-helper iframe so it does not add layout space */
iframe[height="0"], iframe[height="0"][width="0"] {
  display: none !important;
  position: absolute !important;
  width: 0 !important;
  height: 0 !important;
  border: 0 !important;
}


/* Mobile-first refinements */
@media (max-width: 768px) {
  .block-container {
    max-width: 100% !important;
    padding-left: 0.9rem !important;
    padding-right: 0.9rem !important;
    padding-top: 0.7rem !important;
    padding-bottom: 2.4rem !important;
  }

  .topbar { margin-bottom: 0.35rem; }
  .topbar-meta { font-size: 0.7rem; letter-spacing: 0.06em; }

  .hero { margin-bottom: 0.85rem; }
  .hero-kicker { font-size: 0.7rem; margin-bottom: 0.5rem; letter-spacing: 0.12em; }
  .hero-brand {
    font-size: clamp(2.15rem, 11vw, 2.75rem) !important;
    margin-bottom: 0.55rem !important;
  }
  .hero-sub {
    font-size: 0.94rem !important;
    line-height: 1.5 !important;
    max-width: none;
  }

  .quick-label { margin-bottom: 0.35rem; }

  div[data-testid="stForm"] {
    border-radius: 16px !important;
    padding: 0.85rem 0.8rem 0.7rem !important;
  }
  .stTextArea textarea {
    min-height: 96px !important;
    font-size: 16px !important; /* avoids iOS zoom on focus */
  }

  .section-head {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.35rem;
    margin: 1.15rem 0 0.6rem 0;
  }
  .section-label { font-size: 1.28rem; }
  .dish-title { font-size: clamp(1.28rem, 6.5vw, 1.55rem) !important; }

  .recipe-shell,
  .empty-stage,
  .feedback-wrap,
  .macro-card {
    border-radius: 14px;
    padding: 0.95rem 0.9rem;
  }
  .empty-stage { margin-top: 1rem; }
  .empty-stage h3 { font-size: 1.15rem; }
  .macro-card { font-size: 0.9rem; }

  /* Touch-friendly controls; allow label wrap */
  .stButton > button,
  .stDownloadButton > button,
  .stFormSubmitButton > button {
    min-height: 2.75rem !important;
    white-space: normal !important;
    line-height: 1.25 !important;
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
  }
  .stButton > button[kind="secondary"] {
    border-radius: 12px !important;
    font-size: 0.82rem !important;
  }

  /* Wrap column rows into usable mobile grids */
  div[data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    gap: 0.4rem !important;
  }
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    min-width: calc(50% - 0.25rem) !important;
    flex: 1 1 calc(50% - 0.25rem) !important;
  }

  .feedback-wrap { margin-top: 1rem; }
  .foot {
    margin-top: 1.5rem;
    font-size: 0.72rem;
    line-height: 1.4;
    padding: 0 0.25rem;
  }

  [data-testid="stSidebar"] {
    border-right: none !important;
    width: min(20rem, 88vw) !important;
  }
  [data-testid="stSidebar"][aria-expanded="true"] {
    box-shadow:
      8px 0 40px rgba(0, 0, 0, 0.55),
      0 0 0 100vmax rgba(6, 10, 8, 0.82) !important;
  }
  [data-testid="stSidebar"] .block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }

  /* Left-edge cue: swipe right from here to open the menu */
  .stApp::before {
    content: "";
    position: fixed;
    left: 0;
    top: 42%;
    width: 5px;
    height: 72px;
    border-radius: 0 6px 6px 0;
    background: linear-gradient(180deg, transparent, var(--accent), transparent);
    opacity: 0.55;
    z-index: 999998;
    pointer-events: none;
    box-shadow: 0 0 12px var(--accent-glow);
  }
}

@media (max-width: 420px) {
  .block-container {
    padding-left: 0.7rem !important;
    padding-right: 0.7rem !important;
  }
  .hero-sub { font-size: 0.9rem !important; }
  .stButton > button[kind="secondary"] { font-size: 0.78rem !important; }
}
"""

COMPACT_CSS = """
.block-container { max-width: 720px !important; }
"""

WIDE_CSS = """
.block-container { max-width: 1040px !important; }
@media (max-width: 768px) {
  .block-container { max-width: 100% !important; }
}
"""

LARGE_TYPE_CSS = """
html, body, .stApp { font-size: 17px !important; }
.hero-brand { font-size: clamp(2.9rem, 7vw, 4.1rem) !important; }
.recipe-shell { font-size: 1.05rem !important; line-height: 1.8 !important; }
@media (max-width: 768px) {
  html, body, .stApp { font-size: 16px !important; }
  .hero-brand { font-size: clamp(2.25rem, 11vw, 2.9rem) !important; }
}
"""

REDUCED_MOTION_CSS = """
*, *::before, *::after {
  animation: none !important;
  transition: none !important;
}
"""

DARK_THEME_CSS = """
:root {
  --app-bg:
    radial-gradient(900px 420px at 12% -8%, rgba(124, 176, 131, 0.22), transparent 58%),
    radial-gradient(700px 380px at 96% 8%, rgba(226, 194, 122, 0.12), transparent 52%),
    linear-gradient(165deg, #121a16 0%, #0f1412 42%, #0c100e 100%);
  --panel: rgba(24, 36, 30, 0.82);
  --line: rgba(214, 232, 214, 0.12);
  --text: #eef6ef;
  --muted: #9fb4a6;
  --accent: #7cb083;
  --accent-glow: rgba(124, 176, 131, 0.18);
  --accent-2: #e2c27a;
  --accent-2-line: rgba(226, 194, 122, 0.28);
  --input-bg: rgba(10, 14, 12, 0.72);
  --chip-bg: rgba(255,255,255,0.03);
  --recipe-bg: linear-gradient(180deg, rgba(28, 42, 35, 0.92), rgba(16, 24, 20, 0.92));
  --macro-bg: rgba(226, 194, 122, 0.07);
  --empty-bg: linear-gradient(135deg, rgba(124, 176, 131, 0.08), transparent 45%), rgba(18, 28, 23, 0.55);
  --shadow: 0 30px 80px rgba(0, 0, 0, 0.35);
  --cta-bg: linear-gradient(120deg, #8fbf94, #7cb083 40%, #c9b06a);
  --cta-text: #102016;
  --cta-shadow: 0 12px 30px rgba(124, 176, 131, 0.22);
}
"""


def init_preferences() -> None:
    defaults = {
        "layout_density": "Comfortable",
        "type_scale": "Standard",
        "show_starters": True,
        "show_macros": True,
        "reduced_motion": False,
        "result_limit": 5,
        "inventory_input": "",
        "diet_input": [],
        "recipe_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Drop legacy theme / compact prefs
    st.session_state.pop("theme_mode", None)
    if st.session_state.pop("compact_mode", None):
        st.session_state["layout_density"] = "Compact"


def inject_sidebar_swipe() -> None:
    """Mobile swipe: right from left edge opens sidebar; left swipe closes it."""
    components.html(
        """
<script>
(function () {
  const win = window.parent;
  const doc = win.document;
  if (win.__chefbotSwipeBound) return;
  win.__chefbotSwipeBound = true;

  const EDGE_PX = 56;
  const MIN_DX = 55;
  const MOBILE_MAX = 900;

  function isMobile() {
    return win.innerWidth <= MOBILE_MAX;
  }

  function sidebarExpanded() {
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    return !!(sb && sb.getAttribute("aria-expanded") === "true");
  }

  function clickExpand() {
    const btn =
      doc.querySelector('[data-testid="stExpandSidebarButton"]') ||
      doc.querySelector('button[kind="header"]') ||
      doc.querySelector('[data-testid="stBaseButton-header"]');
    if (btn) btn.click();
  }

  function clickCollapse() {
    const collapse =
      doc.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
      doc.querySelector('[data-testid="stSidebarCollapseButton"]') ||
      doc.querySelector('[data-testid="stBaseButton-headerNoPadding"]');
    if (collapse) collapse.click();
  }

  let startX = 0;
  let startY = 0;
  let tracking = false;
  let fromEdge = false;
  let fromSidebar = false;

  doc.addEventListener(
    "touchstart",
    function (e) {
      if (!isMobile() || e.touches.length !== 1) return;
      const t = e.touches[0];
      const target = e.target;
      startX = t.clientX;
      startY = t.clientY;
      tracking = true;
      fromEdge = startX <= EDGE_PX;
      const sb = doc.querySelector('[data-testid="stSidebar"]');
      fromSidebar = !!(sb && sb.contains(target));
    },
    { passive: true }
  );

  doc.addEventListener(
    "touchend",
    function (e) {
      if (!tracking || !isMobile()) {
        tracking = false;
        return;
      }
      tracking = false;
      const t = e.changedTouches[0];
      const dx = t.clientX - startX;
      const dy = t.clientY - startY;
      if (Math.abs(dx) < MIN_DX || Math.abs(dx) < Math.abs(dy) * 1.15) return;

      const open = sidebarExpanded();

      // Swipe right from the left edge (sidebar side) -> open
      if (!open && dx > MIN_DX && fromEdge) {
        clickExpand();
        return;
      }

      // Swipe left while open (especially on drawer) -> close
      if (open && dx < -MIN_DX && (fromSidebar || startX < win.innerWidth * 0.9)) {
        clickCollapse();
      }
    },
    { passive: true }
  );
})();
</script>
        """,
        height=0,
        width=0,
    )


def parse_inventory(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def split_recipe_and_macros(text: str) -> tuple[str, str | None]:
    parts = MACRO_SPLIT.split(text, maxsplit=1)
    if len(parts) == 1:
        return text.strip(), None
    return parts[0].strip(), parts[1].strip()


def extract_dish_title(body: str) -> tuple[str | None, str]:
    lines = [line.strip() for line in body.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    if not lines:
        return None, body
    first = lines[0]
    title_match = re.match(r"^#{1,3}\s+(.+)$", first)
    if title_match:
        return title_match.group(1).strip(), "\n".join(lines[1:]).strip()
    bold = re.match(r"^\*\*(.+?)\*\*\s*$", first)
    if bold:
        return bold.group(1).strip(), "\n".join(lines[1:]).strip()
    if len(first) <= 80 and not first.endswith(":") and not first.startswith("-"):
        return first, "\n".join(lines[1:]).strip()
    return None, body


def friendly_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "429" in message or "resource_exhausted" in lowered or "quota" in lowered:
        return "The kitchen is at capacity right now. Give it a minute and try again."
    if "connect" in lowered or "refused" in lowered or "timed out" in lowered or "timeout" in lowered:
        return "We couldn't reach the recipe service. Please try again in a moment."
    if "502" in message or "retrieval failed" in lowered:
        return "Recipe search is temporarily unavailable. Please try again shortly."
    if "422" in message:
        return "Add at least one ingredient so we know what we're cooking with."
    return "Something went wrong while generating your recipe. Please try again."


def stream_recipe(payload: dict[str, Any]) -> Iterator[str]:
    timeout = httpx.Timeout(connect=20.0, read=55.0, write=30.0, pool=20.0)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                with client.stream("POST", GENERATE_URL, json=payload) as response:
                    if response.status_code in {429, 502, 503} and attempt == 0:
                        body = response.read().decode("utf-8", errors="replace")
                        last_error = RuntimeError(
                            f"API error {response.status_code}: {body}"
                        )
                        time.sleep(2 ** (attempt + 1))
                        continue
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")
                        raise RuntimeError(f"API error {response.status_code}: {body}")
                    transaction_id = response.headers.get("X-Transaction-Id")
                    if transaction_id:
                        st.session_state["last_transaction_id"] = transaction_id
                    yielded = False
                    for chunk in response.iter_text():
                        if chunk:
                            yielded = True
                            yield chunk
                    if not yielded:
                        raise RuntimeError("API returned an empty recipe stream.")
                    return
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"Recipe service connection failed: {exc}") from exc
    if last_error:
        raise RuntimeError(str(last_error)) from last_error


def post_feedback(transaction_id: str, feedback: str) -> None:
    response = httpx.post(
        FEEDBACK_URL,
        json={"transaction_id": transaction_id, "feedback": feedback},
        timeout=20.0,
    )
    response.raise_for_status()


def fetch_monitoring_summary() -> dict[str, Any] | None:
    try:
        response = httpx.get(MONITORING_SUMMARY_URL, timeout=15.0)
        if response.status_code >= 400:
            return None
        data = response.json()
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def fetch_monitoring_dashboard(days: int = 14) -> dict[str, Any] | None:
    try:
        response = httpx.get(
            MONITORING_DASHBOARD_URL,
            params={"days": days},
            timeout=20.0,
        )
        if response.status_code >= 400:
            return None
        data = response.json()
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _rows_to_chart_frame(
    rows: list[dict[str, Any]] | None,
    *,
    index_key: str,
    value_key: str,
    value_label: str | None = None,
):
    """Build a small chart table; returns None when empty."""
    if not rows:
        return None
    try:
        import pandas as pd
    except ImportError:
        return None

    cleaned: list[dict[str, Any]] = []
    label = value_label or value_key
    for row in rows:
        if not isinstance(row, dict):
            continue
        index_val = row.get(index_key)
        value = row.get(value_key)
        if index_val is None or value is None:
            continue
        cleaned.append({index_key: str(index_val), label: float(value)})
    if not cleaned:
        return None
    frame = pd.DataFrame(cleaned).set_index(index_key)
    return frame


def render_monitoring_dashboard() -> None:
    """Monitoring page: feedback stats and charts from Postgres."""
    st.markdown(
        """
        <div class="section-head">
          <p class="section-label">Monitoring</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("## Kitchen dashboard")
    st.caption(
        "Postgres interaction logs + LLM-as-judge scores. "
        "User thumbs feedback is collected on every plated recipe."
    )

    days = st.slider("Lookback window (days)", min_value=3, max_value=30, value=14)
    refresh = st.button("Refresh dashboard", use_container_width=False)
    cache_key = f"monitoring_dashboard_{days}"
    if refresh or cache_key not in st.session_state:
        st.session_state[cache_key] = fetch_monitoring_dashboard(days)

    payload = st.session_state.get(cache_key)
    if not payload:
        st.warning(
            "Monitoring dashboard unavailable. Confirm the API is up and "
            "`DATABASE_URL` points at Postgres."
        )
        return

    summary = payload.get("summary") or {}
    interactions = summary.get("interactions") or {}
    evaluations = summary.get("evaluations") or {}
    charts = payload.get("charts") or {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Plates logged", interactions.get("interactions", 0))
    m2.metric("Avg latency (ms)", interactions.get("avg_latency_ms") or "—")
    m3.metric(
        "Thumbs",
        f"↑{interactions.get('thumbs_up', 0)} / ↓{interactions.get('thumbs_down', 0)}",
    )
    m4.metric("Judge scored", evaluations.get("evaluations", 0))

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 1. Interactions / day")
        frame = _rows_to_chart_frame(
            charts.get("interactions_per_day"),
            index_key="day",
            value_key="interactions",
            value_label="interactions",
        )
        if frame is None:
            st.caption("No interaction volume yet.")
        else:
            st.bar_chart(frame, height=220)

    with c2:
        st.markdown("##### 2. Avg latency / day")
        frame = _rows_to_chart_frame(
            charts.get("avg_latency_per_day"),
            index_key="day",
            value_key="avg_latency_ms",
            value_label="avg_latency_ms",
        )
        if frame is None:
            st.caption("No latency samples yet.")
        else:
            st.line_chart(frame, height=220)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### 3. Status breakdown")
        frame = _rows_to_chart_frame(
            charts.get("status_breakdown"),
            index_key="status",
            value_key="count",
            value_label="count",
        )
        if frame is None:
            st.caption("No status rows yet.")
        else:
            st.bar_chart(frame, height=220)

    with c4:
        st.markdown("##### 4. Feedback breakdown")
        frame = _rows_to_chart_frame(
            charts.get("feedback_breakdown"),
            index_key="feedback",
            value_key="count",
            value_label="count",
        )
        if frame is None:
            st.caption("No feedback rows yet.")
        else:
            st.bar_chart(frame, height=220)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("##### 5. Judge verdicts")
        frame = _rows_to_chart_frame(
            charts.get("judge_verdicts"),
            index_key="verdict",
            value_key="count",
            value_label="count",
        )
        if frame is None:
            st.caption("Run `evaluate.py` to populate judge verdicts.")
        else:
            st.bar_chart(frame, height=220)

    with c6:
        st.markdown("##### 6. Judge score averages")
        frame = _rows_to_chart_frame(
            charts.get("judge_score_averages"),
            index_key="metric",
            value_key="score",
            value_label="score",
        )
        if frame is None:
            st.caption("No judge scores yet.")
        else:
            st.bar_chart(frame, height=220)

    st.markdown("##### 7. Feedback rate / day")
    frame = _rows_to_chart_frame(
        charts.get("feedback_rate_per_day"),
        index_key="day",
        value_key="feedback_rate",
        value_label="feedback_rate",
    )
    if frame is None:
        st.caption("No daily feedback rate yet.")
    else:
        st.line_chart(frame, height=220)

    st.caption(
        f"Window: last {payload.get('window_days', days)} days · "
        f"source={payload.get('source', 'postgres')}"
    )


def push_history(recipe_text: str, payload: dict[str, Any]) -> None:
    title, _ = extract_dish_title(split_recipe_and_macros(recipe_text)[0])
    entry = {
        "title": title or "Untitled plate",
        "recipe": recipe_text,
        "payload": payload,
        "at": datetime.now().strftime("%H:%M"),
    }
    history = list(st.session_state.get("recipe_history") or [])
    history.insert(0, entry)
    st.session_state["recipe_history"] = history[:8]


def render_recipe(text: str) -> None:
    body, macros = split_recipe_and_macros(text)
    title, remainder = extract_dish_title(body)
    st.markdown('<div class="recipe-shell">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<p class="dish-title">{title}</p>', unsafe_allow_html=True)
        if remainder:
            st.markdown(remainder)
    else:
        st.markdown(body)
    st.markdown("</div>", unsafe_allow_html=True)
    if macros and st.session_state.get("show_macros", True):
        st.markdown('<div class="macro-card">', unsafe_allow_html=True)
        st.markdown("### Estimated macros")
        st.markdown(macros)
        st.markdown("</div>", unsafe_allow_html=True)


def generate_and_render(payload: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="section-head">
          <p class="section-label">Your plate</p>
          <span class="live-pill"><i></i> Streaming</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status = st.status("Chef is plating your recipe...", expanded=False)
    st.caption("Usually under a minute. If this stalls near ~55s, try again.")
    try:
        chunks: list[str] = []
        placeholder = st.empty()
        for piece in stream_recipe(payload):
            chunks.append(piece)
            live = "".join(chunks)
            title, remainder = extract_dish_title(MACRO_SPLIT.split(live, maxsplit=1)[0])
            with placeholder.container():
                st.markdown('<div class="recipe-shell">', unsafe_allow_html=True)
                if title:
                    st.markdown(
                        f'<p class="dish-title">{title}</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(remainder or "")
                else:
                    st.markdown(live)
                st.markdown("</div>", unsafe_allow_html=True)

        recipe_text = "".join(chunks)
        placeholder.empty()
        render_recipe(recipe_text)
        status.update(label="Ready to cook", state="complete")
        st.session_state["last_recipe"] = recipe_text
        st.session_state["last_payload"] = payload
        st.session_state["show_feedback"] = True
        push_history(recipe_text, payload)
    except Exception as exc:  # noqa: BLE001
        status.update(label="Couldn't finish this plate", state="error")
        st.error(friendly_error(exc))
        with st.expander("Technical details"):
            st.code(str(exc), language="text")
        st.session_state["show_feedback"] = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ChefBot AI",
    page_icon="🍳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

init_preferences()

layout = st.session_state.get("layout_density", "Comfortable")
layout_css = COMPACT_CSS if layout == "Compact" else (WIDE_CSS if layout == "Wide" else "")
type_css = LARGE_TYPE_CSS if st.session_state.get("type_scale") == "Large" else ""
motion_css = REDUCED_MOTION_CSS if st.session_state.get("reduced_motion") else ""
st.markdown(
    f"<style>{DARK_THEME_CSS}{SHARED_CSS}{layout_css}{type_css}{motion_css}</style>",
    unsafe_allow_html=True,
)
inject_sidebar_swipe()

# --- Sidebar: preferences & history ---
with st.sidebar:
    st.markdown("### Navigate")
    st.radio(
        "View",
        options=["Cook", "Monitoring"],
        key="app_view",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### Experience")
    st.markdown(
        '<p class="pref-hint">On mobile, swipe right from the left edge to open this menu; swipe left to close.</p>',
        unsafe_allow_html=True,
    )
    st.selectbox(
        "Layout",
        options=["Compact", "Comfortable", "Wide"],
        key="layout_density",
        help="How wide the cooking workspace feels.",
    )
    st.selectbox(
        "Text size",
        options=["Standard", "Large"],
        key="type_scale",
    )
    st.toggle("Show starter ideas", key="show_starters")
    st.toggle("Show macro card", key="show_macros")
    st.toggle("Reduce motion", key="reduced_motion")
    st.slider(
        "Recipes to ground on",
        min_value=3,
        max_value=8,
        key="result_limit",
        help="More context can improve grounding; fewer is faster.",
    )

    st.divider()
    st.markdown("### Actions")
    if st.button("New plate", use_container_width=True):
        st.session_state["last_recipe"] = ""
        st.session_state["show_feedback"] = False
        st.session_state["last_transaction_id"] = None
        st.rerun()
    if st.button("Clear fridge input", use_container_width=True):
        st.session_state["inventory_input"] = ""
        st.session_state["diet_input"] = []
        st.rerun()
    if st.session_state.get("last_payload") and st.button(
        "Regenerate last plate", use_container_width=True
    ):
        st.session_state["force_regenerate"] = True
        st.rerun()
    if st.session_state.get("recipe_history") and st.button(
        "Clear recent plates", use_container_width=True
    ):
        st.session_state["recipe_history"] = []
        st.rerun()

    st.divider()
    st.markdown("### Kitchen metrics")
    st.caption("Open Monitoring view for the full chart dashboard.")
    if st.button("Refresh metrics", use_container_width=True):
        st.session_state.pop("monitoring_summary", None)
        for key in list(st.session_state.keys()):
            if str(key).startswith("monitoring_dashboard_"):
                st.session_state.pop(key, None)
    if "monitoring_summary" not in st.session_state:
        st.session_state["monitoring_summary"] = fetch_monitoring_summary()
    summary = st.session_state.get("monitoring_summary")
    if not summary:
        st.caption("Metrics unavailable right now.")
    else:
        interactions = summary.get("interactions") or {}
        evaluations = summary.get("evaluations") or {}
        st.caption(
            f"Plates logged: {interactions.get('interactions', 0)} · "
            f"ok {interactions.get('ok_count', 0)} · "
            f"errors {interactions.get('error_count', 0)}"
        )
        latency = interactions.get("avg_latency_ms")
        if latency is not None:
            st.caption(f"Avg latency: {latency} ms")
        st.caption(
            f"Thumbs: ↑ {interactions.get('thumbs_up', 0)} · "
            f"↓ {interactions.get('thumbs_down', 0)}"
        )
        if evaluations.get("evaluations"):
            st.caption(
                f"Judge: {evaluations.get('evaluations')} scored · "
                f"avg {evaluations.get('avg_overall')} · "
                f"pass {evaluations.get('pass_count')} / "
                f"fail {evaluations.get('fail_count')}"
            )

    st.divider()
    st.markdown("### Recent plates")
    history = st.session_state.get("recipe_history") or []
    if not history:
        st.caption("Your generated plates will appear here.")
    for index, item in enumerate(history):
        with st.container():
            st.markdown(
                f'<div class="history-card"><strong>{item["title"]}</strong>'
                f'<br/><span style="color:var(--muted);font-size:0.8rem;">'
                f'{item["at"]}</span></div>',
                unsafe_allow_html=True,
            )
            hist_cols = st.columns(2)
            with hist_cols[0]:
                if st.button("Open", key=f"open_hist_{index}", use_container_width=True):
                    st.session_state["last_recipe"] = item["recipe"]
                    st.session_state["last_payload"] = item["payload"]
                    st.session_state["show_feedback"] = True
                    st.rerun()
            with hist_cols[1]:
                if st.button("Reuse", key=f"reuse_hist_{index}", use_container_width=True):
                    payload = item.get("payload") or {}
                    st.session_state["inventory_input"] = ", ".join(
                        payload.get("inventory") or []
                    )
                    diets = payload.get("dietary_choices") or ""
                    st.session_state["diet_input"] = [
                        d.strip() for d in diets.split(",") if d.strip()
                    ]
                    st.rerun()

if st.session_state.get("app_view", "Cook") == "Monitoring":
    render_monitoring_dashboard()
    st.markdown(
        '<p class="foot">ChefBot AI · recipe logs · feedback</p>',
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown(
    f'<div class="topbar"><span class="topbar-meta">{layout} layout</span></div>',
    unsafe_allow_html=True,
)

# --- Hero ---
st.markdown(
    """
    <div class="hero">
      <p class="hero-brand">ChefBot AI</p>
      <p class="hero-sub">
        Enter what’s in your fridge and any diet limits. ChefBot looks up real
        recipes from the cookbook index and streams one adapted plate.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Quick starts (2x2 so labels stay readable on phones) ---
if st.session_state.get("show_starters", True):
    st.markdown('<div class="quick-label">Try a starter</div>', unsafe_allow_html=True)
    starter_items = list(QUICK_STARTS.items())
    for row_start in range(0, len(starter_items), 2):
        row = starter_items[row_start : row_start + 2]
        chip_cols = st.columns(2)
        for col, (label, ingredients) in zip(chip_cols, row):
            with col:
                if st.button(label, key=f"chip_{label}", use_container_width=True):
                    st.session_state["inventory_input"] = ingredients
                    st.rerun()

# --- Form ---
with st.form("recipe_form", clear_on_submit=False):
    inventory_raw = st.text_area(
        "What's in the fridge?",
        placeholder="chicken, garlic, tomatoes, rice, olive oil",
        help="Comma-separated ingredients you have on hand.",
        height=118,
        key="inventory_input",
    )
    dietary_choices = st.multiselect(
        "Allergies & diets",
        options=DIET_OPTIONS,
        help="Treated as hard constraints.",
        key="diet_input",
    )
    submitted = st.form_submit_button("Generate recipe")

# --- Force regenerate from sidebar ---
if st.session_state.pop("force_regenerate", False) and st.session_state.get("last_payload"):
    generate_and_render(st.session_state["last_payload"])

# --- Results ---
elif submitted:
    inventory = parse_inventory(inventory_raw)
    if not inventory:
        st.error("Add at least one ingredient so we know what we're cooking with.")
        st.stop()

    payload = {
        "inventory": inventory,
        "dietary_choices": ", ".join(dietary_choices) if dietary_choices else "",
        "limit": int(st.session_state.get("result_limit", 5)),
    }
    generate_and_render(payload)

elif st.session_state.get("last_recipe") and st.session_state.get("show_feedback"):
    st.markdown(
        '<div class="section-head"><p class="section-label">Your plate</p></div>',
        unsafe_allow_html=True,
    )
    render_recipe(st.session_state["last_recipe"])

just_generated = bool(
    st.session_state.get("last_recipe") and st.session_state.get("show_feedback")
)
if just_generated:
    st.download_button(
        "Download recipe",
        data=st.session_state["last_recipe"],
        file_name="chefbot-recipe.md",
        mime="text/markdown",
        use_container_width=True,
    )
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Regenerate", use_container_width=True) and st.session_state.get(
            "last_payload"
        ):
            st.session_state["force_regenerate"] = True
            st.rerun()
    with action_cols[1]:
        if st.button("New plate", key="new_plate_main", use_container_width=True):
            st.session_state["last_recipe"] = ""
            st.session_state["show_feedback"] = False
            st.rerun()
elif not submitted:
    st.markdown(
        """
        <div class="empty-stage">
          <h3>Start with your fridge</h3>
          <p>
            List a few ingredients, set any diet constraints, then generate.
            Layout and recent plates are in the sidebar.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.session_state.get("show_feedback") and st.session_state.get("last_recipe"):
    st.markdown('<div class="feedback-wrap">', unsafe_allow_html=True)
    st.markdown("#### How was it?")
    st.caption("A quick thumbs-up or down helps improve future plates.")
    feedback = st.feedback("thumbs", key="recipe_feedback")
    if feedback is not None:
        label = "thumbs_up" if feedback == 1 else "thumbs_down"
        st.session_state["last_feedback"] = label
        transaction_id = st.session_state.get("last_transaction_id")
        if transaction_id:
            try:
                post_feedback(transaction_id, label)
                st.success("Thanks - feedback saved.")
            except Exception:
                st.success("Thanks - feedback noted for this session.")
        else:
            st.success("Thanks - feedback noted for this session.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<p class="foot">ChefBot AI</p>',
    unsafe_allow_html=True,
)
