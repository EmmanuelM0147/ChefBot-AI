# ChefBot AI

Grounded, inventory-aware recipe generation with **FastAPI**, **Qdrant**, **Gemini**, **PostgreSQL monitoring**, and a **Streamlit** UI.

ChefBot retrieves real recipes from a vector index, constrains generation to that context, streams the answer with **Gemini 2.5 Flash**, appends **tool-calculated** macro estimates, and logs every transaction for evaluation.

---

## Architecture

```mermaid
flowchart LR
  subgraph UI["Presentation"]
    ST["Streamlit<br/>app.py"]
  end

  subgraph API["Application"]
    FA["FastAPI<br/>main.py"]
    RT["Retrieval<br/>retrieval.py"]
    TOOL["estimate_macros<br/>Python tool"]
    MON["monitor.py"]
  end

  subgraph AI["Google GenAI"]
    EMB["gemini-embedding-001<br/>768-d vectors"]
    GEN["gemini-2.5-flash<br/>generate_content_stream"]
  end

  subgraph Data["Data plane"]
    JSON["dataset/recipes.json<br/>structured JSONL"]
    ING["ingest.py"]
    QD[("Qdrant<br/>chefbot_recipes")]
    PG[("PostgreSQL<br/>chefbot_interactions")]
  end

  ST -->|"POST /api/generate-recipe<br/>HTTP stream"| FA
  ST -->|"POST /api/feedback<br/>thumbs"| FA
  FA --> RT
  RT --> EMB
  RT -->|"cosine + ingredient filter<br/>(filter-only fallback)"| QD
  FA --> GEN
  GEN -->|"function call"| TOOL
  FA -->|"StreamingResponse"| ST
  FA -->|"BackgroundTasks log"| MON
  MON --> PG
  JSON --> ING
  ING --> EMB
  ING -->|"PointStruct upsert"| QD
```

### Request path (runtime)

1. User enters fridge inventory + diet/allergy constraints in Streamlit.
2. UI opens an async streaming `POST` to `/api/generate-recipe`.
3. FastAPI calls `search_recipes()`: embed the query, filter Qdrant payloads by inventory ingredients, return top matches.
4. If Gemini embedding quota is exhausted, retrieval **falls back** to ingredient-filter-only Qdrant results (no vector ranking).
5. Gemini 2.5 Flash streams a Michelin-style, allergen-safe recipe **using only retrieved context**.
6. The hardcoded `estimate_macros` tool is invoked via Gemini function calling; the **server** appends the calculated macros to the stream.
7. After the stream completes, FastAPI **background-logs** the transaction to PostgreSQL (`user_query`, best recipe id, LLM output, latency).
8. Streamlit shows thumbs feedback and posts to `/api/feedback` using `X-Transaction-Id`.

---

## Why structured JSON instead of CSV

The source dump shipped as both `recipes.csv` and `recipes.json` (~62k recipes, identical content). ChefBot standardizes on **clean structured JSON (JSONL)** for ingestion and retrieval.

| Concern | CSV | Structured JSON |
|---|---|---|
| Nested fields | `ingredients` / `directions` stored as stringified lists (need fragile `ast.literal_eval` / regex) | Native arrays (type-safe for payloads, filters, and prompts) |
| Streaming ingest | Row parser must handle embedded commas/quotes inside list strings | One object per line (line-oriented, resumable, memory-friendly) |
| Schema fidelity | Everything is text; counts and lists lose meaning | Preserves `list[str]`, ints (`num_steps`), and nested structure |
| Vector payload quality | Easy to corrupt ingredient tokens during CSV escape/unescape | Exact strings land in Qdrant for `MatchText` inventory filters |
| RAG prompt assembly | Extra parse step before building context | Direct join of title / category / ingredients |

**Engineering decision:** keep CSV only as an analysis / spreadsheet artifact; treat **JSONL as the system of record** for the ML and search pipeline. That removes a whole class of parse bugs from the hot path (`ingest -> embed -> filter -> generate`).

---

## Gemini 2.5 Flash streaming

ChefBot uses the modern `google-genai` SDK (`from google import genai`) with:

| Capability | Model / API | Role in ChefBot |
|---|---|---|
| Embeddings | `gemini-embedding-001` (`output_dimensionality=768`) | Query + document vectors for Qdrant cosine search |
| Generation | `gemini-2.5-flash` via `client.models.generate_content_stream` | Low-latency, token-by-token recipe prose |
| Tools | Python `estimate_macros` passed in `tools=[...]` | Deterministic macros; server appends results (no invented nutrition) |

### Why stream?

- **Time-to-first-token:** users see the dish title and method appear immediately instead of waiting for a full completion.
- **Backpressure-friendly UI:** Streamlit's `st.write_stream` consumes the FastAPI `StreamingResponse` chunk-by-chunk.
- **Safer tool UX:** narrative streams first; calculated macros are appended after function calling so numbers never come from free-form model text.

System instructions force context-only cooking (no improvisation outside the retrieved recipe set) and forbid the model from inventing calorie/macro figures.

### Embedding quota resilience

Gemini free-tier embed limits can block search (HTTP 429). ChefBot handles this in two places:

| Layer | Behavior |
|---|---|
| `ingest.py` | Paces batches, retries short-term 429s, **resumes** from existing Qdrant point count (use `--recreate` only for a full rebuild) |
| `retrieval.py` | Retries brief rate limits, caches identical query vectors in-process, and on daily quota exhaustion **falls back to filter-only** Qdrant scroll so generation can still run |

---

## Monitoring (PostgreSQL)

`monitor.py` follows the LLM Zoomcamp pattern: a local Postgres table that records each generation for later evaluation.

**Table:** `chefbot_interactions`

| Column | Meaning |
|---|---|
| `id` | Transaction UUID (also returned as `X-Transaction-Id`) |
| `user_query` | Inventory string sent by the user |
| `dietary_choices` | Diet / allergy selections |
| `best_recipe_id` | Top Qdrant point id matched |
| `best_recipe_title` | Top recipe title |
| `llm_output` | Full streamed LLM response text |
| `response_latency_ms` | End-to-end latency for the request |
| `user_feedback` | `thumbs_up` / `thumbs_down` (via `/api/feedback`) |
| `model_name` / `status` | Model used and outcome (`ok`, `retrieval_error`, ...) |

Logging is wired through FastAPI `BackgroundTasks` so it does not block the stream. If Postgres is down, the API still serves recipes (monitoring failures are non-fatal).

Local DB (port **5433** to avoid clashing with a host Postgres on 5432):

```bash
docker compose up -d
python -u monitor.py   # optional smoke test: creates table + sample row
```

---

## Repository layout

```text
ChefBot AI/
|-- app.py              # Streamlit UI (inventory, diets, live stream, feedback)
|-- main.py             # FastAPI: /api/generate-recipe + /api/feedback
|-- retrieval.py        # Async Qdrant search + Gemini embeddings (+ fallback)
|-- ingest.py           # Batch embed + upsert into chefbot_recipes
|-- monitor.py          # PostgreSQL interaction logging
|-- dataset/
|   |-- recipes.json    # Primary structured dataset (JSONL)
|   `-- recipes.csv     # Optional tabular twin (not used at runtime)
|-- docker-compose.yml  # Local Postgres for monitoring (:5433)
|-- requirements.txt
|-- .env                # Secrets / service URLs (not committed)
`-- .cursorrules
```

---

## Prerequisites

- Python **3.11+** (3.13 tested)
- **Docker** (for local Postgres monitoring)
- A **Gemini API key** ([Google AI Studio](https://aistudio.google.com/))
- A **Qdrant** instance (Cloud URL + API key, or local Docker on `:6333`)
- Dataset file at `dataset/recipes.json` (or `dataset/2_Recipe_json.json`)

---

## Configuration

Copy and fill `.env`:

```env
GEMINI_API_KEY="your-gemini-api-key"
QDRANT_URL="https://YOUR-CLUSTER.aws.cloud.qdrant.io"
QDRANT_API_KEY="your-qdrant-api-key"
CHEFBOT_API_URL="http://localhost:8000"
DATABASE_URL="postgresql://user:password@localhost:5433/chefbot_monitoring"
```

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Embeddings + Gemini 2.5 Flash generation |
| `QDRANT_URL` / `QDRANT_API_KEY` | Vector store for `chefbot_recipes` |
| `CHEFBOT_API_URL` | Streamlit -> FastAPI base URL |
| `DATABASE_URL` | Monitoring DB (`docker compose` maps Postgres to `localhost:5433`) |

---

## Setup

```bash
cd "ChefBot AI"
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Git Bash) / macOS / Linux
source .venv/Scripts/activate   # or: source .venv/bin/activate

pip install -r requirements.txt
docker compose up -d
```

Confirm Postgres is healthy:

```bash
docker compose ps
python -u monitor.py
```

---

## Launch the stack

### 0. Start monitoring database

```bash
docker compose up -d
```

### 1. Ingest recipes into Qdrant (once / when refreshing)

```bash
python -u ingest.py
```

- Embeds the first **1,000** recipes (test limit) with `gemini-embedding-001`.
- Upserts into collection `chefbot_recipes` (768 dims, Cosine).
- **Resumes** automatically if interrupted (free-tier quota). Use `--recreate` only for a full rebuild:

```bash
python -u ingest.py --recreate
```

### 2. Start the FastAPI backend

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

> Use `main:app` (module path). `uvicorn main.py:app` will fail to import.

On startup, FastAPI calls `init_monitoring_table()` so `chefbot_interactions` exists.

- Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Start the Streamlit UI

```bash
streamlit run app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501), enter inventory, select diets/allergies, then **Generate Recipe**. Use the thumbs control afterward to write feedback into Postgres.

### Quick API smoke test

```bash
curl -N -D - -X POST "http://127.0.0.1:8000/api/generate-recipe" ^
  -H "Content-Type: application/json" ^
  -d "{\"inventory\":[\"chicken\",\"garlic\",\"tomato\"],\"dietary_choices\":\"high protein\",\"limit\":5}"
```

Look for response header `X-Transaction-Id`, then:

```bash
curl -X POST "http://127.0.0.1:8000/api/feedback" ^
  -H "Content-Type: application/json" ^
  -d "{\"transaction_id\":\"PASTE-UUID-HERE\",\"feedback\":\"thumbs_up\"}"
```

*(Git Bash / macOS / Linux: use `\` line continuations and single-quoted JSON.)*

---

## API contract

`POST /api/generate-recipe`

```json
{
  "inventory": ["chicken", "garlic", "tomato"],
  "dietary_choices": "high protein, dairy-free",
  "limit": 5
}
```

Response: `text/plain` **stream** with recipe narrative tokens, then a tool-calculated macros appendix. Header: `X-Transaction-Id`.

`POST /api/feedback`

```json
{
  "transaction_id": "uuid",
  "feedback": "thumbs_up"
}
```

Allowed feedback values: `thumbs_up`, `thumbs_down`.

---

## Design principles

- **Grounding first:** generation is constrained to Qdrant-retrieved recipes.
- **Typed data plane:** JSON arrays in payloads enable reliable ingredient filters.
- **Stream by default:** Gemini 2.5 Flash + FastAPI `StreamingResponse` + Streamlit `st.write_stream`.
- **Tools for facts:** macros come from Python lookup math, not model imagination.
- **Observe everything:** Postgres logs query, match, output, latency, and feedback.
- **Degrade gracefully:** embed quota exhaustion falls back to filter-only retrieval; monitoring outages do not take down the API.
- **Modern GenAI SDK:** `google-genai` only (see `.cursorrules`).

---

## License

Proprietary / all rights reserved unless otherwise specified by the project owner.
