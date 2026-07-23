# ChefBot AI

Inventory-aware recipe app built with FastAPI, Qdrant, Gemini, PostgreSQL, and Streamlit.

Given what’s in the fridge (and any diet limits), ChefBot retrieves matching recipes from a vector index, generates an adapted dish with Gemini, appends tool-calculated macros, and logs each run for later review.

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
  RT -->|"rewrite → cosine + filter → rerank<br/>(filter-only fallback)"| QD
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
3. FastAPI calls `search_recipes()`: rewrite the inventory query, embed it, filter Qdrant payloads by inventory ingredients, re-rank by ingredient overlap, return top matches.
4. If Gemini embedding quota is exhausted, retrieval **falls back** to ingredient-filter-only Qdrant results (still re-ranked when possible).
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

`monitor.py` writes each generation to Postgres so you can inspect latency, feedback, and outputs later.

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

### Production monitoring (Vercel)

Use a hosted Postgres provider for live logs. Preferred: **Supabase**, **Neon**, or **Vercel Postgres**. Do **not** use Render for this project.

**Monitoring:** Postgres tables hold the logs.

- `chefbot_interactions` — every generation + feedback + latency
- `chefbot_evaluations` — offline judge scores
- thumbs feedback on every plate (`/api/feedback`)
- compact sidebar metrics via `GET /api/monitoring/summary`
- Streamlit **Monitoring** view via `GET /api/monitoring/dashboard`

1. Create a free Postgres project (Supabase is fine).
2. Copy the **connection pooler** URI (Supabase transaction mode is typically port `6543`).
3. Set `DATABASE_URL` in the Vercel project env (Production + Preview as needed).
4. Redeploy the API. On boot, `init_monitoring_table()` creates `chefbot_interactions` automatically.

`monitor.py` adds `sslmode=require` for remote hosts and disables prepared statements for Supabase/PgBouncer poolers.

### Streamlit monitoring dashboard

In the sidebar, switch **View → Monitoring**. Charts (7):

1. Interactions / day  
2. Avg latency / day  
3. Status breakdown  
4. Feedback breakdown (thumbs up / down / none)  
5. Judge verdicts  
6. Judge score averages  
7. Feedback rate / day  

API: `GET /api/monitoring/dashboard?days=14`

### Retrieval evaluation (offline)

`evaluate_retrieval.py` compares **three** retrieval approaches on a fixed inventory query set (`evals/retrieval_queries.json`):

| Mode | What it does |
|---|---|
| `hybrid` | Cosine rank **with** inventory `MatchText` filter (production default) |
| `vector_only` | Cosine rank **without** inventory filter |
| `filter_only` | Inventory filter scroll only (no vector ranking) |

**Relevance proxy:** a hit is relevant if ≥2 inventory ingredient tokens appear in the recipe's ingredients text (substring, case-insensitive). Metrics: **Hit@k**, **MRR**, **Precision@k**, and mean inventory **coverage**.

```bash
python -u evaluate_retrieval.py --k 5 --min-hits 2
```

**Latest run** (`evals/retrieval_results.json`, k=5, 12 queries):

| Mode | Hit@5 | MRR | P@5 | Coverage |
|---|---:|---:|---:|---:|
| **hybrid** ★ | 1.000 | 1.000 | 0.950 | 0.883 |
| vector_only | 1.000 | 1.000 | 0.950 | 0.883 |
| filter_only | 1.000 | 0.662 | 0.367 | 0.662 |

Production uses **`hybrid`** with **query rewriting** + **ingredient-overlap re-ranking** enabled (`retrieval.py`).

### Retrieval extras

| Feature | Implementation |
|---|---|
| Hybrid search | Dense cosine vectors **plus** inventory `MatchText` payload filter (evaluated vs `vector_only` / `filter_only`) |
| User query rewriting | `rewrite_query_for_retrieval()` expands fridge lists + diet into a richer embedding query (synonyms / cooking intent) |
| Document re-ranking | Over-fetch `3×k` candidates, then `rerank_by_ingredient_overlap()` blends vector score with inventory coverage |

Disable for experiments: `search_recipes_with_mode(..., rewrite_query=False, rerank=False)`.

### Prompt comparison (offline)

`evaluate_llm.py` compares two system prompts on the same retrieved context, then scores each answer with a Gemini judge:

| Approach | Intent |
|---|---|
| `grounded_structured` | Hard RAG constraints + structured sections + allergy discipline (production) |
| `loose_creative` | Casual home-cook; may invent complementary ingredients |

```bash
python -u evaluate_llm.py --limit 4 --pause 2
```

**Latest run** (`evals/llm_results.json`, 3 queries):

| Approach | Overall | Groundedness | Relevance | Safety | n |
|---|---:|---:|---:|---:|---:|
| **grounded_structured** ★ | 4.33 | 4.00 | 4.33 | 5.00 | 3 |
| loose_creative | 3.50 | 3.00 | 5.00 | 5.00 | 2* |

\* `loose_creative` missed q03 due to Gemini free-tier quota; still trails on overall/groundedness. Production keeps `prompts.DEFAULT_APPROACH = "grounded_structured"`.

`evaluate.py` remains available to judge **logged production** interactions in Postgres (`chefbot_evaluations`).

Run production-log judging from the **project root**:

```bash
cd "C:\Users\pc\Desktop\ChefBot AI"
.\.venv\Scripts\Activate.ps1
python -u evaluate.py --limit 20
```

- Default: only unevaluated `ok` interactions
- Re-judge everything in the batch: `python -u evaluate.py --limit 50 --all`

---

## What’s in the repo

| Area | Main files |
|---|---|
| Problem / flow | This README |
| Retrieval | `retrieval.py`, Qdrant `chefbot_recipes` |
| Retrieval eval | `evaluate_retrieval.py`, `evals/` |
| Generation eval | `evaluate_llm.py`, `evaluate.py` |
| UI / API | `app.py`, `main.py` |
| Ingest | `ingest.py`, `prepare_dataset.py` |
| Monitoring | `monitor.py`, Streamlit Monitoring view |
| Docker | `docker-compose.yml`, `Dockerfile.api`, `Dockerfile.ui` |
| Deps / data | `requirements.txt`, `dataset/sample_recipes.jsonl` |

Retrieval extras: hybrid search (vector + ingredient text filter), query rewrite, and overlap re-ranking.

### Screenshots

Add captures under `docs/screenshots/` if you want them in the README:

```text
docs/screenshots/
|-- cook-ui.png
|-- monitoring-dashboard.png
|-- sample-recipe.png
```

```markdown
![Cook UI](docs/screenshots/cook-ui.png)
![Monitoring](docs/screenshots/monitoring-dashboard.png)
![Sample recipe](docs/screenshots/sample-recipe.png)
```

---

## Repository layout

```text
ChefBot AI/
|-- app.py                 # Streamlit UI (inventory, diets, live stream, feedback)
|-- main.py                # FastAPI: /api/generate-recipe + /api/feedback
|-- retrieval.py           # Async Qdrant search + Gemini embeddings (+ fallback)
|-- ingest.py              # Batch embed + upsert into chefbot_recipes
|-- monitor.py             # PostgreSQL interaction logging
|-- evaluate.py             # Offline LLM-as-a-judge over logged interactions
|-- evaluate_retrieval.py   # Compare hybrid / vector_only / filter_only
|-- evaluate_llm.py         # Prompt comparison with shared retrieval + judge
|-- prompts.py              # Prompt approaches; DEFAULT_APPROACH = production winner
|-- evals/
|   |-- retrieval_queries.json
|   |-- retrieval_results.json  # From evaluate_retrieval.py
|   `-- llm_results.json        # From evaluate_llm.py
|-- prepare_dataset.py      # Fetch/copy recipe corpus for ingest
|-- dataset/
|   |-- README.md          # Data access + schema
|   |-- sample_recipes.jsonl  # Tracked 200-recipe demo corpus
|   |-- recipes.json       # Full/local dump (gitignored; via prepare_dataset.py)
|   `-- recipes.csv        # Optional tabular twin (not used at runtime)
|-- docker-compose.yml     # Full local stack: Postgres + Qdrant + API + UI
|-- Dockerfile.api         # FastAPI / uvicorn image
|-- Dockerfile.ui          # Streamlit image
|-- .dockerignore
|-- requirements.txt       # Pinned direct dependencies
|-- requirements.lock.txt  # Full transitive pin set (optional)
`-- .env                   # Secrets / service URLs (not committed)
```

---

## Prerequisites

- Python **3.11+** (3.13 tested) — optional if you run via Docker only
- **Docker** + Docker Compose (full local stack)
- A **Gemini API key** ([Google AI Studio](https://aistudio.google.com/))
- Dataset via `prepare_dataset.py` (tracked sample or full dump URL)

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
| `CHEFBOT_API_URL` | Streamlit -> FastAPI base URL (local or Vercel) |
| `DATABASE_URL` | Monitoring DB: local `docker compose` (`localhost:5433`) or hosted Supabase/Neon/Vercel Postgres (not Render) |
| `CHEFBOT_DATASET_URL` | Optional HTTP(S) URL for `prepare_dataset.py` full corpus download |
| `CORS_ORIGINS` | Comma-separated browser origins allowed by FastAPI CORS |

---

## Dataset (reproducibility)

Full recipe dumps (~62k rows, ~80MB+) are **not** committed. To run locally:

```bash
# Demo corpus (200 recipes, tracked in git)
python -u prepare_dataset.py --from-sample

# Or full dump you already have / host yourself
# copy into dataset/recipes.json
# or:
python -u prepare_dataset.py --url "https://YOUR-HOST/recipes.jsonl" --force
```

Details and schema: [`dataset/README.md`](dataset/README.md).

**Dependencies** are pinned in `requirements.txt`. For an exact transitive freeze (CI/repro machines):

```bash
pip install -r requirements.txt
# optional stricter install:
pip install -r requirements.lock.txt
```

---

## Remote fullstack (Streamlit Cloud + Vercel)

1. **Backend (Vercel)**
   - Deploy this repo (Vercel detects `main.py` FastAPI `app`).
   - Set env vars in Vercel: `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `DATABASE_URL` (hosted Postgres URI), `CORS_ORIGINS`.
   - For `DATABASE_URL`, use Supabase/Neon/Vercel Postgres pooler URI (not Render). Table bootstrap is automatic on deploy.
   - Recommended `CORS_ORIGINS`:
     `https://chefbot-ai-9v272ty2jahksxappfpvhqg.streamlit.app,http://localhost:8501`
   - `vercel.json` sets `maxDuration: 60` for long recipe streams.

2. **Frontend (Streamlit Community Cloud)**
   - App: https://chefbot-ai-9v272ty2jahksxappfpvhqg.streamlit.app/
   - Backend: https://chef-bot-ai-one.vercel.app/
   - In **App settings -> Secrets**, set:
     ```toml
     CHEFBOT_API_URL = "https://chef-bot-ai-one.vercel.app"
     ```
   - Redeploy / reboot the Streamlit app after saving secrets.

3. **Note on CORS**
   - Streamlit Cloud calls your API from Python (`httpx`) on Streamlit's servers (server-to-server), so CORS is not always required for that path.
   - CORS is still configured so browser clients, previews, and future web UIs can call the Vercel API safely from the Streamlit origin.

---

## Setup

```bash
cd "ChefBot AI"
cp .env.example .env   # then set GEMINI_API_KEY
python -u prepare_dataset.py --from-sample   # or use your full recipes.json
```

### Option A — full stack with Docker

Brings up **Postgres**, **Qdrant**, **FastAPI**, and **Streamlit**:

```bash
docker compose up --build -d
```

| Service | URL |
|---|---|
| UI | [http://localhost:8501](http://localhost:8501) |
| API | [http://localhost:8000/docs](http://localhost:8000/docs) |
| Qdrant | [http://localhost:6333/dashboard](http://localhost:6333/dashboard) |
| Postgres | `localhost:5433` (user/password/chefbot_monitoring) |

Compose overrides `DATABASE_URL` / `QDRANT_URL` to Docker DNS (`postgres`, `qdrant`).  
`GEMINI_API_KEY` is read from your host `.env`.

Ingest once (from the host, against the compose Qdrant):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # or: source .venv/bin/activate
pip install -r requirements.txt
python -u prepare_dataset.py --from-sample   # skip if dataset/recipes.json already exists
# Ensure .env has QDRANT_URL=http://localhost:6333 for host-side ingest
python -u ingest.py
```

Stop the stack:

```bash
docker compose down
```

### Option B — local Python + compose dependencies

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Git Bash) / macOS / Linux
source .venv/Scripts/activate   # or: source .venv/bin/activate

pip install -r requirements.txt
docker compose up -d postgres qdrant
```

Confirm Postgres is healthy:

```bash
docker compose ps
python -u monitor.py
```

---

## Launch the stack (Option B — without containerizing the apps)

### 0. Start Postgres + Qdrant

```bash
docker compose up -d postgres qdrant
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
- **GenAI SDK:** `google-genai` (`from google import genai`).

---

## License

Proprietary / all rights reserved unless otherwise specified by the project owner.
