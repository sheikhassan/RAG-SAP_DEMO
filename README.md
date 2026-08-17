# Drive Medical SAP Training Assistant

Vibeathon 2026 — a **production-oriented RAG training chatbot** for SAP processes
(ECC + S/4HANA), grounded in Drive Medical's own documentation.

The stack is still **fully local and free** by default, but retrieval is no
longer a toy Chroma cosine search. It is a hybrid index with an HNSW
nearest-neighbor graph, BM25 keyword search, token-budgeted context, and
measurable retrieval quality.

| Layer            | Component                                                         |
|------------------|-------------------------------------------------------------------|
| Auth             | CSV-backed users, **PBKDF2-SHA256** + **JWT (HS256)**             |
| RBAC             | Per-role document visibility + upload permissions                 |
| Vector store     | **Qdrant** (embedded `./qdrant_db`, or a server via `QDRANT_URL`) |
| ANN index        | **HNSW** (Hierarchical Navigable Small World), cosine             |
| Keyword search   | **BM25** sparse vectors (FastEmbed `Qdrant/bm25`)                 |
| Fusion           | **Reciprocal Rank Fusion (RRF)** — hybrid = semantic + keyword    |
| Embeddings       | **sentence-transformers/all-MiniLM-L6-v2** (384-d, local)         |
| LLM              | **Ollama + qwen2.5:3b** with a 32k context budget                 |
| Metrics          | Hit Rate, P@k, R@k, MRR, nDCG@k, latency, context utilization     |
| UI               | Streamlit (Chat, Upload, Knowledge base, Metrics)                 |
| Doc ingestion    | Markdown / Text / PDF / DOCX with live re-indexing                |

---

## 1. Architecture at a glance

```
+------------+       +-------------+       +-------------------+
| Login page | --->  |    JWT      | --->  | Streamlit tabs    |
| (CSV +     |       | (role, exp) |       | - Chat            |
|  PBKDF2)   |       +-------------+       | - Upload          |
+------------+                              | - Knowledge base  |
                                           | - Metrics         |
                                           +---------+---------+
                                                     |
                          +--------------------------+--------------------+
                          |                          |                    |
                          v                          v                    v
                +--------------------+   +-----------------------+  +-------------+
                | hybrid retrieve()  |   | upload pipeline       |  | sidebar:    |
                | HNSW + BM25 + RRF  |   | save -> chunk ->      |  | RBAC + last |
                | role-filter        |   | dense+sparse upsert   |  | query stats |
                +---------+----------+   +-----------+-----------+  +-------------+
                          |
                          v
                +--------------------+
                | Qdrant             |
                | dense: HNSW/cosine |
                | sparse: BM25       |
                | payload: role,     |
                | system, source_id  |
                +---------+----------+
                          |
                          v
                +--------------------+
                | token packer       |
                | fit chunks into    |
                | LLM context window |
                +---------+----------+
                          |
                          v
                +--------------------+
                | qwen2.5:3b (Ollama)|
                | grounded answer +  |
                | citations          |
                +--------------------+
```

### Why hybrid search (and what "HSW" actually is)

**HNSW** — Hierarchical Navigable Small World — is the approximate
nearest-neighbor algorithm Qdrant uses for **semantic** search. It builds a
multi-layer graph so a query vector can jump long distances on the top layer,
then refine neighbors on lower layers. Tunables:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `HNSW_M` | 16 | Bi-directional links per node. Higher = better recall, more RAM. |
| `HNSW_EF_CONSTRUCT` | 128 | Build-time candidate list. Higher = better index quality, slower ingest. |
| `HNSW_EF_SEARCH` | 64 | Query-time candidate list. Higher = better recall, more latency. |

**BM25** is classic **keyword** search (the same family as Elasticsearch). It
excels at SAP T-codes (`MIRO`, `ME21N`, `F110`) and exact field names that
dense embeddings sometimes smear.

**Hybrid** runs both, then **Reciprocal Rank Fusion** merges the two ranked
lists. Switch modes in the sidebar: Hybrid / Semantic only / Keyword only.

---

## 2. Quick start

> Prereqs:
>
> 1. Python 3.10+ (project tested on 3.10).
> 2. **Ollama** running locally with `qwen2.5:3b` pulled:
>    ```bash
>    ollama pull qwen2.5:3b
>    ```

```powershell
# 1. Create + activate venv (Windows PowerShell)
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Index the bundled SAP docs into Qdrant (dense + BM25)
python ingest.py --reset

# 4. Optional: measure retrieval quality
python eval.py --compare

# 5. Launch
streamlit run app.py
# or always-correct-interpreter:
.\.venv\Scripts\python.exe -m streamlit run app.py
# or:
.\run_app.ps1
```

Open <http://localhost:8501> and sign in.

Embedded Qdrant locks the `./qdrant_db` folder. Stop Streamlit before
re-running `ingest.py`, or run Qdrant as a server (see §7).

---

## 3. Authentication & RBAC

### Demo accounts

`auth/users.csv` is auto-seeded on first run with these accounts. **Change all
passwords before any real deployment.**

| Username     | Password     | Role         | Can upload? | Sees docs tagged                                           |
|--------------|--------------|--------------|-------------|------------------------------------------------------------|
| `admin`      | `admin123`   | admin        | Yes (any)   | common, finance, procurement, planning, manager, hr, restricted |
| `hr`         | `hr123`      | hr           | Yes (hr+common) | common, hr                                            |
| `manager`    | `manager123` | manager      | No          | common, finance, procurement, planning, manager            |
| `alice`      | `alice123`   | finance      | No          | common, finance                                            |
| `bob`        | `bob123`     | procurement  | No          | common, procurement                                        |
| `carol`      | `carol123`   | planning     | No          | common, planning                                           |

### How it works

1. **Login** — username + password are checked against `auth/users.csv`
   (PBKDF2-SHA256, 200 000 iterations, 16-byte salt).
2. **JWT** — on success, a HS256 token is issued with payload
   `{sub, role, name, email, iat, exp, iss}` and stored in
   `st.session_state["token"]`. TTL is `JWT_TTL_HOURS` from `.env`
   (default **8 h**).
3. **Per-request check** — every Streamlit rerun re-verifies the token
   (signature + exp). Tampered or expired tokens drop the user back to
   the login page.
4. **RBAC at retrieval** — `src/config.ROLE_VISIBILITY` is the single
   source of truth. Qdrant payload filters restrict hits to
   `role IN visible_doc_roles(user.role)`, so the model **never even sees**
   documents outside the user's scope.
5. **Upload permissions** — `src/config.UPLOAD_ALLOWED_ROLES` controls
   who can upload and which categories they can target. The Upload tab
   is hidden for non-uploaders, and the API is double-checked in
   `src/upload.ingest_uploaded_file()`.

### Adding / changing users

Edit `auth/users.csv` directly (header:
`username,password_hash,role,full_name,email`). To regenerate a
password hash quickly:

```bash
python -c "from auth.auth import _hash_password; print(_hash_password('NEW_PASSWORD'))"
```

---

## 4. Document ingestion

### Static / batch (markdown under `data/`)

```bash
python ingest.py            # incremental upsert
python ingest.py --reset    # drop and rebuild collection
```

`data/<role>/*.md` files use YAML frontmatter:

```yaml
---
title: Vendor Invoice (MIRO)
role: finance
system: both
source_id: FI-AP-001
---
```

### Live upload (Admin or HR)

1. Sign in as `admin` or `hr`.
2. Open the **Upload** tab.
3. Choose:
   - **Document category** (only the categories your role is allowed to
     upload to are shown).
   - **SAP system** (ECC / S4HANA / both).
   - Optional **display title**.
   - The file (`.md`, `.txt`, `.pdf`, `.docx`).
4. Submit. The file is:
   - Saved under `data/<category>/<safe_name>`.
   - Text-extracted (`pypdf` or `python-docx` as needed).
   - Chunked (paragraph-aware, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`).
   - Embedded as a **dense** vector and a **BM25 sparse** vector.
   - Upserted into Qdrant — **immediately retrievable**, no restart.

If you re-upload a file with the same name, prior chunks for that
`source_id` are deleted first (clean replace).

---

## 5. RAG flow

For every chat message:

1. Identity / smalltalk short-circuits (`hello`, `who are you`, …).
2. Retrieve top-`TOP_K` chunks from Qdrant, **filtered to
   `ROLE_VISIBILITY[user.role]`** + optional SAP system.
   - **Hybrid (default):** HNSW semantic prefetch + BM25 prefetch → RRF.
   - **Semantic:** HNSW cosine only; drop hits below `MIN_DENSE_SCORE` (0.30).
   - **Keyword:** BM25 only (strong on T-codes and exact terms).
3. **Pack context** so retrieved text fits
   `min(MAX_CONTEXT_TOKENS, LLM_CONTEXT_WINDOW − system − query − generation reserve)`.
   Lowest-ranked chunks are dropped rather than overflowing the model.
4. Build the grounded prompt and call **Ollama → qwen2.5:3b** with
   `num_ctx=LLM_CONTEXT_WINDOW` and low temperature. The system prompt
   requires answers only from retrieved context, ending with
   `Sources: <source_id>, ...`.
5. Render the answer + an expandable **Sources** panel (title, role,
   system, file path, hybrid/semantic/keyword score).
6. Record **query metrics**: retrieve ms, generate ms, context tokens used,
   utilization, HNSW parameters, source ids.

---

## 6. Metrics

Open the **Metrics** tab, or run the CLI:

```powershell
python eval.py --compare
```

| Metric | What it tells you |
|--------|-------------------|
| **Hit Rate** | Share of questions where the expected `source_id` appears in top-k |
| **Precision@k** | Fraction of the top-k that are relevant |
| **Recall@k** | Fraction of relevant docs that made it into top-k |
| **MRR** | Mean Reciprocal Rank — how high the first relevant hit sits |
| **nDCG@k** | Discounted ranking quality (order-aware) |
| **Retrieve / generate ms** | Latency breakdown per chat turn |
| **Context utilization** | Tokens packed vs the allowed context budget |

The golden set in `src/eval_set.py` mixes paraphrases (semantic) and T-codes
(keyword) so you can see why hybrid usually wins.

---

## 7. Configuration (`.env`)

```dotenv
# LLM
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:3b

# Auth
JWT_SECRET=dev-secret-change-me-please-use-32-random-bytes
JWT_TTL_HOURS=8

# Hybrid search + HNSW
SEARCH_MODE=hybrid
HNSW_M=16
HNSW_EF_CONSTRUCT=128
HNSW_EF_SEARCH=64
TOP_K=5

# Context window (qwen2.5:3b = 32k)
LLM_CONTEXT_WINDOW=32768
MAX_CONTEXT_TOKENS=6000
GENERATION_RESERVE_TOKENS=1024

# Optional Qdrant server (otherwise embedded ./qdrant_db)
# QDRANT_URL=http://127.0.0.1:6333
# QDRANT_API_KEY=

# Optional Gemini fallback
# LLM_PROVIDER=gemini
# GOOGLE_API_KEY=...
# GEMINI_MODEL_NAME=gemini-2.5-flash
```

> **Always** set a strong `JWT_SECRET` for any non-demo deployment.

### Optional Qdrant server (recommended if ingest + UI run together)

```powershell
docker compose up -d
# then set QDRANT_URL=http://127.0.0.1:6333 in .env
python ingest.py --reset
```

---

## 8. Project layout

```
app.py                 # Streamlit entry: login + chat + upload + KB + metrics
ingest.py              # CLI: batch ingest data/ -> Qdrant (dense + BM25)
eval.py                # CLI: Hit Rate / MRR / nDCG for hybrid vs baselines
run_app.ps1            # PowerShell helper: always uses .venv interpreter
docker-compose.yml     # Optional Qdrant server
requirements.txt
.env / .env.example

auth/
  __init__.py
  auth.py              # PBKDF2 hashing, JWT issue/verify, CSV seed
  rbac.py              # visibility + upload permission helpers
  users.csv            # auto-seeded on first run (gitignored)

src/
  config.py            # paths, HNSW, hybrid, context window, RBAC, JWT
  embeddings.py        # SentenceTransformer dense wrapper
  sparse.py            # FastEmbed BM25 sparse wrapper
  document_loader.py   # md/txt/pdf/docx -> Chunk objects
  vector_store.py      # Qdrant client + hybrid / semantic / keyword query
  tokens.py            # token estimate + context packer
  metrics.py           # per-query stats + golden-set eval
  eval_set.py          # labeled retrieval questions
  ollama_client.py     # /api/chat HTTP client (num_ctx + temperature)
  rag_engine.py        # retrieve + pack + generate
  upload.py            # save -> chunk -> embed -> upsert pipeline

data/                  # SAP markdown docs (seed knowledge base)
  common/
  finance/
  procurement/
  planning/

qdrant_db/             # embedded Qdrant store (gitignored)
```

---

## 9. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ModuleNotFoundError: No module named 'sentence_transformers'` | Streamlit is using the system Python instead of the venv. Use `.\.venv\Scripts\python.exe -m streamlit run app.py` or `.\run_app.ps1`. |
| `Cannot reach Ollama` | Start the Ollama desktop app/service. Verify `Invoke-WebRequest http://127.0.0.1:11434/api/tags` returns 200. |
| `HTTP 404: model '...' not found` | The model in `.env` (`OLLAMA_MODEL`) is not pulled. Run `ollama pull qwen2.5:3b`. |
| Qdrant lock / "already accessed by another instance" | Embedded mode allows one process. Stop Streamlit before `ingest.py`, or use `QDRANT_URL` + Docker. |
| Wrong model used after editing `.env` | `app.py` loads `.env` **before** importing config; restart Streamlit. |
| Logged in but Upload tab missing | Your role is not in `UPLOAD_ALLOWED_ROLES`. Sign in as `admin` or `hr`. |
| User sees "no answer" but the doc exists | The doc's `role` tag is outside the user's `ROLE_VISIBILITY`. Either reclassify the doc, or sign in with a higher-privileged user. |
| Keyword mode misses a paraphrase | Expected — switch to **Hybrid** or **Semantic**. T-codes belong in Keyword/Hybrid. |

---

## 10. Demo script (4 steps)

1. Sign in as **Alice (`alice` / `alice123`)** — role = finance.
   Ask: *"How do I post a vendor invoice using MIRO?"* — get answer
   citing `FI-AP-001`. Open **Metrics** and note retrieve ms + context tokens.
2. Sign out, sign in as **Carol (`carol` / `carol123`)** — role =
   planning. Ask the same question. The retrieval hits zero finance
   docs (only `common` + `planning`), proving RBAC isolation.
3. Sign in as **Admin (`admin` / `admin123`)**. Open **Upload**, drop
   any HR PDF tagged as category `hr`. Open **Metrics** → *Run retrieval eval*
   and compare Hybrid vs Semantic vs Keyword.
4. Sign in as **HR (`hr` / `hr123`)**. Ask a question about that PDF —
   it appears with citations. No restart, no retrain.
