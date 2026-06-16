# LLM-Zoomcamp 2026

My working repository for [DataTalksClub's **LLM Zoomcamp**](https://github.com/DataTalksClub/llm-zoomcamp) — a free, project-based course on building real applications with Large Language Models.

This repo is my hands-on path from *"I've used ChatGPT"* to *"I can design and ship LLM-powered systems."* Each module ships as its own folder with code, notebooks, notes, and homework solutions.

---

## About this journey

I'm working through the zoomcamp to acquire practical AI knowledge — not by reading papers, but by building things that retrieve documents, talk to models, evaluate themselves, and act as agents. The course is structured as a series of modules, each with a video lesson and a homework. I'm tackling them in order, committing my work as I go.

This README will grow as more modules are completed.

---

## Progress

| # | Module | Status | Highlights |
|---|--------|--------|------------|
| 01 | **Agentic RAG** | ✅ Completed | RAG over course lessons, chunking, agent with tool use |
| 02 | **Vector Search** | ✅ Completed | Embeddings, ONNX local inference, in-memory / sqlite-vec / pgvector |
| 03 | Orchestration | ⏳ Pending | Building larger LLM pipelines |
| 04 | Evaluation | ⏳ Pending | Measuring quality, LLM-as-judge |
| 05 | Monitoring | ⏳ Pending | Observability for LLM apps |
| 06 | Best Practices | ⏳ Pending | Hybrid search, prompt engineering, cost control |
| 07 | Project Example | ⏳ Pending | End-to-end reference project |

---

## Repository layout

```
llm-zoomcamp-2026-code/
├── 01-agentic-rag/             # Module 1 — completed
│   ├── ingest.py               # Load lesson markdown, build minsearch index
│   ├── rag_helper.py           # RAG class: search → context → prompt → LLM (returns answer + usage)
│   ├── agent.py                # Agentic version with a search tool (toyaikit)
│   └── rag_ingest.ipynb        # Walkthrough notebook (Q1–Q6 of the homework)
│
├── 02-vector-search/           # Module 2 — completed
│   ├── vector_search.py        # Embedder + three pluggable engines (in-memory, sqlite-vec, pgvector)
│   └── vector_search_demo.ipynb# Head-to-head notebook: same data, same embedder, three storage backends
│
├── 03-orchestration/           # (pending)
├── ...
│
├── .env                        # API key + base URL — not committed
├── .gitignore
├── pyproject.toml              # uv-managed project metadata
├── uv.lock
└── README.md                   # this file
```

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.13 |
| Env / deps | [`uv`](https://github.com/astral-sh/uv), `.venv` |
| Notebooks | JupyterLab |
| LLM access | `openai` SDK against an OpenAI-compatible endpoint (`devstral` via course infrastructure) |
| Keyword retrieval | [`minsearch`](https://github.com/alexeygrigorev/minsearch) — tiny in-memory full-text index |
| Embeddings | [`fastembed`](https://github.com/qdrant/fastembed) — ONNX models running on CPU, no GPU required |
| Vector storage | NumPy (in-memory), [`sqlite-vec`](https://github.com/asg017/sqlite-vec), [`pgvector`](https://github.com/pgvector/pgvector) |
| Data loading | [`gitsource`](https://github.com/alexeygrigorev/gitsource) — pull files from a GitHub repo at a pinned commit |
| Agents | [`toyaikit`](https://github.com/alexeygrigorev/toyaikit) — minimal function-calling agent framework |
| HTTP | `httpx` (with custom TLS settings for the course's private CA) |

---

## Getting started

### 1. Clone and enter the project

```bash
git clone https://github.com/rauleteee/llm-zoomcamp-2026-code.git
cd llm-zoomcamp-2026-code
```

### 2. Set up the environment

Using `uv` (recommended):

```bash
uv sync
source .venv/bin/activate
```

Or with vanilla Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt    # if present, otherwise install per-module
```

### 3. Configure secrets

Create a `.env` file at the project root:

```bash
OPENAI_API_KEY=your-key-here
BASE_URL=https://your-llm-endpoint/v1
```

`.env` is git-ignored — never commit it.

### 4. Run a module

```bash
cd 01-agentic-rag
jupyter lab        # open rag_ingest.ipynb
```

---

## Module 1: Agentic RAG

The first module builds a Retrieval-Augmented Generation system over the course lessons themselves, then turns it into an autonomous agent.

### What's inside

1. **Data ingestion** — `ingest.py` pulls every `.md` lesson from the course repo at a pinned commit and parses them into documents.
2. **Indexing** — builds a `minsearch.Index` with `content` as a text field and `filename` as a keyword field.
3. **RAG pipeline** — `rag_helper.py` defines a `RAG` class that wires together search, context building, prompt construction, and LLM calls. Returns both the answer and the token usage.
4. **Chunking** — splits long lessons into overlapping windows to keep prompts smaller and retrieval more precise.
5. **Agentic version** — `agent.py` exposes the chunked search as a tool and lets the LLM call it autonomously, iterating until it has enough context to answer.

### Run it

```bash
cd 01-agentic-rag
jupyter lab rag_ingest.ipynb         # full Q1–Q6 walkthrough
python agent.py                       # standalone agent run
```

### Key takeaways from this module

- **Chunking dramatically reduces prompt size** without hurting answer quality — I saw ~3× fewer input tokens after switching to 2000-char windows with 1000-char overlap.
- **Agentic loops trade cost for adaptability.** A plain RAG pipeline runs once; an agent decides when (and how often) to search, and can retry with different keywords if results are weak.
- **Tool schemas come from type hints and docstrings.** Frameworks like `toyaikit` introspect the function's signature to build the JSON schema the LLM sees — clean type hints and good docstrings *are* the contract.

---

## Module 2: Vector Search

Module 2 trades keyword matching for semantic search. Same dataset (the chunked course lessons from module 1), but instead of matching words, we match *meaning* — by embedding text into a vector space and finding nearest neighbours.

### What's inside

1. **Local ONNX embedder** — `vector_search.py` wraps `fastembed` with the `BAAI/bge-small-en-v1.5` model. Runs on CPU, no GPU, no API costs.
2. **Three storage engines** sharing one interface (`add` / `search` / `close`):
   - `InMemoryEngine` — NumPy matrix, dot-product search. Zero setup.
   - `SQLiteEngine` — `sqlite-vec` extension, embedded DB in a `.db` file.
   - `PGVectorEngine` — Postgres + `pgvector`, real database with cosine-distance indexes.
3. **Embedding sanity check** — short notebook section confirming that semantically related phrases really do produce close vectors (*"dog chasing a ball"* vs *"puppy playing fetch"* scores higher than either against *"financial market analysis"*).
4. **Head-to-head benchmark** — same query, same data, all three engines side by side. Latency, top result, top score.

### Run it

```bash
cd 02-vector-search
jupyter lab vector_search_demo.ipynb
```

For the pgvector engine you need Postgres running. The fastest path:

```bash
docker run -d --name pgvector \
    -e POSTGRES_PASSWORD=postgres \
    -p 5432:5432 \
    pgvector/pgvector:pg16
```

The notebook gracefully skips pgvector if it can't connect, so the rest still works.

### Key takeaways from this module

- **Embeddings are the interesting part; storage is plumbing.** Swap the database, keep the model, and you get the same neighbours. Picking a vector DB is an ops decision, not an ML one.
- **`sqlite-vec` is shockingly capable for personal projects.** A single file, no daemon, full vector search. Perfect for prototypes and edge deployments.
- **Local CPU embeddings are completely viable.** `fastembed` + ONNX runs in seconds on a laptop, sidestepping the cost and latency of calling an embeddings API.
- **Scores from different engines aren't directly comparable** (cosine vs L2 vs negative-distance). What matters is the *ordering*, and across all three engines, ordering is stable.

---

## Conventions used in this repo

- One folder per module, named with a two-digit prefix matching the course (`01-`, `02-`, ...).
- Notebooks are scratch space and walkthroughs; reusable logic lives in `.py` files.
- Token-counting and cost-related code reads `response.usage` directly rather than estimating.
- All external LLM calls go through a single `OpenAI` client configured from `.env`.
- Each module folder ships with at least one `.py` for reusable code and one `.ipynb` to walk through it.

---

## What I'm learning

Beyond the course content itself, this repo doubles as a sandbox for general AI-engineering hygiene:

- Treating prompts as code — version them, test them, measure them.
- Keeping retrieval and generation decoupled, so I can swap either independently.
- Watching token counts the way I'd watch latency in a regular backend.
- Letting models "think out loud" via tool calls instead of cramming everything into one prompt.
- Treating the database under the embeddings as an ops choice, not a modeling choice.

---

## Acknowledgements

- The [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) team at [DataTalksClub](https://datatalks.club/) for putting together a genuinely excellent free course.
- [Alexey Grigorev](https://github.com/alexeygrigorev) for the supporting libraries (`minsearch`, `gitsource`, `toyaikit`) that make the course's project-based approach possible.
- The [`sqlite-vec`](https://github.com/asg017/sqlite-vec), [`pgvector`](https://github.com/pgvector/pgvector), and [`fastembed`](https://github.com/qdrant/fastembed) maintainers — vector search would be a much heavier lift without them.

---

## License

Personal learning project — code is mine to share, but the course materials referenced throughout belong to DataTalksClub under their respective licenses.

---

*🚧 Work in progress — updated as I complete each module.*