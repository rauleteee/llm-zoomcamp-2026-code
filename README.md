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
| 02 | Vector Search | ⏳ Pending | Embeddings, vector DBs, semantic search |
| 03 | Orchestration | ⏳ Pending | Building larger LLM pipelines |
| 04 | Evaluation | ⏳ Pending | Measuring quality, LLM-as-judge |
| 05 | Monitoring | ⏳ Pending | Observability for LLM apps |
| 06 | Best Practices | ⏳ Pending | Hybrid search, prompt engineering, cost control |
| 07 | Project Example | ⏳ Pending | End-to-end reference project |

---

## Repository layout

```
llm-zoomcamp-2026-code/
├── 01-agentic-rag/         # Module 1 — completed
│   ├── ingest.py           # Load lesson markdown from the course repo, build minsearch index
│   ├── rag_helper.py       # RAG class: search → build context → prompt → LLM, returns answer + token usage
│   ├── agent.py            # Agentic version with a search tool (toyaikit)
│   └── rag_ingest.ipynb    # Walkthrough notebook covering Q1–Q6 of the homework
│
├── 02-vector-search/       # (pending)
├── ...
│
├── .env                    # API key + base URL — not committed
├── .gitignore
├── pyproject.toml          # uv-managed project metadata
├── uv.lock
└── README.md               # this file
```

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.13 |
| Env / deps | [`uv`](https://github.com/astral-sh/uv), `.venv` |
| Notebooks | JupyterLab |
| LLM access | `openai` SDK against an OpenAI-compatible endpoint (`devstral` model via course infrastructure) |
| Retrieval | [`minsearch`](https://github.com/alexeygrigorev/minsearch) — tiny in-memory full-text index |
| Data loading | [`gitsource`](https://github.com/alexeygrigorev/gitsource) — pull files straight from a GitHub repo at a pinned commit |
| Agents | [`toyaikit`](https://github.com/alexeygrigorev/toyaikit) — minimal function-calling agent framework |
| HTTP | `httpx` (with custom TLS settings for the course's private CA) |

---

## Getting started

### 1. Clone and enter the project

```bash
git clone <this-repo-url>
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

## Conventions used in this repo

- One folder per module, named with a two-digit prefix matching the course (`01-`, `02-`, ...).
- Notebooks are scratch space and walkthroughs; reusable logic lives in `.py` files.
- Token-counting and cost-related code reads `response.usage` directly rather than estimating.
- All external LLM calls go through a single `OpenAI` client configured from `.env`.

---

## What I'm learning

Beyond the course content itself, this repo doubles as a sandbox for general AI-engineering hygiene:

- Treating prompts as code — version them, test them, measure them.
- Keeping retrieval and generation decoupled, so I can swap either independently.
- Watching token counts the way I'd watch latency in a regular backend.
- Letting models "think out loud" via tool calls instead of cramming everything into one prompt.

---

## Acknowledgements

- The [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) team at [DataTalksClub](https://datatalks.club/) for putting together a genuinely excellent free course.
- [Alexey Grigorev](https://github.com/alexeygrigorev) for the supporting libraries (`minsearch`, `gitsource`, `toyaikit`) that make the course's project-based approach possible.

---

## License

Personal learning project — code is mine to share, but the course materials referenced throughout belong to DataTalksClub under their respective licenses.

---

*🚧 Work in progress — updated as I complete each module.*
