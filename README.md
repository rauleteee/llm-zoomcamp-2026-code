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
| 02 | **Vector Search** | ✅ Completed | ONNX embeddings, hand-rolled cosine, minsearch, hybrid search with RRF |
| 03 | **Orchestration** | ✅ Completed | Kestra flows, RAG vs agentic AI, multi-agent systems |
| 04 | **Evaluation** | ✅ Completed | Ground truth generation, Hit Rate / MRR, hybrid wins (k=1) |
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
│   ├── download.py             # Fetches the ONNX model from HuggingFace
│   ├── embedder.py             # Embedder class (encode / encode_batch)
│   ├── homework.ipynb          # Full Q1–Q6 walkthrough: by-hand → library → hybrid
│   └── README.md               # Module-specific deep dive
│
├── 03-orchestration/           # Module 3 — completed
│   ├── flows/                  # Kestra YAML flows
│   └── README.md               # Module-specific deep dive
│
├── 04-evaluation/              # Module 4 — completed
│   ├── ground-truth.csv        # 360 LLM-generated questions, each tagged with source filename
│   ├── homework.ipynb          # Q1–Q6 walkthrough: ground-truth gen → Hit Rate / MRR → hybrid tuning
│   └── README.md               # Module-specific deep dive
│
├── bacterio_compat.py          # Shim: course helpers re-implemented on chat.completions for the course endpoint
│
├── 05-monitoring/              # (pending)
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
| Structured output | Pydantic + Chat Completions JSON mode (via `bacterio_compat.py`) |
| Keyword retrieval | [`minsearch`](https://github.com/alexeygrigorev/minsearch) — tiny in-memory full-text + vector index |
| Embeddings | [`onnxruntime`](https://onnxruntime.ai/) + `tokenizers` — `all-MiniLM-L6-v2` on CPU, no GPU/PyTorch/CUDA |
| Data loading | [`gitsource`](https://github.com/alexeygrigorev/gitsource) — pull files from a GitHub repo at a pinned commit |
| Agents | [`toyaikit`](https://github.com/alexeygrigorev/toyaikit) — minimal function-calling agent framework |
| Orchestration | [Kestra](https://kestra.io) — workflow engine with native AI Agent / RAG / MCP plugins |
| LLM providers (via Kestra) | Google Gemini, OpenAI; Tavily for web search |
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
uv run jupyter lab        # open rag_ingest.ipynb
```

Each module folder also has its own `README.md` with module-specific setup and conclusions.

---

## Module 1: Agentic RAG

The first module builds a Retrieval-Augmented Generation system over the course lessons themselves, then turns it into an autonomous agent.

### What's inside

1. **Data ingestion** — `ingest.py` pulls every `.md` lesson from the course repo at a pinned commit and parses them into documents.
2. **Indexing** — builds a `minsearch.Index` with `content` as a text field and `filename` as a keyword field.
3. **RAG pipeline** — `rag_helper.py` defines a `RAG` class that wires together search, context building, prompt construction, and LLM calls. Returns both the answer and the token usage.
4. **Chunking** — splits long lessons into overlapping windows to keep prompts smaller and retrieval more precise.
5. **Agentic version** — `agent.py` exposes the chunked search as a tool and lets the LLM call it autonomously, iterating until it has enough context to answer.

### Key takeaways

- **Chunking dramatically reduces prompt size** without hurting answer quality — I saw ~3× fewer input tokens after switching to 2000-char windows with 1000-char overlap.
- **Agentic loops trade cost for adaptability.** A plain RAG pipeline runs once; an agent decides when (and how often) to search, and can retry with different keywords if results are weak.
- **Tool schemas come from type hints and docstrings.** Frameworks like `toyaikit` introspect the function's signature to build the JSON schema the LLM sees — clean type hints and good docstrings *are* the contract.

---

## Module 2: Vector Search

Module 2 trades keyword matching for semantic search. Same dataset (the chunked course lessons from module 1), but instead of matching words we match *meaning* — by embedding text into a vector space and finding nearest neighbours. The homework redo uses the **lightweight ONNX runtime** (no PyTorch, no CUDA — install is ~30× smaller than `sentence-transformers`).

### What's inside

1. **ONNX embedder** — `download.py` pulls `Xenova/all-MiniLM-L6-v2`; `embedder.py` provides a clean `encode` / `encode_batch` interface.
2. **Hand-rolled vector search** — build the corpus matrix, score with `X.dot(v)`, no library involved. The whole point: prove to yourself that vector search is just a matrix multiply.
3. **Library-backed search** — same operation through `minsearch.VectorSearch`, showing that the library is bookkeeping, not magic.
4. **Vector vs text comparison** — running the same query against `VectorSearch` and `Index` to see where each method wins.
5. **Hybrid search with RRF** — combining vector and text rankings using Reciprocal Rank Fusion (`1 / (k + rank)`, `k=60`). Documents that appear in both lists rise to the top by consensus.

Full walkthrough in [`02-vector-search/README.md`](./02-vector-search/README.md).

### Key takeaways

- **The embedder is the intelligence; everything else is logistics.** Pick the model carefully; the database is an ops decision.
- **Cosine similarity is just a dot product when vectors are normalized.** The single fact that makes semantic search fast at scale.
- **Vector and text search are complementary, not competitive.** Text wins on exact terms (names, codes, jargon). Vector wins on meaning and paraphrasing. Hybrid wins almost everywhere.
- **Hybrid search is the production default.** RRF is one of the simplest and most effective ways to fuse multiple rankings — and it works because consensus across methods beats individual brilliance.
- **Start with text search; upgrade only when it breaks.** Vector search adds real overhead. Most products handle 80% of queries with keyword search at a fraction of the operational cost.

---

## Module 3: Orchestration

Module 3 zooms out from "build a single RAG call" to "operate a workflow that may or may not need AI at all." The tool here is **Kestra**, an open-source orchestrator with native plugins for AI agents, RAG, content retrievers, and MCP tools.

### What's inside

YAML flows that progressively layer in capabilities:

1. **Plain chat** — direct LLM call, no retrieval. Baseline.
2. **Chat with RAG** — same model, but grounded in indexed documentation. Shows the accuracy lift from giving the model real context.
3. **Simple agent** — declarative LLM with structured prompts.
4. **Web research agent** — agent decides autonomously when to invoke the Tavily search tool. The flow specifies the goal; the model figures out the how.
5. **Multi-agent system** — a senior analyst agent delegates web research to a specialized agent exposed as a tool (`io.kestra.plugin.ai.tool.AIAgent`).

### Key takeaways

- **Context is usually the bug.** When an LLM gives wrong answers, the model itself is rarely at fault — what's missing is current data, the right tool, or grounded sources. RAG and tools are how you fix that.
- **You specify the goal, the agent decides the how.** In an agentic flow, there's no explicit "call search now" step. The agent is given a goal, a system message that nudges it, and a set of capabilities. The execution path emerges at runtime.
- **Agents-as-tools is a powerful composition pattern.** A specialist agent (narrow scope, narrow toolkit) wrapped as a tool for an orchestrator agent (broad scope, planning) cleanly separates concerns.
- **Agents are non-deterministic. That's a feature *and* a bug.** Great for research, exploration, fuzzy questions. Terrible for financial reporting, audits, regulated workflows. For those, use traditional task-based pipelines: explicit steps, same inputs → same outputs, code-reviewable, auditable.
- **Agents aren't an upgrade. They're a tool for a specific class of problem.** Pick based on requirements, not fashion.

---

## Module 4: Evaluation

Modules 1 and 2 built three search methods (keyword, vector, hybrid) and left the most important question open: **which one actually works best?** Module 4 closes that loop. We generate a ground-truth question set with an LLM, then measure each method against it with Hit Rate and MRR.

### What's inside

1. **Ground-truth generation** — for each lesson page, the LLM writes 5 plausible student questions answered by that page. Because the question came from a known page, we know the correct answer in advance. Pattern: **A → Q\* → A'**.
2. **Structured output via Pydantic** — questions come back as typed `Questions` objects, not free-form text. Reliable downstream parsing.
3. **Search evaluation** — `compute_relevance`, `hit_rate`, `mrr`, and a generic `evaluate(ground_truth, search_function)` wrapper that runs any search method against the full 360-question set.
4. **Hybrid tuning** — sweep over the RRF constant `k` ∈ {1, 50, 100, 200} and pick the value that maximizes MRR.

Full walkthrough in [`04-evaluation/README.md`](./04-evaluation/README.md).

### Results

| Method | Hit Rate | MRR |
|---|---:|---:|
| Text search | 0.758 | 0.594 |
| Vector search | 0.725 | 0.549 |
| **Hybrid (k=1)** | **0.839** | **0.648** |

### Key takeaways

- **Measurement overrides intuition.** Going in, the expectation was "vector search will beat text search on paraphrased questions." The data said otherwise — text outperformed vector on this corpus. The only way to know was to measure.
- **Hybrid wins, but the gap is smaller than the hype suggests.** RRF lifts MRR ~9% above text-only. Real, but not magical. In a tight ops budget, text-only is a defensible choice.
- **Why hybrid wins isn't sophistication — it's consensus.** Two independent methods agreeing on a result is a stronger signal than either method's individual confidence.
- **Synthetic data inflates the numbers.** LLM-generated questions share vocabulary patterns with their source documents. Treat these scores as upper bounds; real user queries will perform worse.
- **Search is the foundation everything else stands on.** If retrieval is bad, no amount of prompt engineering or agent orchestration can save the answer. That's why this module spends the most time on retrieval before touching RAG quality or agent metrics.

---

## A note on `bacterio_compat.py`

The course's helpers assume the OpenAI Responses API (`responses.parse` with `text_format=PydanticModel`). The course endpoint used for this work only fully supports Chat Completions, and even its partial Responses support doesn't enforce structured output. So `bacterio_compat.py` ships drop-in replacements (`llm_structured`, `llm_structured_retry`) built on `chat.completions.create` with `response_format={"type": "json_object"}` and the Pydantic schema embedded in the system prompt.

Same function names, same arguments — notebooks just import from `bacterio_compat` instead of `evaluation_utils`. Anyone running this on the real OpenAI API can swap the import back.

---

## Conventions used in this repo

- One folder per module, named with a two-digit prefix matching the course (`01-`, `02-`, ...).
- Each module folder has its own `README.md` with the deep dive; this root README is the table of contents.
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
- Treating the database under the embeddings as an ops choice, not a modeling choice.
- Picking the right execution model — deterministic pipeline vs. agentic loop — based on the requirement, not the hype.
- Defining "good" as a number before iterating, instead of trusting intuition.

---

## Acknowledgements

- The [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) team at [DataTalksClub](https://datatalks.club/) for putting together a genuinely excellent free course.
- [Alexey Grigorev](https://github.com/alexeygrigorev) for the supporting libraries (`minsearch`, `gitsource`, `toyaikit`) that make the course's project-based approach possible.
- The [Kestra](https://kestra.io) team for an orchestrator that takes AI workflows seriously as first-class citizens.
- The maintainers of [`onnxruntime`](https://onnxruntime.ai), [`sqlite-vec`](https://github.com/asg017/sqlite-vec), and [`pgvector`](https://github.com/pgvector/pgvector) — vector search would be a much heavier lift without them.

---

## License

Personal learning project — code is mine to share, but the course materials referenced throughout belong to DataTalksClub under their respective licenses.

---

*🚧 Work in progress — updated as I complete each module.*