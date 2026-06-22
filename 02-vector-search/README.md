# Module 2 — Vector Search (Homework)

Homework redo of LLM-Zoomcamp Module 2 using the **lightweight ONNX runtime** instead of `sentence-transformers`.

Both produce identical vectors. The ONNX path needs no PyTorch, no CUDA, and the install is ~30× smaller — it runs anywhere, including a basic Codespace.

This folder walks through the full retrieval pipeline: hand-rolled dot products → library-backed vector search → keyword search → hybrid fusion. The notebook is a sequential narrative; each cell builds on the previous one.

---

## Setup

```bash
cd 02-vector-search

# Helper scripts from the course repo
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/02-vector-search/embed
wget $PREFIX/download.py
wget $PREFIX/embedder.py

# Download the ONNX model (~90 MB, one-time)
uv run python download.py

# Launch the notebook
uv run jupyter lab homework.ipynb
```

Dependencies (already in the project's `pyproject.toml`):

```
onnxruntime  tokenizers  numpy  tqdm  minsearch  gitsource
```

---

## What's in this folder

```
02-vector-search/
├── download.py        # Fetches Xenova/all-MiniLM-L6-v2 from HuggingFace
├── embedder.py        # Embedder class with `encode` / `encode_batch`
├── homework.ipynb     # Sequential walkthrough of Q1–Q6
├── onnx/              # Model weights (gitignored, ~90 MB)
└── README.md          # this file
```

---

## What each question taught me

### Q1 — Embedding a query

Embedded *"How does approximate nearest neighbor search work?"* with the ONNX `Embedder`. Returned a `(384,)` float vector.

**Takeaway:** an embedding is just a list of numbers. The "intelligence" comes from a model that has learned to put semantically similar text close together in that 384-dimensional space. The vector itself is opaque to humans; we only care about how vectors relate to each other.

### Q2 — Cosine similarity, by hand

Embedded a specific lesson page and compared it to the Q1 query with a dot product.

**Takeaway:** when vectors are normalized to unit length, **cosine similarity is just a dot product** — no division, no trigonometry. That's why every modern embedder ships normalized vectors by default. It turns "how related are these two pieces of text?" into a single multiplication, which is the entire reason vector search is fast at scale.

### Q3 — Chunking and search by hand

Re-chunked the corpus (`size=2000, step=1000`), embedded every chunk, stacked them into a matrix `X`, and ranked by `X · v`.

**Takeaway:** a "vector database" is mostly a matrix multiplication. Once you have a corpus matrix `X` (shape `n_chunks × 384`) and a query vector `v`, **`scores = X.dot(v)`** ranks every document in one line. Everything a vector DB does — indexes, ANN structures, sharding — is engineering on top of that core operation, optimizing it for billions of rows instead of hundreds.

Chunking also mattered: page-level embeddings dilute a single page's many topics into one blurry vector. Chunking sharpens the signal.

### Q4 — Vector search with `minsearch.VectorSearch`

Same operation as Q3, but using `minsearch.VectorSearch` instead of hand-rolled NumPy.

**Takeaway:** libraries don't change what's happening — they just clean up the bookkeeping (storing the payload alongside the vectors, returning top-k with metadata, etc.). The principle from Q3 still applies: search is a dot product. **In production you use the library; in learning you should always do it once by hand first**, so the library doesn't feel like magic.

### Q5 — Vector vs text search

Indexed the same chunks with `minsearch.Index` (keyword search) and ran the query *"How do I store vectors in PostgreSQL?"* through both.

**Takeaway — and the most important one of the homework:**

> **Vector search and text search are not better/worse — they are complementary.**

- **Text search** wins on exact terms (product codes, names, technical jargon). It found the pgvector lesson because the literal word "PostgreSQL" appears there.
- **Vector search** wins on meaning and paraphrasing. It surfaced lessons that *talk about* storing vectors without naming Postgres specifically — pages a keyword index would never find.

Neither method is sufficient on its own. That's the setup for Q6.

### Q6 — Hybrid search with Reciprocal Rank Fusion

Combined both result lists with RRF:

```
RRF(d) = sum over lists of 1 / (k + rank(d))     where k = 60
```

The winning file (`14-agentic-loop.md`) **wasn't first in either search individually** — it won because it appeared in the top results of *both*.

**Takeaway:** RRF is the elegant trick that makes hybrid search work. It throws away the raw scores (which live on different scales and can't be compared directly) and looks only at *position*. A document that two independent search methods both consider relevant is more trustworthy than one a single method ranks highly. **Consensus beats individual brilliance.**

The constant `k=60` (from the original 2009 paper) controls how harshly lower ranks are penalized. Higher `k` flattens the difference between rank 0 and rank 5; lower `k` makes the top spots disproportionately valuable. You rarely need to tune it.

---

## The pipeline, end to end

The homework climbs a ladder of abstraction:

| Question | Layer | What changes |
|---|---|---|
| Q1 | Raw embedding | One vector |
| Q2 | Hand cosine similarity | Two vectors, one dot product |
| Q3 | Hand vector search | One query against all chunks |
| Q4 | Library vector search | Same thing, less bookkeeping |
| Q5 | Vector vs text | Two complementary methods |
| Q6 | Hybrid (RRF) | Both methods, fused |

Every step is a thin wrapper around the previous one. There's no point where something magic happens — just more convenient interfaces on top of "embeddings + linear algebra."

---

## Overarching conclusions

1. **The embedder is the intelligence; everything else is logistics.** Choose the model carefully (size, language, domain). The database is an ops decision, not an ML one.

2. **Cosine similarity is "just" a dot product when vectors are normalized.** This single fact is why semantic search scales — the inner loop of every vector DB is a matrix multiply.

3. **Chunk before embedding.** A page covering many topics produces a vector that's an average of all of them — bad for retrieval. Smaller, overlapping chunks sharpen the signal.

4. **Vector and text search are not in competition.** They catch different kinds of relevance:
   - Text → exact terms, names, codes
   - Vector → meaning, intent, paraphrasing
   - Hybrid → both, fused

5. **Hybrid search is the production default.** Almost every serious search system (Elasticsearch, Vespa, Weaviate, modern Postgres setups) runs both methods and merges the results. RRF is one of the simplest and most effective ways to do that merge.

6. **Start with text search; upgrade only when it breaks.** Vector search adds an embedding model, vector storage, normalization, distance metrics, and operational overhead. For most products, keyword search handles 80% of queries at a fraction of the complexity. Only add the vector layer when you have evidence (real query logs) showing semantic matches matter and keyword search is missing them.

---

## What this homework changed in how I think

Before the module: *"Vector search is the new way to do search. RAG needs it. Use it everywhere."*

After the module: *"Vector search is one of two complementary tools. Text search still wins for exact terms. Hybrid is the default in production. The embedder matters more than the database. And the simplest path is still: start with text, measure where it fails, upgrade specifically there."*

Less hype, more nuance — which is the whole reason for doing the homework by hand instead of just running a library.

---

*Part of the [`llm-zoomcamp-2026-code`](../) learning-in-public project.*
