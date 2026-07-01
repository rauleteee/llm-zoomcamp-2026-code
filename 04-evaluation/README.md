# Module 4 — Evaluation (Homework)

This folder is where the question *"which search method is best?"* finally gets a numerical answer.

For modules 1 and 2 we built three search methods — **keyword (text), vector, and hybrid** — and ended with an open question: which one actually works best? Up to now we've been comparing by intuition, by looking at single queries, by *vibes*. That stops here.

This homework generates a test dataset, runs all three search methods on it, and produces hard numbers. By the end, you can point at a row in a spreadsheet and say *"this one wins, by this much."* That's the entire point of evaluation.

---

## The mental model: why this is harder than it sounds

Evaluating a search system has a chicken-and-egg problem:

To measure quality, you need a set of **questions** paired with the **correct document** for each question. That's called **ground truth**. The catch is that this is exactly the thing you don't have — you have a knowledge base of documents, but no list of "people will ask X and the answer is Y."

The trick we use in this homework: **generate the questions from the documents themselves with an LLM.** For each lesson page, we ask an LLM to write 5 plausible student questions that the page answers. Because the question came from a known page, we know in advance which page is the correct answer.

This pattern is called **A → Q\* → A'**:

- **A** is the original document (we have it)
- **Q\*** is a question generated *from* A by an LLM
- **A'** is what our search system returns when we ask Q\*

If A' contains A, search worked. If it doesn't, it didn't. Repeat across 360 questions and you get a real, comparable number.

---

## What's in this folder

```
04-evaluation/
├── ground-truth.csv         # 360 generated questions, each tagged with its source filename
├── bacterio_compat.py       # Local adapter — uses chat.completions instead of responses.parse
├── evaluation_utils.py      # Course-provided helpers (mostly bypassed because of API mismatch)
├── rag_helper.py            # Reused from module 1
├── homework.ipynb           # Full Q1–Q6 walkthrough
└── README.md                # this file
```

---

## How each question fits into the bigger picture

### Q1 — Generate questions for 3 pages

We generated 5 questions for each of the first three lesson pages and measured the input tokens.

**Result:** average input tokens = **1583**, closest option → **1400**.

**What it means:** every time we ask the LLM to generate questions, we send the entire lesson page (plus the system prompt and JSON schema) as input. That's why each call costs ~1500 tokens of input. Doing this for all 72 pages would be ~110K input tokens — small money on a real provider, but worth knowing.

The point of this question isn't really the number — it's the realization that **synthetic ground truth has a real cost**, and the cost scales linearly with the corpus size. Now we know what we'd be signing up for if we wanted to redo this for a larger system.

### Q2 — First text-search result for the first ground truth question

The question was *"What exactly is a retrieval-augmented generation system, and why does it help with answers that the model wouldn't know on its own?"* — generated from `01-agentic-rag/lessons/01-intro.md`.

**Text search returned:** `01-agentic-rag/lessons/03-rag.md` at rank 1. **Wrong.**

(Well, *technically* wrong by the homework's definition. The 03-rag.md page is also a perfectly reasonable answer to that question — but the ground truth says "intro," so we count it as a miss.)

This is a quirk worth keeping in mind: our evaluation is **single-correct-answer**, which is a simplification. In real life, multiple pages might answer the same question. The metric is strict by design.

### Q3 — First vector-search result for the same question

**Vector search returned:** `01-agentic-rag/lessons/01-intro.md` at rank 1. **Correct.**

So for *this one query*, vector search won. The homework is explicit about why this matters:

> *"Notice that one method finds the right page at the top and the other doesn't. That's exactly why we measure across the whole dataset instead of trusting one query."*

A single query is anecdote. The next questions move from anecdote to data.

### Q4 — Text search across all 360 questions

```python
{'hit_rate': 0.758, 'mrr': 0.594}
```

Translation:
- **Hit Rate 0.758** → For 75.8% of the 360 questions, the correct page appears somewhere in the top 5 results.
- **MRR 0.594** → On average, the correct page is roughly between rank 1 and 2 when found.

**Closest hit rate option:** **0.76**.

### Q5 — Vector search across all 360 questions

```python
{'hit_rate': 0.725, 'mrr': 0.549}
```

**Closest MRR option:** **0.55**.

**This is the surprise of the homework:** vector search performs **slightly worse** than text search on this dataset.

| Method | Hit Rate | MRR |
|---|---:|---:|
| Text   | 0.758 | 0.594 |
| Vector | 0.725 | 0.549 |

The intuition most people start with — *"semantic search must be better than keyword search"* — turns out to be wrong here. By both metrics, text search is the stronger single method.

Why might this be?

- **The corpus is technical documentation.** Lesson pages use specific terminology (`RAG`, `pgvector`, `embedding`, `chunking`) that exact-match keyword search nails. Vector search blurs those exact terms into a semantic neighbourhood, and sometimes the neighbourhood includes the wrong page.
- **The questions overlap lexically with the source.** Even though the prompt told the LLM to *"use as few words as possible from the page,"* generated questions still naturally share vocabulary with what they're asking about. Keyword search profits from that.
- **The embedder isn't tuned for this domain.** A general-purpose embedder treats all "introductory" pages as similar (they share generic words like *introduction, basics, overview*), which makes it hard to distinguish between, say, an intro to RAG and an intro to evaluation.

**This is exactly why we evaluate.** Without these numbers we'd have been confident (and wrong) about vector search winning.

### Q6 — Tuning hybrid search

Hybrid search runs both methods and fuses their result lists with **Reciprocal Rank Fusion (RRF)**:

```
RRF(d) = sum over lists of  1 / (k + rank(d))
```

The `k` constant controls how heavily we weight top positions. Smaller `k` = top positions dominate. Larger `k` = lower positions still meaningfully contribute.

We tried `k = 1, 50, 100, 200`:

```text
  k   hit_rate     mrr
  1     0.839    0.648    ← winner
 50     0.836    0.638
100     0.836    0.638
200     0.836    0.638
```

**Best `k` = 1.** The tie among 50/100/200 is broken by picking the smallest, so `k=1` is the unambiguous answer.

What this tells us:

- **`k=1` works best because we want the top rank to dominate.** When either text or vector search puts the correct page at rank 1, that's a strong signal — fuse with a small `k` and that signal wins. As `k` grows, lower-ranked (mostly noisy) results dilute the strong rank-1 signal.
- **The differences between 50, 100, 200 are noise.** Three decimal places of identical MRR means the parameter has saturated — once `k` is big enough to flatten position weighting, making it bigger doesn't change anything. The interesting range is `k < 50`.
- **Hybrid clearly wins over both individual methods.** Text-only MRR was 0.594, vector-only was 0.549, hybrid (with `k=1`) is 0.648 — a 9% improvement over text and 18% over vector.

---

## The headline finding

| Method | Hit Rate | MRR | Note |
|---|---:|---:|---|
| Text search | 0.758 | 0.594 | Surprising winner among single methods |
| Vector search | 0.725 | 0.549 | Underperforms text here |
| **Hybrid (k=1)** | **0.839** | **0.648** | **Clear overall winner** |

Three things to take away:

1. **The two methods catch different queries.** Text gets 75.8% by itself, vector gets 72.5%, but hybrid hits 83.9% — meaning each method is catching ~10% of queries the other one misses. Both are valuable; neither is sufficient alone.

2. **Consensus beats brilliance.** Hybrid doesn't win because it's a "smarter" algorithm; it wins because two independent methods agreeing is more reliable than either method individually.

3. **Intuition was wrong.** Going in, the natural assumption was *"vector search will beat text search on paraphrased questions."* The data says otherwise. This is the entire point of evaluation: it overrides intuition with numbers.

---

## Why this homework matters more than it looks

Everything else in this course — RAG quality, agentic loops, multi-agent systems, production deployment — sits on top of search. If retrieval brings back the wrong documents, **no amount of prompt engineering or fancy agent orchestration can rescue the answer.**

Before this homework, we had three search methods and no way to tell which was best. We picked based on what sounded reasonable or what the course taught last. Now we have evidence-based answers:

- For this corpus, on this question set, **hybrid with `k=1` is the best retrieval setting.**
- Vector search alone is the *worst* of the three options, despite being the newest and shiniest.
- The hybrid improvement over text-only is real but modest (~9% MRR gain), so the added complexity of vector + RRF is justified — but barely. In a tighter operational budget, text-only would be a defensible choice.

**The mindset shift:** "Sounds reasonable" is not a substitute for measurement. Most "obvious" choices in retrieval systems are wrong, and the only way to know is to build a ground truth set and grind through it. That's the discipline this module is trying to instill.

---

## What I changed from the course materials

A note on the helper code in this folder, because future-me will be confused by it:

The course's `evaluation_utils.llm_structured` uses the OpenAI Responses API (`client.responses.parse`). My course endpoint (`bacterio.telcryp`) only fully supports Chat Completions, and even the partial Responses support doesn't enforce the structured-output `text_format`. So I built `bacterio_compat.llm_structured` as a drop-in replacement using `chat.completions.create` with `response_format={"type": "json_object"}` and the Pydantic schema embedded in the system prompt.

End result: same function name, same arguments, but works on this endpoint. The notebook imports from `bacterio_compat` instead of `evaluation_utils`. Anyone reproducing this work on a real OpenAI account can swap the import back.

---

## Limitations to keep in mind

The metrics here are real numbers, but they have caveats:

- **Synthetic data inflates scores.** LLM-generated questions share vocabulary patterns with their source documents. Real user questions will be messier and worse-performing. Treat 75-85% hit rates as upper bounds on what real users would experience.
- **Single-correct-answer is a simplification.** Other pages in the top 5 might also be reasonable answers. The metric doesn't credit them.
- **One corpus, one experiment.** These numbers describe the LLM-Zoomcamp lessons specifically. Don't extrapolate to "vector search is bad" globally — on a different corpus (long-form prose, multilingual content, user reviews), the ranking could flip.
- **The right threshold depends on use case.** Is 83.9% hit rate good enough? Depends on what your RAG pipeline does when retrieval fails. For a critical-info system, no. For a "help me explore this corpus" tool, probably yes.

---

## What's next

Module 4 also covers RAG-level evaluation (LLM-as-judge) and agent-level evaluation (trajectory analysis). Search evaluation is the foundation; those layers build on top of it. The same `evaluate(ground_truth, search_function)` pattern extends to `evaluate(ground_truth, rag_pipeline)` and `evaluate(ground_truth, agent)` — same structure, different metric inside.

The mindset to take forward: **define what "good" means as a number, then iterate by measurement, not by hope.**

---

*Part of the [`llm-zoomcamp-2026-code`](../) learning-in-public project.*