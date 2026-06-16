"""Vector search playground for LLM-Zoomcamp module 2.

Three engines, one interface, one embedder:

  - InMemoryEngine — minsearch's VectorIndex, NumPy under the hood. Zero setup.
  - SQLiteEngine   — sqlite-vec extension, embedded DB in a single file.
  - PGVectorEngine — Postgres + pgvector, real database, real indexing.

All three use the same ONNX-backed `Embedder` (FastEmbed) so the only
thing that changes between them is the storage and search layer.

Each engine implements:
    .add(docs)           # ingest a list of {"id", "filename", "content"}
    .search(query, k=5)  # returns top-k results with cosine similarity
    .close()             # release resources
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np
from fastembed import TextEmbedding


# ---------- Embedder ----------

class Embedder:
    """Wraps a FastEmbed ONNX model. Cached so we only load weights once."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        # FastEmbed downloads and caches the ONNX model on first use.
        self._model = TextEmbedding(model_name=model_name)
        # Probe dimensionality by embedding a dummy string once.
        self.dim = len(next(self._model.embed(["probe"])))

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed many texts at once. Returns shape (n, dim) float32 array."""
        vectors = list(self._model.embed(texts))
        return np.asarray(vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ---------- Shared types ----------

@dataclass
class SearchResult:
    id: int
    filename: str
    content: str
    score: float  # cosine similarity in [-1, 1]; higher is better


class VectorEngine(Protocol):
    name: str
    def add(self, docs: list[dict]) -> None: ...
    def search(self, query: str, k: int = 5) -> list[SearchResult]: ...
    def close(self) -> None: ...


# ---------- Engine 1: in-memory (NumPy) ----------

@dataclass
class InMemoryEngine:
    embedder: Embedder
    name: str = "in-memory (numpy)"
    _matrix: np.ndarray | None = field(default=None, init=False)
    _docs: list[dict] = field(default_factory=list, init=False)

    def add(self, docs: list[dict]) -> None:
        self._docs = list(docs)
        texts = [d["content"] for d in self._docs]
        vectors = self.embedder.embed(texts)
        # Normalize so a dot product equals cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._matrix = vectors / np.clip(norms, 1e-12, None)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if self._matrix is None:
            raise RuntimeError("Call .add(docs) before .search().")
        q = self.embedder.embed_one(query)
        q /= max(np.linalg.norm(q), 1e-12)
        scores = self._matrix @ q  # shape: (n_docs,)
        top = np.argsort(-scores)[:k]
        return [
            SearchResult(
                id=int(i),
                filename=self._docs[int(i)]["filename"],
                content=self._docs[int(i)]["content"],
                score=float(scores[int(i)]),
            )
            for i in top
        ]

    def close(self) -> None:
        pass


# ---------- Engine 2: SQLite + sqlite-vec ----------

def _pack_floats(vec: np.ndarray) -> bytes:
    """sqlite-vec expects vectors as a packed bytes blob of float32 values."""
    return struct.pack(f"{len(vec)}f", *vec.astype(np.float32))


@dataclass
class SQLiteEngine:
    embedder: Embedder
    db_path: str = ":memory:"
    name: str = "sqlite-vec"
    _conn: sqlite3.Connection | None = field(default=None, init=False)

    def __post_init__(self):
        import sqlite_vec  # pip install sqlite-vec
        self._conn = sqlite3.connect(self.db_path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        cur = self._conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS docs ("
            "  id INTEGER PRIMARY KEY,"
            "  filename TEXT NOT NULL,"
            "  content TEXT NOT NULL"
            ")"
        )
        cur.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_docs "
            f"USING vec0(embedding float[{self.embedder.dim}])"
        )
        self._conn.commit()

    def add(self, docs: list[dict]) -> None:
        assert self._conn is not None
        texts = [d["content"] for d in docs]
        vectors = self.embedder.embed(texts)
        cur = self._conn.cursor()
        cur.execute("DELETE FROM docs")
        cur.execute("DELETE FROM vec_docs")
        for i, (doc, vec) in enumerate(zip(docs, vectors)):
            cur.execute(
                "INSERT INTO docs (id, filename, content) VALUES (?, ?, ?)",
                (i, doc["filename"], doc["content"]),
            )
            cur.execute(
                "INSERT INTO vec_docs (rowid, embedding) VALUES (?, ?)",
                (i, _pack_floats(vec)),
            )
        self._conn.commit()

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        assert self._conn is not None
        q_vec = self.embedder.embed_one(query)
        cur = self._conn.cursor()
        # sqlite-vec exposes a `distance` column (L2 distance for vec0).
        rows = cur.execute(
            """
            SELECT d.id, d.filename, d.content, v.distance
            FROM vec_docs v
            JOIN docs d ON d.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (_pack_floats(q_vec), k),
        ).fetchall()
        # Convert L2 distance to a similarity-ish score so it's comparable
        # to cosine. Vectors aren't unit-normalized here, so this is rough.
        return [
            SearchResult(id=r[0], filename=r[1], content=r[2], score=-float(r[3]))
            for r in rows
        ]

    def close(self) -> None:
        if self._conn:
            self._conn.close()


# ---------- Engine 3: Postgres + pgvector ----------

@dataclass
class PGVectorEngine:
    """Requires a running Postgres with the pgvector extension.

    Quick start with Docker:
        docker run -d --name pgvector \
            -e POSTGRES_PASSWORD=postgres \
            -p 5432:5432 \
            pgvector/pgvector:pg16
    """

    embedder: Embedder
    dsn: str = "postgresql://postgres:postgres@localhost:5432/postgres"
    name: str = "pgvector"
    _conn: object | None = field(default=None, init=False)

    def __post_init__(self):
        import psycopg              # pip install "psycopg[binary]"
        from pgvector.psycopg import register_vector  # pip install pgvector

        self._conn = psycopg.connect(self.dsn, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)

        with self._conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS docs_vec")
            cur.execute(
                f"""
                CREATE TABLE docs_vec (
                    id        SERIAL PRIMARY KEY,
                    filename  TEXT NOT NULL,
                    content   TEXT NOT NULL,
                    embedding vector({self.embedder.dim})
                )
                """
            )

    def add(self, docs: list[dict]) -> None:
        import numpy as np
        texts = [d["content"] for d in docs]
        vectors = self.embedder.embed(texts)
        with self._conn.cursor() as cur:
            for doc, vec in zip(docs, vectors):
                cur.execute(
                    "INSERT INTO docs_vec (filename, content, embedding) "
                    "VALUES (%s, %s, %s)",
                    (doc["filename"], doc["content"], np.asarray(vec)),
                )
            # IVFFlat index — fast approximate search.
            # For tiny datasets a sequential scan is faster; uncomment for >10k rows.
            # cur.execute(
            #     "CREATE INDEX ON docs_vec USING ivfflat (embedding vector_cosine_ops) "
            #     "WITH (lists = 100)"
            # )

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        import numpy as np
        q_vec = np.asarray(self.embedder.embed_one(query))
        with self._conn.cursor() as cur:
            # The `<=>` operator is cosine distance (0=identical, 2=opposite).
            rows = cur.execute(
                """
                SELECT id, filename, content, 1 - (embedding <=> %s) AS similarity
                FROM docs_vec
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (q_vec, q_vec, k),
            ).fetchall()
        return [
            SearchResult(id=r[0], filename=r[1], content=r[2], score=float(r[3]))
            for r in rows
        ]

    def close(self) -> None:
        if self._conn:
            self._conn.close()


# ---------- Convenience: benchmark all engines on the same data ----------

def benchmark(engines: Iterable[VectorEngine], docs: list[dict], queries: list[str], k: int = 3):
    """Run the same queries across every engine and return a tidy list of dicts."""
    import time

    # Ingest once per engine (cost we usually pay during setup, not at query time).
    for eng in engines:
        t0 = time.perf_counter()
        eng.add(docs)
        print(f"[{eng.name}] indexed {len(docs)} docs in {time.perf_counter()-t0:.2f}s")

    out = []
    for q in queries:
        for eng in engines:
            t0 = time.perf_counter()
            results = eng.search(q, k=k)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            out.append(
                {
                    "engine": eng.name,
                    "query": q,
                    "top_filename": results[0].filename if results else None,
                    "top_score": results[0].score if results else None,
                    "latency_ms": elapsed_ms,
                }
            )
    return out
