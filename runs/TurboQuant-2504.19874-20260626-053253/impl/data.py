"""Load the paper's empirical-validation dataset: DBpedia entities encoded with OpenAI3
text-embedding-3-large at 1536 dimensions (sec:exp_valivation / sec:nn_exp).

The paper samples 100,000 training points + 1,000 query points. We stream the public
HF dataset, take the first (100k + 1k) rows, L2-normalize to the unit sphere (the
algorithms and theorems are stated for x in S^{d-1}), and cache the arrays as .npy so
subsequent eval runs are fast and offline.

Override the cache dir with PAPER_REPRISE_DATA_DIR (default /scratch/$USER/turboquant).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DATASET = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M"
DIM = 1536
N_TRAIN = 100_000
N_QUERY = 1_000


def _cache_dir() -> Path:
    base = os.environ.get("PAPER_REPRISE_DATA_DIR") or f"/scratch/{os.environ.get('USER','user')}/turboquant"
    d = Path(base)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _embedding_column(example: dict) -> str:
    """Find the column holding the 1536-d embedding (a list/array of DIM floats)."""
    for k, v in example.items():
        try:
            if hasattr(v, "__len__") and len(v) == DIM and not isinstance(v, str):
                return k
        except TypeError:
            continue
    raise RuntimeError(f"no {DIM}-d embedding column found; keys={list(example.keys())}")


def _download() -> np.ndarray:
    """Stream the first N_TRAIN+N_QUERY embeddings into a (N, DIM) float64 array."""
    from datasets import load_dataset

    need = N_TRAIN + N_QUERY
    ds = load_dataset(DATASET, split="train", streaming=True)
    rows = []
    col = None
    for ex in ds:
        if col is None:
            col = _embedding_column(ex)
        rows.append(np.asarray(ex[col], dtype=np.float64))
        if len(rows) >= need:
            break
    if len(rows) < need:
        raise RuntimeError(f"only {len(rows)} rows available, need {need}")
    return np.vstack(rows)


def load(n_query: int = N_QUERY, n_train: int | None = None):
    """Return (train, query) unit-normalized float64 arrays.

    train: up to N_TRAIN rows (capped at n_train if given, for speed); query: n_query rows.
    The split is disjoint (query taken from the tail). Cached as embeddings_raw.npy."""
    cache = _cache_dir() / "embeddings_raw.npy"
    if cache.exists():
        raw = np.load(cache)
    else:
        raw = _download()
        np.save(cache, raw)
    train = _unit(raw[:N_TRAIN])
    query = _unit(raw[N_TRAIN:N_TRAIN + n_query])
    if n_train is not None:
        train = train[:n_train]
    return train, query
