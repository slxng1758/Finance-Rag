"""Shared embedding model used by both the change-detection engine and (if built) the
stretch Q&A/retrieval layer, so similarity scores are computed consistently and the
model is loaded once.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

PRIMARY_MODEL = "BAAI/bge-large-en-v1.5"
FAST_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def get_model(fast: bool = False) -> SentenceTransformer:
    """Return the shared embedding model. `fast=True` swaps in the smaller
    low-resource fallback (MiniLM) instead of the primary bge-large model."""
    return _load_model(FAST_MODEL if fast else PRIMARY_MODEL)


def embed(texts: list[str], fast: bool = False) -> np.ndarray:
    """Embed a list of texts, L2-normalized, so dot product == cosine similarity."""
    model = get_model(fast=fast)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between embedding matrix `a` (n, d) and `b` (m, d) -> (n, m).
    Assumes inputs are already L2-normalized (true for anything from `embed`)."""
    return a @ b.T
