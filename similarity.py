"""
similarity.py — Cosine similarity utilities for CLIP embeddings.

All functions operate on plain numpy arrays (float32, shape (512,)) as
produced by clip_service.  No model loading happens here.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

STRONG_MATCH: float = 0.80   # score >= 0.80  → likely same/edited image
WEAK_MATCH: float = 0.60     # 0.60 <= score < 0.80 → possibly related
NO_MATCH: float = 0.60       # score < 0.60   → unrelated image


def _validate_embedding(embedding: np.ndarray, name: str = "embedding") -> None:
    """Raise ValueError if *embedding* is not a valid 1-D float array."""
    if embedding is None:
        raise ValueError(f"{name} is None.")
    if not isinstance(embedding, np.ndarray):
        raise ValueError(f"{name} must be a numpy array, got {type(embedding)}.")
    if embedding.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D, got shape {embedding.shape}."
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def cosine_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Both embeddings are L2-normalised before the dot product so the result
    is identical to the angle-based cosine similarity even when vectors are
    not pre-normalised (CLIP embeddings from clip_service are already
    unit-normalised, so this is a no-op there).

    Args:
        embedding1: numpy array of shape (D,).
        embedding2: numpy array of shape (D,).

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 if either vector is zero.
    """
    _validate_embedding(embedding1, "embedding1")
    _validate_embedding(embedding2, "embedding2")

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0.0 or norm2 == 0.0:
        logger.warning("One or both embeddings are zero-vectors; returning 0.0.")
        return 0.0

    score = float(np.dot(embedding1 / norm1, embedding2 / norm2))

    # Clamp to [0, 1] to guard against floating-point drift.
    return max(0.0, min(1.0, score))


def get_match_level(score: float) -> str:
    """
    Map a similarity score to a human-readable match level string.

    Args:
        score: Float in [0.0, 1.0].

    Returns:
        One of "strong_match", "weak_match", or "no_match".
    """
    if score >= STRONG_MATCH:
        return "strong_match"
    if score >= WEAK_MATCH:
        return "weak_match"
    return "no_match"


def find_most_similar(
    query_embedding: np.ndarray,
    embedding_list: List[Dict],
) -> Optional[Dict]:
    """
    Find the single most-similar embedding to *query_embedding*.

    Args:
        query_embedding: numpy array of shape (D,).
        embedding_list:  List of dicts, each with keys:
                           "id"        → str identifier
                           "embedding" → numpy array of shape (D,)

    Returns:
        Dict with keys "id", "similarity", "match_level", or None if
        *embedding_list* is empty or every comparison fails.
    """
    if not embedding_list:
        return None

    best_id: Optional[str] = None
    best_score: float = -1.0

    for item in embedding_list:
        try:
            score = cosine_similarity(query_embedding, item["embedding"])
            if score > best_score:
                best_score = score
                best_id = item["id"]
        except Exception as exc:
            logger.warning("Skipping item '%s' due to error: %s", item.get("id"), exc)

    if best_id is None:
        return None

    return {
        "id": best_id,
        "similarity": round(best_score, 6),
        "match_level": get_match_level(best_score),
    }


def batch_compare(
    query_embedding: np.ndarray,
    embeddings_dict: Dict[str, np.ndarray],
) -> List[Tuple[str, float]]:
    """
    Compare *query_embedding* against every embedding in *embeddings_dict*.

    Args:
        query_embedding:  numpy array of shape (D,).
        embeddings_dict:  Mapping of id → numpy array of shape (D,).

    Returns:
        List of (id, score) tuples sorted from highest to lowest similarity.
    """
    results: List[Tuple[str, float]] = []

    for item_id, embedding in embeddings_dict.items():
        try:
            score = cosine_similarity(query_embedding, embedding)
            results.append((item_id, round(score, 6)))
        except Exception as exc:
            logger.warning(
                "Skipping embedding '%s' in batch_compare: %s", item_id, exc
            )

    # Sort descending by score.
    results.sort(key=lambda x: x[1], reverse=True)
    return results
