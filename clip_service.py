"""
clip_service.py — Singleton wrapper around the OpenAI CLIP model.

Responsibilities:
- Load ViT-B/32 CLIP model once at startup (CPU only).
- Expose helper functions that accept a PIL Image, raw bytes, or a URL
  and return a 512-dimensional numpy embedding vector.
"""

import io
import logging
from typing import Optional

import clip
import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton state — model and preprocessor are loaded once and reused.
# ---------------------------------------------------------------------------

_model = None
_preprocess = None
_model_loaded: bool = False


def load_model() -> bool:
    """
    Load the CLIP ViT-B/32 model into module-level singletons.

    Called once during FastAPI lifespan startup.  Returns True on success,
    False if loading fails (e.g. network unavailable, out of memory).
    """
    global _model, _preprocess, _model_loaded

    try:
        import torch
        logger.info("Loading CLIP ViT-B/32 model on CPU …")
        _model, _preprocess = clip.load("ViT-B/32", device="cpu")
        _model.eval()          # inference-only; disables dropout etc.
        _model_loaded = True
        logger.info("CLIP model loaded successfully.")
        return True
    except Exception as exc:
        logger.error("Failed to load CLIP model: %s", exc)
        _model_loaded = False
        return False


def is_model_loaded() -> bool:
    """Return True when the singleton model is ready for inference."""
    return _model_loaded


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _embed(image: Image.Image) -> Optional[np.ndarray]:
    """
    Run CLIP encoding on a PIL Image.

    Returns a (512,) float32 numpy array normalised to unit length,
    or None when the model is not loaded.
    """
    if not _model_loaded:
        logger.error("CLIP model is not loaded; cannot generate embedding.")
        return None

    import torch

    try:
        # Preprocess converts the PIL image to the tensor shape CLIP expects.
        tensor = _preprocess(image).unsqueeze(0)   # shape: (1, 3, 224, 224)

        with torch.no_grad():
            features = _model.encode_image(tensor)   # shape: (1, 512)

        # Normalise to unit length so cosine similarity == dot product.
        features = features / features.norm(dim=-1, keepdim=True)

        # Return as a plain 1-D numpy array.
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    except Exception as exc:
        logger.error("Error generating CLIP embedding: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_embedding(image: Image.Image) -> Optional[np.ndarray]:
    """
    Generate a 512-dim CLIP embedding from a PIL Image object.

    Args:
        image: A PIL.Image.Image instance (any mode; will be converted to RGB).

    Returns:
        numpy array of shape (512,) or None on failure.
    """
    try:
        # CLIP preprocessing expects RGB; convert regardless of source mode.
        rgb_image = image.convert("RGB")
        return _embed(rgb_image)
    except Exception as exc:
        logger.error("get_embedding failed: %s", exc)
        return None


def get_embedding_from_file(file_bytes: bytes) -> Optional[np.ndarray]:
    """
    Generate a 512-dim CLIP embedding from raw image bytes.

    Args:
        file_bytes: Raw bytes of a supported image file (JPEG, PNG, WEBP …).

    Returns:
        numpy array of shape (512,) or None on failure.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return get_embedding(image)
    except Exception as exc:
        logger.error("get_embedding_from_file failed: %s", exc)
        return None


def get_embedding_from_url(url: str) -> Optional[np.ndarray]:
    """
    Download an image from *url* and return its 512-dim CLIP embedding.

    Args:
        url: Publicly accessible image URL.

    Returns:
        numpy array of shape (512,) or None on failure.
    """
    try:
        # Send a browser-like User-Agent so image hosts don't block the request.
        headers = {"User-Agent": "Mozilla/5.0 (compatible; image-lifecycle-ml/1.0)"}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()

        # Validate that the server returned an image content-type.
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            logger.error(
                "URL did not return an image (Content-Type: %s).", content_type
            )
            return None

        return get_embedding_from_file(response.content)

    except requests.RequestException as exc:
        logger.error("Failed to download image from URL '%s': %s", url, exc)
        return None
    except Exception as exc:
        logger.error("get_embedding_from_url failed: %s", exc)
        return None
