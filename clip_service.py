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


def classify_editing(file_bytes: bytes) -> dict:
    """
    Zero-shot CLIP classification for editing / manipulation detection.

    Compares the image against two sets of text prompts:
      • Edited indicators  — text overlays, watermarks, social posts, screenshots, etc.
      • Original indicators — raw photographs, unedited camera images

    Uses softmax over all prompts so scores are calibrated probabilities.

    Returns
    -------
    {
        "edit_probability": float,    # 0–1; higher = more likely edited/processed
        "top_indicators":  list[str], # up to 3 editing prompts that matched strongly
        "is_edited":       bool,      # edit_probability >= 0.55
    }
    """
    if not _model_loaded:
        return {"edit_probability": 0.0, "top_indicators": [], "is_edited": False}

    import torch

    EDITED_PROMPTS = [
        "a photo with text written on it",
        "a social media post with a caption",
        "a watermarked image",
        "a screenshot of a phone or computer",
        "a thumbnail with text overlay",
        "a photoshopped or digitally manipulated image",
        "an edited photo with filters or effects",
        "a meme with text",
        "an image with a logo or brand overlay",
        "a promotional or marketing graphic",
        "a composite image made from multiple photos",
        "a photo with stickers or emoji",
        "an image with a title or headline text",
        "a digitally enhanced photograph",
        "an image with a black or white border added",
    ]

    ORIGINAL_PROMPTS = [
        "an original unedited photograph",
        "a natural raw photo from a camera",
        "an authentic unmanipulated image",
    ]

    all_prompts = EDITED_PROMPTS + ORIGINAL_PROMPTS

    try:
        image  = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        tensor = _preprocess(image).unsqueeze(0)

        tokens = clip.tokenize(all_prompts)

        with torch.no_grad():
            img_features  = _model.encode_image(tensor)
            text_features = _model.encode_text(tokens)

            img_features  = img_features  / img_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarities scaled by CLIP's learned temperature (100×)
            logits = (100.0 * img_features @ text_features.T).squeeze(0)
            probs  = logits.softmax(dim=-1).cpu().numpy()

        n_edited   = len(EDITED_PROMPTS)
        edited_prob = float(probs[:n_edited].sum())

        # Top editing indicators (prompts with highest individual probability)
        edited_probs = list(zip(EDITED_PROMPTS, probs[:n_edited].tolist()))
        edited_probs.sort(key=lambda x: x[1], reverse=True)
        top_indicators = [p for p, _ in edited_probs[:3] if _ > 0.01]

        logger.debug(
            "classify_editing: edit_prob=%.3f  top=%s",
            edited_prob, top_indicators[:1],
        )

        return {
            "edit_probability": round(edited_prob, 4),
            "top_indicators":   top_indicators,
            "is_edited":        edited_prob >= 0.55,
        }

    except Exception as exc:
        logger.error("classify_editing failed: %s", exc)
        return {"edit_probability": 0.0, "top_indicators": [], "is_edited": False}


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
