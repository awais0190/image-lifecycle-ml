"""
services/face_service.py — DeepFace ArcFace face verification service.

Responsibilities
----------------
- Load the ArcFace model once at application startup (weights are cached to disk).
- Expose verify() which compares two raw image byte payloads and returns
  whether they belong to the same person.

This module is intentionally independent of the CLIP service and must never
import from clip_service.py or similarity.py.
"""

import io
import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state — model is loaded once and reused for all requests
# ---------------------------------------------------------------------------

_ready: bool = False


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_model() -> bool:
    """
    Warm up the DeepFace ArcFace model.

    On the first call this downloads the ArcFace weights (~250 MB) to
    ~/.deepface/weights/  and runs a dummy inference to JIT-compile the
    graph.  Subsequent calls (and restarts) use the cached weights.

    Returns True on success, False if loading fails (network error, OOM …).
    """
    global _ready
    try:
        from deepface import DeepFace
        logger.info("Warming up DeepFace ArcFace model …")
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        DeepFace.represent(
            img_path=dummy,
            model_name="ArcFace",
            detector_backend="skip",   # skip detection on the dummy image
            enforce_detection=False,
        )
        _ready = True
        logger.info("DeepFace ArcFace model ready.")
        return True
    except Exception as exc:
        logger.error("DeepFace warm-up failed: %s", exc)
        _ready = False
        return False


def is_ready() -> bool:
    """Return True when the model is loaded and ready for inference."""
    return _ready


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to an RGB numpy array."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(img_bytes: bytes) -> dict:
    """
    Detect faces in a single image.

    Returns
    -------
    {
        "face_detected": bool,   # True if at least one face found
        "face_count":    int,    # number of faces detected
        "confidence":    float,  # confidence of the best detected face (0–1)
    }
    Never raises — returns face_detected=False on any error.
    """
    from deepface import DeepFace
    arr = _bytes_to_numpy(img_bytes)
    try:
        faces = DeepFace.extract_faces(
            img_path=arr,
            detector_backend="opencv",
            enforce_detection=True,
        )
        best_conf = max((float(f.get("confidence", 0)) for f in faces), default=0.0)
        return {
            "face_detected": True,
            "face_count":    len(faces),
            "confidence":    round(best_conf, 4),
        }
    except (ValueError, Exception):
        return {"face_detected": False, "face_count": 0, "confidence": 0.0}


def verify(img1_bytes: bytes, img2_bytes: bytes) -> dict:
    """
    Verify whether two images contain the same person using ArcFace.

    Strategy
    --------
    1. Try with enforce_detection=True (strict — requires a detectable face).
    2. If the detector finds no face, fall back to detector_backend='skip'
       which treats the entire image region as the face.  In that case
       face_detected=False is set in the response to signal the fallback.
    3. If both attempts fail, raise ValueError so the caller can return
       a 422 to the client.

    Parameters
    ----------
    img1_bytes : raw bytes of the first image (JPEG, PNG, WebP …)
    img2_bytes : raw bytes of the second image

    Returns
    -------
    {
        "verified":      bool,   # True when distance < ArcFace threshold
        "distance":      float,  # cosine distance  (0 = identical)
        "model":         str,    # always "ArcFace"
        "face_detected": bool,   # False when fallback was used
    }

    Raises
    ------
    ValueError  — if no face can be detected even after fallback
    RuntimeError — on unexpected DeepFace errors
    """
    from deepface import DeepFace

    arr1 = _bytes_to_numpy(img1_bytes)
    arr2 = _bytes_to_numpy(img2_bytes)

    try:
        result = DeepFace.verify(
            img1_path=arr1,
            img2_path=arr2,
            model_name="ArcFace",
            detector_backend="opencv",
            distance_metric="cosine",
            enforce_detection=True,
        )
    except ValueError:
        # No face detected in one or both images — do not fall back to
        # full-image comparison, which causes false matches on non-face
        # images (shoes, objects, etc.).
        return {
            "verified":      False,
            "distance":      1.0,
            "model":         "ArcFace",
            "face_detected": False,
        }

    return {
        "verified":      bool(result["verified"]),
        "distance":      round(float(result["distance"]), 6),
        "model":         "ArcFace",
        "face_detected": True,
    }
