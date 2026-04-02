"""
services/partial_match_service.py — Multi-scale template matching for crop/partial detection.

Detects whether one image is a cropped or partial sub-region of another image.
Uses OpenCV TM_CCOEFF_NORMED at multiple scales so it works even when the crop
has been resized after extraction.

This answers the question: "Is image B a piece cut out of image A?" — something
pHash and CLIP both fail on because a crop changes the hash completely and
reduces CLIP similarity to ~0.65, which falls below the "related" threshold.
"""

import io
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Template must be at least this many pixels on each side
_MIN_TEMPLATE_PX = 32

# Template (at tested scale) must cover at least this fraction of the source area.
# Prevents tiny patch matches (e.g. a 10×10 template matching random texture).
_MIN_AREA_RATIO = 0.08   # template must be ≥ 8% of source area

# How many scale steps to try (more = slower but more accurate)
_NUM_SCALES = 20

# TM_CCOEFF_NORMED confidence threshold.
# Genuine crops (resized/compressed) typically score 0.68–0.92.
# Random image pairs (people, objects) typically score 0.20–0.55.
# 0.70 cleanly separates real crops from coincidental texture matches.
PARTIAL_MATCH_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_gray(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to a grayscale numpy array (uint8)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    return np.array(img, dtype=np.uint8)


def _try_match(source: np.ndarray, template_orig: np.ndarray) -> float:
    """
    Try to find template_orig inside source at multiple scales.

    We resize the TEMPLATE (not the source) to account for the common case where
    the crop was resized/compressed after being cut out of the original.
    Scales range from 0.20 to 1.0 of the template's current dimensions.

    Constraints that prevent false positives:
    - Template side must be ≥ _MIN_TEMPLATE_PX pixels
    - Template area must be ≥ _MIN_AREA_RATIO × source area (no tiny patch matches)
    - Template must physically fit inside source

    Returns the best TM_CCOEFF_NORMED score (0.0–1.0).
    """
    src_h, src_w = source.shape
    src_area = src_h * src_w
    best = 0.0

    for scale in np.linspace(0.20, 1.0, _NUM_SCALES):
        new_w = max(1, int(template_orig.shape[1] * scale))
        new_h = max(1, int(template_orig.shape[0] * scale))

        # Template side too small — noisy match
        if new_w < _MIN_TEMPLATE_PX or new_h < _MIN_TEMPLATE_PX:
            continue
        # Template must fit inside source
        if new_w > src_w or new_h > src_h:
            continue
        # Template must cover a meaningful area of source (no micro-patch matches)
        if (new_w * new_h) < _MIN_AREA_RATIO * src_area:
            continue

        template = cv2.resize(template_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val > best:
            best = float(max_val)

    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_partial_match(img1_bytes: bytes, img2_bytes: bytes) -> dict:
    """
    Detect whether one image is a cropped sub-region of the other.

    Both orderings are tried (img2 inside img1, img1 inside img2) to handle
    the case where either image could be the "original" containing the other.

    Returns
    -------
    {
        "is_partial":  bool,          # True when best confidence >= threshold
        "confidence":  float,         # 0.0–1.0
        "which":       str | None,    # "B_in_A" | "A_in_B" | null
    }
    """
    try:
        gray1 = _to_gray(img1_bytes)
        gray2 = _to_gray(img2_bytes)

        # Try both orderings — we don't know which image is larger / the source
        score_b_in_a = _try_match(source=gray1, template_orig=gray2)  # img2 crop of img1
        score_a_in_b = _try_match(source=gray2, template_orig=gray1)  # img1 crop of img2

        best_score = max(score_b_in_a, score_a_in_b)
        is_partial = best_score >= PARTIAL_MATCH_THRESHOLD

        which = None
        if is_partial:
            which = "B_in_A" if score_b_in_a >= score_a_in_b else "A_in_B"

        logger.debug(
            "partial_match: B_in_A=%.3f  A_in_B=%.3f  best=%.3f  is_partial=%s",
            score_b_in_a, score_a_in_b, best_score, is_partial,
        )

        return {
            "is_partial": is_partial,
            "confidence": round(best_score, 4),
            "which":      which,
        }

    except Exception as exc:
        logger.error("detect_partial_match failed: %s", exc)
        return {"is_partial": False, "confidence": 0.0, "which": None}
