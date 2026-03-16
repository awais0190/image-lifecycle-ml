"""
detectors/color_detector.py — Color change and filter detection.

analyze_single: Detects signs of color manipulation in one image
                (filter, heavy saturation boost, desaturation, etc.)

compare_two:    Compares two versions of an image and reports hue shift,
                saturation change, brightness change, and % of pixels
                that changed colour.

Uses only OpenCV + NumPy (both already installed).
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

_SATURATION_HIGH   = 140    # mean HSV-S > this → heavy saturation boost suspected
_HUE_CHANGE_DEG    = 10.0   # mean hue shift (in degrees) → hue change
_SAT_CHANGE        = 20.0   # mean saturation diff → saturation change
_BRIGHTNESS_CHANGE = 15.0   # mean brightness diff → brightness change
_DIFF_PIXEL_THRESH = 15     # per-pixel combined diff to count as "affected"
_MAX_DIM           = 800    # resize large images before analysis


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = _MAX_DIM / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def _histogram_smoothness(channel: np.ndarray) -> float:
    """
    0 → very jagged (natural photo).
    1 → very smooth (filter applied = uniform tone curve).
    """
    hist  = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
    diffs = np.abs(np.diff(hist))
    return float(1.0 - min(1.0, diffs.std() / 50.0))


def _get_dominant_colors(img: np.ndarray, k: int = 2) -> list:
    """Return k dominant colours as [R, G, B] via k-means (sampled to ≤5k pixels)."""
    try:
        pixels = img.reshape(-1, 3).astype(np.float32)
        if len(pixels) > 5000:
            idx    = np.random.choice(len(pixels), 5000, replace=False)
            pixels = pixels[idx]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3,
                                   cv2.KMEANS_RANDOM_CENTERS)
        # OpenCV is BGR → return RGB
        return [[int(c[2]), int(c[1]), int(c[0])] for c in centers]
    except Exception:
        return []


def _safe_result() -> dict:
    return {
        "color_changed":    False,
        "confidence":       0.0,
        "change_type":      "none",
        "change_intensity": 0.0,
        "details": {
            "hue_shift":            0.0,
            "saturation_change":    0.0,
            "brightness_change":    0.0,
            "affected_percentage":  0.0,
            "dominant_colors":      [],
        },
    }


# ── Detector ──────────────────────────────────────────────────────────────────

class ColorChangeDetector:

    def analyze_single(self, image: np.ndarray) -> dict:
        """
        Analyze one image for signs of colour manipulation.
        Returns change_type: "filter" | "hue_shift" | "desaturated" |
                             "brightened" | "darkened" | "none"
        """
        try:
            img = _resize(image.copy())
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h_ch, s_ch, v_ch = cv2.split(hsv)

            h_smooth   = _histogram_smoothness(h_ch)
            s_smooth   = _histogram_smoothness(s_ch)
            v_smooth   = _histogram_smoothness(v_ch)
            smoothness = h_smooth * 0.3 + s_smooth * 0.4 + v_smooth * 0.3

            sat_mean = float(s_ch.mean())
            val_mean = float(v_ch.mean())

            change_type = "none"
            confidence  = 0.0

            if smoothness > 0.78:
                change_type = "filter"
                confidence  = min(0.90, smoothness)
            elif sat_mean > _SATURATION_HIGH:
                change_type = "filter"
                confidence  = min(0.80,
                                  (sat_mean - _SATURATION_HIGH) / 60.0 + 0.50)
            elif sat_mean < 40:
                change_type = "desaturated"
                confidence  = min(0.85, (40 - sat_mean) / 40.0 * 0.85)
            elif val_mean > 210:
                change_type = "brightened"
                confidence  = 0.55
            elif val_mean < 45:
                change_type = "darkened"
                confidence  = 0.55

            color_changed = change_type != "none"
            dominant      = _get_dominant_colors(img, k=2)

            return {
                "color_changed":    color_changed,
                "confidence":       round(confidence, 3),
                "change_type":      change_type,
                "change_intensity": round(confidence, 3),
                "details": {
                    "hue_shift":           0.0,
                    "saturation_change":   round(sat_mean, 1),
                    "brightness_change":   round(val_mean, 1),
                    "affected_percentage": 100.0 if color_changed else 0.0,
                    "dominant_colors":     dominant,
                },
            }

        except Exception as exc:
            logger.error("ColorDetector.analyze_single failed: %s", exc)
            return _safe_result()

    def compare_two(self, img1: np.ndarray, img2: np.ndarray) -> dict:
        """
        Compare two images for colour differences.
        Detects hue shift, saturation change, brightness change.
        """
        try:
            a = _resize(img1.copy())
            b = _resize(img2.copy())

            # Align to same resolution (smaller of the two)
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a = cv2.resize(a, (w, h))
            b = cv2.resize(b, (w, h))

            hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV).astype(np.float32)

            # Hue is circular in OpenCV (0–180).  Min of raw diff and wrapped diff.
            h_diff  = np.abs(hsv_b[:, :, 0] - hsv_a[:, :, 0])
            h_diff  = np.minimum(h_diff, 180.0 - h_diff)
            hue_shift = float(h_diff.mean() * 2.0)   # × 2 → degrees

            sat_diff = float((hsv_b[:, :, 1] - hsv_a[:, :, 1]).mean())
            val_diff = float((hsv_b[:, :, 2] - hsv_a[:, :, 2]).mean())

            # Affected-pixel percentage
            combined_diff = (
                h_diff
                + np.abs(hsv_b[:, :, 1] - hsv_a[:, :, 1]) / 255.0 * 180.0
            ) / 2.0
            affected_pct = float((combined_diff > _DIFF_PIXEL_THRESH).mean() * 100.0)

            hue_changed = hue_shift  > _HUE_CHANGE_DEG
            sat_changed = abs(sat_diff) > _SAT_CHANGE
            val_changed = abs(val_diff) > _BRIGHTNESS_CHANGE
            color_changed = hue_changed or sat_changed or val_changed

            if hue_changed:
                change_type = "hue_shift"
            elif sat_diff < -_SAT_CHANGE:
                change_type = "desaturated"
            elif sat_diff > _SAT_CHANGE:
                change_type = "filter"
            elif val_diff > _BRIGHTNESS_CHANGE:
                change_type = "brightened"
            elif val_diff < -_BRIGHTNESS_CHANGE:
                change_type = "darkened"
            else:
                change_type = "none"

            signals = [
                min(1.0, hue_shift      / 30.0) if hue_changed else 0.0,
                min(1.0, abs(sat_diff)  / 60.0) if sat_changed else 0.0,
                min(1.0, abs(val_diff)  / 40.0) if val_changed else 0.0,
            ]
            confidence       = max(signals) if color_changed else 0.0
            change_intensity = min(1.0, affected_pct / 80.0) if color_changed else 0.0

            dom_a = _get_dominant_colors(a, k=1)
            dom_b = _get_dominant_colors(b, k=1)

            return {
                "color_changed":    color_changed,
                "confidence":       round(confidence, 3),
                "change_type":      change_type,
                "change_intensity": round(change_intensity, 3),
                "details": {
                    "hue_shift":           round(hue_shift, 1),
                    "saturation_change":   round(sat_diff, 1),
                    "brightness_change":   round(val_diff, 1),
                    "affected_percentage": round(affected_pct, 1),
                    "dominant_colors":     dom_a + dom_b,
                },
            }

        except Exception as exc:
            logger.error("ColorDetector.compare_two failed: %s", exc)
            return _safe_result()
