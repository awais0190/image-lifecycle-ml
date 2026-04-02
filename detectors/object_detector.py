"""
detectors/object_detector.py — Object addition / removal detection.

compare_two:    Computes an absolute-difference map between two image versions,
                finds contours of significant change regions, classifies them as
                "added" / "removed" / "modified", and returns a base64 heatmap.

analyze_single: Returns a no-op (object detection inherently requires two images).

Uses only OpenCV + NumPy (both already installed).
"""

import base64
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DIFF_THRESHOLD       = 30      # pixel-level threshold to count as changed
_MIN_CONTOUR_AREA_PCT = 0.005   # ignore contours < 0.5 % of image (JPEG noise)
_MAX_ALIGN_DIM        = 800     # max side before resizing for alignment


# ── Helpers ───────────────────────────────────────────────────────────────────

def _align(img1: np.ndarray, img2: np.ndarray):
    """
    Resize both images to the same dimensions while preserving img1's aspect
    ratio.  Uses the smaller of the two widths / heights, capped at _MAX_ALIGN_DIM.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    target_h = min(h1, h2, _MAX_ALIGN_DIM)
    target_w = min(w1, w2, _MAX_ALIGN_DIM)

    scale = min(target_w / w1, target_h / h1)
    new_w, new_h = int(w1 * scale), int(h1 * scale)

    a = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return a, b


def _heatmap_b64(diff: np.ndarray, base: np.ndarray, regions: list) -> str | None:
    """
    Blend the amplified diff map with the base image and draw bounding boxes
    around changed regions.  Returns a base64-encoded PNG string.
    """
    try:
        amplified = cv2.multiply(diff, 4)
        heatmap   = cv2.applyColorMap(amplified, cv2.COLORMAP_JET)

        base_r = cv2.resize(base, (heatmap.shape[1], heatmap.shape[0]))
        blended = cv2.addWeighted(base_r, 0.5, heatmap, 0.5, 0)

        for r in regions:
            x, y, rw, rh = r["bbox"]
            color = (
                (0, 200, 50)   if r["type"] == "added"    else   # green
                (50, 50, 255)  if r["type"] == "removed"  else   # red
                (0, 200, 200)                                     # yellow
            )
            cv2.rectangle(blended, (x, y), (x + rw, y + rh), color, 2)

        _, buf = cv2.imencode(".png", blended)
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception as exc:
        logger.warning("Heatmap generation failed: %s", exc)
        return None


# ── Detector ──────────────────────────────────────────────────────────────────

class ObjectDetector:

    def analyze_single(self, image: np.ndarray) -> dict:
        """
        Single-image object analysis — no reference image available so we cannot
        detect additions/removals.  Returns an empty (no-op) result.
        """
        return {
            "objects_changed":      False,
            "copy_move_detected":   False,
            "regions":              [],
            "total_changed_area":   0.0,
            "diff_heatmap_base64":  None,
            "change_intensity":     0.0,
        }

    def compare_two(self, img1: np.ndarray, img2: np.ndarray) -> dict:
        """
        Find regions that were added, removed, or modified between img1
        (treated as the reference / original) and img2.
        """
        try:
            a, b      = _align(img1, img2)
            h, w      = a.shape[:2]
            total_px  = h * w

            gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

            # Absolute diff → blur to suppress JPEG noise → threshold
            diff    = cv2.absdiff(gray_a, gray_b)
            blurred = cv2.GaussianBlur(diff, (5, 5), 0)
            _, mask = cv2.threshold(blurred, _DIFF_THRESHOLD, 255,
                                    cv2.THRESH_BINARY)

            # Morphological close to merge nearby changed pixels into blobs
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            min_area    = _MIN_CONTOUR_AREA_PCT * total_px

            regions       = []
            total_changed = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue

                x, y, cw, ch = cv2.boundingRect(cnt)
                area_pct     = area / total_px * 100.0

                # Classify by average brightness change in the ROI
                roi_diff = (gray_b[y:y+ch, x:x+cw].astype(float)
                            - gray_a[y:y+ch, x:x+cw].astype(float)).mean()

                rtype = (
                    "added"    if roi_diff >  10 else
                    "removed"  if roi_diff < -10 else
                    "modified"
                )

                regions.append({
                    "type":             rtype,
                    "area_percentage":  round(area_pct, 2),
                    "bbox":             [int(x), int(y), int(cw), int(ch)],
                    "confidence":       round(min(0.95, area_pct / 20.0 + 0.40), 2),
                })
                total_changed += area

            total_changed_pct = min(100.0, total_changed / total_px * 100.0)
            objects_changed   = total_changed_pct > 1.0

            heatmap_b64      = _heatmap_b64(diff, a, regions)
            change_intensity = min(1.0, total_changed_pct / 40.0)

            return {
                "objects_changed":     objects_changed,
                "copy_move_detected":  False,
                "regions":             regions[:10],
                "total_changed_area":  round(total_changed_pct, 2),
                "diff_heatmap_base64": heatmap_b64,
                "change_intensity":    round(change_intensity, 3),
            }

        except Exception as exc:
            logger.error("ObjectDetector.compare_two failed: %s", exc)
            return {
                "objects_changed":     False,
                "copy_move_detected":  False,
                "regions":             [],
                "total_changed_area":  0.0,
                "diff_heatmap_base64": None,
                "change_intensity":    0.0,
            }
