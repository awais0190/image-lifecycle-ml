"""
analyzers/edit_analyzer.py — Master edit analysis.

Combines ColorChangeDetector + ObjectDetector into a FullEditReport.

analyze_single_bytes(raw)            → color analysis only (single image)
analyze_comparison_bytes(raw1, raw2) → color + object diff (two images)
quick_check_bytes(raw1, raw2)        → color only, no heatmap (fast)
"""

import io
import logging

import cv2
import numpy as np
from PIL import Image

from detectors.color_detector  import ColorChangeDetector
from detectors.object_detector import ObjectDetector

logger = logging.getLogger(__name__)

_color_detector  = ColorChangeDetector()
_object_detector = ObjectDetector()


# ── Image decoding ────────────────────────────────────────────────────────────

def _decode(raw: bytes) -> np.ndarray:
    """Decode raw image bytes → OpenCV BGR array.  Falls back to PIL."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


# ── Report assembly ───────────────────────────────────────────────────────────

def _severity(edit_types: list, color: dict, objects: dict) -> str:
    if not edit_types:
        return "none"
    if len(edit_types) >= 2:
        return "major"
    if "object_change" in edit_types and objects.get("total_changed_area", 0) > 10.0:
        return "moderate"
    if "color_change" in edit_types:
        return "moderate" if color.get("change_intensity", 0.0) > 0.5 else "minor"
    return "minor"


def _summary(edit_types: list, color: dict, objects: dict, comparison: bool) -> str:
    if not edit_types:
        return "No edits detected. Image appears authentic."

    parts = []

    if "color_change" in edit_types:
        ct     = color.get("change_type", "filter")
        detail = color.get("details", {})
        if ct == "hue_shift":
            parts.append(
                f"Hue shift of {detail.get('hue_shift', 0):.0f}° detected.")
        elif ct == "desaturated":
            parts.append("Image appears desaturated — colour removed.")
        elif ct == "filter":
            sat = detail.get("saturation_change", 0.0)
            if comparison and sat:
                parts.append(
                    f"Colour filter applied — saturation changed by {sat:+.0f}.")
            else:
                parts.append("Colour filter or grading detected.")
        elif ct == "brightened":
            parts.append("Brightness significantly increased.")
        elif ct == "darkened":
            parts.append("Brightness significantly decreased.")

    if "object_change" in edit_types:
        area    = objects.get("total_changed_area", 0.0)
        regions = objects.get("regions", [])
        added   = sum(1 for r in regions if r["type"] == "added")
        removed = sum(1 for r in regions if r["type"] == "removed")
        if added and removed:
            parts.append(
                f"Objects added and removed ({area:.0f}% of image changed).")
        elif added:
            parts.append(
                f"Content added in {added} region(s) ({area:.0f}% of image).")
        elif removed:
            parts.append(
                f"Content removed from {removed} region(s) ({area:.0f}% of image).")
        else:
            parts.append(f"Content modified — {area:.0f}% of image area changed.")

    return " ".join(parts)


def _build_report(color: dict, objects: dict, comparison: bool) -> dict:
    edit_types: list[str] = []
    if color.get("color_changed"):
        edit_types.append("color_change")
    if objects.get("objects_changed"):
        edit_types.append("object_change")

    is_edited = bool(edit_types)
    sev       = _severity(edit_types, color, objects)
    summ      = _summary(edit_types, color, objects, comparison)

    confidence = 0.1
    if is_edited:
        confidence = max(
            color.get("confidence", 0.0),
            min(0.90, objects.get("change_intensity", 0.0) + 0.10)
            if objects.get("objects_changed") else 0.0,
        )

    return {
        "overall": {
            "is_edited":  is_edited,
            "confidence": round(confidence, 3),
            "edit_types": edit_types,
            "severity":   sev,
            "summary":    summ,
        },
        "color":   color,
        "objects": objects,
    }


def _error_report(msg: str) -> dict:
    return {
        "overall": {
            "is_edited":  False,
            "confidence": 0.0,
            "edit_types": [],
            "severity":   "none",
            "summary":    f"Analysis failed: {msg}",
        },
        "color":   {"error": msg, "skipped": True},
        "objects": {"error": msg, "skipped": True},
    }


# ── Analyzer ──────────────────────────────────────────────────────────────────

class EditAnalyzer:

    def analyze_single_bytes(self, raw: bytes) -> dict:
        try:
            img     = _decode(raw)
            color   = _color_detector.analyze_single(img)
            objects = _object_detector.analyze_single(img)
            return _build_report(color, objects, comparison=False)
        except Exception as exc:
            logger.error("EditAnalyzer.analyze_single_bytes: %s", exc)
            return _error_report(str(exc))

    def analyze_comparison_bytes(self, raw1: bytes, raw2: bytes) -> dict:
        try:
            img1    = _decode(raw1)
            img2    = _decode(raw2)
            color   = _color_detector.compare_two(img1, img2)
            objects = _object_detector.compare_two(img1, img2)
            return _build_report(color, objects, comparison=True)
        except Exception as exc:
            logger.error("EditAnalyzer.analyze_comparison_bytes: %s", exc)
            return _error_report(str(exc))

    def quick_check_bytes(self, raw1: bytes, raw2: bytes) -> dict:
        """Fast check — colour only, no heatmap.  Target latency < 1 s."""
        try:
            img1  = _decode(raw1)
            img2  = _decode(raw2)
            color = _color_detector.compare_two(img1, img2)
            edit_types = ["color_change"] if color.get("color_changed") else []
            return {
                "is_edited":  bool(edit_types),
                "edit_types": edit_types,
                "confidence": color.get("confidence", 0.0),
            }
        except Exception as exc:
            logger.error("EditAnalyzer.quick_check_bytes: %s", exc)
            return {"is_edited": False, "edit_types": [], "confidence": 0.0}


# Module-level singleton — instantiated once at import time (no heavy model)
edit_analyzer = EditAnalyzer()
