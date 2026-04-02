"""
app.py — FastAPI application for the image-lifecycle-ml microservice.

Endpoints
---------
GET  /health   — liveness check; reports whether CLIP model is loaded.
POST /embed    — generate a 512-dim CLIP embedding from an image file or URL.
POST /compare  — cosine-similarity score between two pre-computed embeddings.
POST /analyze  — embed a new image AND compare it against existing embeddings
                 (the main endpoint called by the Next.js app).
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from functools import partial
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import clip_service
import similarity as sim
from services import face_service, partial_match_service
from analyzers.edit_analyzer import edit_analyzer

# ---------------------------------------------------------------------------
# Logging — timestamp + level + message on every line
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Set of MIME types we accept as image uploads.
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
}

# Maximum allowed upload size: 10 MB.
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# CLIP ViT-B/32 always produces 512-dimensional embeddings.
EMBEDDING_DIM = 512

# ---------------------------------------------------------------------------
# Lifespan — load the CLIP model once at startup, release at shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load CLIP and DeepFace models before the server starts accepting requests."""
    logger.info("Starting up — loading CLIP model …")
    clip_service.load_model()
    logger.info("Starting up — loading DeepFace ArcFace model …")
    await _run_in_thread(face_service.load_model)
    yield
    logger.info("Shutting down image-lifecycle-ml service.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="image-lifecycle-ml",
    description="CLIP-based image similarity microservice",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins so the Next.js dev server (localhost:3000) and any
# production domain can call this service without CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request with method, path, and response time."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s  →  %s  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    embedding1: List[float]
    embedding2: List[float]


class ExistingEmbedding(BaseModel):
    id: str
    embedding: List[float]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _validate_image_content_type(upload: UploadFile) -> None:
    """Raise HTTPException(400) if the uploaded file is not an image."""
    ct = upload.content_type or ""
    # Also accept 'application/octet-stream' — some clients send this for images.
    if ct not in ALLOWED_IMAGE_TYPES and not ct.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file must be an image. Got content-type: '{ct}'.",
        )


def _validate_file_size(raw: bytes) -> None:
    """Raise HTTPException(413) if the file exceeds the size limit."""
    if len(raw) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )


def _validate_embedding_dim(values: List[float], name: str = "embedding") -> None:
    """Raise HTTPException(400) if the embedding is not exactly 512 dimensions."""
    if len(values) != EMBEDDING_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"'{name}' must have exactly {EMBEDDING_DIM} dimensions, got {len(values)}.",
        )


def _embedding_to_list(arr: np.ndarray) -> List[float]:
    """Convert a numpy array to a plain Python list of Python floats."""
    return arr.tolist()


def _list_to_embedding(values: List[float]) -> np.ndarray:
    """Convert a list of floats received from JSON into a numpy array."""
    return np.array(values, dtype=np.float32)


async def _run_in_thread(fn, *args):
    """
    Run a CPU-bound function in a thread pool so it does not block
    the async event loop while CLIP is processing.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness check")
async def health():
    """
    Returns service status and whether the CLIP and face models are loaded.

    Response
    --------
    { "status": "ok", "clip_loaded": true, "face_loaded": true }
    """
    return {
        "status":       "ok",
        "clip_loaded":  clip_service.is_model_loaded(),
        "face_loaded":  face_service.is_ready(),
    }


@app.post("/embed", summary="Generate a CLIP embedding")
async def embed(
    file: Optional[UploadFile] = File(default=None),
    image_url: Optional[str] = Form(default=None),
):
    """
    Generate a 512-dimensional CLIP embedding for the provided image.

    Accepts **one** of:
    - `file`      — multipart/form-data image upload.
    - `image_url` — form field containing a publicly accessible image URL.

    Returns
    -------
    { "embedding": [512 floats], "shape": 512, "status": "success" }
    """
    if not clip_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="CLIP model is not loaded yet.")

    embedding: Optional[np.ndarray] = None

    if file is not None:
        _validate_image_content_type(file)
        raw = await file.read()
        _validate_file_size(raw)
        # Run CPU-bound CLIP inference in a thread pool to avoid blocking the event loop.
        embedding = await _run_in_thread(clip_service.get_embedding_from_file, raw)
        if embedding is None:
            raise HTTPException(
                status_code=422,
                detail="Could not process the uploaded image file.",
            )

    elif image_url:
        # URL download + inference both run in the thread pool.
        embedding = await _run_in_thread(clip_service.get_embedding_from_url, image_url)
        if embedding is None:
            raise HTTPException(
                status_code=422,
                detail=f"Could not download or process image from URL: {image_url}",
            )

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either a 'file' upload or an 'image_url' form field.",
        )

    return {
        "embedding": _embedding_to_list(embedding),
        "shape": int(embedding.shape[0]),
        "status": "success",
    }


@app.post("/compare", summary="Compare two embeddings")
async def compare(body: CompareRequest):
    """
    Compute cosine similarity between two pre-computed CLIP embeddings.

    Accepts JSON body:
    ```json
    { "embedding1": [512 floats], "embedding2": [512 floats] }
    ```

    Returns
    -------
    {
        "similarity": 0.87,
        "match_level": "strong_match" | "weak_match" | "no_match",
        "status": "success"
    }
    """
    _validate_embedding_dim(body.embedding1, "embedding1")
    _validate_embedding_dim(body.embedding2, "embedding2")

    try:
        emb1 = _list_to_embedding(body.embedding1)
        emb2 = _list_to_embedding(body.embedding2)
        score = sim.cosine_similarity(emb1, emb2)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "similarity": round(score, 6),
        "match_level": sim.get_match_level(score),
        "status": "success",
    }


@app.post("/analyze", summary="Embed + compare against existing embeddings")
async def analyze(
    file: Optional[UploadFile] = File(default=None),
    image_url: Optional[str] = Form(default=None),
    existing_embeddings: Optional[str] = Form(default=None),
):
    """
    The **main endpoint** called by the Next.js application.

    1. Generates a CLIP embedding for the submitted image.
    2. Compares it against all embeddings in `existing_embeddings`.
    3. Returns the embedding, the best match, and the full sorted score list.

    Form fields
    -----------
    - `file`               — image upload (optional if image_url provided).
    - `image_url`          — URL of the image (optional if file provided).
    - `existing_embeddings`— JSON string: list of { "id": str, "embedding": [floats] }

    Returns
    -------
    ```json
    {
        "embedding": [512 floats],
        "most_similar": {
            "id": "abc123",
            "similarity": 0.87,
            "match_level": "strong_match"
        },
        "all_scores": [
            { "id": "abc123", "similarity": 0.87 },
            ...
        ],
        "status": "success"
    }
    ```
    `most_similar` and `all_scores` are `null` / `[]` when no existing
    embeddings were provided.
    """
    if not clip_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="CLIP model is not loaded yet.")

    # --- 1. Generate embedding for the submitted image -------------------
    query_embedding: Optional[np.ndarray] = None

    if file is not None:
        _validate_image_content_type(file)
        raw = await file.read()
        _validate_file_size(raw)
        query_embedding = await _run_in_thread(clip_service.get_embedding_from_file, raw)
        if query_embedding is None:
            raise HTTPException(
                status_code=422,
                detail="Could not process the uploaded image file.",
            )

    elif image_url:
        query_embedding = await _run_in_thread(clip_service.get_embedding_from_url, image_url)
        if query_embedding is None:
            raise HTTPException(
                status_code=422,
                detail=f"Could not download or process image from URL: {image_url}",
            )

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either a 'file' upload or an 'image_url' form field.",
        )

    # --- 2. Parse existing_embeddings (optional) -------------------------
    most_similar = None
    all_scores: List[dict] = []

    if existing_embeddings:
        try:
            raw_list = json.loads(existing_embeddings)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"'existing_embeddings' is not valid JSON: {exc}",
            )

        if not isinstance(raw_list, list):
            raise HTTPException(
                status_code=400,
                detail="'existing_embeddings' must be a JSON array.",
            )

        # Convert each item's embedding list → numpy array.
        embedding_list = []
        for item in raw_list:
            try:
                _validate_embedding_dim(item["embedding"], f"existing_embeddings[id={item['id']}]")
                embedding_list.append(
                    {
                        "id": item["id"],
                        "embedding": _list_to_embedding(item["embedding"]),
                    }
                )
            except (KeyError, TypeError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid embedding item: {exc}",
                )

        if embedding_list:
            # Build dict for batch_compare and list for find_most_similar.
            embeddings_dict = {e["id"]: e["embedding"] for e in embedding_list}

            most_similar = sim.find_most_similar(query_embedding, embedding_list)
            sorted_pairs = sim.batch_compare(query_embedding, embeddings_dict)
            all_scores = [
                {"id": pair_id, "similarity": score}
                for pair_id, score in sorted_pairs
            ]

    return {
        "embedding": _embedding_to_list(query_embedding),
        "most_similar": most_similar,
        "all_scores": all_scores,
        "status": "success",
    }


@app.post("/face/detect", summary="Detect faces in a single image")
async def face_detect(
    image: UploadFile = File(...),
):
    """
    Detect whether a face is present in one image.

    Accepts **multipart/form-data** with field:
    - ``image`` — the image to analyse

    Returns
    -------
    ```json
    { "face_detected": true, "face_count": 1, "confidence": 0.99 }
    ```
    Always returns 200 — ``face_detected`` is ``false`` when no face is found.
    """
    if not face_service.is_ready():
        raise HTTPException(status_code=503, detail="Face model is not loaded yet.")

    _validate_image_content_type(image)
    raw = await image.read()
    _validate_file_size(raw)

    result = await _run_in_thread(face_service.detect, raw)
    return result


@app.post("/face/verify", summary="Verify whether two images show the same person")
async def face_verify(
    img1: UploadFile = File(...),
    img2: UploadFile = File(...),
):
    """
    Verify whether the faces in two uploaded images belong to the same person.

    Uses DeepFace ArcFace with cosine distance.  The model is loaded once at
    startup and reused for every request.

    Accepts **multipart/form-data** with fields:
    - ``img1`` — first image (JPEG / PNG / WebP / GIF / BMP / TIFF)
    - ``img2`` — second image

    Returns
    -------
    ```json
    {
        "verified":      true,
        "distance":      0.42,
        "model":         "ArcFace",
        "face_detected": true
    }
    ```

    ``face_detected`` is ``false`` when no face was found by the OpenCV
    detector and the service fell back to treating the whole image as the
    face region.

    Errors
    ------
    - **503** — model not loaded yet (startup still in progress)
    - **422** — no face could be detected in one or both images
    - **413** — one of the uploaded files exceeds the 10 MB limit
    """
    if not face_service.is_ready():
        raise HTTPException(status_code=503, detail="Face model is not loaded yet.")

    _validate_image_content_type(img1)
    _validate_image_content_type(img2)

    raw1 = await img1.read()
    raw2 = await img2.read()
    _validate_file_size(raw1)
    _validate_file_size(raw2)

    try:
        result = await _run_in_thread(face_service.verify, raw1, raw2)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Face verification error: %s", exc)
        raise HTTPException(status_code=500, detail="Face verification failed.")

    return result


@app.post("/classify/editing", summary="Zero-shot CLIP editing/manipulation classification")
async def classify_editing(
    file: UploadFile = File(...),
):
    """
    Classify whether an image has been edited or digitally manipulated using
    CLIP zero-shot classification.

    Compares the image against prompts describing edited images (text overlays,
    watermarks, social-media posts, screenshots, composites …) and original
    photographs, returning a calibrated editing probability.

    Accepts **multipart/form-data** with field:
    - ``file`` — the image to classify

    Returns
    -------
    ```json
    {
        "edit_probability": 0.78,
        "top_indicators":   ["a photo with text written on it", "a watermarked image"],
        "is_edited":        true
    }
    ```
    """
    if not clip_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="CLIP model is not loaded yet.")

    _validate_image_content_type(file)
    raw = await file.read()
    _validate_file_size(raw)

    result = await _run_in_thread(clip_service.classify_editing, raw)
    return result


@app.post("/image/partial-match", summary="Detect if one image is a crop/region of the other")
async def image_partial_match(
    img1: UploadFile = File(...),
    img2: UploadFile = File(...),
):
    """
    Detect whether one image is a cropped or partial sub-region of the other.

    Uses multi-scale OpenCV template matching (TM_CCOEFF_NORMED) to find if
    one image physically appears within the other — even after resizing.

    Accepts **multipart/form-data** with fields:
    - ``img1`` — first image
    - ``img2`` — second image

    Returns
    -------
    ```json
    {
        "is_partial":  true,
        "confidence":  0.73,
        "which":       "B_in_A"
    }
    ```
    ``which`` is ``"B_in_A"`` (img2 is inside img1) or ``"A_in_B"`` (img1 is inside img2).
    ``which`` is ``null`` when ``is_partial`` is ``false``.
    """
    _validate_image_content_type(img1)
    _validate_image_content_type(img2)

    raw1 = await img1.read()
    raw2 = await img2.read()
    _validate_file_size(raw1)
    _validate_file_size(raw2)

    result = await _run_in_thread(partial_match_service.detect_partial_match, raw1, raw2)
    return result


# ---------------------------------------------------------------------------
# Edit detection endpoints
# ---------------------------------------------------------------------------

@app.get("/edit/health", summary="Which edit detectors are ready")
async def edit_health():
    """
    Reports readiness of each edit detector.

    ColorChangeDetector and ObjectDetector use only OpenCV/NumPy and are
    always ready immediately (no model loading required).

    Returns
    -------
    { "color_detector": true, "object_detector": true, "all_ready": true }
    """
    return {
        "color_detector":  True,
        "object_detector": True,
        "all_ready":       True,
    }


@app.post("/edit/analyze-single", summary="Colour-change analysis for one image")
async def edit_analyze_single(
    image: UploadFile = File(...),
):
    """
    Analyse a single image for signs of colour manipulation (filter, hue shift,
    desaturation, brightness boost).

    Returns a **FullEditReport** — overall severity + per-detector results.
    No heavy model required; typical latency < 500 ms.
    """
    _validate_image_content_type(image)
    raw = await image.read()
    _validate_file_size(raw)

    result = await _run_in_thread(edit_analyzer.analyze_single_bytes, raw)
    return result


@app.post("/edit/analyze-comparison", summary="Full edit analysis comparing two images")
async def edit_analyze_comparison(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    """
    Compare two images and detect specific edit types:
    - **Colour change** — filter, hue shift, saturation/brightness change
    - **Object change** — regions added, removed, or modified; diff heatmap

    Returns a **FullEditReport** with a base64-encoded diff heatmap PNG.
    Typical latency 1–3 s depending on image size.

    ``image1`` is treated as the reference (original).
    ``image2`` is the version being examined.
    """
    _validate_image_content_type(image1)
    _validate_image_content_type(image2)
    raw1 = await image1.read()
    raw2 = await image2.read()
    _validate_file_size(raw1)
    _validate_file_size(raw2)

    result = await _run_in_thread(
        edit_analyzer.analyze_comparison_bytes, raw1, raw2
    )
    return result


@app.post("/edit/quick-check", summary="Fast colour-only edit check (< 1 s)")
async def edit_quick_check(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    """
    Fast two-image colour comparison — skips object diff and heatmap generation.
    Designed for real-time use where latency matters.

    Returns
    -------
    { "is_edited": bool, "edit_types": ["color_change"], "confidence": float }
    """
    _validate_image_content_type(image1)
    _validate_image_content_type(image2)
    raw1 = await image1.read()
    raw2 = await image2.read()
    _validate_file_size(raw1)
    _validate_file_size(raw2)

    result = await _run_in_thread(edit_analyzer.quick_check_bytes, raw1, raw2)
    return result


# ---------------------------------------------------------------------------
# Global exception handler — catches any unhandled exception and returns
# a JSON error response instead of a 500 HTML page.
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)},
    )
