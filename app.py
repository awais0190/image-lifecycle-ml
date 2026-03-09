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
    """Load CLIP model before the server starts accepting requests."""
    logger.info("Starting up — loading CLIP model …")
    clip_service.load_model()
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
    Returns service status and whether the CLIP model is loaded.

    Response
    --------
    { "status": "ok", "model_loaded": true }
    """
    return {"status": "ok", "model_loaded": clip_service.is_model_loaded()}


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
