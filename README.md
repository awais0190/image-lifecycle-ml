# image-lifecycle-ml

> A Python microservice that detects whether two images are the same, similar, or completely different — even if one has been edited, cropped, or resaved.

---

## What problem does this solve?

Imagine you have thousands of images in a system.  Someone uploads a new image — but is it brand new, or is it a cropped version of an image you already have?  Is it a re-exported copy?  A lightly edited variant?

This service answers that question automatically.

It uses **OpenAI CLIP** — an AI model originally built by OpenAI — to understand the *visual meaning* of an image, not just its pixels.  Two photos of the same subject will score high similarity even if they have different file sizes, slight color adjustments, or minor edits.

---

## How it works — the simple version

```
Your image
    │
    ▼
CLIP AI model
    │
    ▼
A "fingerprint" (called an embedding) — 512 numbers that describe what the image looks like
    │
    ▼
Compare that fingerprint against fingerprints of other images
    │
    ▼
Get a similarity score  (0.0 = completely different  →  1.0 = identical)
```

The similarity score is then labelled:

| Label          | Score     | Meaning                                      |
|----------------|-----------|----------------------------------------------|
| `strong_match` | ≥ 0.80    | Almost certainly the same image or a direct edit |
| `weak_match`   | 0.60–0.79 | Possibly related (same subject, different shot) |
| `no_match`     | < 0.60    | Unrelated images                             |

---

## Where does this fit in the bigger system?

This is a **standalone Python service**.  The main application (built with Next.js) calls this service over HTTP whenever it needs to:

1. Register a new image — generate its fingerprint and store it.
2. Check an uploaded image — compare it against all known fingerprints.
3. Build an image family tree — find which images are parents, children, or variants of each other.

```
Next.js App
    │
    │  HTTP requests
    ▼
image-lifecycle-ml  (this service, port 8000)
    │
    │  loads
    ▼
CLIP ViT-B/32 AI model  (runs locally on CPU)
```

---

## Requirements

- Python 3.10 or higher
- ~2 GB disk space (for the CLIP model download on first run)
- No GPU needed — runs entirely on CPU

---

## Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd image-lifecycle-ml
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
```

### 3. Activate the virtual environment
```bash
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate.bat     # Windows
```

You will see `(venv)` appear at the start of your terminal prompt — that means it is active.

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

> First install downloads PyTorch and CLIP — this can take a few minutes depending on your connection.

### 5. Start the server
```bash
uvicorn app:app --reload --port 8000
```

When you see this, the service is ready:
```
INFO  CLIP model loaded successfully.
INFO  Uvicorn running on http://0.0.0.0:8000
```

> The CLIP model (~338 MB) is downloaded automatically on first startup and cached at `~/.cache/clip/`.  Subsequent starts are fast.

---

## API endpoints

### `GET /health` — Is the service running?

Use this to check the service is up and the AI model is loaded before making other calls.

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok", "model_loaded": true }
```

---

### `POST /embed` — Generate a fingerprint for an image

Takes an image (uploaded file or URL) and returns its 512-number fingerprint.  Store this fingerprint in your database alongside the image ID.

**Upload a file:**
```bash
curl -X POST http://localhost:8000/embed \
  -F "file=@/path/to/photo.jpg"
```

**Pass a URL:**
```bash
curl -X POST http://localhost:8000/embed \
  -F "image_url=https://example.com/photo.jpg"
```

**Response:**
```json
{
  "embedding": [0.023, -0.014, 0.061, ...],
  "shape": 512,
  "status": "success"
}
```

---

### `POST /compare` — How similar are two images?

Pass the fingerprints of two images (that you stored earlier) and get back a similarity score.

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "embedding1": [0.023, -0.014, ...],
    "embedding2": [0.031, -0.009, ...]
  }'
```

**Response:**
```json
{
  "similarity": 0.872341,
  "match_level": "strong_match",
  "status": "success"
}
```

---

### `POST /analyze` — The main endpoint

This is the single call the Next.js app uses most.  It does everything in one shot:

1. Generates the fingerprint for the new image.
2. Compares it against all fingerprints you pass in.
3. Returns the fingerprint + the best match + all scores ranked highest to lowest.

**Form fields:**

| Field                 | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| `file`                | Image file upload                                               |
| `image_url`           | Or an image URL — provide one of these two                      |
| `existing_embeddings` | Optional JSON array of previously stored fingerprints to compare against |

**`existing_embeddings` format:**
```json
[
  { "id": "img_abc123", "embedding": [0.023, -0.014, ...] },
  { "id": "img_def456", "embedding": [0.031, -0.009, ...] }
]
```

**Example request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@/path/to/new-photo.jpg" \
  -F 'existing_embeddings=[{"id":"img_abc123","embedding":[...]}]'
```

**Response:**
```json
{
  "embedding": [0.023, -0.014, ...],
  "most_similar": {
    "id": "img_abc123",
    "similarity": 0.872341,
    "match_level": "strong_match"
  },
  "all_scores": [
    { "id": "img_abc123", "similarity": 0.872341 },
    { "id": "img_def456", "similarity": 0.412000 }
  ],
  "status": "success"
}
```

If you don't pass `existing_embeddings`, `most_similar` is `null` and `all_scores` is `[]` — the endpoint still returns the embedding for the new image.

---

## Project structure

```
image-lifecycle-ml/
├── app.py             # All HTTP endpoints (FastAPI)
├── clip_service.py    # Loads the CLIP model; converts images to fingerprints
├── similarity.py      # Compares fingerprints; returns scores and match labels
├── .env               # Config (port, environment)
├── requirements.txt   # Python dependencies
└── venv/              # Virtual environment (not committed to git)
```

---

## Configuration (`.env`)

| Variable      | Default       | Description                   |
|---------------|---------------|-------------------------------|
| `PORT`        | `8000`        | Port the server listens on    |
| `ENVIRONMENT` | `development` | `development` or `production` |

---

## Technical notes

- **No GPU required** — the CLIP model runs on CPU.
- **CORS is open** — any origin (including `localhost:3000`) can call this service.
- **Every request is logged** to the console with timestamp, method, path, status code, and response time.
- **Accepted image formats:** JPEG, PNG, WEBP, GIF, BMP, TIFF.
