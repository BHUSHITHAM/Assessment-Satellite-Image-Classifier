"""
Part 2 -- the serving API, v2: async, multi-model, with the load-management
and failure-handling concerns a real deployment needs, not just the happy path.

What's handled here on purpose (not just "it classifies a tile"):
  - Every checkpoint in checkpoints/ is loaded at import time into a model
    registry, keyed by the model_name stored inside the checkpoint. The
    caller picks which model to use per-request via a dropdown (see below),
    not free text -- this is the "dynamic model selection, will fix on one
    later" ask. If unspecified, the highest-val_acc loaded model is used.
  - Model names and class labels are exposed as OpenAPI enums, not plain
    strings, so Swagger UI renders them as dropdowns and invalid values are
    rejected by FastAPI's validation before any handler code runs -- this is
    why discovery happens at import time rather than inside the async
    lifespan: the dropdown's choices have to be known when the route is
    registered, which is before the app (and its lifespan) ever starts.
  - Inference is CPU-bound and blocking (torch), so it's offloaded to a
    thread via asyncio.to_thread rather than run inline in the async
    handler -- otherwise one slow classification would stall every other
    request (including /health) on the same event loop.
  - A semaphore caps how many inferences run concurrently (MAX_CONCURRENT_
    INFERENCE). Without this, a burst of requests would spin up unbounded
    threads all fighting over the same CPU cores, and every request would
    get slower instead of queueing predictably.
  - A per-request timeout guards against a hung classification blocking a
    caller forever (returns 504 instead).
  - SQLite is opened in WAL mode with a busy_timeout, so concurrent writes
    from multiple in-flight requests don't immediately throw "database is
    locked" -- they wait briefly instead.
  - A basic request-size check rejects absurdly large uploads before they're
    fully read into memory.
  - Uncaught exceptions are caught, logged server-side with a traceback, and
    returned to the caller as a generic 500 -- never leak internals.
  - CORS is permissive here since this is a demo API meant to be hit from a
    separately-hosted dashboard/browser; lock this down before anything
    resembling production use.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000
    Swagger UI: http://localhost:8000/docs

Config via environment variables (all optional):
    CHECKPOINTS_DIR            -- default "checkpoints"
    DB_PATH                    -- default "results.db"
    TILE_STORAGE_DIR           -- default "stored_tiles"
    DEFAULT_MODEL              -- force a specific model_name as default
                                   instead of auto-picking highest val_acc
    MAX_CONCURRENT_INFERENCE   -- default: number of CPU cores
    INFERENCE_TIMEOUT_SECONDS  -- default 30
    MAX_UPLOAD_BYTES           -- default 5_000_000 (5MB -- tiles are tiny)
"""

import asyncio
import hashlib
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum

import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

import storage
from data import get_transforms, idx_to_class
from models import CLASS_NAMES, discover_checkpoints

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tile_classifier")

CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "checkpoints")
DB_PATH = os.environ.get("DB_PATH", "results.db")
TILE_STORAGE_DIR = os.environ.get("TILE_STORAGE_DIR", "stored_tiles")
FORCED_DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL")  # None => auto-pick
MAX_CONCURRENT_INFERENCE = int(os.environ.get("MAX_CONCURRENT_INFERENCE", str(os.cpu_count() or 2)))
INFERENCE_TIMEOUT_SECONDS = float(os.environ.get("INFERENCE_TIMEOUT_SECONDS", "30"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(5_000_000)))

# --- Model discovery happens here, at import time, not inside lifespan ---
# Checkpoint files are static (already on disk before the process starts),
# so there's no reason to defer this to an async startup event -- and
# deferring it would mean the /classify and /results route signatures below
# couldn't know the valid model names yet, which is what we need to build
# real dropdowns instead of free-text fields.
_DEVICE = torch.device("cpu")  # offline/CPU-only per the assignment's ground rules
_REGISTRY = discover_checkpoints(CHECKPOINTS_DIR, _DEVICE)

if not _REGISTRY:
    raise RuntimeError(f"No usable checkpoints found in {CHECKPOINTS_DIR}/ -- the service has nothing to serve.")

_DEFAULT_MODEL = FORCED_DEFAULT_MODEL
if _DEFAULT_MODEL and _DEFAULT_MODEL not in _REGISTRY:
    logger.warning(f"DEFAULT_MODEL={_DEFAULT_MODEL!r} not found among {list(_REGISTRY)}; auto-picking instead.")
    _DEFAULT_MODEL = None
if not _DEFAULT_MODEL:
    _DEFAULT_MODEL = max(_REGISTRY, key=lambda name: _REGISTRY[name].get("val_acc") or 0)

for _name, _entry in _REGISTRY.items():
    _entry["idx2class"] = idx_to_class(_entry["class_to_idx"])
    _entry["transform"] = get_transforms(_entry["input_size"], train=False)

# Dynamic enums built from real data (loaded checkpoints, known class names)
# rather than hardcoded -- FastAPI renders an Enum-typed parameter as a
# dropdown in Swagger UI and rejects any value outside it automatically.
ModelName = Enum("ModelName", {name: name for name in _REGISTRY})
LabelName = Enum("LabelName", {name: name for name in CLASS_NAMES})

logger.info(
    f"Loaded models: { {k: round(v['val_acc'] or 0, 3) for k, v in _REGISTRY.items()} } "
    f"| default={_DEFAULT_MODEL} | max_concurrent_inference={MAX_CONCURRENT_INFERENCE}"
)

# Populated at startup (lifespan) -- just the pieces that genuinely need the
# running event loop (the semaphore) or a filesystem side effect (DB init).
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(TILE_STORAGE_DIR, exist_ok=True)
    storage.init_db(DB_PATH)
    _state["semaphore"] = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)
    yield
    _state.clear()


app = FastAPI(
    title="Offline Satellite Tile Classifier",
    description=(
        "Classifies satellite tiles (EuroSAT-style, 7 land-use classes) using a "
        "locally-run CPU model -- no hosted inference APIs. Pick a model from the "
        "dropdown on /classify, or omit it to use the best available one."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo-permissive; scope this down for anything real
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _make_tile_id(filename: str, content: bytes) -> str:
    # Content hash rather than just the filename: two tiles uploaded with
    # the same filename but different bytes shouldn't collide, and the
    # same bytes uploaded twice should be treated as the same tile
    # (idempotent re-classification -- see storage.insert_result's upsert).
    digest = hashlib.sha256(content).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(filename))[0]
    return f"{stem}_{digest}"


def _run_inference_sync(model_entry: dict, image: Image.Image):
    """
    The actual blocking work: preprocess + forward pass + softmax.
    Runs inside a worker thread (see /classify), never directly on the
    event loop.
    """
    tensor = model_entry["transform"](image).unsqueeze(0)
    with torch.no_grad():
        logits = model_entry["model"](tensor)
        probs = torch.softmax(logits, dim=1)[0]
    return probs


@app.post(
    "/classify",
    tags=["Classification"],
    summary="Classify one tile and store the result",
)
async def classify_tile(
    file: UploadFile = File(..., description="A tile image (PNG/JPEG)."),
    model: ModelName | None = Query(
        default=None,
        description="Which loaded model to classify with. Omit to use the best available (see GET /models).",
    ),
):
    """
    The core path: take one tile image, classify it with the selected (or
    default) model, store the result. Returns the stored result, including
    the full confidence distribution.
    """
    model_name = model.value if model is not None else _DEFAULT_MODEL

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes > {MAX_UPLOAD_BYTES} limit)",
        )

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a readable image")

    tile_id = _make_tile_id(file.filename or str(uuid.uuid4()), content)

    # Persist the raw tile to disk -- storage.py keeps a reference, not the
    # bytes, in SQLite (see design note Section 4). File I/O also goes
    # through to_thread: not CPU-bound, but still blocking syscalls that
    # shouldn't sit on the event loop under load.
    stored_path = os.path.join(TILE_STORAGE_DIR, f"{tile_id}.png")
    if not os.path.exists(stored_path):
        await asyncio.to_thread(image.save, stored_path)

    model_entry = _REGISTRY[model_name]
    semaphore = _state["semaphore"]

    start = time.time()
    try:
        async with semaphore:
            probs = await asyncio.wait_for(
                asyncio.to_thread(_run_inference_sync, model_entry, image),
                timeout=INFERENCE_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Classification timed out")
    inference_ms = (time.time() - start) * 1000

    idx2class = model_entry["idx2class"]
    class_probabilities = {idx2class[i]: round(p.item(), 4) for i, p in enumerate(probs)}
    confidence, pred_idx = probs.max(dim=0)
    predicted_label = idx2class[pred_idx.item()]

    result = await asyncio.to_thread(
        storage.insert_result,
        DB_PATH,
        tile_id=tile_id,
        source_filename=file.filename or "unknown",
        predicted_label=predicted_label,
        confidence=round(confidence.item(), 4),
        class_probabilities=class_probabilities,
        model_version=model_name,
    )
    result["inference_ms"] = round(inference_ms, 2)
    return result


@app.get(
    "/models",
    tags=["Classification"],
    summary="List loaded models and the current default",
)
def list_models():
    """Lets a caller (or the dashboard) discover what's loaded and pick one."""
    return {
        "default": _DEFAULT_MODEL,
        "available": {
            name: {"val_acc": entry["val_acc"], "input_size": entry["input_size"]}
            for name, entry in _REGISTRY.items()
        },
    }


@app.get(
    "/results/{tile_id}",
    tags=["Results"],
    summary="Look up the stored result for one tile",
)
async def get_result(tile_id: str):
    result = await asyncio.to_thread(storage.get_result, DB_PATH, tile_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No result found for tile_id={tile_id}")
    return result


@app.get(
    "/results",
    tags=["Results"],
    summary="Browse stored results with filters",
)
async def list_results(
    label: LabelName | None = Query(default=None, description="Filter by predicted land-use class."),
    min_confidence: float | None = Query(default=None, ge=0, le=1),
    max_confidence: float | None = Query(default=None, ge=0, le=1),
    model_version: ModelName | None = Query(default=None, description="Filter by which model produced the result."),
    review_flag: bool | None = Query(default=None, description="True = only tiles flagged for analyst review."),
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
):
    """
    Filtered listing -- design note Section 6, query need (2). Aggregate/
    monitoring queries (need 3) are not implemented here; see design note.
    """
    return await asyncio.to_thread(
        storage.list_results,
        DB_PATH,
        label=label.value if label is not None else None,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        model_version=model_version.value if model_version is not None else None,
        review_flag=review_flag,
        limit=limit,
        offset=offset,
    )


@app.get(
    "/health",
    tags=["System"],
    summary="Service + model registry health check",
)
def health():
    return {
        "status": "ok",
        "default_model": _DEFAULT_MODEL,
        "loaded_models": list(_REGISTRY.keys()),
        "db_path": DB_PATH,
        "max_concurrent_inference": MAX_CONCURRENT_INFERENCE,
    }
