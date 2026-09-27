# Offline Satellite Tile Classifier -- GalaxEye Take-Home

See [`design_note.md`](design_note.md) for Part 1 (architecture, trade-offs,
assumptions, questions) and [`part3_answers.md`](part3_answers.md) for
Part 3 (written answers). This README covers running the thing (Part 2).

## What's here

```
design_note.pdf              -- Part 1: design note
part3_answers.txt            -- Part 3: written answers
models.py, data.py           -- shared architectures + dataset loading
train_cnn.py                 -- trains SmallCNN from scratch (real result: 81-85% val acc)
train_finetune.py            -- fine-tunes pretrained ResNet18 (real result: 95-96% val acc)
eval.py                      -- scores a checkpoint against eval_set/ + eval_labels.csv
storage.py                   -- SQLite result storage (WAL mode, upserts, aggregate helpers)
main.py                      -- FastAPI serving slice (Part 2)
dashboard.py                 -- Streamlit analyst dashboard, auto-refreshes every 15s
checkpoints/                 -- trained model files (small_cnn.pt, resnet18_finetune.pt)
results/                     -- sample prediction CSVs from eval.py runs
README_TRAINING.md           -- detailed training setup (Colab + local)
requirements.txt             -- training + API deps
requirements-dashboard.txt   -- dashboard-only deps (no torch needed)
```

## Real results (already run, not hypothetical)

| Model | Eval accuracy | Notes |
|---|---|---|
| SmallCNN (from scratch) | ~81-85% | No internet needed even at training time |
| ResNet18 (fine-tuned) | ~95-96% | Needs one-time internet access to fetch ImageNet weights |

Both are served by the API; which one answers a given request is chosen
per-request (see below), not hardcoded.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate

# CPU-only torch install (skips several GB of unneeded nvidia-cu* packages):
pip install torch --no-deps
pip install torchvision --no-deps
pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

In another terminal, for the dashboard:

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard.py
```

Both read/write the same `results.db` / `stored_tiles/` in the current
directory by default, so just run them from the same folder (or point both
at the same paths via `DB_PATH` / `TILE_STORAGE_DIR` env vars).

## API usage

```bash
# what's loaded, and which one answers by default
curl http://localhost:8000/health
curl http://localhost:8000/models

# classify with the default (highest-accuracy) model
curl -X POST http://localhost:8000/classify -F "file=@some_tile.png"

# classify with a specific model explicitly
curl -X POST "http://localhost:8000/classify?model=small_cnn" -F "file=@some_tile.png"

# point lookup
curl http://localhost:8000/results/<tile_id>

# filtered listing
curl "http://localhost:8000/results?label=Forest&min_confidence=0.8"
curl "http://localhost:8000/results?review_flag=true"   # low-confidence, needs a human look
```

Interactive API docs (Swagger UI) are auto-generated at `/docs` once the
server's running.

### What the API deliberately handles (see main.py's docstring for the full list)
- Dynamic model selection per request, not a single hardcoded model
- Async request handling; blocking inference runs in a worker thread so it
  never stalls the event loop
- A semaphore caps concurrent inferences (`MAX_CONCURRENT_INFERENCE`) so a
  burst of requests queues predictably instead of thrashing the CPU
- Per-request inference timeout (`INFERENCE_TIMEOUT_SECONDS`)
- Upload size limit (`MAX_UPLOAD_BYTES`)
- SQLite in WAL mode with a busy timeout, so concurrent writes don't
  immediately throw "database is locked"
- Uncaught exceptions are logged server-side and returned as a generic 500,
  never leaking internals to the caller

### What it deliberately does NOT handle (see design_note.md)
- Auth (none -- add before exposing this beyond a demo)
- An ingest queue for sustained high-throughput bursts
- Aggregate/monitoring endpoints (the dashboard covers this by reading the
  DB directly instead)

## Dashboard

`streamlit run dashboard.py` -- filterable table of stored results (by
label, model version, confidence range, review-flag status), a label
distribution chart, and summary stats. Auto-refreshes every 15 seconds, so
classifying a tile via the API shows up there shortly after without a
manual reload.

## Deploying a live demo

Render and Railway both support deploying straight from a Python repo --
you give them a build command and a start command:

**API service:**
- Build command: `pip install torch --no-deps && pip install torchvision --no-deps && pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

**Dashboard service:**
- Build command: `pip install -r requirements-dashboard.txt`
- Start command: `streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0`



