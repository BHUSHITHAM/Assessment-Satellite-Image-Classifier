"""
Analyst dashboard -- filterable preview of stored classification results,
auto-refreshing every 15 seconds so new tiles classified via the API show
up without a manual reload.

Reads the SQLite results DB directly (read-only) rather than going through
the serving API's HTTP endpoints -- simpler, no networking/CORS concerns for
what's fundamentally a read-only internal view, and it's the same file the
API writes to, so this is always current, not a separate copy of the data.

Run with:
    streamlit run dashboard.py

Config via environment variables (should match whatever the API is using):
    DB_PATH           -- default "results.db"
    TILE_STORAGE_DIR  -- default "stored_tiles" (used for thumbnail previews)
"""

import os

import streamlit as st
from PIL import Image
from streamlit_autorefresh import st_autorefresh

import storage

DB_PATH = os.environ.get("DB_PATH", "results.db")
TILE_STORAGE_DIR = os.environ.get("TILE_STORAGE_DIR", "stored_tiles")

st.set_page_config(page_title="Tile Classifier -- Analyst Dashboard", layout="wide")

# Refresh the whole script every 15s. This is a full rerun (Streamlit's
# model), not a partial DOM patch -- fine for a dashboard at this data
# volume; would need something heavier (e.g. a websocket push) at real scale.
st_autorefresh(interval=15_000, key="auto_refresh")

st.title("Satellite Tile Classifier -- Results Dashboard")
st.caption("Auto-refreshes every 15 seconds. Reads directly from the results database.")

if not os.path.exists(DB_PATH):
    st.warning(f"No results database found yet at `{DB_PATH}` -- classify a tile via the API first.")
    st.stop()

# --- Summary row ---
stats = storage.get_summary_stats(DB_PATH)
col1, col2, col3 = st.columns(3)
col1.metric("Total classified", stats["total"])
col2.metric("Average confidence", f"{stats['avg_confidence']:.1%}" if stats["avg_confidence"] else "--")
col3.metric("Flagged for review", stats["flagged_count"])

# --- Label distribution ---
label_counts = storage.get_label_counts(DB_PATH)
if label_counts:
    st.subheader("Predicted label distribution")
    st.bar_chart(label_counts)

st.divider()

# --- Filters ---
st.subheader("Browse results")
model_versions = storage.get_distinct_model_versions(DB_PATH)
all_labels = sorted(label_counts.keys()) if label_counts else []

filter_cols = st.columns(4)
with filter_cols[0]:
    label_filter = st.selectbox("Predicted label", options=["All"] + all_labels)
with filter_cols[1]:
    model_filter = st.selectbox("Model version", options=["All"] + model_versions)
with filter_cols[2]:
    conf_range = st.slider("Confidence range", 0.0, 1.0, (0.0, 1.0))
with filter_cols[3]:
    review_only = st.checkbox("Flagged for review only")

results = storage.list_results(
    DB_PATH,
    label=None if label_filter == "All" else label_filter,
    model_version=None if model_filter == "All" else model_filter,
    min_confidence=conf_range[0],
    max_confidence=conf_range[1],
    review_flag=True if review_only else None,
    limit=100,
)

if not results:
    st.info("No results match the current filters.")
else:
    st.write(f"Showing {len(results)} result(s), most recent first.")
    for r in results:
        cols = st.columns([1, 3, 6])
        thumb_path = os.path.join(TILE_STORAGE_DIR, f"{r['tile_id']}.png")
        with cols[0]:
            if os.path.exists(thumb_path):
                st.image(Image.open(thumb_path), width=64)
            else:
                st.write("(no image)")
        with cols[1]:
            st.write(f"**{r['predicted_label']}**")
            st.caption(f"{r['confidence']:.1%} confidence")
            if r["review_flag"]:
                st.caption("Flagged for review")
        with cols[2]:
            st.caption(f"tile_id: `{r['tile_id']}` | model: {r['model_version']} | {r['created_at']}")
            probs_sorted = sorted(r["class_probabilities"].items(), key=lambda kv: -kv[1])[:3]
            st.caption(" / ".join(f"{name}: {p:.1%}" for name, p in probs_sorted))
        st.divider()
