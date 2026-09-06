# SHOAL — Spatio-temporal Habitat & Organism Analytics Layer

**Challenge:** CHE01 — AI-Based Underwater Fish Detection and Tracking for Spatio-Temporal Marine Monitoring

SHOAL turns raw underwater video into trajectory data, then into the specific
quantities a marine scientist already uses — MaxN abundance, Shannon diversity with
confidence intervals, behavioural state segmentation — and exports them in standard
formats. Every number on screen is measured at runtime, not asserted. Where the
method is uncertain, the interface says so.

This repository is at **Phase 1** (mentor-demo scope) per `SHOAL_build_plan.md §13`.
See [`docs/PHASE1_STATUS.md`](docs/PHASE1_STATUS.md) for exactly what is built, what
is deferred to Phase 2, and every deviation from the plan.

---

## Quick start

```bash
# 1. Environment (Python 3.11 or 3.12; uv manages the interpreter)
uv sync --group phase2

# 2. Fetch demo clips + detector weights (cached in data/ and weights/, never re-downloaded)
uv run shoal-fetch

# 3. Preflight — must print all green (plan §13, 0:00–0:20 acceptance test)
uv run shoal-check

# 4. Calibrate the visibility index against the anchor clips (writes config/visibility_calibration.json)
uv run shoal-calibrate

# 5. Run the pipeline on a clip -> outputs/<run_id>/
uv run shoal-run --clip data/clips/reef_clear.mp4

# 6. Dashboard
uv run streamlit run app/dashboard.py
```

Tests: `uv run pytest`.  Demo walkthrough + pitch: [`docs/DEMO.md`](docs/DEMO.md).

## Pipeline

```
video -> ingest -> visibility index -> adaptive restoration (gated)
      -> detection (single-class fish) -> BoT-SORT tracking
      -> track repair (smoothing / gap-fill / stitching / quality flags)
      -> MovingPandas trajectory analytics (residency, stops, activity nodes, flows)
      -> behaviour (rule-based classifier; HMM in Phase 2)
      -> ecology (MaxN, Shannon/Pielou/Simpson/richness, bootstrap CI, honesty guard)
      -> exports (tracks.csv, trajectories.geojson, results_MOT.txt)
```

The one architectural decision defended out loud: **detection and species ID are
decoupled**. Detection runs single-class and high-recall; species identification is
a separate track-level step (Phase 2) because it is far more reliable over ~50–200
views of one animal than frame-by-frame, and a per-track class change would
otherwise break the tracker's association logic.

## Units

Without camera calibration, **every distance is in pixels and every speed is in
px/s or body-lengths/s (BL/s)**. The dashboard says so in the axis labels. A metre
value is shown only after the user supplies a scale in the UI. BL/s is a standard
unit in fish swimming ecology and is the default.

## Repository layout

```
config/     pipeline.yaml, botsort_underwater.yaml, visibility_calibration.json
shoal/      the pipeline modules (one per plan module; see docstrings for the "why")
app/        Streamlit dashboard
benchmark/  HOTA/MOTA evaluation (Phase 2)
data/       clips + datasets (gitignored)
weights/    model checkpoints (gitignored)
outputs/    per-run artefact directories
```

## Licensing note

Ultralytics YOLO is AGPL-3.0. The detector sits behind the interface in
`shoal/detect.py`; swapping in a permissively-licensed detector is a config change,
not a rewrite.
