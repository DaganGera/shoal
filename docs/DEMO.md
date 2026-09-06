# SHOAL — running the demo

## One-time setup

```bash
uv sync --group phase2
uv run shoal-fetch          # ~90 MB weights + 4 clips, cached
uv run shoal-check          # must be all green (2 yellow are expected pre-calibrate)
uv run shoal-calibrate      # fits the visibility index to the anchor clips
```

## Produce the demo runs

```bash
for c in anemone_static reef_clear reef_snapper reef_turbid_matched; do
  uv run shoal-run --clip data/clips/$c.mp4 --run-id $c
done
uv run streamlit run app/dashboard.py
```

Each run writes a self-contained `outputs/<run_id>/` (annotated video, all CSVs,
GeoJSON, MOT file, residency PNG, `run.json` provenance). The dashboard reads a run
directory; pick the run in the sidebar.

## Which clip shows what

| Run | Lead with | Why |
|---|---|---|
| `reef_clear` | **tab 5 abundance bounds + tab 1 tracking** | a real school → MaxN in the 20s; the gap to the unique-track count is the honest "track count is an upper bound, and this camera pans" story |
| `reef_turbid_matched` | **tab 2 before/after + A/B ablation** | matched-pair fixture: identical scene, only visibility changed, so the ablation isolates the restoration effect |
| `anemone_static` | **tab 1 tracking + tab 3 per-track table** | tightest tracking (two clownfish); the residency heatmap is the most legible single visual |
| `reef_snapper` | **tab 3 residency heatmap** | one fish cruising a reef — clean single trajectory |

Every clip trips the camera-motion guard (stock footage is never tripod-steady), so
tab 3 shows the "image coordinates, not world coordinates" banner. Say it out loud:
that banner is the guard working, and on a fixed seabed rig it goes green.

## The three-sentence pitch (plan §17)

1. **Problem.** Underwater video is abundant and cheap; analysis of it is scarce and
   expensive. The bottleneck is the human hours between footage and finding.
2. **What was built.** SHOAL turns raw underwater video into trajectory data, then
   into the quantities a marine scientist already uses — MaxN abundance, Shannon
   diversity with confidence intervals, behavioural state segmentation — and exports
   them in standard formats (`tracks.csv`, `trajectories.geojson`, `results_MOT.txt`).
3. **Why it holds up.** Every number on screen is measured, not asserted
   (`run.json` records the config, library versions, and code path for each run).
   Where the method is uncertain the interface says so — the diversity panel greys
   itself out under one collapsed class, the camera-motion guard flags a non-static
   rig, and the restoration module proves its own value with a live A/B test instead
   of a claim.

## The one architectural point to defend out loud

**Detection and species ID are decoupled.** Detection runs single-class and
high-recall (`shoal/detect.py` collapses every fish-family class to `fish` before
the tracker sees it). Species ID is a separate track-level step (Phase 2) because it
is far more reliable over ~50–200 views of one animal than frame-by-frame, and a
per-frame class flip would break the tracker's association. If a juror asks "why not
train a 17-class YOLO?", that is the answer — an engineering argument, not a
rationalisation.

## Honest caveats to state before being asked

- The Phase 1 demo clips are **CC0 stock footage**, not the plan's Brackish/DeepFish
  (those need Kaggle credentials). No ground truth → the A/B ablation reports
  detection-count and confidence deltas but no mAP; that code path activates
  automatically on Brackish. See `docs/PHASE1_STATUS.md`.
- `reef_turbid_matched.mp4` is a **labelled turbidity augmentation** of the clear
  reef clip — a controlled fixture for the matched-pair ablation, not real footage.
- The species head (marine-detect FishInv, regionally scoped to Tioman, Malaysia) is
  Phase 2. It is "a drop-in regional species head — swap the head, keep the
  pipeline", which makes the architecture the contribution rather than the weights.
