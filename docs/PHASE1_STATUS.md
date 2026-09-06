# SHOAL — Phase 1 status

Phase 1 = the ~3-hour mentor-demo scope from `SHOAL_build_plan.md §13`. This file
records **exactly what is built, what is deferred, and every deviation from the
plan**, so the Phase 2 work starts from a known state (plan §16: "Stop after Phase 1
for review").

---

## What runs end to end today

`uv run shoal-check` → `uv run shoal-fetch` → `uv run shoal-calibrate` →
`uv run shoal-run --clip data/clips/reef_clear.mp4` → `uv run streamlit run app/dashboard.py`

| Plan module | File | Phase 1 state |
|---|---|---|
| [1] Ingest + camera-motion guard | `shoal/ingest.py` | **Done.** Frame iterator with real timestamps; camera motion measured by LK feature tracking + partial-affine RANSAC, chained frame-to-frame for a net-drift estimate; verdict shown in the dashboard next to every spatial figure. All four stock clips flag non-static (they are). |
| [2] Visibility index | `shoal/visibility.py` | **Done.** Four components (contrast, colour-cast, sharpness, dark-channel haze), fused with weights **calibrated against anchor clips** by `shoal/calibrate.py`, not hand-set. |
| [3] Adaptive restoration | `shoal/restore.py` | **Done.** Grey-world WB → Ancuti red compensation → CLAHE-L → unsharp → luma-driven gamma, **gated** on the fused visibility score. |
| [4] Detection + fallback chain | `shoal/detect.py` | **Done.** 3-rung chain (fine-tuned → marine-detect FishInv → YOLO-World), landing logged. Single-class collapse via `classes=` filter + a post-process callback. |
| [5] Tracking | `shoal/track.py` | **Done.** Ultralytics BoT-SORT + Re-ID, underwater tuning in `config/botsort_underwater.yaml`, tracker YAML reconciled against the installed ultralytics keys. |
| [6] Track repair | `shoal/repair.py` | **Done.** Savitzky-Golay (window validated against clip fps), gap interpolation with flags, conservative fragment stitching, quality flags, min-track filter with a shown "rejected" tally. Emits the §7 `tracks_df` / `track_summary_df` contract (`shoal/schema.py` validates it). |
| [8] Trajectory analytics | `shoal/trajectory.py` | **Done.** MovingPandas `TrajectoryCollection` on a synthetic planar CRS; `TrajectoryStopDetector` → DBSCAN activity nodes → zone-crossing O/D flow matrix; time-weighted residency heatmap. |
| [9] Behaviour | `shoal/behaviour.py` | **Rule-based only** (Phase 1 scope). Cut-points are the clip's own kinematic percentiles, not absolutes. HMM + collective metrics are Phase 2 (see below). |
| [10] Ecology | `shoal/ecology.py` | **Done.** MaxN + unique-track bounds; Shannon/Pielou/Simpson on per-species MaxN; **bootstrap CI over per-track species confidence**; **honesty guard** greys out diversity when the species head is absent/low-confidence (it always is in Phase 1 → guard always fires, correctly). |
| [7] Species head | `shoal/species.py` | **Phase 2 stub.** Real interface + design docstring; raises `NotImplementedError`. Detector runs single-class in Phase 1. |
| [11] Exports | `shoal/export.py` | **csv / geojson / MOTChallenge done.** Darwin Core is Phase 2. |
| Report | `shoal/report.py` | **Phase 2 stub.** All inputs (`run.json`, figures) already produced; dashboard tab 7 shows the provenance. |
| Benchmark | `benchmark/evaluate.py` | **Phase 2 stub.** `shoal.track` is already parametrised over `method`. |
| Dashboard | `app/dashboard.py` | **Tabs 1–5 done.** Tabs 6 (Benchmark) and 7 (Report) are labelled Phase-2 placeholders. |
| A/B ablation | `shoal/ablation.py` | **Done** (plan §4.3). Real inference, raw vs restored, count + confidence deltas, mAP@0.5 delta when GT is present, matched-pair example frames saved. No hardcoded numbers. |

### Additions beyond `SHOAL_build_plan.md §14` (flagged per plan §16)

These are infrastructure, not features:

- `shoal/pipeline.py` — the orchestrator that runs [1]…[11] and writes a
  self-contained `outputs/<run_id>/` directory. §14 lists the modules but no glue.
- `shoal/runlog.py` — run context + provenance capture (the plan §16 "log which
  model / tracker config / code path" requirement made structural).
- `shoal/schema.py` — the §7 dataframe contract as validated column sets.
- `shoal/config.py` — YAML config + deep-merged overrides.
- `shoal/fetch.py`, `shoal/calibrate.py`, `shoal/figures.py` — acquisition,
  visibility calibration (§4.1 requires it), headless figure rendering.
- `config/pipeline.yaml` — one config file with the rationale for every value inline.

---

## Deferred to Phase 2 (unchanged from plan §13 priority list)

1. **HMM behavioural state segmentation** (`hmmlearn`) — replaces the rule-based classifier as the headline method.
2. **Polarisation Φ / nearest-neighbour distance** collective metrics + the shoal-vs-school discriminator.
3. **Species head** (marine-detect FishInv, 15 classes) + track-level confidence voting → real per-individual species + confidence feeding the diversity bootstrap. This is what makes the diversity panel light up instead of being honesty-guarded.
4. **HOTA / MOTA / IDF1 benchmark**, 2×2 ablation (ByteTrack × BoT-SORT) × (raw × restored) on the Brackish test split. `benchmark/evaluate.py` is a stub.
5. **Darwin Core Occurrence export** (`occurrences_dwc.csv`).
6. **Event detection** (startle/escape conjunction) with auto-clipping.
7. **Fine-tuned YOLO26** single-class on merged Brackish + DeepFish (train on Kaggle GPU).
8. **pydeck TripsLayer** animated trajectory playback.
9. **Auto-generated PDF report** (`shoal/report.py`).
10. **ONNX export** + measured FPS table across model sizes.

---

## Known limitations of the Phase 1 demo data (important)

The plan's primary datasets — **Brackish** (bounding-box GT + turbid video + fixed
seabed station) and **DeepFish** — are on Kaggle and need an account. This
environment has no Kaggle credentials, so Phase 1 ships with **CC0 Mixkit stock
clips** instead:

| clip | role | caveat |
|---|---|---|
| `reef_clear.mp4` | primary demo, abundance / tracking / behaviour; "clear" calibration anchor | camera **pans** strongly. Track count ≫ MaxN partly because panning fragments tracks — which is exactly the "track count is an upper bound" point, shown honestly. |
| `anemone_static.mp4` | tightest tracking (two clownfish), darker scene | camera **drifts slowly** (~⅓ frame over the clip) — still flagged non-static |
| `reef_snapper.mp4` | single-subject trajectory | camera **follows the fish** — the most mobile of the four |
| `reef_turbid_matched.mp4` | matched-pair restoration ablation; "turbid" calibration anchor | **synthetic turbidity augmentation** of `reef_clear.mp4` (ffmpeg colour push + haze + blur + noise), clearly labelled. A controlled fixture, not footage. |

**All four stock clips are non-static** — none is a tripod shot, so the camera-motion
guard flags every one and the dashboard reads residency/flow in **image** coordinates,
not fixed world coordinates. This is the guard doing its job, and it is the honest
demo of it: on a real fixed rig (Brackish's seabed station) the same guard validates.
The Phase 2 fix is simply better source footage, not a code change.

Consequences to fix in Phase 2:

- **Visibility calibration** is half-synthetic (clear anchor real, turbid anchor
  augmented). Re-run `shoal-calibrate --clear <DeepFish clip> --turbid <Brackish clip>`
  once those are fetched (`shoal-fetch --brackish` works the moment a Kaggle token
  is present). The *method* is real and reproducible; only the anchor values change.
- **The A/B ablation has no `mAP@0.5` number** because the stock clips have no GT.
  The code path for it is built and tested against MOTChallenge `gt.txt`; it
  activates automatically on Brackish.
- No shoal / schooling footage, so Φ (Phase 2) will first be exercised on Brackish's
  small-fish schools.

---

## Verification checklist (plan §16 — owner confirms before Phase 2)

- [x] Visibility index weights calibrated against anchor clips, not hand-chosen — `shoal/calibrate.py`, sanity check enforces clear > turbid ordering.
- [x] Restoration ablation produces measured numbers on real inference — `shoal/ablation.py`, `outputs/<run>/ablation.json`.
- [x] Savitzky-Golay window justified against clip frame rate — `repair._validated_savgol_window` caps the window at ~0.4 s of frames.
- [~] Track stitching does not merge distinct individuals — conservative rules (gap ≤ 15 frames, plausible speed, heading continuity); **still needs the "inspect 10 stitched tracks by eye" pass** with real GT.
- [x] MaxN and track count both displayed, both labelled as bounds — dashboard tab 5.
- [x] Bootstrap CI actually resamples species confidence — `ecology.compute_diversity` (currently exercised with placeholder conf=1.0 since Phase 1 has one class; guard fires first anyway).
- [x] Every axis label carries its unit — px / px·s⁻¹ / BL·s⁻¹ throughout; metres only with a user scale.
- [x] Honesty guard fires when the species head is disabled — it fires on every Phase 1 run by construction.
- [x] Tensor shapes verified through detect → track → repair — `shoal-check` smoke test + full runs on all four clips + `tests/` (29 passing).

---

## Phase 1 acceptance tests (plan §13 table)

| Milestone | Acceptance test | Status |
|---|---|---|
| Repo scaffold, deps, data + weights | `shoal-check` prints all green | **pass** (2 yellow = provisional-calibration-before-you-run-calibrate, and no fine-tuned weights which is a Phase-2 artefact; 0 red) |
| Detection + tracking → annotated MP4 | video exists, IDs persist across frames | **pass** — `outputs/<run>/annotated.mp4` (H.264, browser-playable), track IDs persist across frames with fading trajectory tails |
| Visibility index + restoration + A/B | ablation prints two real measured numbers | **pass** — count delta + confidence delta from real inference |
| Track repair + MovingPandas + residency | `track_summary_df` populated, heatmap renders | **pass** |
| MaxN, diversity + CI, behaviour | ecology panel shows numbers with intervals | **pass** — MaxN bounds shown; diversity honesty-guarded (correct for 1 class) |
| Streamlit dashboard tabs 1–5 | runs locally end to end without error | **pass** |
| Fallback demo video | fallback MP4 exists | **pass** — three runnable clips, annotated MP4s cached in `outputs/` |
