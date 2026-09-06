# SHOAL — Spatio-temporal Habitat & Organism Analytics Layer

**Challenge:** CHE01 — AI-Based Underwater Fish Detection and Tracking for Spatio-Temporal Marine Monitoring
**Target hardware:** Acer Nitro Lite 16, 6 GB VRAM, Arch Linux
**Build agent:** Claude Code
**Owner:** solo build (team exists but plan assumes one person)

---

## 0. How to use this document

This is a build plan, not a spec to follow blindly. Every module below states **why** a design choice was made so the decision can be checked rather than accepted. Where a number is estimated rather than measured, it is marked `[ESTIMATE]` and must be replaced with a measured value before it appears in any slide or dashboard.

**Two deadlines, one codebase:**

- **Phase 1 (~3 hours)** — working demo for the online mentor pitch. Strict subset of Phase 2. Nothing built here gets thrown away.
- **Phase 2 (48 hours)** — the actual hackathon build. Adds the benchmark, the collective-behaviour layer, the standards-compliant export, and the polish.

Claude Code should build Phase 1 completely and stop for review before starting Phase 2.

---

## 1. Reading the problem statement literally

The brief is worth parsing clause by clause, because each clause is a scoring opportunity and most teams will only answer the first two.

| Clause in the brief | What it demands | Where it is handled |
|---|---|---|
| "detecting and tracking fish in underwater video sequences" | detection + MOT | §5, §6 |
| "acquired under **varying visibility conditions**" | the pipeline must *adapt*, not just run | §4 — adaptive restoration gated on a measured visibility index |
| "transform underwater imagery into valuable **spatio-temporal information**" | trajectories as first-class data, not an overlay | §8 — MovingPandas trajectory objects |
| "analysis of fish **trajectories and movement patterns**" | movement *patterns*, i.e. behaviour, not just paths | §9 — HMM behavioural state segmentation |
| "fisheries management, biodiversity studies, aquaculture, ocean observation" | four named downstream users | §10 — MaxN, diversity indices, Darwin Core export |
| "**non-invasive** monitoring tools" | passive video only, no tagging | inherent; state it explicitly in the pitch |
| "**actionable insights**" | outputs a domain expert can act on | §10, §11 — report generator, GBIF-compatible export |
| Skills wanted: **oceanographer, data analyst** | the jury is not primarily a CV audience | drives the whole weighting: analytics layer > model novelty |

**The single most important strategic read:** the challenge owner asked for an *oceanographer and a data analyst*, not an ML engineer. Most teams will present a YOLO demo with boxes on fish. The differentiation is entirely in what happens **after** the boxes — MaxN, diversity with confidence intervals, behavioural state segmentation, shoal-vs-school discrimination, and an export that plugs into real biodiversity infrastructure. Weight effort accordingly.

---

## 2. Architecture

```
video source
  │
  ├─[1] INGEST ─────────── frame iterator, metadata, camera-motion guard
  │
  ├─[2] VISIBILITY INDEX ─ per-frame turbidity/contrast/colour-cast score  ──┐
  │                                                                          │
  ├─[3] RESTORATION ────── adaptive; only fires when visibility < threshold ←─┘
  │
  ├─[4] DETECTION ──────── class-agnostic "fish" detector (high recall)
  │
  ├─[5] TRACKING ───────── BoT-SORT w/ Re-ID, persist across frames
  │
  ├─[6] TRACK REPAIR ───── smoothing, gap-fill, stitching, quality flags
  │
  ├─[7] SPECIES HEAD ───── crop classifier + track-level confidence voting
  │
  ├─[8] TRAJECTORY ─────── MovingPandas: speed, sinuosity, stops, clusters, flows
  │
  ├─[9] BEHAVIOUR ──────── HMM state segmentation + collective metrics
  │
  ├─[10] ECOLOGY ───────── MaxN, Shannon H′ + CI, Pielou J′, Simpson, richness
  │
  └─[11] OUTPUT ────────── Streamlit dashboard, exports, auto report
```

### The one architectural decision to defend out loud

**Detection and species identification are decoupled.** Module [4] runs a single-class, high-recall "fish" detector. Module [7] runs a separate classifier on tracked crops.

Why this matters and why it is not the obvious choice:

1. A multi-class detector couples localisation to classification. If the model flips a parrotfish to a snapper on frame 40, the tracker sees a class change and the association logic can break the track. Single-class detection gives the tracker one stable job.
2. Species identification is far more reliable at the **track** level than the **frame** level. A track gives ~50–200 views of the same animal at different angles and distances. Confidence-weighted voting over the whole track converts a mediocre per-frame classifier into a good per-individual one — and, critically, yields a *per-individual confidence* that propagates into the diversity index.
3. Recall matters more than precision for the ecology layer. A missed fish is a missing data point that biases MaxN downward. A false positive can be filtered later by track length and quality flags. Single-class training lets recall be tuned aggressively.

If a juror asks "why not just train a 17-class YOLO?", this is the answer. It is a real engineering argument, not a rationalisation.

---

## 3. Data — acquire before anything else

Wifi is good, so fetch everything up front. Cache to `data/` and never re-download.

### 3.1 Primary: Brackish (Aalborg University)

- **Source:** Kaggle — `aalborguniversity/brackish-dataset`. Also mirrored on Roboflow Universe in YOLO format.
- **Why this one:** it is the only easily-available underwater set that gives you *all three* of: bounding-box ground truth, video sequences (not just stills), and **genuinely varying turbidity** — which is the literal wording of the challenge brief. It is also from a **fixed seabed station**, which is what makes residency maps and flow matrices scientifically meaningful rather than decorative.
- **Classes:** fish, small fish, crab, shrimp, jellyfish, starfish.
- **Role:** ground truth for the benchmark; the "hard visibility" demo clip.

### 3.2 Secondary: DeepFish (JCU)

- **Source:** `github.com/alzayats/DeepFish`
- **Why:** ~40k images from 20 tropical Australian habitats, frame-mounted cameras. Cleaner water, more visually appealing, and habitat diversity supports the biodiversity narrative.
- **Role:** the "good visibility" demo clip and a supplementary training source.

### 3.3 Species model: marine-detect (Orange Business × Tēnaka)

- **Source:** `github.com/Orange-OpenSource/marine-detect` — README links to downloadable `FishInv` and `MegaFauna` model files plus YOLO-format datasets.
- **Coverage:** 17 bio-indicating species (Chaetodontidae, Serranidae, Scaridae, Lutjanidae, Muraenidae, Haemulidae, *Cromileptes altivelis*, *Cheilinus undulatus*, *Bolbometopon muricatum*, giant clam, urchin, sea cucumber, lobster, crown-of-thorns, plus sharks/turtles/rays).
- **Honest caveat that must be stated in the pitch:** these models were trained on footage from Tioman, Malaysia. They are *regionally scoped*. Presenting them as a universal species identifier is a claim the model does not support. Present them as "a drop-in regional species head — swap the head, keep the pipeline" and the architecture becomes the contribution rather than the weights.

### 3.4 Synthetic clip (optional, clearly labelled)

Generated video is **not** the demo spine, for three reasons: generated fish morph and duplicate between frames, producing ID switches the tracker doesn't deserve blame for; there is no ground truth, so no metrics; and a marine-science jury will recognise it immediately.

It has exactly one good use: an explicitly-labelled **out-of-distribution stress test** — "here is footage the model has never seen anything like, and here is how the confidence-gating behaves." Framed that way it is a strength. Framed as real footage it is a liability.

**Veo / Gemini video prompt (for the labelled stress-test clip):**

```
A static, fixed underwater camera mounted on a sandy seabed at 8 metres depth,
looking horizontally across a temperate coastal reef. Natural daylight filtering
from above, visible blue-green colour cast and moderate suspended particulate
matter drifting through the frame. A loose aggregation of 12-18 silver-bodied
schooling fish, roughly 20cm each, swims across the frame from left to right at
a steady pace, maintaining alignment. Two larger solitary fish move slowly in the
opposite direction in the background. Occasional small particles drift past the
lens. The camera does not move, pan, or zoom at any point. No cuts, no transitions,
no text, no people, no divers, no equipment. Documentary nature-footage realism,
slightly soft focus at distance due to water turbidity. 10 seconds, continuous
single shot.
```

Key constraints in that prompt and why each is there: *static camera* (so the spatial analytics remain valid), *no cuts* (a cut destroys every track), *consistent fish count and direction* (gives the polarisation metric something real to measure), *visible turbidity* (exercises the restoration module).

---

## 4. Module [2] + [3] — Visibility index and adaptive restoration

### 4.1 Visibility index (build this first, it is the hook)

Compute per frame, all cheap OpenCV operations:

```python
def visibility_index(frame_bgr) -> dict:
    """
    Returns a dict of the four raw components plus a fused 0-1 score.
    All components are computed on the frame as-is; no restoration applied.
    """
    # 1. contrast      -> std dev of the L channel in LAB
    # 2. colour_cast   -> distance of (mean_a, mean_b) from the neutral point (128,128)
    # 3. sharpness     -> variance of the Laplacian
    # 4. haze          -> mean of the dark channel (min across BGR, 15x15 erosion)
```

Fuse into a single 0–1 score with fixed weights. **Calibrate the weights on real clips, do not invent them** — run the four components over a clear DeepFish clip and a turbid Brackish clip, normalise each component against those two anchors, then fuse. Record the anchor values in `config/visibility_calibration.json` so the number is reproducible and defensible.

### 4.2 Adaptive restoration

**The gate is the point.** Restoration only fires when `visibility < threshold`. This is not a performance optimisation dressed up as science — over-enhancing already-clear frames introduces halo artefacts around fish edges and measurably hurts detection. The gate is the direct engineering answer to "varying visibility conditions" in the brief.

Enhancement chain (all OpenCV, CPU, target < 8 ms at 720p — **measure this, do not quote the estimate**):

1. **Grey-world white balance in LAB** — shifts the a/b channels toward neutral.
2. **Red channel compensation** (Ancuti-style): `I_r' = I_r + α·(mean_g − mean_r)·(1 − I_r)·I_g`. Red attenuates fastest in water; this recovers it from the green channel, which survives longest. α ≈ 1.0, tune on data.
3. **CLAHE on the L channel only** — clip limit 2.0, tile grid 8×8. L-only avoids the colour-noise amplification you get from CLAHE on RGB.
4. **Mild unsharp mask** — restores edge definition lost to scattering.
5. **Gamma correction** — driven by the measured frame luminance, not a constant.

### 4.3 The ablation that makes this module undeniable

Do not claim restoration helps. **Measure it, live, in the dashboard.**

Run detection twice on a held-out sample of N frames — once raw, once restored — and report:

- detection count delta
- mean confidence delta
- if GT is available (Brackish): **mAP@0.5 delta**, which is the real number

Display as a two-bar chart with the actual numbers. If restoration turns out *not* to help on a given clip, show that too and let the gate turn it off. A team that shows a negative result on one clip and a positive on another has demonstrated methodology. A team that shows only positives has demonstrated marketing.

**Acceptance test:** the ablation runs end to end and produces real numbers from real inference. No hardcoded values anywhere in this module.

---

## 5. Module [4] — Detection

### Model choice for 6 GB VRAM

- **Primary:** `yolo26s` fine-tuned to single-class fish. YOLO26 is NMS-free end-to-end, which removes a latency-variable stage and simplifies the ONNX export path later.
- **Fallback if YOLO26 fine-tuning misbehaves:** `yolo11s`. Still current and explicitly recommended for stable production workloads alongside YOLO26.
- **Emergency fallback:** an off-the-shelf Roboflow Universe fish detector, zero training.

Build the fallback chain as actual code — `models/loader.py` tries each in order and logs which one it landed on. Do not leave it as a note in a README.

### Training

- Merge Brackish + DeepFish detection subset, collapse all fish-like classes to a single `fish` class. Keep crab/starfish/jellyfish as separate classes only if time permits; they are a bonus, not the target.
- `imgsz=640`, `batch=8` on 6 GB (`batch=16` if it fits after checking `nvidia-smi` under load), AMP on, ~40–60 epochs.
- **Run this on Kaggle GPU, not the laptop.** Kaggle quota is the right resource for a one-off training run; the laptop is the demo machine and should not be tied up.
- Augmentation targeted at the actual failure mode: heavy HSV jitter (colour cast varies enormously underwater), moderate blur, no vertical flip (fish have a dorsal/ventral orientation and vertical flipping teaches the model a body plan that doesn't exist).

### Confidence threshold

Set it **low** (0.15–0.25) and let track-level filtering clean up. Rationale: turbid water depresses confidence scores across the board, and the tracker's low-score association stage (§6) is specifically designed to recover exactly those detections. A high threshold throws away the signal the tracker was built to use.

---

## 6. Module [5] — Tracking

Use Ultralytics' built-in tracking: `model.track(source=..., persist=True, tracker="botsort.yaml")`.

**BoT-SORT with Re-ID as primary**, ByteTrack as the comparison arm in the benchmark. Ultralytics shipped a Re-ID upgrade to the tracking stack in June 2026 targeting identity stability under occlusion and in crowded scenes — which is the defining failure mode for shoaling fish.

### Underwater-specific tuning (put these in `config/botsort_underwater.yaml`)

| Parameter | Change | Reason |
|---|---|---|
| `track_buffer` | raise to ~60–90 frames | fish leave frame and return; default buffer kills the ID |
| `match_thresh` | loosen slightly | erratic, non-linear fish motion violates the constant-velocity assumption |
| `new_track_thresh` | raise | suppresses spurious tracks from particulate matter |
| low-conf association | keep enabled | the whole point in turbid water |

### The narrative link worth making in the pitch

Turbidity depresses detection confidence. Low-confidence detections are exactly what ByteTrack's second association stage was designed to recover. So the visibility problem and the tracking solution are the same problem viewed from two ends — and that is why the restoration gate and the tracker configuration are tuned together rather than independently. This is a coherent systems story, and coherence is what separates a demo from a prototype in a jury's eyes.

---

## 7. Module [6] — Track repair

**This module is unglamorous and load-bearing. Do not skip it.**

Raw bounding-box centroids jitter by several pixels frame to frame. Speed is the first derivative of position, so that jitter is *amplified* — computing speed on raw centroids produces numbers that are mostly noise. Every downstream ecological metric would then be built on sand.

1. **Smoothing:** Savitzky-Golay filter on (x, y) independently. Window 9, polyorder 2. Preserves the peaks that matter for burst detection, unlike a moving average which flattens them.
2. **Gap filling:** linear interpolation across gaps ≤ 5 frames. Flag interpolated points so they can be excluded from metrics.
3. **Stitching:** link track fragments separated by ≤ 15 frames where the motion is consistent and (if Re-ID is on) appearance embeddings match. Reduces the ID-fragmentation that would otherwise inflate individual counts.
4. **Quality flags** per track: `n_frames`, `frac_interpolated`, `mean_detection_conf`, `n_gaps`, `bbox_area_stability`.
5. **Minimum viable track:** ≥ 15 frames. Shorter tracks are excluded from ecology metrics but still counted in a "rejected" tally that gets shown, so the filtering is visible rather than hidden.

**Output contract — every downstream module consumes this schema:**

```
tracks_df:
  track_id | frame | timestamp | x | y | w | h | conf
  | is_interpolated | speed_px_s | heading_rad | turn_angle_rad
```

```
track_summary_df:
  track_id | n_frames | duration_s | path_length_px | net_displacement_px
  | sinuosity | mean_speed_px_s | max_speed_px_s | species | species_conf
  | quality_flag | behaviour_state
```

---

## 8. Module [8] — Trajectory analytics (MovingPandas)

Pixel coordinates are treated as a **local planar CRS**. This is legitimate — MovingPandas operates on any planar coordinate system, it does not require geographic coordinates. Wrap the track table in a `GeoDataFrame` with a synthetic CRS and build a `TrajectoryCollection`.

### Units — non-negotiable

Without camera calibration, **every distance is in pixels and every speed is in px/s**. The dashboard must say so, in the axis labels, not in a footnote.

Provide two optional calibration paths in the UI:

1. **Scale bar** — user enters a known real-world distance and drags a line across the frame. Yields px→m.
2. **Body-length normalisation** — express speed in *body lengths per second* (BL/s) using the median bounding-box width of a track as the body-length proxy. This requires no calibration at all and BL/s is a **standard unit in fish swimming ecology**, directly comparable to published literature. This is the better default and it is scientifically stronger than a guessed metre conversion.

The moment a scale is supplied, the entire dashboard switches to metres and the units update everywhere. Until then, it says px/s and BL/s. Never display a metre value derived from an assumed scale.

### Derived quantities

- Per-point speed, heading, turning angle, acceleration
- **Sinuosity** = path length / net displacement (1.0 = straight line; high = tortuous, characteristic of foraging)
- Per-track summary statistics

### Spatial aggregation (follows the MovingPandas `9-aggregating-trajectories` recipe)

1. `TrajectoryStopDetector` → stop points (max diameter, min duration tuned to the clip)
2. **DBSCAN** on stop points → activity nodes / habitat hotspots
3. Voronoi or convex-hull cells around node centroids → data-driven habitat zones
4. Segment trajectories by zone crossing → **origin-destination flow matrix**
5. Render as a directed flow map with edge width ∝ transition count

### Residency heatmap

Time-weighted 2D occupancy histogram over the frame. Gaussian-smoothed, overlaid on a representative frame. This is the single most immediately legible visual in the whole project — a juror understands it without explanation.

---

## 9. Module [9] — Behaviour and collective dynamics

**This is the section that wins the challenge.** It is what an oceanographer would build and what a CV team would never think to.

### 9.1 Behavioural state segmentation — Hidden Markov Model

Do **not** use hand-tuned thresholds as the headline method. Use an HMM on (step length, turning angle) per track.

This is not a novelty choice — **HMM state segmentation is the standard method in movement ecology** (the same approach behind tools like `momentuHMM`). Using the field's own standard method, rather than inventing rules, is precisely the signal that lands with a domain jury.

- Implementation: `hmmlearn`, Gaussian emissions, 3 hidden states, fitted per clip.
- Interpret states post hoc by their fitted parameters:
  - low step length + high turning variance → **area-restricted search / foraging**
  - high step length + low turning variance → **transit / cruising**
  - very high step length + burst acceleration → **escape / startle**
- Report the fitted transition matrix. It is a real, interpretable result — "fish transition from transit to foraging with probability 0.08 per frame in the reef zone" is an ecological finding, not a demo feature.
- Keep the rule-based classifier as a fallback if the HMM fails to converge on short clips, and label it as such in the UI.

### 9.2 Collective behaviour — the shoal/school discriminator

Computed per frame across all simultaneously tracked individuals:

- **Polarisation order parameter** Φ = |(1/N) Σ v̂ᵢ| where v̂ᵢ is the unit velocity vector of individual *i*. Φ ≈ 0 → disordered aggregation. Φ ≈ 1 → coordinated alignment.
- **Mean nearest-neighbour distance** (NND)
- **Group membership** via DBSCAN on per-frame positions

These two numbers together make a **real biological distinction that most people conflate**: a *shoal* is fish aggregated socially but not aligned (low NND, low Φ); a *school* is fish aggregated **and** polarised (low NND, high Φ). Being able to say "at t=14s this shoal transitions to schooling, Φ rises from 0.2 to 0.8" is a genuine behavioural-ecology observation delivered from video, automatically.

Plot Φ and NND as time series beneath the video. Watching Φ spike as the fish align is the most compelling ten seconds of the demo.

### 9.3 Event detection

A **startle/escape response** has a specific, measurable signature: a simultaneous spike in mean speed, collapse in Φ, and expansion of NND. Detect the conjunction, flag the timestamp, auto-extract the clip.

Frame this correctly in the pitch: the value is **triage**. A monitoring station generates thousands of hours of video. An automatic flag saying "something happened at 04:17:33" is what makes continuous monitoring tractable for a human researcher. That is the "actionable insights" clause of the brief, answered concretely.

---

## 10. Module [10] — Ecological indicators

### 10.1 Abundance — use MaxN, not track count

**MaxN** = the maximum number of individuals of a species visible in any single frame. This is the **standard abundance metric in baited/remote underwater video fisheries surveys**, and using it instead of a raw track count is the clearest possible signal of domain literacy.

The technical reason it is also just *better*: MaxN is robust to ID switches. If the tracker fragments one fish into three tracks, the track count triples but MaxN is unchanged. Track count is an *upper bound* on abundance; MaxN is a conservative *lower bound*.

**Report both, side by side, labelled as bounds.** "MaxN = 7 (conservative), unique tracks = 11 (upper bound, sensitive to ID fragmentation)." A juror who knows the field will recognise instantly that you know what you're doing; one who doesn't will appreciate the honesty about uncertainty.

### 10.2 Diversity indices

Computed on **per-species MaxN**, not per-detection counts. Per-detection counting weights by how long each fish happened to stay in frame, which is a residency artefact, not a diversity measurement.

- **Shannon–Wiener** H′ = −Σ pᵢ ln pᵢ
- **Pielou evenness** J′ = H′ / ln(S)
- **Simpson** 1 − D
- **Richness** S

### 10.3 Confidence intervals — the credibility move

A bare H′ = 1.84 invites the question "how sure are you?" and there is no good answer.

Instead: **bootstrap the index over the per-track species confidences.** Each track carries a species label and a confidence from the track-level voting in §7. Resample track species assignments according to those confidences, recompute H′ 1000 times, report the median and the 95% interval.

Output: **H′ = 1.84 [1.61 – 2.02]**.

This costs about twenty lines of code and it is the difference between a number and a measurement. It also means that when species classification is weak, the interval widens visibly and honestly rather than the dashboard printing a confident-looking wrong number.

### 10.4 Mandatory honesty guard

If the species head is unavailable or its mean track confidence falls below a set threshold, the dashboard must **grey out the diversity panel** and display: *"Species identification confidence too low for diversity estimation on this clip. Abundance metrics remain valid."*

Build this as an actual code path, not a policy. Diversity computed over a single collapsed `fish` class is mathematically zero-information and presenting it would be the one thing in this project a knowledgeable juror could legitimately attack.

---

## 11. Module [11] — Outputs

### 11.1 Streamlit dashboard

Tabs:

1. **Live / Playback** — annotated video, boxes with track IDs, fading trajectory tails, live counter
2. **Visibility & Restoration** — before/after slider, visibility index time series, the A/B ablation bar chart
3. **Trajectories** — residency heatmap, activity nodes, flow map, per-track table
4. **Behaviour** — HMM state ribbon over time, Φ and NND time series, flagged events with clip thumbnails
5. **Ecology** — MaxN and track-count bounds, diversity indices with CIs, species composition
6. **Benchmark** — the ablation table (Phase 2)
7. **Report** — one-click PDF generation

### 11.2 Exports — the viability answer

- `tracks.csv` — full per-frame table
- `trajectories.geojson` — LineString per track, attributes attached
- `results_MOT.txt` — MOTChallenge format, so any standard tracking evaluator can verify the results independently
- **`occurrences_dwc.csv` — Darwin Core Occurrence format**

The Darwin Core export is the highest-leverage twenty minutes in the entire build. Darwin Core is the GBIF biodiversity data standard. Emitting a compliant occurrence record means the pipeline's output can flow directly into global biodiversity infrastructure without a human retyping anything.

Minimum viable fields: `occurrenceID`, `eventDate`, `decimalLatitude`, `decimalLongitude`, `scientificName`, `individualCount`, `basisOfRecord` = `MachineObservation`, `identificationVerificationStatus`, `associatedMedia`.

When the jury asks the standard viability question — "who would actually use this and how does it fit their workflow?" — the answer is a file, on screen, in the format their existing systems already ingest. That is a materially stronger answer than a slide.

### 11.3 Auto-generated report

One page: clip metadata, visibility profile, abundance bounds, diversity with CIs, behavioural state breakdown, flagged events, method summary with library versions and model checkpoints. Reproducibility metadata included, because a scientific tool that doesn't record its own configuration isn't a scientific tool.

---

## 12. Benchmark (Phase 2)

Run on Brackish test split with ground truth. Use `motmetrics` or TrackEval.

**2×2 ablation:**

| | ByteTrack | BoT-SORT + Re-ID |
|---|---|---|
| **Raw video** | HOTA / MOTA / IDF1 | HOTA / MOTA / IDF1 |
| **Restored video** | HOTA / MOTA / IDF1 | HOTA / MOTA / IDF1 |

Report **HOTA as the headline**, not MOTA. HOTA balances detection and association performance; MOTA is dominated by detection count and systematically flatters a good detector paired with a mediocre tracker. Choosing HOTA and being able to say why is itself a credibility signal.

Compute **offline, once**. Cache to JSON. Render as a static card. Never re-run during a live demo.

---

## 13. Build order

### Phase 1 — 3 hours, mentor demo

| Time | Milestone | Acceptance test |
|---|---|---|
| 0:00–0:20 | Repo scaffold, venv, deps, data + weights fetched and cached | `python -m shoal.check` prints all green |
| 0:20–0:45 | Detection + tracking on a real clip → annotated MP4 out | video file exists, IDs visibly persist across frames |
| 0:45–1:05 | Visibility index + adaptive restoration + A/B ablation | ablation prints two real measured numbers |
| 1:05–1:35 | Track repair + MovingPandas trajectories + residency heatmap | `track_summary_df` populated, heatmap renders |
| 1:35–2:05 | MaxN, diversity + bootstrap CI, rule-based behaviour classifier | ecology panel shows numbers with intervals |
| 2:05–2:45 | Streamlit dashboard, tabs 1–5 | runs locally end to end without error |
| 2:45–3:00 | Record fallback demo video, sanity check | fallback MP4 exists |

**Cut line if behind schedule:** drop behaviour classification first, then the flow map. Never cut the residency heatmap (highest visual impact per minute), the visibility ablation (the only measured result), or the units labelling (the honesty guard).

### Phase 2 — 48 hours

Priority order, highest value first:

1. HMM behavioural state segmentation (replaces the rule-based classifier)
2. Polarisation / NND collective metrics + shoal-vs-school discriminator
3. Species head + track-level confidence voting + honesty guard
4. HOTA/MOTA benchmark, 2×2 ablation
5. Darwin Core export
6. Event detection with auto-clipping
7. Fine-tuned YOLO26 on merged Brackish + DeepFish
8. pydeck TripsLayer animated trajectory playback
9. Auto-generated PDF report
10. ONNX export + measured FPS table across model sizes

---

## 14. Repository layout

```
shoal/
├── config/
│   ├── botsort_underwater.yaml
│   ├── visibility_calibration.json
│   └── pipeline.yaml
├── shoal/
│   ├── ingest.py           # frame source, camera-motion guard
│   ├── visibility.py       # [2] index
│   ├── restore.py          # [3] adaptive enhancement
│   ├── detect.py           # [4] + fallback chain
│   ├── track.py            # [5]
│   ├── repair.py           # [6] smoothing, stitching, quality
│   ├── species.py          # [7] crop classifier + voting
│   ├── trajectory.py       # [8] MovingPandas
│   ├── behaviour.py        # [9] HMM + collective metrics
│   ├── ecology.py          # [10] MaxN, indices, bootstrap
│   ├── export.py           # [11] CSV/GeoJSON/MOT/DwC
│   ├── report.py           # PDF generation
│   └── check.py            # preflight: deps, weights, data, GPU
├── app/
│   └── dashboard.py        # Streamlit
├── benchmark/
│   └── evaluate.py
├── data/                   # gitignored
├── weights/                # gitignored
└── outputs/
```

---

## 15. Dependencies

```
ultralytics>=8.3          # YOLO26 + BoT-SORT Re-ID
opencv-python-headless
numpy, pandas, scipy
movingpandas, geopandas, shapely
scikit-learn              # DBSCAN
hmmlearn                  # behavioural state segmentation
streamlit, plotly, pydeck
motmetrics                # benchmark
reportlab                 # PDF report
```

**Licensing note worth having ready:** Ultralytics YOLO is AGPL-3.0. For a hackathon prototype this is fine, but "viability" is a judging criterion and a sharp juror may ask about commercialisation. The prepared answer: the pipeline is model-agnostic by design — the detector sits behind an interface in `detect.py`, and swapping in a permissively-licensed detector is a config change, not a rewrite. Having thought about this before being asked is worth more than the answer itself.

---

## 16. Constraints for Claude Code

**Do not:**
- Hardcode any metric, FPS figure, or accuracy number. Every number displayed must trace to a computation performed at runtime.
- Display distances in metres unless a calibration has been explicitly supplied by the user.
- Compute or display diversity indices when the species head is absent or low-confidence — trigger the honesty guard instead.
- Compute speed from unsmoothed centroids.
- Re-run the benchmark during a live demo.
- Add a feature not listed in this plan without flagging it.

**Do:**
- Write the fallback chains as real, tested code paths.
- Put a docstring on every non-obvious function stating *why* the approach was chosen, not just what it does.
- Keep module boundaries clean — each consumes and emits the dataframe schemas in §7.
- Log which model, which tracker config, and which code path was actually taken on every run.
- Stop after Phase 1 for review.

**Verification checklist (owner reads and confirms each before Phase 2):**

- [ ] Visibility index weights calibrated against real anchor clips, not chosen by hand
- [ ] Restoration ablation produces measured numbers on real inference
- [ ] Savitzky-Golay window and order justified against the clip frame rate
- [ ] Track stitching does not merge distinct individuals — inspect 10 stitched tracks by eye
- [ ] MaxN and track count both displayed, both labelled as bounds
- [ ] Bootstrap CI actually resamples species confidence, not a placeholder
- [ ] Every axis label carries its unit
- [ ] Honesty guard fires correctly when the species head is disabled
- [ ] Tensor shapes verified through detect → track → repair

---

## 17. The pitch narrative

Three sentences, in this order:

1. **Problem framing.** "Underwater video is abundant and cheap; analysis of it is scarce and expensive. The bottleneck is not cameras, it's the human hours between footage and finding."
2. **What was built.** "SHOAL turns raw underwater video into trajectory data, then into the specific quantities a marine scientist already uses — MaxN abundance, Shannon diversity with confidence intervals, and behavioural state segmentation — and exports them in the GBIF standard format."
3. **Why it holds up.** "Every number on screen is measured, not asserted. Where the method is uncertain, the interface says so. The restoration module proves its own value with a live A/B test, and the tracker is benchmarked against ground truth with HOTA."

Lead the live demo with the residency heatmap and the polarisation time series. Those two visuals are legible in under five seconds without explanation, which is roughly the attention budget a juror gives any single screen.
