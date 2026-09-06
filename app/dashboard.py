"""SHOAL dashboard (plan §11.1).

Reads a completed run directory (``outputs/<run_id>/``). It never runs the
pipeline itself, so every number shown was computed and cached by ``shoal.pipeline``
and traces to that run's ``run.json``.

    uv run streamlit run app/dashboard.py

Tabs (Phase 1 = 1..5; 6 and 7 are Phase-2 placeholders kept for structure):
  1 Live / Playback      annotated video, per-frame count, camera-motion verdict
  2 Visibility & Restore  before/after, visibility time series, the A/B ablation
  3 Trajectories          residency heatmap, activity nodes, flows, per-track table
  4 Behaviour             rule-based state ribbon (HMM + collective metrics = Phase 2)
  5 Ecology               MaxN vs track-count bounds, diversity + CI (honesty guard)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = REPO_ROOT / "outputs"
UPLOAD_DIR = REPO_ROOT / "data" / "clips" / "uploads"

st.set_page_config(page_title="SHOAL", layout="wide", page_icon="🐟")

# Visual pass, matched to the landing page (web/). Colours, fonts and radius come
# from .streamlit/config.toml; this only trims Streamlit chrome and tightens a
# few components. Every rule degrades to a no-op if a selector moves.
st.markdown(
    """
    <style>
      [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
      .block-container { padding-top: 2.6rem; padding-bottom: 3rem; max-width: 1440px; }
      h1 { letter-spacing: -0.021em; font-weight: 600; }
      h2, h3, h4 { letter-spacing: -0.014em; }
      .stTabs [data-baseweb="tab-list"] { gap: 0.15rem; border-bottom: 1px solid #26333b; }
      .stTabs [data-baseweb="tab"] { padding: 0.45rem 0.85rem; font-size: 0.88rem; }
      [data-testid="stMetric"] {
        background: #121a22; border: 1px solid #222e37;
        border-radius: 10px; padding: 0.85rem 1rem;
      }
      [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 500; }
      [data-testid="stMetricLabel"] { opacity: 0.75; }
      section[data-testid="stSidebar"] { border-right: 1px solid #222e37; }
      hr { margin: 0.9rem 0; border-color: #222e37; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def list_runs() -> list[str]:
    if not OUTPUTS.exists():
        return []
    return sorted((p.name for p in OUTPUTS.iterdir()
                   if p.is_dir() and (p / "results.json").exists()), reverse=True)


@st.cache_data(show_spinner=False)
def load_run(run_id: str) -> dict:
    d = OUTPUTS / run_id
    out: dict = {"dir": d, "results": json.loads((d / "results.json").read_text())}
    for name in ("ecology", "camera_motion", "ablation", "run"):
        p = d / f"{name}.json"
        out[name] = json.loads(p.read_text()) if p.exists() else None
    for name in ("visibility", "tracks", "track_summary", "behaviour_timeline",
                 "abundance_per_frame", "stops", "activity_nodes", "flow_matrix"):
        p = d / f"{name}.csv"
        out[name] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    hist_p = d / "residency_hist.npy"
    out["residency_hist"] = np.load(hist_p) if hist_p.exists() else None
    meta_p = d / "residency_meta.json"
    out["residency_meta"] = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    return out


@st.cache_data(show_spinner=False)
def clip_frame(clip_path: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(clip_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, bgr = cap.read()
    cap.release()
    return bgr[..., ::-1].copy() if ok else None


def _np_to_data_uri(rgb: np.ndarray) -> str:
    import base64

    ok, buf = cv2.imencode(".jpg", rgb[..., ::-1])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def units_note(scale_px_per_m: float | None) -> str:
    if scale_px_per_m:
        return f"scale set: {scale_px_per_m:.1f} px/m, so metre values are shown alongside px"
    return "no scale supplied, so distances are in **pixels** and speeds in **px/s** and **body-lengths/s**"


def run_pipeline_ui(clip_path: Path, run_id: str, max_frames: int | None,
                    device: str, ablation: bool) -> bool:
    """Run `python -m shoal.pipeline` as a subprocess, streaming its log into the
    page. Returns True when outputs/<run_id>/results.json was produced."""
    cmd = [sys.executable, "-m", "shoal.pipeline", "--clip", str(clip_path),
           "--run-id", run_id, "--device", device]
    if max_frames:
        cmd += ["--max-frames", str(int(max_frames))]
    if not ablation:
        cmd += ["--no-ablation"]

    with st.status(f"Processing {clip_path.name} ...", expanded=True) as status:
        log_box = st.empty()
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1", "UV_LINK_MODE": "copy"},
        )
        tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line or "Stone Soup" in line or "warnings.warn" in line \
                    or "not enough matching" in line or "nonzero nanoseconds" in line:
                continue
            tail.append(line.split(" INFO ", 1)[-1] if " INFO " in line else line)
            log_box.code("\n".join(tail[-16:]), language="text")
        rc = proc.wait()
        ok = rc == 0 and (OUTPUTS / run_id / "results.json").exists()
        if ok:
            status.update(label=f"Done: {run_id}", state="complete", expanded=False)
        else:
            status.update(label=f"Pipeline exited with code {rc} (no results.json)", state="error")
    return ok


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
runs = list_runs()

with st.sidebar:
    st.markdown("## SHOAL")
    st.caption("Spatio-temporal Habitat & Organism Analytics Layer")

    with st.expander("▶  Process a new clip", expanded=True):
        up = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv", "webm", "m4v"])
        path_in = st.text_input("...or a path already on disk",
                                placeholder="data/clips/reef_clear.mp4")
        quick = st.checkbox("Quick preview", value=True,
                            help="cap at the first 150 frames for a fast result")
        dev = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
        do_abl = st.checkbox("Also run the restoration A/B", value=False,
                             help="adds ~40 s; the A/B tab needs this")
        if st.button("Run pipeline", type="primary", width="stretch"):
            src: Path | None = None
            if up is not None:
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                src = UPLOAD_DIR / up.name
                src.write_bytes(up.getbuffer())
            elif path_in.strip():
                p = Path(path_in.strip()).expanduser()
                src = p if p.is_absolute() else (REPO_ROOT / p)
            if src is None or not src.exists():
                st.error("Choose a file or enter a path that exists.")
            else:
                rid = f"{src.stem}_{datetime.now():%m%d-%H%M%S}"
                if run_pipeline_ui(src, rid, 150 if quick else None, dev, do_abl):
                    list_runs.clear()
                    load_run.clear()
                    st.session_state["run_id"] = rid
                    st.rerun()

    runs = list_runs()
    if not runs:
        st.info("No completed runs yet. Use **Process a new clip** above, or run "
                "`uv run shoal-run --clip data/clips/reef_clear.mp4` in a terminal.")
        st.stop()

    if st.session_state.get("run_id") not in runs:
        st.session_state["run_id"] = runs[0]
    run_id = st.selectbox("Run", runs, key="run_id")
    run = load_run(run_id)
    R = run["results"]

    st.markdown("---")
    st.markdown("### Scale calibration (optional)")
    st.caption("Plan §8: no metre value is shown unless a scale is supplied here.")
    use_scale = st.checkbox("I have a known distance in the frame")
    scale_px_per_m = None
    if use_scale:
        known_m = st.number_input("Known real-world distance (m)", 0.01, 1000.0, 1.0, 0.1)
        known_px = st.number_input("...spans this many pixels", 1.0, 10000.0, 200.0, 1.0)
        scale_px_per_m = known_px / known_m
    st.info(units_note(scale_px_per_m))

    st.markdown("---")
    meta = R["clip_meta"]
    st.markdown(f"**clip** `{Path(R['clip']).name}`  \n"
                f"{meta['width']}×{meta['height']}, {meta['fps']:.0f} fps, "
                f"{meta['duration_s']:.1f}s  \n"
                f"**device** {R['device']}  ")
    if run["run"]:
        pv = run["run"].get("package_versions", {})
        st.caption("key library versions: " + ", ".join(
            f"{k} {v}" for k, v in list(pv.items())[:6]))


st.title(f"SHOAL  ·  {Path(R['clip']).name}")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["1 · Live / Playback", "2 · Visibility & Restoration", "3 · Trajectories",
     "4 · Behaviour", "5 · Ecology", "6 · Benchmark", "7 · Report"]
)


# --------------------------------------------------------------------------- #
# tab 1: live / playback
# --------------------------------------------------------------------------- #
with tab1:
    cam = R["camera_motion"]
    (st.success if cam["is_static"] else st.warning)(f"**Camera-motion guard:** {cam['verdict']}")

    c1, c2 = st.columns([3, 2])
    with c1:
        vid = run["dir"] / "annotated.mp4"
        if vid.exists():
            st.video(str(vid))
        else:
            st.info("annotated.mp4 not found for this run")
    with c2:
        st.metric("Tracks accepted", R["n_tracks_accepted"],
                  help="tracks ≥ min_track_frames; short tracks are excluded from ecology")
        rp = R["repair"]
        st.metric("Rejected as short", rp.get("n_tracks_rejected_short", 0))
        st.metric("Interpolated points", rp.get("n_points_interpolated", 0),
                  help="gap-filled positions, flagged and excluded from metrics")
        st.metric("Stitched fragments", len(rp.get("stitched_track_ids", [])))

    apf = run["abundance_per_frame"]
    if not apf.empty:
        fig = px.area(apf, x="timestamp", y="count",
                      labels={"timestamp": "time (s)", "count": "fish in frame (distinct track ids)"})
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# tab 2: visibility and restoration
# --------------------------------------------------------------------------- #
with tab2:
    vis = run["visibility"]
    gate = float(run["run"]["config"]["restore"]["gate_threshold"]) if run["run"] else 0.55

    if vis.empty:
        st.info("no visibility trace for this run")
    else:
        prov = R["visibility"].get("provisional", True)
        if prov:
            st.warning("Visibility calibration is **provisional**. Run `uv run shoal-calibrate`. "
                       "Scores are still consistent within this clip.")
        st.markdown("#### Before / after restoration")
        st.caption("The restoration chain is applied here on demand so you can inspect any frame.")
        fidx = st.slider("frame", int(vis["frame"].min()), int(vis["frame"].max()),
                         int(vis.loc[vis["visibility_score"].idxmin(), "frame"]))
        vrow = vis.iloc[(vis["frame"] - fidx).abs().idxmin()]
        st.caption(f"frame {fidx}: visibility score **{vrow['visibility_score']:.2f}**, "
                   f"gate is {gate:.2f}, so restoration "
                   f"**{'fires' if vrow['visibility_score'] < gate else 'does not fire'}** here")
        bgr = None
        cap = cv2.VideoCapture(R["clip"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, bgr = cap.read()
        cap.release()
        if ok:
            from shoal import restore as _restore

            rcfg = run["run"]["config"]["restore"]
            enhanced = _restore.enhance(bgr, rcfg)
            a, b = st.columns(2)
            a.image(bgr[..., ::-1], caption="raw", width="stretch")
            b.image(enhanced[..., ::-1], caption="restored (grey-world WB → red comp → CLAHE-L → unsharp → gamma)",
                    width="stretch")

        st.markdown("#### Per-frame visibility index")
        fig = go.Figure()
        fig.add_scatter(x=vis["timestamp"], y=vis["visibility_score"], name="visibility",
                        mode="lines")
        fig.add_hline(y=gate, line_dash="dash", line_color="crimson",
                      annotation_text=f"restoration gate {gate:.2f}")
        below = vis[vis["visibility_score"] < gate]
        fig.add_scatter(x=below["timestamp"], y=below["visibility_score"], mode="markers",
                        marker=dict(color="crimson", size=5), name=f"{len(below)} frames restored")
        fig.update_layout(height=280, yaxis_range=[0, 1], xaxis_title="time (s)",
                          yaxis_title="visibility (0 to 1)", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        with st.expander("visibility components (raw)"):
            comp_cols = [c for c in vis.columns if c.startswith("raw_")]
            st.plotly_chart(px.line(vis, x="timestamp", y=comp_cols).update_layout(height=260),
                            width="stretch")

    st.markdown("#### Restoration A/B ablation")
    abl = run["ablation"]
    if not abl:
        st.info("ablation not run for this run (`--no-ablation`)")
    else:
        st.markdown(f"> {abl['verdict']}")
        m = st.columns(4)
        m[0].metric("detections raw → restored", f"{abl['raw_detections']} → {abl['restored_detections']}",
                    f"{abl['detection_count_delta']:+.0f}")
        m[1].metric("mean confidence", f"{abl['raw_mean_conf']:.3f} → {abl['restored_mean_conf']:.3f}",
                    f"{abl['mean_conf_delta']:+.3f}")
        if abl.get("gt_available"):
            m[2].metric("mAP@0.5", f"{abl['map50_raw']:.3f} → {abl['map50_restored']:.3f}",
                        f"{abl['map50_delta']:+.3f}")
        else:
            m[2].metric("mAP@0.5", "n/a", help="no ground truth for this clip")
        m[3].metric("enhance time / frame", f"{abl['enhance_ms_mean']:.1f} ms",
                    help="target < 8 ms at 720p (plan §4.2); this is measured, not quoted")

        bars = pd.DataFrame({
            "condition": ["raw", "restored"],
            "detections": [abl["raw_detections"], abl["restored_detections"]],
            "mean_conf": [abl["raw_mean_conf"], abl["restored_mean_conf"]],
        })
        cc = st.columns(2)
        cc[0].plotly_chart(px.bar(bars, x="condition", y="detections", color="condition",
                                  title="fish detections (sampled frames)").update_layout(height=300, showlegend=False),
                           width="stretch")
        cc[1].plotly_chart(px.bar(bars, x="condition", y="mean_conf", color="condition",
                                  title="mean detection confidence").update_layout(height=300, showlegend=False, yaxis_range=[0, 1]),
                           width="stretch")

        ex = abl.get("provenance", {}).get("ab_example", {})
        raw_p = run["dir"] / ex.get("raw", "")
        res_p = run["dir"] / ex.get("restored", "")
        if raw_p.exists() and res_p.exists():
            st.caption(f"Lowest-visibility sampled frame ({ex.get('frame')}, score {ex.get('visibility')})")
            e1, e2 = st.columns(2)
            e1.image(str(raw_p), caption="raw", width="stretch")
            e2.image(str(res_p), caption="restored", width="stretch")


# --------------------------------------------------------------------------- #
# tab 3: trajectories
# --------------------------------------------------------------------------- #
with tab3:
    cam = R["camera_motion"]
    if not cam["is_static"]:
        st.warning(f"⚠️ Camera is **non-static** ({cam['verdict']}). "
                   "Residency and flow figures below are indicative only.")
    st.caption(units_note(scale_px_per_m))

    hist = run["residency_hist"]
    if hist is not None and hist.size:
        rmeta = run["residency_meta"]
        maxn_frame = R["ecology"]["abundance"]["maxn_frame"]
        bg = clip_frame(R["clip"], max(0, maxn_frame))
        fig = go.Figure()
        if bg is not None:
            fig.add_layout_image(dict(source=_np_to_data_uri(bg), xref="x", yref="y",
                                      x=0, y=0, sizex=bg.shape[1], sizey=bg.shape[0],
                                      sizing="stretch", layer="below", opacity=0.55))
        z = hist.copy()
        z[z <= np.percentile(z[z > 0], 8) if np.any(z > 0) else 0] = np.nan
        fig.add_heatmap(z=z, colorscale="Inferno", opacity=0.8,
                        x0=0, dx=R["clip_meta"]["width"] / z.shape[1],
                        y0=0, dy=R["clip_meta"]["height"] / z.shape[0],
                        colorbar=dict(title="fish·s / bin"))
        fig.update_yaxes(autorange="reversed", scaleanchor="x", constrain="domain")
        fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="x (px)", yaxis_title="y (px)")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"time-weighted occupancy · {rmeta.get('total_dwell_s', 0):.0f} fish-seconds total")

    nodes, flows = run["activity_nodes"], run["flow_matrix"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Activity nodes** (DBSCAN on stop points)")
        if nodes.empty:
            st.caption("no activity nodes: fish did not dwell long enough to form stop clusters "
                       "(expected for a short clip of continuously-cruising fish)")
        else:
            st.dataframe(nodes, width="stretch", hide_index=True)
    with c2:
        st.markdown("**Origin → destination flows**")
        if flows.empty:
            st.caption("no zone-to-zone transitions recorded")
        else:
            st.dataframe(flows, width="stretch", hide_index=True)

    st.markdown("**Per-track summary** (the §7 `track_summary_df` contract)")
    ts = run["track_summary"].copy()
    if scale_px_per_m and not ts.empty:
        ts["path_length_m"] = ts["path_length_px"] / scale_px_per_m
        ts["mean_speed_m_s"] = ts["mean_speed_px_s"] / scale_px_per_m
    st.dataframe(ts, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# tab 4: behaviour
# --------------------------------------------------------------------------- #
with tab4:
    st.info("Phase 1 ships the **rule-based** classifier with cut-points from this clip's own "
            "kinematic percentiles. The **HMM state segmentation** (plan §9.1) and the "
            "**polarisation Φ / nearest-neighbour** collective metrics (plan §9.2) are Phase 2.")
    beh = R["behaviour"]
    thr = beh.get("thresholds", {})
    st.caption(f"cut-points ({thr.get('speed_unit', '')}): "
               f"transit ≥ {thr.get('transit_speed', 0):.2f}, "
               f"forage |turn| ≥ {thr.get('forage_turn', 0):.2f} rad, "
               f"escape speed ≥ {thr.get('escape_speed', 0):.2f}")

    tl = run["behaviour_timeline"]
    if not tl.empty:
        melt = tl.melt(id_vars=["frame", "timestamp"],
                       value_vars=["n_foraging", "n_transit", "n_escape"],
                       var_name="state", value_name="n")
        fig = px.area(melt, x="timestamp", y="n", color="state",
                      color_discrete_map={"n_foraging": "#2a9d8f", "n_transit": "#457b9d",
                                          "n_escape": "#e63946"},
                      labels={"timestamp": "time (s)", "n": "fish in state"})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    counts = beh.get("state_counts", {})
    if counts:
        st.plotly_chart(px.bar(x=list(counts.keys()), y=list(counts.values()),
                               labels={"x": "state", "y": "tracks (majority vote)"})
                        .update_layout(height=260), width="stretch")


# --------------------------------------------------------------------------- #
# tab 5: ecology
# --------------------------------------------------------------------------- #
with tab5:
    eco = run["ecology"]
    ab = eco["abundance"]
    st.markdown("#### Abundance, reported as bounds (plan §10.1)")
    c = st.columns(3)
    c[0].metric("MaxN (conservative lower bound)", ab["maxn_total"],
                help=f"max fish in any single frame; at t={ab['maxn_timestamp']}s")
    c[1].metric("Unique tracks (upper bound)", ab["unique_tracks_total"],
                help="sensitive to ID fragmentation")
    c[2].metric("MaxN : tracks", f"{ab['maxn_total']} : {ab['unique_tracks_total']}")
    st.caption(ab["interpretation"])

    st.markdown("#### Diversity")
    dv = eco["diversity"]
    if not dv["available"]:
        st.warning(f"**Diversity panel disabled (honesty guard, plan §10.4).**\n\n{dv['reason']}")
    else:
        cols = st.columns(4)
        cols[0].metric("Richness S", dv["richness"])
        for i, (key, label) in enumerate(
            [("shannon_h", "Shannon H′"), ("pielou_j", "Pielou J′"), ("simpson_1_minus_d", "Simpson 1−D")], 1
        ):
            v = dv.get(key)
            if v:
                ci = v.get("ci95")
                cols[i].metric(label, f"{v['value']:.2f}",
                               f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else None,
                               delta_color="off")
        st.caption(f"bootstrap over per-track species confidence, {dv['bootstrap_iters']} iterations "
                   f"(mean track species conf {dv['mean_species_conf']})")

    st.markdown("#### Species composition")
    st.caption("Phase 1: detector runs single-class. The regional species head (marine-detect "
               "FishInv, 15 classes) + track-level confidence voting is Phase 2 (plan §7, §13).")
    comp = pd.DataFrame([{"species": k, "MaxN": v} for k, v in ab["maxn_by_species"].items()])
    if not comp.empty:
        st.dataframe(comp, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# tabs 6 / 7: Phase 2 placeholders
# --------------------------------------------------------------------------- #
with tab6:
    st.info("**Benchmark (Phase 2).** 2×2 ablation ByteTrack vs BoT-SORT+ReID × raw vs restored, "
            "HOTA / MOTA / IDF1 on the Brackish test split with ground truth. Computed offline once "
            "and cached to JSON; rendered as a static card. See `benchmark/evaluate.py`.")
    bench = run["dir"] / "benchmark.json"
    if bench.exists():
        st.json(json.loads(bench.read_text()))

with tab7:
    st.info("**Report (Phase 2).** One-click PDF: clip metadata, visibility profile, abundance "
            "bounds, diversity with CIs, behavioural breakdown, flagged events, and reproducibility "
            "metadata (library versions + model checkpoints). The `run.json` in this run directory "
            "already contains that provenance:")
    if run["run"]:
        st.json(run["run"].get("provenance", {}))
