"""Pipeline orchestrator.

Not one of the numbered plan modules — this is the glue that runs [1]..[11] in order
and writes a self-contained ``outputs/<run_id>/`` directory the dashboard reads.
Flagged as an addition beyond SHOAL_build_plan.md §14 in docs/PHASE1_STATUS.md.

Stages (Phase 1 subset of plan §2):

    [1]  ingest + camera-motion guard
    [2]  visibility trace (every frame)
    [3]  adaptive restoration (gated, applied inside tracking)
    [4]  detection (fallback chain)
    [5]  BoT-SORT tracking
    [6]  track repair -> tracks_df, track_summary_df (§7 contract)
    [8]  MovingPandas trajectory analytics + residency heatmap
    [9]  behaviour: rule-based classifier (HMM is Phase 2)
    [10] ecology: MaxN + diversity (honesty guard) + bootstrap CI
    [11] exports (csv / geojson / MOT) + annotated video
    [4.3] restoration A/B ablation

Nothing is hardcoded: every number written traces to a computation here.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from shoal import annotate, behaviour, ecology, export, figures, trajectory
from shoal.config import load_config, resolve_device
from shoal.detect import build_detector
from shoal.ingest import assess_camera_motion, iter_frames, probe
from shoal.repair import repair_tracks
from shoal.runlog import basic_logging, new_run
from shoal.track import run_tracking
from shoal.visibility import VisibilityCalibration, VisibilityTrace, score_frame


def _visibility_trace(clip_path, cfg, calibration, log, max_frames, frame_stride) -> tuple[VisibilityTrace, set[int]]:
    from shoal.visibility import _COMPONENTS

    patch = int(cfg.get("visibility.dark_channel_patch", 15))
    gate = float(cfg.get("restore.gate_threshold", 0.55))
    frames, ts, scores = [], [], []
    raw = {c: [] for c in _COMPONENTS}
    norm = {c: [] for c in _COMPONENTS}
    restored: set[int] = set()
    t0 = time.perf_counter()
    for fr in iter_frames(clip_path, frame_stride=frame_stride, max_frames=max_frames):
        s = score_frame(fr.bgr, calibration, patch)
        frames.append(fr.index)
        ts.append(fr.timestamp)
        scores.append(s["score"])
        for c in _COMPONENTS:
            raw[c].append(s["components_raw"][c])
            norm[c].append(s["components_norm"][c])
        if s["score"] < gate:
            restored.add(fr.index)
    dt = time.perf_counter() - t0
    log.info("visibility: %d frames scored in %.1fs (%.1f ms/frame); %d below gate %.2f; provisional=%s",
             len(frames), dt, 1e3 * dt / max(len(frames), 1), len(restored), gate,
             not calibration.calibrated)
    trace = VisibilityTrace(frames, ts, scores, raw, norm,
                            provisional=not calibration.calibrated, method=calibration.method)
    return trace, restored


def run(clip_path: str | Path, config_path: str | Path | None = None,
        overrides: list[str] | None = None, device: str | None = None,
        max_frames: int | None = None, frame_stride: int | None = None,
        run_ablation: bool = True, run_id: str | None = None) -> Path:
    clip_path = Path(clip_path)
    if not clip_path.exists():
        raise FileNotFoundError(clip_path)

    inline: dict = {"run": {}}
    if device:
        inline["run"]["device"] = device
    if max_frames is not None:
        inline["run"]["max_frames"] = max_frames
    if frame_stride is not None:
        inline["run"]["frame_stride"] = frame_stride
    cfg = load_config(config_path, overrides, inline)

    dev = resolve_device(cfg.get("run.device", "auto"))
    ctx = new_run(clip_path, dev, cfg.as_dict(), run_id=run_id)
    log = ctx.log
    max_frames = cfg.get("run.max_frames")
    frame_stride = int(cfg.get("run.frame_stride", 1))
    seed = int(cfg.get("run.seed", 20260906))

    meta = probe(clip_path)
    ctx.note("clip_meta", {"fps": meta.fps, "size": [meta.width, meta.height],
                           "n_frames": meta.n_frames, "duration_s": round(meta.duration_s, 2),
                           "fourcc": meta.fourcc})
    log.info("clip: %dx%d  %.2f fps  %d frames  %.1fs  (%s)",
             meta.width, meta.height, meta.fps, meta.n_frames, meta.duration_s, meta.fourcc)

    # -- [1] camera-motion guard -------------------------------------------
    cam = assess_camera_motion(clip_path, cfg.get("ingest.camera_motion_guard", {}))
    ctx.note("camera_motion", cam.as_dict())
    (ctx.path("camera_motion.json")).write_text(json.dumps(cam.as_dict(), indent=2))
    log.info("camera-motion guard: %s", cam.verdict)

    # -- [2] visibility trace --------------------------------------------
    calib = VisibilityCalibration.load(cfg.get("visibility.calibration_file"))
    if not calib.calibrated:
        log.warning("visibility calibration is PROVISIONAL — run `uv run shoal-calibrate`. "
                    "Every visibility score this run is tagged provisional=true.")
    trace, restored_frames = _visibility_trace(clip_path, cfg, calib, log, max_frames, frame_stride)
    vis_df = trace.to_frame()
    vis_df.to_csv(ctx.path("visibility.csv"), index=False)
    ctx.note("visibility_summary", trace.summary())

    # -- [4]+[5] detection + tracking (restoration [3] applied inside) -----
    detector = build_detector(cfg["detect"], dev, log)
    ctx.note("detector", {"name": detector.name, "kind": detector.kind, "source": detector.source,
                          "device": detector.device, "conf": detector.conf,
                          "fish_class_ids": detector.fish_class_ids,
                          "underlying_classes": detector.class_names})

    prog = _progress(log, "track")
    raw_tracks, track_prov = run_tracking(
        detector, clip_path, meta, {**cfg["track"], "dark_channel_patch": cfg.get("visibility.dark_channel_patch", 15)},
        log, frame_stride=frame_stride, max_frames=max_frames,
        restore_cfg=cfg["restore"], visibility_calibration=calib,
        visibility_scores=dict(zip(trace.frames, trace.scores)), on_progress=prog,
    )
    ctx.note("tracking", track_prov)
    raw_tracks.to_csv(ctx.path("tracks_raw.csv"), index=False)

    # -- [6] track repair ------------------------------------------------
    tracks_df, summary_df, repair_prov = repair_tracks(raw_tracks, meta.fps, cfg["repair"], log)
    ctx.note("repair", repair_prov)

    # -- [8] trajectory analytics --------------------------------------
    traj = trajectory.analyse(tracks_df, summary_df, meta.width, meta.height, meta.fps,
                              cfg["trajectory"], log)
    ctx.note("trajectory", traj.provenance)
    traj.stop_points.to_csv(ctx.path("stops.csv"), index=False)
    traj.activity_nodes.to_csv(ctx.path("activity_nodes.csv"), index=False)
    traj.flow_matrix.to_csv(ctx.path("flow_matrix.csv"), index=False)
    np.save(ctx.path("residency_hist.npy"), np.asarray(traj.residency["histogram"]))
    (ctx.path("residency_meta.json")).write_text(json.dumps(
        {k: v for k, v in traj.residency.items() if k != "histogram"}, indent=2, default=str))

    # -- [9] behaviour (rule-based) ---------------------------------
    beh = behaviour.classify(tracks_df, summary_df, cfg["behaviour"], log)
    tracks_df, summary_df = beh.tracks_df, beh.summary_df
    beh.state_timeline.to_csv(ctx.path("behaviour_timeline.csv"), index=False)
    ctx.note("behaviour", {**beh.provenance, "thresholds": beh.thresholds})

    # -- [10] ecology ------------------------------------------------
    species_head_available = detector.kind == "finetuned" or any(
        s not in ("fish", "", None) for s in summary_df["species"].unique()
    )
    eco = ecology.analyse(tracks_df, summary_df, cfg["ecology"], species_head_available, log, seed=seed)
    ctx.note("ecology", eco.provenance)
    (ctx.path("ecology.json")).write_text(json.dumps(
        {"abundance": eco.abundance.as_dict(), "diversity": eco.diversity.as_dict(),
         "provenance": eco.provenance}, indent=2))
    eco.abundance.per_frame_counts.to_csv(ctx.path("abundance_per_frame.csv"), index=False)

    # -- [11] exports + annotated video ----------------------------
    written = export.run_exports(tracks_df, summary_df, ctx.out_dir, meta.height,
                                 cfg.get("export.formats", ["csv", "geojson", "mot"]), log)
    ctx.note("exports", written)

    annotate.render(clip_path, tracks_df, ctx.path("annotated.mp4"), log,
                    visibility_df=vis_df, restored_frames=restored_frames,
                    max_frames=max_frames)

    bg = _representative_frame(clip_path, eco.abundance.maxn_frame)
    figures.residency_png(traj.residency, bg, ctx.path("residency.png"))
    figures.visibility_png(vis_df, float(cfg.get("restore.gate_threshold", 0.55)),
                           ctx.path("visibility.png"))

    # -- [4.3] restoration ablation -------------------------------
    ablation_dict = None
    if run_ablation:
        from shoal.ablation import run_ablation as _abl

        abl = _abl(detector, clip_path, cfg.as_dict(), calib, log,
                   n_sample=int(cfg.get("ablation.n_sample", 60)), seed=seed,
                   example_dir=ctx.out_dir)
        ablation_dict = abl.as_dict()
        (ctx.path("ablation.json")).write_text(json.dumps(ablation_dict, indent=2))
        figures.ablation_png(ablation_dict, ctx.path("ablation.png"))
        ctx.note("ablation", {"verdict": abl.verdict, "map50_delta": abl.map50_delta,
                              "enhance_ms_mean": round(abl.enhance_ms_mean, 2)})

    # -- results.json (dashboard entry point) --------------------
    results = {
        "run_id": ctx.run_id,
        "clip": str(clip_path),
        "clip_meta": {"fps": meta.fps, "width": meta.width, "height": meta.height,
                      "n_frames": meta.n_frames, "duration_s": meta.duration_s},
        "device": dev,
        "camera_motion": cam.as_dict(),
        "visibility": trace.summary(),
        "tracking": track_prov,
        "repair": repair_prov,
        "trajectory": traj.provenance,
        "behaviour": {**beh.provenance, "thresholds": beh.thresholds},
        "ecology": {"abundance": eco.abundance.as_dict(), "diversity": eco.diversity.as_dict()},
        "ablation": ablation_dict,
        "artifacts": {
            "annotated_video": "annotated.mp4",
            "residency_png": "residency.png",
            "visibility_png": "visibility.png",
            "ablation_png": "ablation.png" if ablation_dict else None,
            "tracks_csv": "tracks.csv",
            "track_summary_csv": "track_summary.csv",
            "trajectories_geojson": "trajectories.geojson",
            "results_mot": "results_MOT.txt",
        },
        "n_tracks_accepted": repair_prov.get("n_tracks_accepted", 0),
        "phase": 1,
    }
    (ctx.path("results.json")).write_text(json.dumps(results, indent=2, default=str))
    ctx.write_manifest()
    log.info("DONE. run directory: %s", ctx.out_dir)
    _print_summary(results)
    return ctx.out_dir


def _progress(log: logging.Logger, tag: str):
    state = {"last": -1}

    def cb(idx: int, total: int):
        if total <= 0:
            return
        pct = int(100 * idx / total)
        if pct >= state["last"] + 10:
            state["last"] = pct
            log.info("%s: %d%% (frame %d/%d)", tag, pct, idx, total)

    return cb


def _representative_frame(clip_path, frame_idx: int) -> np.ndarray | None:
    try:
        target = max(0, int(frame_idx))
        for fr in iter_frames(clip_path):
            if fr.index >= target:
                return fr.bgr
    except Exception:
        return None
    return None


def _print_summary(r: dict) -> None:
    a = r["ecology"]["abundance"]
    d = r["ecology"]["diversity"]
    print("\n" + "=" * 68)
    print(f"  SHOAL run {r['run_id']}   ({r['device']})")
    print("=" * 68)
    print(f"  clip            {Path(r['clip']).name}  "
          f"({r['clip_meta']['duration_s']:.1f}s, {r['clip_meta']['n_frames']} frames)")
    print(f"  camera          {'STATIC' if r['camera_motion']['is_static'] else 'NON-STATIC'}")
    print(f"  visibility      mean {r['visibility']['score_mean']:.2f}"
          f"  (provisional={r['visibility']['provisional']})")
    print(f"  tracks          {r['n_tracks_accepted']} accepted "
          f"(+{r['repair'].get('n_tracks_rejected_short', 0)} rejected as short)")
    print(f"  abundance       MaxN = {a['maxn_total']}  |  unique tracks = {a['unique_tracks_total']}"
          f"   (bounds)")
    if d["available"]:
        h = d["shannon_h"]
        print(f"  diversity       H' = {h['value']} {h.get('ci95', '')}   S = {d['richness']}")
    else:
        print(f"  diversity       [honesty guard] {d['reason'][:60]}...")
    if r["ablation"]:
        print(f"  restoration     {r['ablation']['verdict'][:80]}")
    print("=" * 68 + "\n")


def main(argv: list[str] | None = None) -> int:
    basic_logging()
    ap = argparse.ArgumentParser(description="Run the SHOAL pipeline on a clip.")
    ap.add_argument("--clip", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--override", action="append", default=[], help="extra YAML config to deep-merge")
    ap.add_argument("--device", default=None, help="auto | cuda | cpu | cuda:0")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--frame-stride", type=int, default=None)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    out = run(args.clip, args.config, args.override, args.device, args.max_frames,
              args.frame_stride, run_ablation=not args.no_ablation, run_id=args.run_id)
    print(f"outputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
