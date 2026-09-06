"""Module [5] — TRACKING.

Ultralytics' built-in tracker (``model.track(persist=True, tracker=...)``), with
BoT-SORT + Re-ID as the Phase 1 primary and the underwater tuning in
``config/botsort_underwater.yaml`` (plan §6). ByteTrack is the Phase 2 benchmark
comparison arm and is loadable through the same entry point.

The narrative this stage completes: turbidity depresses detection confidence, and
low-confidence detections are exactly what the tracker's second association stage
recovers. So ``detect.conf`` is deliberately low and ``track_low_thresh`` is
deliberately low — they are tuned together, not independently.

Output of ``run_tracking`` is the RAW per-frame track table (pre-repair):

    track_id | frame | timestamp | x | y | w | h | conf
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from shoal.config import REPO_ROOT
from shoal.detect import Detector
from shoal.ingest import ClipMeta

_ULTRALYTICS_KNOWN_KEYS = {
    "tracker_type", "track_high_thresh", "track_low_thresh", "new_track_thresh",
    "track_buffer", "match_thresh", "fuse_score", "gmc_method", "proximity_thresh",
    "appearance_thresh", "with_reid", "model",
}


def _prepare_tracker_yaml(path: str | Path, method: str, log: logging.Logger) -> str:
    """Validate our tracker YAML against the keys Ultralytics understands, warn on
    anything unrecognised (a version bump surfaces loudly), and force the tracker
    type to match ``method``. Returns a path Ultralytics can consume."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    unknown = set(cfg) - _ULTRALYTICS_KNOWN_KEYS
    if unknown:
        log.warning("tracker config %s has keys Ultralytics may ignore: %s", path.name, sorted(unknown))
    want_type = "bytetrack" if method == "bytetrack" else "botsort"
    if cfg.get("tracker_type") != want_type:
        log.info("overriding tracker_type %s -> %s to match method=%s",
                 cfg.get("tracker_type"), want_type, method)
        cfg["tracker_type"] = want_type
    if want_type == "bytetrack":
        # ByteTrack ignores the BoT-SORT-only appearance keys; strip them to avoid noise.
        for k in ("gmc_method", "proximity_thresh", "appearance_thresh", "with_reid", "model"):
            cfg.pop(k, None)
    resolved = REPO_ROOT / "outputs" / "_tracker_resolved.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(resolved)


def run_tracking(
    detector: Detector,
    clip_path: str | Path,
    meta: ClipMeta,
    cfg: dict,
    log: logging.Logger,
    frame_stride: int = 1,
    max_frames: int | None = None,
    restore_cfg: dict | None = None,
    visibility_calibration=None,
    visibility_scores: dict[int, float] | None = None,
    on_progress=None,
) -> tuple[pd.DataFrame, dict]:
    """Track fish through the clip.

    Restoration is applied per-frame *before* detection when a calibration is
    supplied and the frame's visibility score falls below the gate — the tracker
    then runs on exactly the frames the detector sees. Returns the raw track table
    and a provenance dict (tracker file, method, frame counts, restored-frame count).
    """
    method = cfg.get("method", "botsort")
    tracker_yaml = _prepare_tracker_yaml(cfg["tracker_config"], method, log)
    detector.install_single_class_hook(log)

    prov: dict = {
        "tracker_method": method,
        "tracker_config_file": str(cfg["tracker_config"]),
        "tracker_config_resolved": tracker_yaml,
        "frames_seen": 0,
        "frames_restored": 0,
        "detections_total": 0,
    }

    gate_active = visibility_calibration is not None and restore_cfg is not None
    if gate_active:
        from shoal import restore, visibility

    rows: list[dict] = []
    from shoal.ingest import iter_frames

    for frame in iter_frames(clip_path, frame_stride=frame_stride, max_frames=max_frames):
        bgr = frame.bgr
        if gate_active:
            if visibility_scores is not None and frame.index in visibility_scores:
                vscore = visibility_scores[frame.index]
            else:
                vscore = visibility.score_frame(bgr, visibility_calibration,
                                                int(cfg.get("dark_channel_patch", 15)))["score"]
            rr = restore.restore_if_needed(bgr, vscore, restore_cfg)
            bgr = rr.frame
            if rr.restored:
                prov["frames_restored"] += 1

        result = detector.model.track(
            bgr, persist=True, tracker=tracker_yaml, conf=detector.conf, iou=detector.iou,
            imgsz=detector.imgsz, max_det=detector.max_det, classes=detector.fish_class_ids,
            device=detector.device, verbose=False,
        )[0]
        prov["frames_seen"] += 1

        boxes = result.boxes
        if boxes is not None and boxes.id is not None and len(boxes) > 0:
            xywh = boxes.xywh.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for (cx, cy, w, h), tid, c in zip(xywh, ids, confs):
                rows.append({
                    "track_id": int(tid), "frame": int(frame.index),
                    "timestamp": float(frame.timestamp), "x": float(cx), "y": float(cy),
                    "w": float(w), "h": float(h), "conf": float(c),
                })
            prov["detections_total"] += len(ids)

        if on_progress is not None:
            on_progress(frame.index, meta.n_frames)

    raw = pd.DataFrame(rows, columns=["track_id", "frame", "timestamp", "x", "y", "w", "h", "conf"])
    n_ids = raw["track_id"].nunique() if not raw.empty else 0
    log.info("tracking done: %d frames, %d detections, %d raw track ids (method=%s)",
             prov["frames_seen"], prov["detections_total"], n_ids, method)
    prov["raw_track_ids"] = int(n_ids)
    return raw, prov
