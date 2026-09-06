"""Module [4.3] — the restoration A/B ablation.

Do not claim restoration helps. Measure it, live (plan §4.3).

Run the detector twice on a held-out sample of N frames — once raw, once restored —
and report the deltas that come straight out of real inference:

    * detection count delta
    * mean confidence delta
    * mean inference time (so the "< 8 ms enhancement" claim is measured, not quoted)
    * mAP@0.5 delta  IF ground-truth boxes are available for the clip (Brackish)

If restoration turns out not to help on a clip, that is shown too, and the gate is
free to leave it off. A negative result on one clip and a positive on another
demonstrates methodology; only-positives demonstrates marketing.

NO HARDCODED NUMBERS. Every value returned traces to a call made in this function.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from shoal import restore, visibility
from shoal.detect import Detector
from shoal.ingest import iter_frames, probe


@dataclass
class AblationResult:
    n_frames: int
    frame_indices: list[int]
    raw_detections: int
    restored_detections: int
    raw_mean_conf: float
    restored_mean_conf: float
    raw_infer_ms: float
    restored_infer_ms: float
    enhance_ms_mean: float
    detection_count_delta: float
    mean_conf_delta: float
    gate_would_fire_frac: float          # fraction of sampled frames below the gate
    map50_raw: float | None = None
    map50_restored: float | None = None
    map50_delta: float | None = None
    gt_available: bool = False
    verdict: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sample_frame_indices(n_total: int, n_sample: int, seed: int) -> list[int]:
    if n_total <= n_sample:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_total, size=n_sample, replace=False).tolist())


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (M,4) against every box in b (N,4), xyxy."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _ap50(all_scores: list[float], all_tp: list[int], n_gt: int) -> float:
    """Single-class AP at IoU 0.5, VOC-style 101-point interpolation."""
    if n_gt == 0:
        return float("nan")
    if not all_scores:
        return 0.0
    order = np.argsort(-np.asarray(all_scores))
    tp = np.asarray(all_tp)[order]
    fp = 1 - tp
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / n_gt
    precision = tp_c / np.clip(tp_c + fp_c, 1, None)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precision[recall >= t]
        ap += (p.max() if p.size else 0.0) / 101
    return float(ap)


def _load_gt(clip_path: str | Path) -> dict[int, np.ndarray] | None:
    """Look for a sibling MOT ground-truth file: <clip>.gt.txt or gt/gt.txt.

    Format: frame,id,bb_left,bb_top,w,h,conf,... (MOTChallenge). Returns {frame: (N,4) xyxy}.
    Brackish / BrackishMOT ship exactly this; stock clips have none, and the ablation
    then simply reports count/confidence deltas without mAP.
    """
    clip_path = Path(clip_path)
    candidates = [clip_path.with_suffix(".gt.txt"), clip_path.parent / "gt" / "gt.txt",
                  clip_path.parent / f"{clip_path.stem}_gt.txt"]
    gt_file = next((c for c in candidates if c.exists()), None)
    if gt_file is None:
        return None
    by_frame: dict[int, list[list[float]]] = {}
    for line in gt_file.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        f = int(float(parts[0])) - 1  # MOT is 1-indexed
        x, y, w, h = map(float, parts[2:6])
        by_frame.setdefault(f, []).append([x, y, x + w, y + h])
    return {f: np.asarray(v, dtype=np.float32) for f, v in by_frame.items()}


def run_ablation(detector: Detector, clip_path: str | Path, cfg: dict,
                 calibration: visibility.VisibilityCalibration, log: logging.Logger,
                 n_sample: int = 60, seed: int = 20260906,
                 example_dir: str | Path | None = None) -> AblationResult:
    meta = probe(clip_path)
    n_total = meta.n_frames if meta.n_frames > 0 else 10_000
    want = set(_sample_frame_indices(n_total, n_sample, seed))
    gt = _load_gt(clip_path)
    restore_cfg = cfg["restore"]
    dc_patch = int(cfg.get("visibility", {}).get("dark_channel_patch", 15))

    raw_dets = res_dets = 0
    raw_confs: list[float] = []
    res_confs: list[float] = []
    raw_ms: list[float] = []
    res_ms: list[float] = []
    enh_ms: list[float] = []
    below_gate = 0
    seen = 0
    example: dict[str, Any] | None = None   # lowest-visibility sampled frame, raw + restored

    # per-frame data for mAP
    raw_scores: list[float] = []
    raw_tp: list[int] = []
    res_scores: list[float] = []
    res_tp: list[int] = []
    n_gt_total = 0

    for frame in iter_frames(clip_path):
        if frame.index not in want:
            continue
        seen += 1
        bgr = frame.bgr

        vscore = visibility.score_frame(bgr, calibration, dc_patch)["score"]
        if vscore < float(restore_cfg.get("gate_threshold", 0.55)):
            below_gate += 1

        t0 = time.perf_counter()
        raw_boxes = detector.predict_frame(bgr)
        raw_ms.append((time.perf_counter() - t0) * 1e3)
        raw_dets += len(raw_boxes)
        raw_confs.extend(raw_boxes[:, 4].tolist())

        te = time.perf_counter()
        restored = restore.enhance(bgr, restore_cfg)
        enh_ms.append((time.perf_counter() - te) * 1e3)

        if example is None or vscore < example["visibility"]:
            example = {"frame": frame.index, "visibility": vscore,
                       "raw": bgr.copy(), "restored": restored.copy()}
        t1 = time.perf_counter()
        res_boxes = detector.predict_frame(restored)
        res_ms.append((time.perf_counter() - t1) * 1e3)
        res_dets += len(res_boxes)
        res_confs.extend(res_boxes[:, 4].tolist())

        if gt is not None and frame.index in gt:
            g = gt[frame.index]
            n_gt_total += len(g)
            for boxes, scores, tps in ((raw_boxes, raw_scores, raw_tp),
                                       (res_boxes, res_scores, res_tp)):
                if len(boxes) == 0:
                    continue
                iou = _iou(boxes[:, :4], g)
                matched = set()
                order = np.argsort(-boxes[:, 4])
                for i in order:
                    scores.append(float(boxes[i, 4]))
                    j = int(np.argmax(iou[i])) if iou.shape[1] else -1
                    if j >= 0 and iou[i, j] >= 0.5 and j not in matched:
                        matched.add(j)
                        tps.append(1)
                    else:
                        tps.append(0)

        if frame.index >= max(want):
            break

    raw_mc = float(np.mean(raw_confs)) if raw_confs else 0.0
    res_mc = float(np.mean(res_confs)) if res_confs else 0.0
    map_raw = map_res = map_delta = None
    if gt is not None and n_gt_total > 0:
        map_raw = _ap50(raw_scores, raw_tp, n_gt_total)
        map_res = _ap50(res_scores, res_tp, n_gt_total)
        map_delta = map_res - map_raw

    count_delta = res_dets - raw_dets
    conf_delta = res_mc - raw_mc

    prefix = (f"On {seen} sampled frames ({below_gate} below the visibility gate), "
              f"restoration changed ")
    if map_delta is not None:
        # Ground truth present: mAP@0.5 is the real number and it decides.
        headline = f"mAP@0.5 {map_delta:+.3f} ({map_raw:.3f} -> {map_res:.3f}), " \
                   f"detections {count_delta:+.0f}, mean confidence {conf_delta:+.3f}"
        if map_delta > 0.005:
            tail = "Restoration helps on this clip (mAP up); the gate keeps it on."
        elif map_delta < -0.005:
            tail = "Restoration hurts on this clip (mAP down); the gate is right to leave it off."
        else:
            tail = "Restoration is neutral on this clip (mAP unchanged within noise)."
    else:
        # No ground truth: report both deltas, and only claim a direction when they agree.
        headline = f"detections {count_delta:+.0f}, mean confidence {conf_delta:+.3f} " \
                   f"(no ground truth on this clip, so no mAP)"
        if count_delta > 0 and conf_delta > 0:
            tail = "Both signals point the same way: restoration helps here."
        elif count_delta < 0 and conf_delta < 0:
            tail = "Both signals point the same way: restoration hurts here."
        else:
            tail = ("Signals disagree (more detections but lower mean confidence, or vice "
                    "versa) — inspect the raw/restored A/B frames before trusting either "
                    "direction. This is exactly why the gate is measured, not assumed.")
    verdict = prefix + headline + ". " + tail

    example_paths: dict[str, str] = {}
    if example is not None and example_dir is not None:
        example_dir = Path(example_dir)
        example_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(example_dir / "ab_example_raw.png"), example["raw"])
        cv2.imwrite(str(example_dir / "ab_example_restored.png"), example["restored"])
        example_paths = {
            "raw": "ab_example_raw.png", "restored": "ab_example_restored.png",
            "frame": int(example["frame"]), "visibility": round(float(example["visibility"]), 3),
        }

    log.info("ablation: %s", verdict)
    return AblationResult(
        n_frames=seen, frame_indices=sorted(want),
        raw_detections=raw_dets, restored_detections=res_dets,
        raw_mean_conf=raw_mc, restored_mean_conf=res_mc,
        raw_infer_ms=float(np.mean(raw_ms)) if raw_ms else 0.0,
        restored_infer_ms=float(np.mean(res_ms)) if res_ms else 0.0,
        enhance_ms_mean=float(np.mean(enh_ms)) if enh_ms else 0.0,
        detection_count_delta=float(count_delta), mean_conf_delta=float(conf_delta),
        gate_would_fire_frac=below_gate / seen if seen else 0.0,
        map50_raw=map_raw, map50_restored=map_res, map50_delta=map_delta,
        gt_available=gt is not None and n_gt_total > 0, verdict=verdict,
        provenance={"detector": detector.name, "n_gt_boxes": n_gt_total,
                    "gate_threshold": float(restore_cfg.get("gate_threshold", 0.55)),
                    "enhance_target_ms": 8.0, "ab_example": example_paths},
    )
