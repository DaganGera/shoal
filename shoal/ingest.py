"""Module [1] — INGEST: frame iterator, clip metadata, camera-motion guard.

Two jobs:

1. Give every downstream stage the same view of the clip: frames indexed by their
   ORIGINAL frame number and stamped with their REAL timestamp, so a ``frame_stride``
   for fast iteration never corrupts a speed (which is metres — or pixels — per
   *second*, not per *processed frame*).

2. Measure camera motion. The residency map and the origin/destination flow matrix
   are only physically meaningful for a fixed camera (plan §3.1 — the Brackish set
   matters *because* it is a fixed seabed station). Rather than assume the rig is
   static, we estimate global motion from tracked corner features and hand the
   pipeline a verdict + magnitude. The dashboard prints that verdict next to every
   spatial figure. A measured limitation shown is a strength; an unmeasured one hidden
   is the thing a juror catches.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ClipMeta:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int          # as reported by the container (may be approximate for some codecs)
    duration_s: float
    fourcc: str

    @property
    def diagonal_px(self) -> float:
        return float(np.hypot(self.width, self.height))


@dataclass(frozen=True)
class Frame:
    index: int             # original frame number in the source clip
    timestamp: float       # seconds from clip start
    bgr: np.ndarray        # H x W x 3, uint8


@dataclass
class CameraMotionReport:
    enabled: bool
    is_static: bool
    n_samples: int
    per_frame_translation_frac: float   # median inter-sample shift / frame diagonal
    cumulative_drift_frac: float        # net displacement of the scene / frame diagonal
    max_rotation_deg: float
    verdict: str                        # human-readable, shown in the dashboard

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "is_static": self.is_static,
            "n_samples": self.n_samples,
            "per_frame_translation_frac": round(self.per_frame_translation_frac, 5),
            "cumulative_drift_frac": round(self.cumulative_drift_frac, 5),
            "max_rotation_deg": round(self.max_rotation_deg, 3),
            "verdict": self.verdict,
        }


def probe(path: str | Path) -> ClipMeta:
    """Read container metadata without decoding the whole clip."""
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((raw_fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
    cap.release()
    if fps <= 1e-3:
        fps = 30.0  # last-resort default; logged by the caller via ClipMeta
    duration_s = n_frames / fps if n_frames > 0 else 0.0
    return ClipMeta(path, fps, width, height, n_frames, duration_s, fourcc or "unknown")


def iter_frames(
    path: str | Path,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[Frame]:
    """Yield frames at the given stride, each carrying its real timestamp.

    ``max_frames`` caps the number of *yielded* frames (post-stride), which is what
    "process the first 300 frames" means when iterating quickly.
    """
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(frame_stride))
    idx = 0
    yielded = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield Frame(index=idx, timestamp=idx / fps, bgr=bgr)
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# Camera-motion guard
# --------------------------------------------------------------------------- #
_LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def assess_camera_motion(path: str | Path, cfg: dict) -> CameraMotionReport:
    """Estimate global (ego) motion from tracked Shi-Tomasi corners.

    For each sampled consecutive pair we fit a partial-affine transform (translation
    + rotation + uniform scale) with RANSAC — fish are moving foreground, and RANSAC
    treats them as outliers against the static background, which is the majority of
    the pixels in a usable monitoring clip.

    Two numbers come out:
      * per-frame jitter — the median inter-sample translation (fraction of the
        frame diagonal)
      * cumulative drift — the CHAINED transform composed frame to frame, tracking
        where the frame centre ends up. Chaining (not summing translation
        magnitudes) means a wobble that returns to place nets to zero, while a
        genuine slow pan accumulates.
    """
    if not cfg.get("enabled", True):
        return CameraMotionReport(False, True, 0, 0.0, 0.0, 0.0, "guard disabled")

    max_features = int(cfg.get("max_features", 300))
    sample_stride = max(1, int(cfg.get("sample_stride", 3)))
    t_warn = float(cfg.get("translation_frac_warn", 0.006))
    d_warn = float(cfg.get("drift_frac_warn", 0.06))

    meta = probe(path)
    diag = meta.diagonal_px or 1.0
    centre = np.array([[meta.width / 2.0], [meta.height / 2.0], [1.0]])

    cap = cv2.VideoCapture(str(path))
    prev_gray = None
    prev_pts = None
    per_frame_shifts: list[float] = []
    rotations: list[float] = []
    cumulative = np.eye(3)          # chained frame-0 -> frame-i transform
    max_drift = 0.0
    idx = 0
    n_samples = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % sample_stride != 0:
                idx += 1
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None and prev_pts is not None and len(prev_pts) >= 8:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **_LK_PARAMS)
                good_prev = prev_pts[status.ravel() == 1]
                good_nxt = nxt[status.ravel() == 1]
                if len(good_prev) >= 8:
                    m, _ = cv2.estimateAffinePartial2D(
                        good_prev, good_nxt, method=cv2.RANSAC, ransacReprojThreshold=3
                    )
                    if m is not None:
                        per_frame_shifts.append(np.hypot(m[0, 2], m[1, 2]) / diag)
                        rotations.append(abs(float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))))
                        cumulative = np.vstack([m, [0, 0, 1]]) @ cumulative
                        moved = cumulative @ centre
                        max_drift = max(max_drift, float(
                            np.hypot(moved[0, 0] - centre[0, 0], moved[1, 0] - centre[1, 0]) / diag))
                        n_samples += 1
            prev_gray = gray
            prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=max_features, qualityLevel=0.01, minDistance=12, blockSize=7
            )
            idx += 1
    finally:
        cap.release()

    if n_samples == 0:
        return CameraMotionReport(
            True, True, 0, 0.0, 0.0, 0.0,
            "insufficient texture to assess camera motion — treated as static",
        )

    per_frame = float(np.median(per_frame_shifts))
    # The chained estimate accumulates estimation error on long, feature-poor clips;
    # past ~1 diagonal it only tells us "well beyond static", so clamp the reported
    # figure and say so rather than printing an implausible 384%.
    drift_raw = float(max_drift)
    drift = min(drift_raw, 1.2)
    drift_note = " (clamped; chained estimate saturates on long clips)" if drift_raw > 1.2 else ""
    max_rot = float(np.max(rotations)) if rotations else 0.0
    is_static = per_frame <= t_warn and drift <= d_warn

    if is_static:
        verdict = (
            f"static rig: per-frame scene shift {per_frame*100:.2f}% of the diagonal, "
            f"cumulative drift {drift*100:.1f}% — spatial analytics valid in frame coordinates"
        )
    else:
        verdict = (
            f"NON-STATIC camera: per-frame scene shift {per_frame*100:.2f}% "
            f"(warn > {t_warn*100:.2f}%), cumulative drift {drift*100:.0f}%{drift_note} "
            f"(warn > {d_warn*100:.1f}%). Residency map and flow matrix are read in IMAGE "
            f"coordinates, not fixed world coordinates; BoT-SORT GMC still compensates the tracks."
        )
    return CameraMotionReport(True, is_static, n_samples, per_frame, drift, max_rot, verdict)
