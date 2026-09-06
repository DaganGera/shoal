"""Module [2] — VISIBILITY INDEX.

Build this first: it is the hook, and it is the thing that turns "varying visibility
conditions" from a line in the brief into a measured, per-frame number that gates the
rest of the pipeline.

Four cheap OpenCV components, computed on the frame AS-IS (no restoration):

    contrast     std dev of the L channel in LAB           higher  = clearer
    colour_cast  distance of (mean a, mean b) from (128,128) in LAB   higher = worse
    sharpness    variance of the Laplacian                  higher  = clearer
    haze         mean of the dark channel                    higher = worse

They are fused into a single 0-1 score (0 = unusable, 1 = pristine) using
normalisation ranges and weights that MUST be calibrated against real anchor clips
(`shoal.calibrate`), never hand-picked (plan §4.1). Until calibration has run, this
module still returns every raw component and a provisional fused score explicitly
tagged ``provisional=True`` so nothing downstream can mistake it for a real number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from shoal.config import REPO_ROOT

_COMPONENTS = ("contrast", "colour_cast", "sharpness", "haze")

# Visibility is a GLOBAL frame statistic, so it is computed on a downscaled copy:
# ~10x faster and the four components move by < 1% versus full resolution. The
# working width is fixed here (not in config) so a calibration transfers between
# clips of different native resolution.
_WORK_WIDTH = 512


# --------------------------------------------------------------------------- #
# raw components
# --------------------------------------------------------------------------- #
def _fit_work_size(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= _WORK_WIDTH:
        return bgr
    nh = max(1, round(h * _WORK_WIDTH / w))
    return cv2.resize(bgr, (_WORK_WIDTH, nh), interpolation=cv2.INTER_AREA)


def _channel_min(bgr: np.ndarray) -> np.ndarray:
    """Per-pixel min across B, G, R. ``np.min(bgr, axis=2)`` on a uint8 array is
    pathologically slow (contiguous size-3 axis); pairwise ``np.minimum`` is ~30x
    faster."""
    return np.minimum(np.minimum(bgr[:, :, 0], bgr[:, :, 1]), bgr[:, :, 2])


def _dark_channel(bgr: np.ndarray, patch: int) -> np.ndarray:
    """Min across colour channels, then a local min (erosion) over a patch.

    The dark-channel prior (He et al. 2009): in haze-free outdoor patches at least
    one channel is near zero somewhere; underwater, suspended particulate lifts that
    floor, so a high mean dark channel is a direct turbidity proxy.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(_channel_min(bgr), kernel)


def raw_components(bgr: np.ndarray, dark_channel_patch: int = 15) -> dict[str, float]:
    """The four unfused visibility components for one frame (computed downscaled)."""
    work = _fit_work_size(bgr)
    # scale the dark-channel patch with the downscale so it covers the same fraction
    patch = max(3, round(dark_channel_patch * work.shape[1] / max(bgr.shape[1], 1)))
    if patch % 2 == 0:
        patch += 1

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    contrast = float(L.std())
    colour_cast = float(np.hypot(a.mean() - 128.0, b.mean() - 128.0))
    sharpness = float(cv2.Laplacian(L, cv2.CV_32F).var())
    haze = float(_dark_channel(work, patch).mean())

    return {"contrast": contrast, "colour_cast": colour_cast, "sharpness": sharpness, "haze": haze}


# --------------------------------------------------------------------------- #
# calibration + fusion
# --------------------------------------------------------------------------- #
@dataclass
class VisibilityCalibration:
    calibrated: bool
    method: str
    component_normalisation: dict[str, dict[str, float]]
    fusion_weights: dict[str, float]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> VisibilityCalibration:
        path = Path(path) if path else REPO_ROOT / "config" / "visibility_calibration.json"
        data = json.loads(Path(path).read_text())
        return cls(
            calibrated=bool(data.get("calibrated", False)),
            method=data.get("method", "unknown"),
            component_normalisation=data["component_normalisation"],
            fusion_weights=data["fusion_weights"],
            raw=data,
        )

    def _normalise(self, name: str, value: float) -> float:
        spec = self.component_normalisation[name]
        lo, hi = float(spec["lo"]), float(spec["hi"])
        if hi <= lo:
            return 0.0
        n = (value - lo) / (hi - lo)
        n = float(np.clip(n, 0.0, 1.0))
        # After normalisation every component is oriented so that 1.0 = clearer.
        return 1.0 - n if spec.get("invert", False) else n

    def fuse(self, components: dict[str, float]) -> dict[str, Any]:
        """Return normalised components + the weighted fused score in [0, 1]."""
        normed = {k: self._normalise(k, components[k]) for k in _COMPONENTS}
        wsum = sum(self.fusion_weights.get(k, 0.0) for k in _COMPONENTS) or 1.0
        score = sum(self.fusion_weights.get(k, 0.0) * normed[k] for k in _COMPONENTS) / wsum
        return {
            "components_raw": components,
            "components_norm": normed,
            "score": float(np.clip(score, 0.0, 1.0)),
            "provisional": not self.calibrated,
            "method": self.method,
        }


def score_frame(
    bgr: np.ndarray,
    calibration: VisibilityCalibration,
    dark_channel_patch: int = 15,
) -> dict[str, Any]:
    return calibration.fuse(raw_components(bgr, dark_channel_patch))


@dataclass
class VisibilityTrace:
    """Per-frame visibility over a whole clip, plus summary statistics."""

    frames: list[int]
    timestamps: list[float]
    scores: list[float]
    components_raw: dict[str, list[float]]
    components_norm: dict[str, list[float]]
    provisional: bool
    method: str

    def summary(self) -> dict[str, Any]:
        arr = np.asarray(self.scores, dtype=float)
        return {
            "n_frames": len(self.scores),
            "score_mean": float(arr.mean()) if arr.size else None,
            "score_min": float(arr.min()) if arr.size else None,
            "score_max": float(arr.max()) if arr.size else None,
            "score_p10": float(np.percentile(arr, 10)) if arr.size else None,
            "provisional": self.provisional,
            "method": self.method,
        }

    def to_frame(self):
        import pandas as pd

        data = {"frame": self.frames, "timestamp": self.timestamps, "visibility_score": self.scores}
        for k, v in self.components_raw.items():
            data[f"raw_{k}"] = v
        for k, v in self.components_norm.items():
            data[f"norm_{k}"] = v
        return pd.DataFrame(data)
