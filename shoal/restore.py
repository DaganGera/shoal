"""Module [3] — ADAPTIVE RESTORATION.

The GATE is the contribution (plan §4.2). ``restore_if_needed`` enhances a frame
*only* when its fused visibility score is below the configured threshold. Enhancing
an already-clear frame adds halo artefacts around fish edges and measurably hurts
detection — which is precisely what ``shoal.ablation`` measures live rather than
asserting.

Enhancement chain, all OpenCV / CPU (target < 8 ms at 720p — measured, not quoted,
by ``shoal.ablation`` and printed in the dashboard):

    1. grey-world white balance in LAB
    2. Ancuti-style red-channel compensation (recover red from green)
    3. CLAHE on the L channel only
    4. mild unsharp mask
    5. luminance-driven gamma correction
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RestoreResult:
    frame: np.ndarray
    restored: bool          # did the gate fire?
    visibility_score: float
    elapsed_ms: float
    steps: tuple[str, ...]


def _grey_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    """Shift the LAB a/b channels so the frame's average colour is neutral.

    Underwater illumination is strongly non-neutral (blue-green). Grey-world in LAB
    (rather than per-channel RGB gain) keeps luminance untouched and only moves
    chroma, which is what a colour cast actually is. Implemented with saturating
    integer ``cv2.subtract`` of the scalar a/b offsets (fast; the per-pixel
    L-weighting in the textbook form buys almost nothing here).
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    means = cv2.mean(lab)
    shift = (0.0, means[1] - 128.0, means[2] - 128.0)  # leave L alone
    lab = cv2.subtract(lab, np.array(shift, dtype=np.float64))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _red_compensation(bgr: np.ndarray, alpha: float) -> np.ndarray:
    """I_r' = I_r + alpha * (mean_g - mean_r) * (1 - I_r) * I_g   (Ancuti et al.).

    Red light attenuates fastest in water; green survives longest. Where red is
    both dark and green is bright, borrow from green. The ``(1 - I_r)`` term stops
    already-bright reds (a diver's suit, a fish's fin) from blowing out.

    All arithmetic is kept in float32 — a stray float64 scalar here silently
    promotes the whole per-pixel array and triples the cost.
    """
    f = bgr.astype(np.float32)
    f *= np.float32(1.0 / 255.0)
    g = f[:, :, 1]
    r = f[:, :, 2]
    k = np.float32(alpha) * np.float32(float(g.mean()) - float(r.mean()))
    r += k * (np.float32(1.0) - r) * g
    np.clip(f, 0.0, 1.0, out=f)
    f *= np.float32(255.0)
    return f.astype(np.uint8)


def _clahe_on_L(bgr: np.ndarray, clip: float, grid: int) -> np.ndarray:
    """CLAHE on L only — CLAHE on RGB amplifies chroma noise, which turbid frames
    already have too much of."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _unsharp(bgr: np.ndarray, amount: float) -> np.ndarray:
    """Restore edge definition lost to forward scattering. Mild by design — a strong
    unsharp mask is exactly what creates the halos the gate exists to avoid."""
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.4)
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def _gamma_toward(bgr: np.ndarray, target_luma: float) -> np.ndarray:
    """Pick gamma from the frame's own mean luminance, not a constant."""
    b_m, g_m, r_m, _ = cv2.mean(bgr)      # scalar per-channel means, no big array
    luma = float(np.clip((0.114 * b_m + 0.587 * g_m + 0.299 * r_m) / 255.0, 1e-3, 1 - 1e-3))
    gamma = float(np.clip(np.log(target_luma) / np.log(luma), 0.5, 2.0))
    lut = np.clip((np.arange(256) / 255.0) ** (1.0 / gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(bgr, lut)


def enhance(bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Run the full enhancement chain unconditionally (the gate lives in the caller)."""
    out = bgr
    if cfg.get("grey_world_white_balance", True):
        out = _grey_world_white_balance(out)
    out = _red_compensation(out, float(cfg.get("red_compensation_alpha", 1.0)))
    out = _clahe_on_L(out, float(cfg.get("clahe_clip", 2.0)), int(cfg.get("clahe_grid", 8)))
    out = _unsharp(out, float(cfg.get("unsharp_amount", 0.6)))
    out = _gamma_toward(out, float(cfg.get("gamma_target_luma", 0.5)))
    return out


def _chain_steps(cfg: dict) -> tuple[str, ...]:
    steps = []
    if cfg.get("grey_world_white_balance", True):
        steps.append("grey_world_wb")
    steps += ["red_compensation", "clahe_L", "unsharp", "gamma"]
    return tuple(steps)


def restore_if_needed(bgr: np.ndarray, visibility_score: float, cfg: dict) -> RestoreResult:
    """Enhance the frame iff ``visibility_score < cfg['gate_threshold']``."""
    threshold = float(cfg.get("gate_threshold", 0.55))
    if visibility_score >= threshold:
        return RestoreResult(bgr, False, visibility_score, 0.0, ())
    t0 = time.perf_counter()
    out = enhance(bgr, cfg)
    return RestoreResult(
        out, True, visibility_score, (time.perf_counter() - t0) * 1e3, _chain_steps(cfg)
    )
