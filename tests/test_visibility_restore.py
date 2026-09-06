"""Visibility index + adaptive restoration (plan §4)."""

from __future__ import annotations

import json

import numpy as np

from shoal import restore
from shoal.visibility import VisibilityCalibration, raw_components


def _turbid_frame(h=200, w=320):
    """A hazy, low-contrast, blue-green frame."""
    f = np.full((h, w, 3), (150, 120, 40), np.uint8)  # BGR: lots of blue
    f = f + np.random.default_rng(0).integers(-8, 8, f.shape, dtype=np.int16)
    return np.clip(f, 0, 255).astype(np.uint8)


def _clear_frame(h=200, w=320):
    rng = np.random.default_rng(1)
    f = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)  # high contrast, neutral
    return f


def test_components_separate_clear_from_turbid():
    clear = raw_components(_clear_frame())
    turbid = raw_components(_turbid_frame())
    assert clear["contrast"] > turbid["contrast"]
    assert clear["sharpness"] > turbid["sharpness"]
    assert turbid["haze"] > clear["haze"]


def test_calibration_fuse_orders_clear_above_turbid_and_clamps():
    calib = VisibilityCalibration.load()  # repo config (calibrated by shoal-calibrate)
    sc = calib.fuse(raw_components(_clear_frame()))["score"]
    st = calib.fuse(raw_components(_turbid_frame()))["score"]
    assert 0.0 <= st <= sc <= 1.0


def test_provisional_flag_follows_calibration_state(tmp_path):
    prov = {
        "calibrated": False, "method": "provisional",
        "component_normalisation": {k: {"lo": 0, "hi": 100, "invert": False}
                                    for k in ("contrast", "colour_cast", "sharpness", "haze")},
        "fusion_weights": {k: 0.25 for k in ("contrast", "colour_cast", "sharpness", "haze")},
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(prov))
    calib = VisibilityCalibration.load(p)
    out = calib.fuse(raw_components(_clear_frame()))
    assert out["provisional"] is True


def test_restoration_gate_only_fires_below_threshold():
    frame = _turbid_frame()
    cfg = {"gate_threshold": 0.55}
    assert restore.restore_if_needed(frame, 0.8, cfg).restored is False
    r = restore.restore_if_needed(frame, 0.3, cfg)
    assert r.restored is True
    assert r.frame.shape == frame.shape
    assert r.elapsed_ms > 0
    assert not np.array_equal(r.frame, frame)          # it actually changed the pixels


def test_restoration_reduces_blue_cast_on_turbid_frame():
    frame = _turbid_frame()
    out = restore.enhance(frame, {"grey_world_white_balance": True, "red_compensation_alpha": 1.0,
                                  "clahe_clip": 2.0, "clahe_grid": 8, "unsharp_amount": 0.6,
                                  "gamma_target_luma": 0.5})
    b0, _, r0 = frame.reshape(-1, 3).mean(0)
    b1, _, r1 = out.reshape(-1, 3).mean(0)
    assert (b1 - r1) < (b0 - r0)                       # blue/red gap narrows
    assert out.dtype == np.uint8
    assert not np.isnan(out).any()


def test_restoration_is_fast_enough_to_be_practical():
    """Not the 8 ms aspiration, but it must stay well under a frame budget so the
    gated path during tracking is usable. Measured, not assumed (plan §4.2)."""
    import time

    frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    cfg = {"grey_world_white_balance": True, "red_compensation_alpha": 1.0, "clahe_clip": 2.0,
           "clahe_grid": 8, "unsharp_amount": 0.6, "gamma_target_luma": 0.5}
    restore.enhance(frame, cfg)  # warm
    t = time.perf_counter()
    for _ in range(10):
        restore.enhance(frame, cfg)
    ms = (time.perf_counter() - t) / 10 * 1e3
    assert ms < 120  # generous ceiling; real number lands in ablation.json
