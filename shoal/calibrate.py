"""Visibility-index calibration (build-plan §4.1, verification checklist item 1).

The fused visibility score's normalisation ranges and weights must be FIT against
real anchor clips, not hand-chosen. This script:

  1. samples frames from a "clear" anchor clip and a "turbid" anchor clip
  2. computes the four raw components (shoal.visibility.raw_components) on each
  3. sets each component's normalisation [lo, hi] from the anchors' distributions
     (5th percentile of the worse clip -> 95th percentile of the better clip)
  4. sets fusion weights from how well each component SEPARATES the two anchors
     (weight_k proportional to the standardised mean difference for component k)
  5. writes config/visibility_calibration.json with ``calibrated: true`` plus the
     anchor values, so the number is reproducible and defensible

Phase 1 anchors: the clearest and murkiest clips available (Mixkit ``reef_clear``
and the labelled ``reef_turbid_matched`` fixture). When Brackish + DeepFish are
fetched, re-run with ``--clear`` / ``--turbid`` pointing at them.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from shoal.config import REPO_ROOT
from shoal.ingest import iter_frames, probe
from shoal.runlog import basic_logging
from shoal.visibility import _COMPONENTS, raw_components

log = logging.getLogger("shoal.calibrate")

CLIPS_DIR = REPO_ROOT / "data" / "clips"
OUT_PATH = REPO_ROOT / "config" / "visibility_calibration.json"

# Which direction is "clearer" for each component (see shoal.visibility).
_INVERT = {"contrast": False, "colour_cast": True, "sharpness": False, "haze": True}


def _sample_components(clip: Path, n: int, patch: int) -> dict[str, np.ndarray]:
    meta = probe(clip)
    total = meta.n_frames if meta.n_frames > 0 else n * 4
    stride = max(1, total // n)
    acc: dict[str, list[float]] = {c: [] for c in _COMPONENTS}
    for frame in iter_frames(clip, frame_stride=stride, max_frames=n):
        comp = raw_components(frame.bgr, patch)
        for c in _COMPONENTS:
            acc[c].append(comp[c])
    log.info("sampled %d frames from %s", len(acc["contrast"]), clip.name)
    return {c: np.asarray(v, dtype=float) for c, v in acc.items()}


def calibrate(clear_clip: Path, turbid_clip: Path, n_frames: int = 80,
              patch: int = 15) -> dict:
    clear = _sample_components(clear_clip, n_frames, patch)
    turbid = _sample_components(turbid_clip, n_frames, patch)

    norm: dict[str, dict[str, float]] = {}
    weights: dict[str, float] = {}
    anchors_clear: dict[str, float] = {}
    anchors_turbid: dict[str, float] = {}

    for c in _COMPONENTS:
        cv, tv = clear[c], turbid[c]
        anchors_clear[c] = float(np.median(cv))
        anchors_turbid[c] = float(np.median(tv))

        # normalisation range: span the union of the two anchor distributions
        lo = float(min(np.percentile(cv, 2), np.percentile(tv, 2)))
        hi = float(max(np.percentile(cv, 98), np.percentile(tv, 98)))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        norm[c] = {"lo": round(lo, 4), "hi": round(hi, 4), "invert": _INVERT[c]}

        # separation weight: standardised mean difference (Cohen's d), abs
        pooled_sd = np.sqrt(0.5 * (cv.var(ddof=1) + tv.var(ddof=1))) + 1e-9
        d = abs(cv.mean() - tv.mean()) / pooled_sd
        weights[c] = float(d)

    wsum = sum(weights.values()) or 1.0
    weights = {c: round(weights[c] / wsum, 4) for c in _COMPONENTS}

    payload = {
        "calibrated": True,
        "method": "anchor-fit: normalisation from 2-98 pctile union, weights from Cohen's d separation",
        "generated_utc": datetime.now(UTC).isoformat(),
        "n_frames_per_anchor": n_frames,
        "dark_channel_patch": patch,
        "anchors": {
            "clear": {"clip": clear_clip.name, "components_median": {k: round(v, 4) for k, v in anchors_clear.items()}},
            "turbid": {"clip": turbid_clip.name, "components_median": {k: round(v, 4) for k, v in anchors_turbid.items()}},
        },
        "component_normalisation": norm,
        "fusion_weights": weights,
        "_sanity": _sanity_check(norm, weights, anchors_clear, anchors_turbid),
    }
    return payload


def _sanity_check(norm, weights, clear, turbid) -> dict:
    """The fused score of the clear anchor MUST exceed that of the turbid anchor."""
    def fuse(vals):
        s = 0.0
        for c in _COMPONENTS:
            lo, hi = norm[c]["lo"], norm[c]["hi"]
            x = np.clip((vals[c] - lo) / (hi - lo), 0, 1)
            if norm[c]["invert"]:
                x = 1 - x
            s += weights[c] * x
        return float(s)

    sc, st = fuse(clear), fuse(turbid)
    ok = sc > st
    log.info("sanity: fused(clear)=%.3f  fused(turbid)=%.3f  ordered=%s", sc, st, ok)
    return {"fused_clear": round(sc, 4), "fused_turbid": round(st, 4), "ordered_correctly": ok}


def main(argv: list[str] | None = None) -> int:
    basic_logging()
    ap = argparse.ArgumentParser(description="Calibrate the visibility index against anchor clips.")
    ap.add_argument("--clear", type=Path, default=CLIPS_DIR / "reef_clear.mp4",
                    help="clear-water anchor clip")
    ap.add_argument("--turbid", type=Path, default=CLIPS_DIR / "reef_turbid_matched.mp4",
                    help="turbid-water anchor clip")
    ap.add_argument("--frames", type=int, default=80, help="frames sampled per anchor")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    for p in (args.clear, args.turbid):
        if not p.exists():
            log.error("anchor clip missing: %s  (run `uv run shoal-fetch`)", p)
            return 1

    payload = calibrate(args.clear, args.turbid, args.frames)
    args.out.write_text(json.dumps(payload, indent=2))
    log.info("wrote %s (calibrated=%s, weights=%s)",
             args.out, payload["calibrated"], payload["fusion_weights"])
    if not payload["_sanity"]["ordered_correctly"]:
        log.error("CALIBRATION FAILED SANITY: clear anchor did not score above turbid anchor. "
                  "Check that --clear and --turbid are not swapped.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
