"""Track repair — the load-bearing module (plan §7). Tests target the properties the
plan calls out explicitly: speed from smoothed (not raw) positions, gap flags,
conservative stitching, the §7 schema, and the frame-rate-aware Savitzky-Golay window.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from shoal.repair import _validated_savgol_window, repair_tracks
from shoal.schema import TRACK_SUMMARY_COLUMNS, TRACKS_COLUMNS

LOG = logging.getLogger("test")


def _straight_track(track_id, n=40, x0=100, y0=100, vx=5.0, vy=0.0, jitter=0.0, fps=30.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n):
        rows.append(dict(
            track_id=track_id, frame=f, timestamp=f / fps,
            x=x0 + vx * f + rng.normal(0, jitter), y=y0 + vy * f + rng.normal(0, jitter),
            w=20.0, h=10.0, conf=0.8,
        ))
    return pd.DataFrame(rows)


def test_savgol_window_is_frame_rate_aware():
    # 0.4 s at 60 fps = 24 frames -> a requested window of 31 must be capped below that
    win = _validated_savgol_window(31, 2, n_points=200, fps=60.0, log=LOG)
    assert win is not None and win % 2 == 1 and win <= 25
    # too few points -> no filtering
    assert _validated_savgol_window(9, 2, n_points=3, fps=30.0, log=LOG) is None


def test_speed_computed_from_smoothed_not_raw_centroids():
    """A dead-straight path corrupted by pixel jitter must yield near-constant speed
    after repair; computing speed on raw centroids would give a noisy series."""
    raw = _straight_track(1, n=60, vx=4.0, jitter=2.5, fps=30.0, seed=1)
    tracks, summary, _ = repair_tracks(raw, fps=30.0,
                                       cfg=dict(savgol_window=9, savgol_polyorder=2,
                                                min_track_frames=15, max_gap_interpolate=5,
                                                max_gap_stitch=15), log=LOG)
    g = tracks[tracks.track_id == 1]
    # true speed is 4 px/frame * 30 fps = 120 px/s
    interior = g.iloc[5:-5]
    assert interior["speed_px_s"].std() < 45  # smoothed: modest spread
    assert abs(interior["speed_px_s"].median() - 120) < 40

    raw_speed = np.hypot(raw.x.diff(), raw.y.diff()).dropna() * 30.0
    assert raw_speed.std() > interior["speed_px_s"].std()  # raw is noisier


def test_gap_interpolation_is_flagged_and_bounded():
    raw = _straight_track(1, n=40, fps=30.0)
    raw = raw[~raw.frame.isin([10, 11, 12])]                     # 3-frame gap: interpolate
    raw = raw[raw.frame < 25].pipe(lambda d: pd.concat([d, _straight_track(1, n=40).iloc[33:]]))  # 8-frame gap: leave a hole
    tracks, summary, prov = repair_tracks(raw, fps=30.0,
                                          cfg=dict(savgol_window=9, savgol_polyorder=2,
                                                   min_track_frames=15, max_gap_interpolate=5,
                                                   max_gap_stitch=0), log=LOG)
    g = tracks[tracks.track_id == 1]
    assert g.loc[g.frame.isin([10, 11, 12]), "is_interpolated"].all()
    assert not g[g.is_interpolated].empty
    assert g["is_interpolated"].dtype == bool
    # the wide gap must NOT have been filled
    assert not set(range(26, 33)).issubset(set(g.frame))


def test_stitching_is_conservative_about_distinct_individuals():
    """Two tracks far apart in space but close in time must NOT be merged."""
    a = _straight_track(1, n=20, x0=100, y0=100, vx=1.0)
    b = _straight_track(2, n=20, x0=900, y0=600, vx=1.0)
    b["frame"] += 22
    b["timestamp"] = b["frame"] / 30.0
    raw = pd.concat([a, b], ignore_index=True)
    _, summary, prov = repair_tracks(raw, fps=30.0,
                                     cfg=dict(savgol_window=9, savgol_polyorder=2,
                                              min_track_frames=10, max_gap_interpolate=5,
                                              max_gap_stitch=15, stitch_max_speed_px_s=800),
                                     log=LOG)
    assert summary["track_id"].nunique() == 2  # not stitched


def test_output_matches_section_7_schema():
    raw = pd.concat([_straight_track(i, n=30, x0=50 * i, seed=i) for i in range(1, 4)],
                    ignore_index=True)
    tracks, summary, _ = repair_tracks(raw, fps=30.0,
                                       cfg=dict(savgol_window=9, savgol_polyorder=2,
                                                min_track_frames=15, max_gap_interpolate=5,
                                                max_gap_stitch=15), log=LOG)
    assert list(tracks.columns) == TRACKS_COLUMNS
    assert list(summary.columns) == TRACK_SUMMARY_COLUMNS
    assert (summary["sinuosity"] >= 1.0 - 1e-6).all()
    assert (tracks["speed_px_s"] >= 0).all()


def test_short_tracks_rejected_but_counted():
    raw = pd.concat([_straight_track(1, n=40), _straight_track(2, n=6)], ignore_index=True)
    _, summary, prov = repair_tracks(raw, fps=30.0,
                                     cfg=dict(savgol_window=9, savgol_polyorder=2,
                                              min_track_frames=15, max_gap_interpolate=5,
                                              max_gap_stitch=0), log=LOG)
    assert prov["n_tracks_rejected_short"] >= 1
    assert (summary["quality_flag"] == "short").any()
