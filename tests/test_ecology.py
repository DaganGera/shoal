"""Ecology — the scoring choices the plan stakes its credibility on (§10):
MaxN robustness to ID fragmentation, the honesty guard, diversity on per-species
MaxN, and a bootstrap CI that widens as species confidence falls.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from shoal import ecology
from shoal.schema import TRACK_SUMMARY_COLUMNS

LOG = logging.getLogger("test")


def _mk(summary_rows, track_frames):
    """summary_rows: list of dict; track_frames: {track_id: (f0, f1)}"""
    summ = pd.DataFrame(summary_rows)
    for c in TRACK_SUMMARY_COLUMNS:
        if c not in summ.columns:
            summ[c] = 0 if c not in ("species", "quality_flag", "behaviour_state") else "ok"
    rows = []
    for tid, (f0, f1) in track_frames.items():
        for f in range(f0, f1):
            rows.append(dict(track_id=tid, frame=f, timestamp=f / 30.0, x=10.0, y=10.0,
                             w=20.0, h=10.0, conf=0.8, is_interpolated=False,
                             speed_px_s=1.0, heading_rad=0.0, turn_angle_rad=0.0,
                             body_length_px=20.0, speed_bl_s=0.05))
    return pd.DataFrame(rows), summ


def test_maxn_is_robust_to_id_fragmentation():
    """One real fish split into 3 non-overlapping track fragments: track count
    triples, MaxN stays 1."""
    frames = {1: (0, 20), 2: (20, 40), 3: (40, 60)}
    summ = [dict(track_id=t, species="fish", species_conf=1.0, quality_flag="ok") for t in frames]
    tracks, summary = _mk(summ, frames)
    ab = ecology.compute_abundance(tracks, summary, dict(maxn_smooth_frames=1), LOG)
    assert ab.maxn_total == 1
    assert ab.unique_tracks_total == 3
    assert ab.maxn_total < ab.unique_tracks_total  # MaxN is the conservative bound


def test_maxn_counts_simultaneous_individuals():
    frames = {1: (0, 30), 2: (0, 30), 3: (10, 20)}   # 3 fish overlap on frames 10-19
    summ = [dict(track_id=t, species="fish", species_conf=1.0, quality_flag="ok") for t in frames]
    tracks, summary = _mk(summ, frames)
    ab = ecology.compute_abundance(tracks, summary, dict(maxn_smooth_frames=1), LOG)
    assert ab.maxn_total == 3


def test_honesty_guard_fires_without_species_head():
    frames = {1: (0, 30), 2: (0, 30)}
    summ = [dict(track_id=t, species="fish", species_conf=1.0, quality_flag="ok") for t in frames]
    tracks, summary = _mk(summ, frames)
    dv = ecology.compute_diversity(tracks, summary, dict(species_conf_floor=0.35),
                                   species_head_available=False, log=LOG)
    assert dv.available is False
    assert "abundance metrics remain valid" in dv.reason.lower()


def test_honesty_guard_fires_on_low_species_confidence():
    frames = {1: (0, 30), 2: (0, 30), 3: (0, 30)}
    summ = [dict(track_id=1, species="Scaridae", species_conf=0.2, quality_flag="ok"),
            dict(track_id=2, species="Serranidae", species_conf=0.15, quality_flag="ok"),
            dict(track_id=3, species="Scaridae", species_conf=0.25, quality_flag="ok")]
    tracks, summary = _mk(summ, frames)
    dv = ecology.compute_diversity(tracks, summary, dict(species_conf_floor=0.35),
                                   species_head_available=True, log=LOG)
    assert dv.available is False


def test_bootstrap_ci_widens_as_species_confidence_falls():
    frames = {i: (0, 30) for i in range(1, 7)}
    species = ["Scaridae", "Scaridae", "Serranidae", "Serranidae", "Lutjanidae", "Lutjanidae"]

    def diversity_at_conf(conf):
        summ = [dict(track_id=i + 1, species=species[i], species_conf=conf, quality_flag="ok")
                for i in range(6)]
        tracks, summary = _mk(summ, frames)
        return ecology.compute_diversity(
            tracks, summary, dict(species_conf_floor=0.0, bootstrap_iters=400, bootstrap_ci=0.95,
                                  _seed=1),
            species_head_available=True, log=LOG)

    high = diversity_at_conf(0.98)
    low = diversity_at_conf(0.55)
    hi_w = high.shannon_ci[1] - high.shannon_ci[0]
    lo_w = low.shannon_ci[1] - low.shannon_ci[0]
    assert lo_w > hi_w  # lower confidence -> wider interval (plan §10.3)


def test_diversity_uses_per_species_maxn_not_detection_counts():
    """Two species, equal MaxN (1 each), but species A lingers 10x longer. Diversity
    must treat them as even (H' ~ ln 2), not weight by residency."""
    frames = {1: (0, 100), 2: (0, 10)}
    summ = [dict(track_id=1, species="Scaridae", species_conf=0.9, quality_flag="ok"),
            dict(track_id=2, species="Serranidae", species_conf=0.9, quality_flag="ok")]
    tracks, summary = _mk(summ, frames)
    dv = ecology.compute_diversity(tracks, summary,
                                   dict(species_conf_floor=0.0, bootstrap_iters=200, _seed=1),
                                   species_head_available=True, log=LOG)
    assert dv.available
    assert abs(dv.shannon_h - np.log(2)) < 0.05
