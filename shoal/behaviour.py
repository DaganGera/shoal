"""Module [9] — BEHAVIOUR (Phase 1: rule-based classifier only).

Plan §9 is the section that wins the challenge, and its headline method is an HMM on
(step length, turning angle) — the standard approach in movement ecology. That HMM
is **Phase 2** (plan §13, priority 1). Phase 1 ships the rule-based classifier that
the HMM will later replace, and it is labelled as provisional everywhere it appears.

The one thing done right even in the rule-based version: the cut-points are the
track population's own percentiles, computed at runtime from this clip's data — not
hand-set absolute speeds. That keeps the classifier meaningful when it moves to a
clip with a different frame rate, fish size, or camera distance.

States (interpreted post hoc, exactly as the HMM states will be):
    foraging : low speed, high turning        -> area-restricted search
    transit  : high speed, low turning        -> cruising / directed travel
    escape   : very high speed + acceleration  -> startle / flight
    unknown  : track too short to classify

Collective metrics (polarisation Φ, nearest-neighbour distance) are Phase 2
(plan §13, priority 2) and are intentionally NOT computed here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BehaviourProducts:
    tracks_df: pd.DataFrame          # + column: behaviour_state (per point)
    summary_df: pd.DataFrame         # behaviour_state filled (per track, majority vote)
    state_timeline: pd.DataFrame     # frame | timestamp | n_foraging | n_transit | n_escape
    method: str
    provisional: bool
    thresholds: dict[str, float]
    provenance: dict[str, Any] = field(default_factory=dict)


def _percentile_thresholds(tracks_df: pd.DataFrame, cfg: dict) -> dict[str, float]:
    """Derive cut-points from this clip's own kinematic distribution."""
    moving = tracks_df[(~tracks_df["is_interpolated"]) & (tracks_df["speed_px_s"] > 0)]
    speed = moving["speed_bl_s"].dropna()
    if speed.empty:                     # no body-length proxy: fall back to px/s
        speed = moving["speed_px_s"].dropna()
        speed_unit = "px_s"
    else:
        speed_unit = "bl_s"
    turn = moving["turn_angle_rad"].abs().dropna()
    accel = moving.groupby("track_id")["speed_px_s"].diff().abs().dropna()

    return {
        "speed_unit": speed_unit,
        "transit_speed": float(np.percentile(speed, cfg.get("transit_speed_pct", 60))) if len(speed) else 0.0,
        "forage_turn": float(np.percentile(turn, cfg.get("forage_turn_pct", 65))) if len(turn) else 0.0,
        "escape_speed": float(np.percentile(speed, cfg.get("escape_speed_pct", 95))) if len(speed) else 0.0,
        "escape_accel": float(np.percentile(accel, 90)) if len(accel) else 0.0,
    }


def _classify_points(g: pd.DataFrame, thr: dict[str, float]) -> np.ndarray:
    speed_col = "speed_bl_s" if thr["speed_unit"] == "bl_s" else "speed_px_s"
    speed = g[speed_col].to_numpy(float)
    speed = np.nan_to_num(speed, nan=0.0)
    turn = np.abs(g["turn_angle_rad"].to_numpy(float))
    accel = np.abs(np.diff(g["speed_px_s"].to_numpy(float), prepend=g["speed_px_s"].to_numpy(float)[:1]))

    state = np.full(len(g), "foraging", dtype=object)
    state[speed >= thr["transit_speed"]] = "transit"
    escape = (speed >= thr["escape_speed"]) & (accel >= thr["escape_accel"])
    state[escape] = "escape"
    # a slow, weakly-turning fish is neither foraging nor transiting cleanly -> hold/mill
    state[(speed < thr["transit_speed"]) & (turn < thr["forage_turn"])] = "foraging"
    return state


def classify(tracks_df: pd.DataFrame, summary_df: pd.DataFrame, cfg: dict,
             log: logging.Logger) -> BehaviourProducts:
    method = cfg.get("method", "rules")
    thr = _percentile_thresholds(tracks_df, cfg)
    log.info("behaviour: rule-based classifier, thresholds (from clip percentiles) = %s", thr)

    tracks_df = tracks_df.copy()
    tracks_df["behaviour_state"] = "unknown"
    summary_df = summary_df.copy()

    short_ids = set(summary_df.loc[summary_df["quality_flag"] == "short", "track_id"])
    per_track_state: dict[int, str] = {}
    for tid, g in tracks_df.groupby("track_id"):
        if tid in short_ids or len(g) < 5:
            per_track_state[tid] = "unknown"
            continue
        states = _classify_points(g, thr)
        tracks_df.loc[g.index, "behaviour_state"] = states
        vals, counts = np.unique(states, return_counts=True)
        per_track_state[tid] = str(vals[np.argmax(counts)])

    summary_df["behaviour_state"] = summary_df["track_id"].map(per_track_state).fillna("unknown")

    # per-frame state counts (the ribbon under the video in the dashboard)
    tl = (tracks_df[tracks_df["behaviour_state"] != "unknown"]
          .groupby(["frame", "behaviour_state"]).size().unstack(fill_value=0))
    for col in ("foraging", "transit", "escape"):
        if col not in tl.columns:
            tl[col] = 0
    tl = tl.reset_index()
    frame_time = tracks_df.groupby("frame")["timestamp"].first()
    tl["timestamp"] = tl["frame"].map(frame_time)
    tl = tl.rename(columns={"foraging": "n_foraging", "transit": "n_transit", "escape": "n_escape"})
    tl = tl[["frame", "timestamp", "n_foraging", "n_transit", "n_escape"]]

    counts = summary_df["behaviour_state"].value_counts().to_dict()
    log.info("behaviour: per-track states %s", counts)
    prov = {
        "method": "rule-based (percentile cut-points)",
        "note": "HMM state segmentation is the Phase 2 headline method and replaces this",
        "state_counts": {k: int(v) for k, v in counts.items()},
    }
    return BehaviourProducts(tracks_df, summary_df, tl, method, True, thr, prov)
