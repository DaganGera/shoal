"""Module [6] — TRACK REPAIR. Unglamorous and load-bearing (plan §7).

Raw bounding-box centroids jitter several pixels frame to frame. Speed is the first
derivative of position, so that jitter is *amplified*: speed computed on raw
centroids is mostly noise, and every downstream ecological metric would then be
built on sand. This module produces the clean ``tracks_df`` and ``track_summary_df``
that the §7 contract promises.

    1. smoothing   Savitzky-Golay on x and y independently (preserves burst peaks)
    2. gap fill    linear interpolation across gaps <= max_gap_interpolate, flagged
    3. stitching   link fragments <= max_gap_stitch apart when motion is consistent
    4. quality     per-track flags: n_frames, frac_interpolated, conf, gaps, area stability
    5. min track   >= min_track_frames to enter ecology; shorter ones counted as rejected
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from shoal.schema import (
    TRACK_SUMMARY_COLUMNS,
    TRACKS_COLUMNS,
    empty_summary,
    empty_tracks,
    validate_summary,
    validate_tracks,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _validated_savgol_window(requested: int, polyorder: int, n_points: int, fps: float,
                             log: logging.Logger) -> int | None:
    """Return an odd window that is >= polyorder+2, <= n_points, and spans <= ~0.4 s.

    The 0.4 s cap (verification checklist item 3) keeps genuine burst accelerations
    from being smoothed away: a startle response develops in ~100-300 ms, so the
    filter must not average over much more than that.
    """
    if n_points < polyorder + 2:
        return None
    max_span_frames = max(polyorder + 2, int(round(0.4 * fps)) | 1)
    win = min(requested, n_points, max_span_frames)
    if win % 2 == 0:
        win -= 1
    if win < polyorder + 2:
        return None
    if win != requested:
        log.debug("savgol window %d -> %d (n_points=%d, fps=%.1f, 0.4s=%d frames)",
                  requested, win, n_points, fps, max_span_frames)
    return win


def _smooth_xy(x: np.ndarray, y: np.ndarray, window: int, poly: int) -> tuple[np.ndarray, np.ndarray]:
    return (savgol_filter(x, window, poly), savgol_filter(y, window, poly))


def _circular_diff(a: np.ndarray) -> np.ndarray:
    """Wrapped difference of an angle series into (-pi, pi]."""
    d = np.diff(a, prepend=a[:1])
    return (d + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# stage 1-2: densify + smooth a single track
# --------------------------------------------------------------------------- #
def _densify_track(g: pd.DataFrame, fps: float, max_gap: int) -> pd.DataFrame:
    """Reindex a track onto every integer frame in its span, linearly interpolating
    each run of missing frames whose WIDTH is <= ``max_gap``, and marking those rows
    ``is_interpolated``. Wider gaps are left as holes (they raise ``n_gaps`` and, if
    narrow enough, become stitch candidates).

    Gap WIDTH — not distance-to-the-nearest-observation — is the criterion: an
    8-frame hole has interior points only 4 frames from an endpoint, but the plan
    (§7) says "interpolate across gaps <= 5 frames", i.e. the whole run must be short.
    """
    g = g.sort_values("frame")
    f0, f1 = int(g["frame"].iloc[0]), int(g["frame"].iloc[-1])
    full = pd.RangeIndex(f0, f1 + 1, name="frame")
    g = g.set_index("frame").reindex(full)

    present = g["x"].notna().to_numpy()

    # classify each missing frame by the width of the contiguous missing run it is in
    short_gap = np.zeros(len(present), dtype=bool)
    i = 0
    while i < len(present):
        if present[i]:
            i += 1
            continue
        j = i
        while j < len(present) and not present[j]:
            j += 1
        if (j - i) <= max_gap and i > 0 and j < len(present):   # bounded run, not a ragged end
            short_gap[i:j] = True
        i = j

    for col in ("x", "y", "w", "h", "conf"):
        g[col] = g[col].astype(float).interpolate(method="linear", limit_direction="both")
    g["is_interpolated"] = (~present) & short_gap
    g = g[present | short_gap]   # drop frames inside wide gaps and ragged ends

    g["timestamp"] = g.index.to_numpy() / fps
    g["track_id"] = int(pd.unique(g["track_id"].dropna())[0]) if g["track_id"].notna().any() else -1
    return g.reset_index()


# --------------------------------------------------------------------------- #
# stage 3: stitching
# --------------------------------------------------------------------------- #
def _stitch(summaries: pd.DataFrame, tracks: pd.DataFrame, cfg: dict,
            log: logging.Logger) -> dict[int, int]:
    """Return a remap {old_track_id -> canonical_track_id} that links fragments.

    A link is made when the tail of track A and the head of track B are separated
    by <= ``max_gap_stitch`` frames, the implied closing speed is plausible, and
    the heading is roughly continuous. Deliberately conservative: over-merging
    distinct individuals would deflate abundance (verification checklist item 4).
    """
    max_gap = int(cfg.get("max_gap_stitch", 15))
    max_speed = float(cfg.get("stitch_max_speed_px_s", 1200))
    fps_guard = 1e-6

    ends = {}
    for tid, g in tracks.groupby("track_id"):
        g = g.sort_values("frame")
        head, tail = g.iloc[0], g.iloc[-1]
        ends[int(tid)] = dict(
            f0=int(head["frame"]), f1=int(tail["frame"]),
            x0=head["x"], y0=head["y"], x1=tail["x"], y1=tail["y"],
            hx=tail["x"] - g.iloc[max(0, len(g) - 4)]["x"],
            hy=tail["y"] - g.iloc[max(0, len(g) - 4)]["y"],
        )

    parent: dict[int, int] = {t: t for t in ends}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    order = sorted(ends, key=lambda t: ends[t]["f1"])
    for a in order:
        ea = ends[a]
        best = None
        for b in ends:
            if b == a:
                continue
            eb = ends[b]
            gap = eb["f0"] - ea["f1"]
            if gap < 1 or gap > max_gap:
                continue
            dist = np.hypot(eb["x0"] - ea["x1"], eb["y0"] - ea["y1"])
            speed = dist / (gap / max(cfg.get("_fps", 30.0), fps_guard))
            if speed > max_speed:
                continue
            # heading continuity: predicted position from A's tail velocity
            pred_x = ea["x1"] + ea["hx"] / 3.0 * gap
            pred_y = ea["y1"] + ea["hy"] / 3.0 * gap
            pred_err = np.hypot(eb["x0"] - pred_x, eb["y0"] - pred_y)
            score = pred_err + 0.5 * dist
            if best is None or score < best[1]:
                best = (b, score)
        if best is not None:
            fa, fb = find(a), find(best[0])
            if fa != fb:
                parent[max(fa, fb)] = min(fa, fb)

    remap = {t: find(t) for t in ends}
    n_merged = len(set(remap.values()))
    if n_merged < len(remap):
        log.info("stitching: %d fragments -> %d tracks", len(remap), n_merged)
    return remap


# --------------------------------------------------------------------------- #
# stage 4-5: per-point kinematics + per-track summary
# --------------------------------------------------------------------------- #
def _kinematics(g: pd.DataFrame, fps: float, window: int, poly: int,
                log: logging.Logger) -> pd.DataFrame:
    g = g.sort_values("frame").reset_index(drop=True)
    x = g["x"].to_numpy(float)
    y = g["y"].to_numpy(float)
    win = _validated_savgol_window(window, poly, len(g), fps, log)
    if win is not None:
        xs, ys = _smooth_xy(x, y, win, poly)
    else:
        xs, ys = x, y  # too short to filter; kinematics will be flagged 'short' anyway
    g["x_smooth"], g["y_smooth"] = xs, ys

    dt = np.diff(g["timestamp"].to_numpy(float), prepend=g["timestamp"].iloc[0] - 1.0 / fps)
    dt[dt <= 0] = 1.0 / fps
    dx = np.diff(xs, prepend=xs[:1])
    dy = np.diff(ys, prepend=ys[:1])
    speed = np.hypot(dx, dy) / dt
    heading = np.arctan2(dy, dx)
    turn = _circular_diff(heading)

    body_len = np.maximum(g["w"].to_numpy(float), g["h"].to_numpy(float))
    body_len = np.where(body_len <= 1.0, np.nan, body_len)
    median_bl = np.nanmedian(body_len) if np.isfinite(body_len).any() else np.nan

    g["speed_px_s"] = np.clip(speed, 0.0, None)
    g["heading_rad"] = heading
    g["turn_angle_rad"] = turn
    g["body_length_px"] = np.where(np.isfinite(body_len), body_len, median_bl)
    g["speed_bl_s"] = g["speed_px_s"] / (median_bl if np.isfinite(median_bl) and median_bl > 0 else np.nan)
    return g


def _summarise(g: pd.DataFrame, fps: float, min_frames: int, stitched: bool) -> dict:
    x = g["x"].to_numpy(float)
    y = g["y"].to_numpy(float)
    seg = np.hypot(np.diff(x), np.diff(y))
    path_len = float(seg.sum())
    net_disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    sinuosity = float(path_len / net_disp) if net_disp > 1e-6 else float("inf")
    n_frames = len(g)
    real = g[~g["is_interpolated"]]
    frames_present = np.sort(g["frame"].to_numpy(int))
    n_gaps = int(np.sum(np.diff(frames_present) > 1))
    area = (g["w"] * g["h"]).to_numpy(float)
    area_stab = float(1.0 - np.clip(np.std(area) / (np.mean(area) + 1e-6), 0.0, 1.0))

    if n_frames < min_frames:
        quality = "short"
    elif area_stab < 0.4 or sinuosity == float("inf"):
        quality = "noisy"
    elif stitched:
        quality = "stitched"
    else:
        quality = "ok"

    bl = np.nanmedian(np.where(np.maximum(g["w"], g["h"]) > 1, np.maximum(g["w"], g["h"]), np.nan))
    return {
        "track_id": int(g["track_id"].iloc[0]),
        "n_frames": n_frames,
        "duration_s": float(g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]),
        "frac_interpolated": float(g["is_interpolated"].mean()),
        "n_gaps": n_gaps,
        "mean_detection_conf": float(real["conf"].mean()) if len(real) else float(g["conf"].mean()),
        "bbox_area_stability": area_stab,
        "path_length_px": path_len,
        "net_displacement_px": net_disp,
        "sinuosity": min(sinuosity, 9999.0),
        "mean_speed_px_s": float(g["speed_px_s"].mean()),
        "max_speed_px_s": float(g["speed_px_s"].max()),
        "mean_speed_bl_s": float(np.nanmean(g["speed_bl_s"])) if g["speed_bl_s"].notna().any() else float("nan"),
        "median_body_length_px": float(bl) if np.isfinite(bl) else float("nan"),
        "species": "fish",          # single class in Phase 1
        "species_conf": 1.0,        # placeholder; real track-vote confidence in Phase 2
        "quality_flag": quality,
        "behaviour_state": "unknown",
    }


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def repair_tracks(raw: pd.DataFrame, fps: float, cfg: dict,
                  log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """raw per-frame tracks -> (tracks_df, track_summary_df, provenance)."""
    prov = {
        "raw_rows": len(raw),
        "raw_track_ids": int(raw["track_id"].nunique()) if not raw.empty else 0,
        "savgol_window_requested": int(cfg.get("savgol_window", 9)),
        "savgol_polyorder": int(cfg.get("savgol_polyorder", 2)),
        "min_track_frames": int(cfg.get("min_track_frames", 15)),
    }
    if raw.empty:
        log.warning("repair: no raw tracks to process")
        return empty_tracks(), empty_summary(), prov

    cfg = {**cfg, "_fps": fps}
    max_gap_interp = int(cfg.get("max_gap_interpolate", 5))
    window = int(cfg.get("savgol_window", 9))
    poly = int(cfg.get("savgol_polyorder", 2))
    min_frames = int(cfg.get("min_track_frames", 15))

    # 1-2: densify + interpolate every raw track
    dens = [_densify_track(g, fps, max_gap_interp) for _, g in raw.groupby("track_id")]
    dens = pd.concat(dens, ignore_index=True)

    # 3: stitch fragments
    prelim_summ = pd.DataFrame(
        [_summarise(_kinematics(g, fps, window, poly, log), fps, min_frames, False)
         for _, g in dens.groupby("track_id")]
    )
    remap = _stitch(prelim_summ, dens, cfg, log)
    dens["orig_track_id"] = dens["track_id"]
    dens["track_id"] = dens["track_id"].map(lambda t: remap.get(int(t), int(t)))
    stitched_ids = {v for k, v in remap.items() if k != v}

    # collapse any duplicate frames created by a merge (keep the higher-conf row)
    dens = (dens.sort_values(["track_id", "frame", "conf"])
                .drop_duplicates(["track_id", "frame"], keep="last"))

    # 4: kinematics + 5: summary on the stitched tracks
    tracks_parts, summ_rows = [], []
    for tid, g in dens.groupby("track_id"):
        gk = _kinematics(g, fps, window, poly, log)
        tracks_parts.append(gk)
        summ_rows.append(_summarise(gk, fps, min_frames, tid in stitched_ids))

    tracks_df = pd.concat(tracks_parts, ignore_index=True)
    tracks_df["is_interpolated"] = tracks_df["is_interpolated"].astype(bool)
    tracks_df = tracks_df[TRACKS_COLUMNS].sort_values(["track_id", "frame"]).reset_index(drop=True)

    summary_df = pd.DataFrame(summ_rows, columns=TRACK_SUMMARY_COLUMNS)
    summary_df = summary_df.sort_values("track_id").reset_index(drop=True)

    accepted = summary_df[summary_df["quality_flag"] != "short"]
    prov.update(
        stitched_track_ids=sorted(int(i) for i in stitched_ids),
        n_tracks_total=len(summary_df),
        n_tracks_accepted=len(accepted),
        n_tracks_rejected_short=int((summary_df["quality_flag"] == "short").sum()),
        n_points_interpolated=int(tracks_df["is_interpolated"].sum()),
    )
    log.info("repair: %d raw ids -> %d tracks (%d accepted, %d rejected as short); %d interpolated points",
             prov["raw_track_ids"], prov["n_tracks_total"], prov["n_tracks_accepted"],
             prov["n_tracks_rejected_short"], prov["n_points_interpolated"])

    validate_tracks(tracks_df)
    validate_summary(summary_df)
    return tracks_df, summary_df, prov
