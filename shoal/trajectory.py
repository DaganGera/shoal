"""Module [8] — TRAJECTORY ANALYTICS (MovingPandas).

Pixel coordinates are treated as a LOCAL PLANAR CRS. This is legitimate:
MovingPandas operates on any planar coordinate system, it does not need geographic
coordinates (plan §8). We wrap the repaired track table in a GeoDataFrame with a
synthetic metre-like CRS and build a TrajectoryCollection.

Everything here is in PIXELS and PIXELS/SECOND (and BL/s where a body-length proxy
exists). No metre value is ever produced from an assumed scale — the dashboard
supplies the optional calibration paths.

Derived:
  * per-track summary already carries speed / sinuosity (from shoal.repair)
  * TrajectoryStopDetector  -> stop points
  * DBSCAN on stop points   -> activity nodes / habitat hotspots
  * convex hulls around nodes -> data-driven habitat zones
  * zone-crossing segmentation -> origin/destination flow matrix
  * time-weighted 2D occupancy histogram -> residency heatmap
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# A synthetic planar CRS. Any projected CRS works; this one has metre units so
# MovingPandas' speed/length helpers return numbers we then relabel as pixels.
SYNTHETIC_CRS = "EPSG:32663"  # World Equidistant Cylindrical (metre units, planar)


@dataclass
class TrajectoryProducts:
    stop_points: pd.DataFrame              # track_id, x, y, t_start, t_end, duration_s
    activity_nodes: pd.DataFrame           # node_id, x, y, n_stops, radius_px
    flow_matrix: pd.DataFrame              # origin_node, dest_node, transitions
    residency: dict[str, Any]              # heatmap array + extent + smoothing meta
    per_track: pd.DataFrame                # trajectory-level extras merged onto the summary
    provenance: dict[str, Any] = field(default_factory=dict)


def _build_trajectory_collection(tracks_df: pd.DataFrame, fps: float):
    import geopandas as gpd
    import movingpandas as mpd
    from shapely.geometry import Point

    df = tracks_df.copy()
    # MovingPandas wants a DatetimeIndex; synthesise one from the real timestamps.
    # Round to whole milliseconds so MovingPandas' pydatetime conversions don't warn
    # about discarded sub-microsecond nanoseconds.
    df["t"] = pd.to_datetime((df["timestamp"] * 1000).round().astype("int64"),
                             unit="ms", origin="2026-01-01")
    gdf = gpd.GeoDataFrame(
        df, geometry=[Point(xy) for xy in zip(df["x"], df["y"])], crs=SYNTHETIC_CRS
    ).set_index("t")
    tc = mpd.TrajectoryCollection(gdf, traj_id_col="track_id", t=None)
    return tc


def _detect_stops(tc, cfg: dict, log: logging.Logger) -> pd.DataFrame:
    import movingpandas as mpd

    detector = mpd.TrajectoryStopDetector(tc)
    max_diameter = float(cfg.get("stop_max_diameter_px", 40))
    min_duration = pd.Timedelta(seconds=float(cfg.get("stop_min_duration_s", 1.2)))
    try:
        stops = detector.get_stop_points(min_duration=min_duration, max_diameter=max_diameter)
    except Exception as exc:
        log.warning("stop detection failed (%s); continuing with no stops", exc)
        return pd.DataFrame(columns=["track_id", "x", "y", "t_start", "t_end", "duration_s"])

    if stops is None or len(stops) == 0:
        return pd.DataFrame(columns=["track_id", "x", "y", "t_start", "t_end", "duration_s"])

    rows = []
    for _, s in stops.iterrows():
        geom = s.geometry
        rows.append({
            "track_id": int(str(s["traj_id"]).split("_")[0]) if "traj_id" in s else -1,
            "x": float(geom.x), "y": float(geom.y),
            "t_start": s.get("start_time"), "t_end": s.get("end_time"),
            "duration_s": float(s["duration_s"]) if "duration_s" in s
            else (s.get("end_time") - s.get("start_time")).total_seconds(),
        })
    log.info("stop detection: %d stop points (max_diameter=%.0f px, min_duration=%.1f s)",
             len(rows), max_diameter, min_duration.total_seconds())
    return pd.DataFrame(rows)


def _activity_nodes(stops: pd.DataFrame, cfg: dict, log: logging.Logger) -> pd.DataFrame:
    if len(stops) < int(cfg.get("dbscan_min_samples", 3)):
        return pd.DataFrame(columns=["node_id", "x", "y", "n_stops", "radius_px"])
    from sklearn.cluster import DBSCAN

    xy = stops[["x", "y"]].to_numpy(float)
    labels = DBSCAN(eps=float(cfg.get("dbscan_eps_px", 60)),
                    min_samples=int(cfg.get("dbscan_min_samples", 3))).fit_predict(xy)
    rows = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        pts = xy[labels == lab]
        centre = pts.mean(axis=0)
        radius = float(np.percentile(np.hypot(*(pts - centre).T), 90))
        rows.append({"node_id": int(lab), "x": float(centre[0]), "y": float(centre[1]),
                     "n_stops": len(pts), "radius_px": radius})
    log.info("activity nodes: %d clusters from %d stops (%d noise)",
             len(rows), len(stops), int(np.sum(labels == -1)))
    return pd.DataFrame(rows)


def _assign_zone(x: float, y: float, nodes: pd.DataFrame) -> int:
    if nodes.empty:
        return -1
    d = np.hypot(nodes["x"].to_numpy() - x, nodes["y"].to_numpy() - y)
    j = int(np.argmin(d))
    return int(nodes["node_id"].iloc[j]) if d[j] <= max(nodes["radius_px"].iloc[j] * 1.5, 80) else -1


def _flow_matrix(tracks_df: pd.DataFrame, nodes: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    if nodes.empty:
        return pd.DataFrame(columns=["origin_node", "dest_node", "transitions"])
    transitions: dict[tuple[int, int], int] = {}
    for _, g in tracks_df.sort_values("frame").groupby("track_id"):
        zone_seq = [_assign_zone(r.x, r.y, nodes) for r in g.itertuples()]
        zone_seq = [z for z in zone_seq if z != -1]
        collapsed = [z for i, z in enumerate(zone_seq) if i == 0 or z != zone_seq[i - 1]]
        for a, b in zip(collapsed, collapsed[1:]):
            transitions[(a, b)] = transitions.get((a, b), 0) + 1
    rows = [{"origin_node": a, "dest_node": b, "transitions": n}
            for (a, b), n in sorted(transitions.items())]
    log.info("flow matrix: %d directed edges between %d nodes", len(rows), len(nodes))
    return pd.DataFrame(rows, columns=["origin_node", "dest_node", "transitions"])


def _residency_heatmap(tracks_df: pd.DataFrame, frame_w: int, frame_h: int,
                       cfg: dict) -> dict[str, Any]:
    """Time-weighted 2D occupancy histogram over the frame. The single most legible
    visual in the project — a juror reads it without explanation (plan §8)."""
    from scipy.ndimage import gaussian_filter

    bins_long = int(cfg.get("residency_bins", 96))
    if frame_w >= frame_h:
        bins_x, bins_y = bins_long, max(1, round(bins_long * frame_h / frame_w))
    else:
        bins_x, bins_y = max(1, round(bins_long * frame_w / frame_h)), bins_long

    # weight each point by its frame duration so a paused fish counts for its dwell time
    dt = tracks_df.groupby("track_id")["timestamp"].transform(
        lambda s: s.diff().fillna(s.diff().median()).clip(lower=0)
    ).fillna(0).to_numpy()
    weights = np.where(dt > 0, dt, np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 1.0)

    hist, xedges, yedges = np.histogram2d(
        tracks_df["x"], tracks_df["y"], bins=[bins_x, bins_y],
        range=[[0, frame_w], [0, frame_h]], weights=weights,
    )
    smoothed = gaussian_filter(hist, sigma=float(cfg.get("residency_sigma_bins", 1.5)))
    return {
        "histogram": smoothed.T,                 # (y, x) for imshow
        "extent": [0, frame_w, frame_h, 0],
        "bins": [bins_x, bins_y],
        "sigma_bins": float(cfg.get("residency_sigma_bins", 1.5)),
        "total_dwell_s": float(weights.sum()),
        "units": "time-weighted occupancy (fish-seconds per bin)",
    }


def analyse(tracks_df: pd.DataFrame, summary_df: pd.DataFrame, frame_w: int, frame_h: int,
            fps: float, cfg: dict, log: logging.Logger) -> TrajectoryProducts:
    prov: dict[str, Any] = {"crs": SYNTHETIC_CRS, "units": "pixels / pixels-per-second"}
    usable = tracks_df[tracks_df["track_id"].isin(
        summary_df.loc[summary_df["quality_flag"] != "short", "track_id"]
    )]
    if usable.empty:
        log.warning("trajectory: no usable tracks after quality filter")
        empty = pd.DataFrame()
        return TrajectoryProducts(empty, empty, empty,
                                  _residency_heatmap(tracks_df if not tracks_df.empty else
                                                     pd.DataFrame({"x": [], "y": [], "timestamp": [],
                                                                   "track_id": []}),
                                                     frame_w, frame_h, cfg),
                                  summary_df.copy(), prov)

    tc = _build_trajectory_collection(usable, fps)
    stops = _detect_stops(tc, cfg, log)
    nodes = _activity_nodes(stops, cfg, log)
    flows = _flow_matrix(usable, nodes, log)
    residency = _residency_heatmap(usable, frame_w, frame_h, cfg)

    # trajectory-level extras onto the summary
    per_track = summary_df.copy()
    conv = {}
    for traj in tc.trajectories:
        tid = int(str(traj.id).split("_")[0])
        try:
            conv[tid] = float(traj.get_length())
        except Exception:
            continue
    per_track["mpd_length_px"] = per_track["track_id"].map(conv)

    prov.update(n_stop_points=len(stops), n_activity_nodes=len(nodes),
                n_flow_edges=len(flows))
    return TrajectoryProducts(stops, nodes, flows, residency, per_track, prov)
