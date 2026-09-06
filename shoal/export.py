"""Module [11] — EXPORTS (Phase 1: CSV, GeoJSON, MOTChallenge).

Darwin Core (``occurrences_dwc.csv``) is Phase 2 (plan §13, priority 5); the writer
slot is stubbed here with a clear message so the pipeline never silently omits it.

    tracks.csv           full per-frame table (the §7 tracks_df, verbatim)
    track_summary.csv    one row per track (the §7 track_summary_df)
    trajectories.geojson LineString per track, summary attributes attached
    results_MOT.txt      MOTChallenge format, so any standard MOT evaluator can
                         independently verify the tracking
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def write_csv(tracks_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path,
              log: logging.Logger) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p1, p2 = out_dir / "tracks.csv", out_dir / "track_summary.csv"
    tracks_df.to_csv(p1, index=False)
    summary_df.to_csv(p2, index=False)
    log.info("export: wrote %s (%d rows), %s (%d rows)", p1.name, len(tracks_df), p2.name, len(summary_df))
    return [p1, p2]


def write_geojson(tracks_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path,
                  frame_h: int, log: logging.Logger) -> Path:
    """One LineString per track in the local pixel CRS.

    GeoJSON is nominally lon/lat, so we declare a local CRS in the file and flip Y
    (image coordinates run top-down) to keep the geometry visually upright in any
    GIS viewer. This is a pixel-space artefact, labelled as such.
    """
    features = []
    summ = summary_df.set_index("track_id")
    for tid, g in tracks_df.sort_values("frame").groupby("track_id"):
        if len(g) < 2:
            continue
        coords = [[float(x), float(frame_h - y)] for x, y in zip(g["x"], g["y"])]
        props = {"track_id": int(tid)}
        if tid in summ.index:
            row = summ.loc[tid]
            props.update({
                k: (None if pd.isna(row[k]) else (float(row[k]) if isinstance(row[k], (int, float, np.floating)) else str(row[k])))
                for k in ("n_frames", "duration_s", "path_length_px", "net_displacement_px",
                          "sinuosity", "mean_speed_px_s", "mean_speed_bl_s", "species",
                          "species_conf", "quality_flag", "behaviour_state")
            })
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                         "properties": props})

    fc = {
        "type": "FeatureCollection",
        "name": "shoal_trajectories",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:local:pixel"}},
        "_units": "pixels; Y flipped so up is up; not geographic",
        "features": features,
    }
    p = out_dir / "trajectories.geojson"
    p.write_text(json.dumps(fc))
    log.info("export: wrote %s (%d line features)", p.name, len(features))
    return p


def write_mot(tracks_df: pd.DataFrame, out_dir: Path, log: logging.Logger) -> Path:
    """MOTChallenge: frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1  (1-indexed frames)."""
    rows = []
    for r in tracks_df.itertuples():
        if r.is_interpolated:
            continue
        left, top = r.x - r.w / 2.0, r.y - r.h / 2.0
        rows.append(f"{int(r.frame) + 1},{int(r.track_id)},{left:.2f},{top:.2f},"
                    f"{r.w:.2f},{r.h:.2f},{r.conf:.4f},-1,-1,-1")
    p = out_dir / "results_MOT.txt"
    p.write_text("\n".join(rows) + ("\n" if rows else ""))
    log.info("export: wrote %s (%d detections, MOTChallenge format)", p.name, len(rows))
    return p


def write_darwin_core(*_args, **_kwargs):
    raise NotImplementedError(
        "Darwin Core Occurrence export is Phase 2 (plan §11.2 / §13 priority 5). "
        "It needs the species head and an event location; both arrive in Phase 2."
    )


def run_exports(tracks_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path,
                frame_h: int, formats: list[str], log: logging.Logger) -> dict[str, str]:
    out_dir = Path(out_dir)
    written: dict[str, str] = {}
    if "csv" in formats:
        for p in write_csv(tracks_df, summary_df, out_dir, log):
            written[p.name] = str(p)
    if "geojson" in formats:
        p = write_geojson(tracks_df, summary_df, out_dir, frame_h, log)
        written[p.name] = str(p)
    if "mot" in formats:
        p = write_mot(tracks_df, out_dir, log)
        written[p.name] = str(p)
    if "darwin_core" in formats:
        log.warning("export: darwin_core requested but is Phase 2 — skipped")
    return written
