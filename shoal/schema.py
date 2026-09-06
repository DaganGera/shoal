"""The dataframe contracts every module consumes and emits (plan §7).

Keeping these in one place is the mechanism behind plan §16's "keep module
boundaries clean". Each stage calls ``validate_tracks`` / ``validate_summary`` on
its output; a missing or misnamed column fails loudly at the boundary instead of
producing a subtly wrong ecology number three stages later.
"""

from __future__ import annotations

import pandas as pd

# Per-frame, per-track record. Emitted by shoal.repair, consumed by everything after.
TRACKS_COLUMNS: list[str] = [
    "track_id",
    "frame",
    "timestamp",       # seconds from clip start (real time, stride-independent)
    "x", "y",          # bounding-box centre, pixels, in the ORIGINAL frame CRS
    "w", "h",          # bounding-box size, pixels
    "conf",            # detection confidence that produced this observation
    "is_interpolated", # True where the point was gap-filled, excluded from metrics
    "speed_px_s",      # first derivative of the SMOOTHED centre (never raw centroids)
    "heading_rad",     # direction of travel, atan2(dy, dx), (-pi, pi]
    "turn_angle_rad",  # change in heading between consecutive points, (-pi, pi]
    "body_length_px",  # per-point body-length proxy = max(w, h)
    "speed_bl_s",      # speed in body-lengths / second (calibration-free, ecology-standard)
]

# One row per track. Emitted by shoal.repair, augmented by species/behaviour.
TRACK_SUMMARY_COLUMNS: list[str] = [
    "track_id",
    "n_frames",
    "duration_s",
    "frac_interpolated",
    "n_gaps",
    "mean_detection_conf",
    "bbox_area_stability",   # 1 - normalised std of bbox area; low = erratic detections
    "path_length_px",
    "net_displacement_px",
    "sinuosity",             # path_length / net_displacement (1.0 = straight)
    "mean_speed_px_s",
    "max_speed_px_s",
    "mean_speed_bl_s",
    "median_body_length_px",
    "species",               # 'fish' in Phase 1 (single class); real label in Phase 2
    "species_conf",          # 1.0 placeholder in Phase 1; track-vote confidence in Phase 2
    "quality_flag",          # 'ok' | 'short' | 'noisy' | 'stitched'
    "behaviour_state",       # 'transit' | 'foraging' | 'escape' | 'unknown'
]


class SchemaError(ValueError):
    """Raised at a module boundary when a dataframe violates the §7 contract."""


def _check(df: pd.DataFrame, required: list[str], name: str) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{name} is missing columns {missing}; has {list(df.columns)}")
    return df


def validate_tracks(df: pd.DataFrame) -> pd.DataFrame:
    _check(df, TRACKS_COLUMNS, "tracks_df")
    if df.empty:
        return df
    if (df["speed_px_s"] < 0).any():
        raise SchemaError("tracks_df has negative speed_px_s")
    if not df["is_interpolated"].dtype == bool:
        raise SchemaError("tracks_df.is_interpolated must be boolean")
    return df


def validate_summary(df: pd.DataFrame) -> pd.DataFrame:
    _check(df, TRACK_SUMMARY_COLUMNS, "track_summary_df")
    if df.empty:
        return df
    if (df["sinuosity"] < 0.999).any():
        raise SchemaError("track_summary_df has sinuosity < 1, which is geometrically impossible")
    return df


def empty_tracks() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in TRACKS_COLUMNS})


def empty_summary() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in TRACK_SUMMARY_COLUMNS})
