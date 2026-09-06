"""Config merge, schema validation, export formats, camera-motion guard, behaviour."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from shoal.config import _deep_merge, load_config, resolve_device
from shoal.schema import SchemaError, empty_summary, empty_tracks, validate_summary, validate_tracks

LOG = logging.getLogger("test")


def test_deep_merge_recurses_mappings_and_replaces_lists():
    base = {"a": {"b": 1, "c": 2}, "l": [1, 2, 3]}
    over = {"a": {"c": 9, "d": 4}, "l": [7]}
    out = _deep_merge(base, over)
    assert out == {"a": {"b": 1, "c": 9, "d": 4}, "l": [7]}
    assert base["a"]["c"] == 2  # base untouched


def test_repo_config_loads_and_has_required_keys():
    cfg = load_config()
    assert cfg.get("detect.conf") is not None
    assert cfg.get("restore.gate_threshold") is not None
    assert cfg.require("repair.min_track_frames") >= 1
    with pytest.raises(KeyError):
        cfg.require("does.not.exist")


def test_resolve_device_maps_auto():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")


def test_schema_validators_reject_bad_frames():
    from shoal.schema import TRACK_SUMMARY_COLUMNS, TRACKS_COLUMNS

    bad = pd.DataFrame([{c: 0.0 for c in TRACKS_COLUMNS}])
    bad["is_interpolated"] = bad["is_interpolated"].astype(bool)
    bad["speed_px_s"] = -1.0                       # impossible (negative speed)
    with pytest.raises(SchemaError):
        validate_tracks(bad)

    s = pd.DataFrame([{c: 1.0 for c in TRACK_SUMMARY_COLUMNS}])
    s["sinuosity"] = 0.5                           # impossible (< 1)
    with pytest.raises(SchemaError):
        validate_summary(s)

    # a well-formed empty frame must pass
    validate_tracks(empty_tracks())
    validate_summary(empty_summary())


def test_mot_export_is_one_indexed_and_skips_interpolated(tmp_path):
    from shoal.export import write_mot

    df = pd.DataFrame([
        dict(track_id=3, frame=0, timestamp=0.0, x=100, y=100, w=20, h=10, conf=0.9,
             is_interpolated=False),
        dict(track_id=3, frame=1, timestamp=0.03, x=105, y=100, w=20, h=10, conf=0.8,
             is_interpolated=True),
    ])
    p = write_mot(df, tmp_path, LOG)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 1                       # interpolated row skipped
    parts = lines[0].split(",")
    assert parts[0] == "1"                       # frame 0 -> MOT frame 1
    assert parts[1] == "3"
    assert float(parts[2]) == 90.0               # bb_left = x - w/2


def test_geojson_export_structure(tmp_path):
    from shoal.export import write_geojson

    tracks = pd.DataFrame([
        dict(track_id=1, frame=f, timestamp=f / 30, x=100 + f, y=50, w=10, h=10, conf=0.9,
             is_interpolated=False) for f in range(5)
    ])
    summary = pd.DataFrame([dict(track_id=1, n_frames=5, duration_s=0.13, path_length_px=4.0,
                                 net_displacement_px=4.0, sinuosity=1.0, mean_speed_px_s=30.0,
                                 mean_speed_bl_s=3.0, species="fish", species_conf=1.0,
                                 quality_flag="ok", behaviour_state="transit")])
    p = write_geojson(tracks, summary, tmp_path, frame_h=200, log=LOG)
    import json

    fc = json.loads(p.read_text())
    assert fc["type"] == "FeatureCollection"
    assert fc["features"][0]["geometry"]["type"] == "LineString"
    # Y is flipped (image coords -> upright): y=50 in a 200-high frame -> 150
    assert fc["features"][0]["geometry"]["coordinates"][0][1] == 150


def test_camera_motion_guard_flags_pan(tmp_path):
    """Synthesise a static clip and a panning clip; the guard must tell them apart."""
    import cv2

    from shoal.ingest import assess_camera_motion

    rng = np.random.default_rng(0)
    texture = rng.integers(0, 255, (240, 640, 3), dtype=np.uint8)

    def write(path, pan_px):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (320, 240))
        for i in range(40):
            off = int(i * pan_px)
            vw.write(texture[:, off:off + 320].copy())
        vw.release()

    static_p, pan_p = tmp_path / "static.mp4", tmp_path / "pan.mp4"
    write(static_p, 0)
    write(pan_p, 5)
    cfg = dict(enabled=True, max_features=200, sample_stride=1,
               translation_frac_warn=0.006, drift_frac_warn=0.06)
    assert assess_camera_motion(static_p, cfg).is_static is True
    assert assess_camera_motion(pan_p, cfg).is_static is False


def test_behaviour_thresholds_are_population_percentiles():
    from shoal.behaviour import _percentile_thresholds

    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(dict(
        track_id=rng.integers(0, 20, n), is_interpolated=False,
        speed_px_s=rng.gamma(2, 20, n), speed_bl_s=rng.gamma(2, 1, n),
        turn_angle_rad=rng.normal(0, 0.5, n),
    ))
    thr = _percentile_thresholds(df, dict(transit_speed_pct=60, forage_turn_pct=65,
                                          escape_speed_pct=95))
    assert thr["transit_speed"] < thr["escape_speed"]
    assert thr["speed_unit"] == "bl_s"
