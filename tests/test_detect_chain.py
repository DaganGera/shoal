"""Detector fallback chain (plan §5, §16 "write the fallback chains as real code paths").

Model loading needs weights + network, so these tests exercise the *chain logic*:
class discovery, and that each rung declines cleanly when its weights are absent so
control falls through to the next.
"""

from __future__ import annotations

import logging

from shoal.detect import _fish_class_ids, _try_finetuned, _try_marine_detect

LOG = logging.getLogger("test")
CFG = {"fish_class_aliases": ["fish", "small fish", "scaridae", "serranidae"],
       "conf": 0.2, "iou": 0.6, "imgsz": 640, "max_det": 300}


def test_fish_class_discovery_matches_aliases_and_the_word_fish():
    names = {0: "Fish", 1: "Crab", 2: "Scaridae", 3: "Jellyfish", 4: "Urchin", 5: "small fish"}
    ids = _fish_class_ids(names, CFG["fish_class_aliases"])
    assert set(ids) == {0, 2, 3, 5}          # Fish, Scaridae, Jellyfish (contains 'fish'), small fish
    assert 1 not in ids and 4 not in ids     # Crab, Urchin excluded


def test_finetuned_rung_declines_when_weights_absent(tmp_path):
    cfg = {**CFG, "finetuned_weights": tmp_path / "nope.pt"}
    assert _try_finetuned(cfg, "cpu", LOG) is None


def test_marine_detect_rung_declines_when_weights_absent(tmp_path):
    cfg = {**CFG, "marine_detect_weights": tmp_path / "nope.pt"}
    assert _try_marine_detect(cfg, "cpu", LOG) is None
