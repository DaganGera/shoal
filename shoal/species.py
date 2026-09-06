"""Module [7] — SPECIES HEAD + track-level confidence voting.  **Phase 2.**

Design (plan §2, §7, §13 priority 3), for when this is built:

    A separate classifier runs on tracked crops — NOT a multi-class detector.
    Rationale: species identification is far more reliable at the TRACK level than
    the FRAME level. A track gives ~50-200 views of one animal at different angles
    and distances; confidence-weighted voting over the whole track turns a mediocre
    per-frame classifier into a good per-individual one, and yields a per-individual
    confidence that propagates into the diversity bootstrap (`shoal.ecology`).

    Phase 1 recall: `shoal.detect` already collapses every fish-family class to a
    single `fish`. This module re-expands it: it reloads the marine-detect FishInv
    weights (15 classes, regionally scoped to Tioman, Malaysia — state that caveat),
    classifies each track's crops, and votes.

    add_species_labels(tracks_df, summary_df, clip_path, cfg) -> summary_df with real
        `species` + `species_conf` per track (replacing the Phase 1 placeholders
        'fish' / 1.0), which is exactly what lets `shoal.ecology`'s honesty guard
        stand down and the diversity panel light up.

Phase 1 status: NOT IMPLEMENTED. The pipeline runs single-class; the honesty guard
in `shoal.ecology` fires by construction, which is correct.
"""

from __future__ import annotations

import logging

import pandas as pd

SPECIES_HEAD_WEIGHTS = "weights/FishInv.pt"  # marine-detect; 15 classes
REGION_CAVEAT = ("marine-detect models are trained on footage from Tioman, Malaysia — "
                 "they are a drop-in REGIONAL species head. Swap the head, keep the pipeline.")


def add_species_labels(tracks_df: pd.DataFrame, summary_df: pd.DataFrame,
                       clip_path: str, cfg: dict, log: logging.Logger) -> pd.DataFrame:
    raise NotImplementedError(
        "Species head + track-level voting is Phase 2 (plan §7 / §13 priority 3). "
        "See this module's docstring for the design. Phase 1 runs single-class and "
        "the diversity honesty guard fires — which is the correct behaviour."
    )
