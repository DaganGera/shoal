"""Module [10] — ECOLOGICAL INDICATORS.

The section that signals domain literacy. Three deliberate choices from plan §10:

1. Abundance is MaxN, not track count. MaxN = the maximum number of individuals of
   a species visible in any single frame. It is the standard metric in remote
   underwater video fisheries surveys and it is robust to ID switches: fragment one
   fish into three tracks and the track count triples while MaxN is unchanged. Track
   count is an UPPER bound; MaxN is a conservative LOWER bound. Both are reported,
   both labelled as bounds.

2. Diversity indices are computed on per-species MaxN, never per-detection counts.
   Per-detection counting weights by how long each fish happened to linger in frame,
   which is a residency artefact, not diversity.

3. Confidence intervals come from bootstrapping the index over per-track species
   confidences. When species classification is weak the interval widens honestly
   rather than the dashboard printing a confident wrong number.

Honesty guard (plan §10.4): with a single collapsed 'fish' class, diversity is
mathematically zero-information. If the species head is absent or its mean track
confidence is below the floor, the diversity result is returned with
``available=False`` and a reason string, and the dashboard greys the panel out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class AbundanceResult:
    maxn_by_species: dict[str, int]
    maxn_total: int
    maxn_frame: int                     # frame index where MaxN_total occurs
    maxn_timestamp: float
    unique_tracks_by_species: dict[str, int]
    unique_tracks_total: int
    per_frame_counts: pd.DataFrame       # frame | timestamp | count  (for the time series)

    def as_dict(self) -> dict[str, Any]:
        return {
            "maxn_by_species": self.maxn_by_species,
            "maxn_total": self.maxn_total,
            "maxn_frame": self.maxn_frame,
            "maxn_timestamp": round(self.maxn_timestamp, 2),
            "unique_tracks_by_species": self.unique_tracks_by_species,
            "unique_tracks_total": self.unique_tracks_total,
            "interpretation": (
                f"MaxN = {self.maxn_total} (conservative lower bound), "
                f"unique tracks = {self.unique_tracks_total} "
                f"(upper bound, sensitive to ID fragmentation)"
            ),
        }


@dataclass
class DiversityResult:
    available: bool
    reason: str
    richness: int | None = None
    shannon_h: float | None = None
    shannon_ci: tuple[float, float] | None = None
    pielou_j: float | None = None
    pielou_ci: tuple[float, float] | None = None
    simpson_1_minus_d: float | None = None
    simpson_ci: tuple[float, float] | None = None
    bootstrap_iters: int = 0
    mean_species_conf: float | None = None

    def as_dict(self) -> dict[str, Any]:
        def fmt(v, ci):
            if v is None:
                return None
            out = {"value": round(v, 3)}
            if ci is not None:
                out["ci95"] = [round(ci[0], 3), round(ci[1], 3)]
            return out

        return {
            "available": self.available,
            "reason": self.reason,
            "richness": self.richness,
            "shannon_h": fmt(self.shannon_h, self.shannon_ci),
            "pielou_j": fmt(self.pielou_j, self.pielou_ci),
            "simpson_1_minus_d": fmt(self.simpson_1_minus_d, self.simpson_ci),
            "bootstrap_iters": self.bootstrap_iters,
            "mean_species_conf": None if self.mean_species_conf is None else round(self.mean_species_conf, 3),
        }


@dataclass
class EcologyProducts:
    abundance: AbundanceResult
    diversity: DiversityResult
    provenance: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# abundance
# --------------------------------------------------------------------------- #
def compute_abundance(tracks_df: pd.DataFrame, summary_df: pd.DataFrame,
                      cfg: dict, log: logging.Logger) -> AbundanceResult:
    accepted = summary_df[summary_df["quality_flag"] != "short"]
    usable = tracks_df[tracks_df["track_id"].isin(accepted["track_id"])]
    species_of = accepted.set_index("track_id")["species"].to_dict()

    if usable.empty:
        return AbundanceResult({}, 0, -1, 0.0, {}, 0,
                               pd.DataFrame(columns=["frame", "timestamp", "count"]))

    usable = usable.assign(species=usable["track_id"].map(species_of).fillna("fish"))
    smooth = int(cfg.get("maxn_smooth_frames", 1))

    per_frame_species = (usable.groupby(["frame", "species"])["track_id"]
                         .nunique().unstack(fill_value=0).sort_index())
    if smooth > 1:
        per_frame_species = per_frame_species.rolling(smooth, min_periods=1, center=True).max()

    maxn_by_species = {sp: int(per_frame_species[sp].max()) for sp in per_frame_species.columns}
    total_per_frame = per_frame_species.sum(axis=1)
    maxn_total = int(total_per_frame.max())
    maxn_frame = int(total_per_frame.idxmax())
    frame_time = usable.groupby("frame")["timestamp"].first()
    maxn_ts = float(frame_time.get(maxn_frame, 0.0))

    per_frame_counts = pd.DataFrame({
        "frame": total_per_frame.index,
        "timestamp": [float(frame_time.get(f, 0.0)) for f in total_per_frame.index],
        "count": total_per_frame.to_numpy(int),
    })

    uniq_by_species = accepted.groupby("species")["track_id"].nunique().to_dict()
    log.info("abundance: MaxN_total=%d @ frame %d (t=%.1fs); unique tracks=%d; by species=%s",
             maxn_total, maxn_frame, maxn_ts, int(accepted["track_id"].nunique()), maxn_by_species)
    return AbundanceResult(
        maxn_by_species={k: int(v) for k, v in maxn_by_species.items()},
        maxn_total=maxn_total, maxn_frame=maxn_frame, maxn_timestamp=maxn_ts,
        unique_tracks_by_species={k: int(v) for k, v in uniq_by_species.items()},
        unique_tracks_total=int(accepted["track_id"].nunique()),
        per_frame_counts=per_frame_counts,
    )


# --------------------------------------------------------------------------- #
# diversity + bootstrap CI
# --------------------------------------------------------------------------- #
def _shannon(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _pielou(counts: np.ndarray) -> float:
    s = int(np.sum(counts > 0))
    if s <= 1:
        return float("nan")
    return _shannon(counts) / np.log(s)


def _simpson(counts: np.ndarray) -> float:
    n = counts.sum()
    if n <= 1:
        return float("nan")
    p = counts / n
    return float(1.0 - np.sum(p * p))


def _species_maxn_vector(per_frame_species: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    species = list(per_frame_species.columns)
    vec = np.array([per_frame_species[s].max() for s in species], dtype=float)
    return species, vec


def compute_diversity(tracks_df: pd.DataFrame, summary_df: pd.DataFrame, cfg: dict,
                      species_head_available: bool, log: logging.Logger) -> DiversityResult:
    accepted = summary_df[summary_df["quality_flag"] != "short"].copy()
    floor = float(cfg.get("species_conf_floor", 0.35))
    distinct_species = sorted(set(accepted["species"]) - {"fish", "", None})
    mean_conf = float(accepted["species_conf"].mean()) if len(accepted) else 0.0

    if not species_head_available or not distinct_species:
        return DiversityResult(
            available=False,
            reason=("Species identification unavailable (single collapsed 'fish' class). "
                    "Diversity over one class is zero-information. Abundance metrics remain valid."),
            mean_species_conf=mean_conf if distinct_species else None,
        )
    if mean_conf < floor:
        return DiversityResult(
            available=False,
            reason=(f"Species identification confidence too low for diversity estimation on this "
                    f"clip (mean track confidence {mean_conf:.2f} < floor {floor:.2f}). "
                    f"Abundance metrics remain valid."),
            mean_species_conf=mean_conf,
        )

    usable = tracks_df[tracks_df["track_id"].isin(accepted["track_id"])].assign(
        species=lambda d: d["track_id"].map(accepted.set_index("track_id")["species"])
    )
    per_frame_species = (usable.groupby(["frame", "species"])["track_id"]
                         .nunique().unstack(fill_value=0))
    _, maxn_vec = _species_maxn_vector(per_frame_species)

    h = _shannon(maxn_vec)
    j = _pielou(maxn_vec)
    d = _simpson(maxn_vec)
    richness = int(np.sum(maxn_vec > 0))

    # bootstrap: resample each track's species label by its per-track confidence,
    # recompute per-species MaxN and the indices (plan §10.3).
    iters = int(cfg.get("bootstrap_iters", 1000))
    rng = np.random.default_rng(int(cfg.get("_seed", 20260906)))
    tid_species = accepted["species"].to_numpy()
    tid_conf = accepted["species_conf"].to_numpy(float).clip(0, 1)
    tid_ids = accepted["track_id"].to_numpy()
    all_species = sorted(set(tid_species))
    sp_index = {s: i for i, s in enumerate(all_species)}

    # frame membership matrix: which accepted tracks are present in which frame
    frame_ids = usable.groupby("frame")["track_id"].apply(lambda s: set(s)).to_dict()
    hs, js, ds = np.empty(iters), np.empty(iters), np.empty(iters)
    for k in range(iters):
        # with prob conf keep label; else assign a random *other* species
        keep = rng.random(len(tid_ids)) < tid_conf
        draw = tid_species.copy()
        if (~keep).any() and len(all_species) > 1:
            for idx in np.where(~keep)[0]:
                alts = [s for s in all_species if s != tid_species[idx]] or all_species
                draw[idx] = rng.choice(alts)
        id_to_sp = dict(zip(tid_ids, draw))
        counts = np.zeros(len(all_species))
        # per-species MaxN under this resampling
        per_frame = np.zeros((len(frame_ids), len(all_species)))
        for fi, ids in enumerate(frame_ids.values()):
            for tid in ids:
                per_frame[fi, sp_index[id_to_sp[tid]]] += 1
        counts = per_frame.max(axis=0)
        hs[k] = _shannon(counts)
        js[k] = _pielou(counts)
        ds[k] = _simpson(counts)

    ci = float(cfg.get("bootstrap_ci", 0.95))
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100

    def interval(a):
        a = a[np.isfinite(a)]
        return (float(np.percentile(a, lo_q)), float(np.percentile(a, hi_q))) if a.size else None

    log.info("diversity: S=%d  H'=%.2f  J'=%.2f  1-D=%.2f  (bootstrap %d, mean species conf %.2f)",
             richness, h, j, d, iters, mean_conf)
    return DiversityResult(
        available=True, reason="ok", richness=richness,
        shannon_h=h, shannon_ci=interval(hs),
        pielou_j=None if not np.isfinite(j) else j, pielou_ci=interval(js),
        simpson_1_minus_d=None if not np.isfinite(d) else d, simpson_ci=interval(ds),
        bootstrap_iters=iters, mean_species_conf=mean_conf,
    )


def analyse(tracks_df: pd.DataFrame, summary_df: pd.DataFrame, cfg: dict,
            species_head_available: bool, log: logging.Logger,
            seed: int = 20260906) -> EcologyProducts:
    cfg = {**cfg, "_seed": seed}
    abundance = compute_abundance(tracks_df, summary_df, cfg, log)
    diversity = compute_diversity(tracks_df, summary_df, cfg, species_head_available, log)
    prov = {
        "abundance_metric": "MaxN (single-frame maximum), reported with unique-track upper bound",
        "diversity_basis": "per-species MaxN",
        "species_head_available": species_head_available,
        "honesty_guard_fired": not diversity.available,
    }
    return EcologyProducts(abundance, diversity, prov)
