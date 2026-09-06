"""HOTA / MOTA / IDF1 benchmark — Phase 2 (plan §12, §13 priority 4).

Scope when built:

    Run on the Brackish test split with ground truth. 2x2 ablation:

                     | ByteTrack | BoT-SORT + Re-ID
        raw video    |  H/M/IDF1 |  H/M/IDF1
        restored     |  H/M/IDF1 |  H/M/IDF1

    Report HOTA as the headline (it balances detection and association; MOTA is
    dominated by detection count and flatters a good detector + mediocre tracker).
    Compute OFFLINE ONCE, cache to outputs/<run>/benchmark.json, render as a static
    card in dashboard tab 6. Never re-run during a live demo (plan §16).

Phase 1 status: NOT IMPLEMENTED. This module exists so the interface and the CLI
entry point are real. It needs:
  * Brackish/BrackishMOT fetched (`uv run shoal-fetch --brackishmot`, needs a Kaggle token)
  * `shoal.track.run_tracking` parametrised over method (already is: `method=bytetrack|botsort`)
  * `motmetrics` (installed) or TrackEval for the HOTA computation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shoal.config import REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SHOAL tracking benchmark (Phase 2 — not yet implemented).")
    ap.add_argument("--sequences", type=Path, default=REPO_ROOT / "data" / "brackishmot",
                    help="directory of MOTChallenge sequences with gt/gt.txt")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "benchmark.json")
    ap.parse_args(argv)

    print(
        "benchmark/evaluate.py is a Phase 2 deliverable (plan §12).\n"
        "It is not implemented in Phase 1. See docs/PHASE1_STATUS.md.\n\n"
        "To unblock it:\n"
        "  1. uv run shoal-fetch --brackishmot        # needs ~/.kaggle/kaggle.json\n"
        "  2. implement the 2x2 loop here using shoal.track.run_tracking(method=...)\n"
        "     and motmetrics / TrackEval for HOTA.\n",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
