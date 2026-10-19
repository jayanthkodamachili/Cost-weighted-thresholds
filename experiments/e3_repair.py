"""E3: what a repair buys, and what it costs.

Calibration is fixable, and the fix is not free: it consumes rows that cannot then be used
for anything else, and it has to be refitted whenever the model is. This measures the
recovery against the size of the split, so the trade can be read rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import table, write  # noqa: E402

from cost_thresholds.config import ScenarioConfig  # noqa: E402
from cost_thresholds.report import build_report  # noqa: E402

SEEDS = (20260219, 11, 404, 2027)
FRACTIONS = (0.02, 0.05, 0.10, 0.25, 0.50)
CALIBRATORS = ("platt", "isotonic")
N, STRENGTH = 60000, 2.0


def run() -> dict:
    cells = []
    for calibrator in CALIBRATORS:
        for frac in FRACTIONS:
            before, after, auc_moves = [], [], []
            for seed in SEEDS:
                config = ScenarioConfig(
                    n_samples=N, distortion="power", distortion_strength=STRENGTH,
                    calibrator=calibrator, calibration_frac=frac, seed=seed,
                )
                report = build_report(config)
                before.append(report.decision.regret)
                after.append(report.calibrated_decision.regret)
                auc_moves.append(report.auc_change)
            cells.append({
                "calibrator": calibrator,
                "calibration_frac": frac,
                "rows_used": int(N * frac),
                "regret_before": float(np.mean(before)),
                "regret_after": float(np.mean(after)),
                "recovered": float(np.mean(before)) - float(np.mean(after)),
                "auc_change": float(np.mean(auc_moves)),
            })

    smallest = min(cells, key=lambda c: c["calibration_frac"])
    best = min(cells, key=lambda c: c["regret_after"])
    platt = [c for c in cells if c["calibrator"] == "platt"]
    isotonic = [c for c in cells if c["calibrator"] == "isotonic"]
    return {
        "experiment": "e3_repair",
        "question": "What does repairing calibration recover, and what does it consume?",
        "seeds": list(SEEDS),
        "cells": cells,
        "smallest_frac": smallest["calibration_frac"],
        "smallest_rows": smallest["rows_used"],
        "smallest_recovered": smallest["recovered"],
        "smallest_regret_after": smallest["regret_after"],
        "best_calibrator": best["calibrator"],
        "best_regret_after": best["regret_after"],
        "platt_mean_auc_change": float(np.mean([c["auc_change"] for c in platt])),
        "isotonic_mean_auc_change": float(np.mean([c["auc_change"] for c in isotonic])),
    }


def main() -> None:
    payload = run()
    rows = [
        [
            c["calibrator"],
            f"{c['calibration_frac']:.0%}",
            f"{c['rows_used']}",
            f"{c['regret_before']:.4f}",
            f"{c['regret_after']:.4f}",
            f"{c['auc_change']:+.6f}",
        ]
        for c in payload["cells"]
    ]
    print(f"E3 what a repair recovers, {len(SEEDS)} seeds, power distortion\n")
    print(table(["calibrator", "split", "rows", "regret before", "after", "AUC moved"],
                rows, [12, 8, 8, 15, 10, 12]))
    print()
    print(
        f"even {payload['smallest_rows']} held-out rows recover "
        f"{payload['smallest_recovered']:.4f} of the regret, leaving "
        f"{payload['smallest_regret_after']:.4f}"
    )
    path = write("e3_repair", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
