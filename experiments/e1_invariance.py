"""E1: the headline. AUC does not move; the cost of acting moves a lot.

Five transformations of one set of scores. Each is strictly monotone, so each leaves the
ranking exactly as it was, and a ranking metric reports the same number for all of them by
construction. What changes is the probability each score represents, and therefore what a
threshold means, and therefore what acting on the model costs.

Averaged over seeds, because the cost of a single draw wobbles and the claim does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import table, write  # noqa: E402

from cost_thresholds.calibration import reliability  # noqa: E402
from cost_thresholds.config import DISTORTIONS, ScenarioConfig  # noqa: E402
from cost_thresholds.decide import decide  # noqa: E402
from cost_thresholds.generate import generate  # noqa: E402

SEEDS = (20260219, 11, 404, 2027, 90210, 13)
N, STRENGTH = 60000, 2.0


def run() -> dict:
    cells = []
    for kind in DISTORTIONS:
        aucs, regrets, costs, trivials, gaps, briers, beats = [], [], [], [], [], [], []
        for seed in SEEDS:
            config = ScenarioConfig(
                n_samples=N, distortion=kind, distortion_strength=STRENGTH, seed=seed
            )
            sample = generate(config)
            decision = decide(sample, config)
            curve = reliability(sample.labels, sample.score, config.n_bins)
            aucs.append(decision.auc)
            regrets.append(decision.regret)
            costs.append(decision.textbook_cost)
            trivials.append(decision.trivial_cost)
            gaps.append(curve.calibration_gap)
            briers.append(curve.brier)
            beats.append(decision.beats_trivial)
        cells.append({
            "distortion": kind,
            "auc": float(np.mean(aucs)),
            "calibration_gap": float(np.mean(gaps)),
            "brier": float(np.mean(briers)),
            "textbook_cost": float(np.mean(costs)),
            "trivial_cost": float(np.mean(trivials)),
            "regret": float(np.mean(regrets)),
            "beats_trivial_seeds": sum(beats),
            "seeds": len(SEEDS),
        })

    aucs = [c["auc"] for c in cells]
    identity = next(c for c in cells if c["distortion"] == "identity")
    worst = max(cells, key=lambda c: c["regret"])
    return {
        "experiment": "e1_invariance",
        "question": "Does a ranking metric see what a monotone distortion does to cost?",
        "seeds": list(SEEDS),
        "strength": STRENGTH,
        "cells": cells,
        "auc_spread": max(aucs) - min(aucs),
        "shared_auc": identity["auc"],
        "calibrated_regret": identity["regret"],
        "worst_distortion": worst["distortion"],
        "worst_regret": worst["regret"],
        "regret_multiple": worst["regret"] / identity["regret"],
        "distortions_losing_to_nothing": sum(
            1 for c in cells if c["beats_trivial_seeds"] == 0
        ),
        "distortion_count": len(cells),
    }


def main() -> None:
    payload = run()
    rows = [
        [
            c["distortion"],
            f"{c['auc']:.6f}",
            f"{c['calibration_gap']:+.4f}",
            f"{c['textbook_cost']:.4f}",
            f"{c['regret']:.4f}",
            f"{c['beats_trivial_seeds']}/{c['seeds']}",
        ]
        for c in payload["cells"]
    ]
    print(f"E1 one ranking, five probability scales, {len(SEEDS)} seeds\n")
    print(table(
        ["distortion", "AUC", "calib gap", "cost", "regret", "beats nothing"],
        rows, [17, 11, 12, 10, 10, 15],
    ))
    print()
    print(
        f"AUC varies by {payload['auc_spread']:.8f} across the table while regret goes "
        f"from {payload['calibrated_regret']:.4f} to {payload['worst_regret']:.4f}, "
        f"a factor of {payload['regret_multiple']:.0f}"
    )
    path = write("e1_invariance", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
