"""E2: how the damage scales with how asymmetric the costs are.

The textbook threshold moves towards zero as a missed positive gets more expensive
relative to a false alarm. A distortion that squashes scores downwards moves the whole
score distribution the same way. The question is whether those two movements cancel or
compound, and the answer decides whether this problem is worse for the teams with the most
asymmetric costs, which is to say fraud, safety and medicine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import table, write  # noqa: E402

from cost_thresholds.config import ScenarioConfig  # noqa: E402
from cost_thresholds.decide import decide  # noqa: E402
from cost_thresholds.generate import generate  # noqa: E402

SEEDS = (20260219, 11, 404, 2027)
RATIOS = (1, 2, 5, 10, 50, 200)
N, STRENGTH = 60000, 2.0


def run() -> dict:
    cells = []
    for ratio in RATIOS:
        regrets, shares, beats = [], [], []
        for seed in SEEDS:
            config = ScenarioConfig(
                n_samples=N, distortion="power", distortion_strength=STRENGTH,
                cost_false_negative=float(ratio), cost_false_positive=1.0, seed=seed,
            )
            decision = decide(generate(config), config)
            regrets.append(decision.regret)
            shares.append(decision.regret_share)
            beats.append(decision.beats_trivial)
        cells.append({
            "cost_ratio": ratio,
            "textbook_threshold": 1.0 / (1.0 + ratio),
            "regret": float(np.mean(regrets)),
            "regret_share": float(np.mean(shares)),
            "beats_trivial_seeds": sum(beats),
            "seeds": len(SEEDS),
        })

    symmetric = cells[0]
    worst = max(cells, key=lambda c: c["regret_share"])
    return {
        "experiment": "e2_cost_ratio",
        "question": "Is this worse for teams whose two mistakes cost very different amounts?",
        "seeds": list(SEEDS),
        "cells": cells,
        "symmetric_regret_share": symmetric["regret_share"],
        "worst_ratio": worst["cost_ratio"],
        "worst_regret_share": worst["regret_share"],
        "share_multiple": worst["regret_share"] / symmetric["regret_share"]
        if symmetric["regret_share"] > 0 else float("inf"),
        "ratios_losing_to_nothing": sum(1 for c in cells if c["beats_trivial_seeds"] == 0),
        "ratio_count": len(cells),
    }


def main() -> None:
    payload = run()
    rows = [
        [
            f"{c['cost_ratio']} to 1",
            f"{c['textbook_threshold']:.4f}",
            f"{c['regret']:.4f}",
            f"{c['regret_share']:.3f}",
            f"{c['beats_trivial_seeds']}/{c['seeds']}",
        ]
        for c in payload["cells"]
    ]
    print(f"E2 regret by cost asymmetry, power distortion, {len(SEEDS)} seeds\n")
    print(table(["cost ratio", "textbook t", "regret", "share", "beats nothing"],
                rows, [14, 12, 10, 9, 15]))
    print()
    print(
        f"regret as a share of the best achievable cost goes from "
        f"{payload['symmetric_regret_share']:.3f} at symmetric costs to "
        f"{payload['worst_regret_share']:.3f} at {payload['worst_ratio']} to 1"
    )
    path = write("e2_cost_ratio", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
