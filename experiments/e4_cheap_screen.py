"""E4: is there a cheap check that predicts the expensive problem?

Computing regret needs a cost matrix and a threshold sweep. Computing the calibration gap
needs one mean and one base rate, and a team can run it on every model they have by
Thursday. This asks whether the cheap number ranks models the way the expensive one does,
across randomly drawn scenarios rather than the handful chosen for the other experiments.

The uncomfortable half is the third column. A screen that correlates well on average can
still pass models that are badly broken, and what matters operationally is not the
correlation but the share of genuinely bad models a cheap threshold lets through.
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

TRIALS = 300
N = 20000
SEED = 540_000
GAP_THRESHOLD = 0.02
BAD_REGRET_SHARE = 0.10


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, computed directly to keep the dependency list short."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denominator) if denominator else 0.0


def run() -> dict:
    rng = np.random.default_rng(SEED)
    rows = []
    for trial in range(TRIALS):
        config = ScenarioConfig(
            n_samples=N,
            base_rate_alpha=float(rng.uniform(1.0, 4.0)),
            base_rate_beta=float(rng.uniform(3.0, 12.0)),
            distortion=str(rng.choice(DISTORTIONS)),
            distortion_strength=float(rng.uniform(1.0, 3.0)),
            cost_false_negative=float(rng.choice([2, 5, 10, 25, 50])),
            cost_false_positive=1.0,
            seed=SEED + trial,
        )
        sample = generate(config)
        decision = decide(sample, config)
        curve = reliability(sample.labels, sample.score, config.n_bins)
        rows.append({
            "abs_gap": abs(curve.calibration_gap),
            "ece": curve.expected_calibration_error,
            "brier": curve.brier,
            "auc": decision.auc,
            "regret_share": decision.regret_share,
        })

    regret = np.array([r["regret_share"] for r in rows])
    bad = regret > BAD_REGRET_SHARE

    screens = []
    for name in ("abs_gap", "ece", "brier", "auc"):
        values = np.array([r[name] for r in rows])
        screens.append({
            "screen": name,
            "spearman_with_regret": spearman(values, regret),
        })

    passed_by_gap = np.array([r["abs_gap"] for r in rows]) <= GAP_THRESHOLD
    missed = int(np.sum(bad & passed_by_gap))
    flagged_fine = int(np.sum(~bad & ~passed_by_gap))

    return {
        "experiment": "e4_cheap_screen",
        "question": "Does a cheap calibration check predict an expensive decision cost?",
        "trials": TRIALS,
        "gap_threshold": GAP_THRESHOLD,
        "bad_regret_share": BAD_REGRET_SHARE,
        "screens": screens,
        "bad_models": int(bad.sum()),
        "passed_by_gap": int(passed_by_gap.sum()),
        "bad_models_missed": missed,
        "miss_rate": missed / max(1, int(bad.sum())),
        "fine_models_flagged": flagged_fine,
        "false_flag_rate": flagged_fine / max(1, int((~bad).sum())),
        "best_screen": max(screens, key=lambda s: abs(s["spearman_with_regret"]))["screen"],
        "auc_spearman": next(
            s["spearman_with_regret"] for s in screens if s["screen"] == "auc"
        ),
    }


def main() -> None:
    payload = run()
    rows = [
        [s["screen"], f"{s['spearman_with_regret']:+.4f}"] for s in payload["screens"]
    ]
    print(f"E4 cheap screens against decision regret, {TRIALS} random scenarios\n")
    print(table(["screen", "rank correlation with regret"], rows, [12, 30]))
    print()
    print(
        f"{payload['bad_models']} of {payload['trials']} models have regret above "
        f"{payload['bad_regret_share']:.0%} of the best achievable cost. A gap screen at "
        f"{payload['gap_threshold']} lets {payload['bad_models_missed']} of them through "
        f"({payload['miss_rate']:.1%}) and flags "
        f"{payload['fine_models_flagged']} that were fine."
    )
    path = write("e4_cheap_screen", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
