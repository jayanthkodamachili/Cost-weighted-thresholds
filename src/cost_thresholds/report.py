"""Put the two readings side by side: what the ranking says, and what acting costs.

A report that printed only the decision numbers would be a better tool and a worse
argument. The AUC is here precisely so a reader can watch it stay still while everything
to its right moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cost_thresholds.calibration import Reliability, fit_calibrator, reliability
from cost_thresholds.config import ScenarioConfig
from cost_thresholds.decide import Decision, decide
from cost_thresholds.generate import Sample, generate, split


@dataclass(frozen=True)
class Report:
    """One scenario, read two ways, optionally after a repair."""

    config: ScenarioConfig
    decision: Decision
    reliability: Reliability
    calibrated_decision: Decision | None = None
    calibrated_reliability: Reliability | None = None

    @property
    def repaired(self) -> bool:
        return self.calibrated_decision is not None

    @property
    def auc_change(self) -> float:
        """How much the repair moved the ranking metric.

        Expected to be near zero for Platt, which is monotone and therefore cannot change
        a ranking at all, and slightly negative for isotonic, which can tie scores
        together and lose resolution. Published because a reader should be able to see
        that the repair is not quietly a different model.
        """
        if self.calibrated_decision is None:
            return 0.0
        return self.calibrated_decision.auc - self.decision.auc

    @property
    def regret_removed(self) -> float:
        if self.calibrated_decision is None:
            return 0.0
        return self.decision.regret - self.calibrated_decision.regret

    def as_dict(self) -> dict:
        payload = {
            "config": self.config.as_dict(),
            "decision": self.decision.as_dict(),
            "reliability": self.reliability.as_dict(),
            "repaired": self.repaired,
        }
        if self.calibrated_decision is not None:
            payload["calibrated_decision"] = self.calibrated_decision.as_dict()
            payload["auc_change"] = round(self.auc_change, 6)
            payload["regret_removed"] = round(self.regret_removed, 6)
        if self.calibrated_reliability is not None:
            payload["calibrated_reliability"] = self.calibrated_reliability.as_dict()
        return payload


def build_report(config: ScenarioConfig, sample: Sample | None = None) -> Report:
    """Run one scenario end to end.

    With no calibrator the whole sample is evaluated. With one, the sample is split and
    only the evaluation half is scored, so the repair never sees the rows it is judged on.
    The two paths therefore report on different row counts, which is stated in the output
    rather than quietly ignored.
    """
    if sample is None:
        sample = generate(config)

    if config.calibrator == "none":
        return Report(
            config=config,
            decision=decide(sample, config),
            reliability=reliability(sample.labels, sample.score, config.n_bins),
        )

    fit, evaluation = split(sample, config.calibration_frac, config.seed)
    calibrate = fit_calibrator(fit, config.calibrator)
    repaired_scores = np.asarray(calibrate(evaluation.score), dtype=float)

    return Report(
        config=config,
        decision=decide(evaluation, config),
        reliability=reliability(evaluation.labels, evaluation.score, config.n_bins),
        calibrated_decision=decide(evaluation, config, score=repaired_scores),
        calibrated_reliability=reliability(
            evaluation.labels, repaired_scores, config.n_bins
        ),
    )
