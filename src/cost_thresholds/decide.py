"""Costs, thresholds and regret: the quantities a ranking metric cannot produce.

Regret is the measurement this repository is built around. It is the extra cost paid by
using the threshold the textbook formula gives, compared with the best threshold available
for these scores. On calibrated probabilities it is near zero, because the formula is
correct there. On distorted ones it is whatever the distortion is worth, and that number
exists whether or not anybody computes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from cost_thresholds.config import ScenarioConfig
from cost_thresholds.generate import Sample


@dataclass(frozen=True)
class CostMatrix:
    """What the two kinds of mistake cost. Correct decisions cost nothing."""

    false_negative: float
    false_positive: float

    @property
    def optimal_threshold(self) -> float:
        return self.false_positive / (self.false_positive + self.false_negative)

    def expected_cost(self, labels: np.ndarray, score: np.ndarray, threshold: float) -> float:
        """Mean cost per row of acting on these scores at this threshold."""
        act = score >= threshold
        false_negatives = int(np.sum((~act) & (labels == 1)))
        false_positives = int(np.sum(act & (labels == 0)))
        total = (
            self.false_negative * false_negatives + self.false_positive * false_positives
        )
        return float(total / labels.size)


@dataclass(frozen=True)
class Decision:
    """A full reading of one set of scores against one cost matrix."""

    auc: float
    textbook_threshold: float
    textbook_cost: float
    best_threshold: float
    best_cost: float
    trivial_cost: float

    @property
    def regret(self) -> float:
        """Extra cost paid for trusting the formula on these scores."""
        return self.textbook_cost - self.best_cost

    @property
    def regret_share(self) -> float:
        """Regret as a share of the best achievable cost."""
        return self.regret / self.best_cost if self.best_cost > 0 else 0.0

    @property
    def beats_trivial(self) -> bool:
        """Whether acting on the model at the textbook threshold beats not having it.

        The trivial policy is whichever of act-on-everything or act-on-nothing is cheaper.
        A model can have a respectable AUC and still lose to that, which is the finding
        most likely to change what somebody does on Monday.
        """
        return self.textbook_cost < self.trivial_cost

    def as_dict(self) -> dict:
        return {
            "auc": round(self.auc, 6),
            "textbook_threshold": round(self.textbook_threshold, 6),
            "textbook_cost": round(self.textbook_cost, 6),
            "best_threshold": round(self.best_threshold, 6),
            "best_cost": round(self.best_cost, 6),
            "trivial_cost": round(self.trivial_cost, 6),
            "regret": round(self.regret, 6),
            "regret_share": round(self.regret_share, 6),
            "beats_trivial": self.beats_trivial,
        }


def sweep_thresholds(
    labels: np.ndarray, score: np.ndarray, costs: CostMatrix, grid: int = 999
) -> tuple[np.ndarray, np.ndarray]:
    """Expected cost across a grid of thresholds.

    A grid rather than every distinct score value. Sweeping unique scores is exact and
    costs time proportional to the sample, and the published figures are means over many
    scenarios where the grid's resolution is far below the sampling error already present.
    ADR-002 carries the comparison that settled it.
    """
    thresholds = np.linspace(0.0, 1.0, grid)
    costs_at = np.array([costs.expected_cost(labels, score, t) for t in thresholds])
    return thresholds, costs_at


def trivial_cost(labels: np.ndarray, costs: CostMatrix) -> float:
    """The cheaper of acting on everything and acting on nothing."""
    positives = int(np.sum(labels == 1))
    negatives = int(labels.size - positives)
    act_on_all = costs.false_positive * negatives / labels.size
    act_on_none = costs.false_negative * positives / labels.size
    return float(min(act_on_all, act_on_none))


def decide(sample: Sample, config: ScenarioConfig, score: np.ndarray | None = None) -> Decision:
    """Read one set of scores: its AUC, its textbook cost, and the cost it could have had."""
    values = sample.score if score is None else score
    costs = CostMatrix(config.cost_false_negative, config.cost_false_positive)

    thresholds, swept = sweep_thresholds(sample.labels, values, costs)
    best_index = int(np.argmin(swept))

    return Decision(
        auc=float(roc_auc_score(sample.labels, values)),
        textbook_threshold=costs.optimal_threshold,
        textbook_cost=costs.expected_cost(sample.labels, values, costs.optimal_threshold),
        best_threshold=float(thresholds[best_index]),
        best_cost=float(swept[best_index]),
        trivial_cost=trivial_cost(sample.labels, costs),
    )
