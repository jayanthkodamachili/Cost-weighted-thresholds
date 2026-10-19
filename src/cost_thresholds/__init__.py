"""Measure what a ranking metric cannot see.

AUC is invariant under every strictly monotone transformation of the scores, by
construction. That is usually described as a virtue, and it is, right up to the moment
somebody sets a threshold. A threshold is a statement about the probability a score
represents, and a transformation that leaves the ranking untouched can move every
probability without AUC changing in the sixth decimal place.

So a model can rank well, score well on the metric the team reports, and cost more to act
on than having no model at all. This measures that gap: the cost of the textbook
threshold, the cost of the best threshold available, and the difference between them.
"""

from cost_thresholds.config import ScenarioConfig
from cost_thresholds.decide import CostMatrix, Decision, decide
from cost_thresholds.report import Report, build_report

__all__ = [
    "CostMatrix", "Decision", "Report", "ScenarioConfig", "build_report", "decide",
]
__version__ = "0.1.0"
