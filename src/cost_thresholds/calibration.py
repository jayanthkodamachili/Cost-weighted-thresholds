"""Calibration measurement, and the two repairs people reach for.

The measurement and the repair are kept apart on purpose. Measuring calibration needs no
held-out data and tells you whether the thresholds you are using mean anything. Repairing
it consumes data you could have trained on, and this repository measures that price rather
than assuming it is small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from cost_thresholds.generate import EPSILON, Sample, _logit


@dataclass(frozen=True)
class Reliability:
    """A binned reliability curve and the scalar summaries drawn from it."""

    bin_centres: tuple[float, ...]
    bin_counts: tuple[int, ...]
    mean_score: tuple[float, ...]
    observed_rate: tuple[float, ...]
    expected_calibration_error: float
    brier: float
    mean_score_overall: float
    base_rate: float

    @property
    def calibration_gap(self) -> float:
        """Mean predicted probability minus the base rate.

        The cheapest possible calibration check, computable without binning anything: a
        model whose average prediction does not match how often the event happens is
        miscalibrated, and no amount of ranking skill fixes that.
        """
        return self.mean_score_overall - self.base_rate

    def as_dict(self) -> dict:
        return {
            "bin_centres": [round(v, 6) for v in self.bin_centres],
            "bin_counts": list(self.bin_counts),
            "mean_score": [round(v, 6) for v in self.mean_score],
            "observed_rate": [round(v, 6) for v in self.observed_rate],
            "expected_calibration_error": round(self.expected_calibration_error, 6),
            "brier": round(self.brier, 6),
            "mean_score_overall": round(self.mean_score_overall, 6),
            "base_rate": round(self.base_rate, 6),
            "calibration_gap": round(self.calibration_gap, 6),
        }


def reliability(
    labels: np.ndarray, score: np.ndarray, n_bins: int = 20
) -> Reliability:
    """Bin by predicted probability and compare each bin's mean score to its hit rate.

    Equal-width bins rather than equal-count. Equal-count bins hide the thing being looked
    for: a model that squashes everything into a narrow band gets bins that all look
    locally fine while the band itself sits in the wrong place.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(score, edges[1:-1], right=False), 0, n_bins - 1)

    centres, counts, means, rates = [], [], [], []
    weighted_error = 0.0
    for b in range(n_bins):
        mask = index == b
        count = int(np.sum(mask))
        centres.append(float((edges[b] + edges[b + 1]) / 2))
        counts.append(count)
        if count == 0:
            means.append(0.0)
            rates.append(0.0)
            continue
        mean_score = float(score[mask].mean())
        observed = float(labels[mask].mean())
        means.append(mean_score)
        rates.append(observed)
        weighted_error += count * abs(mean_score - observed)

    return Reliability(
        bin_centres=tuple(centres),
        bin_counts=tuple(counts),
        mean_score=tuple(means),
        observed_rate=tuple(rates),
        expected_calibration_error=float(weighted_error / labels.size),
        brier=float(np.mean((score - labels) ** 2)),
        mean_score_overall=float(score.mean()),
        base_rate=float(labels.mean()),
    )


def fit_calibrator(fit: Sample, kind: str):
    """Fit a repair on held-out rows, and return a function mapping scores to probabilities."""
    if kind == "none":
        return lambda s: s
    if kind == "platt":
        model = LogisticRegression()
        model.fit(_logit(fit.score).reshape(-1, 1), fit.labels)
        return lambda s: model.predict_proba(_logit(s).reshape(-1, 1))[:, 1]
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(fit.score, fit.labels)
        return lambda s: np.clip(model.predict(s), EPSILON, 1 - EPSILON)
    raise ValueError(f"unreachable: config validation should have rejected {kind!r}")
