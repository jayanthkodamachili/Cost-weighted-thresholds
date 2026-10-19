"""Draw labels from known probabilities, then distort the scores without touching the order.

A real model's scores come with an unknown true probability behind them, which makes the
central measurement here impossible: there is nothing to compare the scores against. So
the true probability is drawn first and the label is drawn from it, which means every
calibration error in this repository is known exactly rather than estimated.

Every distortion below is strictly monotone. That is not a convenience, it is the whole
construction: a strictly monotone transformation leaves every ranking metric unchanged by
definition, so any difference these produce is a difference ranking metrics cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cost_thresholds.config import ScenarioConfig

EPSILON = 1e-9


@dataclass(frozen=True)
class Sample:
    """Labels, the true probabilities that generated them, and the distorted scores."""

    labels: np.ndarray
    true_probability: np.ndarray
    score: np.ndarray

    def __post_init__(self) -> None:
        sizes = {self.labels.size, self.true_probability.size, self.score.size}
        if len(sizes) != 1:
            raise ValueError("labels, true probabilities and scores must be the same length")

    @property
    def base_rate(self) -> float:
        return float(self.labels.mean())

    @property
    def n(self) -> int:
        return int(self.labels.size)


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, EPSILON, 1 - EPSILON)
    return np.log(clipped / (1 - clipped))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def distort(probability: np.ndarray, kind: str, strength: float) -> np.ndarray:
    """Apply a strictly monotone transformation, so the ranking survives exactly.

    `overconfident` and `underconfident` are the two shapes that actually turn up. A
    model trained with too little regularisation pushes its scores towards zero and one;
    one trained with too much, or averaged over an ensemble, pulls them towards the base
    rate. Both leave the ranking untouched and both ruin a threshold.
    """
    if kind == "identity":
        return probability.copy()
    if kind == "power":
        return np.clip(probability, EPSILON, 1.0) ** strength
    if kind == "logit_shift":
        return _expit(_logit(probability) - strength)
    if kind == "overconfident":
        return _expit(_logit(probability) * strength)
    if kind == "underconfident":
        return _expit(_logit(probability) / strength)
    raise ValueError(f"unreachable: config validation should have rejected {kind!r}")


def generate(config: ScenarioConfig) -> Sample:
    """Draw true probabilities, draw labels from them, and distort the scores."""
    rng = np.random.default_rng(config.seed)
    true_probability = rng.beta(
        config.base_rate_alpha, config.base_rate_beta, size=config.n_samples
    )
    labels = (rng.random(config.n_samples) < true_probability).astype(np.int64)
    score = distort(true_probability, config.distortion, config.distortion_strength)
    return Sample(labels=labels, true_probability=true_probability, score=score)


def split(sample: Sample, frac: float, seed: int) -> tuple[Sample, Sample]:
    """Carve a calibration split off the evaluation split.

    A calibrator fitted and evaluated on the same rows reports an improvement that does
    not exist, which would be this repository making the error it is about.
    """
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(sample.n)
    cut = int(round(sample.n * frac))
    if cut < 2 or sample.n - cut < 2:
        raise ValueError("split leaves too few rows on one side to be useful")

    fit_idx, eval_idx = order[:cut], order[cut:]
    return (
        Sample(sample.labels[fit_idx], sample.true_probability[fit_idx], sample.score[fit_idx]),
        Sample(sample.labels[eval_idx], sample.true_probability[eval_idx],
               sample.score[eval_idx]),
    )
