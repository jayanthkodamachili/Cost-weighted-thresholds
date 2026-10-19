"""The frozen description of a scenario, including the cost matrix.

The cost matrix is in the config rather than passed to the reporting function because it
is not a presentation choice. Which threshold is correct, and therefore whether a model is
usable at all, is a function of what the two kinds of mistake cost. A tool that treats
that as an afterthought is a tool that will be run with the defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

DISTORTIONS = ("identity", "power", "logit_shift", "overconfident", "underconfident")
CALIBRATORS = ("none", "platt", "isotonic")


class ConfigError(ValueError):
    """Raised when a scenario cannot be constructed, as opposed to merely being awkward."""


@dataclass(frozen=True)
class ScenarioConfig:
    """One fully specified scenario.

    Attributes:
        n_samples: rows drawn.
        base_rate_alpha: first beta shape for the latent true probability.
        base_rate_beta: second beta shape. Together these set both the base rate and how
            separable the problem is, which is deliberate: a tool that let you dial
            separability independently of prevalence would be modelling something that
            does not happen.
        distortion: which monotone transformation is applied to the true probabilities.
            Every option preserves the ranking exactly, which is the point.
        distortion_strength: how hard the transformation pushes.
        cost_false_negative: cost of missing a positive.
        cost_false_positive: cost of acting on a negative.
        calibrator: optional repair fitted on a held-out split.
        calibration_frac: share of rows reserved for fitting the calibrator.
        n_bins: bins used by the reliability curve and the calibration error.
        seed: controls generation and every split.
    """

    n_samples: int = 40000
    base_rate_alpha: float = 2.0
    base_rate_beta: float = 8.0
    distortion: str = "power"
    distortion_strength: float = 2.0
    cost_false_negative: float = 10.0
    cost_false_positive: float = 1.0
    calibrator: str = "none"
    calibration_frac: float = 0.3
    n_bins: int = 20
    seed: int = 20260219

    def __post_init__(self) -> None:
        if self.n_samples < 200:
            raise ConfigError("n_samples must be at least 200 to estimate a reliability curve")
        if self.base_rate_alpha <= 0 or self.base_rate_beta <= 0:
            raise ConfigError("beta shape parameters must be positive")
        if self.distortion not in DISTORTIONS:
            raise ConfigError(f"unknown distortion {self.distortion!r}, expected {DISTORTIONS}")
        if self.calibrator not in CALIBRATORS:
            raise ConfigError(f"unknown calibrator {self.calibrator!r}, expected {CALIBRATORS}")
        if self.cost_false_negative <= 0 or self.cost_false_positive <= 0:
            raise ConfigError("both costs must be positive; a free mistake has no threshold")
        if not 0.0 < self.calibration_frac < 0.9:
            raise ConfigError("calibration_frac must sit strictly between 0 and 0.9")
        if self.n_bins < 2:
            raise ConfigError("n_bins must be at least 2")
        if self.distortion_strength <= 0:
            raise ConfigError("distortion_strength must be positive")

    @property
    def theoretical_threshold(self) -> float:
        """The threshold that minimises expected cost, if the probabilities are honest.

        Act when p * C_fn exceeds (1 - p) * C_fp, which rearranges to p above
        C_fp / (C_fp + C_fn). Every textbook derives this and every derivation assumes
        calibration, which is the assumption this repository is about.
        """
        return self.cost_false_positive / (
            self.cost_false_positive + self.cost_false_negative
        )

    @property
    def cost_ratio(self) -> float:
        return self.cost_false_negative / self.cost_false_positive

    def as_dict(self) -> dict:
        return asdict(self)
