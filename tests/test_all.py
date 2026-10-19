"""The suite. Every test name states the property it pins.

Scenarios are small, because the published figures come from the experiments and a test
only needs a path exercised and a property asserted. The one exception is the invariance
test, which needs enough rows for AUC to be stable to six decimal places, and that is the
central claim of the repository so it earns the seconds.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from cost_thresholds import cli
from cost_thresholds.calibration import fit_calibrator, reliability
from cost_thresholds.config import DISTORTIONS, ConfigError, ScenarioConfig
from cost_thresholds.decide import CostMatrix, decide, sweep_thresholds, trivial_cost
from cost_thresholds.generate import Sample, distort, generate, split
from cost_thresholds.report import build_report

FAST = ["--n-samples", "4000", "--seed", "5"]


@pytest.fixture
def small_config() -> ScenarioConfig:
    return ScenarioConfig(n_samples=4000, seed=5)


@pytest.fixture
def small_sample(small_config) -> Sample:
    return generate(small_config)


# ----- config -----

def test_default_config_is_constructible():
    assert ScenarioConfig().distortion == "power"


def test_config_is_frozen(small_config):
    with pytest.raises(FrozenInstanceError):
        small_config.seed = 1  # type: ignore[misc]


def test_too_few_samples_is_rejected():
    with pytest.raises(ConfigError, match="n_samples"):
        ScenarioConfig(n_samples=50)


@pytest.mark.parametrize("alpha,beta", [(0.0, 1.0), (1.0, -2.0)])
def test_non_positive_beta_shapes_are_rejected(alpha, beta):
    with pytest.raises(ConfigError, match="beta shape"):
        ScenarioConfig(base_rate_alpha=alpha, base_rate_beta=beta)


def test_an_unknown_distortion_is_rejected():
    with pytest.raises(ConfigError, match="unknown distortion"):
        ScenarioConfig(distortion="vibes")


def test_an_unknown_calibrator_is_rejected():
    with pytest.raises(ConfigError, match="unknown calibrator"):
        ScenarioConfig(calibrator="magic")


def test_a_free_mistake_is_rejected():
    """A cost of zero makes the optimal threshold degenerate rather than interesting."""
    with pytest.raises(ConfigError, match="free mistake"):
        ScenarioConfig(cost_false_positive=0.0)


@pytest.mark.parametrize("frac", [0.0, 0.95])
def test_a_calibration_split_outside_its_range_is_rejected(frac):
    with pytest.raises(ConfigError, match="calibration_frac"):
        ScenarioConfig(calibration_frac=frac)


def test_too_few_bins_is_rejected():
    with pytest.raises(ConfigError, match="n_bins"):
        ScenarioConfig(n_bins=1)


def test_a_non_positive_distortion_strength_is_rejected():
    with pytest.raises(ConfigError, match="distortion_strength"):
        ScenarioConfig(distortion_strength=0.0)


def test_the_theoretical_threshold_follows_the_cost_ratio():
    config = ScenarioConfig(cost_false_negative=9.0, cost_false_positive=1.0)
    assert config.theoretical_threshold == pytest.approx(0.1)
    assert config.cost_ratio == pytest.approx(9.0)


def test_equal_costs_give_the_familiar_half_threshold():
    config = ScenarioConfig(cost_false_negative=1.0, cost_false_positive=1.0)
    assert config.theoretical_threshold == pytest.approx(0.5)


def test_as_dict_round_trips(small_config):
    assert ScenarioConfig(**small_config.as_dict()) == small_config


# ----- generation and the invariance claim -----

def test_generation_is_reproducible(small_config):
    np.testing.assert_array_equal(generate(small_config).score, generate(small_config).score)


def test_labels_are_drawn_from_the_true_probabilities():
    """The construction that makes exact calibration error possible."""
    config = ScenarioConfig(n_samples=60000, distortion="identity", seed=2)
    sample = generate(config)
    assert sample.base_rate == pytest.approx(sample.true_probability.mean(), abs=0.01)


@pytest.mark.parametrize("kind", DISTORTIONS)
def test_every_distortion_preserves_the_ranking_exactly(kind):
    """The central claim, asserted directly rather than through AUC.

    A strictly monotone map leaves the order of the scores untouched, so the rank
    correlation with the true probabilities must be exactly one.
    """
    probability = np.linspace(0.01, 0.99, 500)
    distorted = distort(probability, kind, strength=2.0)
    assert np.all(np.diff(distorted) > 0)


@pytest.mark.parametrize("kind", DISTORTIONS)
def test_auc_is_identical_under_every_distortion(kind):
    """The repository in one assertion. Six decimal places, not approximately."""
    config = ScenarioConfig(n_samples=40000, distortion="identity", seed=9)
    sample = generate(config)
    baseline = decide(sample, config).auc
    moved = decide(sample, config, score=distort(sample.score, kind, 2.0)).auc
    assert round(moved, 6) == round(baseline, 6)


def test_an_unknown_distortion_raises_at_the_dispatch():
    with pytest.raises(ValueError, match="unreachable"):
        distort(np.array([0.5]), "wishful", 1.0)


def test_the_power_distortion_pushes_scores_down():
    moved = distort(np.array([0.5]), "power", 2.0)
    assert moved[0] < 0.5


def test_overconfident_and_underconfident_move_in_opposite_directions():
    high = np.array([0.8])
    assert distort(high, "overconfident", 2.0)[0] > high[0]
    assert distort(high, "underconfident", 2.0)[0] < high[0]


def test_mismatched_array_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        Sample(np.array([1, 0]), np.array([0.5]), np.array([0.5, 0.5]))


def test_a_split_separates_the_rows(small_sample):
    fit, evaluation = split(small_sample, 0.3, seed=1)
    assert fit.n + evaluation.n == small_sample.n
    assert fit.n == pytest.approx(small_sample.n * 0.3, abs=2)


def test_a_split_that_leaves_nothing_usable_is_refused(small_sample):
    with pytest.raises(ValueError, match="too few rows"):
        split(small_sample, 0.00001, seed=1)


# ----- decisions -----

def test_the_optimal_threshold_comes_from_the_cost_matrix():
    assert CostMatrix(false_negative=19.0, false_positive=1.0).optimal_threshold == pytest.approx(
        0.05
    )


def test_expected_cost_counts_both_kinds_of_mistake():
    labels = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.1, 0.9, 0.1])
    costs = CostMatrix(false_negative=10.0, false_positive=1.0)
    # at 0.5: one missed positive and one acted-on negative
    assert costs.expected_cost(labels, score, 0.5) == pytest.approx((10.0 + 1.0) / 4)


def test_a_perfect_score_costs_nothing_at_the_right_threshold():
    labels = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.2])
    costs = CostMatrix(false_negative=10.0, false_positive=1.0)
    assert costs.expected_cost(labels, score, 0.5) == 0.0


def test_the_trivial_cost_is_the_cheaper_of_the_two_blanket_policies():
    labels = np.array([1] * 2 + [0] * 8)
    costs = CostMatrix(false_negative=10.0, false_positive=1.0)
    # act on all: 8 false positives at 1 => 0.8; act on none: 2 missed at 10 => 2.0
    assert trivial_cost(labels, costs) == pytest.approx(0.8)


def test_sweeping_returns_a_cost_for_every_threshold():
    labels = np.array([1, 0, 1, 0])
    thresholds, costs = sweep_thresholds(labels, np.array([0.6, 0.4, 0.7, 0.2]),
                                         CostMatrix(2.0, 1.0), grid=51)
    assert thresholds.size == costs.size == 51


def test_calibrated_scores_have_near_zero_regret():
    """The formula is correct when its assumption holds, which is why regret measures
    the assumption rather than the formula."""
    config = ScenarioConfig(n_samples=40000, distortion="identity", seed=4)
    assert decide(generate(config), config).regret < 0.02


def test_distorted_scores_have_large_regret():
    config = ScenarioConfig(n_samples=40000, distortion="power",
                            distortion_strength=2.0, seed=4)
    assert decide(generate(config), config).regret > 0.2


def test_a_distorted_model_can_cost_more_than_having_no_model():
    config = ScenarioConfig(n_samples=40000, distortion="power",
                            distortion_strength=2.0, seed=4)
    assert decide(generate(config), config).beats_trivial is False


def test_the_same_model_calibrated_does_beat_having_no_model():
    config = ScenarioConfig(n_samples=40000, distortion="identity", seed=4)
    assert decide(generate(config), config).beats_trivial is True


def test_regret_share_is_zero_when_the_best_cost_is_zero():
    labels = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.2])
    config = ScenarioConfig(n_samples=4000, cost_false_negative=1.0, cost_false_positive=1.0)
    decision = decide(Sample(labels, score, score), config)
    assert decision.best_cost == 0.0
    assert decision.regret_share == 0.0


# ----- calibration -----

def test_a_calibrated_model_has_a_small_calibration_gap():
    config = ScenarioConfig(n_samples=40000, distortion="identity", seed=6)
    sample = generate(config)
    assert abs(reliability(sample.labels, sample.score, 20).calibration_gap) < 0.01


def test_a_distorted_model_has_a_large_calibration_gap():
    config = ScenarioConfig(n_samples=40000, distortion="power",
                            distortion_strength=2.0, seed=6)
    sample = generate(config)
    assert abs(reliability(sample.labels, sample.score, 20).calibration_gap) > 0.08


def test_the_reliability_curve_covers_every_bin(small_sample):
    curve = reliability(small_sample.labels, small_sample.score, n_bins=10)
    assert len(curve.bin_centres) == 10
    assert sum(curve.bin_counts) == small_sample.n


def test_an_empty_bin_contributes_nothing_rather_than_crashing():
    """Equal-width bins leave gaps when scores cluster, which is the normal case here."""
    labels = np.array([1, 0, 1, 0])
    score = np.array([0.01, 0.02, 0.01, 0.02])
    curve = reliability(labels, score, n_bins=20)
    assert np.isfinite(curve.expected_calibration_error)


def test_the_no_op_calibrator_returns_its_input(small_sample):
    calibrate = fit_calibrator(small_sample, "none")
    np.testing.assert_array_equal(calibrate(small_sample.score), small_sample.score)


@pytest.mark.parametrize("kind", ("platt", "isotonic"))
def test_a_calibrator_reduces_regret_on_a_distorted_model(kind):
    config = ScenarioConfig(n_samples=40000, distortion="power", distortion_strength=2.0,
                            calibrator=kind, seed=7)
    report = build_report(config)
    assert report.regret_removed > 0.1


def test_platt_cannot_change_the_ranking():
    """It is a monotone map, so AUC must be untouched. Isotonic can tie scores and lose
    a little, which is published rather than hidden."""
    config = ScenarioConfig(n_samples=40000, distortion="power", calibrator="platt", seed=7)
    assert abs(build_report(config).auc_change) < 1e-9


def test_an_unknown_calibrator_raises_at_the_dispatch(small_sample):
    with pytest.raises(ValueError, match="unreachable"):
        fit_calibrator(small_sample, "telepathy")


def test_a_report_without_a_calibrator_reports_no_repair(small_config):
    report = build_report(small_config)
    assert report.repaired is False
    assert report.auc_change == 0.0
    assert report.regret_removed == 0.0


def test_a_calibrated_report_scores_only_the_held_out_half():
    """A calibrator judged on the rows it was fitted on reports an improvement that does
    not exist, which would be this repository making the error it is about."""
    config = ScenarioConfig(n_samples=4000, calibrator="isotonic",
                            calibration_frac=0.25, seed=8)
    report = build_report(config)
    assert sum(report.reliability.bin_counts) == pytest.approx(3000, abs=2)


def test_a_report_is_json_serialisable(small_config):
    json.dumps(build_report(small_config).as_dict())


# ----- cli -----

def test_parser_defaults_match_the_documented_scenario():
    args = cli.build_parser().parse_args([])
    assert args.distortion == "power"
    assert args.cost_false_negative == 10.0


def test_a_calibrated_model_exits_zero(capsys):
    code = cli.main([*FAST, "--n-samples", "40000", "--distortion", "identity"])
    capsys.readouterr()
    assert code == cli.EXIT_USABLE


def test_a_distorted_model_exits_four_and_says_so(capsys):
    code = cli.main([*FAST, "--n-samples", "40000", "--distortion", "power"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_WORSE_THAN_NOTHING
    assert "costs more than not having it" in out


def test_a_model_with_regret_but_still_useful_exits_one(capsys):
    """Between the two extremes: the thresholds are wrong and the model still helps."""
    code = cli.main([*FAST, "--n-samples", "40000", "--distortion", "power",
                     "--distortion-strength", "1.2", "--regret-tolerance", "0.001"])
    capsys.readouterr()
    assert code == cli.EXIT_REGRET


def test_an_impossible_scenario_exits_two_without_a_traceback(capsys):
    code = cli.main(["--n-samples", "10"])
    captured = capsys.readouterr()
    assert code == cli.EXIT_BAD_CONFIG
    assert "invalid scenario" in captured.err
    assert "Traceback" not in captured.err


def test_an_internal_failure_exits_three(monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_report", lambda _c: (_ for _ in ()).throw(RuntimeError("x")))
    code = cli.main(FAST)
    assert code == cli.EXIT_INTERNAL_ERROR
    assert "failed to complete" in capsys.readouterr().err


def test_every_exit_code_is_distinct():
    codes = {cli.EXIT_USABLE, cli.EXIT_REGRET, cli.EXIT_BAD_CONFIG,
             cli.EXIT_INTERNAL_ERROR, cli.EXIT_WORSE_THAN_NOTHING}
    assert len(codes) == 5


def test_the_table_puts_auc_next_to_the_costs(capsys):
    cli.main(FAST)
    out = capsys.readouterr().out
    assert "AUC" in out
    assert "cost of having no model" in out


def test_json_mode_is_parseable(capsys):
    cli.main([*FAST, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["seed"] == 5
    assert "regret" in payload["decision"]


def test_the_repair_line_appears_only_with_a_calibrator(capsys):
    cli.main([*FAST, "--calibrator", "platt"])
    assert "after platt" in capsys.readouterr().out


def test_an_unknown_distortion_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--distortion", "telepathy"])


def test_a_calibrated_report_serialises_the_repair_as_well():
    """as_dict has branches only a repaired report reaches, and the JSON output is what
    a pipeline would threshold on, so the repaired payload is pinned too."""
    config = ScenarioConfig(n_samples=4000, distortion="power",
                            calibrator="platt", seed=9)
    payload = build_report(config).as_dict()
    assert payload["repaired"] is True
    assert "calibrated_decision" in payload
    assert "calibrated_reliability" in payload
    assert "auc_change" in payload and "regret_removed" in payload
    json.dumps(payload)
