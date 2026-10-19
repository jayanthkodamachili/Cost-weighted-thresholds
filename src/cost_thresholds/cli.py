"""Command line entry point.

Exit code 4 is the one worth having. A model can rank well, carry an unremarkable AUC, and
still cost more to act on than ignoring it entirely, and no other code in this scheme says
that. A pipeline that treats "the model works" as a boolean will never ask the question.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from cost_thresholds.config import CALIBRATORS, DISTORTIONS, ConfigError, ScenarioConfig
from cost_thresholds.report import Report, build_report

EXIT_USABLE = 0
EXIT_REGRET = 1
EXIT_BAD_CONFIG = 2
EXIT_INTERNAL_ERROR = 3
EXIT_WORSE_THAN_NOTHING = 4

REGRET_TOLERANCE = 0.02


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cost-threshold",
        description="Price the gap between a model's ranking and the cost of acting on it.",
    )
    parser.add_argument("--n-samples", type=int, default=40000)
    parser.add_argument("--base-rate-alpha", type=float, default=2.0)
    parser.add_argument("--base-rate-beta", type=float, default=8.0)
    parser.add_argument("--distortion", choices=DISTORTIONS, default="power")
    parser.add_argument("--distortion-strength", type=float, default=2.0)
    parser.add_argument("--cost-false-negative", type=float, default=10.0)
    parser.add_argument("--cost-false-positive", type=float, default=1.0)
    parser.add_argument("--calibrator", choices=CALIBRATORS, default="none")
    parser.add_argument("--calibration-frac", type=float, default=0.3)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260219)
    parser.add_argument(
        "--regret-tolerance",
        type=float,
        default=REGRET_TOLERANCE,
        help="regret above this share of the best cost exits 1",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ScenarioConfig:
    return ScenarioConfig(
        n_samples=args.n_samples,
        base_rate_alpha=args.base_rate_alpha,
        base_rate_beta=args.base_rate_beta,
        distortion=args.distortion,
        distortion_strength=args.distortion_strength,
        cost_false_negative=args.cost_false_negative,
        cost_false_positive=args.cost_false_positive,
        calibrator=args.calibrator,
        calibration_frac=args.calibration_frac,
        n_bins=args.n_bins,
        seed=args.seed,
    )


def render(report: Report) -> str:
    config, decision, curve = report.config, report.decision, report.reliability
    lines = [
        f"distortion {config.distortion} at strength {config.distortion_strength}, "
        f"cost ratio {config.cost_ratio:.0f} to 1",
        f"base rate {curve.base_rate:.4f}, mean predicted {curve.mean_score_overall:.4f}, "
        f"gap {curve.calibration_gap:+.4f}",
        "",
        f"{'reading':<26}{'value':>12}",
        "-" * 38,
        f"{'AUC':<26}{decision.auc:>12.6f}",
        f"{'expected calibration error':<26}{curve.expected_calibration_error:>12.6f}",
        f"{'Brier score':<26}{curve.brier:>12.6f}",
        "-" * 38,
        f"{'textbook threshold':<26}{decision.textbook_threshold:>12.4f}",
        f"{'cost there':<26}{decision.textbook_cost:>12.4f}",
        f"{'best threshold available':<26}{decision.best_threshold:>12.4f}",
        f"{'cost there':<26}{decision.best_cost:>12.4f}",
        f"{'regret':<26}{decision.regret:>12.4f}",
        f"{'cost of having no model':<26}{decision.trivial_cost:>12.4f}",
        "-" * 38,
    ]
    if not decision.beats_trivial:
        lines.append(
            "acting on this model at the textbook threshold costs more than not having it"
        )
    if report.repaired and report.calibrated_decision is not None:
        repaired = report.calibrated_decision
        lines.append(
            f"after {config.calibrator}: regret {repaired.regret:.4f} "
            f"(removed {report.regret_removed:+.4f}), AUC moved {report.auc_change:+.6f}"
        )
    return "\n".join(lines)


def exit_code_for(report: Report, tolerance: float) -> int:
    if not report.decision.beats_trivial:
        return EXIT_WORSE_THAN_NOTHING
    if report.decision.regret_share > tolerance:
        return EXIT_REGRET
    return EXIT_USABLE


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
    except ConfigError as exc:
        print(f"invalid scenario: {exc}", file=sys.stderr)
        return EXIT_BAD_CONFIG

    try:
        report = build_report(config)
    except Exception as exc:  # noqa: BLE001
        print(f"report failed to complete: {exc}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR

    print(json.dumps(report.as_dict(), indent=2) if args.json else render(report))
    return exit_code_for(report, args.regret_tolerance)


if __name__ == "__main__":
    raise SystemExit(main())
