"""Re-measure every published figure and write docs/metrics.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = ROOT / "docs" / "experiments"
METRICS_PATH = ROOT / "docs" / "metrics.json"

EXPERIMENTS = ("e1_invariance", "e2_cost_ratio", "e3_repair", "e4_cheap_screen")
CHECKED_DOCUMENTS = ("README.md", "docs/adr/ADR-005-what-this-cannot-measure.md")
TABLE_ANCHOR = ["| {} |"]

ANCHORS: dict[str, list[str]] = {
    "tests_total": ["{} tests"],
    "coverage_line_pct": ["{} percent line coverage", "{}% line coverage"],
    "shared_auc": ["all report {}", "auc of {}"],
    "auc_spread": ["varies by {}", "spread of {}"],
    "calibrated_regret": ["regret of {}", "from {} to"],
    "worst_regret": ["to {}", "worst is {}"],
    "regret_multiple": ["factor of {}", "{} times"],
    "losing_distortions": ["{} of the five", "{} of 5 distortions"],
    "symmetric_share": ["{} at symmetric", "share of {} when"],
    "worst_ratio_share": ["{} at fifty", "reaches {}"],
    "worst_ratio": ["at {} to 1", "{} to one"],
    "ratios_losing": ["{} of 6 cost", "{} of the six"],
    "regret_before_repair": ["of the {} available", "{} of regret"],
    "cheap_rows": ["{} held-out rows", "just {} rows"],
    "cheap_recovered": ["recover {}", "removes {}"],
    "cheap_after": ["leaving {}", "down to {}"],
    "platt_auc_change": ["exactly {}", "moves auc by {}"],
    "isotonic_auc_change": ["costs {}", "moves it {}"],
    "gap_correlation": ["correlates {}", "{} with regret"],
    "auc_correlation": ["manages {}", "correlates {}"],
    "bad_models": ["{} of 300", "{} genuinely bad"],
    "missed_by_screen": ["lets {} through", "misses {}"],
    "miss_rate": ["{} of them", "miss rate of {}"],
    "false_flags": ["flags {} that", "{} models that were fine"],
    "trials": ["{} random scenarios", "across {} scenarios"],
}


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, check=False, **kwargs)


def measure_suite(skip_tests: bool) -> dict:
    junit, cov = ROOT / "junit.xml", ROOT / "coverage.json"
    if skip_tests:
        missing = [p.name for p in (junit, cov) if not p.exists()]
        if missing:
            raise SystemExit(f"--skip-tests given but {', '.join(missing)} not found")
    else:
        result = run([sys.executable, "-m", "pytest", f"--junitxml={junit}",
                      "--cov=cost_thresholds", "--cov-report=json", "-q"],
                     capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-4000:], file=sys.stderr)
            raise SystemExit("test suite failed, refusing to publish metrics")
    suite = ET.parse(junit).getroot()
    node = suite if suite.tag == "testsuite" else suite.find("testsuite")
    total = int(node.get("tests", 0)) - int(node.get("skipped", 0))
    return {"tests_total": total,
            "coverage_line_pct": round(
                json.loads(cov.read_text())["totals"]["percent_covered"], 1)}


def ensure_experiments(rerun: bool) -> dict[str, dict]:
    payloads = {}
    for name in EXPERIMENTS:
        path = EXPERIMENT_DIR / f"{name}.json"
        if rerun or not path.exists():
            print(f"running experiment {name}")
            if run([sys.executable, str(ROOT / "experiments" / f"{name}.py")]).returncode:
                raise SystemExit(f"experiment {name} failed")
        payloads[name] = json.loads(path.read_text())
    return payloads


def derive(suite: dict, exp: dict[str, dict]) -> dict:
    e1, e2 = exp["e1_invariance"], exp["e2_cost_ratio"]
    e3, e4 = exp["e3_repair"], exp["e4_cheap_screen"]
    gap = next(s for s in e4["screens"] if s["screen"] == "abs_gap")

    metrics: dict = {
        "tests_total": suite["tests_total"],
        "coverage_line_pct": suite["coverage_line_pct"],
        "shared_auc": round(e1["shared_auc"], 6),
        "auc_spread": round(e1["auc_spread"], 8),
        "calibrated_regret": round(e1["calibrated_regret"], 4),
        "worst_regret": round(e1["worst_regret"], 4),
        "regret_multiple": round(e1["regret_multiple"], 0),
        "losing_distortions": e1["distortions_losing_to_nothing"],
        "symmetric_share": round(e2["symmetric_regret_share"], 3),
        "worst_ratio_share": round(e2["worst_regret_share"], 3),
        "worst_ratio": e2["worst_ratio"],
        "ratios_losing": e2["ratios_losing_to_nothing"],
        "regret_before_repair": round(e3["cells"][0]["regret_before"], 4),
        "cheap_rows": e3["smallest_rows"],
        "cheap_recovered": round(e3["smallest_recovered"], 4),
        "cheap_after": round(e3["smallest_regret_after"], 4),
        "platt_auc_change": round(e3["platt_mean_auc_change"], 6),
        "isotonic_auc_change": round(e3["isotonic_mean_auc_change"], 6),
        "gap_correlation": round(gap["spearman_with_regret"], 4),
        "auc_correlation": round(e4["auc_spearman"], 4),
        "bad_models": e4["bad_models"],
        "missed_by_screen": e4["bad_models_missed"],
        "miss_rate": round(e4["miss_rate"], 3),
        "false_flags": e4["fine_models_flagged"],
        "trials": e4["trials"],
    }

    for cell in e1["cells"]:
        k = cell["distortion"]
        metrics[f"e1_{k}_auc"] = round(cell["auc"], 6)
        metrics[f"e1_{k}_gap"] = round(cell["calibration_gap"], 4)
        metrics[f"e1_{k}_cost"] = round(cell["textbook_cost"], 4)
        metrics[f"e1_{k}_regret"] = round(cell["regret"], 4)
        for s in ("auc", "gap", "cost", "regret"):
            ANCHORS.setdefault(f"e1_{k}_{s}", TABLE_ANCHOR)

    for cell in e2["cells"]:
        k = cell["cost_ratio"]
        metrics[f"e2_{k}_t"] = round(cell["textbook_threshold"], 4)
        metrics[f"e2_{k}_regret"] = round(cell["regret"], 4)
        metrics[f"e2_{k}_share"] = round(cell["regret_share"], 3)
        for s in ("t", "regret", "share"):
            ANCHORS.setdefault(f"e2_{k}_{s}", TABLE_ANCHOR)

    for cell in e3["cells"]:
        k = f"{cell['calibrator']}_{int(cell['calibration_frac'] * 100)}"
        metrics[f"e3_{k}_after"] = round(cell["regret_after"], 4)
        ANCHORS.setdefault(f"e3_{k}_after", TABLE_ANCHOR)

    for screen in e4["screens"]:
        metrics[f"e4_{screen['screen']}_rho"] = round(screen["spearman_with_regret"], 4)
        ANCHORS.setdefault(f"e4_{screen['screen']}_rho", TABLE_ANCHOR)

    return metrics


SIGNED = ("_gap", "_rho", "_change", "correlation")
SIX_PLACE = ("_auc", "shared_auc", "_change")
THREE_PLACE = ("_share", "miss_rate", "symmetric_share", "worst_ratio_share")


def display_strings(metrics: dict) -> dict[str, str]:
    """Formatting declared per metric, because one figure rendering two ways defeats the
    anchor mechanism."""
    out = {}
    for key, value in metrics.items():
        if isinstance(value, int):
            out[key] = str(value)
        elif key == "coverage_line_pct":
            out[key] = f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"
        elif key == "auc_spread":
            out[key] = f"{value:.8f}"
        elif key == "regret_multiple":
            out[key] = f"{value:.0f}"
        elif any(key.endswith(s) or key == s for s in THREE_PLACE):
            out[key] = f"{value:.3f}"
        elif any(key.endswith(s) or key == s for s in SIX_PLACE):
            out[key] = f"{value:+.6f}" if "change" in key else f"{value:.6f}"
        elif any(key.endswith(s) or key == s for s in SIGNED):
            out[key] = f"{value:+.4f}"
        else:
            out[key] = f"{value:.4f}"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-measure every published figure.")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    suite = measure_suite(skip_tests=args.skip_tests)
    metrics = derive(suite, ensure_experiments(rerun=not args.reuse))
    METRICS_PATH.write_text(json.dumps({
        "metrics": metrics, "display": display_strings(metrics), "anchors": ANCHORS,
        "checked_documents": list(CHECKED_DOCUMENTS),
        "note": "every value here is produced by running the suite and the four experiments",
    }, indent=2) + "\n")
    print(f"wrote {METRICS_PATH} with {len(metrics)} metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
