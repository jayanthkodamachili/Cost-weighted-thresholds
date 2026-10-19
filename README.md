# cost-weighted-thresholds

A model's AUC cannot tell you whether acting on it is cheaper than ignoring it.

This repository takes one model's ranking, leaves that ranking exactly intact, and shows
the cost of acting on it moving by a factor of 683. Every figure below is re-measured by
`make verify` from a clean clone, offline.

## The finding

Take a set of scores and push them through strictly monotone transforms: a power curve, a
shift in logit space, a squeeze toward the middle, a stretch toward the ends. A monotone
transform cannot reorder anything, so every ranking metric is pinned. All five score sets
all report 0.706592. The AUC varies by 0.00000000 across them, which is not a rounding
artifact but the exact arithmetic consequence of the ranks being identical.

Now price the decisions. Pick the threshold the textbook formula gives for a cost matrix,
`t* = C_fp / (C_fp + C_fn)`, and compare the cost paid there against the cost at the best
threshold those same scores could have supported. That difference is **regret**, and across
the five distortions it runs from 0.0015 to 1.0397, a factor of 683.

Worse: 3 of the five distortions produce a model that loses to not having a model at all.
Acting on its recommendations at the formula's threshold costs more than the better of
treating everything as positive or treating nothing as positive. A model with a respectable
AUC, deployed by a team that trusted the formula, quietly making things worse than the
null policy.

| distortion | AUC | calibration gap | cost at t* | regret |
| --- | --- | --- | --- | --- |
| identity | 0.706592 | -0.0003 | 0.7322 | 0.0015 |
| power | 0.706592 | -0.1458 | 1.3664 | 0.6352 |
| logit_shift | 0.706592 | -0.1640 | 1.7707 | 1.0397 |
| overconfident | 0.706592 | -0.1062 | 1.1154 | 0.3844 |
| underconfident | 0.706592 | +0.1182 | 0.7965 | 0.0659 |

The calibration gap column is the whole story in one number: mean predicted probability
minus the base rate. It is the cheapest check available, it needs no held-out data, and it
moves when the AUC does not.

## Where it hurts most

Regret is not evenly distributed across problems. It is worst exactly where the two kinds
of mistake cost very different amounts, which is to say in fraud, in churn intervention, in
medical triage, in every setting where anyone bothers to write a cost matrix down.

| cost ratio (FN:FP) | textbook t* | regret | regret as share of best cost |
| --- | --- | --- | --- |
| 1 | 0.5000 | 0.0019 | 0.010 |
| 2 | 0.3333 | 0.0307 | 0.084 |
| 5 | 0.1667 | 0.2652 | 0.432 |
| 10 | 0.0909 | 0.6335 | 0.865 |
| 50 | 0.0196 | 1.2084 | 1.510 |
| 200 | 0.0050 | 1.0607 | 1.326 |

Regret share is 0.010 at symmetric costs and reaches 1.510 at 50 to 1: the avoidable cost
exceeds the unavoidable cost. From 5:1 onward the model stops beating the null policy in
every seed tested, so 4 of the six ratios lose to having no model.

The damage is not monotone in the cost ratio. It peaks near 50:1 and recedes by 200:1,
because once the threshold is low enough the optimal policy is approaching "act on
everything" anyway, and a badly placed threshold has less room to be wrong. A reader who
expected this curve to keep climbing (as I did) should note that the experiment disagreed.

## The repair is cheap, and its price is visible

Calibration is repairable, and the data cost is far lower than the problem suggests.

| held-out share | Platt regret after | isotonic regret after |
| --- | --- | --- |
| 2% | 0.0021 | 0.0012 |
| 5% | 0.0019 | 0.0010 |
| 10% | 0.0023 | 0.0008 |
| 25% | 0.0028 | 0.0031 |
| 50% | 0.0026 | 0.0017 |

1200 held-out rows, 2 percent of the sample, recover 0.6299 of the 0.6319 available,
leaving 0.0021. Spending twenty-five times that much data buys nothing measurable; the
curve is flat from the first column onward.

The repair also does not secretly change the model. Platt scaling is monotone, so it moves
AUC by exactly +0.000000. Isotonic regression can tie scores together and lose resolution,
and it costs -0.001531 on average. Both numbers are published so a reader can confirm that
the calibrated model is the same model, scored differently.

## A screen you can afford to run on everything

If regret is the quantity that matters and nobody computes it, the practical question is
which cheap diagnostic predicts it. Across 300 random scenarios, ranked by Spearman
correlation with regret:

| screen | correlation with regret |
| --- | --- |
| abs_gap | +0.8236 |
| ece | +0.8147 |
| brier | +0.1826 |
| auc | +0.0954 |

The absolute calibration gap correlates +0.8236 with regret and costs one mean and one base
rate to compute. Expected calibration error does no better despite needing a binning scheme
and a bin-count decision. AUC manages +0.0954, which is to say nothing.

As an operational screen: 155 of 300 models carry regret above a tenth of the best
achievable cost. A gap threshold of `0.02` lets 1 through, a miss rate of 0.006 of them,
and flags 54 that were fine. That asymmetry is the right direction for a screen. A false
flag costs one afternoon of investigation; a miss costs a deployed model that is worse
than no model.

## Using it

```bash
make verify          # tests, experiments, figure check, all offline
cost-threshold --distortion logit_shift --cost-false-negative 50 --cost-false-positive 1
cost-threshold --distortion power --calibrator isotonic --calibration-frac 0.05 --json
```

Exit codes carry the verdict, so this runs in a pipeline:

| code | meaning |
| --- | --- |
| 0 | regret within tolerance, model usable at the textbook threshold |
| 1 | regret above tolerance |
| 2 | scenario configuration rejected |
| 3 | internal error |
| 4 | acting on this model costs more than not having it |

Code 4 is the one worth having. No other number in a standard model report says it.

When a calibrator is given, the exit code grades the repaired model, since that is the one
that would be deployed. The first version did not, and a pipeline that correctly repaired a
badly calibrated model was still told its model was worse than no model. That bug was found
by running the documented commands to generate the terminal figure, not by the test suite,
which had full line coverage of the function concerned and never constructed the state that
broke. [ADR-003](docs/adr/ADR-003-exit-code-grades-the-shipped-artifact.md) has it.

## What this cannot measure

The scores here are generated, not learned. That is deliberate: it is the only way to know
the true probability and therefore the true best threshold, and without those the regret
quantity does not exist to be measured. The cost is that nothing here demonstrates how
badly a particular real model is calibrated. It demonstrates that the question is worth
asking and that the asking is cheap.

Three further limits, with the reasoning in [ADR-005](docs/adr/ADR-005-what-this-cannot-measure.md):
the cost matrix is assumed known and constant, which it rarely is; thresholds are swept on
a grid rather than at every distinct score; and all of this is single-population, with no
subgroup for whom the calibration gap might differ in sign.

## Verification

Every number in this README is produced by `tools/collect_metrics.py` running the suite
and the four experiments, then checked by `tools/check_numbers.py` in both directions:
every metric must appear in the prose through a registered anchor phrase, and every decimal
in the prose must correspond to a metric. A figure that drifts breaks the build. The
checker has been deliberately broken in both directions to confirm it fails when it should.

The suite is 71 tests at 100 percent line coverage. `make verify` runs offline from a clean
clone and reproduces every experiment JSON byte for byte.


## Layout

```
src/cost_thresholds/   config, generate, decide, calibration, report, cli
experiments/           the four experiments, each writing JSON to docs/experiments/
tests/                 71 tests
tools/                 metric collection and the two-direction number checker
docs/adr/              five decision records, including what this cannot measure
```
