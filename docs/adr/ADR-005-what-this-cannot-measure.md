# ADR-005: What this repository cannot measure

Status: accepted

## Context

Every result here rests on knowing the true probability behind each row, because regret is
defined against the best threshold those scores could have supported, and "best" is only
knowable when the labels are generated from a probability you chose. That assumption buys
the entire measurement, and it has to be paid for somewhere.

## Decision

State the limits in the README rather than in a footnote, and keep this record of the ones
a reader is most likely to trip over.

## The limits

**Synthetic scores.** Nothing here shows how badly any real model is calibrated. A reader
who wants that number must run the gap check on their own scores, which is the point: it
costs one mean and one base rate.

**The cost matrix is assumed known and constant.** In practice the cost of a false negative
is itself an estimate, often a contested one, and it drifts. A threshold derived from a
wrong cost matrix is wrong for reasons this repository does not model. The regret measured
here is the regret conditional on the matrix being right, which is a lower bound on the
regret available in the real setting.

**Thresholds are swept on a grid.** 999 points rather than every distinct score. ADR-002
carries the comparison that settled this: the grid's resolution sits well below the
sampling error already present in the published means. For a single production decision on
a single model, sweep the actual score values instead.

**Single population.** There is no subgroup here whose calibration gap differs in sign from
the whole. That case is the one that matters most in deployment, because an aggregate gap
near zero can hide two subgroups miscalibrated in opposite directions, and the cost of the
resulting thresholds falls on whichever group is on the wrong side. Measuring it needs a
subgroup structure this generator does not have, and adding one without a real population
to model would be inventing a conclusion rather than measuring one.

**One monotone family.** The five distortions are smooth and parametric. Real
miscalibration can be lumpy, can differ by score region, and can arrive from training
distribution shift rather than from a transform. The invariance argument holds for any
strictly monotone map, so the AUC result generalises; the specific regret magnitudes do
not.

## Consequence

The honest claim this repository supports is narrow and still useful: ranking quality and
decision quality are different measurements, the gap between them is large enough to
invert a deployment decision, and the cheap check for it correlates +0.8236 with the
expensive one.
