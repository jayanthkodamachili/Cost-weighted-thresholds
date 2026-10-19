# ADR-002: Sweep a grid of thresholds rather than every distinct score

Status: accepted

## Context

Finding the best threshold for a score vector can be done exactly, by evaluating cost at
every distinct score value, or approximately, on a fixed grid. The exact method costs time
proportional to the sample size per scenario. The published figures are means over four
seeds, five distortions, six cost ratios, ten repair settings and 300 random screening
scenarios, so the multiplier is not small.

## Decision

Sweep 999 evenly spaced thresholds in [0, 1].

## Why

The comparison was run before the choice was made rather than after. On the scenarios here
the grid's best cost differs from the exact best cost by less than the seed-to-seed
standard deviation of the same quantity, by roughly an order of magnitude. A difference
that sits an order of magnitude below the noise already in the published means cannot
change any conclusion drawn from them.

The honest framing: this is an approximation chosen for speed, justified by a measurement
showing the approximation is invisible at the resolution anything is reported to. It is not
free, and it is not defended as free.

## Consequence

For a single production decision on a single model, this is the wrong trade. Sweep the
actual score values; the cost is one pass and the answer is exact. ADR-005 repeats this
where a reader is more likely to be looking for it.

Scores here live in [0, 1] by construction, so the grid covers the full range. A score
vector that did not would need the grid pinned to its own min and max, and nothing in the
code currently enforces that assumption.
