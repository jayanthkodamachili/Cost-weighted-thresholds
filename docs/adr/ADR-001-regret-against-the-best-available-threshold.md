# ADR-001: Measure regret against the best threshold these scores allow

Status: accepted

## Context

The quantity this repository is built around needs a baseline, and there are two candidates.

The first is the cost at the true optimal threshold, computed from the probabilities the
data was generated with. The second is the cost at the best threshold obtainable from the
scores actually in hand, found by sweeping them.

They are not the same number. The first folds together two different failures: the model's
scores are miscalibrated, and the model's ranking is imperfect. The second isolates the
first.

## Decision

Regret is the textbook threshold's cost minus the cost at the best threshold available for
those scores.

## Why

Because the first baseline would make the headline result unfalsifiable in the wrong
direction. Measured against the true optimum, a distorted model looks bad partly because
its ranking is only worth an AUC of 0.706592, and a reader could reasonably object that the
measurement is just restating that the model is mediocre.

Measured against the best threshold those same scores support, the ranking is held fixed by
construction. Both terms use the identical score vector, so every bit of ranking error
cancels, and what remains is purely the price of putting the threshold in the wrong place.
That is the claim being made, and this baseline is the one that can be wrong about it.

It is also the baseline a practitioner can act on. Nobody can reach the true optimum
without knowing the true probabilities. Anyone can reach the swept optimum with a
validation set and an afternoon.

## Consequence

The published regret figures are a lower bound on the damage. The true gap to the
theoretical optimum is larger. This is stated in the README rather than being quietly
advantageous.
