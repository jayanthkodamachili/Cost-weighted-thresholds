# ADR-003: The exit code grades the model that would be shipped

Status: accepted, after a bug

## Context

The CLI returns 4 when acting on a model costs more than the better of the two trivial
policies. That code is the reason the tool exists in a pipeline: no standard model report
says it.

The first implementation read `report.decision`, which is the reading of the raw scores,
and did so whether or not a calibrator had been applied.

## The bug

Running the tool on a badly calibrated model with `--calibrator isotonic
--calibration-frac 0.05` returned 4, while the same output said the repaired model carried
regret of 0.0071 and beat the trivial policy comfortably. The gate was failing a pipeline
for doing exactly the thing the rest of the repository recommends.

This was found by running the four documented invocations to capture the terminal figure,
not by a test. The test suite had full line coverage of `exit_code_for` and every one of
those tests passed a report with no calibrator attached.

## Decision

Grade `report.calibrated_decision` when a repair ran, and `report.decision` otherwise. The
raw reading stays in the printed output in both cases, so a repair never conceals what it
repaired.

## Why

A gate that punishes the correct response to its own warning does not survive contact with
a team. The first time it blocks a release that should have shipped, somebody passes
`--regret-tolerance 1` and the gate is gone permanently. Gates are only worth having while
they are believed.

## Consequence

Four tests now hold the behaviour, including one asserting that the raw warning line still
appears after a successful repair. The more useful consequence is a note about the limits
of coverage as a signal: 100 percent line coverage of a function says every line ran, not
that every state the function can be handed was constructed in a test. The states that
mattered here were ones no test built.
