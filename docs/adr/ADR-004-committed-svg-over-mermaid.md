# ADR-004: Commit a hand built SVG rather than a Mermaid fence

Status: accepted

## Context

Two diagrams belong in this repository: the architecture, and a terminal session showing
real output.

## Decision

Generate both as committed SVG files from scripts in `tools/`, with intrinsic dimensions
and an explicit light background.

## Why

For the architecture, rendering control. A Mermaid fence is rendered by whatever is reading
the markdown, with that renderer's theme, and inherits the viewer's dark mode in ways that
can leave dark text on a dark background. A committed SVG with its own background renders
identically everywhere, including in previews that do not run Mermaid at all.

For the terminal session, something stronger than aesthetics. A screenshot pasted into a
README is unfalsifiable: a reader cannot tell whether the numbers in the picture came from
this code, from an earlier version, or from nowhere. `tools/capture_cli.py` runs the
commands, captures stdout, records each command with the exit code it actually returned
into `docs/evidence-manifest.json`, and draws the picture from what it captured. The image
is a rendering of a real run, and the manifest is the instruction for reproducing it.

This is the same discipline as the number checker, applied to the one artifact in a README
that readers habitually trust the least and verify the least.

## Consequence

The SVGs are build outputs that live in version control, so a diff can show a figure
changing when the code did. `make figures` regenerates both.
