"""Render the committed architecture SVG.

Hand written rather than produced by a diagramming library, for the reasons in ADR-004: a
committed SVG with intrinsic dimensions and an explicit light background renders the same
on GitHub in either theme, which a Mermaid fence does not reliably do.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "architecture.svg"

W, H = 960, 500
INK = "#1b1f23"
MUTED = "#57606a"
BAD = "#b42318"
GOOD = "#116329"
LINE = "#8c959f"


def box(x, y, w, h, title, subtitle="", stroke=MUTED, fill="#ffffff", bold=False):
    weight = "600" if bold else "400"
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        f'<text x="{x + w / 2}" y="{y + (20 if subtitle else h / 2 + 4)}" '
        f'text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="13" '
        f'font-weight="{weight}" fill="{INK}">{title}</text>',
    ]
    if subtitle:
        for index, line in enumerate(subtitle.split("|")):
            parts.append(
                f'<text x="{x + w / 2}" y="{y + 38 + index * 14}" text-anchor="middle" '
                f'font-family="Helvetica,Arial,sans-serif" font-size="11" '
                f'fill="{MUTED}">{line}</text>'
            )
    return "".join(parts)


def arrow(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" '
        f'stroke-width="1.5" marker-end="url(#a)"{dash}/>'
    )


def label(x, y, text, fill=MUTED, size=11, anchor="middle", weight="400"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Helvetica,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{text}</text>'
    )


def render() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Architecture of cost-weighted-thresholds">',
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/>'
        "</marker></defs>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        label(W / 2, 26, "One ranking, five score sets, five different bills",
              INK, 14, weight="600"),
    ]

    parts.append(box(24, 150, 190, 70, "true probabilities",
                     "and the labels|drawn from them", bold=True))
    parts.append(box(260, 150, 210, 70, "strictly monotone|distortion",
                     "cannot reorder anything", bold=True))
    parts.append(arrow(214, 185, 260, 185))

    # Upper path: the metric that cannot see it.
    parts.append(box(540, 58, 230, 56, "rank the rows",
                     "AUC, ROC, precision at k"))
    parts.append(box(800, 58, 136, 56, "0.706592",
                     "identical, all five", stroke=MUTED, fill="#f6f8fa", bold=True))
    parts.append(arrow(470, 170, 540, 110))
    parts.append(arrow(770, 86, 800, 86))

    # Lower path: the one that can.
    parts.append(box(540, 150, 230, 56, "compare mean score|against the base rate",
                     "", stroke=GOOD, fill="#f2fbf4", bold=True))
    parts.append(box(800, 150, 136, 56, "the gap",
                     "moves with the|distortion", stroke=GOOD, fill="#f2fbf4"))
    parts.append(arrow(470, 185, 540, 182))
    parts.append(arrow(770, 178, 800, 178))

    parts.append(box(540, 262, 230, 70, "threshold at C_fp/(C_fp+C_fn)",
                     "then sweep for the best|threshold these scores allow",
                     stroke=BAD, fill="#fff5f4", bold=True))
    parts.append(box(800, 262, 136, 70, "regret",
                     "0.0015 to 1.0397|a factor of 683",
                     stroke=BAD, fill="#fff5f4", bold=True))
    parts.append(arrow(470, 200, 540, 290))
    parts.append(arrow(770, 297, 800, 297))

    parts.append(label(655, 356, "the only reading that prices a decision", MUTED, 10))

    parts.append(
        box(130, 400, 700, 66,
            "a ranking metric is invariant to exactly the thing that sets the bill",
            "three of five distortions produce a model that costs more to act on "
            "than no model at all", bold=True)
    )
    parts.append(arrow(868, 114, 700, 400))
    parts.append(arrow(868, 332, 480, 400))

    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
