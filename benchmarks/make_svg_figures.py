#!/usr/bin/env python
"""Draw the six figures as SVG from the CSVs ``make_figures.py`` emits.

No plotting library: the figures are simple enough that a few hundred lines of SVG are more
reproducible than a dependency, and vector output is what a paper wants.  Each figure is drawn
from ``figures/<name>.csv``, which in turn carries the receipts it came from - so the chain from
artifact to figure is data end to end.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

W, H = 900, 540
MARGIN = {"left": 90, "right": 210, "top": 50, "bottom": 70}
INK = "#111111"
GRID = "#dddddd"
BAR = "#3b6ea5"
BAR2 = "#a56b3b"
BAD = "#b23b3b"


def esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines))


def frame(title: str, xlabel: str, ylabel: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{MARGIN["left"]}" y="28" font-size="17" font-weight="bold" fill="{INK}">'
        f'{esc(title)}</text>',
        f'<text x="{(W - MARGIN["right"] + MARGIN["left"]) / 2}" y="{H - 22}" font-size="13" '
        f'text-anchor="middle" fill="{INK}">{esc(xlabel)}</text>',
        f'<text x="20" y="{H / 2}" font-size="13" text-anchor="middle" fill="{INK}" '
        f'transform="rotate(-90 20 {H / 2})">{esc(ylabel)}</text>',
    ]


def axes(x0, x1, y0, y1, xticks, yticks, logx=False, logy=False):
    """Return (svg lines, scale functions)."""
    plot_w = W - MARGIN["left"] - MARGIN["right"]
    plot_h = H - MARGIN["top"] - MARGIN["bottom"]

    def sx(v):
        if logx:
            lo, hi = math.log10(max(x0, 1e-9)), math.log10(max(x1, 1e-9))
            return MARGIN["left"] + (math.log10(max(v, 1e-9)) - lo) / (hi - lo) * plot_w
        return MARGIN["left"] + (v - x0) / (x1 - x0) * plot_w

    def sy(v):
        if logy:
            lo, hi = math.log10(max(y0, 1e-9)), math.log10(max(y1, 1e-9))
            return MARGIN["top"] + plot_h - (math.log10(max(v, 1e-9)) - lo) / (hi - lo) * plot_h
        return MARGIN["top"] + plot_h - (v - y0) / (y1 - y0) * plot_h

    out = []
    for tick in yticks:
        y = sy(tick)
        out.append(f'<line x1="{MARGIN["left"]}" y1="{y:.1f}" x2="{W - MARGIN["right"]}" '
                   f'y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{MARGIN["left"] - 8}" y="{y + 4:.1f}" font-size="11" '
                   f'text-anchor="end" fill="{INK}">{esc(tick)}</text>')
    for tick in xticks:
        x = sx(tick)
        out.append(f'<line x1="{x:.1f}" y1="{MARGIN["top"]}" x2="{x:.1f}" '
                   f'y2="{H - MARGIN["bottom"]}" stroke="{GRID}"/>')
        out.append(f'<text x="{x:.1f}" y="{H - MARGIN["bottom"] + 16}" font-size="11" '
                   f'text-anchor="middle" fill="{INK}">{esc(tick)}</text>')
    out.append(f'<line x1="{MARGIN["left"]}" y1="{H - MARGIN["bottom"]}" '
               f'x2="{W - MARGIN["right"]}" y2="{H - MARGIN["bottom"]}" stroke="{INK}"/>')
    out.append(f'<line x1="{MARGIN["left"]}" y1="{MARGIN["top"]}" x2="{MARGIN["left"]}" '
               f'y2="{H - MARGIN["bottom"]}" stroke="{INK}"/>')
    return out, sx, sy


def hbars(rows, label_key, value_key, title, xlabel, lo, hi, colour=BAR, ci_key=None):
    svg = frame(title, xlabel, "arm")
    ticks = [lo + (hi - lo) * i / 4 for i in range(5)]
    lines, sx, sy = axes(0, 1, 0, len(rows), [], [round(t, 2) for t in ticks])
    height = (H - MARGIN["top"] - MARGIN["bottom"]) / max(1, len(rows))
    for index, row in enumerate(rows):
        y = MARGIN["top"] + index * height + height * 0.15
        bar_h = height * 0.7
        value = float(row[value_key])
        x0 = MARGIN["left"]
        width = sx(max(lo, min(hi, value))) - MARGIN["left"] if hi > lo else 0
        if value < 0:
            zero = sx(0) if lo < 0 < hi else MARGIN["left"]
            width = zero - sx(value)
            x0 = sx(value)
        svg.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(width, 0):.1f}" '
                   f'height="{bar_h:.1f}" fill="{colour}" opacity="0.85"/>')
        if ci_key and row.get(ci_key):
            low, high = [float(v) for v in row[ci_key].strip("[]").split(",")]
            svg.append(f'<line x1="{sx(low):.1f}" y1="{y + bar_h / 2:.1f}" '
                       f'x2="{sx(high):.1f}" y2="{y + bar_h / 2:.1f}" stroke="{INK}"/>')
            for edge in (low, high):
                svg.append(f'<line x1="{sx(edge):.1f}" y1="{y + bar_h * 0.25:.1f}" '
                           f'x2="{sx(edge):.1f}" y2="{y + bar_h * 0.75:.1f}" stroke="{INK}"/>')
        svg.append(f'<text x="{W - MARGIN["right"] + 8}" y="{y + bar_h * 0.7:.1f}" font-size="11" '
                   f'fill="{INK}">{esc(row[label_key])[:34]} ({value:g})</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def scatter(rows, x_key, y_key, label_key, title, xlabel, ylabel):
    xs = [float(r[x_key]) for r in rows]
    ys = [float(r[y_key]) for r in rows]
    x0, x1 = min(xs) - 2, max(xs) + 2
    y0, y1 = 0, max(ys) * 1.15
    svg = frame(title, xlabel, ylabel)
    lines, sx, sy = axes(x0, x1, y0, y1,
                         [round(x0 + (x1 - x0) * i / 4, 1) for i in range(5)],
                         [round(y1 * i / 4, 2) for i in range(5)])
    svg += lines
    for row in rows:
        x, y = sx(float(row[x_key])), sy(float(row[y_key]))
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{BAR}"/>')
        svg.append(f'<text x="{x + 9:.1f}" y="{y + 4:.1f}" font-size="10" fill="{INK}">'
                   f'{esc(row[label_key])[:28]}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def vbars(rows, label_key, value_key, title, xlabel, ylabel):
    values = [float(r[value_key]) for r in rows]
    y1 = max(values) * 1.2 if values else 1
    svg = frame(title, xlabel, ylabel)
    lines, sx, sy = axes(0, len(rows), 0, y1,
                         [], [round(y1 * i / 4, 2) for i in range(5)])
    svg += lines
    slot = (W - MARGIN["left"] - MARGIN["right"]) / max(1, len(rows))
    for index, row in enumerate(rows):
        x = MARGIN["left"] + index * slot + slot * 0.2
        y = sy(float(row[value_key]))
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot * 0.6:.1f}" '
                   f'height="{H - MARGIN["bottom"] - y:.1f}" fill="{BAR}" opacity="0.85"/>')
        svg.append(f'<text x="{x + slot * 0.3:.1f}" y="{H - MARGIN["bottom"] + 16}" font-size="10" '
                   f'text-anchor="middle" fill="{INK}">{esc(row[label_key])[:16]}</text>')
        svg.append(f'<text x="{x + slot * 0.3:.1f}" y="{y - 6:.1f}" font-size="10" '
                   f'text-anchor="middle" fill="{INK}">{float(row[value_key]):g}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default="figures")
    args = p.parse_args(argv)
    directory = Path(args.dir)
    written = []

    path = directory / "fig1_state_vs_age.csv"
    if path.exists():
        rows = [r for r in read_csv(path) if int(float(r["n"])) >= 5]
        # one series: turn bucket on x, state on y, with the history it corresponds to as a label
        data = [{"bucket": r["bucket"], "state": r["state_p50"], "history": r["raw_history_p50"],
                 "label": f"{r['bucket']} (hist {float(r['raw_history_p50'])/1000:.0f}K"
                          + (f", {r['fidelity_delta_pp']} pp)" if r["fidelity_delta_pp"] else ")")}
                for r in rows]
        svg = scatter(data, "state", "history", "label",
                      "Figure 1 - the session grows, the state that must move does not",
                      "execution state (tokens)", "raw history (tokens)")
        (directory / "fig1_state_vs_age.svg").write_text(svg)
        written.append("fig1_state_vs_age.svg")

    path = directory / "fig2_two_metrics.csv"
    if path.exists():
        rows = read_csv(path)
        svg = scatter(rows, "fidelity_pp", "decision_f1", "arm",
                      "Figure 2 - fidelity against the decision, every arm at 8,192 tokens",
                      "teacher-forced fidelity delta (pp)", "end-task F1")
        (directory / "fig2_two_metrics.svg").write_text(svg)
        written.append("fig2_two_metrics.svg")

    path = directory / "fig3_compiler_ablation.csv"
    if path.exists():
        rows = read_csv(path)
        data = [{"arm": f"{r['arm']} @{r['budget']}", "value": r["active_f1"]} for r in rows]
        svg = hbars(data, "arm", "value", "Figure 3 - the end-task ladder (same instances)",
                    "end-task F1", 0.0, max(float(r["active_f1"]) for r in rows) * 1.2)
        (directory / "fig3_compiler_ablation.svg").write_text(svg)
        written.append("fig3_compiler_ablation.svg")

    path = directory / "fig4_phase_diagram.csv"
    if path.exists():
        rows = [r for r in read_csv(path) if r.get("advance")]
        data = [{"arm": f"{r['active_tokens']}"
                         + (" (admissible)" if r.get("fidelity_admissible") == "True" else ""),
                 "value": r["advance"]} for r in rows]
        svg = vbars(data, "arm", "value",
                    "Figure 4 - advancing routing cells by active set",
                    "active tokens", "cells")
        (directory / "fig4_phase_diagram.svg").write_text(svg)
        written.append("fig4_phase_diagram.svg")

    path = directory / "fig6_metric_lesson.csv"
    if path.exists():
        rows = read_csv(path)
        data = [{"arm": r["arm"][:18], "value": r["fidelity_pp"]} for r in rows]
        svg = vbars(data, "arm", "value",
                    "Figure 6 - fidelity against how the newest evidence is rendered",
                    "rendering", "fidelity delta (pp)")
        (directory / "fig6_metric_lesson.svg").write_text(svg)
        written.append("fig6_metric_lesson.svg")

    for name in written:
        print("wrote", directory / name)
    if not written:
        print("no figure CSVs found; run make_figures.py first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
