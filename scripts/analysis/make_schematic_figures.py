"""Schematic figures for the manuscript: cohort-construction flow and model
architecture. Drawn with matplotlib boxes/arrows so they match the other figures
and require no external tooling."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("manuscript/figures")
BLUE, GREY, GREEN, ORANGE = "#4C72B0", "#e8e8e8", "#55A868", "#DD8452"


def box(ax, x, y, w, h, text, fc="white", ec="#333", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                linewidth=1.1, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.1, color="#333"))


def cohort_flow():
    steps = [
        ("MIMIC-CXR-JPG: 227,835 studies\n+ MIMIC-IV / MIMIC-IV-ED", BLUE, 1.2),
        ("Frontal (PA/AP) radiographs\nlinkable to an ED stay", "white", 1.2),
        ("Anchor $t_0$ = CXR StudyDate+Time;\ntriage merged on stay id", "white", 1.2),
        ("CheXpert pneumonia label;\nu_ignore (exclude uncertain)", "white", 1.2),
        ("ED-anchored cohort: 9,154 studies", GREEN, 1.0),
        ("Patient-level temporal 80/10/10 split\ntrain 7,144 / val 930 / test 1,080", "white", 1.2),
        ("Filter to radiographs on disk\n$\\Rightarrow$ evaluated test $n$ = 1,075\n(487 pos / 588 neg; prev. 45.3%)", ORANGE, 1.7),
    ]
    gap, margin = 0.45, 0.5
    total = sum(h for _, _, h in steps) + gap * (len(steps) - 1)
    fig, ax = plt.subplots(figsize=(6.8, 8.2)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, total + 2 * margin)
    y = total + margin
    tops, bots = [], []
    for txt, fc, h in steps:
        yb = y - h
        box(ax, 1.2, yb, 7.6, h, txt, fc=fc, bold=(fc in (GREEN, ORANGE, BLUE)), fs=9)
        tops.append(y); bots.append(yb); y = yb - gap
    for i in range(len(steps) - 1):
        arrow(ax, 5, bots[i], 5, tops[i + 1])
    fig.tight_layout(); fig.savefig(OUT / "fig13_cohort_flow.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(7.8, 4.4)); ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 7)
    box(ax, 0.3, 4.6, 2.1, 1.2, "Chest\nradiograph", fc=GREY, fs=9)
    box(ax, 0.3, 1.2, 2.1, 1.2, "Triage vitals\n(+ early labs)", fc=GREY, fs=9)
    box(ax, 3.1, 4.6, 2.6, 1.2, "DenseNet-121\n(ImageNet$\\to$CheXpert)", fc=BLUE, fs=8.5)
    box(ax, 3.1, 1.2, 2.6, 1.2, "TabularMLP\n(2 layers)", fc=GREEN, fs=9)
    box(ax, 6.4, 4.75, 1.5, 0.9, "$e_v$\n1024-d", fc="white", fs=8.5)
    box(ax, 6.4, 1.35, 1.5, 0.9, "$e_t$\n128-d", fc="white", fs=8.5)
    box(ax, 8.5, 3.05, 1.6, 1.3, "Fusion\n(concat /\nattention)", fc=ORANGE, fs=8.5)
    box(ax, 10.5, 3.05, 2.2, 1.3, "$\\hat p$\n(pneumonia)", fc="white", fs=9, bold=True)
    arrow(ax, 2.4, 5.2, 3.1, 5.2); arrow(ax, 2.4, 1.8, 3.1, 1.8)
    arrow(ax, 5.7, 5.2, 6.4, 5.2); arrow(ax, 5.7, 1.8, 6.4, 1.8)
    arrow(ax, 7.9, 5.2, 8.7, 4.35); arrow(ax, 7.9, 1.8, 8.7, 3.05)
    arrow(ax, 10.1, 3.7, 10.5, 3.7)
    fig.tight_layout(); fig.savefig(OUT / "fig14_architecture.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cohort_flow(); architecture()
    print("wrote fig13_cohort_flow.png and fig14_architecture.png")


if __name__ == "__main__":
    main()
