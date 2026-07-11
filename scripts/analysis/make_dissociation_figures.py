"""Generate the three centrepiece figures for the label-provenance reframe.

fig19_dissociation.png  forest plot of the fusion-minus-image AUROC change for the
                        radiographic and clinical labels (pneumonia flagship + heart
                        failure) with patient-clustered CIs, plus the interaction.
fig20_mechanism.png     triage-logistic coefficients for the clinical vs radiographic
                        label, showing which physiology carries clinical signal.
fig21_complementarity.png  pairwise fix/break decomposition: vitals are redundant for
                        the radiographic sign, complementary for the clinical diagnosis.

All numbers are read from the committed artifacts under artifacts/evaluation/clinical_label/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL = ROOT / "artifacts" / "evaluation" / "clinical_label"
OUT = ROOT / "manuscript" / "figures"
BLUE, ORANGE, TEAL, GREY = "#0077BB", "#EE7733", "#009988", "#888888"
matplotlib.rc("font", family="sans-serif", size=11)
matplotlib.rc("axes", titlesize=12, labelsize=11)
matplotlib.rc("xtick", labelsize=10)
matplotlib.rc("ytick", labelsize=10)
DPI = 300


def load(name):
    return json.load(open(EVAL / name))


def fig_forest():
    diss = load("dissociation.json")["labels"]
    mc = load("multicondition.json")
    fi = load("flagship_interaction.json")
    # rows top-to-bottom; store (label, delta, lo, hi, color)
    rows = [
        ("Pneumonia,  radiographic", diss["radiographic_chexpert"]["fusion_minus_image"],
         *diss["radiographic_chexpert"]["delta_95ci"], BLUE),
        ("Pneumonia,  clinical", diss["clinical_icd"]["fusion_minus_image"],
         *diss["clinical_icd"]["delta_95ci"], ORANGE),
        ("Heart failure,  radiographic", mc["heart_failure"]["radiographic"]["fusion_minus_image"],
         *mc["heart_failure"]["radiographic"]["ci95"], BLUE),
        ("Heart failure,  clinical", mc["heart_failure"]["clinical"]["fusion_minus_image"],
         *mc["heart_failure"]["clinical"]["ci95"], ORANGE),
        ("Interaction (pneumonia)", fi["interaction_clinical_minus_radiographic"],
         *fi["interaction_ci95"], TEAL),
        ("Interaction (heart failure)", mc["heart_failure"]["interaction_clinical_minus_radiographic"]["estimate"],
         *mc["heart_failure"]["interaction_clinical_minus_radiographic"]["ci95"], TEAL),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ypos = list(range(len(rows)))[::-1]
    for y, (lab, d, lo, hi, c) in zip(ypos, rows):
        marker = "D" if c == TEAL else "o"
        ax.plot([lo, hi], [y, y], color=c, lw=2.2, solid_capstyle="round", zorder=2)
        ax.scatter([d], [y], color=c, s=55, marker=marker, zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.text(hi + 0.002, y, f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=8.6)
    ax.axvline(0, color=GREY, lw=1.0, ls="--", zorder=1)
    ax.axhline(1.5, color="0.8", lw=0.8)  # divider above the interaction rows
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(r"$\Delta$AUROC  (fusion $-$ image)")
    ax.set_xlim(-0.03, 0.085)
    ax.set_title("Adding triage vitals: change in discrimination by target label",
                 fontsize=11.5, pad=8)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=8, label="Radiographic label"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=8, label="Clinical label"),
           Line2D([0], [0], marker="D", color="w", markerfacecolor=TEAL, markersize=8, label="Interaction")]
    ax.legend(handles=leg, loc="lower left", frameon=False, fontsize=8.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig19_dissociation.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_mechanism():
    m = load("mechanism_complementarity.json")["labels"]
    rad = m["radiographic_chexpert"]["triage_coefficients"]
    clin = m["clinical_icd"]["triage_coefficients"]
    order = sorted(clin, key=lambda k: abs(clin[k]))  # ascending -> largest at top
    y = np.arange(len(order))
    h = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.barh(y + h / 2, [clin[k] for k in order], height=h, color=ORANGE, label="Clinical label")
    ax.barh(y - h / 2, [rad[k] for k in order], height=h, color=BLUE, label="Radiographic label")
    ax.axvline(0, color=GREY, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Standardized triage-logistic coefficient")
    ax.set_title("Which physiology carries the signal for each label", fontsize=11.5, pad=8)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig20_mechanism.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_complementarity():
    m = load("mechanism_complementarity.json")["labels"]
    cats = [("Radiographic\nlabel", m["radiographic_chexpert"]["complementarity"]),
            ("Clinical\nlabel", m["clinical_icd"]["complementarity"])]
    x = np.arange(len(cats))
    w = 0.34
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    fixes = [c["fixes_per_pair"] for _, c in cats]
    breaks = [c["breaks_per_pair"] for _, c in cats]
    ax.bar(x - w / 2, fixes, width=w, color=TEAL, label="pairs the vitals fix")
    ax.bar(x + w / 2, breaks, width=w, color=GREY, label="pairs the vitals break")
    for i, (_, c) in enumerate(cats):
        top = max(c["fixes_per_pair"], c["breaks_per_pair"])
        ax.text(i, top + 0.003, f"ratio {c['fix_to_break_ratio']}\nnet {c['net_gain']:+.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cats])
    ax.set_ylabel("Fraction of ranked case pairs")
    ax.set_ylim(0, max(fixes) + 0.022)
    ax.set_title("Redundant vs complementary: how vitals reorder cases",
                 fontsize=11.5, pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig21_complementarity.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_forest()
    fig_mechanism()
    fig_complementarity()
    print("wrote fig19_dissociation.png, fig20_mechanism.png, fig21_complementarity.png")
