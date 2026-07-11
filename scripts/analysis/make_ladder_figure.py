"""Centrepiece figure: the fusion ladder.

Left panel: fusion-minus-image AUROC change at each rung, ordered from radiographic sign to
patient outcome, with patient-clustered 95% CIs, coloured by band (diagnosis / treatment /
outcome). Right panel: image-only vs triage-only AUROC per rung, showing the crossover where
physiology overtakes the radiograph as the target becomes an outcome.

Reads artifacts/evaluation/clinical_label/fusion_ladder_fair.json (fair image baseline).
Output: manuscript/figures/fig23_ladder.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
J = ROOT / "artifacts" / "evaluation" / "clinical_label" / "fusion_ladder_fair.json"
OUT = ROOT / "manuscript" / "figures" / "fig23_ladder.png"
BLUE, ORANGE, TEAL, GREY = "#0077BB", "#EE7733", "#009988", "#888888"
matplotlib.rc("font", family="sans-serif", size=11)

# rung order (image -> outcome) and display names / band colours
ORDER = [
    ("radiographic", "Radiographic sign", BLUE),
    ("ed_diagnosis", "ED diagnosis", BLUE),
    ("discharge_diagnosis", "Discharge diagnosis", BLUE),
    ("culture_confirmed", "Culture-confirmed", BLUE),
    ("ed_antibiotic", "Antibiotic given", TEAL),
    ("admission", "Hospital admission", ORANGE),
    ("icu_transfer", "ICU transfer", ORANGE),
    ("hospital_mortality", "In-hospital death", ORANGE),
    ("mortality_30d", "30-day death", ORANGE),
]


def main():
    r = json.load(open(J))
    rows = [(name, r[k], c) for k, name, c in ORDER if k in r and "fusion_minus_image" in r[k]]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})

    y = np.arange(len(rows))[::-1]
    for yi, (name, e, c) in zip(y, rows):
        lo, hi = e["ci95"]
        ax.plot([lo, hi], [yi, yi], color=c, lw=2.4, solid_capstyle="round", zorder=2)
        ax.scatter([e["fusion_minus_image"]], [yi], color=c, s=55, zorder=3, edgecolor="white", linewidth=0.8)
        ax.text(hi + 0.003, yi, f"{e['fusion_minus_image']:+.3f}", va="center", fontsize=8.6)
    ax.axvline(0, color=GREY, lw=1.0, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in rows])
    ax.set_xlabel(r"$\Delta$AUROC from adding triage  (fusion $-$ image)")
    ax.set_title("Fusion value climbs the image-to-outcome ladder", fontsize=11.5, pad=8)

    # right: image vs triage AUROC per rung
    img = [e["image_auroc"] for _, e, _ in rows]
    tri = [e["triage_auroc"] for _, e, _ in rows]
    ax2.plot(img, y, "-o", color=BLUE, lw=2, label="Image (radiograph)")
    ax2.plot(tri, y, "-o", color=ORANGE, lw=2, label="Triage (physiology)")
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlabel("AUROC of each modality alone")
    ax2.set_title("Where physiology overtakes the film", fontsize=11.5, pad=8)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.axvline(0.5, color=GREY, lw=0.8, ls=":")
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print("wrote", OUT.name, "with", len(rows), "rungs")


if __name__ == "__main__":
    main()
