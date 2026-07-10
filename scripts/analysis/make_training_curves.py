"""Training-curves figure (fig16): loss and validation AUPRC per epoch.

Three panels (image-only, multimodal-concat, attention fusion) drawn from the committed
canonical training histories (the same checkpoints whose calibration is reported in the
paper). Left axis: train and validation BCE loss. Right axis: validation AUPRC (the
model-selection criterion), with the best epoch marked. The curves are shown as-is; they
honestly display the image-only overfitting and the attention fusion's unstable
validation despite its early convergence. One shared legend sits below the panels so it
never overlaps the curves.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("manuscript/figures/fig16_training_curves.png")
M = "artifacts/models"
PANELS = [
    (f"{M}/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3", "Image-only fine-tuning"),
    (f"{M}/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3", "Multimodal-concat fine-tuning"),
    (f"{M}/multimodal_pneumonia_attn_fusion_u_ignore_temporal_v1", "Attention fusion fine-tuning"),
]
LOSS_TR, LOSS_VA, AUPRC = "#4C72B0", "#8172B3", "#DD8452"


def load(path):
    h = json.load(open(Path(path) / "history.json"))
    ep = [e["epoch"] for e in h]
    return ep, [e["train_loss"] for e in h], [e["val_loss"] for e in h], [e["val_auprc"] for e in h]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    handles = None
    for ax, (path, title) in zip(axes, PANELS):
        ep, tr, va, ap = load(path)
        best = ep[ap.index(max(ap))]
        l1, = ax.plot(ep, tr, color=LOSS_TR, lw=1.8, label="Train loss")
        l2, = ax.plot(ep, va, color=LOSS_VA, lw=1.8, ls="--", label="Val loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (BCE)")
        ax.set_title(title, fontsize=11)
        ax.axvline(best, color="gray", lw=1, ls=":")
        ax.text(0.03, 0.97, f"best ep {best}, val AUPRC {max(ap):.3f}",
                transform=ax.transAxes, fontsize=8, color="gray", va="top", ha="left")
        ax2 = ax.twinx()
        l3, = ax2.plot(ep, ap, color=AUPRC, lw=2.0, marker="s", ms=3.5, label="Val AUPRC")
        ax2.set_ylabel("Validation AUPRC", color=AUPRC)
        ax2.tick_params(axis="y", labelcolor=AUPRC)
        ax2.set_ylim(0.50, 0.86)
        ax.grid(alpha=0.25)
        handles = [l1, l2, l3]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
