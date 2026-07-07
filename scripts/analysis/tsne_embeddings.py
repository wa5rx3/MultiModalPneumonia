"""t-SNE of image vs image+triage representations (fusion-neutrality, visual).

Extracts, from the trained concat fusion model (seed 42), the image embedding e_v
(1024-d), the triage embedding e_t (128-d) and the fused vector [e_v; e_t] on the
held-out test set, and projects each to 2-D with t-SNE. If appending the triage
embedding does not reorganise the representation into better class-separated geometry,
the fusion-neutral discrimination result has a direct visual counterpart.

Uses the committed checkpoint and the identical eval transform / tabular preprocessing
the model was trained with. Output: manuscript/figures/fig15_tsne.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.datasets.cxr_multimodal_dataset import CXRMultimodalDataset
from src.models.multimodal_model import MultimodalPneumoniaModel
from src.training.train_multimodal_pneumonia import (
    TRIAGE_NUMERIC_COLS, TRIAGE_CATEGORICAL_COLS,
    build_tabular_preprocessor, prepare_tabular_df, build_transforms)

TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
CKPT = "artifacts/models/multiseed/concat_seed42/checkpoints/best.pt"
OUT = Path("manuscript/figures/fig15_tsne.png")
SEED = 42


@torch.no_grad()
def extract():
    df = pd.read_parquet(TABLE)
    pre = build_tabular_preprocessor(TRIAGE_NUMERIC_COLS, TRIAGE_CATEGORICAL_COLS)
    pre.fit(prepare_tabular_df(df[df.temporal_split == "train"]))
    te = df[df.temporal_split == "test"].reset_index(drop=True)
    Xte = pre.transform(prepare_tabular_df(te)).astype(np.float32)

    ckpt = torch.load(CKPT, map_location="cpu")
    sd = ckpt["model_state_dict"]
    tab_dim = sd["tabular_branch.net.0.weight"].shape[1]
    assert tab_dim == Xte.shape[1], f"tab dim mismatch: ckpt {tab_dim} vs data {Xte.shape[1]}"

    model = MultimodalPneumoniaModel(tabular_input_dim=tab_dim, tabular_hidden_dim=128,
                                     fusion_hidden_dim=256, dropout=0.2)
    model.load_state_dict(sd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    _, eval_tf = build_transforms(224)
    loader = DataLoader(CXRMultimodalDataset(df=te, tabular_array=Xte, transform=eval_tf),
                        batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    ev, et, y = [], [], []
    for b in loader:
        img = b["image"].to(device); tab = b["tabular"].to(device)
        ev.append(model.image_backbone(img).cpu().numpy())
        et.append(model.tabular_branch(tab).cpu().numpy())
        y.append(b["target"].numpy())
    ev = np.concatenate(ev); et = np.concatenate(et); y = np.concatenate(y).astype(int)
    return ev, et, np.concatenate([ev, et], axis=1), y


def tsne2d(X):
    Xs = StandardScaler().fit_transform(X)
    return TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
               random_state=SEED).fit_transform(Xs)


def main():
    ev, et, fused, y = extract()
    panels = [("Image embedding $e_v$ (1024-d)", tsne2d(ev)),
              ("Triage embedding $e_t$ (128-d)", tsne2d(et)),
              ("Fused $[e_v; e_t]$ (1152-d)", tsne2d(fused))]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (title, Z) in zip(axes, panels):
        for lab, c, name in [(0, "#4C72B0", "No pneumonia"), (1, "#C44E52", "Pneumonia")]:
            m = y == lab
            ax.scatter(Z[m, 0], Z[m, 1], s=8, c=c, alpha=0.55, linewidths=0, label=name)
        ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=9, loc="best", markerscale=1.6, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {OUT}  (test n={len(y)}, pos={int(y.sum())})")


if __name__ == "__main__":
    main()
