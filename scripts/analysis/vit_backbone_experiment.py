"""End-to-end modern-backbone experiment (reviewer-proofing for P4b).

Fine-tunes a Vision Transformer (ViT-Small/16, ImageNet-pretrained) end-to-end for
binary pneumonia, both image-only and with triage-concat fusion, on the identical
u_ignore cohort/split (seed 42). Tests whether a contemporary non-DenseNet backbone,
fully fine-tuned, extracts a fusion synergy the DenseNet baseline missed. If
ViT-fusion does not beat ViT-image, the discrimination-neutrality conclusion is not an
artefact of backbone choice or of using a frozen probe.

Output: artifacts/evaluation/vit_backbone/vit_metrics.json + predictions CSVs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.cxr_multimodal_dataset import CXRMultimodalDataset
from src.models.multimodal_model import TabularMLP
from src.evaluation.calibration_analysis import compute_ece_mce
from src.training.train_multimodal_pneumonia import (
    TRIAGE_NUMERIC_COLS, TRIAGE_CATEGORICAL_COLS, build_tabular_preprocessor, prepare_tabular_df)

TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
OUT = Path("artifacts/evaluation/vit_backbone")
SEED = 42
VIT = "vit_small_patch16_224"


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def tfm(train):
    ops = [transforms.Resize((224, 224))]
    if train:
        ops.append(transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 0.3))], p=0.1))
    ops += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(ops)


class ViTImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(VIT, pretrained=True, num_classes=0)
        self.head = nn.Linear(self.vit.num_features, 1)

    def forward(self, img, tab=None):
        return self.head(self.vit(img))


class ViTFusion(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        self.vit = timm.create_model(VIT, pretrained=True, num_classes=0)
        self.tab = TabularMLP(tab_dim, 128, dropout=0.2, use_batchnorm=True)
        self.head = nn.Sequential(
            nn.Linear(self.vit.num_features + 128, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, img, tab):
        return self.head(torch.cat([self.vit(img), self.tab(tab)], dim=1))


def loaders(df, tr_tab, va_tab, te_tab):
    tr = df[df.temporal_split == "train"].reset_index(drop=True)
    va = df[df.temporal_split == "validate"].reset_index(drop=True)
    te = df[df.temporal_split == "test"].reset_index(drop=True)
    dl = lambda sub, arr, t, sh: DataLoader(
        CXRMultimodalDataset(df=sub, tabular_array=arr, transform=tfm(t)),
        batch_size=16, shuffle=sh, num_workers=4, pin_memory=True)
    return dl(tr, tr_tab, True, True), dl(va, va_tab, False, False), dl(te, te_tab, False, False), len(tr), (tr.target == 1).sum(), (tr.target == 0).sum()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, ps, ids = [], [], []
    for b in loader:
        img = b["image"].to(device); tab = b["tabular"].to(device)
        p = torch.sigmoid(model(img, tab)).squeeze(1).cpu().numpy()
        ps.append(p); ys.append(b["target"].numpy())
        ids.append(pd.DataFrame({"subject_id": b["subject_id"], "study_id": b["study_id"], "dicom_id": b["dicom_id"]}))
    y = np.concatenate(ys); p = np.concatenate(ps)
    idf = pd.concat(ids, ignore_index=True); idf["target"] = y.astype(int); idf["pred_prob"] = p
    return y, p, idf


def metrics(y, p):
    ece, _, _ = compute_ece_mce(y, p, 10)
    return {"n": int(len(y)), "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "ece": float(ece), "brier": float(brier_score_loss(y, p))}


def train_model(model, tr, va, te, device, pos, neg, name, epochs=15, patience=5):
    pw = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    bb = [p for n, p in model.named_parameters() if n.startswith("vit.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("vit.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": 1e-5}, {"params": rest, "lr": 5e-5}], weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    best_ap, best_state, no_imp = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for b in tr:
            img = b["image"].to(device); tab = b["tabular"].to(device)
            y = b["target"].to(device).float().unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = crit(model(img, tab), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
        yv, pv, _ = evaluate(model, va, device); ap = average_precision_score(yv, pv)
        print(f"  [{name}] epoch {ep}: val AUPRC {ap:.4f}", flush=True)
        if ap > best_ap:
            best_ap, best_state, no_imp = ap, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
    model.load_state_dict(best_state)
    return evaluate(model, te, device)


def main():
    set_seed(SEED); OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    df = pd.read_parquet(TABLE)
    pre = build_tabular_preprocessor(TRIAGE_NUMERIC_COLS, TRIAGE_CATEGORICAL_COLS)
    tr_df = df[df.temporal_split == "train"]
    Xtr = pre.fit_transform(prepare_tabular_df(tr_df)).astype(np.float32)
    tab_dim = Xtr.shape[1]
    Xva = pre.transform(prepare_tabular_df(df[df.temporal_split == "validate"])).astype(np.float32)
    Xte = pre.transform(prepare_tabular_df(df[df.temporal_split == "test"])).astype(np.float32)
    tr, va, te, ntr, pos, neg = loaders(df, Xtr, Xva, Xte)

    results = {"backbone": VIT, "seed": SEED}
    print("=== ViT image-only ==="); set_seed(SEED)
    y, p, idf = train_model(ViTImage().to(device), tr, va, te, device, pos, neg, "img")
    idf.to_csv(OUT / "vit_image_test_predictions.csv", index=False); results["vit_image"] = metrics(y, p)
    print("=== ViT + triage fusion ==="); set_seed(SEED)
    y, p, idf = train_model(ViTFusion(tab_dim).to(device), tr, va, te, device, pos, neg, "fus")
    idf.to_csv(OUT / "vit_fusion_test_predictions.csv", index=False); results["vit_fusion"] = metrics(y, p)
    results["fusion_minus_image_auroc"] = results["vit_fusion"]["auroc"] - results["vit_image"]["auroc"]

    json.dump(results, open(OUT / "vit_metrics.json", "w"), indent=2)
    print("\n=== ViT-Small end-to-end (seed 42) ===")
    print(f"  image : AUROC {results['vit_image']['auroc']:.4f}  ECE {results['vit_image']['ece']:.4f}")
    print(f"  fusion: AUROC {results['vit_fusion']['auroc']:.4f}  ECE {results['vit_fusion']['ece']:.4f}")
    print(f"  fusion - image AUROC = {results['fusion_minus_image_auroc']:+.4f}")


if __name__ == "__main__":
    main()
