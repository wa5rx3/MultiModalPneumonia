"""How the two fusion gains degrade as triage vitals go missing at test time.

Trains the triage model and meta-learner on the clean training data, then evaluates on
test sets in which an increasing fraction of the triage vital values is masked (set missing,
median-imputed, missingness flag raised), simulating deployment-time missingness. Reports the
fusion-minus-image AUROC change for the radiographic and clinical labels at each fraction,
averaged over several masking seeds.

Output: artifacts/evaluation/clinical_label/missingness_robustness.json
        manuscript/figures/fig22_missingness.png
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.training.train_multimodal_pneumonia import build_tabular_preprocessor, prepare_tabular_df

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
KEYS = ["subject_id", "study_id", "dicom_id"]
SEEDS = [42, 123, 456, 789, 1000]
VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]
PHYS = VITALS + [f"{v}_missing" for v in VITALS]
DIAG = "D:/mimic_iv_ed/diagnosis.csv.gz"
TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
OUT = ROOT / "artifacts" / "evaluation" / "clinical_label" / "missingness_robustness.json"
FIG = ROOT / "manuscript" / "figures" / "fig22_missingness.png"
FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MASK_SEEDS = 20
BLUE, ORANGE = "#0077BB", "#EE7733"


def logit(p, e=1e-6):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def ens(split):
    dfs = [pd.read_csv(f"artifacts/models/multiseed/image_seed{s}/{split}_predictions.csv")
           [KEYS + ["pred_prob"]].rename(columns={"pred_prob": f"p{s}"}) for s in SEEDS]
    m = dfs[0]
    for x in dfs[1:]:
        m = m.merge(x, on=KEYS)
    m["img"] = m[[f"p{s}" for s in SEEDS]].mean(axis=1)
    return m[KEYS + ["img"]]


def icd_label(table):
    d = pd.read_csv(DIAG, usecols=["stay_id", "icd_code", "icd_version"], dtype={"icd_code": str})

    def is_pneu(code, ver):
        c = str(code).strip().upper().replace(".", "")
        if ver == 10:
            return c[:3] in {"J12", "J13", "J14", "J15", "J16", "J17", "J18"} or c[:4] == "J690"
        return c[:3] in {"480", "481", "482", "483", "484", "485", "486"} or c[:4] == "5070"

    d["pneu"] = [is_pneu(c, v) for c, v in zip(d.icd_code, d.icd_version)]
    table["icd"] = table.stay_id.map(d.groupby("stay_id").pneu.max()).fillna(False).astype(int)
    return table


def masked_frame(base, frac, rng):
    """Return a copy of the test feature frame with fraction frac of vital values masked."""
    df = base.copy()
    for v in VITALS:
        m = rng.random(len(df)) < frac
        df.loc[m, v] = np.nan
        df.loc[m, f"{v}_missing"] = 1
    return df


def main():
    d = pd.read_parquet(TABLE)
    d = icd_label(d)
    pre = build_tabular_preprocessor(PHYS, [])
    tr = d[d.temporal_split == "train"]
    pre.fit(prepare_tabular_df(tr, PHYS, []))
    va = d[d.temporal_split == "validate"].merge(ens("val"), on=KEYS)
    te = d[d.temporal_split == "test"].merge(ens("test"), on=KEYS)
    Xtr = pre.transform(prepare_tabular_df(tr, PHYS, []))
    Xva = pre.transform(prepare_tabular_df(va, PHYS, []))
    base_te = prepare_tabular_df(te, PHYS, [])
    img_t = te.img.to_numpy()

    models = {}
    for lab in ["target", "icd"]:
        tri = LogisticRegression(max_iter=2000).fit(Xtr, tr[lab])
        tv = tri.predict_proba(Xva)[:, 1]
        meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(va.img), logit(tv)]), va[lab])
        models[lab] = (tri, meta)

    res = {"fractions": FRACTIONS, "radiographic_delta": [], "clinical_delta": []}
    for frac in FRACTIONS:
        deltas = {"target": [], "icd": []}
        for s in range(MASK_SEEDS):
            rng = np.random.default_rng(1000 + s)
            Xte = pre.transform(masked_frame(base_te, frac, rng)) if frac > 0 else pre.transform(base_te)
            for lab in ["target", "icd"]:
                tri, meta = models[lab]
                tt = tri.predict_proba(Xte)[:, 1]
                fus = meta.predict_proba(np.column_stack([logit(img_t), logit(tt)]))[:, 1]
                y = te[lab].to_numpy()
                deltas[lab].append(roc_auc_score(y, fus) - roc_auc_score(y, img_t))
            if frac == 0:  # deterministic, no need to repeat
                break
        res["radiographic_delta"].append(round(float(np.mean(deltas["target"])), 4))
        res["clinical_delta"].append(round(float(np.mean(deltas["icd"])), 4))
        print(f"frac {frac:.1f}: radiographic {res['radiographic_delta'][-1]:+.4f}, "
              f"clinical {res['clinical_delta'][-1]:+.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    matplotlib.rc("font", family="sans-serif", size=11)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    x = [100 * f for f in FRACTIONS]
    ax.plot(x, res["clinical_delta"], "-o", color=ORANGE, lw=2, label="Clinical label")
    ax.plot(x, res["radiographic_delta"], "-o", color=BLUE, lw=2, label="Radiographic label")
    ax.axhline(0, color="0.6", lw=1, ls="--")
    ax.set_xlabel("Triage vitals masked at test time (%)")
    ax.set_ylabel(r"$\Delta$AUROC  (fusion $-$ image)")
    ax.set_title("The clinical gain degrades as the vitals go missing", fontsize=11.5, pad=8)
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG.name}")


if __name__ == "__main__":
    main()
