"""Operating-point analysis at clinically relevant fixed sensitivities.

ED triage for pneumonia prioritises catching disease (high sensitivity). At a
fixed target sensitivity, we report the threshold and the specificity / PPV / NPV
each model achieves, then compare image vs concat. Discrimination differences
that are invisible in AUROC sometimes appear (or vanish) at a chosen operating
point; this makes the clinical trade explicit. Summarised across seeds.

Output: artifacts/evaluation/multiseed/operating_point.{json,csv}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis._pred_utils import available_seeds, load_pred, summarize

EVAL_OUT = Path("artifacts/evaluation/multiseed")


def metrics_at_sensitivity(y: np.ndarray, p: np.ndarray, target_sens: float) -> dict:
    """Lowest threshold achieving sensitivity >= target (most specific such point)."""
    pos = y == 1
    neg = y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    # candidate thresholds = unique predicted probs (descending)
    thr_candidates = np.unique(p)[::-1]
    best = None
    for t in thr_candidates:
        pred = p >= t
        tp = int((pred & pos).sum())
        fn = n_pos - tp
        tn = int((~pred & neg).sum())
        fp = n_neg - tn
        sens = tp / n_pos if n_pos else 0.0
        if sens >= target_sens:
            spec = tn / n_neg if n_neg else 0.0
            ppv = tp / (tp + fp) if (tp + fp) else 0.0
            npv = tn / (tn + fn) if (tn + fn) else 0.0
            best = {
                "threshold": float(t), "sensitivity": float(sens), "specificity": float(spec),
                "ppv": float(ppv), "npv": float(npv), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }
            break
    if best is None:  # cannot reach target sensitivity even at threshold 0
        best = {"threshold": 0.0, "sensitivity": 1.0, "specificity": 0.0, "ppv": float(n_pos / (n_pos + n_neg)),
                "npv": 0.0, "tp": n_pos, "fp": n_neg, "tn": 0, "fn": 0}
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1000])
    ap.add_argument("--archs", nargs="+", default=["image", "concat"])
    ap.add_argument("--sensitivities", nargs="+", type=float, default=[0.90, 0.95])
    args = ap.parse_args()
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    collected: dict[tuple[str, float], dict[str, list]] = {}
    for arch in args.archs:
        for seed in available_seeds(arch, args.seeds):
            df = load_pred(arch, seed)
            y, p = df["target"].to_numpy(), df["pred_prob"].to_numpy()
            for ts in args.sensitivities:
                m = metrics_at_sensitivity(y, p, ts)
                rows.append({"arch": arch, "seed": seed, "target_sens": ts, **m})
                c = collected.setdefault((arch, ts), {"specificity": [], "ppv": [], "npv": [], "threshold": []})
                for k in c:
                    c[k].append(m[k])

    if not rows:
        raise SystemExit("No multi-seed predictions found yet.")

    pd.DataFrame(rows).to_csv(EVAL_OUT / "operating_point.csv", index=False)

    summary: dict = {"sensitivities": args.sensitivities, "per_arch": {}}
    for (arch, ts), c in collected.items():
        summary["per_arch"].setdefault(arch, {})[f"sens_{ts:.2f}"] = {k: summarize(v) for k, v in c.items()}

    with open(EVAL_OUT / "operating_point.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Operating points (mean +/- std across seeds) ===")
    for arch, byts in summary["per_arch"].items():
        for sk, md in byts.items():
            print(f"{arch:8s} @{sk}: spec={md['specificity']['mean']:.3f}+/-{md['specificity']['std']:.3f} "
                  f"ppv={md['ppv']['mean']:.3f} npv={md['npv']['mean']:.3f} thr={md['threshold']['mean']:.3f}")
    print(f"\nWrote {EVAL_OUT}/operating_point.json/.csv")


if __name__ == "__main__":
    main()
