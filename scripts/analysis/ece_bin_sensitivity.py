"""ECE bin-count and binning-scheme sensitivity, across seeds.

ECE point estimates are known to depend on the number of bins and the binning
scheme at moderate sample sizes (Nixon et al. 2019). This reports ECE for the
image and concat models under uniform-width bins {5,10,15,20} and equal-frequency
(quantile) bins {10,15}, summarised across seeds (mean +/- std), plus the paired
dECE (concat - image) for each scheme.

Output: artifacts/evaluation/multiseed/ece_bin_sensitivity.{json,csv}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.analysis._pred_utils import available_seeds, ece_quantile, ece_uniform, load_pred, summarize

EVAL_OUT = Path("artifacts/evaluation/multiseed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1000])
    ap.add_argument("--archs", nargs="+", default=["image", "concat"])
    args = ap.parse_args()
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    schemes = [("uniform", n) for n in (5, 10, 15, 20)] + [("quantile", n) for n in (10, 15)]

    rows = []
    per_arch_seed_ece: dict[tuple[str, str, int], dict[int, float]] = {}
    for arch in args.archs:
        seeds = available_seeds(arch, args.seeds)
        for seed in seeds:
            df = load_pred(arch, seed)
            y, p = df["target"].to_numpy(), df["pred_prob"].to_numpy()
            for scheme, nb in schemes:
                ece = ece_uniform(y, p, nb) if scheme == "uniform" else ece_quantile(y, p, nb)
                rows.append({"arch": arch, "seed": seed, "scheme": scheme, "n_bins": nb, "ece": ece})
                per_arch_seed_ece.setdefault((arch, scheme, nb), {})[seed] = ece

    if not rows:
        raise SystemExit("No multi-seed predictions found yet.")

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(EVAL_OUT / "ece_bin_sensitivity.csv", index=False)

    summary: dict = {"schemes": [{"scheme": s, "n_bins": n} for s, n in schemes], "per_arch": {}, "paired_delta": {}}
    for arch in args.archs:
        summary["per_arch"][arch] = {}
        for scheme, nb in schemes:
            d = per_arch_seed_ece.get((arch, scheme, nb), {})
            if d:
                summary["per_arch"][arch][f"{scheme}_{nb}"] = summarize(d.values())

    # paired dECE per scheme (concat - image), across common seeds
    for scheme, nb in schemes:
        di = per_arch_seed_ece.get(("image", scheme, nb), {})
        dc = per_arch_seed_ece.get(("concat", scheme, nb), {})
        common = sorted(set(di) & set(dc))
        if common:
            deltas = [dc[s] - di[s] for s in common]
            summary["paired_delta"][f"{scheme}_{nb}"] = {
                **summarize(deltas),
                "frac_favoring_multimodal": float(sum(d < 0 for d in deltas) / len(deltas)),
            }

    with open(EVAL_OUT / "ece_bin_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== ECE by binning scheme (mean +/- std across seeds) ===")
    for arch in args.archs:
        if arch not in summary["per_arch"]:
            continue
        print(f"\n{arch}:")
        for key, s in summary["per_arch"][arch].items():
            if s["n"]:
                print(f"  {key:12s}: {s['mean']:.4f} +/- {s['std']:.4f}")
    print("\n=== paired dECE (concat - image) ===")
    for key, s in summary["paired_delta"].items():
        print(f"  {key:12s}: {s['mean']:+.4f} +/- {s['std']:.4f}  frac_favoring_mm={s['frac_favoring_multimodal']:.2f}")
    print(f"\nWrote {EVAL_OUT}/ece_bin_sensitivity.json/.csv")


if __name__ == "__main__":
    main()
