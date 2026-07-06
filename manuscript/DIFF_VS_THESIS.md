# What changed: thesis → manuscript

Diff-style summary of how the journal manuscript differs from the BSc thesis
(`Yazan_thesis_v2_overleaf`). The through-line: single-seed claims were re-tested,
the story was made honest and more rigorous, and three new experiment blocks were
added. Every change is backed by a committed artifact/script.

## Headline reframing (the big one)
- **[CHANGED] H3 calibration claim.** Thesis: "multimodal is better calibrated,
  ECE 0.040 vs 0.067 (40% relative reduction), supported at seed 42." Manuscript:
  multi-seed shows the image ECE alone ranges 0.041–0.060 and the paired advantage
  shrinks to ΔECE −0.013 ± 0.016 (range crosses zero). **Single-seed evaluation
  overstated the effect ~2×.** Reframed from a positive finding to a reproducibility
  caution. Brier unchanged (distributional alignment, not refinement).
- **[STRENGTHENED] H2 non-inferiority.** Thesis seed-42 ΔAUROC = −0.009 (multimodal
  slightly worse). Multi-seed: +0.004 ± 0.009 (image vs concat statistically
  indistinguishable), still inside the pre-specified ±0.05 margin. The seed-42
  negative was within ordinary seed noise.
- **[UNCHANGED] H1.** CXR models beat triage-only baselines by >12 AUROC points.

## New experiments added (not in the thesis)
- **[NEW] Multi-seed replication** (5 seeds) for image / concat / attention — the
  thesis was single-seed (42). This is now the backbone of every claim.
- **[NEW] Labs as a third modality**, time-safe (≤ t0, 24 h lookback). Finding:
  a small +0.009 AUROC gain that a flags-only ablation shows is **100% missing-data
  structure, 0% chemistry**, and is not leakage (3-way audit). New methodological
  caution. Also: labs are unavailable at imaging time for ~75% of studies
  (CRP/procalcitonin <1%).
- **[NEW] Subgroup / fairness audit** (sex, race, view, acuity) — the thesis listed
  this as future work. Finding: real gaps (male 0.77 vs female 0.71; White 0.75 vs
  Hispanic 0.64) that fusion does not close.
- **[NEW] Operating-point analysis** at fixed sensitivity and **ECE bin-sensitivity**
  scan (5/10/15/20 uniform + quantile).
- **[NEW] Modern backbone baseline** (torchxrayvision CheXpert DenseNet, zero-shot
  0.667 + frozen probe 0.687 < our 0.737) — confirms the backbone choice.
- **[PENDING] External validation** on NIH ChestX-ray14 (image branch).

## Infrastructure / integrity
- **[CHANGED] Cohort rebuilt** from raw MIMIC on D:; verified to reproduce the
  thesis's exact evaluated test set (n = 1,075, prevalence 45.3%, 487/588). Every
  manuscript number now traces to one pipeline run + multi-seed training.
- **[NEW] Leakage audits** made explicit (temporal 0/314,145 post-t0; 0 cross-split
  subjects; preprocessing fit on train only).
- **[NEW] Reproducible analysis scripts** under `scripts/multiseed/` and
  `scripts/analysis/`; figures regenerated from artifacts (no hand-copied numbers).

## Scope / framing
- Thesis = "multimodal vs image-only for ED pneumonia, with a calibration win."
- Manuscript = "a rigorous, time-safe evaluation showing imaging dominates and
  clinical fusion is discrimination-neutral, plus two transferable reproducibility
  cautions (single-seed calibration inflation; missingness-as-signal)." Negative /
  nuanced result, framed as methodological contribution.

## Removed from scope (for the paper)
- Grad-CAM, SHAP, and full decision-curve analysis are de-emphasised (kept in the
  repo/thesis) to focus the paper on the evaluation and reproducibility story.
  Corresponding citations dropped rather than kept as filler.
