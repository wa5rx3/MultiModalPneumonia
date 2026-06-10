# Manuscript Conversion — Decision & Experiment Log

Running log of the thesis → journal manuscript effort. Append-only narrative: what
I tried, what I found, judgment calls and why (including dead ends). Numbers here
are provisional working notes; the authoritative values always live in the
regenerated artifacts under `artifacts/` and are tied to a named data split.

Branch: `paper/manuscript-2026` (off `master`). master left untouched until merge.
Author of record: Yazan (wa5rx3@gmail.com).

---

## 2026-06-10 — Orientation

**Environment confirmed**
- Repo: `C:\MultiModalFinal`. Thesis source of truth: `Yazan_thesis_v2_overleaf/`
  (chapters 01–05 + appendix A; `thesis_documentation/` and `thesis_new_docs/`
  are stale and ignored per instructions).
- Dataset on D: confirmed present and matches `configs/paths.local.yaml`:
  - `mimic_cxr_root: D:/mimic_data` (CXR-JPG under `files_pXX/mimic-cxr-jpg/2.1.0/`,
    plus chexpert/metadata/split csv.gz)
  - `mimic_iv_root: D:/mimic_iv` (patients, admissions, d_labitems, labevents/ ~300 shards)
  - `mimic_iv_ed_root: D:/mimic_iv_ed` (triage, edstays)
- Compute: 1× RTX 3060 Laptop, **6 GB VRAM** (shared with desktop apps). CUDA OK
  (torch 2.6.0+cu124, Python 3.13 venv). Constrains batch size; multi-seed
  fine-tuning is feasible, full re-pretraining is slow but avoidable (reuse backbone).

**State of the work (verified against repo + AUDIT_NOTES.md)**
- All trained checkpoints present on disk (pretrained backbone, image-only,
  multimodal concat, attention, 16 ablations) + 26 `test_predictions.csv`.
  => existing thesis numbers are re-verifiable from artifacts without retraining.
- Every locked thesis number matches its artifact JSON to 3 decimals (prior audit).
- The `u_ignore` cohort **parquet tables are missing** (only `u_one`/`u_zero` present);
  retraining on the canonical policy requires rebuilding the cohort from raw MIMIC.

**Honest assessment of journal-readiness (the gaps that matter)**
1. **Single seed.** Canonical models trained once at seed 42. H3 (calibration
   advantage, ECE 0.040 vs 0.067) is explicitly flagged single-seed and borderline:
   ΔECE 95% CI [−0.041, +0.003] crosses zero; P(ΔECE<0)=0.961. The thesis itself
   lists multi-seed replication as required future work. **This is the central fork.**
2. **No external validation.** Non-ED MIMIC AUROC 0.534 is near-chance AND not
   independent (backbone was pretrained on that population). No true external set.
3. **No subgroup / fairness audit.** Aggregate metrics only.
4. **Not novel.** DenseNet-121 + concat-MLP; the multimodal-vs-image discrimination
   result is a (clean, well-powered) negative result.

**Plan (spine):**
- P0. Reproduce existing numbers from artifacts (trust check). [cheap]
- P1. Rebuild `u_ignore` cohort from raw MIMIC (also independently verifies
  n=9,137 / test n=1,075 that the audit could not check). [moderate]
- P2. **Multi-seed replication** of image-only + concat (+ attention) fine-tunes,
  shared pretrained backbone, ≥5 seeds. Report mean±std for AUROC/AUPRC/ECE/Brier
  and the distribution of ΔAUROC, ΔECE. **Outcome decides the paper's story.** [feasible]
- P3. Subgroup/fairness audit + operating-point + ECE bin-sensitivity. [cheap, high reviewer value]
- P4 (scope-dependent): labs as 3rd modality (stronger negative or a positive),
  and/or external image validation. [heavy — decide after P2 + time budget]
- P5. Reframe manuscript honestly around what P2 shows; build figures/tables; venue.

**Venue:** deferred until P2. If multi-seed confirms a calibration benefit →
Computers in Biology and Medicine / BSPC framing as a clinically-useful trade.
If it does not survive → rigorous well-powered negative result, better fit for
Scientific Reports. Will not commit the story ahead of the data.
