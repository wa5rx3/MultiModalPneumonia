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

### P1 result (2026-06-10): cohort rebuilt, reproduces thesis at evaluation level
Ran all 19 pipeline steps from raw MIMIC on D: (exit 0). Reconciliation of the
count delta vs thesis:
- u_ignore manifest: 9,154 studies; temporal split test=1,080 / val=930 / train=7,144.
- Thesis reported 9,137 studies / test 1,075.
- **Cause:** both the image and multimodal datasets filter to images that exist
  on disk; ~17 JPGs (5 in the test split) are absent on this D: snapshot, so the
  actually-trained/evaluated cohort is 1,075 test rows — an EXACT match to the
  thesis (rebuilt test: n=1,075 after existence filter; prevalence 45.37%,
  490 pos / 590 neg vs thesis 487/588). One study also lacks a chexpert label.
- **Decision:** build the entire manuscript on this freshly-rebuilt, fully
  reproducible cohort. Every number will trace to this pipeline run + multi-seed
  training. The ~0.5% manifest delta is immaterial and documented here.
- Verified image and concat test predictions align to the same 1,075 rows
  (targets match) -> paired dAUROC/dECE analysis is valid.
- Demographic columns available for P3 subgroups: gender, race, arrival_transport,
  acuity, view (is_pa/is_ap). Age (anchor_age) to be merged from patients.csv.
- Smoke-tested both training scripts (1 epoch each) on the rebuilt tables: clean.

### P2 preliminary (2026-06-11): image model across 5 seeds
All 5 image-only fine-tunes done on the rebuilt cohort (shared pretrain backbone,
lr 5e-5/1e-5). Per-seed test (n=1,075):
- AUROC 0.7373 +/- 0.0027 (very stable)
- AUPRC 0.7191 +/- 0.0040
- ECE   0.0531 +/- 0.0083, range [0.041, 0.060]
- Brier 0.2067 +/- 0.0010
Key early observation: the seed-42 image ECE here is 0.060, and the thesis
headline used the canonical seed-42 image ECE of 0.067 -- both at the TOP of the
across-seed range. The image model's ECE alone wanders 0.041-0.060 by seed.
This foreshadows that the H3 calibration gap (mm 0.040 vs img 0.067) may shrink
substantially once concat is evaluated across seeds. NOT a conclusion: the paper
rests on the paired dECE across seeds; concat multi-seed in progress (1/5 done).

### Venue: deferred until P2. If multi-seed confirms a calibration benefit →
Computers in Biology and Medicine / BSPC framing as a clinically-useful trade.
If it does not survive → rigorous well-powered negative result, better fit for
Scientific Reports. Will not commit the story ahead of the data.

### P0 result (2026-06-10): existing numbers reproduce exactly
Re-ran the paired bootstrap and calibration on the committed `test_predictions.csv`:
- ΔAUROC (multimodal − image) = −0.0090746, 95% CI [−0.022671, +0.004695],
  P(Δ>0)=0.10 — matches `final_publication_report.json` to every decimal.
- ECE: Image 0.06735, Multimodal 0.04034; Brier 0.20631 / 0.20686 — exact match.
- Eval code is deterministic (seed 42) in this env; committed predictions intact.
Conclusion: the artifacts are trustworthy; the pipeline is safe to build on.
Also verified CXR images resolve on D: via `make_expected_image_path` (layout
`files_pXX/mimic-cxr-jpg/2.1.0/files/pXX/pSUBJ/sSTUDY/dicom.jpg`), 5/5 sampled exist.

### Scope decision (user, 2026-06-10): full ambitious track confirmed. Do the core
rigor pass AND all three extensions: P4a labs-as-3rd-modality, P4b modern
backbone/fusion baselines, P4c external image validation. Dataset path on D:
confirmed by user — proceed as configured. This is the specialty-journal
(CBM/BSPC) effort, multi-week. Task list created (#1–#8) with P2/P4a/P4b gated on
the P1 cohort rebuild.
