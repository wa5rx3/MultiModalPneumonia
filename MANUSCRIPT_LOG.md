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

### P2 VERDICT (2026-06-11): multi-seed image vs concat (5 seeds each)
Across-seed mean +/- std, test n=1,075:
- image : AUROC 0.7373+/-0.0027, AUPRC 0.7191+/-0.0040, ECE 0.0531+/-0.0083, Brier 0.2067+/-0.0010
- concat: AUROC 0.7410+/-0.0062, AUPRC 0.7154+/-0.0052, ECE 0.0404+/-0.0136, Brier 0.2050+/-0.0024
Paired (concat - image), per seed across 5 seeds:
- dAUROC +0.0037 +/- 0.0087, range [-0.0066,+0.0145], 3/5 favor mm
- dAUPRC -0.0037 +/- 0.0079, 2/5 favor mm
- dECE   -0.0127 +/- 0.0162, range [-0.0368,+0.0040], 4/5 favor mm
- dBrier -0.0017 +/- 0.0030, ~0

**Interpretation (the fork, decided):**
1. H2 (non-inferior discrimination) -- ROBUSTLY CONFIRMED and slightly strengthened.
   Across seeds dAUROC = +0.004 +/- 0.009, well inside the +/-0.05 margin; the
   thesis seed-42 value (-0.009) was within ordinary seed noise. Image and concat
   are statistically indistinguishable on discrimination.
2. H3 (better calibration) -- REAL BUT MODEST AND SEED-SENSITIVE, not the clean
   40% reduction the single checkpoint implied. Mean ECE 0.040 (concat) vs 0.053
   (image); mean dECE -0.013, favoring mm in 4/5 seeds, but across-seed std (0.016)
   exceeds the mean effect and the range crosses zero. On the rebuilt seed-42 the
   gap is only -0.004 (vs thesis -0.027): the thesis headline (img ECE 0.067) used
   a high-ECE image checkpoint at the top of the across-seed range.
   => The honest, publishable message: SINGLE-SEED EVALUATION OVERSTATED THE
   CALIBRATION ADVANTAGE ~2x. Multi-seed is necessary for calibration claims.
   Brier is unchanged, so the effect is distributional alignment, not refinement.

**Decision:** reframe the paper from "multimodal is better calibrated" to a
rigorous reproducibility-aware evaluation: imaging dominates; triage fusion is
discrimination-neutral and yields only a small, seed-fragile calibration gain
that single-seed reporting overstates. This is a stronger, more honest contribution
and is exactly what a careful Q1 reviewer rewards. Venue firms up after P4
(labs/external/backbones); leaning Scientific Reports (rigorous nuanced/negative
result) or Computers in Biology and Medicine if labs+external strengthen the
clinical pipeline angle. n=5 seeds is the main analysis; may extend to 10 to
tighten the dECE interval if time allows.

### P3 results (2026-06-11): rigor + fairness analyses (5 seeds, image vs concat)
- ECE bin-sensitivity: dECE (concat-image) ranges -0.005 (5 uniform bins, 2/5
  favor mm) to -0.013 (10 uniform, 4/5). Calibration gain is small and shrinks
  under coarser/quantile binning -> the thesis 10-bin choice flatters it.
- Operating points (ED-relevant): at sens 0.90 spec 0.31 (img) vs 0.32 (concat);
  at 0.95, 0.17 vs 0.18. Clinically equivalent.
- Subgroup/fairness (mean AUROC across seeds): meaningful disparities --
  sex: Male 0.77 vs Female 0.71; race: White 0.75, Asian 0.76, Black 0.71,
  Hispanic 0.64 (small n, noisy); view PA 0.75 > AP 0.72; acuity high 0.76 >
  low 0.71. Fusion does NOT consistently close gaps; in small minority strata it
  can worsen calibration (Asian ECE 0.155->0.206). Report per-subgroup n and
  caveat small strata. Strong, honest limitations/fairness section
  (CLAIM / TRIPOD+AI). Artifacts in artifacts/evaluation/multiseed/.

### Tooling fix (2026-06-17): live Grad-CAM tab in streamlit_app.py
Pre-existing bug (present on master, not introduced by this work): the
"Live from checkpoint" Grad-CAM tab requires an `image_path` column, but the
shipped test_predictions.csv only stores subject_id/study_id/dicom_id, so the
tab always errored out before running. Fix: added `ensure_image_path_column()`
that reconstructs image_path from the IDs via the pipeline's own
`make_expected_image_path(mimic_cxr_root, ...)` (root read from
configs/paths.local.yaml). Graceful no-op if config/IDs missing. Verified:
reconstructed paths resolve 1075/1075 on D:. Saved-PNG-gallery tab was always
fine (40 overlays under artifacts/interpretability/). Fix applied on branch only.

Also fixed the Artifacts tab (`render_artifact_gallery`): it previously showed
only a hardcoded list of <=2 images (reliability diagram + decision curve).
Replaced with a dynamic gallery that enumerates all figures under
artifacts/evaluation/ grouped by folder (23 figures vs 2), plus a listing of all
result data files (75, incl. the 10 multiseed CSV/JSON). Excludes archive/ and
the _repro_check/ scratch. Branch only.

### P4a labs (2026-06-17): time-safe extraction done; KEY coverage finding
Extracted labevents at/before t0 (24h lookback, subject+encounter fallback,
strictly time-safe). Feature table = last value <= t0 per concept + missingness
flags. Built triage+labs table on the rebuilt cohort (recomputed missingness
post-join). tabular_input_dim 110 (18 triage + 52 lab cols, one-hot).
**Coverage reality (verified by dtype-normalized key intersection, not a bug):**
only 25.4% of u_ignore studies (272/1080 test) have ANY resulted lab <=t0;
CBC/BMP ~25%, lactate 9%, liver ~8%, and the pneumonia-specific inflammatory
markers CRP 0.5% / procalcitonin 0.6% essentially absent. Interpretation: at the
moment of ED imaging most labs have not resulted yet -> labs are largely
UNAVAILABLE at decision time. This is itself a clinically meaningful result and
explains why immediately-available triage vitals are the relevant pre-imaging
modality. (Prior repo "overlap=9137" report was NOT lab-filtered; 25% is truth.)
Plan: (1) full-cohort image+triage+labs multi-seed (5 seeds) vs image/concat on
the identical 1075 test (labs imputed where absent + missing flags) -- primary,
expected ~null; (2) lab-present subcohort (n~272) sensitivity: do labs help WHERE
available? 1-epoch smoke passed (AUROC 0.738, 1075 test rows). Labs 5-seed
training launched.

### P4a labs VERDICT (2026-07-06): small gain, but it's a missingness/acuity proxy
Full-cohort multi-seed (5 seeds, identical 1075 test):
- image: AUROC 0.7373+/-0.0027, ECE 0.0531, Brier 0.2067
- concat(triage): AUROC 0.7410+/-0.0062, ECE 0.0404, Brier 0.2050
- labs(triage+labs): AUROC 0.7466+/-0.0035, ECE 0.0427, Brier 0.2033
Paired labs - image: dAUROC +0.0093+/-0.0051 (5/5 seeds, range [+0.002,+0.014]
EXCLUDES zero), dBrier -0.0034 (4/5). So labs give a small, seed-consistent
discrimination + proper-score gain -- a POSITIVE result, unlike triage-concat.
BUT lab-present subcohort (n=269, where labs ARE available): image 0.776, concat
0.782, labs 0.778 AUROC -- discrimination TIED; lab VALUES do not help where
available (labs calibration still better: ECE 0.056 vs 0.071). => the full-cohort
AUROC gain is NOT lab chemistry; it tracks the missingness pattern (whether labs
were drawn = acuity/admission proxy). Running the decisive ablation now:
triage + lab MISSINGNESS FLAGS ONLY (no values), 5 seeds. If it reproduces the
+0.009, the labs 'benefit' is confirmed as a missingness/acuity artifact, not
chemistry. Honest paper claim regardless: imaging carries discrimination; neither
triage nor lab chemistry adds reliable discrimination; the tabular branch modestly
aids calibration; and labs are anyway unavailable at imaging time for ~75% of
patients. Story stays a rigorous nuanced/negative result.

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

### Leakage audit of the labs gain (2026-07-06, prompted by user challenge)
Ruled out leakage 3 ways: (1) temporal -- 0/314145 lab rows have charttime > t0
(all in (t0-24h, t0], median lead 1h); (2) split -- 0 subjects in >1 split, 0 dup
studies; (3) preprocessing -- imputer/scaler fit on TRAIN only (transform on
val/test). ALSO: univariate AUROC of 'any lab drawn' vs pneumonia on test = 0.464
(near-null, slightly NEGATIVE). => corrected my earlier overstatement: missingness
is NOT a strong acuity proxy. The +0.009 labs-vs-image gain is not leakage, not
chemistry (subcohort), not a strong missingness confound; it is small and within
the patient-level bootstrap CI. Flags-only ablation will finalize the mechanism.
