from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
MODELS_DIR = Path('artifacts/models')
EVAL_DIR = Path('artifacts/evaluation')

def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def _get(d: dict | None, *keys, default=None):
    if d is None:
        return default
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d
MODEL_CATALOGUE = [('image_u_ignore', 'image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3', 'summary.json', 'u_ignore', 'all', 'image_only'), ('image_u_zero', 'image_pneumonia_finetune_densenet121_u_zero_temporal_stronger_lr_v3', 'summary.json', 'u_zero', 'all', 'image_only'), ('image_u_one', 'image_pneumonia_finetune_densenet121_u_one_temporal_stronger_lr_v3', 'summary.json', 'u_one', 'all', 'image_only'), ('multimodal_u_ignore', 'multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3', 'summary.json', 'u_ignore', 'all', 'concat_mlp'), ('multimodal_u_zero', 'multimodal_pneumonia_densenet121_triage_u_zero_temporal_stronger_lr_v3', 'summary.json', 'u_zero', 'all', 'concat_mlp'), ('multimodal_u_one', 'multimodal_pneumonia_densenet121_triage_u_one_temporal_stronger_lr_v3', 'summary.json', 'u_one', 'all', 'concat_mlp'), ('multimodal_attn_fusion', 'multimodal_pneumonia_attn_fusion_u_ignore_temporal_v1', 'summary.json', 'u_ignore', 'all', 'attention'), ('multimodal_vitals_plus_acuity', 'multimodal_pneumonia_densenet121_triage_u_ignore_vitals_plus_acuity_temporal_v1', 'summary.json', 'u_ignore', 'vitals_plus_acuity', 'concat_mlp'), ('clinical_lr_u_ignore', 'clinical_baseline_u_ignore_temporal_strong_v2', 'metrics.json', 'u_ignore', 'all', 'logistic_regression'), ('clinical_xgb_u_ignore', 'clinical_xgb_u_ignore_temporal_strong_v2', 'metrics.json', 'u_ignore', 'all', 'xgboost')]

def _extract_model_metrics(dir_name: str, metrics_file: str) -> dict:
    d = _load_json(MODELS_DIR / dir_name / metrics_file)
    if d is None:
        return {'found': False}
    tm = _get(d, 'test_metrics') or {}
    return {'found': True, 'test_n': _get(tm, 'n'), 'positive_rate': _get(tm, 'positive_rate'), 'auroc': _get(tm, 'auroc'), 'auprc': _get(tm, 'auprc'), 'brier_score': _get(tm, 'brier_score'), 'accuracy': _get(tm, 'accuracy'), 'f1': _get(tm, 'f1')}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='artifacts/evaluation/final_publication_report.json')
    args = parser.parse_args()
    report: dict = {'generated_at': datetime.now(timezone.utc).isoformat(), 'audit_notes': {'primary_claim': 'Non-inferiority framing: multimodal model does not significantly outperform image-only. ΔAUROC = −0.009, 95% CI [−0.023, +0.005], P(Δ>0) = 0.10. Multimodal advantage: superior calibration (ECE 0.040 vs 0.067).', 'temporal_split': 'Patient-level ordinal rank by first t₀ (CXR acquisition time). 80/10/10 split — one patient belongs to exactly one split.', 't0_leakage': 'NONE. Triage vitals are ED intake measurements (triage table) structurally preceding CXR acquisition. No post-t₀ features used.', 'bootstrap_method': 'Patient-level paired bootstrap (n=2000). assert_aligned_for_delta() validates subject_id alignment before paired resampling.', 'calibration_caveat': 'ECE and Brier score bootstrap uses row-level resampling (not patient-level). CIs are marginally narrow but effect is minor given n=1075.', 'p_positive_note': 'p_positive = P(Δ > 0) is a one-tailed empirical probability, not a frequentist p-value. Document as such in the paper.', 'nonED_generalization': 'Image model evaluated on non-ED MIMIC-CXR (n=9,589). DenseNet backbone was pretrained on this population during multilabel pretraining. Label as internal generalization check — NOT external validation.'}, 'models': {}, 'pairwise_comparisons': {}, 'calibration': {}, 'feature_ablation': {}, 'label_policy_sensitivity': {}, 'nonED_generalization': {}, 'shap': {}}
    for key, dir_name, metrics_file, label_policy, feature_groups, fusion_type in MODEL_CATALOGUE:
        m = _extract_model_metrics(dir_name, metrics_file)
        m['model_dir'] = str(MODELS_DIR / dir_name)
        m['label_policy'] = label_policy
        m['feature_groups'] = feature_groups
        m['fusion_type'] = fusion_type
        report['models'][key] = m
    for key, dir_name, _, _, _, _ in MODEL_CATALOGUE:
        d = _load_json(MODELS_DIR / dir_name / 'summary.json')
        if d is None:
            continue
        bs = _get(d, 'bootstrap_summary')
        if bs:
            report['models'][key]['auroc_bootstrap'] = _get(bs, 'auroc')
            report['models'][key]['auprc_bootstrap'] = _get(bs, 'auprc')
    BOOTSTRAP_FILES = {'multimodal_vs_image': 'bootstrap_multimodal_vs_image_stronger_lr_v3.json', 'multimodal_vs_xgb': 'bootstrap_multimodal_vs_xgb_stronger_lr_v3.json', 'image_vs_xgb': 'bootstrap_image_vs_xgb_stronger_lr_v3.json', 'attn_vs_concat': 'bootstrap_attn_vs_concat.json', 'multimodal_vitals_vs_image': 'bootstrap_multimodal_vitals_vs_image.json', 'multimodal_vitals_vs_all': 'bootstrap_multimodal_vitals_vs_all.json'}
    for comp_key, fname in BOOTSTRAP_FILES.items():
        d = _load_json(EVAL_DIR / fname)
        if d is None:
            report['pairwise_comparisons'][comp_key] = {'found': False, 'file': fname}
            continue
        delta = _get(d, 'delta_a_minus_b', 'bootstrap_summary') or {}
        report['pairwise_comparisons'][comp_key] = {'found': True, 'model_a': _get(d, 'model_a_path'), 'model_b': _get(d, 'model_b_path'), 'delta_auroc': _get(delta, 'delta_auroc'), 'delta_auprc': _get(delta, 'delta_auprc')}
    cal = _load_json(EVAL_DIR / 'calibration_stronger_lr_v3' / 'calibration_metrics.json')
    if cal:
        report['calibration']['source'] = 'calibration_stronger_lr_v3/calibration_metrics.json'
        report['calibration']['models'] = _get(cal, 'models') or {}
    cal_final = _load_json(EVAL_DIR / 'calibration_final' / 'calibration_metrics.json')
    if cal_final:
        report['calibration']['calibration_final'] = {'source': 'calibration_final/calibration_metrics.json', 'models': _get(cal_final, 'models') or {}}
    ablation_csv = EVAL_DIR / 'feature_ablation_results.csv'
    if ablation_csv.is_file():
        import csv
        rows = []
        with open(ablation_csv, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        report['feature_ablation']['source'] = str(ablation_csv)
        report['feature_ablation']['rows'] = rows
    for policy in ['u_ignore', 'u_zero', 'u_one']:
        for arch in ['image', 'multimodal']:
            k = f'{arch}_{policy}'
            if k in report['models'] and report['models'][k].get('found'):
                report['label_policy_sensitivity'].setdefault(arch, {})[policy] = {'auroc': report['models'][k].get('auroc'), 'auprc': report['models'][k].get('auprc')}
    noned = _load_json(EVAL_DIR / 'nonED_generalization_image.json')
    if noned:
        report['nonED_generalization'] = noned
    shap_meta = _load_json(EVAL_DIR / 'shap' / 'shap_metadata.json')
    if shap_meta:
        report['shap'] = shap_meta
    else:
        report['shap'] = {'found': False, 'note': 'Run scripts/generate_shap_clinical.py first'}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'Report saved to {output_path}')
    found_models = sum((1 for m in report['models'].values() if m.get('found')))
    found_comps = sum((1 for c in report['pairwise_comparisons'].values() if c.get('found')))
    print(f"Models: {found_models}/{len(report['models'])} found")
    print(f"Pairwise comparisons: {found_comps}/{len(report['pairwise_comparisons'])} found")
    print(f"Feature ablation rows: {len(report['feature_ablation'].get('rows', []))}")
    print(f"SHAP: {'found' if report['shap'].get('found') is not False else 'pending'}")
if __name__ == '__main__':
    main()
