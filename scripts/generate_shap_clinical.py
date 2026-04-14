from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
try:
    import shap
except ImportError:
    print('shap not installed. Run: pip install shap')
    sys.exit(1)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.clinical_xgb import prepare_xgb_matrix

def main() -> None:
    parser = argparse.ArgumentParser(description='SHAP analysis for XGBoost clinical model')
    parser.add_argument('--model-dir', type=str, default='artifacts/models/clinical_xgb_u_ignore_vitals_plus_acuity_temporal_strong_v2')
    parser.add_argument('--data-table', type=str, default='artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet')
    parser.add_argument('--output-dir', type=str, default='artifacts/evaluation/shap')
    parser.add_argument('--feature-groups', type=str, default='vitals_plus_acuity', choices=['all', 'vitals_only', 'demographics_only', 'acuity_only', 'vitals_plus_acuity', 'no_missing_flags'])
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'model.joblib'
    if not model_path.is_file():
        print(f'Model not found: {model_path}')
        sys.exit(1)
    model = joblib.load(model_path)
    print(f'Loaded model from {model_path}')
    df = pd.read_parquet(args.data_table)
    df_split = df[df['temporal_split'] == args.split].copy()
    print(f"Split '{args.split}': {len(df_split)} rows")
    X_test = prepare_xgb_matrix(df_split, feature_groups=args.feature_groups)
    feature_names = list(X_test.columns)
    print(f'Feature matrix shape: {X_test.shape}')
    print(f'Features: {feature_names}')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    id_cols = [c for c in ['subject_id', 'study_id'] if c in df_split.columns]
    if id_cols:
        shap_df = pd.concat([df_split[id_cols].reset_index(drop=True), shap_df], axis=1)
    shap_df.to_csv(output_dir / 'shap_values.csv', index=False)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, max_display=len(feature_names), show=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved beeswarm plot -> {output_dir / 'shap_summary_beeswarm.png'}")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=len(feature_names), show=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bar plot -> {output_dir / 'shap_summary_bar.png'}")
    mean_abs_shap = np.abs(shap_values).mean(axis=0).tolist()
    metadata = {'model_dir': str(model_dir), 'feature_groups': args.feature_groups, 'split': args.split, 'n_samples': int(len(df_split)), 'feature_names': feature_names, 'mean_abs_shap': {f: float(v) for f, v in zip(feature_names, mean_abs_shap)}, 'note': "XGBoost SHAP values are in original clinical units. No StandardScaler applied — tree-based model is scale-invariant. The tabular_preprocessor.joblib belongs to the multimodal model's tabular branch and must NOT be applied here."}
    with open(output_dir / 'shap_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata -> {output_dir / 'shap_metadata.json'}")
    print('\nMean |SHAP| per feature (descending):')
    for feat, val in sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1]):
        print(f'  {feat:30s}: {val:.4f}')
if __name__ == '__main__':
    main()
