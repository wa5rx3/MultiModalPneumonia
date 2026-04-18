import json
import os
import sys
import warnings
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, roc_auc_score
warnings.filterwarnings('ignore')
np.random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / 'artifacts'
MANIFESTS = ARTIFACTS / 'manifests'
MODELS = ARTIFACTS / 'models'
EVAL = ARTIFACTS / 'evaluation'
OUT = ROOT / 'thesis_new_docs' / 'figures' / 'generated_results'
OUT.mkdir(parents=True, exist_ok=True)
PALETTE = {'image': '#0077BB', 'multimodal': '#EE7733', 'attention': '#009988', 'xgboost': '#CC3311', 'lr': '#AA4499'}
LABEL_MAP = {'image': 'Image-only (DenseNet121)', 'multimodal': 'Multimodal-concat', 'attention': 'Multimodal-attention', 'xgboost': 'Clinical XGBoost', 'lr': 'Clinical LR'}
DPI = 300
FONT = {'family': 'sans-serif', 'size': 11}
matplotlib.rc('font', **FONT)
matplotlib.rc('axes', titlesize=12, labelsize=11)
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('legend', fontsize=9.5)
matplotlib.rc('figure', dpi=DPI)
PRED_FILES = {'image': MODELS / 'image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3' / 'test_predictions.csv', 'multimodal': MODELS / 'multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3' / 'test_predictions.csv', 'attention': MODELS / 'multimodal_pneumonia_attn_fusion_u_ignore_temporal_v1' / 'test_predictions.csv', 'xgboost': MODELS / 'clinical_xgb_u_ignore_temporal_strong_v2' / 'test_predictions.csv', 'lr': MODELS / 'clinical_baseline_u_ignore_temporal_strong_v2' / 'test_predictions.csv'}
VAL_PRED_FILES = {'image': MODELS / 'image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3' / 'val_predictions.csv', 'multimodal': MODELS / 'multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3' / 'val_predictions.csv'}
HISTORY_FILES = {'image': MODELS / 'image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3' / 'history.json', 'multimodal': MODELS / 'multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3' / 'history.json'}
_TRAINING_TABLE_CANDIDATES = [MANIFESTS / 'cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet', MANIFESTS / 'cxr_clinical_pneumonia_training_table_u_one_temporal.parquet']
TRAINING_TABLE = next((p for p in _TRAINING_TABLE_CANDIDATES if p.exists()), _TRAINING_TABLE_CANDIDATES[0])
PUBLICATION_REPORT = EVAL / 'final_publication_report.json'
ABLATION_CSV = EVAL / 'feature_ablation_results.csv'
SHAP_VALUES = EVAL / 'shap' / 'shap_values.csv'
SHAP_META = EVAL / 'shap' / 'shap_metadata.json'

def bootstrap_ci(y_true, y_score, metric='auroc', n=2000, seed=42):
    rng = np.random.default_rng(seed)
    subjects = np.unique(np.arange(len(y_true)))
    vals = []
    for _ in range(n):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        yt, ys = (y_true[idx], y_score[idx])
        if len(np.unique(yt)) < 2:
            continue
        if metric == 'auroc':
            vals.append(roc_auc_score(yt, ys))
        else:
            vals.append(average_precision_score(yt, ys))
    vals = np.array(vals)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5))

def load_pred(key):
    df = pd.read_csv(PRED_FILES[key])
    return (df['target'].values, df['pred_prob'].values)

def fig_a1_pr_curves():
    print('  A1: Precision-Recall curves...')
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    models = ['image', 'multimodal', 'attention', 'xgboost', 'lr']
    prevalence = None
    for key in models:
        yt, ys = load_pred(key)
        if prevalence is None:
            prevalence = yt.mean()
        prec, rec, _ = precision_recall_curve(yt, ys)
        ap = average_precision_score(yt, ys)
        ci_lo, ci_hi = bootstrap_ci(yt, ys, metric='auprc')
        label = f'{LABEL_MAP[key]}  AUPRC={ap:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]'
        ax.plot(rec, prec, lw=1.8, color=PALETTE[key], label=label)
    ax.axhline(prevalence, ls='--', lw=1.2, color='#888888', label=f'Random classifier  (prevalence={prevalence:.3f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision (PPV)')
    ax.set_title('Precision-Recall Curves: Test Set (n=1,075, u_ignore)')
    ax.legend(loc='lower left', framealpha=0.95, fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()
    out = OUT / 'fig_a1_pr_curves.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'     {out.name}')
    return str(out)

def compute_calibration_bins(yt, ys, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mids, frac_pos, frac_pred, counts = ([], [], [], [])
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (ys >= lo) & (ys < hi)
        if mask.sum() > 0:
            bin_mids.append(ys[mask].mean())
            frac_pos.append(yt[mask].mean())
            frac_pred.append(ys[mask].mean())
            counts.append(mask.sum())
    return (np.array(bin_mids), np.array(frac_pos), np.array(frac_pred), np.array(counts))

def compute_ece(yt, ys, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(yt)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (ys >= lo) & (ys < hi)
        if mask.sum() == 0:
            continue
        acc = yt[mask].mean()
        conf = ys[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return ece

def hosmer_lemeshow(yt, ys, n_groups=10):
    from scipy.stats import chi2
    df_tmp = pd.DataFrame({'y': yt, 'p': ys})
    df_tmp['decile'] = pd.qcut(df_tmp['p'], n_groups, labels=False, duplicates='drop')
    hl_stat = 0.0
    for _, grp in df_tmp.groupby('decile'):
        obs = grp['y'].sum()
        exp = grp['p'].sum()
        n_g = len(grp)
        if exp > 0 and exp < n_g:
            hl_stat += (obs - exp) ** 2 / (exp * (1 - exp / n_g))
    p_val = 1 - chi2.cdf(hl_stat, df=n_groups - 2)
    return (hl_stat, p_val)

def fig_a2_reliability_panel():
    print('  A2: 2×2 reliability diagram panel...')
    panel_models = [('image', 'Image-only (DenseNet121)'), ('multimodal', 'Multimodal-concat'), ('attention', 'Multimodal-attention'), ('xgboost', 'Clinical XGBoost'), ('lr', 'Clinical LR')]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    axes[5].set_visible(False)
    for i, (key, title) in enumerate(panel_models):
        ax = axes[i]
        yt, ys = load_pred(key)
        bin_mids, frac_pos, frac_pred, counts = compute_calibration_bins(yt, ys)
        ece = compute_ece(yt, ys)
        hl_stat, hl_p = hosmer_lemeshow(yt, ys)
        ax.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.6, label='Perfect calibration')
        ax.plot(frac_pred, frac_pos, 'o-', color=PALETTE[key], lw=2.0, ms=5, label='Model calibration')
        ax.fill_between(frac_pred, frac_pos, frac_pred, alpha=0.12, color=PALETTE[key])
        ax_hist = ax.twinx()
        ax_hist.bar(bin_mids, counts / counts.sum(), width=0.08, color=PALETTE[key], alpha=0.18, align='center')
        ax_hist.set_ylim(0, 1.2)
        ax_hist.set_yticks([])
        ax_hist.set_ylabel('')
        hl_str = f'p={hl_p:.2e}' if hl_p < 0.001 else f'p={hl_p:.3f}'
        ax.text(0.04, 0.93, f'ECE = {ece:.3f}\nH-L {hl_str}', transform=ax.transAxes, va='top', fontsize=9.5, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.9))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction Positive')
        ax.set_title(title, fontsize=11, color=PALETTE[key], fontweight='bold')
        ax.legend(loc='lower right', fontsize=8.5)
        ax.grid(True, alpha=0.25)
    fig.suptitle('Reliability Diagrams: Test Set (n=1,075, u_ignore, 10 uniform bins)', fontsize=12, y=1.01)
    fig.tight_layout()
    out = OUT / 'fig_a2_reliability_panel.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'     {out.name}')
    return str(out)

def fig_a3_ablation_bars():
    print('  A3: Feature ablation bar chart...')
    df = pd.read_csv(ABLATION_CSV)
    df['test_auroc'] = df['test_auroc'].astype(float)
    GROUP_LABELS = {'all': 'All features', 'vitals_only': 'Vitals only', 'vitals_plus_acuity': 'Vitals + acuity', 'no_missing_flags': 'No missing flags', 'demographics_only': 'Demographics only', 'acuity_only': 'Acuity only'}
    model_colors = {'clinical_logistic': PALETTE['lr'], 'clinical_xgb': PALETTE['xgboost'], 'multimodal': PALETTE['multimodal']}
    model_labels = {'clinical_logistic': 'Clinical LR', 'clinical_xgb': 'Clinical XGBoost', 'multimodal': 'Multimodal-concat'}
    group_order = ['all', 'vitals_plus_acuity', 'vitals_only', 'no_missing_flags', 'demographics_only', 'acuity_only']
    model_order = ['clinical_logistic', 'clinical_xgb', 'multimodal']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=False)
    for ax, mtype in zip(axes, model_order):
        sub = df[df['model_type'] == mtype].copy()
        sub = sub.set_index('feature_groups')
        groups_for_model = [g for g in group_order if g in sub.index]
        aurocs = [sub.loc[g, 'test_auroc'] for g in groups_for_model]
        labels = [GROUP_LABELS[g] for g in groups_for_model]
        color = model_colors[mtype]
        bars = ax.barh(labels[::-1], aurocs[::-1], color=color, alpha=0.82, height=0.62, edgecolor='white', linewidth=0.5)
        all_auroc = sub.loc['all', 'test_auroc'] if 'all' in sub.index else None
        if all_auroc is not None:
            ax.axvline(all_auroc, ls=':', lw=1.3, color='#444444', alpha=0.7, label=f"'All features' AUROC = {all_auroc:.3f}")
        for bar, val in zip(bars, aurocs[::-1]):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', ha='left', fontsize=8.5)
        ax.axvline(0.5, ls='--', lw=0.8, color='#bbbbbb', alpha=0.5)
        ax.set_xlim(0.48, 0.8)
        ax.set_xlabel('Test AUROC')
        ax.set_title(model_labels[mtype], color=color, fontweight='bold', fontsize=11)
        ax.grid(True, axis='x', alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        if all_auroc is not None:
            ax.legend(fontsize=8.5, loc='lower right')
    fig.suptitle('Feature Ablation: Test AUROC by Feature Group (u_ignore, n_test=1,075)', fontsize=12)
    fig.tight_layout()
    out = OUT / 'fig_a3_ablation_bars.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'     {out.name}')
    return str(out)

def fig_a4_label_sensitivity():
    print('  A4: Label policy sensitivity chart...')
    with open(PUBLICATION_REPORT) as f:
        report = json.load(f)
    sens = report['label_policy_sensitivity']
    policies = ['u_ignore', 'u_zero', 'u_one']
    pol_labels = {'u_ignore': 'u_ignore\n(definitive only)', 'u_zero': 'u_zero\n(uncertain  neg)', 'u_one': 'u_one\n(uncertain  pos)'}
    pol_n = {'u_ignore': 1075, 'u_zero': 1944, 'u_one': 1944}
    models = [('image', 'Image-only'), ('multimodal', 'Multimodal-concat')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    for ax, (metric_key, metric_label) in zip(axes, [('auroc', 'AUROC'), ('auprc', 'AUPRC')]):
        x = np.arange(len(policies))
        width = 0.32
        for offset, (mkey, mlabel) in zip([-width / 2, width / 2], models):
            vals = [sens[mkey][p][metric_key] for p in policies]
            ax.bar(x + offset, vals, width=width, color=PALETTE[mkey], label=mlabel, alpha=0.85, edgecolor='white', linewidth=0.5)
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([pol_labels[p] for p in policies], fontsize=9.5)
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} by Label Policy')
        ax.legend(fontsize=9.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        for xi, p in zip(x, policies):
            ax.text(xi, 0.02, f'n={pol_n[p]}', ha='center', va='bottom', fontsize=7.5, color='#666666')
    fig.suptitle('Label Policy Sensitivity Analysis: Image and Multimodal Models', fontsize=12)
    fig.tight_layout()
    out = OUT / 'fig_a4_label_sensitivity.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'     {out.name}')
    return str(out)

def fig_a5_training_curves():
    print('  A5: Training curves...')
    all_history_files = {
        'image': MODELS / 'image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3' / 'history.json',
        'multimodal': MODELS / 'multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3' / 'history.json',
        'attention': MODELS / 'multimodal_pneumonia_attn_fusion_u_ignore_temporal_v1' / 'history.json',
    }
    histories = {}
    for key, path in all_history_files.items():
        if path.exists():
            with open(path) as f:
                histories[key] = json.load(f)
    n_models = len(histories)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.8))
    if n_models == 1:
        axes = [axes]
    titles = {'image': 'Image-only Fine-tuning', 'multimodal': 'Multimodal-concat Fine-tuning', 'attention': 'Attention Fusion Fine-tuning'}
    for ax, (key, hist) in zip(axes, histories.items()):
        epochs = [h['epoch'] for h in hist]
        tl = [h['train_loss'] for h in hist]
        vl = [h['val_loss'] for h in hist]
        va = [h['val_auprc'] for h in hist]
        best_ep = epochs[int(np.argmax(va))]
        color = PALETTE[key]
        ax.plot(epochs, tl, '-', color=color, lw=2.0, alpha=0.65, label='Train loss')
        ax.plot(epochs, vl, '--', color=color, lw=2.0, label='Val loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (BCE)', color='#333333')
        ax.set_ylim(bottom=0)
        ax2 = ax.twinx()
        ax2.plot(epochs, va, 's-', color='#CC6600', lw=2.0, ms=4, label='Val AUPRC')
        ax2.set_ylabel('Validation AUPRC', color='#CC6600')
        ax2.tick_params(axis='y', colors='#CC6600')
        ax2.set_ylim(0.5, 0.85)
        best_auprc = max(va)
        ax2.axvline(best_ep, ls=':', lw=1.5, color='#333333', alpha=0.7)
        x_text = best_ep + len(epochs) * 0.15
        y_text = min(best_auprc + 0.06, 0.83)
        ax2.annotate(f'Best ep. {best_ep}\n{best_auprc:.4f}',
                     xy=(best_ep, best_auprc),
                     xytext=(x_text, y_text),
                     fontsize=8, color='#333333',
                     arrowprops=dict(arrowstyle='->', color='#333333', lw=0.8))
        ax.set_title(titles[key], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.25)
        lines1 = [matplotlib.lines.Line2D([0], [0], color=color, lw=2, linestyle='-'),
                   matplotlib.lines.Line2D([0], [0], color=color, lw=2, linestyle='--'),
                   matplotlib.lines.Line2D([0], [0], color='#CC6600', lw=2, linestyle='-', marker='s')]
        ax.legend(lines1, ['Train loss', 'Val loss', 'Val AUPRC'],
                  loc='upper center', bbox_to_anchor=(0.5, -0.14),
                  ncol=3, fontsize=9, frameon=False)
    fig.suptitle('Training Curves: Fine-tuning Stage (val AUPRC selection criterion)', fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = OUT / 'fig_a5_training_curves.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'     {out.name}')
    return str(out)

def fig_a6_missing_heatmap():
    print('  A6: Missing data heatmap...')
    df = pd.read_parquet(TRAINING_TABLE)
    vitals = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity']
    miss_data = {}
    for split in ['train', 'validate', 'test']:
        sub = df[df['temporal_split'] == split]
        miss_data[split] = {v: sub[v].isna().mean() * 100 for v in vitals}
    miss_df = pd.DataFrame(miss_data, index=vitals)
    miss_df = miss_df[['train', 'validate', 'test']]
    feat_labels = {'temperature': 'Temperature (°F)', 'heartrate': 'Heart rate (bpm)', 'resprate': 'Respiratory rate (/min)', 'o2sat': 'O₂ saturation (%)', 'sbp': 'Systolic BP (mmHg)', 'dbp': 'Diastolic BP (mmHg)', 'pain': 'Pain score (0–10)', 'acuity': 'ESI acuity (1–5)'}
    miss_df.index = [feat_labels[v] for v in vitals]
    miss_df.columns = ['Training', 'Validation', 'Test']
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(miss_df, annot=True, fmt='.1f', cmap='YlOrRd', vmin=0, vmax=15, linewidths=0.5, linecolor='#eeeeee', cbar_kws={'label': 'Missing rate (%)', 'shrink': 0.75}, ax=ax)
    ax.set_title('Triage Feature Missing Data Rates by Split (%)', fontsize=12, pad=12)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10, rotation=0)
    fig.tight_layout()
    out = OUT / 'fig_a6_missing_heatmap.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'     {out.name}')
    return str(out)

def fig_a7_vital_distributions():
    print('  A7: Vital sign distributions...')
    df = pd.read_parquet(TRAINING_TABLE)
    train = df[df['temporal_split'] == 'train'].copy()
    train['Pneumonia'] = train['target'].map({0: 'Negative', 1: 'Positive'})
    vitals_config = [('temperature', 'Temperature (°F)', [95.0, 105.8]), ('heartrate', 'Heart rate (bpm)', [30, 220]), ('resprate', 'Respiratory rate (/min)', [5, 60]), ('o2sat', 'O₂ saturation (%)', [50, 100]), ('sbp', 'Systolic BP (mmHg)', [60, 250]), ('dbp', 'Diastolic BP (mmHg)', [30, 150])]
    colors_label = {'Negative': '#0077BB', 'Positive': '#CC3311'}
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5))
    axes = axes.flatten()
    for ax, (col, ylabel, clip_range) in zip(axes, vitals_config):
        sub = train[[col, 'Pneumonia']].dropna()
        parts = ax.violinplot([sub.loc[sub['Pneumonia'] == lbl, col].values for lbl in ['Negative', 'Positive']], positions=[0, 1], showmedians=True, showextrema=False, widths=0.65)
        for body, lbl in zip(parts['bodies'], ['Negative', 'Positive']):
            body.set_facecolor(colors_label[lbl])
            body.set_alpha(0.6)
            body.set_edgecolor('#444444')
            body.set_linewidth(0.8)
        parts['cmedians'].set_color('#222222')
        parts['cmedians'].set_linewidth(2.0)
        ax.axhline(clip_range[0], ls=':', lw=1.0, color='#888888', alpha=0.7)
        ax.axhline(clip_range[1], ls=':', lw=1.0, color='#888888', alpha=0.7)
        ax.text(1.02, clip_range[0], f' clip', va='center', transform=ax.get_yaxis_transform(), fontsize=7.5, color='#888888')
        ax.text(1.02, clip_range[1], f' clip', va='center', transform=ax.get_yaxis_transform(), fontsize=7.5, color='#888888')
        neg_vals = sub.loc[sub['Pneumonia'] == 'Negative', col].values
        pos_vals = sub.loc[sub['Pneumonia'] == 'Positive', col].values
        if len(neg_vals) > 0 and len(pos_vals) > 0:
            _, mwu_p = stats.mannwhitneyu(neg_vals, pos_vals, alternative='two-sided')
            sig = '***' if mwu_p < 0.001 else '**' if mwu_p < 0.01 else '*' if mwu_p < 0.05 else 'ns'
            ax.set_title(f'{ylabel}\n(MWU: {sig})', fontsize=10.5)
        else:
            ax.set_title(ylabel, fontsize=10.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'], fontsize=10)
        ax.set_ylabel(ylabel.split('(')[0].strip(), fontsize=10)
        ax.grid(True, axis='y', alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    patches = [mpatches.Patch(color=c, label=l, alpha=0.7) for l, c in colors_label.items()]
    fig.legend(handles=patches, loc='upper center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.97), frameon=True)
    fig.suptitle('Triage Vital Sign Distributions by Pneumonia Label (Training Set)', fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT / 'fig_a7_vital_distributions.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'     {out.name}')
    return str(out)

def fig_a8_label_distribution():
    print('  A8: Label distribution by policy...')
    policies = {'u_ignore': 'u_ignore\n(definitive only)', 'u_zero': 'u_zero\n(uncertain \u2192 neg)', 'u_one': 'u_one\n(uncertain \u2192 pos)'}
    splits = ['train', 'validate', 'test']
    split_labels = {'train': 'Training', 'validate': 'Validation', 'test': 'Test'}
    data = {}
    u_one_clinical = MANIFESTS / 'cxr_clinical_pneumonia_training_table_u_one_temporal.parquet'
    multi_policy_clinical = {k: MANIFESTS / f'cxr_clinical_pneumonia_training_table_{k}_temporal.parquet' for k in ['u_ignore', 'u_zero', 'u_one']}
    all_exist = all(p.exists() for p in multi_policy_clinical.values())
    if all_exist:
        for pol_key in ['u_ignore', 'u_zero', 'u_one']:
            d = pd.read_parquet(multi_policy_clinical[pol_key])
            data[pol_key] = {}
            for sp in splits:
                sub = d[d['temporal_split'] == sp]
                data[pol_key][sp] = {'positive': int((sub['target'] == 1).sum()), 'negative': int((sub['target'] == 0).sum()), 'total': len(sub)}
    else:
        raw = pd.read_parquet(u_one_clinical)
        if 'pneumonia_chexpert_raw' not in raw.columns:
            image_table = MANIFESTS / 'cxr_image_pneumonia_finetune_table_u_one_temporal.parquet'
            raw = raw.merge(pd.read_parquet(image_table)[['subject_id', 'study_id', 'pneumonia_chexpert_raw']], on=['subject_id', 'study_id'], how='left')
        for sp in splits:
            sub = raw[raw['temporal_split'] == sp]
            n_pos_def = int((sub['pneumonia_chexpert_raw'] == 1).sum())
            n_neg_def = int((sub['pneumonia_chexpert_raw'] == 0).sum())
            n_uncertain = int((sub['pneumonia_chexpert_raw'] == -1).sum())
            data.setdefault('u_ignore', {})[sp] = {'positive': n_pos_def, 'negative': n_neg_def, 'total': n_pos_def + n_neg_def}
            data.setdefault('u_zero', {})[sp] = {'positive': n_pos_def, 'negative': n_neg_def + n_uncertain, 'total': n_pos_def + n_neg_def + n_uncertain}
            data.setdefault('u_one', {})[sp] = {'positive': n_pos_def + n_uncertain, 'negative': n_neg_def, 'total': n_pos_def + n_neg_def + n_uncertain}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=False)
    c_pos = '#CC3311'
    c_neg = '#0077BB'
    legend_handles = None
    for ax, sp in zip(axes, splits):
        x = np.arange(len(policies))
        pols = list(policies.keys())
        pos_vals = [data[p][sp]['positive'] for p in pols]
        neg_vals = [data[p][sp]['negative'] for p in pols]
        totals = [data[p][sp]['total'] for p in pols]
        bars_neg = ax.bar(x, neg_vals, color=c_neg, alpha=0.82, label='Negative', edgecolor='white', linewidth=0.5)
        bars_pos = ax.bar(x, pos_vals, bottom=neg_vals, color=c_pos, alpha=0.82, label='Positive', edgecolor='white', linewidth=0.5)
        if legend_handles is None:
            legend_handles = [bars_neg, bars_pos]
        max_total = max(totals)
        for xi, pos, tot in zip(x, pos_vals, totals):
            ax.text(xi, tot + max_total * 0.015, f'{pos / tot * 100:.1f}%\nprevalence', ha='center', va='bottom', fontsize=8.5, color='#333333')
        ax.set_xticks(x)
        ax.set_xticklabels([policies[p] for p in pols], fontsize=9)
        ax.set_ylabel('Number of studies')
        ax.set_title(f'{split_labels[sp]} split', fontsize=11, fontweight='bold')
        ax.set_ylim(top=max_total * 1.18)
        ax.grid(True, axis='y', alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=2, fontsize=10, frameon=True)
    fig.suptitle('Label Distribution by Policy and Split: CheXpert Uncertainty Handling', fontsize=12, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = OUT / 'fig_a8_label_distribution.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'     {out.name}')
    return str(out)

def fig_a9_shap_dependence():
    print('  A9: SHAP O2sat dependence plot...')
    shap_df = pd.read_csv(SHAP_VALUES)
    u_ignore_parquet = MANIFESTS / 'cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet'
    u_one_parquet = MANIFESTS / 'cxr_clinical_pneumonia_training_table_u_one_temporal.parquet'
    src_parquet = u_ignore_parquet if u_ignore_parquet.exists() else u_one_parquet
    train_df = pd.read_parquet(src_parquet)
    feat_cols = ['subject_id', 'study_id', 'o2sat', 'heartrate', 'temperature', 'target']
    test_df = train_df[train_df['temporal_split'] == 'test'][feat_cols].copy()
    merged = shap_df.merge(test_df, on=['subject_id', 'study_id'], how='inner', suffixes=('_shap', '_raw'))
    x = merged['o2sat_raw']
    y = merged['o2sat_shap']
    c = merged['heartrate_raw']
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    sc = ax.scatter(x, y, c=c, cmap='RdYlBu_r', s=15, alpha=0.65, vmin=c.quantile(0.05), vmax=c.quantile(0.95), linewidths=0, rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('Heart rate (bpm)', fontsize=10)
    ax.axhline(0, ls='--', lw=1.2, color='#444444', alpha=0.6)
    sorted_idx = np.argsort(x.values)
    x_s = x.values[sorted_idx]
    y_s = y.values[sorted_idx]
    window = max(1, len(x_s) // 20)
    y_smooth = pd.Series(y_s).rolling(window=window, center=True, min_periods=1).mean().values
    ax.plot(x_s, y_smooth, '-', color='#222222', lw=2.0, alpha=0.85, label='Smoothed trend')
    ax.axvspan(x.min(), 90, alpha=0.07, color='#CC3311', label='Severe hypoxemia (<90%)')
    ax.axvspan(90, 94, alpha=0.05, color='#FF9800', label='Mild hypoxemia (90–94%)')
    ax.set_xlabel('O\u2082 saturation: measured value (%)', fontsize=11)
    ax.set_ylabel('SHAP value for O\u2082 saturation\n(contribution to log-odds of pneumonia)', fontsize=11)
    ax.set_title('SHAP Dependence Plot: O\u2082 Saturation\n(XGBoost, vitals+acuity group, n=1,075 test samples)', fontsize=12)
    ax.legend(fontsize=9, loc='lower left', framealpha=0.92)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = OUT / 'fig_a9_shap_dependence_o2sat.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'     {out.name}')
    return str(out)
if __name__ == '__main__':
    print(f'Output directory: {OUT}\n')
    results = {}
    generators = [('A1', fig_a1_pr_curves), ('A2', fig_a2_reliability_panel), ('A3', fig_a3_ablation_bars), ('A4', fig_a4_label_sensitivity), ('A5', fig_a5_training_curves), ('A6', fig_a6_missing_heatmap), ('A7', fig_a7_vital_distributions), ('A8', fig_a8_label_distribution), ('A9', fig_a9_shap_dependence)]
    for tag, fn in generators:
        try:
            path = fn()
            results[tag] = ('OK', path)
        except Exception as e:
            import traceback
            results[tag] = ('FAIL', str(e))
            traceback.print_exc()
    print('\n═══ Summary ═══')
    for tag, (status, info) in results.items():
        icon = '' if status == 'OK' else ''
        print(f'  {icon} {tag}: {status}  {(Path(info).name if status == 'OK' else info)}')
    n_ok = sum((1 for s, _ in results.values() if s == 'OK'))
    print(f'\n{n_ok}/{len(results)} figures generated successfully.')
