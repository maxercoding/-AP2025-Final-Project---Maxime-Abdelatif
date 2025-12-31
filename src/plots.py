"""
Plotting functions - Matches Cell 99 from original notebook exactly.
Total: 12 figures

Fig 3.1: Class Distributions (3A, 3B, 3C)
Fig 3.2: Regime Timeline
Fig 3.3: Confusion Matrices (3A vs 3B, 3A vs 3C)
Fig 3.4: PCA Analysis
Fig 4.1: ML Confusion Matrices
Fig 4.2: Feature Importance
Fig 4.3: Model Comparison
Fig 5.1: Rolling Stability
Fig 5.2: Bootstrap Significance
Fig 5.3: Threshold Analysis
Fig 6.1: Portfolio Equity Curves
Fig 6.2: Portfolio Metrics
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

from src.utils import save_fig, show_fig, log


# =============================================================================
# FIGURE CONFIGURATION (matches notebook exactly)
# =============================================================================

COLORS = {
    'bear': '#d62728',
    'neutral': '#dcd066',
    'bull': '#2ca02c',
    'lr': '#1f77b4',
    'rf': '#2ca02c',
    'xgb': '#ff7f0e',
    'static': '#7f7f7f',
    'spx': '#000000',
    'perfect': '#9467bd',
}

REGIME_CMAP = LinearSegmentedColormap.from_list('regime', ['#d62728', '#7f7f7f', '#2ca02c'])

MODEL_COLORS = {
    'logistic_regression': '#1f77b4',
    'random_forest': '#2ca02c',
    'xgboost': '#ff7f0e',
    'LR': '#1f77b4',
    'RF': '#2ca02c',
    'XGB': '#ff7f0e',
    'STACKING': '#9467bd',
}


# =============================================================================
# FIG 3.1: CLASS DISTRIBUTIONS
# =============================================================================

def fig_3_1_class_distributions(
    regime_df: pd.DataFrame,
    filename: str = "fig_3.1_class_distributions.png"
) -> None:
    """
    Fig 3.1: Class Distributions for 3A, 3B, 3C.
    """
    log("Generating Figure 3.1: Class Distributions...")
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 3A: Outcome-based regime
    dist_3a = regime_df["regime_target"].value_counts(normalize=True).reindex([-1, 0, 1])
    axes[0].bar(["Bear", "Neutral", "Bull"], dist_3a.values,
                color=[COLORS['bear'], COLORS['neutral'], COLORS['bull']], edgecolor='black')
    axes[0].set_title("Cell 3A: Outcome-Based Regime", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Proportion")
    axes[0].set_ylim(0, 0.6)
    for i, v in enumerate(dist_3a.values):
        axes[0].text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=10)
    
    # 3B: Benchmark regime
    dist_3b = regime_df["regime_label"].value_counts(normalize=True).reindex([-1, 0, 1])
    axes[1].bar(["Bear", "Neutral", "Bull"], dist_3b.values,
                color=[COLORS['bear'], COLORS['neutral'], COLORS['bull']], edgecolor='black')
    axes[1].set_title("Cell 3B: Rule-Based Benchmark", fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 0.6)
    for i, v in enumerate(dist_3b.values):
        axes[1].text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=10)
    
    # 3C: Unsupervised clusters
    if 'cluster_label' in regime_df.columns:
        dist_3c = regime_df["cluster_label"].value_counts(normalize=True).sort_index()
        axes[2].bar([f"Cluster {i}" for i in dist_3c.index], dist_3c.values,
                    color=['#1f77b4', '#ff7f0e', '#9467bd'][:len(dist_3c)], edgecolor='black')
        axes[2].set_title("Cell 3C: Unsupervised Clusters", fontsize=12, fontweight='bold')
        axes[2].set_ylim(0, 0.6)
        for i, v in enumerate(dist_3c.values):
            axes[2].text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=10)
    
    plt.tight_layout()
    save_fig(filename)
    show_fig()


# =============================================================================
# FIG 3.2: REGIME TIMELINE
# =============================================================================

def fig_3_2_regime_timeline(
    regime_df: pd.DataFrame,
    prices_w: pd.DataFrame,
    train_end: str,
    val_end: str,
    filename: str = "fig_3.2_regime_timeline.png"
) -> None:
    """
    Fig 3.2: Regime Timeline with SPX price overlay.
    """
    log("Generating Figure 3.2: Regime Timeline...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    dates = regime_df.index
    spx_prices = prices_w["SPX"].loc[dates]
    
    regime_sources = [
        ("3A: Outcome-Based", regime_df["regime_target"].values),
        ("3B: Rule-Based", regime_df["regime_label"].values),
    ]
    if 'cluster_label' in regime_df.columns:
        regime_sources.append(("3C: Unsupervised Clusters", regime_df["cluster_label"].values))
    
    for ax_idx, (ax, (title, regime_values)) in enumerate(zip(axes, regime_sources)):
        ax.plot(dates, spx_prices.values, color='black', linewidth=1.5, zorder=3)
        
        if "Cluster" in title:
            cluster_colors = ['#1f77b4', '#ff7f0e', '#9467bd']
            for i in range(len(dates) - 1):
                cluster = int(regime_values[i])
                ax.axvspan(dates[i], dates[i+1], alpha=0.4, color=cluster_colors[cluster], zorder=1)
        else:
            for i in range(len(dates) - 1):
                regime = regime_values[i]
                color = COLORS['bull'] if regime == 1 else (COLORS['bear'] if regime == -1 else COLORS['neutral'])
                ax.axvspan(dates[i], dates[i+1], alpha=0.4, color=color, zorder=1)
        
        ax.axvline(pd.Timestamp(train_end), color='black', linestyle='--', linewidth=2, zorder=4)
        ax.axvline(pd.Timestamp(val_end), color='black', linestyle='--', linewidth=2, zorder=4)
        ax.set_ylabel("SPX Price", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    
    legend_elements = [
        Patch(facecolor=COLORS['bull'], alpha=0.4, label='Bull'),
        Patch(facecolor=COLORS['neutral'], alpha=0.4, label='Neutral'),
        Patch(facecolor=COLORS['bear'], alpha=0.4, label='Bear'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    axes[-1].set_xlabel("Date", fontsize=11)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    
    plt.suptitle("Regime Classification Comparison", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_fig(filename)
    show_fig()


# =============================================================================
# FIG 3.3: CONFUSION MATRICES (3A vs 3B, 3A vs 3C)
# =============================================================================

def fig_3_3_regime_confusion_matrices(
    regime_df: pd.DataFrame,
    unsupervised_analysis: Dict = None,
    filename: str = "fig_3.3_confusion_matrices.png"
) -> None:
    """
    Fig 3.3: Confusion Matrices for 3A vs 3B and 3A vs 3C.
    """
    log("Generating Figure 3.3: Regime Confusion Matrices...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [-1, 0, 1]
    label_names = ["Bear", "Neutral", "Bull"]
    
    y_3a = regime_df["regime_target"].values
    y_3b = regime_df["regime_label"].values
    
    # 3A vs 3B
    cm_3a_3b = np.zeros((3, 3), dtype=int)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm_3a_3b[i, j] = np.sum((y_3a == true_label) & (y_3b == pred_label))
    
    row_sums = cm_3a_3b.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_3a_3b_norm = cm_3a_3b.astype(float) / row_sums
    
    im1 = axes[0].imshow(cm_3a_3b_norm, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    for i in range(3):
        for j in range(3):
            value = cm_3a_3b_norm[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            axes[0].text(j, i, f"{value:.0%}", ha='center', va='center',
                        fontsize=12, fontweight='bold', color=text_color)
    
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_xticklabels(label_names)
    axes[0].set_yticklabels(label_names)
    axes[0].set_xlabel("3B: Rule-Based", fontsize=11)
    axes[0].set_ylabel("3A: Outcome-Based", fontsize=11)
    axes[0].set_title("3A vs 3B Agreement", fontsize=12, fontweight='bold')
    
    # 3A vs 3C
    if 'cluster_label' in regime_df.columns:
        y_3c = regime_df["cluster_label"].values
        cluster_labels = [0, 1, 2]
        cluster_names = ["Cluster 0", "Cluster 1", "Cluster 2"]
        
        cm_3a_3c = np.zeros((3, 3), dtype=int)
        for i, true_label in enumerate(labels):
            for j, cluster in enumerate(cluster_labels):
                cm_3a_3c[i, j] = np.sum((y_3a == true_label) & (y_3c == cluster))
        
        row_sums = cm_3a_3c.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_3a_3c_norm = cm_3a_3c.astype(float) / row_sums
        
        im2 = axes[1].imshow(cm_3a_3c_norm, cmap='Oranges', vmin=0, vmax=1, aspect='equal')
        for i in range(3):
            for j in range(3):
                value = cm_3a_3c_norm[i, j]
                text_color = 'white' if value > 0.5 else 'black'
                axes[1].text(j, i, f"{value:.0%}", ha='center', va='center',
                            fontsize=12, fontweight='bold', color=text_color)
        
        axes[1].set_xticks([0, 1, 2])
        axes[1].set_yticks([0, 1, 2])
        axes[1].set_xticklabels(cluster_names)
        axes[1].set_yticklabels(label_names)
        axes[1].set_xlabel("3C: Unsupervised Cluster", fontsize=11)
        axes[1].set_ylabel("3A: Outcome-Based", fontsize=11)
        
        ari = unsupervised_analysis.get('ari_score', 0) if unsupervised_analysis else 0
        axes[1].set_title(f"3A vs 3C Agreement (ARI={ari:.3f})", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_fig(filename)
    show_fig()


# =============================================================================
# FIG 3.4: PCA ANALYSIS
# =============================================================================

def fig_3_4_pca_analysis(
    regime_df: pd.DataFrame,
    unsupervised_analysis: Dict,
    filename: str = "fig_3.4_pca_analysis.png"
) -> None:
    """
    Fig 3.4: PCA Analysis (3 subplots).
    """
    if unsupervised_analysis is None:
        return
    
    log("Generating Figure 3.4: PCA Analysis...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    pca_components = unsupervised_analysis["pca_components"]
    cluster_labels = unsupervised_analysis["cluster_labels"]
    regime_labels = regime_df["regime_target"].values
    
    # Scatter by cluster
    scatter1 = axes[0].scatter(pca_components[:, 0], pca_components[:, 1],
                                c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
    axes[0].set_xlabel("PC1", fontsize=11)
    axes[0].set_ylabel("PC2", fontsize=11)
    axes[0].set_title("PCA: Colored by Cluster (3C)", fontsize=12, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # Scatter by regime
    scatter2 = axes[1].scatter(pca_components[:, 0], pca_components[:, 1],
                                c=regime_labels, cmap=REGIME_CMAP, alpha=0.6, s=20, vmin=-1, vmax=1)
    axes[1].set_xlabel("PC1", fontsize=11)
    axes[1].set_ylabel("PC2", fontsize=11)
    axes[1].set_title("PCA: Colored by Regime (3A)", fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=axes[1], ticks=[-1, 0, 1])
    cbar2.ax.set_yticklabels(['Bear', 'Neutral', 'Bull'])
    
    # Variance explained
    pca = unsupervised_analysis["pca"]
    var_exp = pca.explained_variance_ratio_
    cumvar = np.cumsum(var_exp)
    
    axes[2].bar(range(1, len(var_exp)+1), var_exp, alpha=0.7, label='Individual')
    axes[2].step(range(1, len(var_exp)+1), cumvar, where='mid', color='red', linewidth=2, label='Cumulative')
    axes[2].axhline(0.8, color='gray', linestyle='--', alpha=0.7)
    axes[2].set_xlabel("Principal Component", fontsize=11)
    axes[2].set_ylabel("Variance Explained", fontsize=11)
    axes[2].set_title("PCA Variance Explained", fontsize=12, fontweight='bold')
    axes[2].legend(loc='center right')
    axes[2].set_ylim(0, 1.05)
    
    plt.tight_layout()
    save_fig(filename)
    show_fig()


# =============================================================================
# FIG 4.1: ML CONFUSION MATRICES
# =============================================================================

def fig_4_1_ml_confusion_matrices(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    metrics: Dict[str, Dict],
    filename: str = "fig_4.1_confusion_matrices.png"
) -> None:
    """
    Fig 4.1: ML Confusion Matrices (4 models).
    """
    log("Generating Figure 4.1: ML Confusion Matrices...")
    
    model_names = ["LR", "RF", "XGB", "STACKING"]
    display_names = ["Logistic Regression", "Random Forest", "XGBoost", "Stacked Ensemble"]
    
    # Filter to available models
    available = [(m, d) for m, d in zip(model_names, display_names) if m in predictions]
    
    fig, axes = plt.subplots(1, len(available), figsize=(4.5 * len(available), 4.5))
    if len(available) == 1:
        axes = [axes]
    
    labels = [-1, 0, 1]
    label_names = ["Bear", "Neutral", "Bull"]
    
    for ax, (name, display_name) in zip(axes, available):
        y_pred = predictions[name]
        
        cm = np.zeros((3, 3), dtype=int)
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
        
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype(float) / row_sums
        
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='equal')
        
        for i in range(3):
            for j in range(3):
                value = cm_norm[i, j]
                text_color = 'white' if value > 0.5 else 'black'
                ax.text(j, i, f"{value:.0%}", ha='center', va='center',
                        fontsize=12, fontweight='bold', color=text_color)
        
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicted", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("True", fontsize=10)
        
        f1 = metrics[name]["macro_f1"]
        ax.set_title(f"{display_name}\nMacro-F1: {f1:.3f}", fontsize=11, fontweight='bold')
    
    plt.suptitle("Confusion Matrices (Normalized by Row)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(filename)
    show_fig()


# =============================================================================
# FIG 4.2: FEATURE IMPORTANCE
# =============================================================================

def fig_4_2_feature_importance(
    models: Dict,
    selected_features_lr: List[str],
    selected_features_tree: List[str],
    filename: str = "fig_4.2_feature_importance.png"
) -> None:
    """
    Fig 4.2: Feature Importance (LR, RF, XGB).
    """
    log("Generating Figure 4.2: Feature Importance...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # LR coefficients (handle Pipeline wrapper)
    if "lr" in models:
        lr_model = models["lr"]
        # LR is wrapped in a Pipeline (StandardScaler + LR)
        if hasattr(lr_model, "named_steps"):
            lr_estimator = lr_model.named_steps.get("lr", lr_model)
        else:
            lr_estimator = lr_model
        
        if hasattr(lr_estimator, "coef_"):
            coef_abs = np.abs(lr_estimator.coef_).mean(axis=0)
            coef_importance = pd.Series(coef_abs, index=selected_features_lr).sort_values(ascending=True).tail(10)
            axes[0].barh(coef_importance.index, coef_importance.values, color='steelblue', edgecolor='black')
            axes[0].set_xlabel("Mean |Coefficient|")
            axes[0].set_title("Logistic Regression\n(Top 10 Features)", fontsize=11, fontweight='bold')
    
    # RF importance
    if "rf" in models:
        rf_model = models["rf"]
        if hasattr(rf_model, "feature_importances_"):
            rf_importance = pd.Series(rf_model.feature_importances_, index=selected_features_tree).sort_values(ascending=True).tail(10)
            axes[1].barh(rf_importance.index, rf_importance.values, color='forestgreen', edgecolor='black')
            axes[1].set_xlabel("Gini Importance")
            axes[1].set_title("Random Forest\n(Top 10 Features)", fontsize=11, fontweight='bold')
    
    # XGB importance
    if "xgb" in models:
        xgb_model = models["xgb"]
        if hasattr(xgb_model, "feature_importances_"):
            xgb_importance = pd.Series(xgb_model.feature_importances_, index=selected_features_tree).sort_values(ascending=True).tail(10)
            axes[2].barh(xgb_importance.index, xgb_importance.values, color='darkorange', edgecolor='black')
            axes[2].set_xlabel("Gain")
            axes[2].set_title("XGBoost\n(Top 10 Features)", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_fig(filename)
    show_fig()


# =============================================================================
# FIG 4.3: MODEL COMPARISON
# =============================================================================

def fig_4_3_model_comparison(
    test_metrics: Dict[str, Dict],
    baseline_metrics: Dict[str, Dict],
    filename: str = "fig_4.3_model_comparison.png"
) -> None:
    """
    Fig 4.3: Model Comparison (F1 scores and per-class recall).
    """
    log("Generating Figure 4.3: Model Comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get metrics
    models = ["Majority", "Random", "Persistence", "LR", "RF", "XGB", "STACKING"]
    f1_scores = [
        baseline_metrics.get("majority_class", {}).get("metrics", {}).get("macro_f1", 0),
        baseline_metrics.get("stratified_random", {}).get("metrics", {}).get("macro_f1", 0),
        baseline_metrics.get("persistence", {}).get("metrics", {}).get("macro_f1", 0),
        test_metrics.get("LR", {}).get("macro_f1", 0),
        test_metrics.get("RF", {}).get("macro_f1", 0),
        test_metrics.get("XGB", {}).get("macro_f1", 0),
        test_metrics.get("STACKING", {}).get("macro_f1", 0),
    ]
    colors_bar = ['#cccccc', '#cccccc', '#cccccc', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    
    bars = axes[0].bar(models, f1_scores, color=colors_bar, edgecolor='black')
    majority_f1 = baseline_metrics.get("majority_class", {}).get("metrics", {}).get("macro_f1", 0)
    axes[0].axhline(majority_f1, color='red', linestyle='--', linewidth=2, label='Majority Baseline')
    axes[0].set_ylabel("Macro-F1", fontsize=11)
    axes[0].set_title("Model Comparison: Macro-F1", fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, max(f1_scores) * 1.15 if f1_scores else 1)
    axes[0].tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars, f1_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Per-class recall
    ml_models = ["LR", "RF", "XGB", "STACKING"]
    x = np.arange(len(ml_models))
    width = 0.25
    
    recall_bear = [test_metrics.get(m, {}).get("recall_bear", 0) for m in ml_models]
    recall_neutral = [test_metrics.get(m, {}).get("recall_neutral", 0) for m in ml_models]
    recall_bull = [test_metrics.get(m, {}).get("recall_bull", 0) for m in ml_models]
    
    axes[1].bar(x - width, recall_bear, width, label='Bear', color='#d62728', edgecolor='black')
    axes[1].bar(x, recall_neutral, width, label='Neutral', color='#7f7f7f', edgecolor='black')
    axes[1].bar(x + width, recall_bull, width, label='Bull', color='#2ca02c', edgecolor='black')
    
    axes[1].set_ylabel("Recall", fontsize=11)
    axes[1].set_title("Per-Class Recall by Model", fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ml_models)
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    save_fig(filename)
    show_fig()


# =============================================================================
# FIG 5.1: ROLLING STABILITY
# =============================================================================

def fig_5_1_rolling_stability(
    rolling_metrics: Dict[str, pd.DataFrame],
    baseline_f1: float,
    filename: str = "fig_5.1_rolling_stability.png"
) -> None:
    """
    Fig 5.1: Rolling Stability (F1 and Bear Recall).
    """
    log("Generating Figure 5.1: Rolling Stability...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    model_colors = {'LR': '#1f77b4', 'RF': '#2ca02c', 'XGB': '#ff7f0e',
                    'logistic_regression': '#1f77b4', 'random_forest': '#2ca02c', 'xgboost': '#ff7f0e'}
    
    # Rolling Macro-F1
    for name, df in rolling_metrics.items():
        color = model_colors.get(name, '#1f77b4')
        axes[0].plot(df.index, df["macro_f1"], label=name.replace("_", " ").title(),
                    color=color, linewidth=1.5)
    
    axes[0].axhline(baseline_f1, color='red', linestyle='--', linewidth=2, label='Majority Baseline')
    axes[0].set_ylabel("Macro-F1", fontsize=11)
    axes[0].set_title("Rolling 52-Week Macro-F1", fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].set_ylim(0, 0.7)
    
    # Rolling Bear Recall
    last_df = None
    for name, df in rolling_metrics.items():
        if "recall_bear" in df.columns:
            color = model_colors.get(name, '#1f77b4')
            axes[1].plot(df.index, df["recall_bear"], label=name.replace("_", " ").title(),
                        color=color, linewidth=1.5)
            last_df = df
    
    if last_df is not None:
        axes[1].axhline(0.2, color='red', linestyle='--', linewidth=2, label='Stress Threshold (20%)')
        axes[1].fill_between(last_df.index, 0, 0.2, alpha=0.2, color='red', label='Stress Zone')
    
    axes[1].set_ylabel("Bear Recall", fontsize=11)
    axes[1].set_xlabel("Date", fontsize=11)
    axes[1].set_title("Rolling 52-Week Bear Recall (Stress Test)", fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].set_ylim(0, 1)
    
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    save_fig(filename)
    show_fig()


# =============================================================================
# FIG 5.2: BOOTSTRAP SIGNIFICANCE
# =============================================================================

def fig_5_2_bootstrap_significance(
    bootstrap_results: Dict[str, Dict],
    baseline_f1: float,
    filename: str = "fig_5.2_bootstrap_significance.png"
) -> None:
    """
    Fig 5.2: Bootstrap Significance with CI.
    """
    log("Generating Figure 5.2: Bootstrap Significance...")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    model_colors = {'LR': '#1f77b4', 'RF': '#2ca02c', 'XGB': '#ff7f0e',
                    'logistic_regression': '#1f77b4', 'random_forest': '#2ca02c', 'xgboost': '#ff7f0e'}
    
    models = list(bootstrap_results.keys())
    observed = [bootstrap_results[m]["observed"] for m in models]
    ci_lower = [bootstrap_results[m]["ci_lower"] for m in models]
    ci_upper = [bootstrap_results[m]["ci_upper"] for m in models]
    
    y_pos = np.arange(len(models))
    colors = [model_colors.get(m, '#1f77b4') for m in models]
    
    ax.barh(y_pos, observed, xerr=[np.array(observed)-np.array(ci_lower),
                                    np.array(ci_upper)-np.array(observed)],
            color=colors, capsize=5, edgecolor='black', height=0.6)
    
    ax.axvline(baseline_f1, color='red', linewidth=2, linestyle='--', label='Majority Baseline')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace("_", " ").title() for m in models])
    ax.set_xlabel("Macro-F1", fontsize=11)
    ax.set_title("Model Performance with 95% Confidence Intervals", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    
    for i, m in enumerate(models):
        if ci_lower[i] > baseline_f1:
            ax.text(ci_upper[i] + 0.01, i, "**", fontsize=14, va='center', color='green', fontweight='bold')
        elif observed[i] > baseline_f1:
            ax.text(ci_upper[i] + 0.01, i, "*", fontsize=14, va='center', color='orange', fontweight='bold')
    
    ax.text(0.95, 0.05, "** = CI excludes baseline\n* = Point estimate beats baseline",
            transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_fig(filename)
    show_fig()


# =============================================================================
# FIG 5.3: THRESHOLD ANALYSIS
# =============================================================================

def fig_5_3_threshold_analysis(
    threshold_results: Dict[str, pd.DataFrame],
    baseline_f1: float,
    filename: str = "fig_5.3_threshold_analysis.png"
) -> None:
    """
    Fig 5.3: Threshold Analysis (4 subplots).
    """
    log("Generating Figure 5.3: Threshold Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    model_colors = {'LR': '#1f77b4', 'RF': '#2ca02c', 'XGB': '#ff7f0e',
                    'logistic_regression': '#1f77b4', 'random_forest': '#2ca02c', 'xgboost': '#ff7f0e'}
    
    # Macro-F1 vs threshold
    for name, df in threshold_results.items():
        color = model_colors.get(name, '#1f77b4')
        axes[0, 0].plot(df["threshold"], df["macro_f1"],
                        label=name.replace("_", " ").title(),
                        color=color, linewidth=2, marker='o', markersize=4)
    
    axes[0, 0].axhline(baseline_f1, color='red', linestyle='--', linewidth=2, label='Majority Baseline')
    axes[0, 0].set_xlabel("Confidence Threshold τ", fontsize=11)
    axes[0, 0].set_ylabel("Macro-F1", fontsize=11)
    axes[0, 0].set_title("Macro-F1 vs Confidence Threshold", fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=9)
    
    # Coverage vs threshold
    for name, df in threshold_results.items():
        color = model_colors.get(name, '#1f77b4')
        axes[0, 1].plot(df["threshold"], df["coverage"],
                        label=name.replace("_", " ").title(),
                        color=color, linewidth=2, marker='o', markersize=4)
    
    axes[0, 1].axhline(0.5, color='gray', linestyle='--', linewidth=2, label='50% Coverage')
    axes[0, 1].set_xlabel("Confidence Threshold τ", fontsize=11)
    axes[0, 1].set_ylabel("Coverage (non-Neutral)", fontsize=11)
    axes[0, 1].set_title("Coverage vs Confidence Threshold", fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best', fontsize=9)
    
    # Bear recall vs threshold
    for name, df in threshold_results.items():
        if "recall_bear" in df.columns:
            color = model_colors.get(name, '#1f77b4')
            axes[1, 0].plot(df["threshold"], df["recall_bear"],
                            label=name.replace("_", " ").title(),
                            color=color, linewidth=2, marker='o', markersize=4)
    
    axes[1, 0].set_xlabel("Confidence Threshold τ", fontsize=11)
    axes[1, 0].set_ylabel("Bear Recall", fontsize=11)
    axes[1, 0].set_title("Bear Recall vs Confidence Threshold", fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='best', fontsize=9)
    
    # Bull recall vs threshold
    for name, df in threshold_results.items():
        if "recall_bull" in df.columns:
            color = model_colors.get(name, '#1f77b4')
            axes[1, 1].plot(df["threshold"], df["recall_bull"],
                            label=name.replace("_", " ").title(),
                            color=color, linewidth=2, marker='o', markersize=4)
    
    axes[1, 1].set_xlabel("Confidence Threshold τ", fontsize=11)
    axes[1, 1].set_ylabel("Bull Recall", fontsize=11)
    axes[1, 1].set_title("Bull Recall vs Confidence Threshold", fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='best', fontsize=9)
    
    plt.suptitle("Threshold Analysis: 'Neutral as Uncertainty'", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(filename)
    show_fig()


# =============================================================================
# FIG 6.1: PORTFOLIO EQUITY CURVES
# =============================================================================

def fig_6_1_equity_curves(
    portfolio_results: Dict[str, Dict],
    filename: str = "fig_6.1_equity_curves.png"
) -> None:
    """
    Fig 6.1: Portfolio Equity Curves and Drawdowns.
    """
    log("Generating Figure 6.1: Portfolio Equity Curves...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    strategy_colors = {
        'LR': '#1f77b4', 'RF': '#2ca02c', 'XGB': '#ff7f0e',
        'STACKING': '#9467bd',
        '60_40': '#7f7f7f', 'BuyHold_SPX': '#000000', 'Oracle': '#9467bd',
        'logistic_regression': '#1f77b4', 'random_forest': '#2ca02c', 'xgboost': '#ff7f0e',
        'static_60_30_10': '#7f7f7f', 'buy_hold_spx': '#000000', 'perfect_foresight': '#9467bd',
    }
    
    strategy_labels = {
        'LR': 'LR Strategy', 'RF': 'RF Strategy', 'XGB': 'XGB Strategy',
        'STACKING': 'Stacking Strategy',
        '60_40': 'Static 60/40', 'BuyHold_SPX': 'Buy & Hold SPX', 'Oracle': 'Perfect Foresight',
    }
    
    # Equity curves
    equity_curves = portfolio_results.get("equity_curves", {})
    performance = portfolio_results.get("performance", {})
    
    for name, equity in equity_curves.items():
        color = strategy_colors.get(name, '#1f77b4')
        linestyle = '--' if name in ['60_40', 'Oracle', 'BuyHold_SPX'] else '-'
        label = strategy_labels.get(name, name)
        
        sharpe = performance.get(name, None)
        if sharpe and hasattr(sharpe, 'sharpe_ratio'):
            label = f"{label} (Sharpe: {sharpe.sharpe_ratio:.2f})"
        
        axes[0].plot(equity.index, equity.values, label=label,
                    color=color, linewidth=1.5, linestyle=linestyle)
    
    axes[0].set_ylabel("Portfolio Value (Start=100)", fontsize=11)
    axes[0].set_title("Portfolio Equity Curves (Test Period)", fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=9, ncol=2)
    
    # Drawdowns
    for name in ['XGB', 'xgboost', '60_40', 'BuyHold_SPX', 'static_60_30_10', 'buy_hold_spx']:
        if name in equity_curves:
            equity = equity_curves[name]
            cummax = equity.cummax()
            drawdown = (equity - cummax) / cummax * 100
            color = strategy_colors.get(name, '#1f77b4')
            label = strategy_labels.get(name, name)
            axes[1].fill_between(equity.index, drawdown.values, 0, alpha=0.4, color=color, label=label)
    
    axes[1].set_ylabel("Drawdown (%)", fontsize=11)
    axes[1].set_xlabel("Date", fontsize=11)
    axes[1].set_title("Drawdown Comparison", fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].set_ylim(-40, 5)
    
    plt.tight_layout()
    save_fig(filename)
    show_fig()


# =============================================================================
# FIG 6.2: PORTFOLIO METRICS COMPARISON
# =============================================================================

def fig_6_2_portfolio_metrics(
    portfolio_results: Dict[str, Dict],
    filename: str = "fig_6.2_portfolio_metrics.png"
) -> None:
    """
    Fig 6.2: Portfolio Metrics (CAGR, Sharpe, MaxDD).
    """
    log("Generating Figure 6.2: Portfolio Metrics...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    performance = portfolio_results.get("performance", {})
    
    strategies = ['LR', 'RF', 'XGB', '60_40', 'BuyHold_SPX']
    labels = ['LR', 'RF', 'XGB', 'Static', 'B&H SPX']
    colors_bar = ['#1f77b4', '#2ca02c', '#ff7f0e', '#7f7f7f', '#000000']
    
    # Filter to available strategies
    available = [(s, l, c) for s, l, c in zip(strategies, labels, colors_bar) if s in performance]
    
    if not available:
        log("No portfolio performance data available for plotting.")
        return
    
    strategies_avail, labels_avail, colors_avail = zip(*available)
    
    # CAGR
    cagr = [performance[s].cagr * 100 if hasattr(performance[s], 'cagr') else 0 for s in strategies_avail]
    axes[0].bar(labels_avail, cagr, color=colors_avail, edgecolor='black')
    axes[0].set_ylabel("CAGR (%)", fontsize=11)
    axes[0].set_title("Annualized Return", fontsize=12, fontweight='bold')
    
    # Sharpe
    sharpe = [performance[s].sharpe_ratio if hasattr(performance[s], 'sharpe_ratio') else 0 for s in strategies_avail]
    axes[1].bar(labels_avail, sharpe, color=colors_avail, edgecolor='black')
    axes[1].set_ylabel("Sharpe Ratio", fontsize=11)
    axes[1].set_title("Risk-Adjusted Return", fontsize=12, fontweight='bold')
    
    # Max Drawdown
    maxdd = [performance[s].max_drawdown * 100 if hasattr(performance[s], 'max_drawdown') else 0 for s in strategies_avail]
    axes[2].bar(labels_avail, maxdd, color=colors_avail, edgecolor='black')
    axes[2].set_ylabel("Max Drawdown (%)", fontsize=11)
    axes[2].set_title("Maximum Drawdown", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_fig(filename)
    show_fig()


def fig_6_3_tc_sensitivity(
    tc_results: Dict[str, pd.DataFrame],
    filename: str = "fig_6.3_tc_sensitivity.png"
) -> None:
    """
    Fig 6.3: Transaction Cost Sensitivity Analysis.
    
    Shows how Sharpe ratio degrades with increasing transaction costs.
    """
    log("Generating Figure 6.3: TC Sensitivity...")
    
    if not tc_results:
        log("  No TC sensitivity data available.")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {'LR': '#1f77b4', 'RF': '#2ca02c', 'XGB': '#ff7f0e', 'STACKING': '#9467bd'}
    markers = {'LR': 'o', 'RF': 's', 'XGB': '^', 'STACKING': 'D'}
    
    for model_name, tc_df in tc_results.items():
        color = colors.get(model_name, '#333333')
        marker = markers.get(model_name, 'o')
        
        ax.plot(
            tc_df['tc_bps'], 
            tc_df['sharpe_ratio'], 
            marker=marker, 
            label=model_name,
            color=color,
            linewidth=2,
            markersize=8
        )
    
    ax.set_xlabel("Transaction Cost (bps)", fontsize=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_title("Transaction Cost Sensitivity Analysis", fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Add annotation for typical institutional TC
    ax.axvline(x=10, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(11, ax.get_ylim()[1]*0.95, 'Typical\nInstitutional', fontsize=9, color='gray')
    
    plt.tight_layout()
    save_fig(filename)
    show_fig()


# =============================================================================
# GENERATE ALL FIGURES (Cell 99 equivalent)
# =============================================================================

def generate_all_figures(
    regime_df: pd.DataFrame,
    prices_w: pd.DataFrame,
    unsupervised_analysis: Dict,
    y_test: pd.Series,
    predictions: Dict[str, np.ndarray],
    test_metrics: Dict[str, Dict],
    baseline_metrics: Dict[str, Dict],
    models: Dict,
    selected_features_lr: List[str],
    selected_features_tree: List[str],
    rolling_metrics: Dict[str, pd.DataFrame],
    bootstrap_results: Dict[str, Dict],
    threshold_results: Dict[str, pd.DataFrame],
    portfolio_results: Dict,
    train_end: str,
    val_end: str,
) -> None:
    """
    Generate all 12 figures (Cell 99 equivalent).
    """
    log("\n" + "="*80)
    log("CELL 99: GENERATING ALL FIGURES")
    log("="*80)
    
    # Fig 3.x
    fig_3_1_class_distributions(regime_df)
    fig_3_2_regime_timeline(regime_df, prices_w, train_end, val_end)
    fig_3_3_regime_confusion_matrices(regime_df, unsupervised_analysis)
    fig_3_4_pca_analysis(regime_df, unsupervised_analysis)
    
    # Fig 4.x
    fig_4_1_ml_confusion_matrices(y_test.values, predictions, test_metrics)
    fig_4_2_feature_importance(models, selected_features_lr, selected_features_tree)
    
    majority_f1 = baseline_metrics.get("majority_class", {}).get("metrics", {}).get("macro_f1", 0.2)
    fig_4_3_model_comparison(test_metrics, baseline_metrics)
    
    # Fig 5.x
    fig_5_1_rolling_stability(rolling_metrics, majority_f1)
    fig_5_2_bootstrap_significance(bootstrap_results, majority_f1)
    fig_5_3_threshold_analysis(threshold_results, majority_f1)
    
    # Fig 6.x
    fig_6_1_equity_curves(portfolio_results)
    fig_6_2_portfolio_metrics(portfolio_results)
    
    log("\n" + "="*80)
    log("✓ CELL 99 COMPLETE: ALL 12 FIGURES GENERATED")
    log("="*80)