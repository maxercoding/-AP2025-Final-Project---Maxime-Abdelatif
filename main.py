#!/usr/bin/env python3
"""
ML Market Regime Detection - Main Pipeline
==========================================
Minimal orchestrator that calls cell functions in notebook order.
Generates all 12 figures matching Cell 99 from original notebook.

Usage:
    python main.py
    python main.py --config config/config.yaml
"""
from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Third-party
import numpy as np
import pandas as pd

# Project modules
from src.config import load_config, validate_config
from src.utils import (
    set_seeds, configure_warnings, set_verbose, set_figure_config,
    get_project_root, create_run_folder, copy_config_to_run, log, log_section,
    log_environment_versions  # for reproducibility logging
)
from src.data import cell1_load_data
from src.features import cell2_build_features
from src.labels import cell3_build_labels
from src.splits import cell4b_create_splits, merge_train_val_purged
from src.models import (
    cell4c_train_models, cell4d_retrain_for_test,
    set_deterministic, export_cv_results
)
from src.evaluation import (
    compute_baselines, compute_rule_based_baseline, cell4d_evaluate_test,
    cell5a_stability_diagnostics, cell5b_threshold_analysis
)
from src.portfolio import cell6_portfolio_backtest, DEFAULT_ALLOCATIONS
from src.plots import (
    fig_3_1_class_distributions, fig_3_2_regime_timeline,
    fig_3_3_regime_confusion_matrices, fig_3_4_pca_analysis,
    fig_4_1_ml_confusion_matrices, fig_4_2_feature_importance,
    fig_4_3_model_comparison,
    fig_5_1_rolling_stability, fig_5_2_bootstrap_significance,
    fig_5_3_threshold_analysis,
    fig_6_1_equity_curves, fig_6_2_portfolio_metrics, fig_6_3_tc_sensitivity
)


# =============================================================================
# CELL 0: SETUP
# =============================================================================

def cell0_setup(config_path: Path = None):
    """Cell 0: Load config and set up environment."""
    log_section("CELL 0: CONFIGURATION & SETUP")
    
    project_root = get_project_root()
    
    if config_path is None:
        config_path = project_root / "config" / "config.yaml"
    
    cfg = load_config(config_path)
    validate_config(cfg)
    
    # Reproducibility settings
    set_seeds(cfg.run.seed)
    set_deterministic(cfg.run.deterministic)
    configure_warnings()
    set_verbose(cfg.model.verbose)
    
    # Create timestamped run folder (audit-proof)
    results_dir = project_root / cfg.paths.results_dir
    results_dirs = create_run_folder(results_dir)
    
    # Copy config to run folder for reproducibility
    copy_config_to_run(config_path, results_dirs["run_dir"])
    
    set_figure_config(cfg.run.save_figs, cfg.run.show_figs, results_dirs["figures"])
    
    # Log environment for reproducibility
    log_environment_versions(
        logfile=results_dirs["logs"] / "run.log",
        save_csv=results_dirs["tables"] / "environment_versions.csv"
    )
    
    log(f"✓ Config loaded | Seed={cfg.run.seed} | H={cfg.model.horizon} | K={cfg.model.k}")
    log(f"  Deterministic: {cfg.run.deterministic}")
    
    return cfg, project_root, results_dirs


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(config_path: Path = None) -> dict:
    """Execute the full ML pipeline."""
    
    start_time = datetime.now()
    
    # -------------------------------------------------------------------------
    # CELL 0: Setup
    # -------------------------------------------------------------------------
    cfg, project_root, results_dirs = cell0_setup(config_path)
    
    # -------------------------------------------------------------------------
    # CELL 1: Data Loading
    # -------------------------------------------------------------------------
    df_daily, df_w, prices_w = cell1_load_data(
        excel_path=project_root / cfg.paths.raw_excel,
        sheet_name=cfg.paths.sheet_name,
        freq=cfg.model.freq,
        vol_win=cfg.model.vol_win,
        horizon=cfg.model.horizon
    )
    
    # -------------------------------------------------------------------------
    # CELL 2: Feature Engineering
    # -------------------------------------------------------------------------
    features_df, prices_w, FEATURE_COLS, FEATURE_META = cell2_build_features(
        df_w=df_w,
        prices_w=prices_w
    )
    
    # -------------------------------------------------------------------------
    # CELL 3: Label Construction
    # -------------------------------------------------------------------------
    regime_df, EXCLUDE_COLS, LABEL_PARAMS, UNSUPERVISED = cell3_build_labels(
        features_df=features_df,
        prices_w=prices_w,
        horizon=cfg.model.horizon,
        vol_win=cfg.model.vol_win,
        k=cfg.model.k,
        persist_frac=cfg.model.persist_frac,
        use_persistence=cfg.model.use_persistence,
        train_end=cfg.model.train_end,
        seed=cfg.run.seed
    )
    
    # -------------------------------------------------------------------------
    # CELL 4B: Splits & Preprocessing
    # -------------------------------------------------------------------------
    splits = cell4b_create_splits(
        regime_df=regime_df,
        exclude_cols=EXCLUDE_COLS,
        train_end=cfg.model.train_end,
        val_end=cfg.model.val_end,
        embargo_weeks=cfg.model.embargo,
        seed=cfg.run.seed
    )
    
    # Export embargo validation for audit
    splits["embargo_validation"].to_csv(
        results_dirs["tables"] / "embargo_validation.csv", index=False
    )
    
    # Baselines
    baselines = compute_baselines(splits["y_train"], splits["y_test"], cfg.run.seed)
    
    # Add rule-based baselines (uses features)
    rule_baselines = compute_rule_based_baseline(
        X_test=splits["X_test"],
        y_test=splits["y_test"],
        return_col="SPX_ret_1w",      # 1-week S&P return
        vol_col="SPX_ret_vol_12w"     # 12-week rolling volatility
    )
    baselines.update(rule_baselines)
    
    # -------------------------------------------------------------------------
    # CELL 4C: Model Training (CV)
    # -------------------------------------------------------------------------
    training = cell4c_train_models(
        X_train=splits["X_train"],
        y_train=splits["y_train"],
        selected_features_lr=splits["selected_features_lr"],
        selected_features_tree=splits["selected_features_tree"],
        class_weights=splits["class_weights"],
        class_weights_tree=splits["class_weights_tree"],  # Damped weights for trees
        tscv=splits["tscv"],
        seed=cfg.run.seed
    )
    
    # Export CV search results for audit
    export_cv_results(
        training["cv_results"],
        results_dirs["tables"] / "cv_search_results.csv"
    )
    
    # -------------------------------------------------------------------------
    # CELL 4D: Retrain & Test Evaluation
    # -------------------------------------------------------------------------
    X_trainval, y_trainval = merge_train_val_purged(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        cfg.model.embargo
    )
    
    models_final = cell4d_retrain_for_test(
        X_trainval=X_trainval,
        y_trainval=y_trainval,
        selected_features_lr=splits["selected_features_lr"],
        selected_features_tree=splits["selected_features_tree"],
        class_weights=splits["class_weights"],  # Not used, recomputed internally
        best_params=training["best_params"],
        seed=cfg.run.seed
    )
    
    test_eval = cell4d_evaluate_test(
        models=models_final,
        X_test=splits["X_test"],
        y_test=splits["y_test"],
        selected_features_lr=splits["selected_features_lr"],
        selected_features_tree=splits["selected_features_tree"],
        baselines=baselines,
        seed=cfg.run.seed
    )
    
    # Save results table
    test_eval["results_df"].to_csv(results_dirs["tables"] / "test_results.csv")
    
    # -------------------------------------------------------------------------
    # Select best ML model (exclude baselines)
    # -------------------------------------------------------------------------
    ml_models = list(test_eval["predictions"].keys())
    ml_results = test_eval["results_df"].loc[test_eval["results_df"].index.isin(ml_models)]
    best_model = ml_results["macro_f1"].idxmax()
    log(f"\n✓ Best ML model: {best_model} (Macro-F1={test_eval['metrics'][best_model]['macro_f1']:.4f})")
    
    # -------------------------------------------------------------------------
    # CELL 5A: Stability Diagnostics
    # -------------------------------------------------------------------------
    baseline_f1 = baselines["stratified_random"]["metrics"]["macro_f1"]
    majority_f1 = baselines["majority_class"]["metrics"]["macro_f1"]
    
    stability = cell5a_stability_diagnostics(
        y_test=splits["y_test"],
        y_pred=test_eval["predictions"][best_model],
        baseline_f1=baseline_f1,
        seed=cfg.run.seed
    )
    
    # -------------------------------------------------------------------------
    # CELL 5B: Threshold Analysis
    # -------------------------------------------------------------------------
    # Get majority class from training set for fallback policy
    majority_class = int(splits["y_train"].mode().iloc[0])
    
    threshold = cell5b_threshold_analysis(
        y_test=splits["y_test"],
        y_proba=test_eval["probas"][best_model],
        majority_class=majority_class
    )
    
    # Export policy comparison
    if "policy_comparison" in threshold:
        threshold["policy_comparison"].to_csv(
            results_dirs["tables"] / "abstain_policy_comparison.csv", index=False
        )
    
    # -------------------------------------------------------------------------
    # CELL 6: Portfolio Backtest
    # -------------------------------------------------------------------------
    portfolio = cell6_portfolio_backtest(
        prices_w=prices_w,
        y_test=splits["y_test"],
        predictions=test_eval["predictions"],
        allocation_map=DEFAULT_ALLOCATIONS,
        transaction_cost_bps=10.0
    )
    
    # Save portfolio summary
    portfolio["summary_df"].to_csv(results_dirs["tables"] / "portfolio_summary.csv")
    
    # Save TC sensitivity results
    if "tc_sensitivity" in portfolio:
        # Combine all models into one DataFrame
        tc_dfs = []
        for model_name, tc_df in portfolio["tc_sensitivity"].items():
            tc_dfs.append(tc_df)
        if tc_dfs:
            tc_combined = pd.concat(tc_dfs, ignore_index=True)
            tc_combined.to_csv(results_dirs["tables"] / "tc_sensitivity.csv", index=False)
    
    # -------------------------------------------------------------------------
    # CELL 99: GENERATE ALL FIGURES
    # -------------------------------------------------------------------------
    if cfg.run.save_figs:
        log_section("CELL 99: GENERATING ALL FIGURES")
        
        # Fig 3.1: Class Distributions
        fig_3_1_class_distributions(regime_df)
        
        # Fig 3.2: Regime Timeline
        fig_3_2_regime_timeline(regime_df, prices_w, cfg.model.train_end, cfg.model.val_end)
        
        # Fig 3.3: Confusion Matrices (3A vs 3B, 3A vs 3C)
        fig_3_3_regime_confusion_matrices(regime_df, UNSUPERVISED)
        
        # Fig 3.4: PCA Analysis
        fig_3_4_pca_analysis(regime_df, UNSUPERVISED)
        
        # Fig 4.1: ML Confusion Matrices
        fig_4_1_ml_confusion_matrices(
            splits["y_test"].values,
            test_eval["predictions"],
            test_eval["metrics"]
        )
        
        # Fig 4.2: Feature Importance
        fig_4_2_feature_importance(
            models_final,
            splits["selected_features_lr"],
            splits["selected_features_tree"]
        )
        
        # Fig 4.3: Model Comparison
        fig_4_3_model_comparison(test_eval["metrics"], baselines)
        
        # Fig 5.1: Rolling Stability (need rolling metrics for multiple models)
        # For now, use single model rolling metrics
        rolling_metrics_dict = {best_model: stability["rolling_metrics"]}
        fig_5_1_rolling_stability(rolling_metrics_dict, majority_f1)
        
        # Fig 5.2: Bootstrap Significance
        bootstrap_results = {
            best_model: {
                "observed": stability["observed"],
                "ci_lower": stability["ci_lower"],
                "ci_upper": stability["ci_upper"]
            }
        }
        fig_5_2_bootstrap_significance(bootstrap_results, majority_f1)
        
        # Fig 5.3: Threshold Analysis
        threshold_results = {best_model: threshold["threshold_df"]}
        fig_5_3_threshold_analysis(threshold_results, majority_f1)
        
        # Fig 6.1: Equity Curves
        fig_6_1_equity_curves(portfolio)
        
        # Fig 6.2: Portfolio Metrics
        fig_6_2_portfolio_metrics(portfolio)
        
        # Fig 6.3: TC Sensitivity (NEW)
        if "tc_sensitivity" in portfolio:
            fig_6_3_tc_sensitivity(portfolio["tc_sensitivity"])
        
        log("✓ All figures generated")
    
    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    duration = (datetime.now() - start_time).total_seconds()
    
    log_section("PIPELINE COMPLETE")
    log(f"  Runtime: {duration:.1f}s")
    log(f"  Best ML model: {best_model} (Macro-F1={test_eval['metrics'][best_model]['macro_f1']:.4f})")
    log(f"  Bootstrap 95% CI: [{stability['ci_lower']:.4f}, {stability['ci_upper']:.4f}]")
    log(f"  Significant vs random: {'Yes ✓' if stability['significant'] else 'No'}")
    
    if best_model in portfolio["performance"]:
        perf = portfolio["performance"][best_model]
        log(f"  Portfolio Sharpe: {perf.sharpe_ratio:.2f}")
        log(f"  Portfolio CAGR: {perf.cagr:.1%}")
    
    log(f"\n  📁 Run folder: {results_dirs['run_dir']}")
    log(f"     Tables: {results_dirs['tables'].name}/")
    log(f"     Figures: {results_dirs['figures'].name}/")
    
    # Return all results
    return {
        "cfg": cfg,
        "regime_df": regime_df,
        "prices_w": prices_w,
        "UNSUPERVISED": UNSUPERVISED,
        "splits": splits,
        "baselines": baselines,
        "training": training,
        "models_final": models_final,
        "test_eval": test_eval,
        "best_model": best_model,
        "stability": stability,
        "threshold": threshold,
        "portfolio": portfolio,
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ML Market Regime Detection")
    parser.add_argument("--config", "-c", type=Path, default=None)
    args = parser.parse_args()
    
    try:
        run_pipeline(config_path=args.config)
        return 0
    except Exception as e:
        log(f"✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())