#!/usr/bin/env python3
"""
Test all imports before running pipeline.

Usage:
    python test_imports.py
    
Expected output:
    ✓ ALL IMPORTS SUCCESSFUL!
"""

print("Testing imports...")

# =============================================================================
# STANDARD LIBRARY
# =============================================================================
import argparse
import sys
from pathlib import Path
from datetime import datetime
print("  ✓ Standard library")

# =============================================================================
# THIRD-PARTY PACKAGES
# =============================================================================
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
print("  ✓ Third-party packages")

# =============================================================================
# PROJECT MODULES
# =============================================================================

# Config
from src.config import load_config, validate_config, ModelConfig, RunConfig
print("  ✓ src.config")

# Utils
from src.utils import (
    set_seeds, configure_warnings, set_verbose, set_figure_config,
    get_project_root, create_run_folder, copy_config_to_run,
    ensure_results_dirs, log, log_section, log_debug,
    log_environment_versions, get_full_environment_info
)
print("  ✓ src.utils")

# Data loading
from src.data import cell1_load_data
print("  ✓ src.data")

# Features
from src.features import cell2_build_features
print("  ✓ src.features")

# Labels
from src.labels import cell3_build_labels, LABEL_COLS, BENCHMARK_COLS
print("  ✓ src.labels")

# Splits
from src.splits import (
    cell4b_create_splits, merge_train_val_purged,
    validate_embargo_gaps, extract_features_target
)
print("  ✓ src.splits")

# Models
from src.models import (
    cell4c_train_models, cell4d_retrain_for_test,
    set_deterministic, export_cv_results,
    create_lr_model, create_rf_model, create_xgb_model,
    CVResult, LR_PARAM_GRID, RF_PARAM_GRID, XGB_PARAM_GRID
)
print("  ✓ src.models")

# Evaluation
from src.evaluation import (
    compute_baselines, compute_rule_based_baseline,
    cell4d_evaluate_test, cell5a_stability_diagnostics, cell5b_threshold_analysis,
    compute_metrics, encode_labels, decode_labels,
    standardize_probas, CANONICAL_CLASSES, get_proba_columns,
    AbstainPolicy, apply_abstain_policy, evaluate_abstain_policy, compare_abstain_policies
)
print("  ✓ src.evaluation")

# Portfolio
from src.portfolio import (
    cell6_portfolio_backtest, DEFAULT_ALLOCATIONS,
    compute_buy_hold_benchmark, compute_60_40_benchmark, compute_risk_parity_benchmark,
    tc_sensitivity_analysis, run_tc_sensitivity_all_models,
    PerformanceMetrics, compute_performance_metrics
)
print("  ✓ src.portfolio")

# Plots
from src.plots import (
    fig_3_1_class_distributions, fig_3_2_regime_timeline,
    fig_3_3_regime_confusion_matrices, fig_3_4_pca_analysis,
    fig_4_1_ml_confusion_matrices, fig_4_2_feature_importance,
    fig_4_3_model_comparison,
    fig_5_1_rolling_stability, fig_5_2_bootstrap_significance,
    fig_5_3_threshold_analysis,
    fig_6_1_equity_curves, fig_6_2_portfolio_metrics, fig_6_3_tc_sensitivity
)
print("  ✓ src.plots")

# =============================================================================
# VALIDATION
# =============================================================================

# Verify key components
assert callable(load_config), "load_config should be callable"
assert callable(cell1_load_data), "cell1_load_data should be callable"
assert callable(cell4c_train_models), "cell4c_train_models should be callable"
assert callable(cell6_portfolio_backtest), "cell6_portfolio_backtest should be callable"

# Verify LR returns Pipeline (Step 7 fix)
lr_model = create_lr_model()
assert isinstance(lr_model, Pipeline), "LR model should be a Pipeline with StandardScaler"
print("  ✓ LR model is Pipeline (StandardScaler included)")

# Verify canonical classes (Step 8 fix)
assert list(CANONICAL_CLASSES) == [-1, 0, 1], "Canonical classes should be [-1, 0, 1]"
print("  ✓ Canonical class order verified")

# Verify AbstainPolicy exists (Step 11)
policy = AbstainPolicy(threshold=0.45, fallback="neutral")
assert policy.threshold == 0.45, "AbstainPolicy should have threshold attribute"
print("  ✓ AbstainPolicy dataclass verified")

print("\n" + "="*50)
print("✓ ALL IMPORTS SUCCESSFUL!")
print("="*50)