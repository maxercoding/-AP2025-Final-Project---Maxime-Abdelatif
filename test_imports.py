#!/usr/bin/env python3
"""
Test all imports before running pipeline.

Usage:
    conda activate ap2025-maxime
    python test_imports.py
    
Expected output:
    ✓ ALL IMPORTS SUCCESSFUL!

If packages are missing, run:
    conda env create -f environment.yml
    conda activate ap2025-maxime
"""

import sys
import os

print("="*60)
print("ENVIRONMENT CHECK")
print("="*60)

# =============================================================================
# 1) CHECK PYTHON VERSION
# =============================================================================
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"Python version: {python_version}")

if sys.version_info < (3, 10):
    print("⚠️  WARNING: Python 3.10+ recommended. You have:", python_version)

# =============================================================================
# 2) CHECK CONDA ENVIRONMENT
# =============================================================================
conda_env = os.environ.get("CONDA_DEFAULT_ENV", None)
expected_env = "ap2025-maxime"

if conda_env is None:
    print("⚠️  WARNING: No conda environment detected!")
    print("   Run: conda activate ap2025-maxime")
elif conda_env != expected_env:
    print(f"⚠️  WARNING: Wrong environment active!")
    print(f"   Active:   {conda_env}")
    print(f"   Expected: {expected_env}")
    print(f"   Run: conda activate {expected_env}")
else:
    print(f"✓ Correct environment: {conda_env}")

print()
print("="*60)
print("TESTING IMPORTS")
print("="*60)

# =============================================================================
# STANDARD LIBRARY
# =============================================================================
import argparse
from pathlib import Path
from datetime import datetime
print("  ✓ Standard library")

# =============================================================================
# THIRD-PARTY PACKAGES
# =============================================================================
# If any import fails, the error message will tell you which package is missing.
# To install missing packages:
#   conda activate ap2025-maxime
#   conda env update -f environment.yml
# Or individually:
#   pip install <package_name>

try:
    import numpy as np
    print(f"  ✓ numpy ({np.__version__})")
except ImportError:
    print("  ✗ numpy - Install: pip install numpy")
    sys.exit(1)

try:
    import pandas as pd
    print(f"  ✓ pandas ({pd.__version__})")
except ImportError:
    print("  ✗ pandas - Install: pip install pandas")
    sys.exit(1)

try:
    import yaml
    print(f"  ✓ pyyaml")
except ImportError:
    print("  ✗ pyyaml - Install: pip install pyyaml")
    sys.exit(1)

try:
    import matplotlib
    print(f"  ✓ matplotlib ({matplotlib.__version__})")
except ImportError:
    print("  ✗ matplotlib - Install: pip install matplotlib")
    sys.exit(1)

try:
    import seaborn
    print(f"  ✓ seaborn ({seaborn.__version__})")
except ImportError:
    print("  ✗ seaborn - Install: pip install seaborn")
    sys.exit(1)

try:
    import sklearn
    print(f"  ✓ scikit-learn ({sklearn.__version__})")
except ImportError:
    print("  ✗ scikit-learn - Install: pip install scikit-learn")
    sys.exit(1)

try:
    import xgboost as xgb
    print(f"  ✓ xgboost ({xgb.__version__})")
except ImportError:
    print("  ✗ xgboost - Install: pip install xgboost")
    sys.exit(1)

try:
    import openpyxl
    print(f"  ✓ openpyxl ({openpyxl.__version__})")
except ImportError:
    print("  ✗ openpyxl - Install: pip install openpyxl")
    sys.exit(1)

try:
    import streamlit
    print(f"  ✓ streamlit ({streamlit.__version__})")
except ImportError:
    print("  ✗ streamlit - Install: pip install streamlit")
    sys.exit(1)

# Sklearn submodules
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
print("  ✓ sklearn submodules")

print()
print("="*60)
print("TESTING PROJECT MODULES")
print("="*60)

# =============================================================================
# PROJECT MODULES
# =============================================================================

try:
    from src.config import load_config, validate_config, ModelConfig, RunConfig
    print("  ✓ src.config")
except ImportError as e:
    print(f"  ✗ src.config - {e}")
    sys.exit(1)

try:
    from src.utils import (
        set_seeds, configure_warnings, set_verbose, set_figure_config,
        get_project_root, create_run_folder, copy_config_to_run,
        ensure_results_dirs, log, log_section, log_debug,
        log_environment_versions, get_full_environment_info
    )
    print("  ✓ src.utils")
except ImportError as e:
    print(f"  ✗ src.utils - {e}")
    sys.exit(1)

try:
    from src.data import cell1_load_data
    print("  ✓ src.data")
except ImportError as e:
    print(f"  ✗ src.data - {e}")
    sys.exit(1)

try:
    from src.features import cell2_build_features
    print("  ✓ src.features")
except ImportError as e:
    print(f"  ✗ src.features - {e}")
    sys.exit(1)

try:
    from src.labels import cell3_build_labels, LABEL_COLS, BENCHMARK_COLS
    print("  ✓ src.labels")
except ImportError as e:
    print(f"  ✗ src.labels - {e}")
    sys.exit(1)

try:
    from src.splits import (
        cell4b_create_splits, merge_train_val_purged,
        validate_embargo_gaps, extract_features_target
    )
    print("  ✓ src.splits")
except ImportError as e:
    print(f"  ✗ src.splits - {e}")
    sys.exit(1)

try:
    from src.models import (
        cell4c_train_models, cell4d_retrain_for_test,
        set_deterministic, export_cv_results,
        create_lr_model, create_rf_model, create_xgb_model,
        CVResult, LR_PARAM_GRID, RF_PARAM_GRID, XGB_PARAM_GRID
    )
    print("  ✓ src.models")
except ImportError as e:
    print(f"  ✗ src.models - {e}")
    sys.exit(1)

try:
    from src.evaluation import (
        compute_baselines, compute_rule_based_baseline,
        cell4d_evaluate_test, cell5a_stability_diagnostics, cell5b_threshold_analysis,
        compute_metrics, encode_labels, decode_labels,
        standardize_probas, CANONICAL_CLASSES, get_proba_columns,
        AbstainPolicy, apply_abstain_policy, evaluate_abstain_policy, compare_abstain_policies
    )
    print("  ✓ src.evaluation")
except ImportError as e:
    print(f"  ✗ src.evaluation - {e}")
    sys.exit(1)

try:
    from src.portfolio import (
        cell6_portfolio_backtest, DEFAULT_ALLOCATIONS,
        compute_buy_hold_benchmark, compute_60_40_benchmark, compute_risk_parity_benchmark,
        tc_sensitivity_analysis, run_tc_sensitivity_all_models,
        PerformanceMetrics, compute_performance_metrics
    )
    print("  ✓ src.portfolio")
except ImportError as e:
    print(f"  ✗ src.portfolio - {e}")
    sys.exit(1)

try:
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
except ImportError as e:
    print(f"  ✗ src.plots - {e}")
    sys.exit(1)

print()
print("="*60)
print("VALIDATION CHECKS")
print("="*60)

# Verify key components
assert callable(load_config), "load_config should be callable"
assert callable(cell1_load_data), "cell1_load_data should be callable"
assert callable(cell4c_train_models), "cell4c_train_models should be callable"
assert callable(cell6_portfolio_backtest), "cell6_portfolio_backtest should be callable"
print("  ✓ Key functions are callable")

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

print()
print("="*60)
print("✓ ALL IMPORTS SUCCESSFUL!")
print("="*60)
print()
print("You can now run:")
print("  python main.py                    # Run full pipeline")
print("  python Dashboard/run_dashboard.py # Run dashboard")
print()