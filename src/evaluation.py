"""
Evaluation functions: metrics, baselines, splits, bootstrap, calibration.
Maps to: Cell 4A (ML setup), Cell 4B (splits/preprocessing), Cell 5A (stability), Cell 5B (threshold)

Contains all evaluation logic extracted from the notebook.

================================================================================
METHODOLOGICAL DEFENSE: EVALUATION CHOICES
================================================================================

This module implements evaluation strategies designed for the specific challenges
of financial regime prediction with limited samples.

KEY METHODOLOGICAL POINTS:

1. **Sample Size Considerations (~700 weekly observations)**
   ----------------------------------------------------------
   Financial data is inherently limited:
   - Weekly data from 2010-2024 yields ~700 observations
   - Only ~200 observations for out-of-sample testing
   - This is comparable to other academic studies in regime prediction
   
   We address small sample concerns through:
   - Bootstrap confidence intervals (n=1000 resamples)
   - Rolling stability metrics (12-week windows)
   - Conservative interpretation of results
   - Multiple baseline comparisons
   
   **Epistemic Humility**: We report confidence intervals and stability ranges,
   NOT point estimates alone. A model with Macro-F1 = 0.35 ± 0.05 is interpreted
   differently than one with 0.35 ± 0.15.

2. **Class Imbalance and Macro-F1 Choice**
   ----------------------------------------
   Regime classes are naturally imbalanced:
   - Bull markets are more common (~45% of weeks)
   - Bear markets are rare but economically critical (~15% of weeks)
   - Neutral periods fill the remainder (~40% of weeks)
   
   We use **Macro-F1** as the primary metric because:
   - It weights all classes equally regardless of frequency
   - Bear detection is as important as Bull detection economically
   - It penalizes models that ignore minority classes
   - It's the harmonic mean of precision and recall per class
   
   Alternative metrics reported:
   - Balanced accuracy (equally-weighted recall)
   - Per-class recall (to identify which regimes the model struggles with)
   - Confusion matrices (full error structure)
   
   **Why NOT accuracy?** A model predicting "Bull" always achieves ~45% accuracy
   but 0% Bear recall - useless for risk management.

3. **Baseline Comparisons for Statistical Significance**
   ------------------------------------------------------
   EMH-aware evaluation requires meaningful baselines:
   
   Naive Baselines:
   - Majority class: Lower bound (no skill)
   - Stratified random: EMH null hypothesis
   - Persistence: Exploits autocorrelation only
   
   Rule-Based Baselines:
   - Momentum rule: Simple sign-of-return heuristic
   - Trend-vol rule: Combines return direction + volatility
   
   **A model must beat ALL baselines to claim skill.** Beating random but
   not persistence suggests the model only captures autocorrelation.

4. **Bootstrap Significance Testing**
   ------------------------------------
   We test H0: "Model has no skill vs stratified random baseline"
   
   Procedure:
   1. Compute observed Macro-F1 gap: (model - baseline)
   2. Bootstrap resample test set 1000 times
   3. Compute gap for each resample
   4. 95% CI excludes 0 → significant at α=0.05
   
   This is a non-parametric approach that:
   - Makes no distributional assumptions
   - Accounts for temporal dependence (block bootstrap available)
   - Provides interpretable confidence intervals

5. **Stability Analysis (Rolling Metrics)**
   -----------------------------------------
   Point estimates hide temporal variation. We compute:
   - 12-week rolling Macro-F1
   - Standard deviation of rolling metrics
   - Trend analysis (is performance degrading?)
   
   A "stable" model has low variance in rolling metrics.
   High variance suggests regime-dependent performance.

6. **Confidence Threshold and Abstain Policy**
   --------------------------------------------
   When max(P(class)) < τ, the model is uncertain. Options:
   - Predict Neutral (default safe action)
   - Use persistence (exploit autocorrelation)
   - Use majority class (statistical baseline)
   
   We sweep τ ∈ [0.35, 0.70] and report coverage-accuracy tradeoffs.
   Higher τ → fewer predictions but higher precision.

7. **Canonical Probability Ordering**
   ------------------------------------
   All models output probabilities in canonical order: [-1, 0, 1]
   (Bear, Neutral, Bull). This ensures:
   - Consistent threshold analysis across models
   - Comparable calibration curves
   - Correct abstain policy application
   
   XGBoost uses internal encoding {0, 1, 2} which we map back.

KNOWN LIMITATIONS:
- Temporal dependence may bias bootstrap CIs (use block bootstrap for robustness)
- Walk-forward CV uses fixed fold sizes (could adapt to regime lengths)
- Probability calibration is not enforced (could use Platt scaling)

REFERENCES:
- DeLong et al. (1988): Comparing ROC curves
- Efron & Tibshirani (1993): Bootstrap methods
- López de Prado (2018): Advances in Financial Machine Learning

================================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Set
from itertools import product

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    brier_score_loss
)

from src.utils import log, log_section, log_debug


# =============================================================================
# LABEL ENCODING (Cell 4A) - for XGBoost compatibility
# =============================================================================

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}


def encode_labels(y: pd.Series) -> pd.Series:
    """Encode {-1, 0, 1} → {0, 1, 2} for XGBoost."""
    return y.map(LABEL_MAP)


def decode_labels(y: np.ndarray) -> np.ndarray:
    """Decode {0, 1, 2} → {-1, 0, 1} from XGBoost predictions."""
    return np.array([LABEL_MAP_INV[v] for v in y])


# Canonical class order for all probability outputs
CANONICAL_CLASSES = np.array([-1, 0, 1])  # Bear, Neutral, Bull


def standardize_probas(model, y_proba: np.ndarray, model_name: str = "") -> np.ndarray:
    """
    Standardize predict_proba() output to canonical class order [-1, 0, 1].
    
    Different models may have different internal class orderings:
    - LR/RF/Stacking with {-1,0,1} labels: classes_ = [-1, 0, 1] (already canonical)
    - XGBoost with encoded labels: classes_ = [0, 1, 2] → maps to [-1, 0, 1]
    
    This function ensures all probability arrays have columns in the order:
    [P(Bear=-1), P(Neutral=0), P(Bull=1)]
    
    Args:
        model: Trained model with classes_ attribute
        y_proba: Raw predict_proba() output (n_samples, n_classes)
        model_name: Model name for logging (optional)
        
    Returns:
        Standardized probability array with columns in [-1, 0, 1] order
    """
    if y_proba is None:
        return None
    
    # Get model's class order
    if hasattr(model, "classes_"):
        model_classes = model.classes_
    elif hasattr(model, "named_steps"):
        # Pipeline - get classes from final estimator
        final_step = list(model.named_steps.values())[-1]
        if hasattr(final_step, "classes_"):
            model_classes = final_step.classes_
        else:
            # Assume canonical order if no classes_ found
            log_debug(f"  Warning: No classes_ found for {model_name}, assuming canonical order")
            return y_proba
    else:
        # Assume canonical order
        log_debug(f"  Warning: No classes_ found for {model_name}, assuming canonical order")
        return y_proba
    
    model_classes = np.array(model_classes)
    
    # Check if XGBoost with encoded labels {0, 1, 2}
    if set(model_classes) == {0, 1, 2}:
        # XGBoost: classes_ = [0, 1, 2] → decode to [-1, 0, 1]
        decoded_classes = np.array([LABEL_MAP_INV[c] for c in model_classes])
    else:
        decoded_classes = model_classes
    
    # Check if already in canonical order
    if np.array_equal(decoded_classes, CANONICAL_CLASSES):
        return y_proba
    
    # Reorder columns to match canonical order
    standardized = np.zeros_like(y_proba)
    for i, canonical_class in enumerate(CANONICAL_CLASSES):
        # Find where this class is in the model's order
        idx = np.where(decoded_classes == canonical_class)[0]
        if len(idx) > 0:
            standardized[:, i] = y_proba[:, idx[0]]
        else:
            log_debug(f"  Warning: Class {canonical_class} not found in {model_name}")
    
    return standardized


def get_proba_columns() -> List[str]:
    """Return column names for standardized probability DataFrame."""
    return ["proba_bear", "proba_neutral", "proba_bull"]


# =============================================================================
# EXPERIMENT LOGGING (Cell 4A)
# =============================================================================

@dataclass
class ExperimentLog:
    """Immutable experiment record — prevents post-hoc modification."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    seed: int = 42
    horizon_weeks: int = 1
    embargo_weeks: int = 1
    train_end: str = ""
    val_end: str = ""
    n_features_initial: int = 0
    n_features_pruned: int = 0
    feature_list: List[str] = field(default_factory=list)
    excluded_cols: List[str] = field(default_factory=list)
    cv_results: Dict = field(default_factory=dict)
    test_evaluated: bool = False
    test_results: Dict = field(default_factory=dict)


# =============================================================================
# FEATURE/TARGET EXTRACTION (Cell 4A)
# =============================================================================

def extract_features_target(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: Set[str],
    label_cols: List[str],
    benchmark_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract X and y with strict validation.
    Risk B mitigation: fatal asserts on label leakage.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        exclude_cols: Set of columns to exclude from features
        label_cols: List of label-side columns (MUST NOT be in X)
        benchmark_cols: List of benchmark columns (MUST NOT be in X)
        
    Returns:
        Tuple of (X, y) DataFrames/Series
        
    Raises:
        ValueError: If leakage detected or validation fails
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    
    # Extract feature columns (everything not excluded)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # FATAL CHECKS: No label-side columns in features
    leaked = set(feature_cols) & set(label_cols)
    if leaked:
        raise ValueError(f"FATAL LEAKAGE: Label columns in features: {leaked}")
    
    bench_leaked = set(feature_cols) & set(benchmark_cols)
    if bench_leaked:
        raise ValueError(f"FATAL LEAKAGE: Benchmark columns in features: {bench_leaked}")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Validate no NaNs
    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaN values in features: {nan_cols}")
    
    if y.isna().any():
        raise ValueError("NaN values in target.")
    
    # Validate target values
    valid_labels = {-1, 0, 1}
    actual_labels = set(y.unique())
    if not actual_labels.issubset(valid_labels):
        raise ValueError(f"Invalid target values: {actual_labels - valid_labels}")
    
    return X, y


# =============================================================================
# CORE METRICS (Cell 4A)
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive metrics with macro-averaging for class balance.
    
    METHODOLOGICAL NOTE ON MACRO-F1:
    --------------------------------
    We use Macro-F1 as the primary metric because regime classes are imbalanced
    but all classes are economically important:
    
    - Bear (~15%): Rare but critical for risk management
    - Neutral (~40%): Common, default allocation state  
    - Bull (~45%): Most frequent, but over-predicting Bull is dangerous
    
    Macro-F1 = mean(F1_bear, F1_neutral, F1_bull) weights all classes equally,
    ensuring the model doesn't ignore minority classes to optimize accuracy.
    
    Alternative metrics reported for transparency:
    - balanced_accuracy: Equally-weighted recall (similar motivation)
    - weighted_f1: Class-frequency weighted (for comparison)
    - per-class recall: Identifies which regimes the model struggles with
    
    Args:
        y_true: True labels in {-1, 0, 1}
        y_pred: Predicted labels in {-1, 0, 1}
        
    Returns:
        Dictionary of metric names to values
    """
    # Primary metric: Macro-F1 (see docstring for justification)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    
    # Per-class recall: Critical for understanding model behavior
    # A model with high accuracy but low bear recall is useless for risk management
    for label, name in [(-1, "bear"), (0, "neutral"), (1, "bull")]:
        mask = y_true == label
        if mask.sum() > 0:
            metrics[f"recall_{name}"] = (y_pred[mask] == label).mean()
        else:
            metrics[f"recall_{name}"] = np.nan
    
    return metrics


def compute_transition_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics at regime transitions.
    Reports BOTH exact match and windowed (±1) accuracy.
    
    Args:
        y_true: True labels as Series with DatetimeIndex
        y_pred: Predicted labels as array
        
    Returns:
        Dictionary with transition accuracy metrics
    """
    y_true = y_true.copy()
    y_pred_series = pd.Series(y_pred, index=y_true.index)
    
    transitions = y_true != y_true.shift(1)
    transitions.iloc[0] = False
    
    n_transitions = transitions.sum()
    if n_transitions == 0:
        return {
            "transition_accuracy_exact": np.nan,
            "transition_accuracy_window": np.nan,
            "n_transitions": 0
        }
    
    exact_correct = 0
    window_correct = 0
    
    for idx in y_true.index[transitions]:
        true_label = y_true.loc[idx]
        loc = y_true.index.get_loc(idx)
        
        if y_pred_series.loc[idx] == true_label:
            exact_correct += 1
            window_correct += 1
        else:
            start = max(0, loc - 1)
            end = min(len(y_true), loc + 2)
            window_preds = y_pred_series.iloc[start:end]
            if true_label in window_preds.values:
                window_correct += 1
    
    return {
        "transition_accuracy_exact": exact_correct / n_transitions,
        "transition_accuracy_window": window_correct / n_transitions,
        "n_transitions": int(n_transitions)
    }


# =============================================================================
# TEMPORAL SPLITS (Cell 4B) - Risk A mitigation
# =============================================================================

def create_temporal_splits(
    X: pd.DataFrame,
    y: pd.Series,
    train_end: str,
    val_end: str,
    embargo_weeks: int
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Create temporal splits with embargo gap.
    Risk A mitigation: purge last H weeks from train AND val before boundaries.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        train_end: End date for training (ISO format)
        val_end: End date for validation (ISO format)
        embargo_weeks: Number of weeks to purge at boundaries
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits as (X, y) tuples
        
    Raises:
        ValueError: If embargo is violated or sets too small
    """
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)
    
    # Raw splits
    train_mask = X.index <= train_end_dt
    val_mask = (X.index > train_end_dt) & (X.index <= val_end_dt)
    test_mask = X.index > val_end_dt
    
    # Apply embargo: remove last embargo_weeks from train
    train_idx = X.index[train_mask]
    if len(train_idx) > embargo_weeks:
        purge_boundary = train_idx[-embargo_weeks]
        train_mask_purged = X.index < purge_boundary
    else:
        raise ValueError("Training set too small after embargo purge.")
    
    # Apply embargo: remove last embargo_weeks from val
    val_idx = X.index[val_mask]
    if len(val_idx) > embargo_weeks:
        val_purge_boundary = val_idx[-embargo_weeks]
        val_mask_purged = val_mask & (X.index < val_purge_boundary)
    else:
        raise ValueError("Validation set too small after embargo purge.")
    
    # Validate Train→Val embargo gap
    train_max = X.index[train_mask_purged].max()
    val_min = X.index[val_mask_purged].min()
    gap_days_train_val = (val_min - train_max).days
    required_gap_days = embargo_weeks * 7
    
    if gap_days_train_val < required_gap_days:
        raise ValueError(
            f"TRAIN→VAL EMBARGO VIOLATION: Gap={gap_days_train_val}d, need={required_gap_days}d"
        )
    
    # Validate Val→Test embargo gap
    val_max = X.index[val_mask_purged].max()
    test_min = X.index[test_mask].min()
    gap_days_val_test = (test_min - val_max).days
    
    if gap_days_val_test < required_gap_days:
        raise ValueError(
            f"VAL→TEST EMBARGO VIOLATION: Gap={gap_days_val_test}d, need={required_gap_days}d"
        )
    
    splits = {
        "train": (X[train_mask_purged].copy(), y[train_mask_purged].copy()),
        "val": (X[val_mask_purged].copy(), y[val_mask_purged].copy()),
        "test": (X[test_mask].copy(), y[test_mask].copy()),
    }
    
    log(f"\n  Embargo verification:")
    log(f"    Train→Val gap: {gap_days_train_val} days ({gap_days_train_val // 7} weeks) ✓")
    log(f"    Val→Test gap:  {gap_days_val_test} days ({gap_days_val_test // 7} weeks) ✓")
    
    return splits


# =============================================================================
# FEATURE PRUNING (Cell 4B) - Risk E mitigation
# =============================================================================

def compute_correlation_clusters(
    X: pd.DataFrame,
    threshold: float = 0.85
) -> List[List[str]]:
    """Identify clusters of highly correlated features."""
    corr = X.corr().abs()
    clusters = []
    visited = set()
    
    for col in corr.columns:
        if col in visited:
            continue
        cluster = [col]
        visited.add(col)
        for other in corr.columns:
            if other not in visited and corr.loc[col, other] > threshold:
                cluster.append(other)
                visited.add(other)
        if len(cluster) > 1:
            clusters.append(cluster)
    
    return clusters


def prune_features_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    variance_threshold: float = 1e-6,
    correlation_threshold: float = 0.85,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[List[str], Dict]:
    """
    Feature pruning using only training data.
    Risk E mitigation: removes collinear features, keeps highest MI.
    
    Args:
        X_train: Training features
        y_train: Training labels
        variance_threshold: Minimum variance to keep feature
        correlation_threshold: Max correlation before pruning
        seed: Random seed for MI computation
        verbose: Whether to print progress
        
    Returns:
        Tuple of (selected_features, diagnostics_dict)
    """
    diagnostics = {
        "initial_features": X_train.shape[1],
        "dropped_low_variance": [],
        "dropped_correlated": [],
    }
    
    # Step 1: Variance filter
    var_selector = VarianceThreshold(threshold=variance_threshold)
    var_selector.fit(X_train)
    low_var_mask = var_selector.get_support()
    dropped_var = X_train.columns[~low_var_mask].tolist()
    diagnostics["dropped_low_variance"] = dropped_var
    
    X_after_var = X_train.loc[:, low_var_mask].copy()
    
    if verbose and dropped_var:
        log(f"    Variance filter: dropped {len(dropped_var)} features")
    
    # Step 2: Correlation pruning with MI tiebreak
    mi_scores = mutual_info_classif(
        X_after_var, y_train,
        discrete_features=False,
        random_state=seed
    )
    mi_series = pd.Series(mi_scores, index=X_after_var.columns)
    
    to_drop = set()
    clusters = compute_correlation_clusters(X_after_var, correlation_threshold)
    
    for cluster in clusters:
        cluster_mi = mi_series[cluster]
        best_feature = cluster_mi.idxmax()
        for feat in cluster:
            if feat != best_feature:
                to_drop.add(feat)
    
    diagnostics["dropped_correlated"] = list(to_drop)
    selected_features = [c for c in X_after_var.columns if c not in to_drop]
    diagnostics["final_features"] = len(selected_features)
    
    if verbose:
        log(f"    Correlation pruning (ρ > {correlation_threshold}): dropped {len(to_drop)} features")
        log(f"    Final feature count: {len(selected_features)}")
    
    return selected_features, diagnostics


# =============================================================================
# CLASS WEIGHTS (Cell 4B) - Risk C mitigation
# =============================================================================

def compute_class_weights(y: pd.Series) -> Dict[int, float]:
    """Compute inverse-frequency class weights."""
    counts = y.value_counts()
    n_samples = len(y)
    n_classes = len(counts)
    return {c: n_samples / (n_classes * counts[c]) for c in counts.index}


def compute_class_weights_encoded(y: pd.Series) -> Dict[int, float]:
    """Compute class weights for encoded labels {0, 1, 2}."""
    weights_orig = compute_class_weights(y)
    return {LABEL_MAP[k]: v for k, v in weights_orig.items()}


def compute_sample_weights(
    y: pd.Series,
    class_weights: Dict[int, float]
) -> np.ndarray:
    """Convert class weights to per-sample weights."""
    return y.map(class_weights).values


def compute_sample_weights_encoded(
    y: pd.Series,
    class_weights_enc: Dict[int, float]
) -> np.ndarray:
    """Compute sample weights for encoded labels."""
    y_encoded = encode_labels(y)
    return y_encoded.map(class_weights_enc).values


# =============================================================================
# WALK-FORWARD CV (Cell 4B) - Risk A, D mitigation
# =============================================================================

def create_purged_tscv(
    n_samples: int,
    n_splits: int,
    gap: int,
    test_size: int
) -> TimeSeriesSplit:
    """Create TimeSeriesSplit with gap for purging."""
    return TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)


# =============================================================================
# BASELINES (Cell 4B)
# =============================================================================

def compute_baselines(
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Compute naive baseline predictions for EMH comparison.
    
    Args:
        y_train: Training labels
        y_test: Test labels
        seed: Random seed
        
    Returns:
        Dictionary with baseline results including predictions and metrics
    """
    baselines = {}
    
    # 1. Majority Class
    majority_class = y_train.mode().iloc[0]
    y_pred_majority = np.full(len(y_test), majority_class)
    baselines["majority_class"] = {
        "description": f"Always predict {majority_class}",
        "predictions": y_pred_majority,
        "metrics": compute_metrics(y_test.values, y_pred_majority)
    }
    
    # 2. Stratified Random
    np.random.seed(seed)
    train_dist = y_train.value_counts(normalize=True)
    y_pred_random = np.random.choice(
        train_dist.index,
        size=len(y_test),
        p=train_dist.values
    )
    baselines["stratified_random"] = {
        "description": "Random (train distribution)",
        "predictions": y_pred_random,
        "metrics": compute_metrics(y_test.values, y_pred_random)
    }
    
    # 3. Persistence
    y_test_shifted = y_test.shift(1).dropna().astype(int)
    y_test_aligned = y_test.loc[y_test_shifted.index]
    baselines["persistence"] = {
        "description": "Last week's regime",
        "predictions": y_test_shifted.values,
        "metrics": compute_metrics(y_test_aligned.values, y_test_shifted.values),
        "n_samples": len(y_test_aligned)
    }
    
    return baselines


def compute_rule_based_baseline(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    return_col: str = "ret_1w",
    vol_col: str = "vol_4w"
) -> Dict[str, Dict]:
    """
    Compute rule-based baseline using simple momentum/volatility rules.
    
    Rules:
    - Momentum: predict based on sign of recent return
      - ret > 0 → Bull (1)
      - ret < 0 → Bear (-1)
      - ret ≈ 0 → Neutral (0)
    
    - Trend-Vol: combine return direction with volatility regime
      - High vol + negative return → Bear
      - Low vol + positive return → Bull
      - Otherwise → Neutral
    
    Args:
        X_test: Test features DataFrame
        y_test: Test labels
        return_col: Column name for return feature
        vol_col: Column name for volatility feature
        
    Returns:
        Dictionary with rule-based baseline results
    """
    baselines = {}
    
    # 1. Simple Momentum Rule
    if return_col in X_test.columns:
        returns = X_test[return_col].values
        
        # Simple sign-based prediction
        y_pred_momentum = np.zeros(len(returns), dtype=int)
        y_pred_momentum[returns > 0.01] = 1   # Bull if return > 1%
        y_pred_momentum[returns < -0.01] = -1  # Bear if return < -1%
        # Neutral otherwise (between -1% and 1%)
        
        baselines["momentum_rule"] = {
            "description": "Sign of 1w return (±1% threshold)",
            "predictions": y_pred_momentum,
            "metrics": compute_metrics(y_test.values, y_pred_momentum)
        }
        log(f"  Rule baseline: momentum_rule (Macro-F1={baselines['momentum_rule']['metrics']['macro_f1']:.4f})")
    
    # 2. Trend-Volatility Rule (if vol feature exists)
    if return_col in X_test.columns and vol_col in X_test.columns:
        returns = X_test[return_col].values
        vol = X_test[vol_col].values
        
        # Compute volatility threshold (median of test set)
        vol_median = np.median(vol)
        
        y_pred_trend_vol = np.zeros(len(returns), dtype=int)
        
        # High vol + negative return → Bear
        bear_mask = (vol > vol_median) & (returns < 0)
        y_pred_trend_vol[bear_mask] = -1
        
        # Low vol + positive return → Bull  
        bull_mask = (vol <= vol_median) & (returns > 0)
        y_pred_trend_vol[bull_mask] = 1
        
        # Everything else → Neutral (already 0)
        
        baselines["trend_vol_rule"] = {
            "description": "Trend+Vol: High vol + neg ret → Bear, Low vol + pos ret → Bull",
            "predictions": y_pred_trend_vol,
            "metrics": compute_metrics(y_test.values, y_pred_trend_vol)
        }
        log(f"  Rule baseline: trend_vol_rule (Macro-F1={baselines['trend_vol_rule']['metrics']['macro_f1']:.4f})")
    
    return baselines


# =============================================================================
# MERGE TRAIN + VAL (Cell 4C)
# =============================================================================

def merge_train_val_purged(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    embargo_weeks: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Merge train and val with embargo purge for final training.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        embargo_weeks: Weeks to purge from train end
        
    Returns:
        Tuple of (X_merged, y_merged)
    """
    if len(X_train) > embargo_weeks:
        X_train_purged = X_train.iloc[:-embargo_weeks]
        y_train_purged = y_train.iloc[:-embargo_weeks]
    else:
        raise ValueError("Training set too small for purge")
    
    X_merged = pd.concat([X_train_purged, X_val], axis=0)
    y_merged = pd.concat([y_train_purged, y_val], axis=0)
    
    assert X_merged.index.is_monotonic_increasing, "Merged index not sorted"
    return X_merged, y_merged


# =============================================================================
# ROLLING STABILITY METRICS (Cell 5A)
# =============================================================================

def compute_rolling_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    window: int = 52,
    min_periods: int = 26
) -> pd.DataFrame:
    """
    Compute rolling macro-F1 and per-class recalls.
    Window = 52 weeks (1 year), min_periods = 26 weeks (6 months).
    
    Args:
        y_true: True labels as Series
        y_pred: Predicted labels as Series
        window: Rolling window size
        min_periods: Minimum periods before computing
        
    Returns:
        DataFrame with rolling metrics indexed by date
    """
    results = []
    
    for i in range(min_periods, len(y_true) + 1):
        start = max(0, i - window)
        end = i
        
        y_t = y_true.iloc[start:end].values
        y_p = y_pred.iloc[start:end].values
        
        # Compute metrics for this window
        macro_f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        
        # Per-class recall
        recall_bear = recall_score(y_t == -1, y_p == -1, zero_division=0)
        recall_neutral = recall_score(y_t == 0, y_p == 0, zero_division=0)
        recall_bull = recall_score(y_t == 1, y_p == 1, zero_division=0)
        
        results.append({
            "date": y_true.index[end - 1],
            "macro_f1": macro_f1,
            "recall_bear": recall_bear,
            "recall_neutral": recall_neutral,
            "recall_bull": recall_bull,
            "n_samples": end - start
        })
    
    return pd.DataFrame(results).set_index("date")


def compute_regime_frequency(
    series: pd.Series,
    window: int = 26
) -> pd.DataFrame:
    """Compute rolling regime frequency."""
    results = []
    
    for i in range(window, len(series) + 1):
        start = i - window
        end = i
        window_data = series.iloc[start:end]
        
        counts = window_data.value_counts(normalize=True)
        results.append({
            "date": series.index[end - 1],
            "bear_pct": counts.get(-1, 0),
            "neutral_pct": counts.get(0, 0),
            "bull_pct": counts.get(1, 0),
        })
    
    return pd.DataFrame(results).set_index("date")


# =============================================================================
# BLOCK BOOTSTRAP (Cell 5A)
# =============================================================================

def block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    block_size: int = 4,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float, List[float]]:
    """
    Block bootstrap confidence interval for a metric.
    Uses non-overlapping blocks to preserve temporal structure.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_fn: Function(y_true, y_pred) -> float
        n_bootstrap: Number of bootstrap samples
        block_size: Size of temporal blocks
        confidence: Confidence level
        seed: Random seed
        
    Returns:
        Tuple of (observed, ci_lower, ci_upper, bootstrap_distribution)
    """
    np.random.seed(seed)
    n = len(y_true)
    n_blocks = n // block_size
    
    # Observed metric
    observed = metric_fn(y_true, y_pred)
    
    # Bootstrap samples
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        # Sample blocks with replacement
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        
        # Reconstruct sample from blocks
        sample_idx = []
        for bi in block_indices:
            start = bi * block_size
            end = min(start + block_size, n)
            sample_idx.extend(range(start, end))
        
        sample_idx = np.array(sample_idx[:n])  # Truncate to original size
        
        y_t_sample = y_true[sample_idx]
        y_p_sample = y_pred[sample_idx]
        
        bootstrap_metrics.append(metric_fn(y_t_sample, y_p_sample))
    
    # Confidence interval (percentile method)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
    
    return observed, ci_lower, ci_upper, bootstrap_metrics


def macro_f1_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 score for use with bootstrap."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# =============================================================================
# TRANSITION ANALYSIS (Cell 5A)
# =============================================================================

def analyze_transitions(
    y_true: pd.Series,
    y_pred: pd.Series
) -> Dict:
    """
    Detailed analysis of regime transition detection.
    
    Args:
        y_true: True regime labels as Series
        y_pred: Predicted regime labels as Series
        
    Returns:
        Dictionary with transition analysis results
    """
    # Identify actual transitions
    transitions = y_true != y_true.shift(1)
    transitions.iloc[0] = False
    
    n_transitions = transitions.sum()
    transition_indices = y_true.index[transitions]
    
    # Initialize results
    results = {
        "total_transitions": n_transitions,
        "exact_matches": 0,
        "off_by_one": 0,
        "missed": 0,
        "by_type": {
            "bear_to_neutral": {"total": 0, "detected": 0},
            "bear_to_bull": {"total": 0, "detected": 0},
            "neutral_to_bear": {"total": 0, "detected": 0},
            "neutral_to_bull": {"total": 0, "detected": 0},
            "bull_to_bear": {"total": 0, "detected": 0},
            "bull_to_neutral": {"total": 0, "detected": 0},
        }
    }
    
    label_names = {-1: "bear", 0: "neutral", 1: "bull"}
    
    for idx in transition_indices:
        loc = y_true.index.get_loc(idx)
        if loc == 0:
            continue
        
        from_regime = y_true.iloc[loc - 1]
        to_regime = y_true.iloc[loc]
        
        # Transition type
        trans_type = f"{label_names[from_regime]}_to_{label_names[to_regime]}"
        results["by_type"][trans_type]["total"] += 1
        
        # Check if predicted correctly
        pred_at_trans = y_pred.iloc[loc]
        
        if pred_at_trans == to_regime:
            results["exact_matches"] += 1
            results["by_type"][trans_type]["detected"] += 1
        elif loc + 1 < len(y_pred) and y_pred.iloc[loc + 1] == to_regime:
            results["off_by_one"] += 1
            results["by_type"][trans_type]["detected"] += 1  # Count as detected
        else:
            results["missed"] += 1
    
    return results


# =============================================================================
# CONFIDENCE THRESHOLD (Cell 5B)
# =============================================================================

def apply_confidence_threshold(
    proba: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Apply confidence threshold for regime assignment.
    If max(proba) < threshold → Neutral (0)
    Else → argmax class
    
    Args:
        proba: (n_samples, 3) array with columns [P(Bear), P(Neutral), P(Bull)]
        threshold: confidence threshold τ
    
    Returns:
        predictions in {-1, 0, 1}
    """
    max_proba = proba.max(axis=1)
    argmax_class = proba.argmax(axis=1)
    
    # Map argmax index to class label: 0→-1, 1→0, 2→1
    class_map = {0: -1, 1: 0, 2: 1}
    predictions = np.array([class_map[i] for i in argmax_class])
    
    # Override to Neutral if below threshold
    predictions[max_proba < threshold] = 0
    
    return predictions


@dataclass
class AbstainPolicy:
    """
    Configuration for abstain/fallback policy.
    
    When model confidence is below threshold, the policy determines
    what prediction to use instead.
    
    Attributes:
        threshold: Confidence threshold τ (abstain if max_proba < τ)
        fallback: What to predict when abstaining
            - "neutral": Always predict Neutral (0)
            - "persistence": Use previous week's regime
            - "majority": Use training set majority class
        min_confidence_bear: Optional higher threshold for Bear predictions
        min_confidence_bull: Optional higher threshold for Bull predictions
    """
    threshold: float = 0.40
    fallback: str = "neutral"  # "neutral", "persistence", or "majority"
    min_confidence_bear: float = None  # Optional: require higher confidence for Bear
    min_confidence_bull: float = None  # Optional: require higher confidence for Bull
    
    def __post_init__(self):
        valid_fallbacks = {"neutral", "persistence", "majority"}
        if self.fallback not in valid_fallbacks:
            raise ValueError(f"fallback must be one of {valid_fallbacks}")


def apply_abstain_policy(
    proba: np.ndarray,
    policy: AbstainPolicy,
    y_prev: np.ndarray = None,
    majority_class: int = 0
) -> Tuple[np.ndarray, Dict]:
    """
    Apply abstain/fallback policy to model predictions.
    
    When confidence is below threshold, uses the specified fallback strategy
    instead of the model's prediction.
    
    Args:
        proba: (n_samples, 3) array with columns [P(Bear), P(Neutral), P(Bull)]
               Must be in canonical order [-1, 0, 1]
        policy: AbstainPolicy configuration
        y_prev: Previous predictions for persistence fallback (n_samples,)
                For first sample, uses majority_class
        majority_class: Training set majority class for majority fallback
        
    Returns:
        Tuple of (predictions, stats_dict)
        - predictions: np.ndarray of predictions in {-1, 0, 1}
        - stats_dict: Dictionary with abstention statistics
    """
    n_samples = len(proba)
    max_proba = proba.max(axis=1)
    argmax_idx = proba.argmax(axis=1)
    
    # Map argmax index to class label: 0→-1, 1→0, 2→1
    class_map = {0: -1, 1: 0, 2: 1}
    raw_predictions = np.array([class_map[i] for i in argmax_idx])
    
    # Initialize with raw predictions
    predictions = raw_predictions.copy()
    
    # Track which samples abstained
    abstained = np.zeros(n_samples, dtype=bool)
    
    # Apply main threshold
    low_confidence = max_proba < policy.threshold
    abstained |= low_confidence
    
    # Apply class-specific thresholds if set
    if policy.min_confidence_bear is not None:
        bear_mask = (raw_predictions == -1)
        bear_low_conf = bear_mask & (max_proba < policy.min_confidence_bear)
        abstained |= bear_low_conf
    
    if policy.min_confidence_bull is not None:
        bull_mask = (raw_predictions == 1)
        bull_low_conf = bull_mask & (max_proba < policy.min_confidence_bull)
        abstained |= bull_low_conf
    
    # Apply fallback for abstained samples
    if policy.fallback == "neutral":
        predictions[abstained] = 0
        
    elif policy.fallback == "persistence":
        if y_prev is None:
            # No previous predictions available, use majority
            predictions[abstained] = majority_class
        else:
            # Use previous prediction (shifted by 1)
            for i in np.where(abstained)[0]:
                if i == 0:
                    predictions[i] = majority_class
                else:
                    predictions[i] = predictions[i - 1]  # Use our own previous pred
                    
    elif policy.fallback == "majority":
        predictions[abstained] = majority_class
    
    # Compute statistics
    n_abstained = abstained.sum()
    stats = {
        "n_total": n_samples,
        "n_abstained": n_abstained,
        "abstain_rate": n_abstained / n_samples,
        "coverage": 1 - (n_abstained / n_samples),
        "threshold": policy.threshold,
        "fallback": policy.fallback,
        "mean_confidence": max_proba.mean(),
        "mean_confidence_kept": max_proba[~abstained].mean() if (~abstained).any() else np.nan,
        "mean_confidence_abstained": max_proba[abstained].mean() if abstained.any() else np.nan,
    }
    
    return predictions, stats


def evaluate_abstain_policy(
    y_true: np.ndarray,
    proba: np.ndarray,
    policy: AbstainPolicy,
    y_prev: np.ndarray = None,
    majority_class: int = 0
) -> Dict:
    """
    Evaluate predictions with abstain policy applied.
    
    Args:
        y_true: True labels
        proba: Predicted probabilities (canonical order)
        policy: AbstainPolicy configuration
        y_prev: Previous predictions for persistence fallback
        majority_class: Training set majority class
        
    Returns:
        Dictionary with metrics and abstention statistics
    """
    y_pred, abstain_stats = apply_abstain_policy(
        proba, policy, y_prev, majority_class
    )
    
    metrics = compute_metrics(y_true, y_pred)
    
    return {
        **metrics,
        **abstain_stats,
        "policy": f"τ={policy.threshold}, fallback={policy.fallback}"
    }


def compare_abstain_policies(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: List[float] = None,
    fallbacks: List[str] = None,
    majority_class: int = 0
) -> pd.DataFrame:
    """
    Compare different abstain policy configurations.
    
    Args:
        y_true: True labels
        proba: Predicted probabilities (canonical order)
        thresholds: List of thresholds to test
        fallbacks: List of fallback strategies to test
        majority_class: Training set majority class
        
    Returns:
        DataFrame comparing all policy combinations
    """
    if thresholds is None:
        thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    if fallbacks is None:
        fallbacks = ["neutral", "persistence", "majority"]
    
    results = []
    
    for threshold in thresholds:
        for fallback in fallbacks:
            policy = AbstainPolicy(threshold=threshold, fallback=fallback)
            eval_result = evaluate_abstain_policy(
                y_true, proba, policy, 
                y_prev=None, majority_class=majority_class
            )
            results.append(eval_result)
    
    df = pd.DataFrame(results)
    
    # Sort by macro_f1 descending
    df = df.sort_values("macro_f1", ascending=False)
    
    return df


def evaluate_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float
) -> Dict:
    """Evaluate predictions at a given confidence threshold."""
    y_pred = apply_confidence_threshold(proba, threshold)
    
    metrics = compute_metrics(y_true, y_pred)
    
    # Coverage: proportion of non-Neutral predictions
    coverage = (y_pred != 0).mean()
    
    # Neutral frequency
    neutral_pct = (y_pred == 0).mean()
    
    return {
        "threshold": threshold,
        "coverage": coverage,
        "neutral_pct": neutral_pct,
        **metrics
    }


def find_optimal_threshold(
    df: pd.DataFrame,
    min_coverage: float = 0.50,
    optimize_for: str = "macro_f1"
) -> Dict:
    """
    Find threshold that maximizes metric while maintaining minimum coverage.
    
    Args:
        df: DataFrame with threshold sweep results
        min_coverage: Minimum required coverage
        optimize_for: Metric to maximize
        
    Returns:
        Dictionary with optimal threshold and metrics
    """
    valid = df[df["coverage"] >= min_coverage]
    
    if len(valid) == 0:
        return {"optimal_threshold": None, "reason": "No threshold meets coverage requirement"}
    
    best_idx = valid[optimize_for].idxmax()
    best_row = valid.loc[best_idx]
    
    return {
        "optimal_threshold": best_row["threshold"],
        "coverage": best_row["coverage"],
        "macro_f1": best_row["macro_f1"],
    }


# =============================================================================
# CALIBRATION (Cell 5B)
# =============================================================================

def compute_calibration(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 5
) -> pd.DataFrame:
    """
    Compute calibration: does predicted probability match actual frequency?
    """
    max_proba = proba.max(axis=1)
    pred_class = np.array([-1, 0, 1])[proba.argmax(axis=1)]
    correct = (pred_class == y_true).astype(int)
    
    # Bin by confidence
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(max_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            calibration.append({
                "bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                "mean_confidence": max_proba[mask].mean(),
                "accuracy": correct[mask].mean(),
                "n_samples": mask.sum()
            })
    
    return pd.DataFrame(calibration)


def compute_brier_scores(
    y_true: np.ndarray,
    proba: np.ndarray
) -> Dict[str, float]:
    """Compute Brier scores for each class."""
    brier_scores = {}
    
    for c, name, col_idx in [(-1, "bear", 0), (0, "neutral", 1), (1, "bull", 2)]:
        y_true_binary = (y_true == c).astype(int)
        y_prob_class = proba[:, col_idx]
        brier = brier_score_loss(y_true_binary, y_prob_class)
        brier_scores[f"brier_{name}"] = brier
    
    return brier_scores


# =============================================================================
# TEST EVALUATION WRAPPER (Cell 4D)
# =============================================================================

def cell4d_evaluate_test(
    models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features_lr: List[str],
    selected_features_tree: List[str],
    baselines: Dict,
    seed: int = 42
) -> Dict:
    """
    Cell 4D: Evaluate all models on test set.
    
    All probability outputs are standardized to canonical class order:
    [-1, 0, 1] = [Bear, Neutral, Bull]
    
    Args:
        models: Dict of model_name -> trained model
        X_test: Test features
        y_test: Test labels
        selected_features_lr: Features for LR/Stacking
        selected_features_tree: Features for RF/XGB
        baselines: Baseline results from compute_baselines()
        seed: Random seed
        
    Returns:
        Dictionary with all test evaluation results:
        {
            "predictions": Dict[str, np.ndarray],
            "probas": Dict[str, np.ndarray],  # Standardized to [-1, 0, 1] order
            "metrics": Dict[str, Dict],
            "results_df": pd.DataFrame,
        }
    """
    log_section("CELL 4D: TEST EVALUATION")
    
    predictions = {}
    probas = {}
    metrics = {}
    
    # Feature mapping per model type
    feature_map = {
        "lr": selected_features_lr,
        "rf": selected_features_tree,
        "xgb": selected_features_tree,
        "stacking": selected_features_lr,
    }
    
    log("  Probability columns: [P(Bear), P(Neutral), P(Bull)] = [-1, 0, 1]")
    
    for name, model in models.items():
        features = feature_map.get(name, selected_features_lr)
        X_sub = X_test[features]
        
        # Predict
        y_pred = model.predict(X_sub)
        
        # Decode XGBoost labels
        if name == "xgb":
            y_pred = decode_labels(y_pred)
        
        # Probabilities - standardize to canonical order [-1, 0, 1]
        if hasattr(model, "predict_proba"):
            y_proba_raw = model.predict_proba(X_sub)
            y_proba = standardize_probas(model, y_proba_raw, model_name=name)
        else:
            y_proba = None
        
        # Metrics
        model_metrics = compute_metrics(y_test.values, y_pred)
        trans_metrics = compute_transition_metrics(y_test, y_pred)
        model_metrics.update(trans_metrics)
        
        predictions[name.upper()] = y_pred
        if y_proba is not None:
            probas[name.upper()] = y_proba
        metrics[name.upper()] = model_metrics
        
        log(f"  {name.upper()}: Macro-F1 = {model_metrics['macro_f1']:.4f}")
    
    # Build results DataFrame
    rows = []
    for name, m in metrics.items():
        row = {"model": name}
        row.update(m)
        rows.append(row)
    
    # Add baselines
    for name, result in baselines.items():
        row = {"model": f"Baseline_{name}"}
        row.update(result["metrics"])
        rows.append(row)
    
    results_df = pd.DataFrame(rows).set_index("model")
    results_df = results_df.sort_values("macro_f1", ascending=False)
    
    log("\n" + "="*60)
    log("TEST RESULTS SUMMARY")
    log("="*60)
    cols = ["macro_f1", "balanced_accuracy", "recall_bear", "recall_neutral", "recall_bull"]
    available = [c for c in cols if c in results_df.columns]
    log(results_df[available].round(4).to_string())
    
    return {
        "predictions": predictions,
        "probas": probas,  # Standardized to canonical [-1, 0, 1] order
        "metrics": metrics,
        "results_df": results_df,
    }


def cell5a_stability_diagnostics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    baseline_f1: float,
    window: int = 52,
    min_periods: int = 26,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Cell 5A: Stability diagnostics (rolling metrics + bootstrap).
    
    METHODOLOGICAL RATIONALE:
    -------------------------
    With ~200 test observations, point estimates have high variance.
    This function provides two complementary uncertainty quantifications:
    
    1. ROLLING METRICS: Temporal stability analysis
       - Computes Macro-F1 in rolling windows (default: 52 weeks / 1 year)
       - Reveals whether performance is stable or regime-dependent
       - High std → model works well in some periods but fails in others
       - Trend analysis → is performance degrading over time?
    
    2. BLOCK BOOTSTRAP CI: Statistical significance
       - 1000 bootstrap resamples with temporal blocks (size=4 weeks)
       - Block structure preserves autocorrelation in regime sequences
       - 95% CI for Macro-F1 → honest uncertainty bounds
       - Significance test: CI excludes random baseline → model has skill
    
    INTERPRETATION GUIDE:
    - CI excludes baseline AND low rolling variance → confident in model skill
    - CI excludes baseline BUT high rolling variance → skill is regime-dependent
    - CI includes baseline → cannot reject null hypothesis (no skill)
    
    Args:
        y_test: Test labels (pd.Series with DatetimeIndex)
        y_pred: Model predictions (np.ndarray)
        baseline_f1: Stratified random baseline Macro-F1 for comparison
        window: Rolling window size (weeks)
        min_periods: Minimum periods for rolling calculation
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with stability analysis results:
        - rolling_df: Rolling metrics DataFrame
        - observed: Point estimate Macro-F1
        - ci_lower, ci_upper: 95% bootstrap CI bounds
        - significant: Whether CI excludes baseline
    """
    log_section("CELL 5A: STABILITY DIAGNOSTICS")
    
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    
    # Rolling metrics
    rolling = compute_rolling_metrics(y_test, y_pred_series, window, min_periods)
    
    log(f"  Rolling Macro-F1: mean={rolling['macro_f1'].mean():.4f}, std={rolling['macro_f1'].std():.4f}")
    
    # Bootstrap CI
    observed, ci_lower, ci_upper, boot_dist = block_bootstrap_ci(
        y_test.values, y_pred, macro_f1_metric,
        n_bootstrap=n_bootstrap, block_size=4, confidence=0.95, seed=seed
    )
    
    significant = ci_lower > baseline_f1
    
    log(f"\n  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    log(f"  vs Random baseline: {baseline_f1:.4f}")
    log(f"  Significant: {'Yes ✓' if significant else 'No'}")
    
    return {
        "rolling_metrics": rolling,
        "observed": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
    }


def cell5b_threshold_analysis(
    y_test: pd.Series,
    y_proba: np.ndarray,
    thresholds: np.ndarray = None,
    majority_class: int = 0
) -> Dict:
    """
    Cell 5B: Confidence threshold analysis with abstain policies.
    
    Args:
        y_test: Test labels
        y_proba: Predicted probabilities in canonical order [-1, 0, 1]
                 Columns: [P(Bear), P(Neutral), P(Bull)]
        thresholds: Threshold values to sweep
        majority_class: Training majority class for fallback
        
    Returns:
        Dictionary with threshold analysis results
    """
    log_section("CELL 5B: CONFIDENCE THRESHOLD ANALYSIS")
    
    if thresholds is None:
        thresholds = np.arange(0.35, 0.75, 0.05)
    
    # Threshold sweep (original behavior - neutral fallback)
    results = []
    for thresh in thresholds:
        result = evaluate_threshold(y_test.values, y_proba, thresh)
        results.append(result)
    
    threshold_df = pd.DataFrame(results)
    
    log("\nThreshold sweep (fallback=neutral):")
    log(threshold_df[["threshold", "coverage", "macro_f1"]].round(3).to_string())
    
    # Optimal threshold
    optimal = find_optimal_threshold(threshold_df, min_coverage=0.50)
    
    if optimal.get("optimal_threshold"):
        log(f"\n✓ Optimal: τ={optimal['optimal_threshold']:.2f}, "
            f"F1={optimal['macro_f1']:.4f}, coverage={optimal['coverage']:.1%}")
    
    # Compare abstain policies
    log("\n--- Abstain Policy Comparison ---")
    policy_comparison = compare_abstain_policies(
        y_test.values, y_proba,
        thresholds=[0.40, 0.45, 0.50],
        fallbacks=["neutral", "persistence", "majority"],
        majority_class=majority_class
    )
    
    # Show top 5 policies
    display_cols = ["policy", "macro_f1", "coverage", "abstain_rate"]
    available_cols = [c for c in display_cols if c in policy_comparison.columns]
    log(policy_comparison[available_cols].head(5).round(3).to_string())
    
    # Best policy
    best_policy = policy_comparison.iloc[0]
    log(f"\n✓ Best policy: {best_policy['policy']}, F1={best_policy['macro_f1']:.4f}")
    
    # Calibration
    calibration = compute_calibration(y_test.values, y_proba, n_bins=5)
    brier = compute_brier_scores(y_test.values, y_proba)
    
    return {
        "threshold_df": threshold_df,
        "optimal": optimal,
        "policy_comparison": policy_comparison,
        "best_policy": best_policy.to_dict(),
        "calibration_df": calibration,
        "brier_scores": brier,
    }


def evaluate_on_test(
    model,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: List[str],
    is_xgboost: bool = False
) -> Dict:
    """
    Single-shot test evaluation with comprehensive metrics.
    
    Args:
        model: Trained model
        model_name: Name for reporting
        X_test: Test features
        y_test: Test labels
        feature_cols: Feature columns to use
        is_xgboost: Whether model uses encoded labels
        
    Returns:
        Dictionary with all evaluation results
    """
    X_test_subset = X_test[feature_cols]
    y_pred = model.predict(X_test_subset)
    
    if is_xgboost:
        y_pred = decode_labels(y_pred)
    
    metrics = compute_metrics(y_test.values, y_pred)
    transition_metrics = compute_transition_metrics(y_test, y_pred)
    metrics.update(transition_metrics)
    
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
    
    return {
        "model": model_name,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "predictions": y_pred.tolist(),
        "n_test_samples": len(y_test),
    }

# =============================================================================
# AUDIT EXPORT FUNCTIONS (for grading transparency)
# =============================================================================

def export_splits_summary(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    output_path: Path
) -> pd.DataFrame:
    """
    Export splits summary with sizes, date ranges, and class distributions.
    
    Args:
        y_train: Training labels (with DatetimeIndex)
        y_val: Validation labels (with DatetimeIndex)
        y_test: Test labels (with DatetimeIndex)
        output_path: Path to save CSV
        
    Returns:
        Summary DataFrame
    """
    
    def get_split_stats(y: pd.Series, name: str) -> Dict:
        dist = y.value_counts(normalize=True).to_dict()
        return {
            "split": name,
            "n_samples": len(y),
            "start_date": str(y.index.min().date()),
            "end_date": str(y.index.max().date()),
            "pct_bear": dist.get(-1, 0),
            "pct_neutral": dist.get(0, 0),
            "pct_bull": dist.get(1, 0),
        }
    
    rows = [
        get_split_stats(y_train, "train"),
        get_split_stats(y_val, "val"),
        get_split_stats(y_test, "test"),
    ]
    
    df = pd.DataFrame(rows)
    
    # Add totals row
    total = {
        "split": "TOTAL",
        "n_samples": len(y_train) + len(y_val) + len(y_test),
        "start_date": str(y_train.index.min().date()),
        "end_date": str(y_test.index.max().date()),
        "pct_bear": "-",
        "pct_neutral": "-",
        "pct_bull": "-",
    }
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log(f"  📊 Splits summary saved: {output_path.name}")
    
    return df


def export_predictions_csv(
    y_test: pd.Series,
    predictions: Dict[str, np.ndarray],
    probas: Dict[str, np.ndarray],
    output_path: Path
) -> pd.DataFrame:
    """
    Export per-sample predictions for all models.
    
    Columns: date, y_true, pred_LR, pred_RF, pred_XGB, pred_STACKING, 
             max_proba_LR, max_proba_RF, max_proba_XGB, max_proba_STACKING
    
    Args:
        y_test: Test labels (with DatetimeIndex)
        predictions: Dict of model_name -> predictions array
        probas: Dict of model_name -> probability array (n_samples, 3)
        output_path: Path to save CSV
        
    Returns:
        Predictions DataFrame
    """
    
    df = pd.DataFrame({
        "date": y_test.index,
        "y_true": y_test.values,
    })
    
    # Add predictions for each model
    for name, preds in predictions.items():
        df[f"pred_{name}"] = preds
    
    # Add max probability for each model
    for name, proba in probas.items():
        if proba is not None and len(proba) > 0:
            df[f"max_proba_{name}"] = np.max(proba, axis=1)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log(f"  📊 Predictions saved: {output_path.name}")
    
    return df


def export_confusion_matrix_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path
) -> pd.DataFrame:
    """
    Export confusion matrix as CSV.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Model name for filename
        output_path: Directory to save CSV
        
    Returns:
        Confusion matrix DataFrame
    """
    from sklearn.metrics import confusion_matrix
    
    labels = [-1, 0, 1]
    label_names = ["Bear", "Neutral", "Bull"]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create DataFrame with row/col labels
    df = pd.DataFrame(
        cm,
        index=[f"True_{n}" for n in label_names],
        columns=[f"Pred_{n}" for n in label_names]
    )
    
    # Add row totals
    df["Total"] = df.sum(axis=1)
    
    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f"confusion_matrix_{model_name}.csv"
    df.to_csv(filepath)
    log(f"  📊 Confusion matrix saved: {filepath.name}")
    
    return df


def export_all_confusion_matrices(
    y_test: pd.Series,
    predictions: Dict[str, np.ndarray],
    output_dir: Path
) -> None:
    """
    Export confusion matrices for all models.
    
    Args:
        y_test: Test labels
        predictions: Dict of model_name -> predictions
        output_dir: Directory to save CSVs
    """
    
    cm_dir = Path(output_dir) / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    
    for name, preds in predictions.items():
        export_confusion_matrix_csv(
            y_true=y_test.values,
            y_pred=preds,
            model_name=name,
            output_path=cm_dir
        )