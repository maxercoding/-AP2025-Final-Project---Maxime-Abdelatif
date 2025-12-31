"""
Train/Validation/Test splits with embargo enforcement.
Maps to: Split logic from Cell 4B

Handles:
- Temporal splits with purge/embargo
- Feature/target extraction with leakage validation
- Walk-forward CV setup
- Feature pruning (CV-pure)

Outputs: splits dict, X_train, y_train, X_val, y_val, X_test, y_test, selected_features
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from src.utils import log, log_section, log_debug
from src.labels import LABEL_COLS, BENCHMARK_COLS


# =============================================================================
# FEATURE/TARGET EXTRACTION (Cell 4A)
# =============================================================================

def extract_features_target(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: Set[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract X and y with strict validation.
    Risk B mitigation: fatal asserts on label leakage.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        exclude_cols: Set of columns to exclude from features
        
    Returns:
        Tuple of (X, y)
        
    Raises:
        ValueError: If leakage detected or validation fails
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    
    # Extract feature columns (everything not excluded)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # FATAL CHECKS: No label-side columns in features
    leaked = set(feature_cols) & set(LABEL_COLS)
    if leaked:
        raise ValueError(f"FATAL LEAKAGE: Label columns in features: {leaked}")
    
    bench_leaked = set(feature_cols) & set(BENCHMARK_COLS)
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


def validate_embargo_gaps(
    splits: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    embargo_weeks: int,
    save_csv: str = None
) -> pd.DataFrame:
    """
    Validate embargo gaps and export summary for audit.
    
    Hard asserts that no data overlap exists between splits.
    Exports CSV confirming embargo compliance.
    
    
    Args:
        splits: Dict with 'train', 'val', 'test' splits
        embargo_weeks: Required embargo gap in weeks
        save_csv: Optional path to save validation CSV
        
    Returns:
        DataFrame with embargo validation summary
        
    Raises:
        AssertionError: If any embargo violation detected
    """
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    
    required_gap_days = embargo_weeks * 7
    
    # Extract date boundaries
    train_start = X_train.index.min()
    train_end = X_train.index.max()
    val_start = X_val.index.min()
    val_end = X_val.index.max()
    test_start = X_test.index.min()
    test_end = X_test.index.max()
    
    # Calculate gaps
    gap_train_val = (val_start - train_end).days
    gap_val_test = (test_start - val_end).days
    
    # HARD ASSERTS - no overlap allowed
    assert train_end < val_start, \
        f"FATAL: Train/Val overlap! Train ends {train_end}, Val starts {val_start}"
    assert val_end < test_start, \
        f"FATAL: Val/Test overlap! Val ends {val_end}, Test starts {test_start}"
    
    # Check index intersection (should be empty)
    train_val_overlap = set(X_train.index) & set(X_val.index)
    val_test_overlap = set(X_val.index) & set(X_test.index)
    train_test_overlap = set(X_train.index) & set(X_test.index)
    
    assert len(train_val_overlap) == 0, \
        f"FATAL: {len(train_val_overlap)} overlapping dates between train/val"
    assert len(val_test_overlap) == 0, \
        f"FATAL: {len(val_test_overlap)} overlapping dates between val/test"
    assert len(train_test_overlap) == 0, \
        f"FATAL: {len(train_test_overlap)} overlapping dates between train/test"
    
    # Check embargo is sufficient
    assert gap_train_val >= required_gap_days, \
        f"EMBARGO VIOLATION: Train→Val gap {gap_train_val}d < required {required_gap_days}d"
    assert gap_val_test >= required_gap_days, \
        f"EMBARGO VIOLATION: Val→Test gap {gap_val_test}d < required {required_gap_days}d"
    
    # Build summary DataFrame
    summary = pd.DataFrame([
        {
            "split": "train",
            "start_date": train_start.strftime("%Y-%m-%d"),
            "end_date": train_end.strftime("%Y-%m-%d"),
            "n_samples": len(X_train),
            "gap_to_next_days": gap_train_val,
            "gap_to_next_weeks": gap_train_val // 7,
            "required_gap_weeks": embargo_weeks,
            "embargo_ok": gap_train_val >= required_gap_days,
        },
        {
            "split": "val",
            "start_date": val_start.strftime("%Y-%m-%d"),
            "end_date": val_end.strftime("%Y-%m-%d"),
            "n_samples": len(X_val),
            "gap_to_next_days": gap_val_test,
            "gap_to_next_weeks": gap_val_test // 7,
            "required_gap_weeks": embargo_weeks,
            "embargo_ok": gap_val_test >= required_gap_days,
        },
        {
            "split": "test",
            "start_date": test_start.strftime("%Y-%m-%d"),
            "end_date": test_end.strftime("%Y-%m-%d"),
            "n_samples": len(X_test),
            "gap_to_next_days": None,
            "gap_to_next_weeks": None,
            "required_gap_weeks": embargo_weeks,
            "embargo_ok": True,  # No next split
        },
    ])
    
    # Save CSV if path provided
    if save_csv is not None:
        from pathlib import Path
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(save_csv, index=False)
        log(f"  📋 Embargo validation saved: {save_csv}")
    
    log("  ✓ All embargo checks passed (no data leakage)")
    
    return summary


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
    """Compute inverse-frequency class weights for LR."""
    counts = y.value_counts()
    n_samples = len(y)
    n_classes = len(counts)
    return {c: n_samples / (n_classes * counts[c]) for c in counts.index}


def compute_class_weights_damped(y: pd.Series, damping: float = 0.5) -> Dict[int, float]:
    """
    Compute damped class weights for tree models.
    
    Tree models (RF, XGBoost) are very sensitive to class weights and tend to
    over-predict the minority class when weights are too strong. This function
    applies a damping factor to reduce the weight magnitude.
    
    Formula: damped_weight = 1 + (original_weight - 1) * damping
    
    Args:
        y: Target series
        damping: Damping factor in [0, 1]
            - 0.0: No class weighting (all weights = 1)
            - 0.5: Halfway between uniform and full weights (default)
            - 1.0: Full inverse-frequency weights
            
    Returns:
        Dictionary of damped class weights
        
    Example:
        Original weights: {-1: 1.27, 0: 1.07, 1: 0.78}
        Damped (0.5):     {-1: 1.13, 0: 1.04, 1: 0.89}
    """
    full_weights = compute_class_weights(y)
    return {c: 1 + (w - 1) * damping for c, w in full_weights.items()}


# =============================================================================
# WALK-FORWARD CV (Cell 4B)
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
# MAIN CELL FUNCTION
# =============================================================================

def cell4b_create_splits(
    regime_df: pd.DataFrame,
    exclude_cols: Set[str],
    train_end: str,
    val_end: str,
    embargo_weeks: int,
    seed: int = 42
) -> Dict:
    """
    Cell 4B: Create train/val/test splits with preprocessing.
    
    Args:
        regime_df: DataFrame from Cell 3 with features and targets
        exclude_cols: Columns to exclude from features
        train_end: Training end date (ISO format)
        val_end: Validation end date (ISO format)
        embargo_weeks: Embargo gap in weeks
        seed: Random seed
        
    Returns:
        Dictionary with all split artifacts:
        {
            "X_full": pd.DataFrame,
            "y_full": pd.Series,
            "splits": {"train": (X, y), "val": (X, y), "test": (X, y)},
            "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
            "selected_features_lr": List[str],
            "selected_features_tree": List[str],
            "prune_diag_lr": Dict,
            "prune_diag_tree": Dict,
            "class_weights": Dict,
            "tscv": TimeSeriesSplit,
            "adaptive_test_size": int,
        }
    """
    log_section("CELL 4B: DATA SPLITS & PREPROCESSING")
    
    # Extract full dataset
    X_full, y_full = extract_features_target(
        regime_df, 
        target_col="regime_target",
        exclude_cols=exclude_cols
    )
    
    log(f"\n✓ Features extracted: {X_full.shape[1]} columns, {len(X_full)} rows")
    log(f"  Date range: {X_full.index.min().date()} → {X_full.index.max().date()}")
    
    # Leakage audit
    assert len(set(X_full.columns) & exclude_cols) == 0, "FATAL: Leaked columns in features"
    
    # Create temporal splits
    splits = create_temporal_splits(
        X_full, y_full,
        train_end=train_end,
        val_end=val_end,
        embargo_weeks=embargo_weeks
    )
    
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    
    log(f"\n✓ Temporal splits with {embargo_weeks}-week embargo:")
    log(f"  Train: {len(X_train)} rows | {X_train.index.min().date()} → {X_train.index.max().date()}")
    log(f"  Val:   {len(X_val)} rows | {X_val.index.min().date()} → {X_val.index.max().date()}")
    log(f"  Test:  {len(X_test)} rows | {X_test.index.min().date()} → {X_test.index.max().date()}")
    
    # Validate split integrity
    assert X_train.index.max() < X_val.index.min(), "Train/Val overlap detected"
    assert X_val.index.max() < X_test.index.min(), "Val/Test overlap detected"
    
    # Comprehensive embargo validation (with hard asserts)
    embargo_validation = validate_embargo_gaps(splits, embargo_weeks)
    
    # Class distribution per split
    log("\n  Class distributions:")
    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = y_split.value_counts(normalize=True).reindex([-1, 0, 1]).fillna(0)
        log(f"    {name}: Bear={dist[-1]:.1%} | Neutral={dist[0]:.1%} | Bull={dist[1]:.1%}")
    
    # Feature pruning (CV-Pure: use only first 70% of train)
    log("\n--- Feature Pruning (CV-Pure: first 70% of train) ---")
    
    prune_cutoff = int(len(X_train) * 0.70)
    X_train_prune = X_train.iloc[:prune_cutoff]
    y_train_prune = y_train.iloc[:prune_cutoff]
    
    log(f"  Pruning calibrated on {len(X_train_prune)} samples (70% of train)")
    
    log("\n  LR features (moderate pruning ρ > 0.80):")
    selected_features_lr, prune_diag_lr = prune_features_train_only(
        X_train_prune, y_train_prune,
        variance_threshold=1e-6,
        correlation_threshold=0.80,  # Increased from 0.70 to keep more features
        seed=seed,
        verbose=True
    )
    
    log("\n  Tree features (moderate pruning ρ > 0.85):")
    selected_features_tree, prune_diag_tree = prune_features_train_only(
        X_train_prune, y_train_prune,
        variance_threshold=1e-6,
        correlation_threshold=0.85,
        seed=seed,
        verbose=True
    )
    
    log(f"\n✓ Feature sets ready:")
    log(f"    LR: {len(selected_features_lr)} features")
    log(f"    Tree: {len(selected_features_tree)} features")
    
    # Class weights (full for LR, damped for trees)
    class_weights = compute_class_weights(y_train)
    class_weights_tree = compute_class_weights_damped(y_train, damping=0.5)
    
    log(f"\n✓ Class weights (train-only):")
    log(f"    LR weights (full):")
    log(f"      Bear (-1): {class_weights[-1]:.3f}")
    log(f"      Neutral (0): {class_weights[0]:.3f}")
    log(f"      Bull (1): {class_weights[1]:.3f}")
    log(f"    Tree weights (damped 0.5):")
    log(f"      Bear (-1): {class_weights_tree[-1]:.3f}")
    log(f"      Neutral (0): {class_weights_tree[0]:.3f}")
    log(f"      Bull (1): {class_weights_tree[1]:.3f}")
    
    # Walk-forward CV setup
    adaptive_test_size = max(26, min(52, len(X_train) // 10))
    
    tscv = create_purged_tscv(
        n_samples=len(X_train),
        n_splits=5,
        gap=embargo_weeks,
        test_size=adaptive_test_size
    )
    
    log(f"\n✓ Walk-forward CV configured:")
    log(f"    Splits: {tscv.n_splits}")
    log(f"    Gap: {embargo_weeks} weeks")
    log(f"    Test size per fold: {adaptive_test_size} weeks")
    
    return {
        "X_full": X_full,
        "y_full": y_full,
        "splits": splits,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "selected_features_lr": selected_features_lr,
        "selected_features_tree": selected_features_tree,
        "prune_diag_lr": prune_diag_lr,
        "prune_diag_tree": prune_diag_tree,
        "class_weights": class_weights,           # Full weights for LR
        "class_weights_tree": class_weights_tree, # Damped weights for RF/XGB
        "tscv": tscv,
        "adaptive_test_size": adaptive_test_size,
        "embargo_validation": embargo_validation,  # For audit export
    }


# =============================================================================
# MERGE TRAIN + VAL (for final training in Cell 4C)
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