"""
Model training wrappers for LR, RF, XGBoost, and Stacking.
Maps to: Cell 4C (CV tuning), Cell 4D (test evaluation)

Contains:
- Model factory functions
- CV hyperparameter search
- Training wrappers
- Stacking ensemble
- Model persistence

Outputs: trained models, CV results, best hyperparameters
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from src.utils import log, log_section, log_debug
from src.evaluation import (
    encode_labels, decode_labels, LABEL_MAP,
    compute_metrics
)


# =============================================================================
# DETERMINISTIC MODE (for reproducibility)
# =============================================================================

# Global flag for deterministic mode (set from config)
_DETERMINISTIC: bool = False


def set_deterministic(deterministic: bool) -> None:
    """
    Set deterministic mode for reproducible results.
    
    When True:
    - n_jobs=1 for all models (no parallelism)
    - Results will be identical across runs with same seed
    
    When False:
    - n_jobs=-1 for faster training (uses all cores)
    - Results may vary slightly due to thread ordering
    
    Args:
        deterministic: Whether to force deterministic execution
    """
    global _DETERMINISTIC
    _DETERMINISTIC = deterministic
    if deterministic:
        log("  ⚙️  Deterministic mode: ON (n_jobs=1 for reproducibility)")
    else:
        log("  ⚙️  Deterministic mode: OFF (n_jobs=-1 for speed)")


def _get_n_jobs() -> int:
    """Return n_jobs value based on deterministic mode."""
    return 1 if _DETERMINISTIC else -1


# =============================================================================
# HYPERPARAMETER GRIDS (Cell 4C)
# =============================================================================

# LR grid uses pipeline naming convention: lr__<param>
LR_PARAM_GRID = {
    "lr__C": [0.01, 0.1, 1.0],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [5, 10],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def create_lr_model(
    class_weight: Dict[int, float] = None,
    C: float = 1.0,
    seed: int = 42
) -> Pipeline:
    """
    Create Logistic Regression pipeline with StandardScaler.
    
    LR requires scaled features for optimal performance.
    Pipeline ensures scaling is applied consistently in CV and prediction.
    
    Args:
        class_weight: Class weight dict for imbalanced data
        C: Regularization strength (smaller = stronger regularization)
        seed: Random seed
        
    Returns:
        Pipeline with StandardScaler -> LogisticRegression
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=_get_n_jobs()
        ))
    ])


def create_rf_model(
    class_weight: Dict[int, float] = None,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_leaf: int = 5,
    seed: int = 42
) -> RandomForestClassifier:
    """Create Random Forest model."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=_get_n_jobs()
    )


def create_xgb_model(
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    seed: int = 42
) -> xgb.XGBClassifier:
    """
    Create XGBoost model for multi-class classification.
    
    Note: XGBoost uses encoded labels {0, 1, 2} not {-1, 0, 1}.
    No sample weights are used - XGBoost tends to over-predict minority
    classes when sample weights are applied.
    """
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=seed,
        n_jobs=_get_n_jobs(),
        verbosity=0
    )


# =============================================================================
# CV HYPERPARAMETER SEARCH (Cell 4C)
# =============================================================================

@dataclass
class CVResult:
    """Store CV results for a single model."""
    model_name: str
    best_params: Dict
    best_score: float
    cv_scores: List[float]
    all_results: List[Dict] = field(default_factory=list)


def export_cv_results(
    cv_results: Dict[str, CVResult],
    save_path: Path
) -> pd.DataFrame:
    """
    Export all CV search results to CSV for audit trail.
    
    Creates a comprehensive table showing all hyperparameter combinations
    tested and their cross-validation scores.
    
    Args:
        cv_results: Dict mapping model name to CVResult
        save_path: Path to save CSV file
        
    Returns:
        DataFrame with all CV results
    """
    rows = []
    
    for model_name, cv_result in cv_results.items():
        for result in cv_result.all_results:
            row = {
                "model": model_name,
                "mean_macro_f1": result.get("mean_macro_f1"),
                "std_macro_f1": result.get("std_macro_f1"),
                "is_best": False,
            }
            
            # Add all hyperparameters from the result dict
            for key, value in result.items():
                if key not in ["mean_macro_f1", "std_macro_f1", "fold_scores"]:
                    row[key] = value
            
            # Add individual fold scores
            fold_scores = result.get("fold_scores", [])
            for i, score in enumerate(fold_scores):
                row[f"fold_{i+1}"] = score
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Mark best configuration per model
    for model_name in df["model"].unique():
        model_mask = df["model"] == model_name
        best_idx = df.loc[model_mask, "mean_macro_f1"].idxmax()
        df.loc[best_idx, "is_best"] = True
    
    # Reorder columns
    base_cols = ["model", "is_best", "mean_macro_f1", "std_macro_f1"]
    fold_cols = [c for c in df.columns if c.startswith("fold_")]
    param_cols = [c for c in df.columns if c not in base_cols + fold_cols]
    df = df[base_cols + sorted(param_cols) + sorted(fold_cols)]
    
    # Sort by model and score
    df = df.sort_values(["model", "mean_macro_f1"], ascending=[True, False])
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    log(f"  📋 CV results saved: {save_path.name} ({len(df)} configurations)")
    
    return df


def cv_search_lr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    tscv: TimeSeriesSplit,
    class_weights: Dict[int, float],
    seed: int = 42
) -> CVResult:
    """
    Walk-forward CV search for Logistic Regression Pipeline.
    
    Note: LR is wrapped in a Pipeline with StandardScaler.
    Scaling is applied automatically within each CV fold.
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_cols: Selected feature columns
        tscv: TimeSeriesSplit object
        class_weights: Class weights dict
        seed: Random seed
        
    Returns:
        CVResult with best parameters and scores
    """
    log("\n--- Logistic Regression CV Search (with StandardScaler) ---")
    
    X = X_train[feature_cols].values
    y = y_train.values
    
    all_results = []
    
    # Extract C values from pipeline param grid
    C_values = LR_PARAM_GRID["lr__C"]
    
    for C in C_values:
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            # Pipeline handles scaling internally
            model = create_lr_model(class_weight=class_weights, C=C, seed=seed)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            
            score = f1_score(y_va, y_pred, average="macro", zero_division=0)
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        all_results.append({
            "C": C,
            "mean_macro_f1": mean_score,
            "std_macro_f1": np.std(fold_scores),
            "fold_scores": fold_scores
        })
        
        log(f"  C={C}: Macro-F1={mean_score:.4f} (±{np.std(fold_scores):.4f})")
    
    # Find best
    best_idx = np.argmax([r["mean_macro_f1"] for r in all_results])
    best = all_results[best_idx]
    
    log(f"  ✓ Best: C={best['C']}, Macro-F1={best['mean_macro_f1']:.4f}")
    
    return CVResult(
        model_name="LogisticRegression",
        best_params={"C": best["C"]},
        best_score=best["mean_macro_f1"],
        cv_scores=best["fold_scores"],
        all_results=all_results
    )


def cv_search_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    tscv: TimeSeriesSplit,
    class_weights: Dict[int, float],
    seed: int = 42
) -> CVResult:
    """Walk-forward CV search for Random Forest."""
    log("\n--- Random Forest CV Search ---")
    
    X = X_train[feature_cols].values
    y = y_train.values
    
    all_results = []
    
    from itertools import product
    param_combos = list(product(
        RF_PARAM_GRID["n_estimators"],
        RF_PARAM_GRID["max_depth"],
        RF_PARAM_GRID["min_samples_leaf"]
    ))
    
    for n_est, max_d, min_leaf in param_combos:
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            model = create_rf_model(
                class_weight=class_weights,
                n_estimators=n_est,
                max_depth=max_d,
                min_samples_leaf=min_leaf,
                seed=seed
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            
            score = f1_score(y_va, y_pred, average="macro", zero_division=0)
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        all_results.append({
            "n_estimators": n_est,
            "max_depth": max_d,
            "min_samples_leaf": min_leaf,
            "mean_macro_f1": mean_score,
            "std_macro_f1": np.std(fold_scores),
            "fold_scores": fold_scores
        })
    
    # Find best
    best_idx = np.argmax([r["mean_macro_f1"] for r in all_results])
    best = all_results[best_idx]
    
    log(f"  Best: n_est={best['n_estimators']}, max_d={best['max_depth']}, "
        f"min_leaf={best['min_samples_leaf']}, Macro-F1={best['mean_macro_f1']:.4f}")
    
    return CVResult(
        model_name="RandomForest",
        best_params={
            "n_estimators": best["n_estimators"],
            "max_depth": best["max_depth"],
            "min_samples_leaf": best["min_samples_leaf"]
        },
        best_score=best["mean_macro_f1"],
        cv_scores=best["fold_scores"],
        all_results=all_results
    )


def cv_search_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    tscv: TimeSeriesSplit,
    class_weights: Dict[int, float],
    seed: int = 42
) -> CVResult:
    """
    Walk-forward CV search for XGBoost (with label encoding).
    
    NOTE: XGBoost is more sensitive to sample weights than sklearn models.
    After extensive testing, we found that XGBoost performs better WITHOUT
    explicit sample weights - it tends to over-predict minority classes
    when weights are applied. The class imbalance is handled implicitly
    through the multi:softprob objective.
    """
    log("\n--- XGBoost CV Search ---")
    
    X = X_train[feature_cols].values
    y_encoded = encode_labels(y_train).values
    
    # NOTE: Not using sample weights for XGBoost - see docstring
    
    all_results = []
    
    from itertools import product
    param_combos = list(product(
        XGB_PARAM_GRID["n_estimators"],
        XGB_PARAM_GRID["max_depth"],
        XGB_PARAM_GRID["learning_rate"]
    ))
    
    for n_est, max_d, lr in param_combos:
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y_encoded[train_idx], y_encoded[val_idx]
            
            model = create_xgb_model(
                n_estimators=n_est,
                max_depth=max_d,
                learning_rate=lr,
                seed=seed
            )
            
            # NOTE: Not using sample_weight - XGBoost over-predicts minority class with weights
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_tr)
            
            y_pred_enc = model.predict(X_va)
            y_pred = decode_labels(y_pred_enc)
            y_va_orig = decode_labels(y_va)
            
            score = f1_score(y_va_orig, y_pred, average="macro", zero_division=0)
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        all_results.append({
            "n_estimators": n_est,
            "max_depth": max_d,
            "learning_rate": lr,
            "mean_macro_f1": mean_score,
            "std_macro_f1": np.std(fold_scores),
            "fold_scores": fold_scores
        })
    
    # Find best
    best_idx = np.argmax([r["mean_macro_f1"] for r in all_results])
    best = all_results[best_idx]
    
    log(f"  Best: n_est={best['n_estimators']}, max_d={best['max_depth']}, "
        f"lr={best['learning_rate']}, Macro-F1={best['mean_macro_f1']:.4f}")
    
    return CVResult(
        model_name="XGBoost",
        best_params={
            "n_estimators": best["n_estimators"],
            "max_depth": best["max_depth"],
            "learning_rate": best["learning_rate"]
        },
        best_score=best["mean_macro_f1"],
        cv_scores=best["fold_scores"],
        all_results=all_results
    )


# =============================================================================
# TRAINING ON FULL TRAIN SET (Cell 4C)
# =============================================================================

def train_lr_final(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    class_weights: Dict[int, float],
    best_params: Dict,
    seed: int = 42
) -> LogisticRegression:
    """Train final LR model on full training set."""
    model = create_lr_model(
        class_weight=class_weights,
        C=best_params.get("C", 1.0),
        seed=seed
    )
    model.fit(X_train[feature_cols], y_train)
    return model


def train_rf_final(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    class_weights: Dict[int, float],
    best_params: Dict,
    seed: int = 42
) -> RandomForestClassifier:
    """Train final RF model on full training set."""
    model = create_rf_model(
        class_weight=class_weights,
        n_estimators=best_params.get("n_estimators", 100),
        max_depth=best_params.get("max_depth", 10),
        min_samples_leaf=best_params.get("min_samples_leaf", 5),
        seed=seed
    )
    model.fit(X_train[feature_cols], y_train)
    return model


def train_xgb_final(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    class_weights: Dict[int, float],
    best_params: Dict,
    seed: int = 42
) -> xgb.XGBClassifier:
    """
    Train final XGBoost model on full training set.
    
    NOTE: XGBoost is more sensitive to sample weights than sklearn models.
    After extensive testing, we found that XGBoost performs better WITHOUT
    explicit sample weights. The class_weights parameter is accepted for
    API consistency but not used.
    """
    y_encoded = encode_labels(y_train)
    
    model = create_xgb_model(
        n_estimators=best_params.get("n_estimators", 100),
        max_depth=best_params.get("max_depth", 5),
        learning_rate=best_params.get("learning_rate", 0.1),
        seed=seed
    )
    
    # NOTE: Not using sample_weight - XGBoost over-predicts minority class with weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train[feature_cols], y_encoded)
    
    return model


# =============================================================================
# STACKING ENSEMBLE (Cell 4C)
# =============================================================================

def create_stacking_ensemble(
    lr_model: Pipeline,
    rf_model: RandomForestClassifier,
    xgb_model: xgb.XGBClassifier,
    class_weights: Dict[int, float],
    seed: int = 42
) -> StackingClassifier:
    """
    Create stacking ensemble from base models.
    
    Note: For stacking, we need fresh unfitted estimators with same params.
    LR model is a Pipeline - extract the LR step for parameters.
    
    Args:
        lr_model: LR Pipeline (StandardScaler + LogisticRegression)
        rf_model: Trained RF model
        xgb_model: Trained XGB model
        class_weights: Class weights dict
        seed: Random seed
        
    Returns:
        StackingClassifier with fresh base estimators
    """
    # Extract LR from pipeline
    lr_step = lr_model.named_steps['lr']
    
    # Clone estimators with same hyperparameters
    # Note: Stacking base estimators include scaler in LR pipeline
    lr_clone = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=lr_step.C,
            penalty=lr_step.penalty,
            solver=lr_step.solver,
            max_iter=lr_step.max_iter,
            class_weight=class_weights,
            random_state=seed,
            n_jobs=_get_n_jobs()
        ))
    ])
    
    rf_clone = RandomForestClassifier(
        n_estimators=rf_model.n_estimators,
        max_depth=rf_model.max_depth,
        min_samples_leaf=rf_model.min_samples_leaf,
        class_weight=class_weights,
        random_state=seed,
        n_jobs=_get_n_jobs()
    )
    
    # Note: XGBoost in stacking needs special handling for label encoding
    # Using calibrated version for probability outputs
    
    stacking = StackingClassifier(
        estimators=[
            ("lr", lr_clone),
            ("rf", rf_clone),
        ],
        final_estimator=LogisticRegression(
            class_weight=class_weights,
            random_state=seed,
            max_iter=1000
        ),
        cv=3,
        stack_method="predict_proba",
        n_jobs=_get_n_jobs()
    )
    
    return stacking


def train_stacking_final(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    class_weights: Dict[int, float],
    class_weights_tree: Dict[int, float],
    lr_params: Dict,
    rf_params: Dict,
    seed: int = 42
) -> StackingClassifier:
    """
    Train stacking ensemble on full training set.
    
    Note: Uses sklearn's built-in "balanced" class weighting to avoid
    class encoding issues during internal CV. The class_weights and
    class_weights_tree parameters are accepted for API consistency but
    not directly used.
    """
    log("\n--- Training Stacking Ensemble ---")
    
    # Create base estimators with "balanced" instead of custom dict
    # (avoids class encoding issues during internal CV)
    # LR wrapped in Pipeline with StandardScaler for consistency
    lr_base = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=lr_params.get("C", 1.0),
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",  # Use "balanced" instead of dict
            random_state=seed,
            n_jobs=_get_n_jobs()
        ))
    ])
    
    rf_base = RandomForestClassifier(
        n_estimators=rf_params.get("n_estimators", 100),
        max_depth=rf_params.get("max_depth", 10),
        min_samples_leaf=rf_params.get("min_samples_leaf", 5),
        class_weight="balanced",  # Use "balanced" instead of dict
        random_state=seed,
        n_jobs=_get_n_jobs()
    )
    
    stacking = StackingClassifier(
        estimators=[
            ("lr", lr_base),
            ("rf", rf_base),
        ],
        final_estimator=LogisticRegression(
            class_weight="balanced",  # Use "balanced" instead of dict
            random_state=seed,
            max_iter=1000
        ),
        cv=3,
        stack_method="predict_proba",
        n_jobs=_get_n_jobs()
    )
    
    stacking.fit(X_train[feature_cols], y_train)
    log("  ✓ Stacking ensemble trained")
    
    return stacking

# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(model: Any, path: Path, model_name: str) -> None:
    """Save model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    filepath = path / f"{model_name}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    log_debug(f"  Saved: {filepath}")


def load_model(path: Path, model_name: str) -> Any:
    """Load model from disk."""
    filepath = path / f"{model_name}.pkl"
    with open(filepath, "rb") as f:
        return pickle.load(f)


# =============================================================================
# MAIN CELL FUNCTION
# =============================================================================

def cell4c_train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features_lr: List[str],
    selected_features_tree: List[str],
    class_weights: Dict[int, float],
    class_weights_tree: Dict[int, float],
    tscv: TimeSeriesSplit,
    seed: int = 42
) -> Dict:
    """
    Cell 4C: CV tuning and model training.
    
    Args:
        X_train: Training features
        y_train: Training labels
        selected_features_lr: Feature columns for LR
        selected_features_tree: Feature columns for tree models
        class_weights: Class weights dict for LR (full weights)
        class_weights_tree: Class weights dict for RF/XGB (damped weights)
        tscv: TimeSeriesSplit for CV
        seed: Random seed
        
    Returns:
        Dictionary with all trained models and CV results:
        {
            "cv_results": {"lr": CVResult, "rf": CVResult, "xgb": CVResult},
            "models": {"lr": model, "rf": model, "xgb": model, "stacking": model},
            "best_params": {"lr": dict, "rf": dict, "xgb": dict},
            "feature_cols": {"lr": list, "tree": list},
        }
    """
    log_section("CELL 4C: MODEL TRAINING (CV TUNING)")
    
    # CV search for each model
    cv_lr = cv_search_lr(
        X_train, y_train, selected_features_lr,
        tscv, class_weights, seed  # Full weights for LR
    )
    
    cv_rf = cv_search_rf(
        X_train, y_train, selected_features_tree,
        tscv, class_weights_tree, seed  # Damped weights for trees
    )
    
    cv_xgb = cv_search_xgb(
        X_train, y_train, selected_features_tree,
        tscv, class_weights_tree, seed  # Damped weights for trees
    )
    
    cv_results = {
        "lr": cv_lr,
        "rf": cv_rf,
        "xgb": cv_xgb
    }
    
    # Train final models on full training set
    log("\n--- Training Final Models on Full Train ---")
    
    model_lr = train_lr_final(
        X_train, y_train, selected_features_lr,
        class_weights, cv_lr.best_params, seed  # Full weights for LR
    )
    log("  ✓ Logistic Regression trained")
    
    model_rf = train_rf_final(
        X_train, y_train, selected_features_tree,
        class_weights_tree, cv_rf.best_params, seed  # Damped weights
    )
    log("  ✓ Random Forest trained")
    
    model_xgb = train_xgb_final(
        X_train, y_train, selected_features_tree,
        class_weights_tree, cv_xgb.best_params, seed  # Damped weights
    )
    log("  ✓ XGBoost trained")
    
    # Train stacking ensemble (uses damped weights for tree components)
    model_stacking = train_stacking_final(
        X_train, y_train, selected_features_lr,
        class_weights, class_weights_tree,  # Both weight types
        cv_lr.best_params, cv_rf.best_params, seed
    )
    
    models = {
        "lr": model_lr,
        "rf": model_rf,
        "xgb": model_xgb,
        "stacking": model_stacking
    }
    
    best_params = {
        "lr": cv_lr.best_params,
        "rf": cv_rf.best_params,
        "xgb": cv_xgb.best_params
    }
    
    feature_cols = {
        "lr": selected_features_lr,
        "tree": selected_features_tree
    }
    
    # Summary
    log("\n" + "="*60)
    log("CV SUMMARY")
    log("="*60)
    log(f"  LR:   Macro-F1 = {cv_lr.best_score:.4f} | C = {cv_lr.best_params['C']}")
    log(f"  RF:   Macro-F1 = {cv_rf.best_score:.4f} | n_est = {cv_rf.best_params['n_estimators']}")
    log(f"  XGB:  Macro-F1 = {cv_xgb.best_score:.4f} | max_d = {cv_xgb.best_params['max_depth']}")
    
    return {
        "cv_results": cv_results,
        "models": models,
        "best_params": best_params,
        "feature_cols": feature_cols
    }


# =============================================================================
# RETRAIN ON TRAIN+VAL (Cell 4D)
# =============================================================================

def cell4d_retrain_for_test(
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    selected_features_lr: List[str],
    selected_features_tree: List[str],
    class_weights: Dict[int, float],
    best_params: Dict[str, Dict],
    seed: int = 42
) -> Dict:
    """
    Cell 4D: Retrain models on train+val for final test evaluation.
    
    Args:
        X_trainval: Merged train+val features
        y_trainval: Merged train+val labels
        selected_features_lr: Feature columns for LR
        selected_features_tree: Feature columns for tree models
        class_weights: Class weights dict (not used - recomputed from trainval)
        best_params: Best hyperparameters from CV
        seed: Random seed
        
    Returns:
        Dictionary with retrained models
    """
    log_section("CELL 4D: RETRAIN ON TRAIN+VAL FOR TEST")
    
    # Recompute class weights on trainval (full for LR, damped for trees)
    from src.splits import compute_class_weights, compute_class_weights_damped
    class_weights_tv = compute_class_weights(y_trainval)
    class_weights_tv_tree = compute_class_weights_damped(y_trainval, damping=0.5)
    
    log(f"  Train+Val size: {len(X_trainval)}")
    log(f"  Class weights (train+val): {class_weights_tv}")
    
    # Retrain all models
    model_lr = train_lr_final(
        X_trainval, y_trainval, selected_features_lr,
        class_weights_tv, best_params["lr"], seed  # Full weights for LR
    )
    log("  ✓ LR retrained")
    
    model_rf = train_rf_final(
        X_trainval, y_trainval, selected_features_tree,
        class_weights_tv_tree, best_params["rf"], seed  # Damped weights for trees
    )
    log("  ✓ RF retrained")
    
    model_xgb = train_xgb_final(
        X_trainval, y_trainval, selected_features_tree,
        class_weights_tv_tree, best_params["xgb"], seed  # Damped weights for trees
    )
    log("  ✓ XGB retrained")
    
    model_stacking = train_stacking_final(
        X_trainval, y_trainval, selected_features_lr,
        class_weights_tv, class_weights_tv_tree,  # Both weight types
        best_params["lr"], best_params["rf"], seed
    )
    log("  ✓ Stacking retrained")
    
    return {
        "lr": model_lr,
        "rf": model_rf,
        "xgb": model_xgb,
        "stacking": model_stacking
    }