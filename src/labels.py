"""
Label construction, benchmark labels, and unsupervised diagnostics.
Maps to: Cell 3A (outcome-based), Cell 3B (rule-based benchmark), Cell 3C (unsupervised)

Outputs: regime_df, LABEL_COLS, BENCHMARK_COLS, SCORE_COLS, EXCLUDE_COLS, UNSUPERVISED_ANALYSIS

================================================================================
METHODOLOGICAL DEFENSE: OUTCOME-BASED REGIME LABELS
================================================================================

This module implements **outcome-based** regime labeling, which defines regimes
by realized forward returns rather than contemporaneous market characteristics.
This is a deliberate design choice with important implications.

KEY METHODOLOGICAL POINTS:

1. **Forecasting Task, Not Regime Identification**
   -----------------------------------------------
   We are NOT claiming to identify "true" macroeconomic regimes (recession, 
   expansion, etc.). Instead, we frame this as a **return-state forecasting** 
   problem: predicting whether the next H weeks will exhibit bear-like, 
   neutral, or bull-like returns.
   
   This is analogous to forecasting volatility regimes in GARCH models - we
   predict future states based on current information, not identify hidden
   states that exist independently.

2. **Apparent Circularity is Intentional**
   ----------------------------------------
   Critics may note that labels depend on future returns, creating apparent
   circularity. However:
   
   - Features use ONLY lagged (t-1) information
   - Labels use ONLY forward (t+H) information  
   - There is a strict +1 week gap between feature computation and label start
   - This creates a legitimate supervised learning problem: X(t) → Y(t+H)
   
   The "circularity" is actually the standard setup for any forecasting task.
   We would face the same structure predicting next-week returns, volatility,
   or any other forward-looking target.

3. **Volatility-Adjusted Thresholds**
   -----------------------------------
   Regimes are defined as:
   - Bear:    fwd_return < -k * σ
   - Neutral: |fwd_return| ≤ k * σ
   - Bull:    fwd_return > +k * σ
   
   Where σ is rolling realized volatility and k is a tunable parameter (default 0.3).
   
   This approach:
   - Adapts to changing market conditions (2008 vs 2017)
   - Creates more balanced classes than fixed thresholds
   - Reflects economic significance (a 2% move means more in low-vol periods)

4. **Why Not Traditional Regime Models?**
   ---------------------------------------
   Hidden Markov Models (HMM) or Markov-Switching models are alternatives, but:
   - They require strong distributional assumptions
   - Regime identification is retrospective (labels change with new data)
   - Our ML approach allows richer feature sets and non-linear relationships
   - Our labels are fixed once computed, enabling standard train/test evaluation

5. **Limitations We Acknowledge**
   ------------------------------
   - Labels are retrospective (we know them only after the fact)
   - Class boundaries are somewhat arbitrary (k parameter)
   - 3-class discretization loses information vs continuous returns
   - Sample size (~700 weeks) limits model complexity
   
   These limitations are addressed through:
   - Extensive stability analysis (rolling metrics)
   - Bootstrap confidence intervals
   - Multiple baseline comparisons
   - Transaction cost sensitivity analysis

REFERENCES:
- Ang & Bekaert (2002): "Regime Switches in Interest Rates"
- Guidolin & Timmermann (2007): "Asset Allocation under Multivariate Regime Switching"
- Our approach follows the "predict future states" interpretation rather than
  the "identify hidden states" interpretation of regime modeling.

================================================================================
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from src.utils import log, log_section, log_debug


# =============================================================================
# COLUMN DEFINITIONS (used throughout the project)
# =============================================================================

# Label-side columns from Cell 3A (forward-looking, NEVER in X)
LABEL_COLS = [
    "SPX_ret_fwd_H",      # forward return (target construction)
    "vol_H",              # horizon-scaled volatility
    "pos_frac_fwdH",      # forward positive week fraction
    "neg_frac_fwdH",      # forward negative week fraction
    "regime_target",      # the target itself
]

# Benchmark columns from Cell 3B (must not influence ML)
BENCHMARK_COLS = [
    "trend_score",
    "regime_label",
    "regime_label_strict",
    "score_trend_dir",
    "score_trend_strength",
    "score_momentum",
    "score_volatility",
    "score_cross_asset",
    "score_regime_arc",
    "score_streak",
]

# Score columns (subset of BENCHMARK_COLS)
SCORE_COLS = [
    "score_trend_dir",
    "score_trend_strength",
    "score_momentum",
    "score_volatility",
    "score_cross_asset",
    "score_regime_arc",
    "score_streak",
]

# Auxiliary columns
AUX_COLS = [
    "vol_1w",        # intermediate calculation
    "cluster_label", # added by Cell 3C
]


def get_exclude_cols() -> Set[str]:
    """Get the full set of columns to exclude from ML features."""
    return set(LABEL_COLS + BENCHMARK_COLS + AUX_COLS)


# =============================================================================
# CELL 3A: OUTCOME-BASED REGIME TARGET
# =============================================================================

def cell3a_build_regime_target(
    features_df: pd.DataFrame,
    prices_w: pd.DataFrame,
    horizon: int = 1,
    vol_win: int = 52,
    k: float = 0.3,
    persist_frac: float = 0.35,
    use_persistence: bool = False,
    train_end: str = "2017-12-29"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Cell 3A: Build outcome-based regime target (vol-adjusted).
    
    METHODOLOGICAL NOTES:
    ---------------------
    This function implements **outcome-based** regime labeling, which is a
    deliberate design choice distinct from traditional regime identification.
    
    We are NOT claiming to identify "true" macroeconomic regimes. Instead,
    we frame this as a RETURN-STATE FORECASTING problem:
    
        "Given current market conditions X(t), predict whether the next H weeks
         will exhibit bear-like, neutral, or bull-like returns."
    
    The apparent "circularity" (labels depend on future returns) is actually
    the standard structure for any forecasting task. The key safeguards are:
    
    1. Features X(t) use ONLY lagged (t-1 and earlier) information
    2. Labels Y(t) use ONLY forward (t+H) information
    3. Strict +1 week gap between feature computation and label start
    
    Regime Definition:
    ------------------
    - Bull (+1):   fwd_return(t→t+H) >  +k × σ_H(t)
    - Bear (−1):   fwd_return(t→t+H) <  −k × σ_H(t)
    - Neutral (0): |fwd_return| ≤ k × σ_H(t)
    
    Where σ_H is horizon-scaled trailing volatility (computed BEFORE t).
    
    Why volatility-adjusted thresholds?
    - A 3% weekly move is extreme in low-vol 2017 but normal in 2008
    - Adapts regime boundaries to market conditions
    - Creates more balanced class distributions
    
    Args:
        features_df: Feature DataFrame from Cell 2 (strictly lagged features)
        prices_w: Weekly prices aligned to features
        horizon: H weeks forecast horizon (default: 1)
        vol_win: Volatility lookback window in weeks (default: 52)
        k: Volatility multiplier for regime bands (default: 0.3)
        persist_frac: Minimum fraction for persistence check
        use_persistence: Whether to use persistence filtering
        train_end: End date for training (for distribution stats)
        
    Returns:
        Tuple of (regime_df, label_params_dict)
    """
    log_section("CELL 3A: OUTCOME-BASED REGIME TARGET (VOL-ADJ)")
    
    H = horizon
    VOL_WIN = vol_win
    K = k
    PERSIST_FRAC = persist_frac
    USE_PERSISTENCE = use_persistence
    
    # Auto-disable persistence for H=1 (meaningless for single-week horizon)
    if H == 1 and USE_PERSISTENCE:
        log(f"⚠️ USE_PERSISTENCE=True but H=1; persistence disabled (single week).")
        USE_PERSISTENCE = False
    
    # Strict alignment BEFORE any shifting (fatal if mismatch)
    assert features_df.index.equals(prices_w.index), \
        "Feature/Price index mismatch (must be identical)."
    
    spx = prices_w["SPX"].reindex(features_df.index)
    
    # Future outcome (label-side, forward-looking — allowed for target)
    SPX_ret_fwd_H = np.log(spx.shift(-H) / spx)
    
    # Past-only volatility (strictly non-leaking)
    ret_1w = np.log(spx / spx.shift(1))
    vol_1w = ret_1w.rolling(window=VOL_WIN).std().shift(1)  # uses info through t-1 only
    vol_H = vol_1w * np.sqrt(H)  # horizon scaling (mandatory)
    
    # Forward-path persistence (only computed if H > 1 and USE_PERSISTENCE)
    if H > 1 and USE_PERSISTENCE:
        pos_count = pd.Series(0.0, index=features_df.index)
        neg_count = pd.Series(0.0, index=features_df.index)
        
        for i in range(1, H + 1):
            future_ret = ret_1w.shift(-i)
            pos_count += (future_ret > 0).astype(float)
            neg_count += (future_ret < 0).astype(float)
        
        pos_frac_fwdH = pos_count / H
        neg_frac_fwdH = neg_count / H
    else:
        # H=1 or persistence disabled: set to 1.0 (always passes persistence check)
        pos_frac_fwdH = pd.Series(1.0, index=features_df.index)
        neg_frac_fwdH = pd.Series(1.0, index=features_df.index)
    
    # Regime rules (magnitude + optional persistence)
    bull_mag = SPX_ret_fwd_H > (K * vol_H)
    bear_mag = SPX_ret_fwd_H < (-K * vol_H)
    
    if USE_PERSISTENCE:
        bull_cond = bull_mag & (pos_frac_fwdH >= PERSIST_FRAC)
        bear_cond = bear_mag & (neg_frac_fwdH >= PERSIST_FRAC)
    else:
        bull_cond = bull_mag
        bear_cond = bear_mag
    
    # Sanity check: no overlap
    assert not (bull_cond & bear_cond).any(), \
        "Fatal overlap: Bull and Bear simultaneously true."
    
    regime_target = pd.Series(0, index=features_df.index, name="regime_target")
    regime_target.loc[bull_cond] = 1
    regime_target.loc[bear_cond] = -1
    
    # Assemble canonical dataset (label columns must be excluded later from X)
    regime_df = features_df.copy()
    regime_df["SPX_ret_fwd_H"] = SPX_ret_fwd_H
    regime_df["vol_1w"] = vol_1w
    regime_df["vol_H"] = vol_H
    regime_df["pos_frac_fwdH"] = pos_frac_fwdH
    regime_df["neg_frac_fwdH"] = neg_frac_fwdH
    regime_df["regime_target"] = regime_target
    
    # Drop rows where label cannot be computed (warmup + tail)
    cols_needed = ["SPX_ret_fwd_H", "vol_H"]
    rows_before = len(regime_df)
    regime_df = regime_df.dropna(subset=cols_needed).copy()
    dropped = rows_before - len(regime_df)
    
    # Mandatory integrity checks
    assert regime_df.index.is_monotonic_increasing, "Index not sorted."
    assert set(regime_df["regime_target"].unique()).issubset({-1, 0, 1}), \
        "Invalid labels found."
    
    train_mask_for_stats = regime_df.index <= pd.Timestamp(train_end)
    assert train_mask_for_stats.any(), "No training rows left after drops."
    assert regime_df.loc[train_mask_for_stats].shape[0] >= 200, \
        "Too few training rows after warmups."
    
    # Output diagnostics
    persist_str = f"PERSIST={PERSIST_FRAC:.0%}" if USE_PERSISTENCE else "PERSIST=off"
    log(f"✓ Target ready: n={len(regime_df)} | Dropped={dropped} | "
        f"H={H} | VOL_WIN={VOL_WIN} | K={K} | {persist_str}")
    
    label_map = {-1: "Bear", 0: "Neutral", 1: "Bull"}
    train_dist = regime_df.loc[train_mask_for_stats, "regime_target"].value_counts(
        normalize=True).reindex([-1, 0, 1]).fillna(0)
    all_dist = regime_df["regime_target"].value_counts(
        normalize=True).reindex([-1, 0, 1]).fillna(0)
    
    log("Dist (train): " + " | ".join([f"{label_map[k]} {train_dist[k]:.1%}" for k in [-1, 0, 1]]))
    log("Dist (all):   " + " | ".join([f"{label_map[k]} {all_dist[k]:.1%}" for k in [-1, 0, 1]]))
    
    # Neutral fraction check (informational)
    neutral_pct = all_dist[0]
    if neutral_pct > 0.50:
        log(f"⚠️ Neutral > 50% ({neutral_pct:.1%}). Consider lowering K to reduce neutral zone.")
    elif neutral_pct < 0.15:
        log(f"⚠️ Neutral < 15% ({neutral_pct:.1%}). Consider raising K if regimes are too noisy.")
    
    # Store label settings (for documentation/reproducibility)
    LABEL_PARAMS = {
        "horizon_weeks": H,
        "vol_win_weeks": VOL_WIN,
        "k_vol_units": float(K),
        "persist_frac": float(PERSIST_FRAC) if USE_PERSISTENCE else None,
        "use_persistence": bool(USE_PERSISTENCE),
    }
    
    log(f"\nLabel parameters stored: {LABEL_PARAMS}")
    
    return regime_df, LABEL_PARAMS


def compute_k_sensitivity(
    SPX_ret_fwd_H: pd.Series,
    vol_H: pd.Series,
    k_values: List[float] = None
) -> pd.DataFrame:
    """
    K parameter sensitivity analysis.
    
    Args:
        SPX_ret_fwd_H: Forward returns
        vol_H: Horizon-scaled volatility
        k_values: List of K values to test
        
    Returns:
        DataFrame with sensitivity results
    """
    if k_values is None:
        k_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    sensitivity_results = []
    
    for k_test in k_values:
        bull_test = SPX_ret_fwd_H > (k_test * vol_H)
        bear_test = SPX_ret_fwd_H < (-k_test * vol_H)
        
        # Handle NaN
        valid_mask = ~(SPX_ret_fwd_H.isna() | vol_H.isna())
        n_valid = valid_mask.sum()
        
        n_bull = (bull_test & valid_mask).sum()
        n_bear = (bear_test & valid_mask).sum()
        n_neutral = n_valid - n_bull - n_bear
        
        sensitivity_results.append({
            "K": k_test,
            "Bull %": n_bull / n_valid if n_valid > 0 else 0,
            "Neutral %": n_neutral / n_valid if n_valid > 0 else 0,
            "Bear %": n_bear / n_valid if n_valid > 0 else 0,
        })
    
    return pd.DataFrame(sensitivity_results)


# =============================================================================
# CELL 3B: RULE-BASED BENCHMARK
# =============================================================================

def cell3b_build_benchmark(
    regime_df: pd.DataFrame,
    thresh: int = 3,
    thresh_strict: int = 4
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cell 3B: Build rule-based benchmark regime (composite score).
    
    Uses only lagged features (already enforced in Cell 2).
    Does NOT use any label-side columns from Cell 3A.
    
    Args:
        regime_df: DataFrame from Cell 3A
        thresh: Threshold for regime assignment (default: 3 of 7 signals)
        thresh_strict: Stricter threshold (default: 4 of 7)
        
    Returns:
        Tuple of (updated regime_df, SCORE_COLS list)
    """
    log_section("CELL 3B: BENCHMARK RULE-BASED REGIME (WEEKLY COMPOSITE SCORE)")
    
    # Clean-up (prevents silent changes on reruns)
    old_score_cols = [c for c in regime_df.columns if c.startswith("score_")]
    old_bench_cols = [c for c in ["trend_score", "regime_label", "regime_label_strict"] 
                      if c in regime_df.columns]
    if old_score_cols or old_bench_cols:
        regime_df = regime_df.drop(columns=old_score_cols + old_bench_cols).copy()
    
    # Explicit score components (WEEKLY)
    # Notes on weekly mapping: MA200 proxy = 40w, MA50 proxy = 10w, "20d" proxy = 4w return
    
    regime_df["score_trend_dir"] = (
        (regime_df["SPX_vs_ma200"] > 1.00).astype(int) -
        (regime_df["SPX_vs_ma200"] < 0.98).astype(int)
    )
    
    regime_df["score_trend_strength"] = np.select(
        [regime_df["SPX_trend_strength"] > 0.01,
         regime_df["SPX_trend_strength"] < -0.01],
        [1, -1],
        default=0
    )
    
    # Momentum over ~4 weeks (stored as SPX_ret_20d for compatibility)
    regime_df["score_momentum"] = np.select(
        [regime_df["SPX_ret_20d"] > 0.01,
         regime_df["SPX_ret_20d"] < -0.01],
        [1, -1],
        default=0
    )
    
    # Volatility proxy (VIX vs 4w MA; stored as VIX_vs_ma20 for compatibility)
    regime_df["score_volatility"] = np.select(
        [regime_df["VIX_vs_ma20"] > 1.20,
         regime_df["VIX_vs_ma20"] < 0.90],
        [-1, 1],
        default=0
    )
    
    # Cross-asset risk-on/off (stocks vs bonds relative momentum)
    regime_df["score_cross_asset"] = np.sign(regime_df["SPX_vs_TLT_20d"]).astype(int)
    
    # Regime arc (~10w high/low; stored as *_50 for compatibility)
    regime_df["score_regime_arc"] = (
        (regime_df["SPX_vs_high_50"] > 0.98).astype(int) -
        (regime_df["SPX_vs_low_50"] < 1.02).astype(int)
    )
    
    # Weekly streak (>=2 consecutive up/down weeks)
    regime_df["score_streak"] = np.select(
        [regime_df["up_streak"] >= 2,
         regime_df["down_streak"] >= 2],
        [1, -1],
        default=0
    )
    
    # Hard check: no missing components
    missing_score_cols = [c for c in SCORE_COLS if c not in regime_df.columns]
    assert not missing_score_cols, f"Missing score columns: {missing_score_cols}"
    
    # Composite score (integer in [-7, +7])
    regime_df["trend_score"] = regime_df[SCORE_COLS].sum(axis=1).astype(int)
    
    # Map score -> benchmark regimes (vectorized, explicit integer thresholds)
    regime_df["regime_label"] = np.select(
        [regime_df["trend_score"] >= thresh,
         regime_df["trend_score"] <= -thresh],
        [1, -1],
        default=0
    ).astype(int)
    
    regime_df["regime_label_strict"] = np.select(
        [regime_df["trend_score"] >= thresh_strict,
         regime_df["trend_score"] <= -thresh_strict],
        [1, -1],
        default=0
    ).astype(int)
    
    # Diagnostics
    label_map = {-1: "Bear", 0: "Neutral", 1: "Bull"}
    dist = regime_df["regime_label"].value_counts(normalize=True).reindex([-1, 0, 1]).fillna(0)
    
    log(f"✓ Benchmark ready | components={len(SCORE_COLS)} | "
        f"score_range=[{regime_df['trend_score'].min()}, {regime_df['trend_score'].max()}] "
        f"| THRESH={thresh} | THRESH_STRICT={thresh_strict}")
    log("Dist (bench): " + " | ".join([f"{label_map[k]} {dist[k]:.1%}" for k in [-1, 0, 1]]))
    
    # Agreement check (should not be near-identity)
    match_pct = (regime_df["regime_label"] == regime_df["regime_target"]).mean()
    log(f"Agreement with target (regime_target): {match_pct:.1%}")
    if match_pct > 0.90:
        log("⚠️ Warning: benchmark is very close to target; verify no circular construction.")
    
    return regime_df, SCORE_COLS


# =============================================================================
# CELL 3C: UNSUPERVISED REGIME DISCOVERY
# =============================================================================

def cell3c_unsupervised_analysis(
    regime_df: pd.DataFrame,
    exclude_cols: Set[str],
    train_end: str = "2017-12-29",
    n_clusters: int = 3,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Cell 3C: Unsupervised regime discovery (PCA + KMeans).
    
    This is DESCRIPTIVE (not predictive) — a sanity check for latent structure.
    "Do feature clusters align with outcome-defined labels?"
    
    Args:
        regime_df: DataFrame from Cell 3B
        exclude_cols: Set of columns to exclude from clustering
        train_end: End date for training period
        n_clusters: Number of clusters for KMeans
        seed: Random seed
        
    Returns:
        Tuple of (updated regime_df with cluster_label, UNSUPERVISED_ANALYSIS dict)
    """
    log_section("CELL 3C: UNSUPERVISED REGIME DISCOVERY")
    
    # Prepare features for clustering (exclude label columns)
    cluster_features = [c for c in regime_df.columns if c not in exclude_cols]
    
    # Use only training period for fitting (no data leakage)
    train_mask = regime_df.index <= pd.Timestamp(train_end)
    X_cluster_train = regime_df.loc[train_mask, cluster_features].copy()
    X_cluster_all = regime_df[cluster_features].copy()
    
    # Standardize (fit on train only)
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_cluster_train)
    X_scaled_all = scaler.transform(X_cluster_all)
    
    log(f"✓ Clustering features: {len(cluster_features)}")
    log(f"  Training samples: {len(X_scaled_train)}")
    log(f"  All samples: {len(X_scaled_all)}")
    
    # PCA ANALYSIS
    log("\n" + "-"*60)
    log("PCA ANALYSIS")
    log("-"*60)
    
    # Fit PCA on training data
    n_pca_components = min(10, len(cluster_features))
    pca = PCA(n_components=n_pca_components, random_state=seed)
    pca.fit(X_scaled_train)
    
    # Transform all data
    X_pca_train = pca.transform(X_scaled_train)
    X_pca_all = pca.transform(X_scaled_all)
    
    # Variance explained
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    log(f"\nVariance explained by principal components:")
    for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_, cumvar)):
        log(f"  PC{i+1}: {var:.1%} (cumulative: {cum:.1%})")
        if cum > 0.95:
            log(f"  ... (remaining components explain < 5%)")
            break
    
    # Components needed for 80% variance
    n_components_80 = np.argmax(cumvar >= 0.80) + 1
    log(f"\nComponents for 80% variance: {n_components_80}")
    
    # Top loadings for PC1 and PC2
    log("\nTop feature loadings:")
    for pc_idx, pc_name in [(0, "PC1"), (1, "PC2")]:
        loadings = pd.Series(pca.components_[pc_idx], index=cluster_features)
        top_pos = loadings.nlargest(3)
        top_neg = loadings.nsmallest(3)
        log(f"\n  {pc_name}:")
        log(f"    Positive: {', '.join([f'{k}({v:.2f})' for k,v in top_pos.items()])}")
        log(f"    Negative: {', '.join([f'{k}({v:.2f})' for k,v in top_neg.items()])}")
    
    # KMEANS CLUSTERING
    log("\n" + "-"*60)
    log("KMEANS CLUSTERING")
    log("-"*60)
    
    # Test different numbers of clusters
    k_range = range(2, 7)
    silhouette_scores = []
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X_pca_train[:, :3])  # Use first 3 PCs
        
        sil_score = silhouette_score(X_pca_train[:, :3], labels)
        silhouette_scores.append(sil_score)
        inertias.append(kmeans.inertia_)
        
        log(f"  k={k}: Silhouette={sil_score:.3f}, Inertia={kmeans.inertia_:.1f}")
    
    # Optimal k by silhouette
    optimal_k = list(k_range)[np.argmax(silhouette_scores)]
    log(f"\nOptimal k (by silhouette): {optimal_k}")
    
    # Fit final model with k=3 (to match regime structure)
    log("\n" + "-"*60)
    log(f"FINAL CLUSTERING (k={n_clusters} for regime comparison)")
    log("-"*60)
    
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels_train = kmeans_final.fit_predict(X_pca_train[:, :3])
    cluster_labels_all = kmeans_final.predict(X_pca_all[:, :3])
    
    # Add cluster labels to regime_df
    regime_df["cluster_label"] = cluster_labels_all
    
    # CLUSTER vs REGIME COMPARISON
    log("\n" + "-"*60)
    log("CLUSTER vs OUTCOME-BASED REGIME COMPARISON")
    log("-"*60)
    
    # Get true regime labels for training period
    y_train_regime = regime_df.loc[train_mask, "regime_target"].values
    
    # Adjusted Rand Index (measures clustering agreement)
    ari = adjusted_rand_score(y_train_regime, cluster_labels_train)
    log(f"\nAdjusted Rand Index (clusters vs regime_target): {ari:.3f}")
    log("  ARI = 1.0: Perfect agreement")
    log("  ARI = 0.0: Random labeling")
    log("  ARI < 0.0: Worse than random")
    
    # Cross-tabulation
    log("\nCross-tabulation (Training Period):")
    log("  Rows: Outcome-based regime | Cols: KMeans cluster")
    
    regime_labels = regime_df.loc[train_mask, "regime_target"].values
    crosstab = pd.crosstab(
        pd.Series(regime_labels, name="Regime"),
        pd.Series(cluster_labels_train, name="Cluster"),
        margins=True
    )
    log(str(crosstab))
    
    # CLUSTER CHARACTERISTICS
    log("\n" + "-"*60)
    log("CLUSTER CHARACTERISTICS")
    log("-"*60)
    
    cluster_stats = []
    for c in range(n_clusters):
        mask = cluster_labels_train == c
        cluster_regimes = y_train_regime[mask]
        
        # Dominant regime in this cluster
        regime_counts = pd.Series(cluster_regimes).value_counts(normalize=True)
        dominant_regime = regime_counts.idxmax()
        dominant_pct = regime_counts.max()
        
        # Feature means for this cluster (in original scale)
        cluster_means = X_cluster_train.iloc[mask].mean()
        
        stats = {
            "cluster": c,
            "n_samples": mask.sum(),
            "dominant_regime": {-1: "Bear", 0: "Neutral", 1: "Bull"}[dominant_regime],
            "dominant_pct": dominant_pct,
            "SPX_ret_1w_mean": cluster_means.get("SPX_ret_1w", 0),
            "VIX_ma20_mean": cluster_means.get("VIX_ma20", 0),
            "SPX_vs_ma200_mean": cluster_means.get("SPX_vs_ma200", 0),
        }
        cluster_stats.append(stats)
        
        log(f"\nCluster {c}:")
        log(f"  Samples: {stats['n_samples']}")
        log(f"  Dominant regime: {stats['dominant_regime']} ({stats['dominant_pct']:.1%})")
        log(f"  Avg SPX_ret_1w: {stats['SPX_ret_1w_mean']:.4f}")
        log(f"  Avg VIX_ma20: {stats['VIX_ma20_mean']:.2f}")
        log(f"  Avg SPX_vs_ma200: {stats['SPX_vs_ma200_mean']:.4f}")
    
    # INTERPRETATION
    log("\n" + "-"*60)
    log("INTERPRETATION")
    log("-"*60)
    
    cluster_df = pd.DataFrame(cluster_stats)
    
    # Identify which cluster is "risk-on", "risk-off", "transition"
    vix_col = "VIX_ma20_mean"
    if vix_col in cluster_df.columns:
        high_vix_cluster = cluster_df[vix_col].idxmax()
        low_vix_cluster = cluster_df[vix_col].idxmin()
        
        log(f"\nCluster interpretation (based on features):")
        log(f"  High volatility cluster: Cluster {cluster_df.loc[high_vix_cluster, 'cluster']} "
            f"(VIX avg: {cluster_df.loc[high_vix_cluster, vix_col]:.1f})")
        log(f"  Low volatility cluster: Cluster {cluster_df.loc[low_vix_cluster, 'cluster']} "
            f"(VIX avg: {cluster_df.loc[low_vix_cluster, vix_col]:.1f})")
    
    alignment_msg = "shows" if ari > 0.1 else "does not show"
    structure_msg = ("This suggests regimes have structure in feature space." if ari > 0.1 
                     else "This suggests outcome-based regimes are harder to detect from features alone.")
    
    log(f"""
Key Finding:
The unsupervised clustering {alignment_msg} meaningful
alignment with the outcome-based regime labels (ARI = {ari:.3f}).

{structure_msg}
""")
    
    # Store results
    cluster_centers = kmeans_final.cluster_centers_
    distances = kmeans_final.transform(X_pca_all[:, :3])
    cluster_proba = 1 / (1 + distances)  # Simple inverse distance
    cluster_proba = cluster_proba / cluster_proba.sum(axis=1, keepdims=True)
    
    UNSUPERVISED_ANALYSIS = {
        "pca": pca,
        "kmeans": kmeans_final,
        "scaler": scaler,
        "cluster_labels": cluster_labels_all,
        "pca_components": X_pca_all,
        "cluster_stats": cluster_stats,
        "ari_score": ari,
        "silhouette_scores": dict(zip(k_range, silhouette_scores)),
        "feature_list": cluster_features,
        "n_components_80": n_components_80,
    }
    
    log("\n" + "="*80)
    log("✓ CELL 3C COMPLETE")
    log("="*80)
    log(f"  Analyses completed:")
    log(f"    - PCA: {n_components_80} components for 80% variance")
    log(f"    - KMeans: k={n_clusters} clusters (silhouette={silhouette_scores[1]:.3f})")
    log(f"    - Regime alignment: ARI={ari:.3f}")
    log(f"\n  Column 'cluster_label' added to regime_df")
    
    return regime_df, UNSUPERVISED_ANALYSIS


# =============================================================================
# COMBINED CELL 3 FUNCTION
# =============================================================================

def cell3_build_labels(
    features_df: pd.DataFrame,
    prices_w: pd.DataFrame,
    horizon: int = 1,
    vol_win: int = 52,
    k: float = 0.3,
    persist_frac: float = 0.35,
    use_persistence: bool = False,
    train_end: str = "2017-12-29",
    seed: int = 42
) -> Tuple[pd.DataFrame, Set[str], Dict, Dict]:
    """
    Combined Cell 3: Build all labels (3A + 3B + 3C).
    
    Args:
        features_df: Feature DataFrame from Cell 2
        prices_w: Weekly prices aligned to features
        horizon: H weeks forecast horizon
        vol_win: Volatility lookback window
        k: Volatility multiplier for regime bands
        persist_frac: Persistence fraction
        use_persistence: Whether to use persistence
        train_end: Training end date
        seed: Random seed
        
    Returns:
        Tuple of (regime_df, EXCLUDE_COLS, LABEL_PARAMS, UNSUPERVISED_ANALYSIS)
    """
    # Cell 3A: Outcome-based target
    regime_df, LABEL_PARAMS = cell3a_build_regime_target(
        features_df=features_df,
        prices_w=prices_w,
        horizon=horizon,
        vol_win=vol_win,
        k=k,
        persist_frac=persist_frac,
        use_persistence=use_persistence,
        train_end=train_end
    )
    
    # Cell 3B: Rule-based benchmark
    regime_df, _ = cell3b_build_benchmark(regime_df)
    
    # Build EXCLUDE_COLS
    EXCLUDE_COLS = get_exclude_cols()
    log(f"\n✓ EXCLUDE_COLS defined: {len(EXCLUDE_COLS)} columns")
    log(f"  Label-side: {len(LABEL_COLS)}")
    log(f"  Benchmark: {len(BENCHMARK_COLS)}")
    log(f"  Auxiliary: {len(AUX_COLS)}")
    
    # Cell 3C: Unsupervised analysis
    regime_df, UNSUPERVISED_ANALYSIS = cell3c_unsupervised_analysis(
        regime_df=regime_df,
        exclude_cols=EXCLUDE_COLS,
        train_end=train_end,
        n_clusters=3,
        seed=seed
    )
    
    # Update EXCLUDE_COLS to include cluster_label
    EXCLUDE_COLS.add("cluster_label")
    
    return regime_df, EXCLUDE_COLS, LABEL_PARAMS, UNSUPERVISED_ANALYSIS