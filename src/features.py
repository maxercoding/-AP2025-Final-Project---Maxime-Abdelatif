"""
Feature engineering.
Maps to: Cell 2 (feature engineering)

All indicators/technical features + lagging (+1 week "iron rule").
Outputs: features_df, prices_w (aligned), FEATURE_COLS
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils import log, log_section, log_debug

# =============================================================================
# INDIVIDUAL FEATURE FUNCTIONS (exactly as in notebook Cell 2)
# =============================================================================

def _compute_weekly_returns(df_w: pd.DataFrame) -> pd.DataFrame:
    """
    1) Weekly log returns (t uses t-1..t). Will be lagged by +1 later.
    """
    feat = pd.DataFrame(index=df_w.index)
    feat["SPX_ret_1w"] = np.log(df_w["SPX"] / df_w["SPX"].shift(1))
    feat["TLT_ret_1w"] = np.log(df_w["TLT"] / df_w["TLT"].shift(1))
    feat["GOLD_ret_1w"] = np.log(df_w["GOLD"] / df_w["GOLD"].shift(1))
    feat["VIX_ret_1w"] = np.log(df_w["VIX"] / df_w["VIX"].shift(1))
    return feat


def _compute_trend_ma_features(df_w: pd.DataFrame) -> pd.DataFrame:
    """
    2) Trend / MA features (daily->weekly mapping: 20d≈4w, 50d≈10w, 200d≈40w)
    """
    feat = pd.DataFrame(index=df_w.index)
    
    ma_4w = df_w["SPX"].rolling(4).mean()
    ma_10w = df_w["SPX"].rolling(10).mean()
    ma_40w = df_w["SPX"].rolling(40).mean()
    
    feat["SPX_ma200"] = ma_40w  # keep name for downstream compatibility
    feat["SPX_ma50"] = ma_10w
    feat["SPX_vs_ma20"] = df_w["SPX"] / ma_4w
    feat["SPX_vs_ma50"] = df_w["SPX"] / ma_10w
    feat["SPX_vs_ma200"] = df_w["SPX"] / ma_40w
    feat["SPX_trend_strength"] = (ma_10w / ma_40w) - 1.0
    
    return feat


def _compute_momentum_features(df_w: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum windows (keep legacy names where used downstream)
    """
    feat = pd.DataFrame(index=df_w.index)
    feat["SPX_ret_20d"] = np.log(df_w["SPX"] / df_w["SPX"].shift(4))   # ~1 month
    feat["SPX_ret_50d"] = np.log(df_w["SPX"] / df_w["SPX"].shift(10))  # ~1 quarter
    feat["SPX_ret_12w"] = np.log(df_w["SPX"] / df_w["SPX"].shift(12))  # ~3 months
    return feat


def _compute_vix_features(df_w: pd.DataFrame) -> pd.DataFrame:
    """
    3) VIX features
    """
    feat = pd.DataFrame(index=df_w.index)
    vix_ma_4w = df_w["VIX"].rolling(4).mean()
    feat["VIX_ma20"] = vix_ma_4w
    feat["VIX_vs_ma20"] = df_w["VIX"] / vix_ma_4w
    return feat


def _compute_cross_asset_features(
    df_w: pd.DataFrame, 
    spx_ret_4w: pd.Series
) -> pd.DataFrame:
    """
    4) Cross-asset (levels + relative momentum)
    """
    feat = pd.DataFrame(index=df_w.index)
    
    feat["GOLD_SPX_ratio"] = df_w["GOLD"] / df_w["SPX"]
    feat["TLT_SPX_ratio"] = df_w["TLT"] / df_w["SPX"]
    
    gold_ret_4w = np.log(df_w["GOLD"] / df_w["GOLD"].shift(4))
    tlt_ret_4w = np.log(df_w["TLT"] / df_w["TLT"].shift(4))
    
    feat["SPX_vs_GOLD_20d"] = spx_ret_4w - gold_ret_4w
    feat["SPX_vs_TLT_20d"] = spx_ret_4w - tlt_ret_4w
    
    return feat


def _compute_regime_arc_features(df_w: pd.DataFrame) -> pd.DataFrame:
    """
    5) Regime arc (10w ≈ 50d)
    """
    feat = pd.DataFrame(index=df_w.index)
    
    high_10w = df_w["SPX"].rolling(10).max()
    low_10w = df_w["SPX"].rolling(10).min()
    
    feat["SPX_high_past_50"] = high_10w
    feat["SPX_low_past_50"] = low_10w
    feat["SPX_vs_high_50"] = df_w["SPX"] / high_10w
    feat["SPX_vs_low_50"] = df_w["SPX"] / low_10w
    
    feat["near_high_50"] = (feat["SPX_vs_high_50"] > 0.98).astype(int)
    feat["near_low_50"] = (feat["SPX_vs_low_50"] < 1.02).astype(int)
    
    return feat


def _compute_streak_features(spx_ret_1w: pd.Series) -> pd.DataFrame:
    """
    6) Streak features (vectorized)
    """
    feat = pd.DataFrame(index=spx_ret_1w.index)
    
    dir_1w = np.sign(spx_ret_1w).fillna(0)
    grouper = (dir_1w != dir_1w.shift(1)).cumsum()
    streak = dir_1w.groupby(grouper).cumcount() + 1
    
    feat["up_streak"] = np.where(dir_1w > 0, streak, 0)
    feat["down_streak"] = np.where(dir_1w < 0, streak, 0)
    
    return feat


def _compute_seasonal_features(df_w: pd.DataFrame) -> pd.DataFrame:
    """
    7) Seasonal (weekly-safe). Lag it too to keep the "single lag rule".
    """
    feat = pd.DataFrame(index=df_w.index)
    feat["month"] = df_w.index.month
    feat["quarter"] = df_w.index.quarter
    feat["is_quarter_end"] = df_w.index.is_quarter_end.astype(int)
    return feat


def _compute_transition_features(
    df_w: pd.DataFrame,
    spx_ret_1w: pd.Series,
    tlt_ret_1w: pd.Series
) -> pd.DataFrame:
    """
    8) Regime transition features (Markov-ish, past-only)
    """
    feat = pd.DataFrame(index=df_w.index)
    
    # Rolling % weeks up in last 12 weeks
    rolling_up_pct = (spx_ret_1w > 0).rolling(12).mean()
    feat["rolling_up_pct_12w"] = rolling_up_pct
    
    # Rolling volatility of returns (past only)
    feat["SPX_ret_vol_12w"] = spx_ret_1w.rolling(12).std()
    
    # Rolling correlation SPX-TLT (past only, risk-on/off indicator)
    feat["corr_SPX_TLT_12w"] = spx_ret_1w.rolling(12).corr(tlt_ret_1w)
    
    # Drawdown vs 52-week high (past only)
    spx_high_52w = df_w["SPX"].rolling(52).max()
    feat["drawdown_from_52w_high"] = df_w["SPX"] / spx_high_52w - 1
    
    return feat


def _compute_macro_proxy_features(
    df_w: pd.DataFrame,
    spx_ret_1w: pd.Series,
    tlt_ret_1w: pd.Series
) -> pd.DataFrame:
    """
    9) Macro proxy features (cross-asset signals)
    """
    feat = pd.DataFrame(index=df_w.index)
    
    # Flight-to-quality signal
    feat["flight_to_quality"] = np.sign(spx_ret_1w - tlt_ret_1w)
    
    # VIX slope (4w MA vs current — rising or falling vol)
    feat["VIX_slope_4w"] = df_w["VIX"] / df_w["VIX"].rolling(4).mean() - 1
    
    return feat


# =============================================================================
# FEATURE METADATA (from notebook)
# =============================================================================

def generate_feature_metadata(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-generate feature metadata based on naming conventions.
    All features in this project use +1 week lag (IRON RULE in Cell 2).
    """
    metadata = []
    for col in feature_df.columns:
        meta = {
            "feature": col,
            "shift_applied": 1,  # All features shifted +1 in Cell 2
            "forward_looking": False,
            "max_lookback": None,
        }
        
        # Infer lookback from naming
        if "ma200" in col or "40w" in col:
            meta["max_lookback"] = "40 weeks"
        elif "ma50" in col or "10w" in col:
            meta["max_lookback"] = "10 weeks"
        elif "ma20" in col or "4w" in col:
            meta["max_lookback"] = "4 weeks"
        elif "52w" in col or "VOL_WIN" in col:
            meta["max_lookback"] = "52 weeks"
        elif "12w" in col:
            meta["max_lookback"] = "12 weeks"
        elif "1w" in col:
            meta["max_lookback"] = "1 week"
        elif "streak" in col:
            meta["max_lookback"] = "cumulative"
        elif col in ["month", "quarter", "is_quarter_end"]:
            meta["max_lookback"] = "calendar"
        else:
            meta["max_lookback"] = "unknown"
        
        metadata.append(meta)
    
    return pd.DataFrame(metadata)


# =============================================================================
# MAIN CELL FUNCTION
# =============================================================================

def cell2_build_features(
    df_w: pd.DataFrame,
    prices_w: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    """
    Cell 2: Weekly feature engineering with strict lag.
    
    THE IRON RULE: all ML inputs are lagged by exactly +1 week.
    
    Args:
        df_w: Weekly price DataFrame from Cell 1
        prices_w: Weekly prices DataFrame from Cell 1
        
    Returns:
        Tuple of (features_df, prices_w_aligned, FEATURE_COLS, FEATURE_META):
            features_df: Feature DataFrame with +1 week lag applied
            prices_w_aligned: Prices aligned to feature index
            FEATURE_COLS: List of feature column names
            FEATURE_META: Feature metadata DataFrame
    """
    log_section("CELL 2: WEEKLY FEATURE ENGINEERING (STRICT LAG)")
    
    # 0) Feature frame (engineer on df_w, then apply one global lag at the end)
    feat = pd.DataFrame(index=df_w.index)
    
    # 1) Weekly log returns
    returns = _compute_weekly_returns(df_w)
    feat = pd.concat([feat, returns], axis=1)
    
    # 2) Trend / MA features
    trend_ma = _compute_trend_ma_features(df_w)
    feat = pd.concat([feat, trend_ma], axis=1)
    
    # Momentum windows
    momentum = _compute_momentum_features(df_w)
    feat = pd.concat([feat, momentum], axis=1)
    
    # 3) VIX features
    vix_feat = _compute_vix_features(df_w)
    feat = pd.concat([feat, vix_feat], axis=1)
    
    # 4) Cross-asset (levels + relative momentum)
    spx_ret_4w = feat["SPX_ret_20d"]  # reuse (already computed)
    cross_asset = _compute_cross_asset_features(df_w, spx_ret_4w)
    feat = pd.concat([feat, cross_asset], axis=1)
    
    # 5) Regime arc
    regime_arc = _compute_regime_arc_features(df_w)
    feat = pd.concat([feat, regime_arc], axis=1)
    
    # 6) Streak features
    streaks = _compute_streak_features(feat["SPX_ret_1w"])
    feat = pd.concat([feat, streaks], axis=1)
    
    # 7) Seasonal
    seasonal = _compute_seasonal_features(df_w)
    feat = pd.concat([feat, seasonal], axis=1)
    
    # 8) Regime transition features
    transition = _compute_transition_features(
        df_w, feat["SPX_ret_1w"], feat["TLT_ret_1w"]
    )
    feat = pd.concat([feat, transition], axis=1)
    
    # 9) Macro proxy features
    macro = _compute_macro_proxy_features(
        df_w, feat["SPX_ret_1w"], feat["TLT_ret_1w"]
    )
    feat = pd.concat([feat, macro], axis=1)
    
    # 10) Drop rows with undefined features (rolling warmup), then apply ONE global lag
    before = len(feat)
    feat = feat.dropna(how="any").copy()
    
    FEATURE_COLS = feat.columns.tolist()
    
    # THE IRON RULE: all ML inputs are lagged by exactly +1 week
    feat[FEATURE_COLS] = feat[FEATURE_COLS].shift(1)
    feat = feat.dropna(how="any").copy()
    
    # Align weekly prices to the final feature index (prevents downstream index mistakes)
    prices_w_aligned = prices_w.loc[feat.index, ["SPX", "GOLD", "TLT", "VIX"]].copy()
    
    # Final objects used downstream
    features_df = feat.copy()
    
    # Fatal alignment check
    assert features_df.index.equals(prices_w_aligned.index), \
        "Feature/price index misalignment (fatal)."
    
    # Generate metadata
    FEATURE_META = generate_feature_metadata(features_df)
    
    log(f"✓ Features ready: n_weeks={len(features_df)} | n_features={features_df.shape[1]} | "
        f"{features_df.index.min().date()} -> {features_df.index.max().date()}")
    
    return features_df, prices_w_aligned, FEATURE_COLS, FEATURE_META