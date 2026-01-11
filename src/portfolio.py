"""
Portfolio simulation and economic evaluation.
Maps to: Cell 6 (economic evaluation / backtest)

Contains:
- Regime-to-allocation mapping
- Portfolio return computation
- Transaction costs
- Performance metrics (Sharpe, CAGR, MaxDD, etc.)
- Benchmark comparison

Outputs: portfolio_results, equity_curves, performance_summary
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils import log, log_section, log_debug

# =============================================================================
# ALLOCATION STRATEGIES
# =============================================================================

# Default regime-to-allocation mapping
DEFAULT_ALLOCATIONS = {
    1:  {"SPX": 1.0, "TLT": 0.0, "GOLD": 0.0, "Cash": 0.0},   # Bull: 100% equity
    0:  {"SPX": 0.4, "TLT": 0.3, "GOLD": 0.1, "Cash": 0.2},   # Neutral: diversified
    -1: {"SPX": 0.0, "TLT": 0.6, "GOLD": 0.2, "Cash": 0.2},   # Bear: defensive
}

# Conservative variant
CONSERVATIVE_ALLOCATIONS = {
    1:  {"SPX": 0.8, "TLT": 0.1, "GOLD": 0.0, "Cash": 0.1},
    0:  {"SPX": 0.3, "TLT": 0.3, "GOLD": 0.2, "Cash": 0.2},
    -1: {"SPX": 0.0, "TLT": 0.5, "GOLD": 0.3, "Cash": 0.2},
}


def get_allocation(regime: int, allocation_map: Dict = None) -> Dict[str, float]:
    """Get allocation weights for a given regime."""
    if allocation_map is None:
        allocation_map = DEFAULT_ALLOCATIONS
    return allocation_map.get(regime, DEFAULT_ALLOCATIONS[0])


# =============================================================================
# PORTFOLIO COMPUTATION
# =============================================================================

def compute_asset_returns(
    prices_w: pd.DataFrame,
    assets: List[str] = None
) -> pd.DataFrame:
    """
    Compute weekly SIMPLE returns for assets.
    
    IMPORTANT: Uses simple returns (not log returns) for portfolio arithmetic.
    Simple returns compound correctly with (1 + r).cumprod() and allow
    proper portfolio weighting: R_portfolio = sum(w_i * r_i)
    
    Args:
        prices_w: Weekly prices DataFrame
        assets: List of asset columns (default: SPX, TLT, GOLD)
        
    Returns:
        DataFrame of weekly simple returns
    """
    if assets is None:
        assets = ["SPX", "TLT", "GOLD"]
    
    returns = pd.DataFrame(index=prices_w.index)
    for asset in assets:
        if asset in prices_w.columns:
            # SIMPLE returns: (P_t - P_{t-1}) / P_{t-1} = P_t / P_{t-1} - 1
            returns[asset] = prices_w[asset] / prices_w[asset].shift(1) - 1
        elif asset == "Cash":
            # Cash earns zero return (simplification)
            returns["Cash"] = 0.0
    
    return returns.dropna()


def compute_portfolio_returns(
    asset_returns: pd.DataFrame,
    regimes: pd.Series,
    allocation_map: Dict = None,
    transaction_cost_bps: float = 10.0
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Compute portfolio returns based on regime signals.
    
    The regime at time t determines allocation for period t → t+1.
    
    Args:
        asset_returns: DataFrame of asset returns (t → t+1)
        regimes: Series of regime predictions at time t
        allocation_map: Regime-to-allocation mapping
        transaction_cost_bps: Transaction cost in basis points (per trade)
        
    Returns:
        Tuple of (portfolio_returns, weights_df, turnover)
    """
    if allocation_map is None:
        allocation_map = DEFAULT_ALLOCATIONS
    
    # Align regimes and returns
    common_idx = asset_returns.index.intersection(regimes.index)
    asset_returns = asset_returns.loc[common_idx]
    regimes = regimes.loc[common_idx]
    
    # Shift regimes by 1 to get allocation for period t → t+1
    # (regime predicted at t-1 determines allocation at t)
    regimes_shifted = regimes.shift(1)
    
    # Build allocation weights
    assets = list(asset_returns.columns)
    weights_df = pd.DataFrame(index=common_idx, columns=assets, dtype=float)
    
    for t in common_idx:
        regime = regimes_shifted.loc[t]
        if pd.isna(regime):
            # First period: use neutral allocation
            alloc = allocation_map[0]
        else:
            alloc = allocation_map.get(int(regime), allocation_map[0])
        
        for asset in assets:
            weights_df.loc[t, asset] = alloc.get(asset, 0.0)
    
    # Compute turnover (for transaction costs)
    weight_changes = weights_df.diff().abs()
    turnover = weight_changes.sum(axis=1) / 2  # Sum of absolute weight changes / 2
    
    # Compute gross portfolio return
    portfolio_gross = (asset_returns * weights_df).sum(axis=1)
    
    # Apply transaction costs
    tc_cost = turnover * (transaction_cost_bps / 10000)
    portfolio_net = portfolio_gross - tc_cost
    
    return portfolio_net, weights_df, turnover


def compute_equity_curve(returns: pd.Series, initial_value: float = 100.0) -> pd.Series:
    """Compute cumulative equity curve from returns."""
    cumulative = (1 + returns).cumprod()
    return cumulative * initial_value


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in weeks
    calmar_ratio: float
    win_rate: float
    avg_turnover: float
    n_periods: int


def compute_performance_metrics(
    returns: pd.Series,
    turnover: pd.Series = None,
    periods_per_year: int = 52,
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """
    Compute comprehensive performance metrics.
    
    Args:
        returns: Series of periodic returns
        turnover: Series of turnover values
        periods_per_year: Number of periods per year (52 for weekly)
        risk_free_rate: Annual risk-free rate
        
    Returns:
        PerformanceMetrics dataclass
    """
    n = len(returns)
    
    # Total return
    total_return = (1 + returns).prod() - 1
    
    # CAGR
    years = n / periods_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    excess_return = returns.mean() - (risk_free_rate / periods_per_year)
    sharpe_ratio = (excess_return / returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else 0.0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
    sortino_ratio = (excess_return / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Maximum drawdown duration
    in_drawdown = drawdown < 0
    dd_groups = (~in_drawdown).cumsum()
    dd_durations = in_drawdown.groupby(dd_groups).sum()
    max_drawdown_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0
    
    # Calmar ratio
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    # Average turnover
    avg_turnover = turnover.mean() if turnover is not None else 0.0
    
    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        calmar_ratio=calmar_ratio,
        win_rate=win_rate,
        avg_turnover=avg_turnover,
        n_periods=n
    )


def metrics_to_dict(metrics: PerformanceMetrics) -> Dict[str, float]:
    """Convert PerformanceMetrics to dictionary."""
    return {
        "total_return": metrics.total_return,
        "cagr": metrics.cagr,
        "volatility": metrics.volatility,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "max_drawdown": metrics.max_drawdown,
        "max_drawdown_duration": metrics.max_drawdown_duration,
        "calmar_ratio": metrics.calmar_ratio,
        "win_rate": metrics.win_rate,
        "avg_turnover": metrics.avg_turnover,
        "n_periods": metrics.n_periods
    }


# =============================================================================
# BENCHMARK STRATEGIES
# =============================================================================

def compute_buy_hold_benchmark(
    asset_returns: pd.DataFrame,
    asset: str = "SPX"
) -> pd.Series:
    """Compute buy-and-hold benchmark returns."""
    if asset in asset_returns.columns:
        return asset_returns[asset]
    return pd.Series(0.0, index=asset_returns.index)


def compute_60_40_benchmark(
    asset_returns: pd.DataFrame
) -> pd.Series:
    """Compute 60/40 stock/bond benchmark returns."""
    spx_ret = asset_returns.get("SPX", pd.Series(0.0, index=asset_returns.index))
    tlt_ret = asset_returns.get("TLT", pd.Series(0.0, index=asset_returns.index))
    return 0.6 * spx_ret + 0.4 * tlt_ret


def compute_risk_parity_benchmark(
    asset_returns: pd.DataFrame,
    lookback: int = 52
) -> pd.Series:
    """
    Compute simplified risk parity benchmark.
    
    Allocates inversely proportional to trailing volatility.
    
    Args:
        asset_returns: DataFrame of asset returns
        lookback: Lookback window for volatility estimation
        
    Returns:
        Series of portfolio returns
    """
    assets = ["SPX", "TLT", "GOLD"]
    available = [a for a in assets if a in asset_returns.columns]
    
    if len(available) < 2:
        return asset_returns.get("SPX", pd.Series(0.0, index=asset_returns.index))
    
    # Rolling volatility
    vol = asset_returns[available].rolling(lookback, min_periods=12).std()
    
    # Inverse vol weights (normalized)
    inv_vol = 1 / vol.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(1/len(available))
    
    # Portfolio return
    port_ret = (asset_returns[available] * weights).sum(axis=1)
    
    return port_ret


# =============================================================================
# TRANSACTION COST SENSITIVITY
# =============================================================================

def tc_sensitivity_analysis(
    asset_returns: pd.DataFrame,
    pred_series: pd.Series,
    allocation_map: Dict,
    tc_levels: List[float] = None
) -> pd.DataFrame:
    """
    Analyze portfolio performance across different transaction cost levels.
    
    Args:
        asset_returns: Asset returns DataFrame
        pred_series: Model predictions as Series
        allocation_map: Regime-to-allocation mapping
        tc_levels: List of TC levels in bps to test
        
    Returns:
        DataFrame with performance metrics at each TC level
    """
    if tc_levels is None:
        tc_levels = [0, 5, 10, 15, 20, 25, 50]
    
    results = []
    
    for tc_bps in tc_levels:
        port_returns, weights_df, turnover = compute_portfolio_returns(
            asset_returns, pred_series, allocation_map, tc_bps
        )
        
        metrics = compute_performance_metrics(port_returns, turnover)
        
        results.append({
            "tc_bps": tc_bps,
            "cagr": metrics.cagr,
            "sharpe_ratio": metrics.sharpe_ratio,
            "volatility": metrics.volatility,
            "max_drawdown": metrics.max_drawdown,
            "total_return": metrics.total_return,
            "avg_turnover": metrics.avg_turnover,
        })
    
    df = pd.DataFrame(results)
    return df


def run_tc_sensitivity_all_models(
    asset_returns: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    y_test: pd.Series,
    allocation_map: Dict,
    tc_levels: List[float] = None
) -> Dict[str, pd.DataFrame]:
    """
    Run TC sensitivity for all models.
    
    Args:
        asset_returns: Asset returns DataFrame
        predictions: Dict of model_name -> predictions
        y_test: Test labels (for index alignment)
        allocation_map: Regime-to-allocation mapping
        tc_levels: List of TC levels in bps
        
    Returns:
        Dict mapping model_name -> TC sensitivity DataFrame
    """
    if tc_levels is None:
        tc_levels = [0, 5, 10, 15, 20, 25, 50]
    
    results = {}
    
    for model_name, preds in predictions.items():
        pred_series = pd.Series(preds, index=y_test.index[:len(preds)])
        
        tc_df = tc_sensitivity_analysis(
            asset_returns, pred_series, allocation_map, tc_levels
        )
        tc_df["model"] = model_name
        results[model_name] = tc_df
    
    return results


def create_tc_sensitivity_summary(
    tc_results: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Create summary table of TC sensitivity across all models.
    
    Args:
        tc_results: Dict from run_tc_sensitivity_all_models
        
    Returns:
        Wide-format DataFrame with TC levels as rows, models as columns
    """
    all_dfs = []
    for model_name, df in tc_results.items():
        df_pivot = df[["tc_bps", "sharpe_ratio"]].copy()
        df_pivot = df_pivot.rename(columns={"sharpe_ratio": model_name})
        all_dfs.append(df_pivot.set_index("tc_bps"))
    
    summary = pd.concat(all_dfs, axis=1)
    return summary


# =============================================================================
# MAIN CELL FUNCTION
# =============================================================================

def cell6_portfolio_backtest(
    prices_w: pd.DataFrame,
    y_test: pd.Series,
    predictions: Dict[str, np.ndarray],
    model_names: List[str] = None,
    allocation_map: Dict = None,
    transaction_cost_bps: float = 10.0
) -> Dict:
    """
    Cell 6: Portfolio backtest on test period.
    
    Args:
        prices_w: Weekly prices DataFrame
        y_test: True regime labels for test period
        predictions: Dict of model_name -> predictions array
        model_names: List of models to evaluate (default: all in predictions)
        allocation_map: Regime-to-allocation mapping
        transaction_cost_bps: Transaction cost in basis points
        
    Returns:
        Dictionary with all portfolio results:
        {
            "asset_returns": pd.DataFrame,
            "equity_curves": Dict[str, pd.Series],
            "performance": Dict[str, PerformanceMetrics],
            "weights": Dict[str, pd.DataFrame],
            "summary_df": pd.DataFrame,
        }
    """
    log_section("CELL 6: PORTFOLIO BACKTEST")
    
    if allocation_map is None:
        allocation_map = DEFAULT_ALLOCATIONS
    
    if model_names is None:
        model_names = list(predictions.keys())
    
    # Compute asset returns
    asset_returns = compute_asset_returns(prices_w, ["SPX", "TLT", "GOLD"])
    asset_returns["Cash"] = 0.0
    
    # Align to test period
    test_idx = y_test.index
    asset_returns = asset_returns.loc[asset_returns.index.isin(test_idx)]
    
    log(f"  Test period: {asset_returns.index.min().date()} → {asset_returns.index.max().date()}")
    log(f"  Test weeks: {len(asset_returns)}")
    
    # Results containers
    equity_curves = {}
    performance = {}
    weights_dict = {}
    
    # Backtest each model
    for model_name in model_names:
        if model_name not in predictions:
            continue
        
        preds = predictions[model_name]
        pred_series = pd.Series(preds, index=test_idx[:len(preds)])
        
        port_returns, weights_df, turnover = compute_portfolio_returns(
            asset_returns, pred_series, allocation_map, transaction_cost_bps
        )
        
        equity = compute_equity_curve(port_returns)
        metrics = compute_performance_metrics(port_returns, turnover)
        
        equity_curves[model_name] = equity
        performance[model_name] = metrics
        weights_dict[model_name] = weights_df
        
        log(f"\n  {model_name}:")
        log(f"    CAGR: {metrics.cagr:.2%}")
        log(f"    Sharpe: {metrics.sharpe_ratio:.2f}")
        log(f"    Max DD: {metrics.max_drawdown:.2%}")
    
    # Perfect foresight (oracle)
    oracle_series = y_test.copy()
    port_returns_oracle, weights_oracle, turnover_oracle = compute_portfolio_returns(
        asset_returns, oracle_series, allocation_map, transaction_cost_bps
    )
    equity_curves["Oracle"] = compute_equity_curve(port_returns_oracle)
    performance["Oracle"] = compute_performance_metrics(port_returns_oracle, turnover_oracle)
    weights_dict["Oracle"] = weights_oracle
    
    log(f"\n  Oracle (Perfect Foresight):")
    log(f"    CAGR: {performance['Oracle'].cagr:.2%}")
    log(f"    Sharpe: {performance['Oracle'].sharpe_ratio:.2f}")
    
    # Buy & Hold SPX
    spx_returns = compute_buy_hold_benchmark(asset_returns, "SPX")
    equity_curves["BuyHold_SPX"] = compute_equity_curve(spx_returns)
    performance["BuyHold_SPX"] = compute_performance_metrics(spx_returns, None)
    
    log(f"\n  Buy & Hold SPX:")
    log(f"    CAGR: {performance['BuyHold_SPX'].cagr:.2%}")
    log(f"    Sharpe: {performance['BuyHold_SPX'].sharpe_ratio:.2f}")
    
    # 60/40 Benchmark
    benchmark_60_40 = compute_60_40_benchmark(asset_returns)
    equity_curves["60_40"] = compute_equity_curve(benchmark_60_40)
    performance["60_40"] = compute_performance_metrics(benchmark_60_40, None)
    
    log(f"\n  60/40 Benchmark:")
    log(f"    CAGR: {performance['60_40'].cagr:.2%}")
    log(f"    Sharpe: {performance['60_40'].sharpe_ratio:.2f}")
    
    # Risk Parity Benchmark
    risk_parity_returns = compute_risk_parity_benchmark(asset_returns)
    equity_curves["RiskParity"] = compute_equity_curve(risk_parity_returns)
    performance["RiskParity"] = compute_performance_metrics(risk_parity_returns, None)
    
    log(f"\n  Risk Parity:")
    log(f"    CAGR: {performance['RiskParity'].cagr:.2%}")
    log(f"    Sharpe: {performance['RiskParity'].sharpe_ratio:.2f}")
    
    # Transaction Cost Sensitivity Analysis
    log("\n--- Transaction Cost Sensitivity ---")
    tc_results = run_tc_sensitivity_all_models(
        asset_returns=asset_returns,
        predictions=predictions,
        y_test=y_test,
        allocation_map=allocation_map,
        tc_levels=[0, 5, 10, 20, 50]
    )
    
    # Show TC sensitivity for best model (first in list)
    if model_names and model_names[0] in tc_results:
        best_model_tc = tc_results[model_names[0]]
        log(f"\n  TC Sensitivity ({model_names[0]}):")
        log(f"    TC=0bp:  Sharpe={best_model_tc[best_model_tc['tc_bps']==0]['sharpe_ratio'].values[0]:.2f}")
        log(f"    TC=10bp: Sharpe={best_model_tc[best_model_tc['tc_bps']==10]['sharpe_ratio'].values[0]:.2f}")
        log(f"    TC=50bp: Sharpe={best_model_tc[best_model_tc['tc_bps']==50]['sharpe_ratio'].values[0]:.2f}")
    
    # Summary DataFrame
    summary_rows = []
    for name, metrics in performance.items():
        row = {"Strategy": name}
        row.update(metrics_to_dict(metrics))
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.set_index("Strategy")
    
    # Sort by Sharpe ratio
    summary_df = summary_df.sort_values("sharpe_ratio", ascending=False)
    
    log("\n" + "="*60)
    log("PORTFOLIO PERFORMANCE SUMMARY")
    log("="*60)
    
    display_cols = ["cagr", "sharpe_ratio", "max_drawdown", "volatility"]
    log(summary_df[display_cols].round(4).to_string())
    
    return {
        "asset_returns": asset_returns,
        "equity_curves": equity_curves,
        "performance": performance,
        "weights": weights_dict,
        "summary_df": summary_df,
        "tc_sensitivity": tc_results,  # NEW: TC sensitivity results
    }


# =============================================================================
# REGIME TRANSITION ANALYSIS
# =============================================================================

def analyze_regime_performance(
    returns: pd.Series,
    regimes: pd.Series
) -> pd.DataFrame:
    """
    Analyze portfolio performance by regime.
    
    Args:
        returns: Portfolio returns
        regimes: Regime labels
        
    Returns:
        DataFrame with per-regime statistics
    """
    common_idx = returns.index.intersection(regimes.index)
    returns = returns.loc[common_idx]
    regimes = regimes.loc[common_idx]
    
    results = []
    for regime in [-1, 0, 1]:
        mask = regimes == regime
        if mask.sum() > 0:
            regime_returns = returns[mask]
            results.append({
                "regime": {-1: "Bear", 0: "Neutral", 1: "Bull"}[regime],
                "n_weeks": mask.sum(),
                "mean_return": regime_returns.mean(),
                "std_return": regime_returns.std(),
                "total_return": (1 + regime_returns).prod() - 1,
                "win_rate": (regime_returns > 0).mean()
            })
    
    return pd.DataFrame(results)