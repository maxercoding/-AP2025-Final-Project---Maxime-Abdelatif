"""
Data loading, cleaning, and resampling.
Maps to: Cell 1 (data loading + cleaning + resampling)

Reads Excel → renames columns → sets Date index → resamples to weekly.
Outputs: df_daily, df_w, prices_w
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils import log, log_section, log_debug, assert_positive_prices

# =============================================================================
# COLUMN MAPPING
# =============================================================================

RENAME_MAP = {
    "Date": "Date",
    "S&P 500 COMPOSITE - PRICE INDEX": "SPX",
    "Gold, USD FX Comp. U$/Troy Oz": "GOLD",
    "ISHARES 20+ YEAR TREASURY BOND ETF": "TLT",
    "CBOE Volatility S&P 500 Index (^VIX) - Index Value": "VIX",
}

PRICE_COLS = ["SPX", "GOLD", "TLT", "VIX"]


# =============================================================================
# RAW DATA LOADING
# =============================================================================

def load_raw_excel(excel_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all sheets from the raw Excel file.
    
    Args:
        excel_path: Path to Excel file
        
    Returns:
        Dictionary mapping sheet names to DataFrames
        
    Raises:
        FileNotFoundError: If Excel file doesn't exist
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Raw Excel not found at: {excel_path}")
    
    xls = pd.ExcelFile(excel_path)
    sheets = {}
    for name in xls.sheet_names:
        sheets[name] = pd.read_excel(excel_path, sheet_name=name)
    
    return sheets


def _find_data_sheet(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Find the sheet containing price data.
    
    Looks for a sheet with the expected column names.
    """
    if len(sheets) == 1:
        return list(sheets.values())[0]
    
    # Try to find sheet with expected columns
    for name, df in sheets.items():
        if "Date" in df.columns and any(col in df.columns for col in RENAME_MAP.keys()):
            return df
    
    # Fallback to first sheet
    return list(sheets.values())[0]


# =============================================================================
# DATA PROCESSING
# =============================================================================

def _rename_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns from raw Excel to canonical names.
    
    Args:
        df_raw: Raw DataFrame from Excel
        
    Returns:
        DataFrame with renamed columns [Date, SPX, GOLD, TLT, VIX]
        
    Raises:
        ValueError: If expected columns are missing
    """
    missing = [c for c in RENAME_MAP if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing expected columns in Excel: {missing}")
    
    df = df_raw.rename(columns=RENAME_MAP)[["Date"] + PRICE_COLS].copy()
    return df


def _prepare_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert to daily time series with Date as index.
    
    Args:
        df: DataFrame with 'Date' column
        
    Returns:
        DataFrame indexed by Date, sorted chronologically
        
    Raises:
        ValueError: If dates are unparsable or duplicated
    """
    df = df.copy()
    
    # Parse dates strictly
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Unparsable dates found in 'Date' column.")
    
    df = df.set_index("Date").sort_index()
    
    # Check for duplicate dates (do NOT use drop_duplicates - can drop valid data)
    if df.index.duplicated().any():
        raise ValueError("Duplicate dates found in raw index.")
    
    return df


def _resample_to_weekly(df_daily: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    """
    Resample daily data to weekly frequency.
    
    Uses last value of each week (Friday close convention).
    
    Args:
        df_daily: Daily DataFrame indexed by Date
        freq: Resampling frequency (default: W-FRI for Friday)
        
    Returns:
        Weekly DataFrame with last values of each week
    """
    df_w = df_daily.resample(freq).last().dropna(how="any").copy()
    return df_w


# =============================================================================
# VALIDATION
# =============================================================================

def _validate_weekly_data(
    df_w: pd.DataFrame,
    vol_win: int = 52,
    horizon: int = 1,
    min_weeks: int = 100
) -> None:
    """
    Run integrity checks on weekly data.
    
    Args:
        df_w: Weekly DataFrame
        vol_win: Volatility window in weeks
        horizon: Forecast horizon in weeks
        min_weeks: Minimum required weeks of history
        
    Raises:
        AssertionError: If any validation fails
    """
    # Check positive prices
    assert_positive_prices(df_w, ["SPX", "TLT", "GOLD"])
    
    # Check positive VIX
    assert (df_w["VIX"] > 0).all(), "Non-positive VIX detected."
    
    # Check minimum history
    assert len(df_w) >= min_weeks, \
        f"Insufficient weekly history: {len(df_w)} weeks (need {min_weeks})."
    
    # Check enough data after warmups
    min_needed = max(40, vol_win) + horizon + 5
    assert len(df_w) >= min_needed, \
        f"Not enough weekly data after warmups: need >= {min_needed}, have {len(df_w)}"


# =============================================================================
# MAIN CELL FUNCTION
# =============================================================================

def cell1_load_data(
    excel_path: Path,
    sheet_name: str = None,
    freq: str = "W-FRI",
    vol_win: int = 52,
    horizon: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cell 1: Load and prepare weekly price data.
    
    Full pipeline: load Excel → clean → resample to weekly → validate.
    
    Args:
        excel_path: Path to raw Excel file
        sheet_name: Explicit sheet name to load (if None, auto-detect)
        freq: Weekly frequency (default: W-FRI)
        vol_win: Volatility window for validation
        horizon: Forecast horizon for validation
        
    Returns:
        Tuple of (df_daily, df_w, prices_w) where:
            df_daily: Daily price data (for reference)
            df_w: Weekly price data (all columns)
            prices_w: Weekly prices aligned for downstream use
            
    Raises:
        FileNotFoundError: If Excel file doesn't exist
        ValueError: If data validation fails
    """
    log_section("CELL 1: LOADING & WEEKLY RESAMPLING")
    
    # 1) Load raw Excel
    log_debug(f"Loading Excel from: {excel_path}")
    
    if sheet_name:
        # Explicit sheet name from config
        log_debug(f"  Using explicit sheet: {sheet_name}")
        df_raw = pd.read_excel(excel_path, sheet_name=sheet_name)
    else:
        # Fallback to auto-detect (legacy behavior)
        sheets = load_raw_excel(excel_path)
        log_debug(f"  Found {len(sheets)} sheet(s): {list(sheets.keys())}")
        df_raw = _find_data_sheet(sheets)
    
    log_debug(f"  Loaded {len(df_raw)} rows")
    
    # 2) Rename columns
    df_renamed = _rename_columns(df_raw)
    
    # 3) Set Date index
    df_daily = _prepare_daily_index(df_renamed)
    log_debug(f"  Daily data: {len(df_daily)} rows")
    
    # 4) Resample to weekly
    df_w = _resample_to_weekly(df_daily, freq=freq)
    
    # 5) Validate
    _validate_weekly_data(df_w, vol_win=vol_win, horizon=horizon)
    
    # 6) Create prices_w (aligned copy for downstream)
    prices_w = df_w[PRICE_COLS].copy()
    
    log(f"✓ Weekly dataset ready: {len(df_w)} rows | "
        f"{df_w.index.min().date()} -> {df_w.index.max().date()}")
    
    return df_daily, df_w, prices_w


# =============================================================================
# SANITY CHECK OUTPUT
# =============================================================================

def save_raw_sheets_sanity(
    sheets: Dict[str, pd.DataFrame],
    output_path: Path
) -> None:
    """
    Save sanity check table with sheet names and shapes.
    
    Args:
        sheets: Dictionary of sheet names to DataFrames
        output_path: Path for output CSV
    """
    sanity = []
    for k, df in sheets.items():
        sanity.append({
            "sheet": k,
            "rows": df.shape[0],
            "cols": df.shape[1]
        })
    
    sanity_df = pd.DataFrame(sanity).sort_values("sheet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanity_df.to_csv(output_path, index=False)
    log_debug(f"  Saved sanity table: {output_path}")