"""
Utility functions and helpers.
Maps to: small helper pieces scattered across notebook (mostly Cell 0)

Contains:
- Reproducibility helpers (set_seeds)
- Logging/printing helpers (log, log_section, log_df, log_debug)
- Path helpers (get_project_root, ensure_results_dirs)
- Figure saving helpers (save_fig, show_fig)
"""
from __future__ import annotations

import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PATH HELPERS
# =============================================================================

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Assumes this file is at: <project_root>/src/utils.py
    """
    # src/utils.py -> parents[1] = project root
    return Path(__file__).resolve().parents[1]


def ensure_results_dirs(results_dir: Path) -> Dict[str, Path]:
    """
    Ensure results subdirectories exist (legacy mode - no timestamp).
    
    Creates:
        results/figures/
        results/tables/
        results/logs/
    
    Args:
        results_dir: Base results directory path
        
    Returns:
        Dictionary mapping subdir names to paths
    """
    figures = results_dir / "figures"
    tables = results_dir / "tables"
    logs = results_dir / "logs"
    
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    
    return {"figures": figures, "tables": tables, "logs": logs, "run_dir": results_dir}


def create_run_folder(results_dir: Path, timestamp: str = None) -> Dict[str, Path]:
    """
    Create a timestamped run folder for audit-proof outputs.
    
    Structure:
        results/runs/YYYYMMDD_HHMMSS/
        ├── figures/
        ├── tables/
        ├── logs/
        └── config_used.yaml  (copied by main.py)
    
    Args:
        results_dir: Base results directory (e.g., "results")
        timestamp: Optional timestamp string. If None, uses current time.
        
    Returns:
        Dictionary mapping subdir names to paths:
        {
            "run_dir": Path to run folder,
            "figures": Path to figures folder,
            "tables": Path to tables folder,
            "logs": Path to logs folder,
        }
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir = results_dir / "runs" / timestamp
    figures = run_dir / "figures"
    tables = run_dir / "tables"
    logs = run_dir / "logs"
    
    # Create all directories
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    
    log(f"  📁 Run folder: {run_dir}")
    
    return {
        "run_dir": run_dir,
        "figures": figures,
        "tables": tables,
        "logs": logs,
    }


def copy_config_to_run(config_path: Path, run_dir: Path) -> Path:
    """
    Copy config file to run folder for reproducibility audit.
    
    Args:
        config_path: Path to original config.yaml
        run_dir: Path to run folder
        
    Returns:
        Path to copied config file
    """
    import shutil
    
    dest_path = run_dir / "config_used.yaml"
    shutil.copy2(config_path, dest_path)
    log(f"  📋 Config saved: {dest_path.name}")
    
    return dest_path


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy's random generator
    - PYTHONHASHSEED environment variable
    
    Args:
        seed: Random seed value
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def configure_warnings() -> None:
    """
    Configure warning filters for clean output.
    
    - Shows UserWarnings (important)
    - Hides FutureWarnings (noisy)
    """
    warnings.filterwarnings("default", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# LOGGING HELPERS (Controlled Output)
# =============================================================================

# Global verbose level (set by main.py from config)
_VERBOSE_LEVEL: int = 1
_TABLE_ROWS: int = 5


def set_verbose(level: int, table_rows: int = 5) -> None:
    """Set global verbosity level."""
    global _VERBOSE_LEVEL, _TABLE_ROWS
    _VERBOSE_LEVEL = level
    _TABLE_ROWS = table_rows


def log(msg: str, level: int = 1, logfile: Optional[Path] = None) -> None:
    """
    Print message if verbosity level is sufficient.
    
    Args:
        msg: Message to print
        level: Required verbosity level (1=summary, 2=debug)
        logfile: Optional path to also write to log file
    """
    if _VERBOSE_LEVEL >= level:
        print(msg)
    
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


def log_section(title: str, level: int = 1, logfile: Optional[Path] = None) -> None:
    """
    Print section header if verbosity level is sufficient.
    
    Args:
        title: Section title
        level: Required verbosity level
        logfile: Optional path to also write to log file
    """
    if _VERBOSE_LEVEL >= level:
        header = f"\n{'='*80}\n{title}\n{'='*80}"
        print(header)
        
        if logfile is not None:
            logfile.parent.mkdir(parents=True, exist_ok=True)
            with logfile.open("a", encoding="utf-8") as f:
                f.write(header + "\n")


def log_debug(msg: str, logfile: Optional[Path] = None) -> None:
    """
    Print debug message (only if VERBOSE >= 2).
    
    Args:
        msg: Debug message
        logfile: Optional path to also write to log file
    """
    log(msg, level=2, logfile=logfile)


def log_df(df: pd.DataFrame, n: int = None, level: int = 1) -> None:
    """
    Print DataFrame head if verbosity level is sufficient.
    
    Args:
        df: DataFrame to display
        n: Number of rows (defaults to _TABLE_ROWS)
        level: Required verbosity level
    """
    if _VERBOSE_LEVEL >= level:
        n = n or _TABLE_ROWS
        print(df.head(n).to_string())


# Backward-compatible alias
def print_section(title: str) -> None:
    """Backward-compatible alias for log_section."""
    log_section(title)


# =============================================================================
# LOG FILE MANAGEMENT
# =============================================================================

def make_run_logfile(logs_dir: Path) -> Path:
    """
    Create a timestamped log file path.
    
    Args:
        logs_dir: Directory for log files
        
    Returns:
        Path to new log file (not yet created)
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return logs_dir / f"run_{ts}.txt"


# =============================================================================
# FIGURE HELPERS
# =============================================================================

# Global figure settings (set by main.py from config)
_SAVE_FIGS: bool = True
_SHOW_FIGS: bool = False
_FIGURES_DIR: Optional[Path] = None


def set_figure_config(
    save_figs: bool, 
    show_figs: bool, 
    figures_dir: Optional[Path] = None
) -> None:
    """
    Set global figure configuration.
    
    Args:
        save_figs: Whether to save figures to disk
        show_figs: Whether to display figures inline
        figures_dir: Directory for saving figures
    """
    global _SAVE_FIGS, _SHOW_FIGS, _FIGURES_DIR
    _SAVE_FIGS = save_figs
    _SHOW_FIGS = show_figs
    _FIGURES_DIR = figures_dir


def save_fig(filename: str, figures_dir: Optional[Path] = None) -> None:
    """
    Save current figure to results folder.
    
    Args:
        filename: Name of file (e.g., "fig_3.1_distributions.png")
        figures_dir: Override directory (uses global if None)
    """
    if not _SAVE_FIGS:
        return
    
    dir_to_use = figures_dir or _FIGURES_DIR
    if dir_to_use is None:
        dir_to_use = Path("results/figures")
    
    dir_to_use.mkdir(parents=True, exist_ok=True)
    filepath = dir_to_use / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    log_debug(f"  Saved: {filepath}")


def show_fig() -> None:
    """Show figure if SHOW_FIGS is True, else close it."""
    if _SHOW_FIGS:
        plt.show()
    else:
        plt.close()


# =============================================================================
# ENVIRONMENT VERIFICATION
# =============================================================================

# Minimum required versions (for compatibility checks)
REQUIRED_VERSIONS = {
    "python": "3.11",
    "pandas": "2.0",
    "numpy": "1.24",
    "sklearn": "1.3",
    "xgboost": "2.0",
}

# Expected versions from development environment (for reproducibility)
EXPECTED_VERSIONS = {
    "python": "3.11.14",
    "numpy": "2.4.0",
    "pandas": "2.3.3",
    "sklearn": "1.8.0",
    "xgboost": "3.1.2",
    "matplotlib": "3.10.8",
    "seaborn": "0.13.2",
    "openpyxl": "3.1.5",
    "pyyaml": "6.0.3",
}


def check_environment() -> Dict[str, str]:
    """
    Verify environment meets minimum version requirements.
    
    Returns:
        Dictionary of package names to installed versions
    """
    import sys
    from packaging import version as pkg_version
    import sklearn
    import xgboost as xgb
    
    env_info = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgb.__version__,
    }
    
    log_debug("Environment:")
    for pkg, ver in env_info.items():
        required = REQUIRED_VERSIONS.get(pkg, "0.0")
        ok = pkg_version.parse(ver) >= pkg_version.parse(required)
        log_debug(f"  {pkg}: {ver} {'✓' if ok else '⚠️'}")
    
    return env_info


def get_full_environment_info() -> Dict[str, str]:
    """
    Get comprehensive environment information for reproducibility.
    
    Returns:
        Dictionary of all relevant package versions
    """
    import sys
    import sklearn
    import xgboost as xgb
    import matplotlib
    
    env_info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "matplotlib": matplotlib.__version__,
    }
    
    # Optional packages (may not be installed)
    try:
        import seaborn
        env_info["seaborn"] = seaborn.__version__
    except ImportError:
        env_info["seaborn"] = "NOT_INSTALLED"
    
    try:
        import openpyxl
        env_info["openpyxl"] = openpyxl.__version__
    except ImportError:
        env_info["openpyxl"] = "NOT_INSTALLED"
    
    try:
        import yaml
        env_info["pyyaml"] = yaml.__version__
    except ImportError:
        env_info["pyyaml"] = "NOT_INSTALLED"
    
    return env_info


def log_environment_versions(
    logfile: Optional[Path] = None,
    save_csv: Optional[Path] = None
) -> Dict[str, str]:
    """
    Log all package versions for reproducibility.
    
    This function captures and logs the complete environment state,
    which is critical for reproducing results.
    
    Args:
        logfile: Optional path to write versions to log file
        save_csv: Optional path to save versions as CSV
        
    Returns:
        Dictionary of package names to versions
    """
    env_info = get_full_environment_info()
    
    # Build version report
    lines = [
        "=" * 60,
        "ENVIRONMENT VERSIONS (for reproducibility)",
        "=" * 60,
    ]
    
    for pkg, ver in env_info.items():
        expected = EXPECTED_VERSIONS.get(pkg, "")
        if expected and ver != expected:
            status = f"⚠️  (expected {expected})"
        else:
            status = "✓"
        lines.append(f"  {pkg:12s}: {ver:12s} {status}")
    
    lines.append("=" * 60)
    
    # Print to console
    report = "\n".join(lines)
    log(report)
    
    # Write to log file if specified
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a", encoding="utf-8") as f:
            f.write(report + "\n")
    
    # Save as CSV if specified
    if save_csv is not None:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([
            {"package": pkg, "version": ver, "expected": EXPECTED_VERSIONS.get(pkg, "")}
            for pkg, ver in env_info.items()
        ])
        df.to_csv(save_csv, index=False)
        log(f"  Versions saved to: {save_csv}")
    
    return env_info


def verify_reproducibility(env_info: Dict[str, str] = None) -> bool:
    """
    Check if current environment matches expected versions.
    
    Args:
        env_info: Pre-computed environment info (computed if None)
        
    Returns:
        True if all versions match expected, False otherwise
    """
    if env_info is None:
        env_info = get_full_environment_info()
    
    mismatches = []
    for pkg, expected in EXPECTED_VERSIONS.items():
        actual = env_info.get(pkg, "MISSING")
        if actual != expected:
            mismatches.append(f"{pkg}: {actual} (expected {expected})")
    
    if mismatches:
        log("⚠️  Version mismatches detected:")
        for m in mismatches:
            log(f"    {m}")
        log("  Results may differ from original run.")
        return False
    
    log("✓ All package versions match expected values")
    return True


# =============================================================================
# DATA VALIDATION HELPERS
# =============================================================================

def assert_no_future_leakage(
    feature_index: pd.DatetimeIndex,
    label_index: pd.DatetimeIndex
) -> None:
    """
    Assert that feature and label indices are aligned.
    
    Args:
        feature_index: Index of feature DataFrame
        label_index: Index of label Series/DataFrame
        
    Raises:
        AssertionError: If indices don't match
    """
    assert feature_index.equals(label_index), \
        "Feature/Label index mismatch (potential leakage)."


def assert_no_nan(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Assert DataFrame has no NaN values.
    
    Args:
        df: DataFrame to check
        name: Name for error message
        
    Raises:
        AssertionError: If NaN values found
    """
    nan_cols = df.columns[df.isna().any()].tolist()
    assert len(nan_cols) == 0, f"NaN values in {name}: {nan_cols}"


def assert_positive_prices(df: pd.DataFrame, columns: list) -> None:
    """
    Assert price columns are all positive.
    
    Args:
        df: DataFrame with price columns
        columns: List of column names to check
        
    Raises:
        AssertionError: If non-positive values found
    """
    for col in columns:
        assert (df[col] > 0).all(), f"Non-positive values in {col}"