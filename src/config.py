"""
Configuration management.
Maps to: Cell 0 (config/parameters block)

Loads config/config.yaml and provides typed Config dataclass.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# =============================================================================
# DATACLASSES (mirrors notebook Cell 0 Config)
# =============================================================================

@dataclass(frozen=True)
class Paths:
    """File paths configuration."""
    raw_excel: str
    sheet_name: str
    results_dir: str


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration."""
    seed: int
    deterministic: bool      # NEW: Forces n_jobs=1 for reproducibility
    save_figs: bool
    show_figs: bool


@dataclass(frozen=True)
class ProjectConfig:
    """Project metadata."""
    name: str
    version: str = "v1.0-weekly"


@dataclass(frozen=True)
class ModelConfig:
    """ML model parameters."""
    # Time splits
    train_end: str = "2017-12-29"
    val_end: str = "2020-12-25"
    
    # Weekly framing
    freq: str = "W-FRI"
    horizon: int = 1          # H: forecast horizon in weeks
    embargo: int = 1          # Gap between segments (must equal H)
    
    # Labeling parameters
    vol_win: int = 52         # Volatility window in weeks
    k: float = 0.3            # Volatility multiplier for regime bands
    persist_frac: float = 0.35
    use_persistence: bool = False
    
    # Output control
    verbose: int = 1          # 0=silent, 1=summary, 2=debug
    table_rows: int = 5       # Max rows to display in tables


@dataclass(frozen=True)
class Config:
    """
    Master configuration object.
    
    Combines all configuration sections into a single immutable object.
    Mirrors the notebook's CFG dataclass from Cell 0.
    """
    project: ProjectConfig
    paths: Paths
    run: RunConfig
    model: ModelConfig


# =============================================================================
# YAML LOADER
# =============================================================================

def load_config(config_path: Path) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Populated Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If required keys are missing
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with config_path.open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    
    # Build Config from YAML with defaults for optional fields
    project_data = d.get("project", {})
    paths_data = d.get("paths", {})
    run_data = d.get("run", {})
    model_data = d.get("model", {})
    
    return Config(
        project=ProjectConfig(
            name=project_data.get("name", "ML Market Regime Detection"),
            version=project_data.get("version", "v1.0-weekly"),
        ),
        paths=Paths(
            raw_excel=paths_data["raw_excel"],
            sheet_name=paths_data.get("sheet_name", None),
            results_dir=paths_data.get("results_dir", "results"),
        ),
        run=RunConfig(
            seed=int(run_data.get("seed", 42)),
            deterministic=bool(run_data.get("deterministic", True)),  # Default True for safety
            save_figs=bool(run_data.get("save_figs", True)),
            show_figs=bool(run_data.get("show_figs", False)),
        ),
        model=ModelConfig(
            train_end=model_data.get("train_end", "2017-12-29"),
            val_end=model_data.get("val_end", "2020-12-25"),
            freq=model_data.get("freq", "W-FRI"),
            horizon=int(model_data.get("horizon", 1)),
            embargo=int(model_data.get("embargo", 1)),
            vol_win=int(model_data.get("vol_win", 52)),
            k=float(model_data.get("k", 0.3)),
            persist_frac=float(model_data.get("persist_frac", 0.35)),
            use_persistence=bool(model_data.get("use_persistence", False)),
            verbose=int(model_data.get("verbose", 1)),
            table_rows=int(model_data.get("table_rows", 5)),
        ),
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config(cfg: Config) -> None:
    """
    Validate configuration values.
    
    Args:
        cfg: Config object to validate
        
    Raises:
        AssertionError: If validation fails
    """
    # Embargo must equal horizon (purge/overlap leakage control)
    assert cfg.model.embargo == cfg.model.horizon, \
        f"embargo ({cfg.model.embargo}) must equal horizon ({cfg.model.horizon})"
    
    # Volatility window must be positive
    assert cfg.model.vol_win > 0, "vol_win must be positive"
    
    # K must be positive
    assert cfg.model.k > 0, "k must be positive"
    
    # Horizon must be positive
    assert cfg.model.horizon >= 1, "horizon must be >= 1"
    
    # Verbose must be 0, 1, or 2
    assert cfg.model.verbose in [0, 1, 2], "verbose must be 0, 1, or 2"
    