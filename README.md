# ML Market Regime Detection

**Course:** Data Science and Advanced Programming 2025
**Author:** Maxime Abdelatif 
**Date:** January 2026

---

## Project Overview

This project implements a machine learning pipeline for detecting market regimes (Bear/Neutral/Bull) in the S&P 500 and using these predictions to inform tactical asset allocation decisions.

### Key Features

- **Regime Classification:** 3-class prediction (Bear=-1, Neutral=0, Bull=1)
- **ML Models:** Logistic Regression, Random Forest, XGBoost, Stacking Ensemble
- **Evaluation:** Macro-F1, stability analysis, bootstrap significance tests
- **Portfolio Backtest:** Regime-based allocation with transaction cost analysis

---

## Quick Start

### 1. Environment Setup

# Create conda environment (recommended, reproducible)
conda env create -f environment.yml || conda env update -f environment.yml --prune

# To reproduce results, you must run via (running outside this environment is unsupported and may change results)
conda run -n ap2025-maxime***

# Verify and run WITHOUT relying on "conda activate" (avoids wrong Python/interpreter issues)
conda run -n ap2025-maxime python test_imports.py
conda run -n ap2025-maxime python main.py

# (Optional) Perfect lockfile mode — guarantees identical conda builds to the author’s machine
```
Use this only if you provide explicit-spec.txt in the repo (generated with: conda list --explicit > explicit-spec.txt)
  conda create -n ap2025-maxime --file explicit-spec.txt
  conda run -n ap2025-maxime python test_imports.py
  conda run -n ap2025-maxime python main.py
```

### 2. Verify Installation

```bash
python test_imports.py
```
Expected output:
```
✓ ALL IMPORTS SUCCESSFUL!
```

### 3. Run Full Pipeline

```bash
python main.py
```
Or with custom config:
```bash
python main.py --config config/config.yaml
```

---

## Project Structure

```
[AP2025] FINAL PROJECT/
├── config/
│   └── config.yaml           # All hyperparameters and settings
├── data/raw/
│   └── RawData...xlsx        # Input price data (SPX, TLT, GOLD, VIX)
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration loading and validation
│   ├── utils.py              # Logging, seeds, reproducibility utilities
│   ├── data.py               # Data loading (Cell 1)
│   ├── features.py           # Feature engineering (Cell 2)
│   ├── labels.py             # Regime labeling (Cell 3)
│   ├── splits.py             # Train/Val/Test splits with embargo (Cell 4B)
│   ├── models.py             # ML model training and CV (Cell 4C/4D)
│   ├── evaluation.py         # Metrics, baselines, diagnostics (Cell 5)
│   ├── portfolio.py          # Portfolio backtest (Cell 6)
│   └── plots.py              # All figure generation (Cell 99)
├── main.py                   # Pipeline orchestrator
├── test_imports.py           # Import verification script
├── test_cells.py             # src.py cells verification script
├── environment.yml           # Conda environment specification
├── README.md                 # Development roadmap and task tracking
├── Proposal.md               # Project proposal
└── results/
    └── runs/
        └── YYYYMMDD_HHMMSS/  # Timestamped run folder
            ├── figures/      # All generated plots
            ├── tables/       # All CSV outputs
            ├── logs/         # Run logs
            └── config_used.yaml  # Config snapshot
```

---

## Output Files

Each run creates a timestamped folder in `results/runs/` containing:

### Tables (CSV)

| File | Description |
|------|-------------|
| `environment_versions.csv` | Package versions for reproducibility audit |
| `embargo_validation.csv` | Train/Val/Test split dates and embargo verification |
| `cv_search_results.csv` | All hyperparameter combinations tested with CV scores |
| `test_results.csv` | Test set metrics for all models and baselines |
| `abstain_policy_comparison.csv` | Abstain policy threshold sweep results |
| `portfolio_summary.csv` | Portfolio performance metrics (CAGR, Sharpe, MaxDD) |
| `tc_sensitivity.csv` | Transaction cost sensitivity analysis |

### Figures (PNG)

| Figure | Description |
|--------|-------------|
| `fig_3.1_class_distributions.png` | Label distribution across methods |
| `fig_3.2_regime_timeline.png` | Regime labels overlaid on price chart |
| `fig_3.3_regime_confusion_matrices.png` | Agreement between labeling methods |
| `fig_3.4_pca_analysis.png` | PCA visualization of feature space |
| `fig_4.1_ml_confusion_matrices.png` | Per-model confusion matrices |
| `fig_4.2_feature_importance.png` | Top features per model |
| `fig_4.3_model_comparison.png` | Metrics comparison bar chart |
| `fig_5.1_rolling_stability.png` | Rolling F1 over test period |
| `fig_5.2_bootstrap_significance.png` | Bootstrap confidence intervals |
| `fig_5.3_threshold_analysis.png` | Confidence threshold sweep |
| `fig_6.1_equity_curves.png` | Portfolio equity curves |
| `fig_6.2_portfolio_metrics.png` | CAGR, Sharpe, MaxDD comparison |
| `fig_6.3_tc_sensitivity.png` | Sharpe vs transaction costs |

---

## Configuration

All parameters are controlled via `config/config.yaml`:

```yaml
model:
  horizon: 1              # Forecast horizon (weeks)
  k: 0.3                  # Volatility-adjusted threshold for regime labels
  embargo: 1              # Purge gap (weeks) between train/val/test
  train_end: "2017-12-31" # End of training period
  val_end: "2020-12-31"   # End of validation period

run:
  seed: 42                # Random seed for reproducibility
  deterministic: true     # Force n_jobs=1 for exact reproducibility
  save_figs: true         # Save figures to disk
  show_figs: false        # Display figures (set false for batch runs)
```

---

## Reproducibility

### Deterministic Mode

When `deterministic: true` in config:
- All models use `n_jobs=1` (single-threaded)
- Results are identical across runs with the same seed
- Slower but guaranteed reproducible

### Package Versions

The pipeline logs all package versions at startup to `environment_versions.csv`:

```
python      : 3.11.14       ✓
numpy       : 2.4.0        ✓
pandas      : 2.3.3        ✓
sklearn     : 1.8.0        ✓
xgboost     : 3.1.2        ✓
matplotlib  : 3.10.8       ✓
seaborn     : 0.13.2       ✓
```

### Run-Stamped Outputs

Each run creates a unique timestamped folder:
```
results/runs/20241229_143000/
```

The exact config used is copied to `config_used.yaml` within the run folder.

---

## Methodology Notes

### Regime Labels (Cell 3)

Labels are defined based on **forward-looking returns**, making this a forecasting task:

- **Bear (-1):** Forward return < -k × σ
- **Neutral (0):** Forward return within ±k × σ  
- **Bull (+1):** Forward return > +k × σ

Where σ is rolling 4-week volatility and k=0.3 by default.

**Note:** This is outcome-defined labeling (forecasting return states), not identification of true macroeconomic regimes.

### Data Leakage Prevention

1. **Feature Lag:** All features use +1 week lag to prevent look-ahead bias
2. **Embargo Purge:** 1-week gap between train→val and val→test boundaries
3. **CV-Pure Feature Selection:** Features selected on first 70% of training only
4. **Hard Asserts:** Pipeline crashes if any embargo violation detected

### Model Pipeline

- **Logistic Regression:** Wrapped in Pipeline with StandardScaler (required for regularization)
- **Random Forest / XGBoost:** Native feature handling, no scaling needed
- **Stacking:** LR + RF base estimators, LR meta-learner

### Evaluation

- **Primary Metric:** Macro-F1 (handles class imbalance)
- **Baselines:** Majority class, stratified random, persistence, momentum rule, trend-vol rule
- **Significance:** Bootstrap 95% CI, permutation test vs random

---

## Troubleshooting

### Import Errors
```bash
# Verify all imports work
python test_imports.py

# If xgboost fails, install separately
pip install xgboost --break-system-packages
```

### Memory Issues
For large datasets or many CV folds:
```yaml
run:
  deterministic: false  # Allow parallel processing
```

### Missing Data
Ensure `data/raw/` contains the Excel file with sheets: SPX, TLT, GOLD, VIX

### Github Repository ###
***Link:*** https://github.com/maxercoding/-AP2025-Final-Project---Maxime-Abdelatif

### Author Contact
**Author:** Maxime Abdelatif  
**Student ID:** 20416384
**Email:** maxime.abdelatif@unil.ch
