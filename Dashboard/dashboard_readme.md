# Video Support Dashboard

Streamlit-based presentation slides for video recording.

## How to Run

```bash
# From project root:
python Dashboard/run_dashboard.py

# Or directly with streamlit:
streamlit run Dashboard/dashboard.py
```

Opens at `http://localhost:8501`

## Requirements

```bash
pip install streamlit pandas pyyaml
```

## Structure

```
Dashboard/
├── run_dashboard.py      # Entry point
├── dashboard.py          # Main Streamlit app
└── dashboard_readme.md   # This file
```

## Expected Artifacts

The dashboard auto-detects runs from `results/runs/<timestamp>/`:

```
results/runs/20251230_131036/
├── config_used.yaml
├── figures/
│   ├── fig_3.1_class_distributions.png
│   ├── fig_3.2_regime_timeline.png
│   ├── fig_3.3_confusion_matrices.png
│   ├── fig_3.4_pca_analysis.png
│   ├── fig_4.1_confusion_matrices.png
│   ├── fig_4.2_feature_importance.png
│   ├── fig_4.3_model_comparison.png
│   ├── fig_5.1_rolling_stability.png
│   ├── fig_5.2_bootstrap_significance.png
│   ├── fig_5.3_threshold_analysis.png
│   ├── fig_6.1_equity_curves.png
│   ├── fig_6.2_portfolio_metrics.png
│   └── fig_6.3_tc_sensitivity.png
├── tables/
│   ├── test_results.csv
│   ├── embargo_validation.csv
│   ├── abstain_policy_comparison.csv
│   ├── cv_search_results.csv
│   ├── portfolio_summary.csv
│   ├── tc_sensitivity.csv
│   └── environment_versions.csv
└── logs/
    └── run.log
```

## Slide Order

1. **Title & Thesis** - Project overview
2. **Motivation** - Regime timeline
3. **Label Construction** - Class distributions
4. **Leakage-Safe Evaluation** - Embargo validation table
5. **Feature Sanity Check** - PCA analysis
6. **Models & Baselines** - Comparison + test results
7. **Diagnostics** - Confusion matrices
8. **Robustness** - Rolling stability + bootstrap CI
9. **Decision Under Uncertainty** - Threshold analysis
10. **Portfolio Translation** - Equity curves
11. **Conclusion & Future Work** - Best model summary

## Navigation

- Sidebar buttons to jump to any slide
- Previous/Next buttons at bottom
- Run selector dropdown (if multiple runs exist)