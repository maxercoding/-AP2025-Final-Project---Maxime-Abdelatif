# Video Presentation dashboard

Professional presentation dashboard designed to accompany the live video presentation.  
**Matches the final teleprompter script exactly.**

## Quick Start

```bash
python Dashboard/run_dashboard.py
```

Opens at `http://localhost:8501`

## Requirements

```bash
pip install streamlit pandas pyyaml
```

## Structure

The dashboard follows the **exact presentation structure**:

### PART A — Before Live Demo (Slides 1-10)

| Slide | Title | Section |
|-------|-------|---------|
| 1 | Title & Thesis | Problem Motivation |
| 2 | Background & Motivation | Problem Motivation |
| 3 | Problem Statement | Problem Motivation |
| 4 | Research Question | Problem Motivation |
| 5 | Literature Overview | Technical Approach |
| 6 | Data | Technical Approach |
| 7 | Features | Technical Approach |
| 8 | Target Construction | Technical Approach |
| 9 | Leakage Controls | Technical Approach |
| 10 | Models & Metrics | Technical Approach |

### LIVE DEMO

*`python main.py` and walk through the pipeline*

### PART B — After Live Demo (Slides 11-15)

| Slide | Title | Section |
|-------|-------|---------|
| 11 | Baselines First | Results & Learnings |
| 12 | Predictability | Results & Learnings |
| 13 | Reliability | Results & Learnings |
| 14 | Portfolio | Results & Learnings |
| 15 | Conclusion | Results & Learnings |

## Features

- **Progress bar** — Visual indicator of presentation progress
- **Part A/B badges** — Clear indication of before/after demo sections
- **Color-coded groups** — Red (Motivation), Blue (Methodology), Green (Results)
- **Keyboard navigation** — Use sidebar buttons or on-screen Previous/Next
- **Auto-loads artifacts** — Figures and tables from latest run

## Expected Artifacts

Auto-detects runs from `results/runs/<timestamp>/`:

```
results/runs/YYYYMMDD_HHMMSS/
├── config_used.yaml
├── figures/
│   ├── fig_3.1_class_distributions.png
│   ├── fig_3.2_regime_timeline.png
│   ├── fig_4.1_confusion_matrices.png
│   ├── fig_4.3_model_comparison.png
│   ├── fig_5.1_rolling_stability.png
│   ├── fig_5.2_bootstrap_significance.png
│   ├── fig_5.3_threshold_analysis.png
│   ├── fig_6.1_equity_curves.png
│   └── ...
├── tables/
│   ├── test_results.csv
│   ├── embargo_validation.csv
│   ├── abstain_policy_comparison.csv
│   └── ...
└── logs/
    └── run.log
```

## Design

- **Navy + Gold** color scheme (private banking aesthetic)
- **Crimson Pro** serif for headings
- **Inter** sans-serif for body text
- **Card-based** slide design with shadows
- **Responsive** layout optimized for screen recording