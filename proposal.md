# PROPOSAL.md — Multi-Model ETF Market Forecasting System (Archive)

## What I will build
I will build a **weekly market forecasting pipeline** that uses supervised machine learning to predict the **next market state** of US equities. The core objective is forecasting: estimating the **forward market outcome** (over a fixed horizon) from information available at time *t*, then translating that forecast into an interpretable **market regime signal** (e.g., Bear / Neutral / Bull) for analysis and comparison.

## Data I will use
I will use publicly available historical data for a small, non-overlapping asset universe:
- **SPY** (US equities),
- **TLT** (Treasuries),
- **GLD** (Gold),
- **VIX** (risk/volatility proxy).

The timeline will follow a clean out-of-sample split:
- **Train:** 2010–2018  
- **Test:** 2019–2024  
Hyperparameters will be chosen using a **time-ordered validation slice inside the training period** (no shuffling, no overlap).

## Question I will answer
**Can machine learning forecast future equity market conditions out-of-sample (2019–2024), using only lagged features observed up to time *t*?**  
The forecast target will be defined from SPY’s forward behavior over a fixed horizon (e.g., forward return / volatility-adjusted thresholds). For reporting, the continuous forecast will be summarized into a 3-state regime label to make errors economically interpretable (missed drawdowns vs false alarms, etc.).

## Models & evaluation
Models:
- **Logistic Regression** (baseline, interpretable),
- **Random Forest** (core),
- **XGBoost** (core).

Evaluation will focus on **out-of-sample forecasting skill** (e.g., balanced accuracy / Macro-F1 for regime summaries, plus probability quality where applicable), with robustness checks across sub-periods. A **light backtest** will be included to test that forecasts translate into sensible risk-on/risk-off shifts, not as the main deliverable.

Deliverable: a reproducible repository with a single runnable pipeline and clear results tables/figures.