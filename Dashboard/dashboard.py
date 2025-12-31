"""
ML Market Regime Forecasting - Video Support Dashboard (Redesigned)
===================================================================
Professional presentation dashboard matching Video Guidelines structure.

Two modes:
- Storyboard Mode: All sections scrollable (default)
- Presentation Mode: One section at a time

Theme: Private banking sober (off-white, navy accents, gold highlights)
"""

import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "runs"

# Section definitions matching Video Guidelines exactly
SECTIONS = [
    ("title", "Title & Thesis"),
    ("motivation", "Background & Motivation"),
    ("problem", "Problem Statement: Leakage Risks"),
    ("pillars", "Research Question & Two Pillars"),
    ("lit1", "Literature: Regimes Matter"),
    ("lit2", "Literature: Predictability is Weak"),
    ("lit3", "Literature: Integrity Toolkit"),
    ("data", "Methodology: Data"),
    ("features", "Methodology: Features (+1 Week Lag)"),
    ("target", "Methodology: Target Construction"),
    ("leakage", "Methodology: Leakage Controls"),
    ("models", "Models, Baselines, Metrics & Abstention"),
    ("baselines", "Results: Baselines First"),
    ("predictability", "Results: Predictability (Pillar 1)"),
    ("reliability", "Results: Reliability (Pillar 2)"),
    ("portfolio", "Portfolio Translation (Illustrative)"),
    ("conclusion", "Conclusion & Future Work"),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_latest_run_dir() -> Optional[Path]:
    if not RESULTS_DIR.exists():
        return None
    run_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda x: x.name, reverse=True)
    return run_dirs[0]


def get_all_run_dirs() -> list:
    if not RESULTS_DIR.exists():
        return []
    return sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()], reverse=True)


def load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_figure(run_dir: Path, fig_name: str) -> Optional[Path]:
    fig_path = run_dir / "figures" / fig_name
    return fig_path if fig_path.exists() else None


def get_best_model_summary(test_df: pd.DataFrame) -> Dict[str, Any]:
    if test_df is None or test_df.empty:
        return {"model": "Unknown", "macro_f1": 0.0}
    
    f1_col = next((c for c in test_df.columns if 'macro' in c.lower() and 'f1' in c.lower()), None)
    model_col = next((c for c in test_df.columns if 'model' in c.lower()), None)
    
    if not f1_col or not model_col:
        return {"model": "Unknown", "macro_f1": 0.0}
    
    idx = test_df[f1_col].idxmax()
    row = test_df.loc[idx]
    
    result = {"model": row[model_col], "macro_f1": row[f1_col]}
    
    for col in test_df.columns:
        if 'balanced' in col.lower():
            result['balanced_acc'] = row[col]
        elif 'bear' in col.lower() and 'recall' in col.lower():
            result['recall_bear'] = row[col]
        elif 'neutral' in col.lower() and 'recall' in col.lower():
            result['recall_neutral'] = row[col]
        elif 'bull' in col.lower() and 'recall' in col.lower():
            result['recall_bull'] = row[col]
    
    return result


def parse_run_datetime(run_name: str) -> str:
    try:
        dt = datetime.strptime(run_name, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return run_name


# =============================================================================
# STYLING
# =============================================================================

def apply_styling():
    st.markdown("""
    <style>
        .stApp { background-color: #F7F8FA; }
        .main .block-container { max-width: 1400px; padding: 2rem 3rem; }
        
        [data-testid="stSidebar"] { background-color: #1A2F4A; }
        [data-testid="stSidebar"] * { color: #E8ECF0 !important; }
        [data-testid="stSidebar"] hr { border-color: #2A4060; }
        
        h1 { color: #1A2F4A; font-family: Georgia, serif; font-size: 2.2rem; }
        h2 { color: #1A2F4A; font-family: Georgia, serif; font-size: 1.5rem;
             border-bottom: 2px solid #C9A227; padding-bottom: 0.4rem; margin-top: 1.5rem; }
        h3 { color: #2A4060; font-size: 1.05rem; font-weight: 600; }
        
        .section-block {
            background: white; border-radius: 8px; padding: 1.5rem 2rem;
            margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border-left: 4px solid #1A2F4A;
        }
        .section-title {
            font-family: Georgia, serif; font-size: 1.4rem; font-weight: 600;
            color: #1A2F4A; border-bottom: 2px solid #C9A227;
            padding-bottom: 0.5rem; margin-bottom: 1rem;
        }
        .narration-box {
            background: #F0F4F8; border-radius: 6px; padding: 1rem 1.2rem;
            font-size: 0.95rem; line-height: 1.6; color: #333; margin-bottom: 1rem;
        }
        .key-points-box {
            background: #FAFBFC; border: 1px solid #E0E4E8;
            border-radius: 6px; padding: 1rem 1.2rem;
        }
        .key-points-box ul { margin: 0; padding-left: 1.2rem; }
        .key-points-box li { margin-bottom: 0.4rem; color: #2A4060; font-size: 0.92rem; }
        .takeaway-box {
            background: linear-gradient(135deg, #1A2F4A 0%, #2A4060 100%);
            color: white; border-radius: 6px; padding: 0.8rem 1.2rem;
            margin-top: 1rem; font-size: 0.95rem;
        }
        .takeaway-box strong { color: #C9A227; }
        .warning-box {
            background: #FFF8E6; border: 1px solid #E6C84A;
            border-radius: 4px; padding: 0.5rem 1rem;
            color: #8B6914; font-size: 0.85rem; margin: 0.5rem 0;
        }
        .run-summary {
            background: white; border: 1px solid #E0E4E8;
            border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem;
        }
        .stButton > button {
            background: #1A2F4A; color: white; border: none;
            border-radius: 4px; padding: 0.5rem 1.5rem;
        }
        .stButton > button:hover { background: #2A4060; }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# RENDERING HELPERS
# =============================================================================

def render_warning(msg: str):
    st.markdown(f'<div class="warning-box">{msg}</div>', unsafe_allow_html=True)

def render_takeaway(text: str):
    st.markdown(f'<div class="takeaway-box"><strong>Key Takeaway:</strong> {text}</div>', unsafe_allow_html=True)

def render_section_header(title: str, section_num: int):
    st.markdown(f'<div class="section-title">{section_num}. {title}</div>', unsafe_allow_html=True)

def render_narration(text: str):
    st.markdown(f'<div class="narration-box">{text}</div>', unsafe_allow_html=True)

def render_key_points(points: List[str]):
    items = "".join(f"<li>{p}</li>" for p in points)
    st.markdown(f'<div class="key-points-box"><ul>{items}</ul></div>', unsafe_allow_html=True)

def render_figure(run_dir: Path, fig_name: str, caption: str = None):
    fig_path = load_figure(run_dir, fig_name)
    if fig_path:
        st.image(str(fig_path))
        if caption:
            st.caption(caption)
    else:
        render_warning(f"Figure not found: {fig_name}")

def render_table(df: pd.DataFrame, height: int = 200):
    if df is not None and not df.empty:
        st.dataframe(df, height=height)
    else:
        render_warning("Table data not available")


# =============================================================================
# SECTION CONTENT (17 sections matching Video Guidelines)
# =============================================================================

def section_title(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Title & Thesis", 1)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### ML-Based Weekly Market Regime Forecasting")
        st.markdown("#### A Leakage-Safe Pipeline for Decision Support")
        render_narration(
            "This project builds a leakage-safe ML pipeline to forecast weekly market decision "
            "regimes (Bear / Neutral / Bull) using strictly lagged features. The evaluation focuses "
            "on predictability versus strong baselines and reliability under uncertainty-aware rules. "
            "This is decision support, not 'alpha claims'."
        )
        render_key_points([
            "Output: weekly regime forecast (Bear / Neutral / Bull)",
            "Purpose: decision support, not point return forecasts",
            "Evaluation: baseline-first + reliability checks",
            "Core contribution: disciplined, auditable pipeline design",
        ])
    with col2:
        st.markdown("##### Run Configuration")
        if config:
            labels_cfg = config.get('labels', {})
            splits_cfg = config.get('splits', {})
            st.markdown(f"**Horizon H:** {labels_cfg.get('horizon_weeks', 1)} week")
            st.markdown(f"**Vol Window:** {labels_cfg.get('vol_win_weeks', 52)} weeks")
            st.markdown(f"**Threshold K:** {labels_cfg.get('k_vol_units', 0.3)}")
            st.markdown(f"**Seed:** {config.get('seed', 42)}")
        if test_summary.get('model') != 'Unknown':
            st.markdown("##### Best Model")
            st.metric("Model", test_summary['model'])
            st.metric("Macro-F1", f"{test_summary['macro_f1']:.3f}")
    render_takeaway("A disciplined, auditable pipeline matters more than fancy models.")


def section_motivation(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Background & Motivation", 2)
    col1, col2 = st.columns([1, 1])
    with col1:
        render_narration(
            "Instead of predicting precise returns (hard OOS), I forecast decision regimes. "
            "Bull maps to risk-on, Bear to risk-off, and Neutral is an explicit no-edge zone "
            "enabling abstention rather than forcing trades. Weekly horizon matches institutional "
            "rebalancing cadence and reduces turnover compared to daily predictions."
        )
        render_key_points([
            "Regimes map to decisions, not point forecasts",
            "Neutral is operational: supports 'don't trade' decisions",
            "Weekly frequency = noise/turnover trade-off",
            "Three-class structure enables uncertainty-aware decisions",
        ])
        render_takeaway("Neutral makes the system realistic: it supports 'don't trade'.")
    with col2:
        st.markdown("##### Evidence: Regime Timeline")
        render_figure(run_dir, "fig_3.2_regime_timeline.png")


def section_problem(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Problem Statement: Leakage Risks", 3)
    render_narration(
        "The main challenge in financial ML is not model choice—it's leakage. Random splits, "
        "full-data feature selection, and improper tuning create optimistic backtests. This project "
        "prevents those failure modes and produces auditable out-of-sample evidence."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### Random Splits")
        st.markdown("Time series violates i.i.d. assumptions. Random shuffling leaks future info.")
    with col2:
        st.markdown("##### Selection Bias")
        st.markdown("Feature selection on full data before CV inflates estimates systematically.")
    with col3:
        st.markdown("##### Tuning Leakage")
        st.markdown("HP tuning without proper time structure contaminates model selection.")
    render_takeaway("Integrity controls are the core contribution of this project.")


def section_pillars(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Research Question & Two Pillars", 4)
    render_narration(
        "Can a leakage-safe weekly ML pipeline forecast next-horizon regimes from strictly lagged "
        "features, beat strong non-ML baselines, and remain reliable under uncertainty-aware rules? "
        "I answer with two evidence pillars, keeping claims qualified."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Pillar 1: Predictability")
        render_key_points([
            "Do ML models beat strong baselines OOS?",
            "Strong baselines: persistence, momentum rules",
            "Metric: Macro-F1 (handles class imbalance)",
        ])
    with col2:
        st.markdown("##### Pillar 2: Reliability")
        render_key_points([
            "Is performance stable through time? (rolling)",
            "Is it statistically significant? (bootstrap CI)",
            "Does abstention improve trust? (threshold policies)",
        ])
    render_takeaway("Not 'is it above random?'—but 'is it competitive and reliable?'")


def section_lit1(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Literature: Regimes Matter for Allocation", 5)
    col1, col2 = st.columns([2, 1])
    with col1:
        render_narration(
            "Regime-switching frameworks are widely used because regimes materially affect "
            "allocation decisions. State-dependent rules can dominate static allocations. "
            "Multi-state settings with 2–4 regimes are standard, motivating this three-regime framing."
        )
        render_key_points([
            "Regimes enable state-dependent allocation rules",
            "2–4 regimes common (Ang & Bekaert; Guidolin & Timmermann)",
            "Bear/Neutral/Bull maps to risk-off/no-edge/risk-on",
        ])
    with col2:
        st.markdown("##### Key References")
        st.markdown("- Ang & Bekaert (2002)\n- Guidolin & Timmermann (2007)")
    render_takeaway("The regime lens is standard; the contribution is implementing it correctly.")


def section_lit2(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Literature: Predictability is Weak → Baselines First", 6)
    render_narration(
        "Return predictability is typically weak OOS, so evaluation must be baseline-first. "
        "The correct question is not 'above random' but whether ML beats strong heuristics "
        "like persistence and momentum rules. Many variables fail OOS—simple averages often win."
    )
    render_key_points([
        "Baselines can be surprisingly strong (Goyal & Welch, 2008)",
        "Be skeptical of small improvements over heuristics",
        "'ML theater' avoided by baseline-first reporting",
        "The bar: beat persistence AND momentum rule",
    ])
    render_takeaway("Strong baselines define the real bar to clear.")


def section_lit3(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Literature: Integrity Toolkit for Financial ML", 7)
    render_narration(
        "Financial time series violate i.i.d.; overlapping labels leak across folds. I use "
        "purge/embargo (López de Prado), train-only selection (Ambroise & McLachlan), "
        "time-aware tuning (Varma & Simon), and block bootstrap (Künsch)."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Integrity Checklist")
        render_key_points([
            "Purged/embargoed splits for overlapping labels",
            "Train-only feature selection",
            "Time-series CV for HP tuning",
            "Block bootstrap for uncertainty",
        ])
    with col2:
        st.markdown("##### Key References")
        st.markdown("- López de Prado (2018)\n- Ambroise & McLachlan (2002)\n- Varma & Simon (2006)")
    render_takeaway("Finance ML is mainly about getting the evaluation right.")


def section_data(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Methodology: Data", 8)
    col1, col2 = st.columns([2, 1])
    with col1:
        render_narration(
            "I use daily adjusted close for SPX, TLT (20Y Treasury), GLD (Gold), and VIX, "
            "resampled to weekly Friday close. After warmup, the sample is 678 weekly observations "
            "from 2012 to 2024, split into train/validation/test with strict temporal ordering."
        )
        render_key_points([
            "Assets: SPX, TLT, GLD, VIX",
            "Weekly resampling (W-FRI) = institutional cadence",
            "678 weeks after warmup (2012–2024)",
            "Clear train/val/test with embargo gaps",
        ])
    with col2:
        st.markdown("##### Split Summary")
        if config:
            splits = config.get('splits', {})
            st.markdown(f"**Train End:** {splits.get('train_end', '2017-12-31')}")
            st.markdown(f"**Val End:** {splits.get('val_end', '2020-12-31')}")
            st.markdown(f"**Test End:** 2024-12-27")
    render_takeaway("Weekly frequency reduces noise and aligns with rebalancing reality.")


def section_features(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Methodology: Features (+1 Week Strict Lag)", 9)
    col1, col2 = st.columns([1, 1])
    with col1:
        render_narration(
            "I engineer 36 features across trend, momentum, volatility, cross-asset, seasonality, "
            "and drawdown families. A strict global +1 week shift ensures features at week t only "
            "use info available by week t−1. If this shift were wrong, the project collapses."
        )
        render_key_points([
            "36 features across 6 families",
            "Global shift(+1) is conservative & audit-friendly",
            "Predictors at t use info ≤ t−1 only",
            "Single line to verify: `features.shift(1)`",
        ])
        render_takeaway("Lag integrity is explicit and easy to verify.")
    with col2:
        st.markdown("##### Feature Families")
        st.markdown("""
        | Family | Examples |
        |--------|----------|
        | Trend/MA | SPX vs MA20, MA50, MA200 |
        | Momentum | 12w return, 50d return |
        | Volatility | VIX level, VIX slope |
        | Cross-asset | TLT/SPX ratio |
        | Seasonality | Month, quarter |
        | Drawdown | Current DD depth |
        """)


def section_target(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Methodology: Target Construction (Vol-Adjusted)", 10)
    col1, col2 = st.columns([1, 1])
    with col1:
        render_narration(
            "The target is a 3-class label defined by forward 1-week SPX return relative to a "
            "volatility-adjusted band. σ is computed past-only over 52 weeks and shifted by 1 week. "
            "Bull is above +Kσ, Bear below −Kσ, otherwise Neutral. K=0.3 creates balanced classes."
        )
        render_key_points([
            "Label uses forward return vs ±K·σ band",
            "σ is past-only (52 weeks) and shifted",
            "Neutral = explicit no-edge zone",
            "K=0.3 → ~28% Bear, ~28% Neutral, ~44% Bull",
        ])
        render_takeaway("Outcome labels are valid when predictors are strictly lagged.")
    with col2:
        st.markdown("##### Evidence: Class Distribution")
        render_figure(run_dir, "fig_3.1_class_distributions.png")


def section_leakage(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Methodology: Leakage Controls", 11)
    col1, col2 = st.columns([1, 1])
    with col1:
        render_narration(
            "Splits are strictly time-ordered. Purge removes H weeks at boundaries where labels "
            "overlap; embargo adds a gap between splits. Effective separation is 2 weeks when H=1, "
            "validated in embargo_validation.csv. Feature selection uses only first 70% of training."
        )
        render_key_points([
            "Time-ordered splits only (no shuffling)",
            "Purge: remove H weeks at fold boundaries",
            "Embargo: gap between train/val/test",
            "Train-only pruning (first 70% of train)",
        ])
        render_takeaway("The evaluation design is leakage-aware and auditable.")
    with col2:
        st.markdown("##### Evidence: Embargo Validation")
        embargo_df = load_csv(run_dir / "tables" / "embargo_validation.csv")
        render_table(embargo_df, height=180) if embargo_df is not None else render_warning("embargo_validation.csv not found")


def section_models(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Models, Baselines, Metrics & Abstention", 12)
    render_narration(
        "I compare four model families against five baselines. Macro-F1 is the headline metric "
        "(weights classes equally). Abstention: when max class prob < τ, fall back to Neutral or Persistence."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### Models")
        render_key_points(["Logistic Regression", "Random Forest", "XGBoost", "Stacking"])
    with col2:
        st.markdown("##### Baselines")
        render_key_points(["Majority class", "Stratified random", "Persistence", "Momentum rule", "Trend+Vol rule"])
    with col3:
        st.markdown("##### Metrics")
        render_key_points(["Macro-F1", "Per-class recall", "Abstain: max prob < τ"])
    render_takeaway("Macro-F1 + abstention = decision-focused evaluation.")


def section_baselines(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Results: Baselines First", 13)
    col1, col2 = st.columns([1, 1])
    with col1:
        render_narration(
            "Before celebrating ML, establish the baseline bar. Random achieves ~0.262 Macro-F1, "
            "but persistence (~0.357) and momentum rule (~0.364) set the real bar. "
            "If ML doesn't beat these, there's no result."
        )
        render_key_points([
            "Random baseline: ~0.262 Macro-F1",
            "Persistence: ~0.357 Macro-F1",
            "Momentum rule: ~0.364 Macro-F1 (must-beat bar)",
        ])
        render_takeaway("If you don't beat heuristics, you don't have a result.")
    with col2:
        st.markdown("##### Evidence: Baseline Performance")
        test_df = load_csv(run_dir / "tables" / "test_results.csv")
        if test_df is not None:
            baseline_mask = test_df.iloc[:, 0].str.contains('Baseline|baseline', case=False, na=False)
            render_table(test_df[baseline_mask] if baseline_mask.any() else test_df.tail(5), height=200)


def section_predictability(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Results: Predictability (Pillar 1)", 14)
    col1, col2 = st.columns([1, 1])
    with col1:
        render_narration(
            "On the test set, Logistic Regression achieves the best OOS performance. Complex models "
            "(XGBoost, Stacking) underperform and sometimes collapse a class. The strongest model is "
            "the simple linear one—common when sample size is limited."
        )
        best = test_summary
        st.markdown(f"##### Best Model: {best.get('model', 'LR')}")
        st.markdown(f"- **Macro-F1:** {best.get('macro_f1', 0):.3f}")
        if 'balanced_acc' in best:
            st.markdown(f"- **Balanced Acc:** {best.get('balanced_acc', 0):.3f}")
        if 'recall_bear' in best:
            st.markdown(f"- **Bear Recall:** {best.get('recall_bear', 0):.1%}")
        render_takeaway("'Simple + disciplined' beats 'complex + fragile'.")
    with col2:
        st.markdown("##### Evidence: Model Comparison")
        render_figure(run_dir, "fig_4.3_model_comparison.png")
        with st.expander("Confusion Matrices"):
            render_figure(run_dir, "fig_4.1_confusion_matrices.png")


def section_reliability(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Results: Reliability (Pillar 2)", 15)
    render_narration(
        "Reliability: (1) rolling performance shows stability, (2) block bootstrap gives uncertainty "
        "around Macro-F1, (3) abstention shows coverage–reliability tradeoff. Honest takeaway: "
        "clearly above random, competitive with heuristics, but incremental gain is modest."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Rolling Stability")
        render_figure(run_dir, "fig_5.1_rolling_stability.png")
    with col2:
        st.markdown("##### Bootstrap Significance")
        render_figure(run_dir, "fig_5.2_bootstrap_significance.png")
    st.markdown("---")
    col3, col4 = st.columns([1, 1])
    with col3:
        st.markdown("##### Threshold Analysis")
        render_figure(run_dir, "fig_5.3_threshold_analysis.png")
    with col4:
        st.markdown("##### Abstain Policy Comparison")
        abstain_df = load_csv(run_dir / "tables" / "abstain_policy_comparison.csv")
        render_table(abstain_df, height=180) if abstain_df is not None else render_warning("abstain_policy_comparison.csv not found")
    render_takeaway("Reliability is about trust, not just one leaderboard number.")


def section_portfolio(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Portfolio Translation (Illustrative)", 16)
    st.markdown("*This section is secondary—the core contribution is classification methodology.*")
    col1, col2 = st.columns([1, 1])
    with col1:
        render_narration(
            "Regimes translate to allocations: Bear → defensive, Bull → risk-on. During 2021–2024 "
            "(bullish), buy-and-hold dominates. LR Sharpe ~0.35, degrading with transaction costs—"
            "highlighting the gap between statistical predictability and economic value."
        )
        render_key_points([
            "Portfolio is illustrative only",
            "TC sensitivity wipes out small edges",
            "Sharpe: ~0.49 (0 bps) → ~−0.21 (50 bps)",
        ])
        render_takeaway("Economic value is harder than statistical predictability.")
    with col2:
        st.markdown("##### Evidence: Equity Curves")
        render_figure(run_dir, "fig_6.1_equity_curves.png")
    with st.expander("Additional Portfolio Analysis"):
        col3, col4 = st.columns(2)
        with col3:
            render_figure(run_dir, "fig_6.2_portfolio_metrics.png")
        with col4:
            render_figure(run_dir, "fig_6.3_tc_sensitivity.png")


def section_conclusion(run_dir: Path, config: dict, test_summary: dict):
    render_section_header("Conclusion & Future Work", 17)
    col1, col2 = st.columns([2, 1])
    with col1:
        render_narration(
            "Answer: qualified but affirmative. A leakage-safe pipeline shows above-random evidence "
            "and supports uncertainty-aware decisions, but gains over strong heuristics are modest. "
            "Primary contribution: leakage controls + baseline-first evaluation + reliability tools."
        )
        st.markdown("##### Key Conclusions")
        render_key_points([
            f"Best model: {test_summary.get('model', 'LR')} (Macro-F1 = {test_summary.get('macro_f1', 0):.3f})",
            "Statistically above random; competitive with heuristics",
            "Simpler models win when sample is limited",
            "Primary contribution: auditable integrity + reliability",
        ])
        st.markdown("##### Future Work")
        render_key_points([
            "Sensitivity sweeps: H, K, embargo",
            "Probability calibration for abstention",
            "Alternative features: sentiment, macro",
            "Extended sample (pre-2010 data)",
        ])
    with col2:
        st.markdown("##### Final Summary")
        if test_summary.get('model') != 'Unknown':
            st.metric("Best Model", test_summary['model'])
            st.metric("Macro-F1", f"{test_summary['macro_f1']:.3f}")
        env_df = load_csv(run_dir / "tables" / "environment_versions.csv")
        if env_df is not None:
            with st.expander("Environment"):
                render_table(env_df, height=150)
    render_takeaway("Credibility-first ML beats flashy-but-fragile ML.")


# Section dispatcher
SECTION_FUNCS = {
    "title": section_title, "motivation": section_motivation, "problem": section_problem,
    "pillars": section_pillars, "lit1": section_lit1, "lit2": section_lit2, "lit3": section_lit3,
    "data": section_data, "features": section_features, "target": section_target,
    "leakage": section_leakage, "models": section_models, "baselines": section_baselines,
    "predictability": section_predictability, "reliability": section_reliability,
    "portfolio": section_portfolio, "conclusion": section_conclusion,
}


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(page_title="ML Regime Forecasting", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
    apply_styling()
    
    if 'section_idx' not in st.session_state:
        st.session_state.section_idx = 0
    if 'mode' not in st.session_state:
        st.session_state.mode = "Storyboard"
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("## ML Regime Forecasting")
        st.markdown("Video Support Dashboard")
        st.markdown("---")
        
        all_runs = get_all_run_dirs()
        if not all_runs:
            st.error(f"No runs found in {RESULTS_DIR}")
            st.stop()
        
        selected_run = st.selectbox("Select Run", all_runs, index=0,
            format_func=lambda x: f"{x} (latest)" if x == all_runs[0] else x)
        run_dir = RESULTS_DIR / selected_run
        st.markdown(f"**Date:** {parse_run_datetime(selected_run)}")
        st.markdown("---")
        
        st.session_state.mode = st.radio("View Mode", ["Storyboard", "Presentation"],
            index=0 if st.session_state.mode == "Storyboard" else 1, horizontal=True)
        st.markdown("---")
        
        st.markdown("### Sections")
        for i, (key, title) in enumerate(SECTIONS):
            short_title = title[:22] + "..." if len(title) > 25 else title
            if st.button(f"{i+1}. {short_title}", key=f"nav_{key}", use_container_width=True):
                st.session_state.section_idx = i
                st.rerun()
    
    # LOAD DATA
    config = load_yaml(run_dir / "config_used.yaml") or {}
    test_df = load_csv(run_dir / "tables" / "test_results.csv")
    test_summary = get_best_model_summary(test_df)
    
    # MAIN CONTENT
    if st.session_state.mode == "Presentation":
        current_idx = st.session_state.section_idx
        key, title = SECTIONS[current_idx]
        st.markdown('<div class="section-block">', unsafe_allow_html=True)
        SECTION_FUNCS[key](run_dir, config, test_summary)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if current_idx > 0 and st.button("← Previous"):
                st.session_state.section_idx -= 1
                st.rerun()
        with col2:
            st.markdown(f"<div style='text-align:center;color:#666;'>Section {current_idx+1} of {len(SECTIONS)}</div>", unsafe_allow_html=True)
        with col3:
            if current_idx < len(SECTIONS) - 1 and st.button("Next →"):
                st.session_state.section_idx += 1
                st.rerun()
    else:
        st.markdown("# ML-Based Weekly Market Regime Forecasting")
        st.markdown("### Video Support Dashboard — Storyboard View")
        
        st.markdown('<div class="run-summary">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Run", selected_run[:15])
        with col2:
            st.metric("Best Model", test_summary.get('model', 'N/A'))
        with col3:
            st.metric("Macro-F1", f"{test_summary.get('macro_f1', 0):.3f}")
        with col4:
            if config:
                h = config.get('labels', {}).get('horizon_weeks', 1)
                k = config.get('labels', {}).get('k_vol_units', 0.3)
                st.metric("Config", f"H={h}, K={k}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        for i, (key, title) in enumerate(SECTIONS):
            st.markdown('<div class="section-block">', unsafe_allow_html=True)
            SECTION_FUNCS[key](run_dir, config, test_summary)
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()