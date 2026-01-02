"""
ML Market Regime Forecasting — Video Presentation Dashboard
===========================================================
Designed to accompany the live video presentation.
Matches the final teleprompter script structure exactly.

Structure:
  PART A — Before Demo (Slides 1-10)
    • Problem Motivation: Slides 1-4
    • Technical Approach: Slides 5-10
  
  [LIVE DEMO]
  
  PART B — After Demo (Slides 11-15)
    • Results & Learnings: Slides 11-15
"""

import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "runs"

# Slides matching FINAL SCRIPT exactly
SLIDES = [
    # PART A - Problem Motivation (Slides 1-4)
    {"id": "title", "num": 1, "title": "Title & Thesis", "part": "A", "group": "Problem Motivation"},
    {"id": "motivation", "num": 2, "title": "Background & Motivation", "part": "A", "group": "Problem Motivation"},
    {"id": "problem", "num": 3, "title": "Problem Statement", "part": "A", "group": "Problem Motivation"},
    {"id": "pillars", "num": 4, "title": "Research Question", "part": "A", "group": "Problem Motivation"},
    # PART A - Technical Approach (Slides 5-10)
    {"id": "literature", "num": 5, "title": "Literature Overview", "part": "A", "group": "Technical Approach"},
    {"id": "data", "num": 6, "title": "Data", "part": "A", "group": "Technical Approach"},
    {"id": "features", "num": 7, "title": "Features", "part": "A", "group": "Technical Approach"},
    {"id": "target", "num": 8, "title": "Target Construction", "part": "A", "group": "Technical Approach"},
    {"id": "leakage", "num": 9, "title": "Leakage Controls", "part": "A", "group": "Technical Approach"},
    {"id": "models", "num": 10, "title": "Models & Metrics", "part": "A", "group": "Technical Approach"},
    # PART B - Results (Slides 11-15)
    {"id": "baselines", "num": 11, "title": "Baselines First", "part": "B", "group": "Results & Learnings"},
    {"id": "predictability", "num": 12, "title": "Predictability", "part": "B", "group": "Results & Learnings"},
    {"id": "reliability", "num": 13, "title": "Reliability", "part": "B", "group": "Results & Learnings"},
    {"id": "portfolio", "num": 14, "title": "Portfolio", "part": "B", "group": "Results & Learnings"},
    {"id": "conclusion", "num": 15, "title": "Conclusion", "part": "B", "group": "Results & Learnings"},
]

COLORS = {
    "Problem Motivation": "#dc2626",
    "Technical Approach": "#2563eb",
    "Results & Learnings": "#16a34a",
}

# =============================================================================
# HELPERS
# =============================================================================

def get_all_run_dirs() -> list:
    if not RESULTS_DIR.exists():
        return []
    return sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()], reverse=True)

def load_yaml(path: Path) -> Optional[Dict]:
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except:
        return None

def load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except:
        return None

def load_figure(run_dir: Path, fig_name: str) -> Optional[Path]:
    fig_path = run_dir / "figures" / fig_name
    return fig_path if fig_path.exists() else None

def get_best_model(test_df: pd.DataFrame) -> Dict:
    if test_df is None or test_df.empty:
        return {"model": "N/A", "macro_f1": 0.0}
    f1_col = next((c for c in test_df.columns if 'macro' in c.lower() and 'f1' in c.lower()), None)
    model_col = next((c for c in test_df.columns if 'model' in c.lower()), None)
    if not f1_col or not model_col:
        return {"model": "N/A", "macro_f1": 0.0}
    idx = test_df[f1_col].idxmax()
    row = test_df.loc[idx]
    result = {"model": row[model_col], "macro_f1": row[f1_col]}
    for col in test_df.columns:
        if 'balanced' in col.lower() and 'acc' in col.lower():
            result['balanced_acc'] = row[col]
    return result

# =============================================================================
# STYLES
# =============================================================================

def apply_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Crimson+Pro:wght@400;500;600;700&display=swap');
    
    :root {
        --navy: #0f172a;
        --navy-light: #1e293b;
        --slate: #334155;
        --gold: #d97706;
        --gold-light: #fbbf24;
        --cream: #fefce8;
        --white: #ffffff;
        --gray-50: #f8fafc;
        --gray-100: #f1f5f9;
        --gray-200: #e2e8f0;
        --gray-300: #cbd5e1;
        --gray-500: #64748b;
        --gray-700: #334155;
        --gray-900: #0f172a;
        --red: #dc2626;
        --blue: #2563eb;
        --green: #16a34a;
    }
    
    * { box-sizing: border-box; }
    
    .stApp {
        background: linear-gradient(180deg, #fafafa 0%, #f5f5f5 100%);
    }
    
    .main .block-container {
        max-width: 1150px !important;
        padding: 1rem 2.5rem 3rem !important;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Crimson Pro', Georgia, serif;
        color: var(--navy);
    }
    
    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--navy) 0%, var(--navy-light) 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08);
        margin: 0.5rem 0;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: none;
        text-align: left;
        padding: 0.4rem 0.75rem;
        font-size: 0.82rem;
        border-radius: 6px;
        border-left: 3px solid transparent;
        transition: all 0.15s;
        width: 100%;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.06);
        border-left-color: var(--gold);
    }
    
    /* === PROGRESS BAR === */
    .progress-container {
        background: var(--gray-200);
        height: 6px;
        border-radius: 3px;
        margin-bottom: 1.25rem;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 3px;
        transition: width 0.4s ease;
    }
    
    /* === PART INDICATOR === */
    .part-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }
    
    .part-a {
        background: linear-gradient(135deg, var(--navy) 0%, var(--slate) 100%);
        color: white;
    }
    
    .part-b {
        background: linear-gradient(135deg, var(--green) 0%, #15803d 100%);
        color: white;
    }
    
    /* === SLIDE CARD === */
    .slide-card {
        background: var(--white);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.03);
        position: relative;
    }
    
    /* === SLIDE HEADER === */
    .slide-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1.25rem;
        border-bottom: 2px solid var(--gray-100);
    }
    
    .slide-num {
        background: var(--navy);
        color: white;
        font-weight: 800;
        font-size: 1.1rem;
        width: 2.75rem;
        height: 2.75rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        font-family: 'Inter', sans-serif;
    }
    
    .slide-title {
        font-family: 'Crimson Pro', serif;
        font-size: 2.1rem;
        font-weight: 600;
        color: var(--navy);
        margin: 0;
        line-height: 1.15;
        flex-grow: 1;
    }
    
    .slide-group {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 0.3rem 0.75rem;
        border-radius: 4px;
        white-space: nowrap;
    }
    
    /* === CONTENT === */
    .key-message {
        background: linear-gradient(135deg, var(--gray-50) 0%, var(--white) 100%);
        border-left: 4px solid var(--gold);
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        font-size: 1.15rem;
        line-height: 1.7;
        color: var(--gray-700);
        border-radius: 0 12px 12px 0;
    }
    
    .key-message strong, .key-message b {
        color: var(--navy);
        font-weight: 600;
    }
    
    .points-box {
        background: var(--gray-50);
        border: 1px solid var(--gray-200);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    
    .points-title {
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--gray-500);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .points-title::before {
        content: '';
        width: 6px;
        height: 6px;
        background: var(--gold);
        border-radius: 2px;
    }
    
    .points-box ul {
        margin: 0;
        padding-left: 1.2rem;
    }
    
    .points-box li {
        color: var(--gray-700);
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0.4rem;
    }
    
    .points-box li::marker {
        color: var(--gold);
    }
    
    /* === TAKEAWAY === */
    .takeaway {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-top: 1.5rem;
        color: white;
        font-size: 1.05rem;
        line-height: 1.5;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .takeaway-arrow {
        color: var(--gold-light);
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    
    .takeaway strong {
        color: var(--gold-light);
    }
    
    /* === METRICS === */
    .metrics-grid {
        display: flex;
        gap: 0.75rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: var(--white);
        border: 2px solid var(--gray-200);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        flex: 1;
        transition: all 0.2s;
    }
    
    .metric-card:hover {
        border-color: var(--gold);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(217, 119, 6, 0.15);
    }
    
    .metric-card.highlight {
        border-color: var(--gold);
        background: linear-gradient(135deg, #fffbeb 0%, var(--white) 100%);
    }
    
    .metric-value {
        font-family: 'Crimson Pro', serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--navy);
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--gray-500);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.35rem;
    }
    
    /* === FIGURES === */
    .fig-label {
        font-size: 0.65rem;
        font-weight: 700;
        color: var(--gray-500);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }
    
    .fig-box {
        background: var(--gray-50);
        border-radius: 12px;
        padding: 0.75rem;
        border: 1px solid var(--gray-200);
    }
    
    /* === WARNING === */
    .warning-box {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        color: #92400e;
        font-size: 0.85rem;
    }
    
    /* === NAVIGATION === */
    .stButton > button {
        background: var(--navy);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: var(--slate);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.25);
    }
    
    /* === DEMO BREAK === */
    .demo-break {
        text-align: center;
        padding: 2.5rem 1rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, var(--navy) 0%, var(--slate) 100%);
        border-radius: 16px;
        color: white;
    }
    
    .demo-break h2 {
        color: white;
        font-size: 1.75rem;
        margin-bottom: 0.5rem;
    }
    
    .demo-break p {
        color: var(--gray-300);
        font-size: 1rem;
    }
    
    /* === HIDE STREAMLIT === */
    #MainMenu, footer {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="collapsedControl"] {visibility: visible !important;}
    
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        font-weight: 600;
        background: var(--gray-50);
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# COMPONENTS
# =============================================================================

def progress_bar(current: int, total: int, color: str):
    pct = (current / total) * 100
    st.markdown(f'''
    <div class="progress-container">
        <div class="progress-bar" style="width: {pct}%; background: {color};"></div>
    </div>
    ''', unsafe_allow_html=True)

def slide_header(slide: dict):
    color = COLORS.get(slide["group"], "#666")
    part_class = "part-a" if slide["part"] == "A" else "part-b"
    part_label = "Part A — Before Demo" if slide["part"] == "A" else "Part B — After Demo"
    
    st.markdown(f'<span class="part-badge {part_class}">{part_label}</span>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="slide-card">
        <div class="slide-header">
            <div class="slide-num">{slide["num"]}</div>
            <h2 class="slide-title">{slide["title"]}</h2>
            <span class="slide-group" style="background: {color}15; color: {color};">{slide["group"]}</span>
        </div>
    ''', unsafe_allow_html=True)

def slide_footer():
    st.markdown('</div>', unsafe_allow_html=True)

def key_message(text: str):
    st.markdown(f'<div class="key-message">{text}</div>', unsafe_allow_html=True)

def bullet_points(points: List[str], title: str = "Key Points"):
    items = "".join(f"<li>{p}</li>" for p in points)
    st.markdown(f'''
    <div class="points-box">
        <div class="points-title">{title}</div>
        <ul>{items}</ul>
    </div>
    ''', unsafe_allow_html=True)

def takeaway(text: str):
    st.markdown(f'''
    <div class="takeaway">
        <span class="takeaway-arrow">→</span>
        <span><strong>Takeaway:</strong> {text}</span>
    </div>
    ''', unsafe_allow_html=True)

def warning(msg: str):
    st.markdown(f'<div class="warning-box">⚠️ {msg}</div>', unsafe_allow_html=True)

def fig_title(title: str):
    st.markdown(f'<div class="fig-label">{title}</div>', unsafe_allow_html=True)

def show_fig(run_dir: Path, name: str, title: str = None):
    if title:
        fig_title(title)
    path = load_figure(run_dir, name)
    if path:
        st.markdown('<div class="fig-box">', unsafe_allow_html=True)
        st.image(str(path))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        warning(f"Figure not found: {name}")

def show_df(df: pd.DataFrame, height: int = 180, title: str = None):
    if title:
        fig_title(title)
    if df is not None and not df.empty:
        st.dataframe(df, height=height)
    else:
        warning("Data not available")

def metric_cards(data: List[tuple], highlight: int = None):
    html = '<div class="metrics-grid">'
    for i, (val, lbl) in enumerate(data):
        hl = ' highlight' if i == highlight else ''
        html += f'<div class="metric-card{hl}"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def demo_break():
    st.markdown('''
    <div class="demo-break">
        <h2>🖥️ Live Demo</h2>
        <p>Run the full pipeline and walk through the code</p>
    </div>
    ''', unsafe_allow_html=True)

# =============================================================================
# SLIDE CONTENT
# =============================================================================

def slide_1_title(run_dir, config, summary):
    slide_header(SLIDES[0])
    
    st.markdown("## ML-Based Weekly Market Regime Forecasting")
    st.markdown("### A Leakage-Safe Pipeline for Decision Support")
    
    c1, c2 = st.columns([3, 2])
    with c1:
        key_message(
            "Every week, portfolio managers face the same question: <b>risk-on, risk-off, or stay put?</b> "
            "This project explores whether ML can help — without falling into the usual backtesting traps."
        )
        bullet_points([
            "Forecast weekly regimes: <b>Bear / Neutral / Bull</b>",
            "Goal is <b>decision support</b>, not alpha claims",
            "Two pillars: <b>Predictability</b> + <b>Reliability</b>",
        ])
    with c2:
        fig_title("Configuration")
        if config:
            labels = config.get('labels', {})
            metric_cards([
                (f"{labels.get('horizon_weeks', 1)}w", "Horizon"),
                (f"{labels.get('k_vol_units', 0.3)}", "K Threshold"),
            ])
        fig_title("Best Result")
        if summary.get('model') != 'N/A':
            metric_cards([
                (summary['model'], "Best Model"),
                (f"{summary['macro_f1']:.3f}", "Macro-F1"),
            ], highlight=1)
    
    takeaway("A disciplined, auditable pipeline matters more than fancy models.")
    slide_footer()


def slide_2_motivation(run_dir, config, summary):
    slide_header(SLIDES[1])
    
    c1, c2 = st.columns([1, 1])
    with c1:
        key_message(
            "Instead of predicting exact returns — notoriously difficult OOS — I predict <b>decision regimes</b> as a practical abstraction."
        )
        bullet_points([
            "<b>Bull</b> → risk-on conditions",
            "<b>Bear</b> → risk-off conditions",
            "<b>Neutral</b> → no-edge zone, enables <b>abstention</b>",
        ], "Three Regimes")
        bullet_points([
            "Weekly = reduced noise vs daily",
            "Aligns with institutional rebalancing",
        ], "Why Weekly?")
    with c2:
        show_fig(run_dir, "fig_3.2_regime_timeline.png", "Regime Timeline (2012–2024)")
    
    takeaway("Neutral makes the system realistic: it supports 'don't trade' when uncertain.")
    slide_footer()


def slide_3_problem(run_dir, config, summary):
    slide_header(SLIDES[2])
    
    st.markdown("### Challenge 1: Data Leakage")
    key_message(
        "In financial ML, the main risk is <b>not model choice — it's leakage</b>. "
        "These mistakes create backtests that look impressive but fail completely out-of-sample."
    )
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🔀 Random Splits")
        st.markdown("Time series violates i.i.d. assumptions. Random shuffling **leaks future info**.")
    with c2:
        st.markdown("#### 📊 Selection Bias")
        st.markdown("Feature selection on full data **inflates performance** estimates.")
    with c3:
        st.markdown("#### ⚙️ Tuning Leakage")
        st.markdown("HP tuning without time structure **contaminates** model selection.")
    
    st.markdown("---")
    
    st.markdown("### Challenge 2: Regime Definition")
    key_message(
        "Regimes don't have a precise, observable metric — they are <b>interpretations of market trends</b>. "
        "I chose a <b>return/volatility definition</b>, which is most widely used in practice and provides clear, auditable thresholds."
    )
    
    takeaway("Integrity controls + principled regime definition are the core contributions.")
    slide_footer()


def slide_4_pillars(run_dir, config, summary):
    slide_header(SLIDES[3])
    
    key_message(
        "<i>Can a leakage-safe weekly ML pipeline forecast regimes from strictly lagged features, "
        "<b>beat strong baselines</b>, and remain <b>reliable</b> under uncertainty?</i>"
    )
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Pillar 1: Predictability")
        bullet_points([
            "Beat strong baselines OOS?",
            "Not random — realistic heuristics like <b>momentum</b>",
            "Metric: <b>Macro-F1</b>",
        ])
    with c2:
        st.markdown("### Pillar 2: Reliability")
        bullet_points([
            "Stable through time? (Rolling analysis)",
            "Statistically significant? (Bootstrap CI)",
            "Does abstaining help? (Threshold policies)",
        ])
    
    takeaway("Not 'above random?' — but 'competitive and reliable?'")
    slide_footer()


def slide_5_literature(run_dir, config, summary):
    slide_header(SLIDES[4])
    
    key_message("Three ideas from prior work anchor this project — each shapes the methodology.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📈 Regimes Matter")
        st.markdown("State-dependent portfolio rules outperform static allocations. **2–4 regimes** standard.")
        st.caption("Ang & Bekaert; Guidolin & Timmermann")
    with c2:
        st.markdown("#### 📉 Predictability is Weak")
        st.markdown("Many variables fail OOS. Evaluation must be **baseline-first**.")
        st.caption("Goyal & Welch (2008)")
    with c3:
        st.markdown("#### 🔒 Integrity Toolkit")
        st.markdown("Purge/embargo for overlapping labels, train-only selection, block bootstrap.")
        st.caption("López de Prado (2018)")
    
    takeaway("The novelty is combining these into a disciplined, auditable pipeline.")
    slide_footer()


def slide_6_data(run_dir, config, summary):
    slide_header(SLIDES[5])
    
    c1, c2 = st.columns([2, 1])
    with c1:
        key_message(
            "Four assets: <b>SPX</b> (equity), <b>TLT</b> (Treasuries), <b>GLD</b> (Gold), <b>VIX</b> (volatility). "
            "Resampled to <b>weekly Friday close</b>."
        )
        bullet_points([
            "<b>678 weekly observations</b> (2012–2024)",
            "Train ends <b>2017</b>, Val ends <b>2020</b>, Test <b>2021–2024</b>",
            "Strict chronological order — no shuffling",
        ])
    with c2:
        fig_title("Data Splits")
        metric_cards([
            ("312", "Train Weeks"),
            ("155", "Val Weeks"),
        ])
        metric_cards([
            ("209", "Test Weeks"),
            ("678", "Total"),
        ])
    
    takeaway("Weekly frequency reduces noise and aligns with institutional rebalancing.")
    slide_footer()


def slide_7_features(run_dir, config, summary):
    slide_header(SLIDES[6])
    
    c1, c2 = st.columns([1, 1])
    with c1:
        key_message(
            "<b>36 features</b> across six families. Critical: a <b>global +1 week lag</b> ensures "
            "features at week <i>t</i> only use info through week <i>t−1</i>."
        )
        bullet_points([
            "Trend & moving-average ratios",
            "Momentum signals",
            "Volatility & VIX indicators",
            "Cross-asset relationships",
            "Seasonality & drawdown metrics",
        ], "Feature Families")
    with c2:
        st.markdown("#### Sanity Check: No Trivial Separation")
        key_message(
            "PCA + clustering show <b>ARI = 0.005</b> with outcome labels — essentially random. "
            "Features cluster by volatility, <b>not</b> by outcome. Task is genuinely difficult."
        )
        metric_cards([
            ("36", "Features"),
            ("0.005", "ARI Score"),
        ])
    
    takeaway("Lag integrity is explicit — one line of code to verify.")
    slide_footer()


def slide_8_target(run_dir, config, summary):
    slide_header(SLIDES[7])
    
    c1, c2 = st.columns([1, 1])
    with c1:
        key_message(
            "3-class label: forward 1-week SPX return vs a <b>volatility-adjusted band</b>. "
            "Bull if above +Kσ, Bear if below −Kσ, else Neutral."
        )
        bullet_points([
            "σ from <b>past-only</b> 52-week window",
            "Shifted by 1 week — no peek ahead",
            "<b>K = 0.3</b> → ~28/28/44% split",
        ], "Design Choices")
        metric_cards([
            ("28%", "Bear"),
            ("28%", "Neutral"),
            ("44%", "Bull"),
        ])
    with c2:
        show_fig(run_dir, "fig_3.1_class_distributions.png", "Class Distribution")
    
    takeaway("Outcome labels are valid because predictors are strictly lagged.")
    slide_footer()


def slide_9_leakage(run_dir, config, summary):
    slide_header(SLIDES[8])
    
    c1, c2 = st.columns([1, 1])
    with c1:
        key_message(
            "The integrity core: <b>purge + embargo</b> around split boundaries, "
            "<b>train-only</b> feature selection, <b>time-series CV</b> for tuning."
        )
        bullet_points([
            "Splits are <b>strictly time-ordered</b>",
            "Purge removes H weeks at boundaries",
            "Embargo adds 1-week gap → <b>2 weeks total</b>",
            "Feature pruning on first <b>70% of train only</b>",
        ])
    with c2:
        embargo_df = load_csv(run_dir / "tables" / "embargo_validation.csv")
        show_df(embargo_df, 160, "Embargo Validation")
    
    takeaway("Overlapping labels invalidate naïve CV — purge/embargo fixes that.")
    slide_footer()


def slide_10_models(run_dir, config, summary):
    slide_header(SLIDES[9])
    
    key_message(
        "Four model families vs <b>five baselines</b>. Headline metric: <b>Macro-F1</b> (weights classes equally). "
        "Plus an <b>abstention layer</b>: when max prob < τ, fall back to Neutral."
    )
    
    c1, c2, c3 = st.columns(3)
    with c1:
        bullet_points(["Logistic Regression", "Random Forest", "XGBoost", "Stacking"], "Models")
    with c2:
        bullet_points(["Majority class", "Persistence", "Momentum rule", "Trend+Vol rule"], "Baselines")
    with c3:
        bullet_points(["Macro-F1 (primary)", "Per-class recall", "Abstention policies"], "Evaluation")
    
    takeaway("Macro-F1 + abstention = decision-focused evaluation.")
    slide_footer()


def slide_11_baselines(run_dir, config, summary):
    slide_header(SLIDES[10])
    
    c1, c2 = st.columns([1, 1])
    with c1:
        key_message(
            "First, establish the bar. Random is ~0.26, but <b>persistence reaches 0.357</b> and "
            "<b>momentum hits 0.364</b> — that's what ML must beat."
        )
        metric_cards([
            ("0.262", "Random"),
            ("0.357", "Persistence"),
            ("0.364", "Momentum"),
        ], highlight=2)
    with c2:
        test_df = load_csv(run_dir / "tables" / "test_results.csv")
        if test_df is not None:
            mask = test_df.iloc[:, 0].str.contains('Baseline', case=False, na=False)
            show_df(test_df[mask] if mask.any() else test_df.tail(5), 180, "Baseline Performance")
    
    takeaway("If ML doesn't beat strong heuristics, there's no result.")
    slide_footer()


def slide_12_predictability(run_dir, config, summary):
    slide_header(SLIDES[11])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        key_message(
            "<b>Logistic Regression performs best</b> with Macro-F1 = 0.385 vs momentum's 0.364 — "
            "about <b>6% relative improvement</b>. Complex models underperform."
        )
        if summary.get('model') != 'N/A':
            metric_cards([
                (summary['model'], "Best Model"),
                (f"{summary['macro_f1']:.3f}", "Macro-F1"),
                (f"{summary.get('balanced_acc', 0):.3f}", "Balanced Acc"),
            ], highlight=1)
    
    with col2:
        show_fig(run_dir, "fig_4.3_model_comparison.png", "Model Comparison")
    
    st.markdown("---")
    st.markdown("#### Why This Pattern? Design Choices & Tradeoffs")
    
    c1, c2 = st.columns(2)
    with c1:
        bullet_points([
            "<b>XGBoost:</b> Sample weights disabled after testing — caused over-prediction of Bear. Without weights, it under-predicts Neutral (2% recall).",
            "<b>Future fix:</b> Tune <code>scale_pos_weight</code> and regularization parameters.",
        ], "XGBoost (0.260)")
    with c2:
        bullet_points([
            "<b>Stacking:</b> Excludes XGBoost due to label encoding incompatibility. Only LR+RF combined, limiting diversity.",
            "<b>LR wins:</b> L2 regularization prevents extreme predictions, keeps classes balanced.",
        ], "Stacking (0.266) & LR (0.385)")
    
    with st.expander("View Confusion Matrices"):
        show_fig(run_dir, "fig_4.1_confusion_matrices.png")
    
    takeaway("Simple + regularized beats complex + miscalibrated. Implementation choices matter.")
    slide_footer()


def slide_13_reliability(run_dir, config, summary):
    slide_header(SLIDES[12])
    
    key_message("Three reliability checks: <b>rolling stability</b>, <b>block bootstrap CI</b>, and <b>abstention</b>.")
    
    c1, c2 = st.columns(2)
    with c1:
        show_fig(run_dir, "fig_5.1_rolling_stability.png", "Rolling 52-Week Macro-F1")
    with c2:
        show_fig(run_dir, "fig_5.2_bootstrap_significance.png", "Bootstrap 95% CI")
    
    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Statistical Interpretation")
        st.markdown("""
        95% CI: **[0.316, 0.451]**
        - ✓ Fully above random (0.262)
        - ⚠️ **Includes momentum (0.364)** — cannot claim statistical significance over baseline
        
        This is **consistent with EMH**: all features derive from past prices, which theory says contain limited predictive information.
        """)
    with c4:
        st.markdown("#### Abstention Policy")
        st.markdown("""
        At τ=0.4, abstain on **~18%** of weeks.
        
        Trades coverage for trustworthiness — the model says "I don't know" when uncertain, rather than guessing.
        """)
    
    with st.expander("Threshold Analysis & Abstain Policies"):
        cc1, cc2 = st.columns(2)
        with cc1:
            show_fig(run_dir, "fig_5.3_threshold_analysis.png")
        with cc2:
            abstain_df = load_csv(run_dir / "tables" / "abstain_policy_comparison.csv")
            show_df(abstain_df, 160)
    
    takeaway("Modest improvement over heuristics, consistent with market efficiency theory.")
    slide_footer()


def slide_14_portfolio(run_dir, config, summary):
    slide_header(SLIDES[13])
    
    st.caption("⚠️ This is illustrative only — not the main contribution")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        key_message(
            "Regimes → simple allocation rule. During 2021–2024 (bullish), <b>buy-and-hold dominates</b>. "
            "ML achieves Sharpe ~0.35 vs B&H's 0.83."
        )
        metric_cards([
            ("0.35", "LR Sharpe"),
            ("0.83", "B&H Sharpe"),
        ])
        st.markdown("**Transaction costs** further erode any small edge.")
    with c2:
        show_fig(run_dir, "fig_6.1_equity_curves.png", "Equity Curves (2021–2024)")
    
    with st.expander("Additional Portfolio Analysis"):
        cc1, cc2 = st.columns(2)
        with cc1:
            show_fig(run_dir, "fig_6.2_portfolio_metrics.png")
        with cc2:
            show_fig(run_dir, "fig_6.3_tc_sensitivity.png")
    
    takeaway("Statistical predictability doesn't automatically translate to economic value.")
    slide_footer()


def slide_15_conclusion(run_dir, config, summary):
    slide_header(SLIDES[14])
    
    key_message(
        "The answer is <b>qualified but affirmative</b>. A leakage-safe pipeline produces "
        "above-random forecasts — but gains over heuristics are <b>modest and expected</b> given EMH."
    )
    
    c1, c2 = st.columns([1, 1])
    with c1:
        bullet_points([
            f"Best: <b>{summary.get('model', 'LR')}</b> with Macro-F1 = <b>{summary.get('macro_f1', 0):.3f}</b>",
            "~6% improvement over momentum (not statistically significant)",
            "Complex models failed due to implementation choices, not just sample size",
        ], "Key Results")
        
        bullet_points([
            "XGBoost: tune <code>scale_pos_weight</code> and regularization",
            "Stacking: include XGBoost with proper label encoding wrapper",
            "HP search: broader grid for tree regularization parameters",
        ], "Implementation Improvements")
    
    with c2:
        bullet_points([
            "Price-based features have limited predictive power (EMH)",
            "312 training samples with 36 features — dimensionality risk",
            "Test period (2021-2024) was mostly bullish — may not generalize",
        ], "Honest Limitations")
        
        fig_title("Final Summary")
        metric_cards([
            (summary.get('model', 'LR'), "Model"),
            (f"{summary.get('macro_f1', 0):.3f}", "Macro-F1"),
        ], highlight=1)
        metric_cards([
            ("+6%", "vs Momentum"),
            ("~18%", "Abstain Rate"),
        ])
    
    takeaway("Contribution = auditable methodology + honest evaluation. Results are credible because they're modest.")
    slide_footer()


SLIDE_FUNCS = {
    "title": slide_1_title,
    "motivation": slide_2_motivation,
    "problem": slide_3_problem,
    "pillars": slide_4_pillars,
    "literature": slide_5_literature,
    "data": slide_6_data,
    "features": slide_7_features,
    "target": slide_8_target,
    "leakage": slide_9_leakage,
    "models": slide_10_models,
    "baselines": slide_11_baselines,
    "predictability": slide_12_predictability,
    "reliability": slide_13_reliability,
    "portfolio": slide_14_portfolio,
    "conclusion": slide_15_conclusion,
}

# =============================================================================
# MAIN
# =============================================================================

def main():
    st.set_page_config(
        page_title="ML Regime Forecasting",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_styles()
    
    if 'idx' not in st.session_state:
        st.session_state.idx = 0
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("## 📊 ML Regime Forecasting")
        st.caption("Video Presentation Dashboard")
        st.markdown("---")
        
        runs = get_all_run_dirs()
        if not runs:
            st.error("No runs found")
            st.stop()
        
        selected = st.selectbox("Run", runs)
        run_dir = RESULTS_DIR / selected
        
        st.markdown("---")
        
        # Navigation by group
        current_group = None
        for i, slide in enumerate(SLIDES):
            if slide["group"] != current_group:
                if slide["num"] == 11:
                    st.markdown("---")
                    st.caption("🖥️ LIVE DEMO")
                    st.markdown("---")
                st.caption(f"{slide['group'].upper()}")
                current_group = slide["group"]
            
            is_current = (i == st.session_state.idx)
            marker = "▶ " if is_current else ""
            if st.button(f"{marker}{slide['num']}. {slide['title']}", key=f"nav_{i}", use_container_width=True):
                st.session_state.idx = i
                st.rerun()
    
    # LOAD DATA
    config = load_yaml(run_dir / "config_used.yaml") or {}
    test_df = load_csv(run_dir / "tables" / "test_results.csv")
    summary = get_best_model(test_df)
    
    # CONTENT
    idx = st.session_state.idx
    slide = SLIDES[idx]
    color = COLORS.get(slide["group"], "#666")
    
    # Progress bar
    progress_bar(idx + 1, len(SLIDES), color)
    
    # Render slide
    SLIDE_FUNCS[slide["id"]](run_dir, config, summary)
    
    # Navigation
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if idx > 0:
            if st.button("← Previous"):
                st.session_state.idx -= 1
                st.rerun()
    with c2:
        part_label = "Part A" if slide["part"] == "A" else "Part B"
        st.markdown(f"<p style='text-align:center;color:#64748b;font-size:0.9rem;'>{part_label} · {slide['group']} · Slide {idx+1} of {len(SLIDES)}</p>", unsafe_allow_html=True)
    with c3:
        if idx < len(SLIDES) - 1:
            if st.button("Next →"):
                st.session_state.idx += 1
                st.rerun()


if __name__ == "__main__":
    main()