"""
FIFA 2026 Player Performance AI Dashboard
==========================================
A premium, futuristic Streamlit analytics + ML application.

Run locally:
    streamlit run app.py

Author: Built with Claude
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

try:
    from streamlit_lottie import st_lottie
    import requests
    HAS_LOTTIE = True
except Exception:
    HAS_LOTTIE = False


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="FIFA 2026 Player Performance AI Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "fifa_world_cup_2026_player_performance.csv"

# Columns that are composite/derived scores -> excluded from ML features to avoid leakage
LEAKY_COLS = [
    "player_rating", "tournament_rating", "offensive_contribution",
    "defensive_contribution", "possession_impact", "pressure_resistance",
    "creativity_score", "consistency_score", "clutch_performance_score",
]

ID_COLS = [
    "player_id", "player_name", "match_id", "match_date", "stadium", "city",
    "team", "opponent_team", "club_name", "jersey_number",
]


# ============================================================================
# GLOBAL CSS -- Glassmorphism + Neumorphism + Neon Dark Theme
# ============================================================================
def inject_css():
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root{
        --bg-deep:#060818;
        --bg-navy:#0b0f2b;
        --bg-navy2:#121640;
        --purple:#8b5cf6;
        --purple2:#a855f7;
        --cyan:#22d3ee;
        --pink:#ff2e9a;
        --glass:rgba(255,255,255,0.055);
        --glass-border:rgba(255,255,255,0.12);
        --text-main:#eef0ff;
        --text-dim:#9aa1c7;
    }

    html, body, [class*="css"]{
        font-family:'Outfit', sans-serif;
    }

    /* ---------- APP BACKGROUND ---------- */
    .stApp{
        background:
            radial-gradient(circle at 15% 20%, rgba(139,92,246,0.20) 0%, transparent 45%),
            radial-gradient(circle at 85% 10%, rgba(34,211,238,0.14) 0%, transparent 40%),
            radial-gradient(circle at 50% 90%, rgba(255,46,154,0.12) 0%, transparent 45%),
            linear-gradient(160deg, var(--bg-deep) 0%, var(--bg-navy) 45%, var(--bg-navy2) 100%);
        background-attachment: fixed;
        color: var(--text-main);
    }

    /* Animated ambient orbs */
    .stApp::before{
        content:"";
        position:fixed; inset:0; z-index:0; pointer-events:none;
        background-image:
            radial-gradient(2px 2px at 20% 30%, rgba(255,255,255,0.5) 0, transparent 100%),
            radial-gradient(2px 2px at 70% 60%, rgba(255,255,255,0.35) 0, transparent 100%),
            radial-gradient(1.5px 1.5px at 40% 80%, rgba(255,255,255,0.4) 0, transparent 100%),
            radial-gradient(1.5px 1.5px at 90% 20%, rgba(255,255,255,0.3) 0, transparent 100%);
        animation: floatStars 18s ease-in-out infinite alternate;
        opacity:0.6;
    }
    @keyframes floatStars{
        0%{ transform: translateY(0px); }
        100%{ transform: translateY(-25px); }
    }

    #MainMenu{visibility:hidden;}
    footer{visibility:hidden;}
    header[data-testid="stHeader"]{background:transparent;}

    /* ---------- TITLES ---------- */
    .hero-title{
        font-family:'Space Grotesk', sans-serif;
        font-weight:700;
        font-size:2.7rem;
        background: linear-gradient(90deg, #ffffff 0%, var(--cyan) 45%, var(--purple2) 75%, var(--pink) 100%);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        background-size:200% auto;
        animation: shimmer 6s linear infinite;
        margin-bottom:0.1rem;
        letter-spacing:-0.5px;
    }
    @keyframes shimmer{
        0%{background-position:0% center;}
        100%{background-position:200% center;}
    }
    .hero-sub{
        color: var(--text-dim);
        font-size:1.05rem;
        font-weight:400;
        margin-bottom:1.4rem;
    }
    .section-title{
        font-family:'Space Grotesk', sans-serif;
        font-weight:600;
        font-size:1.5rem;
        color:var(--text-main);
        border-left:4px solid var(--purple2);
        padding-left:0.7rem;
        margin:1.4rem 0 1rem 0;
        animation: fadeInUp 0.6s ease both;
    }

    @keyframes fadeInUp{
        from{ opacity:0; transform: translateY(14px); }
        to{ opacity:1; transform: translateY(0); }
    }
    .fade-in{ animation: fadeInUp 0.7s ease both; }

    /* ---------- GLASS CARD ---------- */
    .glass-card{
        background: var(--glass);
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-radius: 20px;
        padding: 1.3rem 1.5rem;
        box-shadow:
            0 8px 32px rgba(0,0,0,0.35),
            inset 0 1px 0 rgba(255,255,255,0.08);
        transition: all 0.35s cubic-bezier(.2,.8,.2,1);
        position:relative;
        overflow:hidden;
        animation: fadeInUp 0.6s ease both;
    }
    .glass-card::before{
        content:"";
        position:absolute; top:-50%; left:-50%;
        width:200%; height:200%;
        background: linear-gradient(115deg, transparent 40%, rgba(255,255,255,0.06) 50%, transparent 60%);
        transform: translateX(-100%);
        transition: transform 0.8s ease;
    }
    .glass-card:hover::before{ transform: translateX(100%); }
    .glass-card:hover{
        transform: translateY(-6px) scale(1.012);
        border-color: rgba(168,85,247,0.55);
        box-shadow:
            0 18px 45px rgba(139,92,246,0.28),
            0 0 0 1px rgba(168,85,247,0.25),
            inset 0 1px 0 rgba(255,255,255,0.12);
    }

    /* ---------- KPI CARD ---------- */
    .kpi-card{
        background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
        border:1px solid var(--glass-border);
        border-radius:18px;
        padding:1.25rem 1.4rem;
        text-align:left;
        backdrop-filter: blur(14px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06);
        transition: all .3s ease;
        animation: fadeInUp 0.6s ease both;
        position:relative;
    }
    .kpi-card:hover{
        transform: translateY(-5px);
        border-color: rgba(34,211,238,0.5);
        box-shadow: 0 14px 34px rgba(34,211,238,0.22), inset 0 1px 0 rgba(255,255,255,0.1);
    }
    .kpi-icon{ font-size:1.7rem; margin-bottom:0.3rem; filter: drop-shadow(0 0 8px rgba(168,85,247,0.6)); }
    .kpi-label{ color:var(--text-dim); font-size:0.82rem; text-transform:uppercase; letter-spacing:1px; font-weight:500;}
    .kpi-value{
        font-family:'Space Grotesk', sans-serif;
        font-size:2.1rem; font-weight:700;
        background: linear-gradient(90deg, var(--cyan), var(--purple2));
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .kpi-delta{ color:#4ade80; font-size:0.8rem; font-weight:500; }

    /* ---------- NEON BADGE / PILL ---------- */
    .neon-pill{
        display:inline-block;
        padding:0.25rem 0.9rem;
        border-radius:999px;
        font-size:0.75rem;
        font-weight:600;
        letter-spacing:0.5px;
        background: rgba(168,85,247,0.14);
        border:1px solid rgba(168,85,247,0.5);
        color:#d8b4fe;
        box-shadow: 0 0 14px rgba(168,85,247,0.35);
    }
    .neon-pill.cyan{ background: rgba(34,211,238,0.12); border-color: rgba(34,211,238,0.5); color:#a5f3fc; box-shadow:0 0 14px rgba(34,211,238,0.3);}
    .neon-pill.pink{ background: rgba(255,46,154,0.12); border-color: rgba(255,46,154,0.5); color:#ffb3dd; box-shadow:0 0 14px rgba(255,46,154,0.3);}

    /* ---------- BUTTONS ---------- */
    .stButton>button, .stDownloadButton>button{
        background: linear-gradient(135deg, var(--purple), var(--cyan) 130%);
        color:white;
        border:none;
        border-radius:14px;
        padding:0.6rem 1.6rem;
        font-weight:600;
        letter-spacing:0.3px;
        box-shadow: 0 4px 18px rgba(139,92,246,0.45);
        transition: all 0.25s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 28px rgba(34,211,238,0.5);
        filter: brightness(1.08);
    }
    .stButton>button:active{ transform: translateY(0px) scale(0.98); }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"]{
        background: linear-gradient(180deg, rgba(11,15,43,0.97), rgba(6,8,24,0.98));
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] .block-container{ padding-top:1.4rem; }

    /* ---------- INPUTS ---------- */
    .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"]{
        background: var(--glass);
        border-radius:12px;
        border:1px solid var(--glass-border);
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input{
        background: rgba(255,255,255,0.04) !important;
        color: var(--text-main) !important;
        border-radius:10px !important;
        border:1px solid var(--glass-border) !important;
    }
    .stSlider [data-baseweb="slider"]{ padding-top:0.4rem; }

    /* ---------- DATAFRAME ---------- */
    [data-testid="stDataFrame"]{
        border-radius:16px;
        overflow:hidden;
        border:1px solid var(--glass-border);
    }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"]{ gap: 6px; }
    .stTabs [data-baseweb="tab"]{
        background: var(--glass);
        border-radius: 12px 12px 0 0;
        border:1px solid var(--glass-border);
        padding: 8px 18px;
        color: var(--text-dim);
    }
    .stTabs [aria-selected="true"]{
        color: white !important;
        background: linear-gradient(135deg, rgba(139,92,246,0.35), rgba(34,211,238,0.25)) !important;
        border-color: rgba(168,85,247,0.5) !important;
    }

    /* ---------- METRIC / PROGRESS ---------- */
    .stProgress > div > div{
        background: linear-gradient(90deg, var(--purple), var(--cyan));
    }

    /* ---------- DIVIDER GLOW ---------- */
    .glow-divider{
        height:1px;
        margin: 1.6rem 0;
        background: linear-gradient(90deg, transparent, rgba(168,85,247,0.6), rgba(34,211,238,0.6), transparent);
        box-shadow: 0 0 10px rgba(139,92,246,0.4);
        border:none;
    }

    /* ---------- PREDICTION RESULT ---------- */
    .predict-result{
        text-align:center;
        padding: 2.2rem 1.5rem;
        border-radius:22px;
        background: linear-gradient(145deg, rgba(139,92,246,0.16), rgba(34,211,238,0.08));
        border: 1px solid rgba(168,85,247,0.5);
        box-shadow: 0 0 40px rgba(139,92,246,0.35), inset 0 1px 0 rgba(255,255,255,0.1);
        animation: popIn 0.55s cubic-bezier(.2,.9,.25,1.2) both, pulseGlow 2.4s ease-in-out infinite;
    }
    @keyframes popIn{
        0%{ opacity:0; transform: scale(0.85); }
        100%{ opacity:1; transform: scale(1); }
    }
    @keyframes pulseGlow{
        0%,100%{ box-shadow: 0 0 30px rgba(139,92,246,0.3); }
        50%{ box-shadow: 0 0 55px rgba(34,211,238,0.45); }
    }
    .predict-value{
        font-family:'Space Grotesk', sans-serif;
        font-size:3.4rem;
        font-weight:800;
        background: linear-gradient(90deg, var(--cyan), var(--purple2), var(--pink));
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }

    /* ---------- SCROLLBAR ---------- */
    ::-webkit-scrollbar{ width:9px; height:9px; }
    ::-webkit-scrollbar-track{ background: var(--bg-deep); }
    ::-webkit-scrollbar-thumb{ background: linear-gradient(var(--purple), var(--cyan)); border-radius:10px; }

    /* ---------- EXPANDER ---------- */
    .streamlit-expanderHeader{
        background: var(--glass) !important;
        border-radius:12px !important;
        border:1px solid var(--glass-border) !important;
    }

    /* ---------- BEST MODEL BADGE ---------- */
    .best-model-badge{
        display:inline-flex; align-items:center; gap:0.5rem;
        padding:0.5rem 1.1rem;
        border-radius:14px;
        background: linear-gradient(135deg, rgba(74,222,128,0.16), rgba(34,211,238,0.12));
        border:1px solid rgba(74,222,128,0.5);
        color:#bbf7d0;
        font-weight:600;
        box-shadow: 0 0 20px rgba(74,222,128,0.25);
    }

    </style>
    """,
        unsafe_allow_html=True,
    )


def glass_card_html(content: str) -> str:
    return f'<div class="glass-card fade-in">{content}</div>'


def kpi_card(icon, label, value, delta=None, accent="purple"):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def animated_counter(placeholder, target, prefix="", suffix="", decimals=0, duration=0.5, steps=18):
    """Simple animated count-up rendered into an st.empty placeholder."""
    try:
        target_val = float(target)
    except Exception:
        placeholder.markdown(f"### {target}")
        return
    for i in range(1, steps + 1):
        val = target_val * (i / steps)
        txt = f"{val:,.{decimals}f}"
        placeholder.markdown(
            f'<div class="kpi-value">{prefix}{txt}{suffix}</div>', unsafe_allow_html=True
        )
        time.sleep(duration / steps)
    txt = f"{target_val:,.{decimals}f}"
    placeholder.markdown(f'<div class="kpi-value">{prefix}{txt}{suffix}</div>', unsafe_allow_html=True)


# ============================================================================
# DATA LOADING & CLEANING
# ============================================================================
@st.cache_data(show_spinner=False)
def load_and_clean_data(path: str):
    df = pd.read_csv(path)
    report = {}
    report["raw_shape"] = df.shape
    report["missing_before"] = int(df.isnull().sum().sum())
    report["duplicates_before"] = int(df.duplicated().sum())

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Handle missing values: numeric -> median, categorical -> mode
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns

    for c in num_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    # Datatype conversions
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    for c in cat_cols:
        if c != "match_date":
            df[c] = df[c].astype(str).str.strip()

    report["missing_after"] = int(df.isnull().sum().sum())
    report["clean_shape"] = df.shape
    return df, report


@st.cache_data(show_spinner=False)
def get_numeric_cols(df):
    return df.select_dtypes(include=np.number).columns.tolist()


# ============================================================================
# ML HELPERS
# ============================================================================
FEATURE_CANDIDATES = [
    "age", "height_cm", "weight_kg", "minutes_played", "shots", "shots_on_target",
    "expected_goals_xg", "expected_assists_xa", "key_passes", "successful_passes",
    "total_passes", "pass_accuracy", "dribbles_attempted", "successful_dribbles",
    "crosses", "successful_crosses", "tackles", "interceptions", "clearances",
    "blocks", "aerial_duels_won", "aerial_duels_lost", "recoveries",
    "defensive_actions", "fouls_committed", "fouls_suffered", "distance_covered_km",
    "sprint_distance_km", "top_speed_kmh", "accelerations", "decelerations",
    "stamina_score", "goals", "assists",
]
CAT_FEATURE_CANDIDATES = ["position", "preferred_foot"]


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame, target: str, feature_cols, cat_cols):
    data = df.copy()
    feature_cols = [c for c in feature_cols if c in data.columns and c != target]
    cat_cols = [c for c in cat_cols if c in data.columns]

    X = data[feature_cols + cat_cols].copy()
    y = data[target].copy()

    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    }

    results = {}
    for name, model in models.items():
        if name == "Linear Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        results[name] = {
            "model": model,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "preds": preds,
            "y_test": y_test,
        }

    return {
        "results": results,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "encoders": encoders,
        "scaler": scaler,
        "X_train_columns": X.columns.tolist(),
    }


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
def sidebar_nav():
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.6rem 0 1.2rem 0;">
                <div style="font-size:2.4rem;">⚽</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.15rem;
                            background:linear-gradient(90deg,#22d3ee,#a855f7);
                            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                    FIFA 2026 AI
                </div>
                <div style="color:#9aa1c7; font-size:0.75rem; letter-spacing:1px;">PERFORMANCE DASHBOARD</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if HAS_OPTION_MENU:
            page = option_menu(
                menu_title=None,
                options=[
                    "Home Dashboard",
                    "Data Exploration",
                    "EDA",
                    "Machine Learning",
                    "Prediction",
                    "Model Insights",
                ],
                icons=["house-door-fill", "table", "bar-chart-line-fill",
                       "cpu-fill", "magic", "graph-up-arrow"],
                default_index=0,
                styles={
                    "container": {"padding": "0", "background-color": "transparent"},
                    "icon": {"color": "#22d3ee", "font-size": "16px"},
                    "nav-link": {
                        "font-size": "14.5px",
                        "text-align": "left",
                        "margin": "4px 0",
                        "border-radius": "12px",
                        "padding": "10px 14px",
                        "color": "#c7cbe8",
                        "--hover-color": "rgba(139,92,246,0.15)",
                    },
                    "nav-link-selected": {
                        "background": "linear-gradient(135deg, rgba(139,92,246,0.35), rgba(34,211,238,0.2))",
                        "color": "white",
                        "font-weight": "600",
                        "box-shadow": "0 0 16px rgba(139,92,246,0.35)",
                    },
                },
            )
        else:
            page = st.radio(
                "Navigate",
                ["Home Dashboard", "Data Exploration", "EDA", "Machine Learning",
                 "Prediction", "Model Insights"],
                label_visibility="collapsed",
            )

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.75rem; color:#9aa1c7; line-height:1.6;">
            <span class="neon-pill cyan">LIVE</span> &nbsp;
            <span class="neon-pill">AI-Powered</span><br><br>
            Built with Streamlit, Plotly &amp; Scikit-Learn.<br>
            Dark glassmorphism UI, 2026 edition.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return page


# ============================================================================
# PAGE: HOME DASHBOARD
# ============================================================================
def page_home(df):
    st.markdown('<div class="hero-title">FIFA 2026 Player Performance AI Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Real-time analytics &amp; machine learning insights across every match, '
        'every nation, every player of the FIFA World Cup 2026.</div>',
        unsafe_allow_html=True,
    )

    total_players = df["player_id"].nunique()
    avg_goals = df["goals"].mean()
    avg_assists = df["assists"].mean()
    top_country = (
        df.groupby("nationality")["performance_score"].mean().sort_values(ascending=False).index[0]
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("👥", "Total Players", f"{total_players:,}", "Across 48 nations")
    with c2:
        kpi_card("🥅", "Avg Goals / Match", f"{avg_goals:.2f}", "Per player appearance")
    with c3:
        kpi_card("🎯", "Avg Assists / Match", f"{avg_assists:.2f}", "Per player appearance")
    with c4:
        kpi_card("🏆", "Top Performing Nation", top_country, "By avg performance score")

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    colA, colB = st.columns([1.3, 1])
    with colA:
        st.markdown('<div class="section-title">🌍 Performance by Nation</div>', unsafe_allow_html=True)
        nation_perf = (
            df.groupby("nationality")["performance_score"].mean().sort_values(ascending=False).head(12).reset_index()
        )
        fig = px.bar(
            nation_perf, x="performance_score", y="nationality", orientation="h",
            color="performance_score", color_continuous_scale=["#22d3ee", "#a855f7", "#ff2e9a"],
        )
        fig = style_fig(fig, height=420)
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.markdown('<div class="section-title">⚔️ Position Mix</div>', unsafe_allow_html=True)
        pos_counts = df["position"].value_counts().reset_index()
        pos_counts.columns = ["position", "count"]
        fig2 = px.pie(
            pos_counts, names="position", values="count", hole=0.58,
            color_discrete_sequence=["#a855f7", "#22d3ee", "#ff2e9a", "#4ade80"],
        )
        fig2 = style_fig(fig2, height=420)
        fig2.update_traces(textfont_color="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">🔥 Tournament Pulse</div>', unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    with c5:
        st.markdown(
            glass_card_html(
                f"""<div class="kpi-icon">🏟️</div>
                <div class="kpi-label">Total Matches Tracked</div>
                <div class="kpi-value">{df['match_id'].nunique():,}</div>"""
            ),
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            glass_card_html(
                f"""<div class="kpi-icon">💰</div>
                <div class="kpi-label">Avg Market Value</div>
                <div class="kpi-value">€{df['market_value_eur'].mean()/1e6:.1f}M</div>"""
            ),
            unsafe_allow_html=True,
        )
    with c7:
        top_scorer = df.groupby("player_name")["goals"].sum().idxmax()
        st.markdown(
            glass_card_html(
                f"""<div class="kpi-icon">⭐</div>
                <div class="kpi-label">Top Goal Scorer</div>
                <div class="kpi-value" style="font-size:1.5rem;">{top_scorer}</div>"""
            ),
            unsafe_allow_html=True,
        )


# ============================================================================
# PAGE: DATA EXPLORATION
# ============================================================================
def page_data_exploration(df, report):
    st.markdown('<div class="hero-title">📊 Data Exploration</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Filter, search, and inspect the cleaned dataset.</div>', unsafe_allow_html=True)

    with st.expander("🧹 Data Cleaning Summary", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Rows", f"{report['raw_shape'][0]:,}")
        c2.metric("Missing Values Fixed", f"{report['missing_before']:,}")
        c3.metric("Duplicates Removed", f"{report['duplicates_before']:,}")
        c4.metric("Final Shape", f"{report['clean_shape'][0]:,} × {report['clean_shape'][1]}")

    st.markdown('<div class="section-title">🔍 Filters</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        nations = st.multiselect("Nationality", sorted(df["nationality"].unique()))
    with f2:
        positions = st.multiselect("Position", sorted(df["position"].unique()))
    with f3:
        age_range = st.slider("Age Range", int(df["age"].min()), int(df["age"].max()),
                               (int(df["age"].min()), int(df["age"].max())))
    with f4:
        search = st.text_input("🔎 Search Player Name")

    filtered = df.copy()
    if nations:
        filtered = filtered[filtered["nationality"].isin(nations)]
    if positions:
        filtered = filtered[filtered["position"].isin(positions)]
    filtered = filtered[(filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])]
    if search:
        filtered = filtered[filtered["player_name"].str.contains(search, case=False, na=False)]

    st.markdown(f'<span class="neon-pill cyan">{len(filtered):,} rows matched</span>', unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True, height=420)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📐 Column-wise Statistics</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Numeric Columns", "Categorical Columns"])
    with tab1:
        num_df = filtered.select_dtypes(include=np.number)
        st.dataframe(num_df.describe().T.style.background_gradient(cmap="PuBu"), use_container_width=True)
    with tab2:
        cat_df = filtered.select_dtypes(exclude=np.number)
        summary = pd.DataFrame({
            "unique_values": cat_df.nunique(),
            "most_frequent": cat_df.mode().iloc[0] if len(cat_df) else None,
        })
        st.dataframe(summary, use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data (CSV)", csv, "filtered_data.csv", "text/csv")


# ============================================================================
# PLOTLY STYLE HELPER
# ============================================================================
def style_fig(fig, height=400):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaff", family="Outfit"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)"),
        transition={"duration": 500, "easing": "cubic-in-out"},
    )
    return fig


# ============================================================================
# PAGE: EDA
# ============================================================================
def page_eda(df):
    st.markdown('<div class="hero-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Interactive visual exploration of player performance patterns.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🥇 Top Players by Goals</div>', unsafe_allow_html=True)
    top_n = st.slider("Number of players", 5, 25, 10, key="topn_goals")
    top_goals = df.groupby("player_name")["goals"].sum().sort_values(ascending=False).head(top_n).reset_index()
    fig = px.bar(
        top_goals, x="goals", y="player_name", orientation="h", color="goals",
        color_continuous_scale=["#22d3ee", "#a855f7", "#ff2e9a"], text="goals",
    )
    fig = style_fig(fig, height=max(400, top_n * 26))
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        st.markdown('<div class="section-title">🌐 Country-wise Performance</div>', unsafe_allow_html=True)
        metric_choice = st.selectbox(
            "Metric", ["performance_score", "goals", "assists", "player_rating"], key="country_metric"
        )
        country_perf = df.groupby("nationality")[metric_choice].mean().sort_values(ascending=False).head(15).reset_index()
        fig2 = px.bar(
            country_perf, x="nationality", y=metric_choice, color=metric_choice,
            color_continuous_scale=["#22d3ee", "#a855f7", "#ff2e9a"],
        )
        fig2 = style_fig(fig2, height=380)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with cB:
        st.markdown('<div class="section-title">📅 Age vs Performance</div>', unsafe_allow_html=True)
        fig3 = px.scatter(
            df.sample(min(3000, len(df)), random_state=1), x="age", y="performance_score",
            color="position", opacity=0.65,
            color_discrete_sequence=["#a855f7", "#22d3ee", "#ff2e9a", "#4ade80"],
            trendline=None,
        )
        fig3 = style_fig(fig3, height=380)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title">🔗 Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_cols = st.multiselect(
        "Select numeric features for correlation",
        get_numeric_cols(df),
        default=["age", "goals", "assists", "shots", "expected_goals_xg", "pass_accuracy",
                  "tackles", "distance_covered_km", "performance_score", "player_rating"],
    )
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr()
        fig4 = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale=["#0b0f2b", "#8b5cf6", "#22d3ee"],
        )
        fig4 = style_fig(fig4, height=500)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Select at least 2 columns to render the heatmap.")

    cC, cD = st.columns(2)
    with cC:
        st.markdown('<div class="section-title">📊 Distribution Plot</div>', unsafe_allow_html=True)
        dist_col = st.selectbox("Column", get_numeric_cols(df), index=get_numeric_cols(df).index("performance_score"))
        fig5 = px.histogram(df, x=dist_col, nbins=40, color_discrete_sequence=["#a855f7"], marginal="box")
        fig5 = style_fig(fig5, height=380)
        st.plotly_chart(fig5, use_container_width=True)

    with cD:
        st.markdown('<div class="section-title">🎻 Performance by Position</div>', unsafe_allow_html=True)
        fig6 = px.violin(
            df, x="position", y="performance_score", color="position", box=True,
            color_discrete_sequence=["#a855f7", "#22d3ee", "#ff2e9a", "#4ade80"],
        )
        fig6 = style_fig(fig6, height=380)
        fig6.update_layout(showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)


# ============================================================================
# PAGE: MACHINE LEARNING
# ============================================================================
def page_ml(df):
    st.markdown('<div class="hero-title">🤖 Machine Learning</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Train and compare regression models to predict player performance.</div>',
        unsafe_allow_html=True,
    )

    target = st.radio("🎯 Prediction Target", ["performance_score", "goals"], horizontal=True)

    with st.spinner("Training models... crunching the numbers ⚙️"):
        bundle = train_models(df, target, FEATURE_CANDIDATES, CAT_FEATURE_CANDIDATES)

    st.session_state["ml_bundle"] = bundle
    st.session_state["ml_target"] = target

    results = bundle["results"]
    comp_df = pd.DataFrame({
        name: {"R² Score": r["r2"], "RMSE": r["rmse"], "MAE": r["mae"]}
        for name, r in results.items()
    }).T.reset_index().rename(columns={"index": "Model"})

    best_model_name = comp_df.loc[comp_df["R² Score"].idxmax(), "Model"]

    st.markdown('<div class="section-title">🏁 Model Performance</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, name in zip([c1, c2, c3], results.keys()):
        with col:
            r = results[name]
            badge = '<div class="best-model-badge">🏆 Best Model</div>' if name == best_model_name else ""
            st.markdown(
                glass_card_html(
                    f"""<div class="kpi-label">{name}</div>
                    <div class="kpi-value">{r['r2']:.3f}</div>
                    <div style="color:#9aa1c7; font-size:0.82rem;">R² Score</div>
                    <div style="margin-top:0.5rem; font-size:0.82rem; color:#9aa1c7;">
                        RMSE: {r['rmse']:.3f} &nbsp;|&nbsp; MAE: {r['mae']:.3f}
                    </div>
                    <div style="margin-top:0.6rem;">{badge}</div>"""
                ),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    cL, cR = st.columns(2)
    with cL:
        st.markdown('<div class="section-title">📊 Model Comparison</div>', unsafe_allow_html=True)
        fig = px.bar(
            comp_df, x="Model", y="R² Score", color="Model", text="R² Score",
            color_discrete_sequence=["#22d3ee", "#a855f7", "#ff2e9a"],
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig = style_fig(fig, height=380)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with cR:
        st.markdown('<div class="section-title">🎯 Predicted vs Actual</div>', unsafe_allow_html=True)
        model_pick = st.selectbox("Model", list(results.keys()), index=list(results.keys()).index(best_model_name))
        r = results[model_pick]
        sample_idx = np.random.choice(len(r["y_test"]), min(1500, len(r["y_test"])), replace=False)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=np.array(r["y_test"])[sample_idx], y=np.array(r["preds"])[sample_idx],
            mode="markers", marker=dict(color="#22d3ee", size=6, opacity=0.55),
            name="Predictions",
        ))
        lims = [min(r["y_test"].min(), r["preds"].min()), max(r["y_test"].max(), r["preds"].max())]
        fig2.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(color="#ff2e9a", dash="dash"), name="Ideal"))
        fig2 = style_fig(fig2, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        f"""<div class="glass-card fade-in">
        <span class="best-model-badge">🏆 Best performing model: {best_model_name} (R² = {comp_df['R² Score'].max():.3f})</span>
        <p style="color:#9aa1c7; margin-top:0.8rem;">
        The best model is selected automatically based on the highest R² score on the held-out test set (20% split).
        Head to the <b>Prediction</b> page to generate a live forecast, or <b>Model Insights</b> to see feature importance.
        </p>
        </div>""",
        unsafe_allow_html=True,
    )


# ============================================================================
# PAGE: PREDICTION
# ============================================================================
def page_prediction(df):
    st.markdown('<div class="hero-title">🔮 Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter player stats and get a real-time performance forecast.</div>', unsafe_allow_html=True)

    if "ml_bundle" not in st.session_state:
        st.warning("⚠️ Please visit the **Machine Learning** page first to train the models.")
        return

    bundle = st.session_state["ml_bundle"]
    target = st.session_state["ml_target"]
    results = bundle["results"]
    comp_r2 = {k: v["r2"] for k, v in results.items()}
    best_name = max(comp_r2, key=comp_r2.get)

    model_choice = st.selectbox("Choose Model for Prediction", list(results.keys()),
                                 index=list(results.keys()).index(best_name))

    st.markdown('<div class="section-title">⚙️ Player Input</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 17, 40, 25)
            position = st.selectbox("Position", sorted(df["position"].unique()))
            preferred_foot = st.selectbox("Preferred Foot", sorted(df["preferred_foot"].unique()))
            minutes_played = st.slider("Minutes Played", 0, 120, 75)
            shots = st.slider("Shots", 0, 12, 3)
            shots_on_target = st.slider("Shots on Target", 0, 10, 1)
        with col2:
            expected_goals_xg = st.slider("Expected Goals (xG)", 0.0, 3.0, 0.3, 0.05)
            expected_assists_xa = st.slider("Expected Assists (xA)", 0.0, 2.0, 0.2, 0.05)
            key_passes = st.slider("Key Passes", 0, 10, 2)
            pass_accuracy = st.slider("Pass Accuracy (%)", 30.0, 100.0, 82.0)
            dribbles_attempted = st.slider("Dribbles Attempted", 0, 15, 3)
            tackles = st.slider("Tackles", 0, 12, 2)
        with col3:
            distance_covered_km = st.slider("Distance Covered (km)", 4.0, 14.0, 9.5)
            sprint_distance_km = st.slider("Sprint Distance (km)", 0.0, 3.0, 0.8)
            top_speed_kmh = st.slider("Top Speed (km/h)", 20.0, 38.0, 30.0)
            stamina_score = st.slider("Stamina Score", 0.0, 100.0, 75.0)
            goals = st.slider("Goals (this match)", 0, 4, 0)
            assists = st.slider("Assists (this match)", 0, 3, 0)

        submitted = st.form_submit_button("🚀 Predict Performance")

    if submitted:
        input_row = {
            "age": age, "height_cm": df["height_cm"].mean(), "weight_kg": df["weight_kg"].mean(),
            "minutes_played": minutes_played, "shots": shots, "shots_on_target": shots_on_target,
            "expected_goals_xg": expected_goals_xg, "expected_assists_xa": expected_assists_xa,
            "key_passes": key_passes, "successful_passes": df["successful_passes"].mean(),
            "total_passes": df["total_passes"].mean(), "pass_accuracy": pass_accuracy,
            "dribbles_attempted": dribbles_attempted, "successful_dribbles": df["successful_dribbles"].mean(),
            "crosses": df["crosses"].mean(), "successful_crosses": df["successful_crosses"].mean(),
            "tackles": tackles, "interceptions": df["interceptions"].mean(),
            "clearances": df["clearances"].mean(), "blocks": df["blocks"].mean(),
            "aerial_duels_won": df["aerial_duels_won"].mean(), "aerial_duels_lost": df["aerial_duels_lost"].mean(),
            "recoveries": df["recoveries"].mean(), "defensive_actions": df["defensive_actions"].mean(),
            "fouls_committed": df["fouls_committed"].mean(), "fouls_suffered": df["fouls_suffered"].mean(),
            "distance_covered_km": distance_covered_km, "sprint_distance_km": sprint_distance_km,
            "top_speed_kmh": top_speed_kmh, "accelerations": df["accelerations"].mean(),
            "decelerations": df["decelerations"].mean(), "stamina_score": stamina_score,
            "goals": goals, "assists": assists,
            "position": position, "preferred_foot": preferred_foot,
        }

        X_new = pd.DataFrame([input_row])
        for c in bundle["cat_cols"]:
            le = bundle["encoders"][c]
            try:
                X_new[c] = le.transform(X_new[c].astype(str))
            except ValueError:
                X_new[c] = 0

        X_new = X_new[bundle["X_train_columns"]]

        model = results[model_choice]["model"]
        if model_choice == "Linear Regression":
            X_new_scaled = bundle["scaler"].transform(X_new)
            prediction = model.predict(X_new_scaled)[0]
        else:
            prediction = model.predict(X_new)[0]

        unit = "" if target == "performance_score" else " goals"
        st.balloons()
        st.markdown(
            f"""
            <div class="predict-result">
                <div style="color:#9aa1c7; letter-spacing:1px; text-transform:uppercase; font-size:0.85rem;">
                    Predicted {target.replace('_',' ').title()}
                </div>
                <div class="predict-value">{prediction:.2f}{unit}</div>
                <div style="margin-top:0.6rem;">
                    <span class="neon-pill cyan">Model: {model_choice}</span>
                    &nbsp;<span class="neon-pill">R² = {results[model_choice]['r2']:.3f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# PAGE: MODEL INSIGHTS
# ============================================================================
def page_model_insights(df):
    st.markdown('<div class="hero-title">📊 Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Understand what drives the model\'s predictions.</div>', unsafe_allow_html=True)

    if "ml_bundle" not in st.session_state:
        st.warning("⚠️ Please visit the **Machine Learning** page first to train the models.")
        return

    bundle = st.session_state["ml_bundle"]
    results = bundle["results"]

    tree_models = {k: v for k, v in results.items() if k in ["Random Forest", "Decision Tree"]}
    model_choice = st.selectbox("Select tree-based model", list(tree_models.keys()))
    model = tree_models[model_choice]["model"]

    importances = pd.DataFrame({
        "feature": bundle["X_train_columns"],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(15)

    st.markdown('<div class="section-title">🌳 Feature Importance</div>', unsafe_allow_html=True)
    fig = px.bar(
        importances, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale=["#22d3ee", "#a855f7", "#ff2e9a"],
    )
    fig = style_fig(fig, height=480)
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 Model Explanation</div>', unsafe_allow_html=True)

    explanations = {
        "Linear Regression": "Fits a straight-line relationship between each feature and the target. "
                              "Fast, interpretable, and a strong baseline — but assumes linear relationships.",
        "Random Forest": "An ensemble of many decision trees, each trained on a random subset of data and features. "
                          "Averaging their predictions reduces overfitting and typically yields the strongest accuracy.",
        "Decision Tree": "Splits the data into branches based on feature thresholds, forming a tree of decisions. "
                          "Easy to visualize and interpret, but can overfit if not depth-limited.",
    }

    cols = st.columns(3)
    for col, (name, desc) in zip(cols, explanations.items()):
        with col:
            st.markdown(
                glass_card_html(
                    f"""<div class="kpi-label">{name}</div>
                    <p style="color:#c7cbe8; font-size:0.9rem; margin-top:0.5rem;">{desc}</p>"""
                ),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">📈 Top 5 Feature Impact Summary</div>', unsafe_allow_html=True)
    top5 = importances.head(5)
    for _, row in top5.iterrows():
        pct = row["importance"] * 100
        st.markdown(f"**{row['feature'].replace('_',' ').title()}** — {pct:.1f}% contribution")
        st.progress(min(1.0, row["importance"] / importances["importance"].max()))


# ============================================================================
# MAIN
# ============================================================================
def main():
    inject_css()

    with st.spinner("Loading FIFA 2026 dataset..."):
        df, report = load_and_clean_data(DATA_PATH)

    page = sidebar_nav()

    if page == "Home Dashboard":
        page_home(df)
    elif page == "Data Exploration":
        page_data_exploration(df, report)
    elif page == "EDA":
        page_eda(df)
    elif page == "Machine Learning":
        page_ml(df)
    elif page == "Prediction":
        page_prediction(df)
    elif page == "Model Insights":
        page_model_insights(df)


if __name__ == "__main__":
    main()
