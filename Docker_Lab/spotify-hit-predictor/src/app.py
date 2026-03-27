"""
Spotify Hit Predictor — Streamlit Dashboard
Sruthilaya | MLOps | Northeastern University
Spotify green/black theme
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Path resolution (Docker vs local) ────────────────────────────────────────
EXCHANGE_DIR = "/exchange" if os.path.exists("/exchange") else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "exchange"
)
DATA_DIR = "/app/data" if os.path.exists("/app/data") else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data"
)
MODEL_PATH   = os.path.join(EXCHANGE_DIR, "model.joblib")
SCALER_PATH  = os.path.join(EXCHANGE_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(EXCHANGE_DIR, "metrics.json")
DATA_PATH    = os.path.join(DATA_DIR, "spotify_subset.csv")

FEATURES = ["energy", "danceability", "tempo", "valence",
            "loudness", "acousticness", "speechiness"]

FEATURE_LABELS = {
    "energy": "Energy", "danceability": "Danceability",
    "tempo": "Tempo (BPM)", "valence": "Valence (mood)",
    "loudness": "Loudness (dB)", "acousticness": "Acousticness",
    "speechiness": "Speechiness",
}
FEATURE_RANGES   = {
    "energy": (0.0, 1.0, 0.01), "danceability": (0.0, 1.0, 0.01),
    "tempo": (50.0, 210.0, 1.0), "valence": (0.0, 1.0, 0.01),
    "loudness": (-40.0, 0.0, 0.5), "acousticness": (0.0, 1.0, 0.01),
    "speechiness": (0.0, 1.0, 0.01),
}
FEATURE_DEFAULTS = {
    "energy": 0.72, "danceability": 0.68, "tempo": 120.0,
    "valence": 0.55, "loudness": -6.5, "acousticness": 0.12, "speechiness": 0.06,
}
FEATURE_INFO = {
    "energy":       "How intense and active the track feels (0 = calm, 1 = intense)",
    "danceability": "How suitable the track is for dancing based on rhythm and beat",
    "tempo":        "Speed of the track in beats per minute (BPM)",
    "valence":      "Musical positiveness — high = happy/euphoric, low = sad/tense",
    "loudness":     "Overall loudness in decibels (dB). Typical range: -60 to 0",
    "acousticness": "Confidence the track is acoustic (0 = electric, 1 = acoustic)",
    "speechiness":  "Presence of spoken words (podcasts ~1.0, music ~0.0–0.1)",
}

# ── Theme — Spotify green/black ───────────────────────────────────────────────
GREEN       = "#1DB954"
GREEN_DIM   = "#158a3e"
GREEN_SOFT  = "#1ed760"
ACCENT      = "#1DB954"
PINK        = "#e05c5c"
CYAN        = "#1DB954"
BG          = "#121212"
BG2         = "#181818"
BG3         = "#242424"
CARD        = "#282828"
TEXT        = "#FFFFFF"
TEXT_MUTED  = "#A7A7A7"
PURPLE      = "#1DB954"
PURPLE_DARK = "#158a3e"

st.set_page_config(
    page_title="Spotify Hit Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  /* Global */
  .stApp {{ background-color: {BG}; color: {TEXT}; }}
  section[data-testid="stSidebar"] {{ background-color: {BG2}; border-right: 1px solid #2a2a2a; }}
  section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

  /* Header */
  .hero {{
    background: {BG2};
    border: 1px solid #2a2a2a;
    border-left: 3px solid {GREEN};
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .hero-title {{ font-size: 1.8rem; font-weight: 800; color: {TEXT}; margin: 0; }}
  .hero-title span {{ color: {GREEN}; }}
  .hero-sub {{ font-size: 0.82rem; color: {TEXT_MUTED}; margin-top: 0.25rem; }}
  .hero-badge {{ background: {BG3}; border: 1px solid #3a3a3a;
    border-radius: 20px; padding: 0.35rem 0.9rem; font-size: 0.78rem; color: {GREEN}; }}

  /* Cards */
  .card {{
    background: {CARD};
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }}
  .card-title {{
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: {PURPLE}; margin-bottom: 0.8rem;
  }}

  /* Prediction result */
  .prob-number {{ font-size: 4rem; font-weight: 900; line-height: 1; }}
  .verdict {{ font-size: 1.2rem; font-weight: 600; margin-top: 0.4rem; }}

  /* Pipeline steps */
  .pipeline {{
    display: flex; gap: 0; align-items: stretch;
    margin: 0.5rem 0;
  }}
  .pipe-step {{
    flex: 1; background: {BG3}; border: 1px solid #2a2a2a;
    padding: 0.8rem 0.6rem; text-align: center; position: relative;
  }}
  .pipe-step:first-child {{ border-radius: 10px 0 0 10px; }}
  .pipe-step:last-child  {{ border-radius: 0 10px 10px 0; }}
  .pipe-step.active {{ background: {PURPLE_DARK}33; border-color: {PURPLE}; }}
  .pipe-num {{ font-size: 0.65rem; color: {PURPLE}; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; }}
  .pipe-label {{ font-size: 0.8rem; color: {TEXT}; font-weight: 600; margin-top: 0.2rem; }}
  .pipe-sub {{ font-size: 0.68rem; color: {TEXT_MUTED}; margin-top: 0.1rem; }}
  .pipe-arrow {{
    position: absolute; right: -10px; top: 50%; transform: translateY(-50%);
    color: {PURPLE}; font-size: 1.1rem; z-index: 10;
    background: {BG}; padding: 0 2px;
  }}

  /* Explainer rows */
  .explain-row {{
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.5rem 0; border-bottom: 1px solid {PURPLE_DARK}22;
  }}
  .explain-bar-wrap {{ flex: 1; background: {BG3}; border-radius: 4px; height: 8px; }}
  .explain-bar {{ height: 8px; border-radius: 4px; }}
  .explain-label {{ width: 120px; font-size: 0.78rem; color: {TEXT_MUTED}; }}
  .explain-val {{ width: 48px; font-size: 0.78rem; color: {TEXT}; text-align: right; }}
  .explain-impact {{ width: 70px; font-size: 0.72rem; text-align: right; font-weight: 600; }}

  /* Dataset stat pills */
  .stat-pill {{
    display: inline-block; background: {BG3};
    border: 1px solid #2a2a2a; border-radius: 10px;
    padding: 0.6rem 1rem; margin: 0.3rem; text-align: center;
  }}
  .stat-num {{ font-size: 1.4rem; font-weight: 800; color: {PURPLE}; }}
  .stat-lbl {{ font-size: 0.72rem; color: {TEXT_MUTED}; }}

  /* Streamlit overrides */
  .stSlider > div {{ color: {TEXT}; }}
  div[data-testid="metric-container"] {{
    background: {CARD}; border: 1px solid #2a2a2a;
    border-radius: 10px; padding: 0.8rem 1rem;
  }}
  div[data-testid="metric-container"] label {{ color: {TEXT_MUTED} !important; font-size: 0.75rem !important; }}
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{ color: {GREEN} !important; font-size: 1.6rem !important; }}
  /* Tabs — proper spacing */
  .stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid #2a2a2a;
    gap: 0;
    padding: 0;
    margin-bottom: 1.5rem;
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {TEXT_MUTED};
    background: transparent;
    border-radius: 0;
    padding: 0.7rem 1.4rem;
    font-size: 0.88rem;
    font-weight: 500;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
  }}
  .stTabs [aria-selected="true"] {{
    color: {TEXT} !important;
    background: transparent !important;
    border-bottom: 2px solid {GREEN} !important;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT} !important;
    background: transparent !important;
  }}
  h2, h3 {{ color: {TEXT} !important; }}
  .stDivider {{ border-color: {PURPLE_DARK}44 !important; }}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return model, scaler, metrics

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

try:
    model, scaler, metrics = load_artifacts()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

df_data = load_dataset()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='color:{PURPLE};font-weight:700;font-size:1rem;margin-bottom:0.5rem;'>🎛️ Track Audio Features</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.78rem;margin-bottom:1rem;'>Adjust to profile your track and get a real-time hit prediction.</div>", unsafe_allow_html=True)

    user_input = {}
    for feat in FEATURES:
        lo, hi, step = FEATURE_RANGES[feat]
        user_input[feat] = st.slider(
            FEATURE_LABELS[feat],
            min_value=float(lo), max_value=float(hi),
            value=float(FEATURE_DEFAULTS[feat]), step=float(step),
            help=FEATURE_INFO[feat],
        )

    st.divider()
    st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.72rem;'>Model: XGBoost · Data: Spotify<br>Hit threshold: popularity ≥ {metrics['hit_threshold']}</div>", unsafe_allow_html=True)


# ── Prediction ─────────────────────────────────────────────────────────────────
input_vec = np.array([[user_input[f] for f in FEATURES]])
input_sc  = scaler.transform(input_vec)
proba     = float(model.predict_proba(input_sc)[0, 1])
is_hit    = proba >= 0.5
prob_color = PURPLE if is_hit else PINK
verdict    = "🎯 HIT" if is_hit else "📉 NOT A HIT"
verdict_sub = "This track has strong hit potential!" if is_hit else "Doesn't quite have the hit profile yet."


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div>
    <div class="hero-title">🎵 Spotify <span>Hit Predictor</span></div>
    <div class="hero-sub">Sruthilaya &nbsp;·&nbsp; MLOps &nbsp;·&nbsp; Northeastern University</div>
  </div>
  <div class="hero-badge">XGBoost · Docker MLOps Pipeline</div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Dataset & Model", "🧠 How It Works", "🔧 Pipeline"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_pred, col_radar, col_imp = st.columns([1.2, 1.4, 1.4])

    # ── Prediction card ──
    with col_pred:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Live Prediction</div>
          <div class="prob-number" style="color:{prob_color};">{proba:.0%}</div>
          <div class="verdict" style="color:{prob_color};">{verdict}</div>
          <div style="color:{TEXT_MUTED};font-size:0.82rem;margin-top:0.4rem;">{verdict_sub}</div>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        st.metric("AUC-ROC",  f"{metrics['auc_roc']:.3f}")
        st.metric("Hit rate in data", f"{metrics['hit_rate']:.1%}")

    # ── Radar ──
    with col_radar:
        st.markdown(f"<div class='card-title' style='color:{PURPLE};font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;'>Audio Fingerprint</div>", unsafe_allow_html=True)
        radar_feats  = ["energy", "danceability", "valence", "acousticness", "speechiness"]
        radar_labels = [FEATURE_LABELS[f] for f in radar_feats]
        norm_vals = [(user_input[f] - FEATURE_RANGES[f][0]) /
                     (FEATURE_RANGES[f][1] - FEATURE_RANGES[f][0]) for f in radar_feats]

        fig_radar = go.Figure(go.Scatterpolar(
            r=norm_vals + [norm_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor=f"rgba(155,89,255,0.15)",
            line=dict(color=PURPLE, width=2),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], showticklabels=False, gridcolor="#333366"),
                angularaxis=dict(gridcolor="#333366", linecolor="#444477", tickfont=dict(color=TEXT_MUTED, size=11)),
            ),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            margin=dict(l=40, r=40, t=20, b=20), height=300,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Feature importance ──
    with col_imp:
        st.markdown(f"<div class='card-title' style='color:{PURPLE};font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;'>Feature Importance</div>", unsafe_allow_html=True)
        fi    = metrics["feature_importance"]
        fi_df = pd.DataFrame({"Feature": [FEATURE_LABELS[k] for k in fi],
                               "Score": list(fi.values())}).sort_values("Score")
        colors = [PURPLE if v >= fi_df["Score"].median() else PINK for v in fi_df["Score"]]
        fig_imp = go.Figure(go.Bar(
            x=fi_df["Score"], y=fi_df["Feature"], orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in fi_df["Score"]], textposition="outside",
            textfont=dict(color=TEXT_MUTED, size=11),
        ))
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(gridcolor="#222244", tickfont=dict(color=TEXT_MUTED, size=11)),
            margin=dict(l=10, r=60, t=10, b=10), height=300,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    # ── Decision explainer ──
    st.markdown(f"<div class='card-title' style='color:{PURPLE};font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;'>Why is this track {'a hit' if is_hit else 'not a hit'}? — Decision Explainer</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.82rem;margin-bottom:1rem;'>Each feature's value vs what the model considers typical for a hit. Green = supports hit prediction, red = works against it.</div>", unsafe_allow_html=True)

    # Per-feature contribution heuristic based on importance × deviation from hit mean
    hit_means = {"energy": 0.72, "danceability": 0.68, "tempo": 122.0,
                 "valence": 0.55, "loudness": -5.5, "acousticness": 0.12, "speechiness": 0.07}

    for feat in FEATURES:
        val = user_input[feat]
        imp = fi.get(feat, 0)
        lo, hi, _ = FEATURE_RANGES[feat]
        norm = (val - lo) / (hi - lo)
        mean_norm = (hit_means[feat] - lo) / (hi - lo)
        diff = norm - mean_norm
        contribution = diff * imp
        bar_color = PURPLE if contribution >= 0 else PINK
        impact_text = f"+{contribution:.3f}" if contribution >= 0 else f"{contribution:.3f}"
        impact_color = PURPLE if contribution >= 0 else PINK

        st.markdown(f"""
        <div class="explain-row">
          <div class="explain-label">{FEATURE_LABELS[feat]}</div>
          <div class="explain-bar-wrap">
            <div class="explain-bar" style="width:{norm*100:.0f}%;background:{bar_color};"></div>
          </div>
          <div class="explain-val">{val:.2f}</div>
          <div class="explain-impact" style="color:{impact_color};">{impact_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.72rem;margin-top:0.5rem;'>Impact = feature importance × deviation from average hit profile. Positive = pushes toward hit, negative = pushes away.</div>", unsafe_allow_html=True)

    # ── Gauge ──
    st.divider()
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown(f"<div class='card-title' style='color:{PURPLE};font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;'>Hit Probability Gauge</div>", unsafe_allow_html=True)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 36, "color": prob_color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT_MUTED, "tickfont": {"color": TEXT_MUTED}},
                "bar":  {"color": prob_color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0,  50], "color": "#1a0a2a"},
                    {"range": [50, 75], "color": "#1a0a35"},
                    {"range": [75, 100],"color": "#220a40"},
                ],
                "threshold": {"line": {"color": CYAN, "width": 2}, "thickness": 0.8, "value": 50},
            },
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT},
            margin=dict(l=30, r=30, t=30, b=10), height=240,
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with col_g2:
        st.markdown(f"<div class='card-title' style='color:{PURPLE};font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;'>Confusion Matrix (Test Set)</div>", unsafe_allow_html=True)
        cm = metrics["confusion_matrix"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Not a hit", "Hit"], y=["Not a hit", "Hit"],
            text=[[f"TN: {cm[0][0]}", f"FP: {cm[0][1]}"],
                  [f"FN: {cm[1][0]}", f"TP: {cm[1][1]}"]],
            texttemplate="%{text}<br>%{z}",
            colorscale=[[0, BG3], [1, PURPLE_DARK]],
            showscale=False,
            textfont={"color": TEXT},
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": TEXT_MUTED},
            margin=dict(l=10, r=10, t=10, b=10), height=240,
        )
        st.plotly_chart(fig_cm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Dataset & Model Stats
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"<div class='card-title' style='color:{PURPLE};font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;'>Dataset Overview</div>", unsafe_allow_html=True)

    total = metrics["train_size"] + metrics["test_size"]
    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-bottom:1.5rem;">
      <div class="stat-pill"><div class="stat-num">{total:,}</div><div class="stat-lbl">Total rows</div></div>
      <div class="stat-pill"><div class="stat-num">{metrics['train_size']:,}</div><div class="stat-lbl">Training rows</div></div>
      <div class="stat-pill"><div class="stat-num">{metrics['test_size']:,}</div><div class="stat-lbl">Test rows</div></div>
      <div class="stat-pill"><div class="stat-num">{metrics['hit_rate']:.1%}</div><div class="stat-lbl">Hit rate</div></div>
      <div class="stat-pill"><div class="stat-num">{metrics['hit_threshold']}</div><div class="stat-lbl">Hit threshold</div></div>
      <div class="stat-pill"><div class="stat-num">7</div><div class="stat-lbl">Features used</div></div>
    </div>
    """, unsafe_allow_html=True)

    if df_data is not None:
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown(f"<div class='card-title' style='color:{PURPLE};font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;'>Popularity Distribution</div>", unsafe_allow_html=True)
            fig_pop = go.Figure(go.Histogram(
                x=df_data["popularity"], nbinsx=40,
                marker_color=PURPLE_DARK, marker_line_color=PURPLE, marker_line_width=0.5,
            ))
            fig_pop.add_vline(x=metrics["hit_threshold"], line_color=CYAN,
                              line_dash="dash", annotation_text="Hit threshold",
                              annotation_font_color=CYAN)
            fig_pop.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Popularity", gridcolor="#222244", tickfont=dict(color=TEXT_MUTED)),
                yaxis=dict(title="Count", gridcolor="#222244", tickfont=dict(color=TEXT_MUTED)),
                margin=dict(l=10, r=10, t=10, b=10), height=280,
            )
            st.plotly_chart(fig_pop, use_container_width=True)

        with col_d2:
            st.markdown(f"<div class='card-title' style='color:{PURPLE};font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;'>Hits vs Non-Hits by Feature</div>", unsafe_allow_html=True)
            df_data["hit"] = (df_data["popularity"] >= metrics["hit_threshold"]).astype(int)
            feat_sel = st.selectbox("Select feature", FEATURES,
                                    format_func=lambda x: FEATURE_LABELS[x], label_visibility="collapsed")
            hits     = df_data[df_data["hit"] == 1][feat_sel]
            nonhits  = df_data[df_data["hit"] == 0][feat_sel]
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=nonhits, name="Not a hit", marker_color=PINK,
                                     line_color=PINK, fillcolor=f"rgba(255,107,203,0.15)"))
            fig_box.add_trace(go.Box(y=hits, name="Hit", marker_color=PURPLE,
                                     line_color=PURPLE, fillcolor=f"rgba(155,89,255,0.15)"))
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="#222244", tickfont=dict(color=TEXT_MUTED)),
                xaxis=dict(tickfont=dict(color=TEXT_MUTED)),
                legend=dict(font=dict(color=TEXT_MUTED), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=10, b=10), height=280,
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown(f"<div class='card-title' style='color:{PURPLE};font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-top:1rem;'>Feature Correlations with Popularity</div>", unsafe_allow_html=True)
        corrs = {FEATURE_LABELS[f]: round(df_data[f].corr(df_data["popularity"]), 3) for f in FEATURES}
        corr_df = pd.DataFrame({"Feature": list(corrs.keys()), "Correlation": list(corrs.values())}).sort_values("Correlation")
        colors_corr = [PURPLE if v >= 0 else PINK for v in corr_df["Correlation"]]
        fig_corr = go.Figure(go.Bar(
            x=corr_df["Correlation"], y=corr_df["Feature"], orientation="h",
            marker_color=colors_corr,
            text=[f"{v:+.3f}" for v in corr_df["Correlation"]], textposition="outside",
            textfont=dict(color=TEXT_MUTED, size=11),
        ))
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#222244", tickfont=dict(color=TEXT_MUTED), zeroline=True, zerolinecolor=TEXT_MUTED),
            yaxis=dict(gridcolor="#222244", tickfont=dict(color=TEXT_MUTED)),
            margin=dict(l=10, r=80, t=10, b=10), height=280,
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — How It Works
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">What is this app doing?</div>
      <p style="color:{TEXT};font-size:0.9rem;line-height:1.7;">
        This app predicts whether a Spotify track will be a <strong style="color:{PURPLE};">hit</strong> — 
        defined as a track with a popularity score of <strong style="color:{PURPLE};">70 or above</strong> on Spotify's 0–100 scale.
        It uses 7 audio features extracted directly from the Spotify API to make that prediction.
      </p>
    </div>

    <div class="card">
      <div class="card-title">The Model — XGBoost Classifier</div>
      <p style="color:{TEXT};font-size:0.9rem;line-height:1.7;">
        <strong style="color:{PURPLE};">XGBoost</strong> (Extreme Gradient Boosting) is an ensemble model 
        that builds hundreds of decision trees, each one correcting the mistakes of the previous. 
        It's one of the most effective models for structured/tabular data and is widely used in industry ML pipelines.
      </p>
      <p style="color:{TEXT};font-size:0.9rem;line-height:1.7;margin-top:0.5rem;">
        Key hyperparameters used: <span style="color:{CYAN};">300 estimators</span>, 
        <span style="color:{CYAN};">max depth 5</span>, 
        <span style="color:{CYAN};">learning rate 0.05</span>, 
        <span style="color:{CYAN};">subsampling 0.8</span>. 
        Class imbalance (few hits vs many non-hits) is handled via 
        <span style="color:{CYAN};">scale_pos_weight</span>.
      </p>
    </div>

    <div class="card">
      <div class="card-title">The 7 Audio Features</div>
    </div>
    """, unsafe_allow_html=True)

    for feat in FEATURES:
        lo, hi, _ = FEATURE_RANGES[feat]
        val  = user_input[feat]
        norm = (val - lo) / (hi - lo)
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:1rem;padding:0.6rem 0;border-bottom:1px solid {PURPLE_DARK}22;">
          <div style="min-width:130px;font-weight:700;color:{PURPLE};font-size:0.85rem;">{FEATURE_LABELS[feat]}</div>
          <div style="flex:1;color:{TEXT_MUTED};font-size:0.82rem;line-height:1.5;">{FEATURE_INFO[feat]}</div>
          <div style="min-width:80px;text-align:right;">
            <div style="background:{BG3};border-radius:4px;height:6px;margin-bottom:3px;">
              <div style="width:{norm*100:.0f}%;background:{PURPLE};height:6px;border-radius:4px;"></div>
            </div>
            <div style="font-size:0.72rem;color:{TEXT_MUTED};">current: {val:.2f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="margin-top:1rem;">
      <div class="card-title">Why does this work?</div>
      <p style="color:{TEXT};font-size:0.9rem;line-height:1.7;">
        Hit songs tend to cluster around specific audio profiles — high energy and danceability, 
        moderate valence, low acousticness. XGBoost learns these boundaries from 5,000 stratified 
        training examples and generalises them into a probability score. 
        An <strong style="color:{PURPLE};">AUC-ROC of {metrics['auc_roc']}</strong> means the model 
        correctly ranks a random hit above a random non-hit {metrics['auc_roc']:.0%} of the time.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Pipeline Diagram
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Docker MLOps Pipeline</div>
      <p style="color:{TEXT_MUTED};font-size:0.82rem;margin-bottom:1.2rem;">
        Two Docker containers communicate via a shared named volume. 
        The serving container only starts after training completes successfully.
      </p>

      <div class="pipeline">
        <div class="pipe-step active">
          <div class="pipe-num">Step 1</div>
          <div class="pipe-label">📦 Data</div>
          <div class="pipe-sub">spotify_subset.csv<br>5k stratified rows</div>
          <div class="pipe-arrow">›</div>
        </div>
        <div class="pipe-step active">
          <div class="pipe-num">Step 2</div>
          <div class="pipe-label">⚙️ Training</div>
          <div class="pipe-sub">model-training<br>container</div>
          <div class="pipe-arrow">›</div>
        </div>
        <div class="pipe-step active">
          <div class="pipe-num">Step 3</div>
          <div class="pipe-label">💾 Artifacts</div>
          <div class="pipe-sub">model_exchange<br>Docker volume</div>
          <div class="pipe-arrow">›</div>
        </div>
        <div class="pipe-step active">
          <div class="pipe-num">Step 4</div>
          <div class="pipe-label">🌐 Serving</div>
          <div class="pipe-sub">streamlit container<br>port 8501</div>
          <div class="pipe-arrow">›</div>
        </div>
        <div class="pipe-step active">
          <div class="pipe-num">Step 5</div>
          <div class="pipe-label">🎵 Dashboard</div>
          <div class="pipe-sub">Live prediction<br>+ explainability</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Key Docker Concepts Demonstrated</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.5rem;">
        <div>
          <div style="color:{PURPLE};font-weight:700;font-size:0.85rem;margin-bottom:0.3rem;">Multi-service compose</div>
          <div style="color:{TEXT_MUTED};font-size:0.82rem;line-height:1.5;">Two independent containers defined in a single docker-compose.yml, each with its own role and lifecycle.</div>
        </div>
        <div>
          <div style="color:{PURPLE};font-weight:700;font-size:0.85rem;margin-bottom:0.3rem;">Named volumes</div>
          <div style="color:{TEXT_MUTED};font-size:0.82rem;line-height:1.5;">model_exchange volume persists model artifacts between containers without copying files manually.</div>
        </div>
        <div>
          <div style="color:{PURPLE};font-weight:700;font-size:0.85rem;margin-bottom:0.3rem;">depends_on + healthcheck</div>
          <div style="color:{TEXT_MUTED};font-size:0.82rem;line-height:1.5;">service_completed_successfully ensures serving never starts before training finishes — a real MLOps pattern.</div>
        </div>
        <div>
          <div style="color:{PURPLE};font-weight:700;font-size:0.85rem;margin-bottom:0.3rem;">Environment parity</div>
          <div style="color:{TEXT_MUTED};font-size:0.82rem;line-height:1.5;">Same python:3.11-slim image used for both containers. One requirements.txt. Reproducible anywhere.</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Git Strategy</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
        <div>
          <div style="color:{PURPLE};font-weight:600;font-size:0.82rem;margin-bottom:0.4rem;">✅ Committed to git</div>
          <div style="color:{TEXT_MUTED};font-size:0.8rem;line-height:1.8;">
            src/model_training.py<br>
            src/app.py<br>
            docker-compose.yml<br>
            requirements.txt<br>
            data/spotify_subset.csv
          </div>
        </div>
        <div>
          <div style="color:{PINK};font-weight:600;font-size:0.82rem;margin-bottom:0.4rem;">🚫 Git-ignored</div>
          <div style="color:{TEXT_MUTED};font-size:0.8rem;line-height:1.8;">
            data/spotify_tracks.csv (114k rows)<br>
            exchange/model.joblib<br>
            exchange/scaler.joblib<br>
            venv/<br>
            __pycache__/
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;color:{TEXT_MUTED};font-size:0.75rem;padding:1.5rem 0 0.5rem;
border-top:1px solid {PURPLE_DARK}33;margin-top:1rem;">
  Sruthilaya &nbsp;·&nbsp; MLOps &nbsp;·&nbsp; Northeastern University &nbsp;·&nbsp;
  XGBoost · Docker · Streamlit · Spotify Audio Features
</div>
""", unsafe_allow_html=True)