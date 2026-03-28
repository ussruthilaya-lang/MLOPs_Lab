"""
Streamlit App — AI vs Human Text Detector
"""
import os, sys, json, pickle, warnings, re
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from features import get_phrase_matches

MODEL_DIR    = os.path.join(BASE_DIR, 'models')
SUMMARY_PATH = os.path.join(MODEL_DIR, 'model_summary.json')
RF_PATH      = os.path.join(MODEL_DIR, 'rf_pipeline.pkl')
XGB_PATH     = os.path.join(MODEL_DIR, 'xgb_pipeline.pkl')

st.set_page_config(page_title="AI vs Human Detector", page_icon="🔍", layout="wide")

st.markdown("""
<style>
  section.main > div { padding-top: 1.5rem; }
  .page-title { font-size:2rem; font-weight:800; margin-bottom:0; }
  .page-sub   { opacity:0.55; font-size:0.9rem; margin-top:2px; }
  .section-label {
    font-size:0.72rem; font-weight:700; letter-spacing:0.12em;
    text-transform:uppercase; opacity:0.5; margin-bottom:12px;
  }
  .model-card { border-radius:14px; padding:22px 24px; border:1.5px solid rgba(128,128,128,0.2); }
  .model-card.prod { border-color:#f0a500; }
  .model-badge { display:inline-block; font-size:11px; font-weight:700; letter-spacing:0.08em;
    text-transform:uppercase; padding:3px 10px; border-radius:20px; margin-bottom:10px; }
  .badge-prod { background:rgba(240,165,0,0.15); color:#f0a500; }
  .badge-base { background:rgba(128,128,128,0.12); color:#888; }
  .model-name { font-size:1.15rem; font-weight:700; margin-bottom:4px; }
  .run-id     { font-size:11px; opacity:0.45; margin-bottom:16px; font-family:monospace; }
  .metric-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:10px; }
  .metric-item { text-align:center; }
  .metric-val  { font-size:1.3rem; font-weight:800; }
  .metric-lbl  { font-size:0.7rem; opacity:0.5; text-transform:uppercase; letter-spacing:0.05em; }
  .text-display { border-radius:10px; padding:18px 20px; line-height:1.9; font-size:0.95rem;
    border:1px solid rgba(128,128,128,0.2); }
  .legend-row  { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 4px; }
  .legend-chip { display:inline-flex; align-items:center; gap:5px; border-radius:20px;
    padding:4px 12px; font-size:12px; font-weight:600; border:1px solid; }
  .pred-card   { border-radius:14px; padding:22px; border:2px solid; text-align:center; }
  .pred-card.ai    { border-color:#e05252; background:rgba(224,82,82,0.06); }
  .pred-card.human { border-color:#4caf50; background:rgba(76,175,80,0.06); }
  .pred-label  { font-size:1.5rem; font-weight:800; margin:8px 0 4px; }
  .pred-sub    { font-size:0.8rem; opacity:0.5; }
  .pred-ai     { color:#e05252; }
  .pred-human  { color:#4caf50; }
  .bar-wrap { background:rgba(128,128,128,0.12); border-radius:8px; height:28px; margin:14px 0 6px; overflow:hidden; }
  .bar-fill { height:28px; border-radius:8px; display:flex; align-items:center;
    justify-content:center; color:#fff; font-weight:700; font-size:13px; }
  .explain-box { border-radius:10px; padding:16px 18px;
    border:1px solid rgba(128,128,128,0.2); margin-top:10px; font-size:0.88rem; }
  .explain-row { display:flex; justify-content:space-between; align-items:center; margin:5px 0; }
  .explain-bar-bg { background:rgba(128,128,128,0.12); border-radius:4px; height:8px; flex:1; margin:0 10px; }
  .explain-bar-fg { height:8px; border-radius:4px; }
  .explain-lbl { font-size:11px; opacity:0.6; min-width:130px; }
  .explain-val { font-size:11px; font-weight:700; min-width:36px; text-align:right; }
  .tip-box  { border-radius:10px; padding:16px 20px; border:1px solid rgba(128,128,128,0.2); margin-top:8px; }
  .tip-item { display:flex; gap:10px; margin:8px 0; font-size:0.88rem; align-items:flex-start; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
PHRASE_STYLES = {
    'hedge_words':       ('rgba(255,193,7,0.25)',  '#f0a500', '🟡', 'Hedge Words',    'Transitional words AI overuses'),
    'filler_phrases':    ('rgba(220,53,69,0.2)',   '#e05252', '🔴', 'AI Filler',      'Phrases typical of AI conclusions'),
    'technical_density': ('rgba(13,110,253,0.2)',  '#4dabf7', '🔵', 'Vague Adverbs',  'Imprecise intensifiers AI favors'),
    'passive_voice':     ('rgba(25,135,84,0.2)',   '#51cf66', '🟢', 'Passive Voice',  'Common in human academic writing'),
    'citation_markers':  ('rgba(111,66,193,0.2)',  '#9775fa', '🟣', 'Citations',      'Strong human signal'),
}
SIGNAL_DIR = {
    'hedge_words':       ('AI signal',    '#e05252'),
    'filler_phrases':    ('AI signal',    '#e05252'),
    'technical_density': ('AI signal',    '#e05252'),
    'passive_voice':     ('Human signal', '#4caf50'),
    'citation_markers':  ('Human signal', '#4caf50'),
}

SAMPLE_AI = (
    "We present a unified framework for Physics-Informed Neural Networks (PINNs) applied "
    "to hypersonic flow simulations. By embedding the compressible Navier-Stokes equations "
    "directly into the loss function of the neural network, we solve for shock wave "
    "propagation and boundary layer transition without the need for labeled training data. "
    "Our method achieves high-fidelity resolution of shock discontinuities while reducing "
    "computational cost by two orders of magnitude compared to traditional CFD solvers."
)
SAMPLE_HUMAN = (
    "Tau protein, encoded by the MAPT gene, is a microtubule-associated protein involved "
    "in the regulation of microtubule stability in neurons, contributing to cell shape "
    "maintenance and intracellular transport. Tau is not found as a unique isoform; "
    "different Tau isoforms arise from alternative mRNA splicing of exons 2, 3, and 10. "
    "[14][22] Dysregulation of Tau has been implicated in neurodegenerative diseases "
    "collectively known as tauopathies, including Alzheimer disease and frontotemporal dementia."
)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open(RF_PATH,  'rb') as f: rf  = pickle.load(f)
    with open(XGB_PATH, 'rb') as f: xgb = pickle.load(f)
    with open(SUMMARY_PATH)   as f: s   = json.load(f)
    return rf, xgb, s

rf_model, xgb_model, summary = load_models()
model_info = summary['models']


def highlight_text(text, matches):
    result = text
    for cat, phrases in matches.items():
        bg = PHRASE_STYLES[cat][0]
        for phrase in sorted(phrases, key=len, reverse=True):
            result = re.compile(re.escape(phrase), re.IGNORECASE).sub(
                f'<mark style="background:{bg};padding:1px 5px;border-radius:4px;font-weight:600">{phrase}</mark>',
                result
            )
    return result


def predict(text, model):
    prob = model.predict_proba([text])[0]
    pred = int(np.argmax(prob))
    return pred, prob[pred] * 100, prob


def conf_bar_html(conf, is_ai):
    color = "#e05252" if is_ai else "#4caf50"
    return (
        f'<div class="bar-wrap"><div class="bar-fill" style="width:{conf:.1f}%;background:{color}">'
        f'{conf:.1f}%</div></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:11px;opacity:0.45">'
        f'<span>0%</span><span>50%</span><span>100%</span></div>'
    )


def explain_html(matches, prob_ai):
    rows = ""
    ai_items    = [(k,v) for k,v in matches.items() if SIGNAL_DIR.get(k,('',''))[0]=='AI signal']
    human_items = [(k,v) for k,v in matches.items() if SIGNAL_DIR.get(k,('',''))[0]=='Human signal']
    for cat, phrases in ai_items + human_items:
        _, color, icon, label, desc = PHRASE_STYLES[cat]
        direction, dcolor = SIGNAL_DIR[cat]
        strength = min(len(phrases) * 25, 100)
        phrase_str = ', '.join(f'<em>{p}</em>' for p in phrases)
        sign = '+' if direction == 'AI signal' else '-'
        rows += (
            f'<div class="explain-row">'
            f'<span class="explain-lbl">{icon} {label}</span>'
            f'<div class="explain-bar-bg"><div class="explain-bar-fg" style="width:{strength}%;background:{dcolor}"></div></div>'
            f'<span class="explain-val" style="color:{dcolor}">{sign}{len(phrases)}</span>'
            f'</div>'
            f'<div style="font-size:10px;opacity:0.45;margin:-2px 0 8px 2px">{desc}: {phrase_str}</div>'
        )
    if not rows:
        rows = '<div style="opacity:0.5;font-size:12px">No phrase signals — model using TF-IDF vocabulary patterns.</div>'
    header = f'<div style="font-weight:700;margin-bottom:10px">Why the model predicts this &mdash; <span style="opacity:0.6;font-weight:400">{prob_ai*100:.0f}% AI probability</span></div>'
    return f'<div class="explain-box">{header}{rows}</div>'


def metric_html(val, lbl):
    return f'<div class="metric-item"><div class="metric-val">{val:.4f}</div><div class="metric-lbl">{lbl}</div></div>'


# ══════════════════════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="page-title">🔍 AI vs Human Text Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">MLflow Lab &nbsp;·&nbsp; Research Paper Authenticity &nbsp;·&nbsp; 4,000 academic abstracts &nbsp;·&nbsp; TF-IDF + RF / XGBoost</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Model Comparison ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📊 Model Comparison — MLflow Tracked & Registered</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="medium")

for col, (name, info) in zip([col1, col2], model_info.items()):
    is_prod   = info['alias'] == 'production'
    badge_cls = 'badge-prod' if is_prod else 'badge-base'
    badge_txt = '🏆 Production' if is_prod else '📌 Baseline'
    card_cls  = 'prod' if is_prod else ''
    m = info['metrics']
    auc  = m.get('AUC')  or m.get('auc')  or 0
    acc  = m.get('Accuracy') or m.get('accuracy') or 0
    f1   = m.get('F1')   or m.get('f1')   or 0
    prec = m.get('Precision') or m.get('precision') or 0
    rec  = m.get('Recall') or m.get('recall') or 0
    ne   = info['params'].get('n_estimators', '—')

    with col:
        st.markdown(f"""
        <div class="model-card {card_cls}">
          <span class="model-badge {badge_cls}">{badge_txt}</span>
          <div class="model-name">{name}</div>
          <div class="run-id">alias: {info['alias']} &nbsp;|&nbsp; run: {info['run_id'][:12]}...</div>
          <div class="metric-grid">
            {metric_html(auc,'AUC')}{metric_html(acc,'Accuracy')}{metric_html(f1,'F1 Score')}
          </div>
          <div class="metric-grid">
            {metric_html(prec,'Precision')}{metric_html(rec,'Recall')}
            <div class="metric-item"><div class="metric-val">{ne}</div><div class="metric-lbl">Estimators</div></div>
          </div>
        </div>""", unsafe_allow_html=True)
        with st.expander("⚙️ Full hyperparameters"):
            for k, v in info['params'].items():
                st.markdown(f"**{k}:** `{v}`")

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Test Input ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">🧪 Test Your Text</div>', unsafe_allow_html=True)
st.caption("Paste any academic abstract — both models score it with phrase-level explanation and editing tips")

b1, b2, _ = st.columns([1.2, 1.5, 5])
with b1:
    if st.button("🤖 Load AI Sample"):
        st.session_state['tinput'] = SAMPLE_AI
with b2:
    if st.button("✍️ Load Human Sample"):
        st.session_state['tinput'] = SAMPLE_HUMAN

text_input = st.text_area(
    "Text:", value=st.session_state.get('tinput', ''),
    height=150, placeholder="Paste an academic abstract here...",
    key="tinput", label_visibility="collapsed"
)
run_btn = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

if run_btn and text_input.strip():
    matches = get_phrase_matches(text_input)

    # ── Phrase Analysis ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">📌 Phrase Analysis</div>', unsafe_allow_html=True)
    highlighted = highlight_text(text_input, matches)
    st.markdown(f'<div class="text-display">{highlighted}</div>', unsafe_allow_html=True)

    if matches:
        chips = ""
        for cat, phrases in matches.items():
            _, color, icon, label, _ = PHRASE_STYLES[cat]
            chips += f'<span class="legend-chip" style="border-color:{color};color:{color}">{icon} {label} ({len(phrases)})</span>'
        st.markdown(f'<div class="legend-row">{chips}</div>', unsafe_allow_html=True)
    else:
        st.caption("No strong signal phrases — model using TF-IDF vocabulary patterns.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Predictions ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">🤖 Model Predictions</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2, gap="medium")

    all_results = []
    for col, name, model in zip([p1, p2],
        ["Random Forest (Baseline)", "XGBoost (Production)"],
        [rf_model, xgb_model]
    ):
        pred, conf, prob = predict(text_input, model)
        is_ai   = pred == 1
        all_results.append(is_ai)
        label   = "🤖 AI-Generated" if is_ai else "✍️ Human-Written"
        cls     = "ai" if is_ai else "human"
        col_css = "pred-ai" if is_ai else "pred-human"
        verdict = ("High AI confidence"    if is_ai and conf > 80 else
                   "Leaning AI"            if is_ai else
                   "High Human confidence" if conf > 80 else
                   "Leaning Human")

        # Build all HTML for this column up front
        card_html = (
            f'<div class="pred-card {cls}">'
            f'<div style="font-size:13px;font-weight:600;opacity:0.6">{name}</div>'
            f'<div class="pred-label {col_css}">{label}</div>'
            f'<div class="pred-sub">{verdict}</div>'
            f'{conf_bar_html(conf, is_ai)}'
            f'<div style="font-size:11px;opacity:0.45;margin-top:4px">'
            f'AI: {prob[1]*100:.1f}% &nbsp;|&nbsp; Human: {prob[0]*100:.1f}%'
            f'</div></div>'
            + explain_html(matches, prob[1])
        )
        with col:
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Edit & Retest Tips ────────────────────────────────────────────────────
    st.markdown('<div class="section-label">✏️ Edit &amp; Retest</div>', unsafe_allow_html=True)
    overall_ai = sum(all_results) >= 1
    tips = []
    if overall_ai:
        if 'filler_phrases' in matches:
            tips.append(('🗑️', 'Remove filler phrases: ' + ', '.join(f'<code>{p}</code>' for p in matches['filler_phrases'])))
        if 'hedge_words' in matches:
            tips.append(('✂️', 'Cut hedge words: ' + ', '.join(f'<code>{p}</code>' for p in matches['hedge_words'])))
        if 'technical_density' in matches:
            tips.append(('🔁', 'Replace vague adverbs: ' + ', '.join(f'<code>{p}</code>' for p in matches['technical_density'])))
        tips.append(('📎', 'Add citations like <code>[1]</code> or <code>(Smith, 2023)</code>'))
        tips.append(('🔬', 'Include specific measurements, sample sizes, or method names'))
    else:
        tips.append(('✅', 'Text shows strong human writing characteristics'))
        tips.append(('📊', 'Citations and passive constructions detected — typical of human academic writing'))

    tip_rows = ''.join(f'<div class="tip-item"><span>{icon}</span><span>{text}</span></div>' for icon, text in tips)
    st.markdown(f'<div class="tip-box">{tip_rows}</div>', unsafe_allow_html=True)
    st.caption("👆 Edit the text above and click **Analyze Text** again to see predictions update live.")

elif run_btn:
    st.warning("Please enter some text to analyze.")