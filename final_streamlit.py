import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Aqua", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #f7f6f2;
    color: #1a1a1a;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 3rem 3rem 5rem !important; max-width: 1200px; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #ebebeb;
}
[data-testid="stSidebar"] > div { padding: 2.5rem 1.6rem; }
[data-testid="stSidebar"] label {
    font-size: .72rem !important;
    font-weight: 500 !important;
    letter-spacing: .04em !important;
    color: #888 !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #555 !important; }

/* slider accent */
[data-testid="stSlider"] > div > div > div { background: #e8e8e2 !important; height: 3px !important; }
[data-testid="stSlider"] > div > div > div > div { background: #2d6a4f !important; }

/* ── DIVIDER ── */
hr { border: none; border-top: 1px solid #e8e8e2; margin: 2rem 0; }

/* ── TYPOGRAPHY ── */
.eyebrow {
    font-size: .67rem; font-weight: 500; letter-spacing: .12em;
    text-transform: uppercase; color: #aaa; margin-bottom: .3rem;
}
.serif-title {
    font-family: 'Instrument Serif', serif;
    font-size: 2.8rem; font-weight: 400; color: #1a1a1a;
    line-height: 1.05; letter-spacing: -.02em;
}
.page-sub { font-size: .82rem; color: #bbb; margin-top: .5rem; font-weight: 300; }

/* ── VERDICT CARD ── */
.verdict {
    border-radius: 12px;
    padding: 2rem 2.2rem;
    margin-bottom: .8rem;
}
.verdict.safe   { background: #edf7f1; border: 1px solid #b7dfc9; }
.verdict.unsafe { background: #fdf2f2; border: 1px solid #f0c4c4; }

.verdict-label {
    font-family: 'Instrument Serif', serif;
    font-size: 2.1rem; font-weight: 400; line-height: 1;
}
.verdict.safe   .verdict-label { color: #1a6640; }
.verdict.unsafe .verdict-label { color: #8b2020; }

.verdict-prob {
    font-size: .75rem; font-weight: 400;
    margin-top: .75rem; letter-spacing: .01em;
}
.verdict.safe   .verdict-prob { color: #5a9e7a; }
.verdict.unsafe .verdict-prob { color: #c06060; }

/* ── STAT GRID ── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
.stat-box {
    background: #fff;
    border: 1px solid #ebebeb;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
}
.stat-box.full { grid-column: span 2; }
.s-label {
    font-size: .63rem; font-weight: 500; letter-spacing: .1em;
    text-transform: uppercase; color: #bbb; margin-bottom: .4rem;
}
.s-value {
    font-size: 1.65rem; font-weight: 300;
    color: #1a1a1a; letter-spacing: -.03em; line-height: 1;
}
.s-value .unit { font-size: 1rem; color: #ccc; margin-left: 2px; }
.s-value.green { color: #1a6640; }
.s-value.red   { color: #8b2020; }

/* ── PREDICTION READOUT ── */
.readout {
    background: #fff; border: 1px solid #ebebeb; border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-family: 'Inter', monospace; font-size: .73rem;
    color: #bbb; line-height: 2;
}
.readout .rk { color: #ccc; display: inline-block; width: 130px; }
.readout .rv { color: #1a1a1a; font-weight: 500; }
.readout .rg { color: #1a6640; font-weight: 500; }
.readout .rr { color: #8b2020; font-weight: 500; }

/* ── TABLE ── */
.clean-table { width: 100%; border-collapse: collapse; font-size: .76rem; }
.clean-table th {
    text-align: left; padding: .5rem .6rem .5rem 0;
    font-size: .62rem; font-weight: 500; letter-spacing: .09em;
    text-transform: uppercase; color: #bbb;
    border-bottom: 1px solid #ebebeb;
}
.clean-table td { padding: .55rem .6rem .55rem 0; border-bottom: 1px solid #f3f3f0; color: #888; }
.clean-table td:first-child { color: #1a1a1a; font-weight: 400; }
.clean-table tr:last-child td { border-bottom: none; }

/* ── PARAM PILLS ── */
.pill-row { display: flex; flex-wrap: wrap; gap: 6px; }
.pill {
    background: #fff; border: 1px solid #e8e8e2; border-radius: 6px;
    padding: .38rem .75rem; font-size: .72rem;
}
.pill .pn { color: #bbb; margin-right: .35rem; }
.pill .pv { color: #1a1a1a; font-weight: 500; }
.pill .pu { color: #ccc; font-size: .63rem; margin-left: .2rem; }

/* ── SECTION LABEL ── */
.sec { font-size: .63rem; font-weight: 500; letter-spacing: .1em; text-transform: uppercase; color: #bbb; margin-bottom: .9rem; }
</style>
""", unsafe_allow_html=True)


# ── MODEL TRAINING ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="training…")
def build_model(path="water_potability.csv"):
    df   = pd.read_csv(path)
    imp  = SimpleImputer(strategy="median")
    data = pd.DataFrame(imp.fit_transform(df), columns=df.columns)

    X = data.drop("Potability", axis=1)
    y = data["Potability"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc  = StandardScaler()
    Xts = sc.fit_transform(X_tr)
    Xvs = sc.transform(X_te)

    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(Xts, y_tr)

    y_pred = clf.predict(Xvs)
    y_prob = clf.predict_proba(Xvs)[:, 1]

    raw = classification_report(y_te, y_pred, target_names=["Non-Potable", "Potable"], output_dict=True)
    rpt = {"0": raw["Non-Potable"], "1": raw["Potable"],
           "macro avg": raw["macro avg"], "weighted avg": raw["weighted avg"]}

    return (
        clf, sc, data, list(X.columns),
        float(accuracy_score(y_te, y_pred)),
        rpt,
        float(roc_auc_score(y_te, y_prob)),
        confusion_matrix(y_te, y_pred).tolist(),
        {c: {"min": float(data[c].min()), "max": float(data[c].max()), "median": float(data[c].median())} for c in X.columns},
        {c: float(v) for c, v in zip(X.columns, clf.feature_importances_)},
    )

try:
    clf, sc, df_clean, features, accuracy, rpt, auc_score, cm_list, stats, feat_imp = build_model()
    loaded = True
except FileNotFoundError:
    loaded = False

PMETA = {
    "ph":              ("pH",          "pH"),
    "Hardness":        ("Hardness",    "mg/L"),
    "Solids":          ("TDS",         "ppm"),
    "Chloramines":     ("Chloramines", "ppm"),
    "Sulfate":         ("Sulfate",     "mg/L"),
    "Conductivity":    ("Conduct.",    "μS/cm"),
    "Organic_carbon":  ("Org. C",      "ppm"),
    "Trihalomethanes": ("THMs",        "μg/L"),
    "Turbidity":       ("Turbidity",   "NTU"),
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Adjust Parameters")
    st.markdown("---")
    user_input = {}
    if loaded:
        for feat in features:
            lbl, unit = PMETA.get(feat, (feat, ""))
            mn  = round(stats[feat]["min"],    2)
            mx  = round(stats[feat]["max"],    2)
            med = round(stats[feat]["median"], 2)
            user_input[feat] = st.slider(f"{lbl} ({unit})", mn, mx, med, round((mx-mn)/200, 4))

if not loaded:
    st.error("`water_potability.csv` not found.")
    st.stop()

# ── PREDICT ───────────────────────────────────────────────────────────────────
pred     = int(clf.predict(sc.transform(pd.DataFrame([user_input])))[0])
proba    = clf.predict_proba(sc.transform(pd.DataFrame([user_input])))[0]
pot_pct  = float(proba[1]) * 100
npot_pct = float(proba[0]) * 100
is_safe  = pred == 1
vcls     = "safe" if is_safe else "unsafe"
vtxt     = "Safe to Drink" if is_safe else "Not Safe"
vsym     = "💧" if is_safe else "⚠️"

tn = cm_list[0][0]; fp = cm_list[0][1]
fn = cm_list[1][0]; tp = cm_list[1][1]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="eyebrow">SDG 6 · Water Quality Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="serif-title">Aqua</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Random Forest · 150 trees · 9 parameters · 80/20 split</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── LAYOUT ────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 1.1], gap="large")

# ── LEFT ──────────────────────────────────────────────────────────────────────
with col_l:

    # Verdict
    st.markdown(f"""
    <div class="verdict {vcls}">
      <div class="verdict-label">{vsym} {vtxt}</div>
      <div class="verdict-prob">
        Potable {pot_pct:.1f}% &nbsp;·&nbsp; Non-Potable {npot_pct:.1f}%
      </div>
    </div>""", unsafe_allow_html=True)

    # Prediction readout
    rk = "rg" if is_safe else "rr"
    st.markdown(f"""
    <div class="readout">
      <span class="rk">prediction</span><span class="{rk}">{vtxt}</span><br>
      <span class="rk">confidence</span><span class="rv">{max(pot_pct, npot_pct):.2f}%</span><br>
      <span class="rk">class label</span><span class="rv">{pred}</span><br>
      <span class="rk">p(potable)</span><span class="rv">{pot_pct:.4f}%</span><br>
      <span class="rk">p(non-potable)</span><span class="rv">{npot_pct:.4f}%</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Parameter pills
    st.markdown('<div class="sec">Sample Values</div>', unsafe_allow_html=True)
    pills = '<div class="pill-row">'
    for feat in features:
        lbl, unit = PMETA.get(feat, (feat, ""))
        pills += f'<div class="pill"><span class="pn">{lbl}</span><span class="pv">{user_input[feat]:.2f}</span><span class="pu">{unit}</span></div>'
    pills += '</div>'
    st.markdown(pills, unsafe_allow_html=True)


# ── RIGHT ─────────────────────────────────────────────────────────────────────
with col_r:

    st.markdown('<div class="sec">Model Performance</div>', unsafe_allow_html=True)

    # Top stats
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-box">
        <div class="s-label">Accuracy</div>
        <div class="s-value">{accuracy*100:.2f}<span class="unit">%</span></div>
      </div>
      <div class="stat-box">
        <div class="s-label">ROC-AUC</div>
        <div class="s-value">{auc_score:.4f}</div>
      </div>
      <div class="stat-box">
        <div class="s-label">F1 · Potable</div>
        <div class="s-value">{rpt['1']['f1-score']*100:.1f}<span class="unit">%</span></div>
      </div>
      <div class="stat-box">
        <div class="s-label">F1 · Non-Potable</div>
        <div class="s-value">{rpt['0']['f1-score']*100:.1f}<span class="unit">%</span></div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Classification report table
    st.markdown('<div class="sec">Classification Report</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <table class="clean-table">
      <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>n</th></tr>
      <tr>
        <td>Non-Potable</td>
        <td>{rpt['0']['precision']:.3f}</td><td>{rpt['0']['recall']:.3f}</td>
        <td>{rpt['0']['f1-score']:.3f}</td><td>{int(rpt['0']['support'])}</td>
      </tr>
      <tr>
        <td>Potable</td>
        <td>{rpt['1']['precision']:.3f}</td><td>{rpt['1']['recall']:.3f}</td>
        <td>{rpt['1']['f1-score']:.3f}</td><td>{int(rpt['1']['support'])}</td>
      </tr>
      <tr>
        <td>Macro avg</td>
        <td>{rpt['macro avg']['precision']:.3f}</td><td>{rpt['macro avg']['recall']:.3f}</td>
        <td>{rpt['macro avg']['f1-score']:.3f}</td><td>{int(rpt['macro avg']['support'])}</td>
      </tr>
    </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion matrix
    st.markdown('<div class="sec">Confusion Matrix</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-box" style="text-align:center">
        <div class="s-label">True Negative</div>
        <div class="s-value green">{tn}</div>
      </div>
      <div class="stat-box" style="text-align:center">
        <div class="s-label">False Positive</div>
        <div class="s-value red">{fp}</div>
      </div>
      <div class="stat-box" style="text-align:center">
        <div class="s-label">False Negative</div>
        <div class="s-value red">{fn}</div>
      </div>
      <div class="stat-box" style="text-align:center">
        <div class="s-label">True Positive</div>
        <div class="s-value green">{tp}</div>
      </div>
    </div>""", unsafe_allow_html=True)


# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="sec">Feature Importance</div>', unsafe_allow_html=True)

imp_sorted = sorted(feat_imp.items(), key=lambda x: x[1])
labels     = [PMETA.get(f, (f, ""))[0] for f, _ in imp_sorted]
values     = [v for _, v in imp_sorted]
colors     = ["#2d6a4f" if v == max(values) else "#d4e9de" for v in values]

fig = go.Figure(go.Bar(
    x=values, y=labels, orientation="h",
    marker_color=colors, marker_line_width=0,
    hovertemplate="<b>%{y}</b>  %{x:.4f}<extra></extra>",
))
fig.update_layout(
    height=280,
    paper_bgcolor="#f7f6f2", plot_bgcolor="#f7f6f2",
    font=dict(family="Inter, sans-serif", size=11, color="#aaa"),
    margin=dict(l=0, r=0, t=4, b=4),
    xaxis=dict(
        showgrid=True, gridcolor="#ebebeb", gridwidth=1,
        zeroline=False, tickfont=dict(size=10, color="#ccc"),
        title=None,
    ),
    yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#888")),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("""
<p style="font-size:.7rem;color:#ccc;text-align:center;margin-top:1rem">
  Aqua · SDG 6 Clean Water &amp; Sanitation · Random Forest Classifier
</p>""", unsafe_allow_html=True)
