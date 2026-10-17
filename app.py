"""
Time-Aware Temporal Transformer (TATT) for early sepsis prediction: results dashboard and patient simulator.
Run locally:   streamlit run app.py
All model inference is performed in NumPy (no PyTorch needed at deployment).
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import curves as C
import data as D
from tatt_numpy import TATTNumpy, tape

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("TATT_RESULTS", os.path.join(BASE, "results"))
MODEL_PATH = os.path.join(BASE, "model", "tatt_weights.npz")
NAVY, ORANGE, GREEN, GREY, RED, AMBER, BLUE, LIGHT = "#1F3A5F", "#C8551B", "#3E7C4A", "#6B6B6B", "#A12727", "#B7791F", "#2F5D8C", "#D9DEE5"
MC = {"TATT": NAVY, "Bi-LSTM": ORANGE, "XGBoost": GREEN}
MODELS = ["TATT", "Bi-LSTM", "XGBoost"]
VN, UN = D.VAR_NAMES, D.UNITS
MODCOL = {"Vital": BLUE, "Lab": AMBER, "Intervention": GREY}

st.set_page_config(page_title="TATT Sepsis Early Warning", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'IBM Plex Sans', sans-serif; }
h1 { font-weight: 600; font-size: 1.85rem; color: #12263A; letter-spacing: -0.01em; }
h2, h3 { font-weight: 600; color: #12263A; }
.block-container { padding-top: 2rem; max-width: 1280px; }
[data-testid="stMetricValue"] { font-size: 1.55rem; color: #12263A; }
[data-testid="stSidebar"] { border-right: 1px solid #D5DCE3; }
.note { border-left: 3px solid #1F3A5F; background: #F2F5F8; padding: .7rem 1rem; margin: .6rem 0 1rem 0; font-size: .92rem; }
.warn { border-left: 3px solid #B7791F; background: #FBF6EC; padding: .7rem 1rem; margin: .6rem 0 1rem 0; font-size: .92rem; }
.alert { border-left: 4px solid #A12727; background: #FBEDEC; padding: .8rem 1rem; font-weight: 500; }
.ok { border-left: 4px solid #1F3A5F; background: #EAEFF5; padding: .8rem 1rem; font-weight: 500; }
</style>""", unsafe_allow_html=True)


def show(fig, h=None):
    fig.update_layout(template="simple_white", font=dict(family="IBM Plex Sans", size=13, color="#12263A"),
                      margin=dict(l=55, r=20, t=30, b=50), legend=dict(orientation="h", y=-0.22))
    if h:
        fig.update_layout(height=h)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def note(txt, kind="note"):
    st.markdown(f'<div class="{kind}">{txt}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------ assets
@st.cache_resource(show_spinner="Loading model and results")
def load_assets():
    res = json.load(open(os.path.join(RES, "results.json")))
    pats = pickle.load(open(os.path.join(RES, "sample_patients.pkl"), "rb"))
    cv = dict(np.load(os.path.join(RES, "curves.npz")))
    return {"res": res, "pats": pats, "curves": cv, "model": TATTNumpy(MODEL_PATH)}


try:
    A = load_assets()
except Exception as e:  # missing files
    st.error(f"Could not load the model and results from '{RES}'. {e}")
    st.stop()
R, PATS, CV, MODEL = A["res"], A["pats"], A["curves"], A["model"]
SEP = [p for p in PATS if not np.isnan(p["onset"])]
NON = [p for p in PATS if np.isnan(p["onset"])]
Y = CV["y"]; SC = {"TATT": CV["TATT"], "Bi-LSTM": CV["Bi_LSTM"], "XGBoost": CV["XGBoost"]}
PREV = R["prevalence"]; THR = R["threshold"]; WTS = C.prevalence_weights(Y, PREV)
M, W = R["metrics"], R["within_septic"]


def wins(p, times):
    return [D.encode_window(p, float(t), MODEL.window, MODEL.max_len) for t in times]


def risk_at(p, times):
    w = wins(p, times); out = []
    for i in range(0, len(w), 64):
        out.append(MODEL.predict(w[i:i + 64]))
    return np.concatenate(out) if out else np.zeros(0)


def explain_at(p, t):
    w = wins(p, [t]); prob, va, imp, _ = MODEL.predict(w, return_attn=True)
    return float(prob[0]), va[0], w[0], imp[0][:len(w[0]["var"])]


def op_point(k):
    p = SC[k] >= THR[k]; tp = int((p & (Y == 1)).sum()); fp = int((p & (Y == 0)).sum()); fn = int((~p & (Y == 1)).sum()); tn = int((~p & (Y == 0)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=pr, recall=rc, specificity=sp, f1=2 * pr * rc / max(pr + rc, 1e-9))


def ci(d, k):
    return f"{d[k]:.3f} ({d[k + '_ci'][0]:.3f}-{d[k + '_ci'][1]:.3f})"


def label(p):
    return f"{p['id']}  ({'sepsis' if not np.isnan(p['onset']) else 'no sepsis'})"


PAGES = ["Overview", "Cohort and data", "Preprocessing", "Model architecture", "Training", "Evaluation", "Interpretability", "Patient simulator", "Methods and limitations"]
st.sidebar.markdown("### TATT Sepsis Early Warning")
page = st.sidebar.radio("Section", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(f"Data: {R['source']}")
st.sidebar.caption(f"{R['n_patients']:,} patients | sepsis prevalence {PREV:.1%}")
st.sidebar.caption("Research prototype. Not validated for clinical use.")

# ============================================================================================ OVERVIEW
if page == "Overview":
    st.title("Early prediction of sepsis in ICU patients")
    st.markdown("**Time-Aware Temporal Transformer (TATT) with a 4-to-6-hour early warning horizon: results and patient simulator**")
    k = st.columns(4)
    k[0].metric("TATT test AUROC", f"{M['TATT']['auroc']:.3f}", f"{M['TATT']['auroc'] - M['XGBoost']['auroc']:+.3f} vs XGBoost")
    k[1].metric("TATT AUPRC (prevalence-adjusted)", f"{M['TATT']['auprc_adj']:.3f}", f"{M['TATT']['auprc_adj'] - M['XGBoost']['auprc_adj']:+.3f} vs XGBoost")
    k[2].metric("TATT within-patient AUROC", f"{W['TATT']['auroc']:.3f}")
    k[3].metric("False alerts per 1,000 h", f"{R['alarm_burden']['TATT']['false_alerts_per_1000h']:.0f}")
    note("<b>Principal finding.</b> On the held-out test partition the TATT was significantly less discriminative than the XGBoost and Bi-LSTM baselines, and its prevalence-adjusted AUPRC stayed below the 0.35 clinical signal threshold at every lead time. "
         "A control that uses only the position of a window within the ICU stay scored higher than every model (AUROC %.3f), which is why the evaluation sections report within-patient tests as well." % R["controls"]["time_only"]["auroc"])
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(pd.DataFrame({
            "Objective": ["(i) Extract, preprocess and harmonise irregular multimodal data", "(ii) Time-aware positional encoding and time-decay self-attention",
                          "(iii) Optimise for a 4-to-6-hour horizon (label shifting, negative windows, Focal Loss)", "(iv) Compare with XGBoost and Bi-LSTM; validate attention clinically"],
            "Dashboard section": ["Cohort and data; Preprocessing", "Model architecture; Training", "Training; Evaluation (lead time, alarm burden)", "Evaluation; Interpretability"]}), hide_index=True)
        st.markdown("Use **Patient simulator** to enter a patient's values, upload observations, or replay a held-out test patient and see the model's risk score, its trajectory, and the variables the model attended to.")
    with c2:
        st.graphviz_chart("""
digraph G { rankdir=TB; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#4A5560", fillcolor="#F2F5F8"];
 a [label="Hourly ICU records\\n(vitals, laboratory results)"];
 b [label="Timestamped observation tokens\\n+ missing-laboratory mask tokens", fillcolor="#E6EEF2"];
 c [label="TATT: time-aware encoding\\n+ time-decay self-attention\\n+ pooling over final 6 h", fillcolor="#CFE0D5"];
 d [label="Sepsis-3 risk and attention weights"];
 e [label="Early-warning alert\\n(4 to 6 h before onset)", fillcolor="#F3D5D2"];
 a -> b -> c -> d -> e; }""")

# ============================================================================================ COHORT
elif page == "Cohort and data":
    st.title("Cohort and data (Objective i)")
    co = R["cohort"]; tv = co["tokens_by_variable"]; tot = sum(tv.values()); sp = R["split"]
    k = st.columns(4)
    k[0].metric("Patients", f"{R['n_patients']:,}"); k[1].metric("Sepsis prevalence", f"{PREV:.1%}")
    k[2].metric("Median stay (h)", f"{np.median(co['stay_hours']):.0f}"); k[3].metric("Median tokens per patient", f"{np.median(co['tokens_per_patient']):.0f}")
    l, r = st.columns(2)
    with l:
        sepn = {"train": 534, "val": 114, "test": 115}
        df = pd.DataFrame({"Partition": ["Training (70%)", "Validation (15%)", "Test (15%)"], "Patients": [sp[x] for x in ("train", "val", "test")],
                           "Windows": [R["windows"][x]["n"] for x in ("train", "val", "test")], "Positive windows": [R["windows"][x]["pos"] for x in ("train", "val", "test")]})
        df["Negative : positive"] = ((df["Windows"] - df["Positive windows"]) / df["Positive windows"]).round(1)
        st.markdown("**Patient-level partitions** (stratified on sepsis status; windows generated after splitting)")
        st.dataframe(df, hide_index=True)
        st.markdown("**Criteria applied**")
        st.dataframe(pd.DataFrame({"Type": ["Inclusion", "Inclusion", "Not applied"], "Criterion": ["ICU record of at least 48 hourly rows", "Sepsis label and ICU length of stay available", "Restrictions on age, sex or pregnancy"]}), hide_index=True)
    with r:
        fig = go.Figure()
        for v, lab, c in ((np.sort(co["stay_hours"]), "ICU stay length (all patients)", BLUE), (np.sort(co["onset_hours"]), "Sepsis-3 onset (septic patients)", RED)):
            fig.add_trace(go.Scatter(x=v, y=np.arange(1, len(v) + 1) / len(v), mode="lines", line=dict(shape="hv", color=c), name=lab))
        fig.add_vline(x=24, line=dict(color=GREY, dash="dot")); fig.update_xaxes(range=[0, 200], title="Hours from ICU admission"); fig.update_yaxes(title="Cumulative proportion"); show(fig, 340)
        st.caption("Most stays end within about 60 hours, whereas most sepsis onsets occur after that point. Septic patients are therefore observed over much longer stays, which allows time-in-stay to act as a shortcut (see Evaluation, Controls).")
    a, b = st.columns(2)
    with a:
        ks = [x for x in VN if tv[x] > 0]
        fig = go.Figure(go.Bar(x=ks, y=[tv[x] for x in ks], marker_color=[MODCOL[D.MODALITY[D.VAR_IDX[x]]] for x in ks])); fig.update_yaxes(type="log", title="Observations (log scale)"); show(fig, 320)
        st.caption("Blue: vital signs. Amber: laboratory results. Intervention flags have no observations in this database.")
    with b:
        fig = go.Figure(go.Histogram(x=co["tokens_per_patient"], nbinsx=50, marker_color=BLUE)); fig.update_xaxes(title="Observation tokens per patient"); fig.update_yaxes(title="Patients"); show(fig, 320)
    nm = ["Training", "Validation", "Test"]; ps = [R["windows"][x]["pos"] for x in ("train", "val", "test")]; ng = [R["windows"][x]["n"] - R["windows"][x]["pos"] for x in ("train", "val", "test")]
    fig = go.Figure([go.Bar(x=nm, y=ng, name="Negative windows", marker_color=BLUE), go.Bar(x=nm, y=ps, name="Positive windows", marker_color=RED)]); fig.update_layout(barmode="stack"); fig.update_yaxes(title="Windows"); show(fig, 300)

# ============================================================================================ PREPROCESSING
elif page == "Preprocessing":
    st.title("Preprocessing and harmonisation (Objective i)")
    st.markdown("Each measurement becomes a timestamped token. Nothing is interpolated or forward-filled for the TATT; laboratories absent from the window are represented by an explicit mask token.")
    p = PATS[[label(x) for x in PATS].index(st.selectbox("Held-out test patient", [label(x) for x in PATS]))]
    vs = st.multiselect("Variables", [v for v in VN if D.MODALITY[D.VAR_IDX[v]] != "Intervention"], default=["HR", "MAP", "Lactate"])
    if vs:
        fig = make_subplots(rows=len(vs), cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=[f"{v} ({UN[v]})" for v in vs])
        for i, v in enumerate(vs, 1):
            j = D.VAR_IDX[v]; m = p["var"] == j; mod = D.MODALITY[j]
            fig.add_trace(go.Scatter(x=p["time"][m], y=p["val"][m], mode="lines+markers" if mod == "Vital" else "markers", marker=dict(size=5 if mod == "Vital" else 9, color=MODCOL[mod]), line=dict(width=1, color=MODCOL[mod]), showlegend=False), row=i, col=1)
            if not np.isnan(p["onset"]):
                fig.add_vrect(x0=p["onset"] - 6, x1=p["onset"] - 4, fillcolor=RED, opacity=.12, line_width=0, row=i, col=1); fig.add_vline(x=p["onset"], line=dict(color=RED, dash="dash"), row=i, col=1)
        fig.update_xaxes(title_text="Hours from ICU admission", row=len(vs), col=1); show(fig, 230 * len(vs) + 60)
        st.caption("Native, irregular observations. The shaded band is the 4-to-6-hour positive window and the dashed line is Sepsis-3 onset.")
    st.subheader("Native observations versus an hourly last-observation-carried-forward grid")
    c1, c2 = st.columns([1, 2])
    with c1:
        v = st.selectbox("Variable", ["Lactate", "Creatinine", "Platelets", "WBC", "BUN", "Bilirubin", "HR", "MAP", "RR", "SpO2", "SBP", "DBP"])
        t_end = st.slider("Window end (hours)", float(D.MIN_PRED_TIME), float(p["T"]), float(min(p["T"], 48.0)), 0.5)
    j = D.VAR_IDX[v]; grid = D.encode_grid(p, t_end, 24.0); ends = t_end - 24 + np.arange(24) + 1.0; locf = grid[:, j] * D._STD[j] + D._MEAN[j]
    m = (p["var"] == j) & (p["time"] <= t_end) & (p["time"] > t_end - 24); first_t = p["time"][p["var"] == j].min() if (p["var"] == j).any() else np.inf
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ends, y=np.where(ends >= first_t, locf, np.nan), mode="lines", line=dict(shape="hv", color=ORANGE, dash="dot"), name="Hourly grid with LOCF (Bi-LSTM input)"))
        fig.add_trace(go.Scatter(x=p["time"][m], y=p["val"][m], mode="markers", marker=dict(color=AMBER, size=9), name="Native observations (TATT input)"))
        fig.update_xaxes(title="Hours"); fig.update_yaxes(title=f"{v} ({UN[v]})"); show(fig, 330)
    w = D.encode_window(p, t_end, MODEL.window, MODEL.max_len)
    st.markdown(f"**Token sequence for this window** ({len(w['var'])} tokens; flag 0 marks an explicit missing-laboratory token; times are hours since window start)")
    dt = np.r_[0, np.diff(w["time"])]
    st.dataframe(pd.DataFrame({"Variable": [VN[i] for i in w["var"]], "Value (normalised)": w["val"].round(3), "Time in window (h)": w["time"].round(2), "Delta t (h)": dt.round(3), "Observed flag": w["flag"].astype(int)}).tail(15), hide_index=True)
    rate = np.array([[(x["var"] == j).sum() / x["T"] for j in range(15)] for x in PATS])
    fig = go.Figure(go.Heatmap(z=np.log10(rate + 0.01), x=VN, y=[x["id"] for x in PATS], colorscale="Viridis", zmin=-2, zmax=0.3, colorbar=dict(title="log10 obs/h")))
    fig.update_yaxes(showticklabels=False, title="Test patients (sample)"); show(fig, 380)
    st.caption("Observation rate per variable. Vital signs are charted almost hourly; laboratory variables are sparse; the three intervention variables are empty in this database.")

# ============================================================================================ ARCHITECTURE
elif page == "Model architecture":
    st.title("Model architecture (Objective ii)")
    st.graphviz_chart("""
digraph T { rankdir=TB; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#4A5560", fillcolor="#F2F5F8"];
 i [label="Observation triplets (variable, value, time) + missingness flag"];
 e [label="Multimodal input embedding: linear projection to d_model (Eq 3.1)", fillcolor="#E6EEF2"];
 p [label="Time-aware positional encoding on elapsed time within the window (Eq 3.2, 3.3)", fillcolor="#CFE0D5"];
 a [label="Time-decay self-attention: softmax(QK^T / sqrt(dk) - alpha * dT) V (Eq 3.4)", fillcolor="#CFE0D5"];
 f [label="Feed-forward (GELU) + layer normalisation, repeated for N layers"];
 g [label="Global average pooling over the final 6 h"];
 h [label="MLP head (GELU, dropout 0.3)"];
 o [label="Sepsis probability", fillcolor="#F3D5D2"];
 i -> e -> p -> a -> f -> g -> h -> o; }""")
    al = MODEL.alphas(); cfg = MODEL.cfg
    st.dataframe(pd.DataFrame({"Setting": ["Layers", "Attention heads", "Model dimension", "Trainable parameters", "Context window (h)", "Maximum tokens", "Learned alpha (layer 1, layer 2)"],
                               "Value": [cfg["n_layers"], cfg["n_heads"], cfg["d_model"], f"{MODEL.n_params():,}", cfg["window"], MODEL.max_len, f"{al[0]:.4f}, {al[1]:.4f}"]}), hide_index=True)
    st.subheader("Time-aware positional encoding")
    t = np.arange(0, 24.01, 0.05); pe = tape(t[None], 32)[0].T
    fig = go.Figure(go.Heatmap(z=pe, x=t, colorscale="RdBu_r", zmid=0, colorbar=dict(title="Value"))); fig.update_xaxes(title="Hours since start of context window"); fig.update_yaxes(title="Embedding dimension", autorange="reversed"); show(fig, 320)
    st.subheader("Time-decay attention explorer")
    st.markdown("The bias -alpha x dT multiplies each attention weight by exp(-alpha x time gap), up to row renormalisation. The charts use the trained first-layer weights on a real patient window.")
    c1, c2 = st.columns([1, 2])
    with c1:
        alpha = st.slider("alpha (per hour)", 0.0, 1.0, float(round(al[0], 3)), 0.005)
        p = PATS[[label(x) for x in PATS].index(st.selectbox("Patient", [label(x) for x in PATS], key="arch_p"))]
        t_end = st.slider("Window end (h)", float(D.MIN_PRED_TIME), float(p["T"]), float(min(p["T"], 48.0)), 0.5, key="arch_t")
    with c2:
        g = np.linspace(0, 24, 200); fig = go.Figure()
        fig.add_trace(go.Scatter(x=g, y=np.exp(-alpha * g), line=dict(color=NAVY, width=3), name=f"alpha = {alpha:.3f}"))
        for i, a_ in enumerate(al, 1): fig.add_trace(go.Scatter(x=g, y=np.exp(-a_ * g), line=dict(dash="dash"), name=f"Learned, layer {i} ({a_:.4f})"))
        fig.update_xaxes(title="Time gap between observations (h)"); fig.update_yaxes(title="exp(-alpha x gap)", range=[0, 1.02]); show(fig, 300)
    w = D.encode_window(p, t_end, MODEL.window, MODEL.max_len)
    a_no = MODEL.first_layer_attention(w, 0.0); a_sel = MODEL.first_layer_attention(w, alpha); a_lrn = MODEL.first_layer_attention(w, None)
    tm = w["time"]; dT = np.abs(tm[:, None] - tm[None, :]); edges = [0, 0.5, 1, 2, 4, 8, 24.01]; labs = [f"{edges[i]:g}-{min(edges[i + 1], 24):g}" for i in range(len(edges) - 1)]
    mean_by = lambda a: [float(a[(dT >= edges[i]) & (dT < edges[i + 1])].mean()) if ((dT >= edges[i]) & (dT < edges[i + 1])).any() else 0.0 for i in range(len(edges) - 1)]
    fig = go.Figure([go.Bar(x=labs, y=mean_by(a_no), name="No decay", marker_color=LIGHT), go.Bar(x=labs, y=mean_by(a_lrn), name="Learned", marker_color=NAVY), go.Bar(x=labs, y=mean_by(a_sel), name=f"alpha = {alpha:.3f}", marker_color=ORANGE)])
    fig.update_layout(barmode="group"); fig.update_xaxes(title="Time-gap bin (h)"); fig.update_yaxes(title="Mean attention weight (layer 0)"); show(fig, 320)

# ============================================================================================ TRAINING
elif page == "Training":
    st.title("Training (Objectives ii and iii)")
    h = R["history"]; bt = int(np.argmax(h["TATT"]["val_auprc"])); bb = int(np.argmax(h["Bi-LSTM"]["val_auprc"]))
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure([go.Scatter(x=h[k]["epoch"], y=h[k]["train_loss"], name=k, line=dict(color=MC[k])) for k in ("TATT", "Bi-LSTM")]); fig.update_xaxes(title="Epoch"); fig.update_yaxes(title="Training Focal Loss"); show(fig, 320)
    with c2:
        fig = go.Figure()
        for k, b in (("TATT", bt), ("Bi-LSTM", bb)):
            fig.add_trace(go.Scatter(x=h[k]["epoch"], y=h[k]["val_auprc"], name=k, line=dict(color=MC[k])))
            fig.add_trace(go.Scatter(x=[h[k]["epoch"][b]], y=[h[k]["val_auprc"][b]], mode="markers", marker=dict(symbol="star", size=13, color=MC[k]), showlegend=False))
        fig.add_hline(y=PREV, line=dict(color=GREY, dash="dot"), annotation_text="Cohort prevalence"); fig.update_xaxes(title="Epoch"); fig.update_yaxes(title="Validation AUPRC"); show(fig, 320)
    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure(go.Scatter(x=h["TATT"]["epoch"], y=h["TATT"]["alpha"], line=dict(color=NAVY))); fig.add_hline(y=0.05, line=dict(color=GREY, dash="dash"), annotation_text="Initial value")
        fig.update_xaxes(title="Epoch"); fig.update_yaxes(title="Mean learned alpha"); show(fig, 300)
    with c4:
        fig = go.Figure([go.Scatter(x=h[k]["epoch"], y=h[k]["val_auroc"], name=k, line=dict(color=MC[k])) for k in ("TATT", "Bi-LSTM")]); fig.add_hline(y=.5, line=dict(color=GREY, dash="dot")); fig.update_xaxes(title="Epoch"); fig.update_yaxes(title="Validation AUROC"); show(fig, 300)
    st.dataframe(pd.DataFrame({"Model": ["TATT", "Bi-LSTM"], "Epochs run": [len(h["TATT"]["epoch"]), len(h["Bi-LSTM"]["epoch"])], "Selected epoch": [bt + 1, bb + 1],
                               "Validation AUPRC": [round(h["TATT"]["val_auprc"][bt], 3), round(h["Bi-LSTM"]["val_auprc"][bb], 3)], "Validation AUROC": [round(h["TATT"]["val_auroc"][bt], 3), round(h["Bi-LSTM"]["val_auroc"][bb], 3)]}), hide_index=True)
    st.dataframe(pd.DataFrame({"Setting": ["Label-shifting horizon", "Negative : positive windows", "Focal Loss gamma / alpha_t", "Optimiser", "Scheduler", "Early stopping", "Learning rate / weight decay / batch", "Seed"],
                               "Value": ["4 to 6 h before Sepsis-3 onset", "5 : 1 (training: 3 within-patient negatives per positive)", "2.0 / 0.75", "AdamW", "ReduceLROnPlateau (validation AUPRC)", "10 epochs without improvement",
                                         f"{R['config']['lr']} / {R['config']['weight_decay']} / {R['config']['batch']}", "0"]}), hide_index=True)
    note(f"Training wall time for all models, evaluation and interpretability analyses: {R['runtime_s'] / 60:.0f} minutes. Validation AUPRC peaked at epoch {bt + 1} for the TATT and then fell while the training loss kept decreasing, which is the signature of overfitting.")

# ============================================================================================ EVALUATION
elif page == "Evaluation":
    st.title("Evaluation on the held-out test partition (Objectives iii and iv)")
    tabs = st.tabs(["Discrimination", "Operating points and calibration", "Statistical comparison", "Controls for confounding", "Lead time and alarm burden"])
    with tabs[0]:
        rows = [{"Model": k, "AUROC": ci(M[k], "auroc"), "AUPRC (raw)": ci(M[k], "auprc"), "AUPRC (adjusted)": ci(M[k], "auprc_adj")} for k in MODELS]
        rows.append({"Model": "Time-in-stay control", "AUROC": f"{R['controls']['time_only']['auroc']:.3f}", "AUPRC (raw)": f"{R['controls']['time_only']['auprc']:.3f}", "AUPRC (adjusted)": f"{R['controls']['time_only']['auprc_adj']:.3f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(f"Brackets: 95% bootstrap intervals. Test windows are sampled at 5:1; the adjusted AUPRC reweights negatives to the cohort prevalence ({PREV:.3f}) and is the figure compared with the 0.35 threshold.")
        adj = st.checkbox("Adjust precision to cohort prevalence", value=True)
        c1, c2 = st.columns(2); fr, fp = go.Figure(), go.Figure(); w = WTS if adj else None
        for k in MODELS:
            f, t = C.roc_curve(Y, SC[k]); fr.add_trace(go.Scatter(x=f, y=t, name=f"{k} ({M[k]['auroc']:.3f})", line=dict(color=MC[k])))
            rc, pr = C.pr_curve(Y, SC[k], w); fp.add_trace(go.Scatter(x=rc, y=pr, name=f"{k} ({M[k]['auprc_adj' if adj else 'auprc']:.3f})", line=dict(color=MC[k])))
        fr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=GREY, dash="dot"), showlegend=False)); fr.update_xaxes(title="False positive rate"); fr.update_yaxes(title="True positive rate")
        fp.add_hline(y=PREV if adj else Y.mean(), line=dict(color=GREY, dash="dot"))
        if adj: fp.add_hline(y=R["threshold_auprc"], line=dict(color=RED, dash="dash"), annotation_text="Clinical signal threshold 0.35")
        fp.update_xaxes(title="Recall"); fp.update_yaxes(title="Precision", range=[0, 1.02])
        with c1: show(fr, 380)
        with c2: show(fp, 380)
        fig = go.Figure([go.Bar(x=MODELS + ["Time-in-stay control"], y=[M[k]["auroc"] for k in MODELS] + [R["controls"]["time_only"]["auroc"]], marker_color=[MC[k] for k in MODELS] + [GREY])])
        fig.add_hline(y=.5, line=dict(color=GREY, dash="dot")); fig.update_yaxes(title="AUROC", range=[0, 1.05]); show(fig, 300)
    with tabs[1]:
        o = {k: op_point(k) for k in MODELS}
        st.dataframe(pd.DataFrame([{"Model": k, "Threshold": round(THR[k], 3), "Precision": round(o[k]["precision"], 3), "Recall": round(o[k]["recall"], 3), "Specificity": round(o[k]["specificity"], 3), "F1": round(o[k]["f1"], 3),
                                    "TP": o[k]["tp"], "FP": o[k]["fp"], "FN": o[k]["fn"], "TN": o[k]["tn"]} for k in MODELS]), hide_index=True)
        st.caption("Thresholds maximise F1 on the validation partition and are then fixed.")
        c1, c2, c3 = st.columns(3)
        for col, k in zip((c1, c2, c3), MODELS):
            with col:
                m_ = np.array([[o[k]["tn"], o[k]["fp"]], [o[k]["fn"], o[k]["tp"]]]); fig = go.Figure(go.Heatmap(z=m_, x=["Pred. negative", "Pred. positive"], y=["True negative", "True positive"], text=m_, texttemplate="%{text}", colorscale="Blues", showscale=False))
                fig.update_yaxes(autorange="reversed"); fig.update_layout(title=dict(text=k, x=.5)); show(fig, 280)
        th = np.linspace(.02, .98, 97); k = st.selectbox("Model for threshold sweep", MODELS)
        pp, rr, ff = [], [], []
        for t in th:
            p_ = SC[k] >= t; tp_ = (p_ & (Y == 1)).sum(); fp_ = (p_ & (Y == 0)).sum(); fn_ = (~p_ & (Y == 1)).sum(); a_ = tp_ / max(tp_ + fp_, 1); b_ = tp_ / max(tp_ + fn_, 1); pp.append(a_); rr.append(b_); ff.append(2 * a_ * b_ / max(a_ + b_, 1e-9))
        fig = go.Figure([go.Scatter(x=th, y=pp, name="Precision", line=dict(color=BLUE)), go.Scatter(x=th, y=rr, name="Recall", line=dict(color=RED)), go.Scatter(x=th, y=ff, name="F1", line=dict(color="#14202B", dash="dash"))])
        fig.add_vline(x=THR[k], line=dict(color=MC[k])); fig.update_xaxes(title="Alert threshold"); show(fig, 300)
        fig = go.Figure()
        for k in MODELS:
            q = np.quantile(SC[k], np.linspace(0, 1, 11)); idx = np.clip(np.digitize(SC[k], q[1:-1]), 0, 9)
            fig.add_trace(go.Scatter(x=[SC[k][idx == i].mean() for i in range(10)], y=[Y[idx == i].mean() for i in range(10)], mode="lines+markers", name=k, line=dict(color=MC[k])))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=GREY, dash="dot"), showlegend=False)); fig.update_xaxes(title="Mean predicted probability (decile)"); fig.update_yaxes(title="Observed positive fraction"); show(fig, 340)
        st.caption("Scores are rankings, not calibrated risks: windows were sampled at 5:1 and Focal Loss up-weights positives.")
    with tabs[2]:
        ST = R["stats"]
        st.dataframe(pd.DataFrame([{"Comparison": f"TATT vs {k}", "AUROC TATT": round(v["delong_auc_tatt"], 3), "AUROC baseline": round(v["delong_auc_other"], 3), "DeLong p": "< 0.001" if v["delong_p"] < 0.001 else f"{v['delong_p']:.3f}",
                                    "AUPRC difference (adj., 95% CI)": f"{v['auprc_diff']:+.3f} ({v['auprc_diff_ci'][0]:+.3f} to {v['auprc_diff_ci'][1]:+.3f})", "Paired bootstrap p": "< 0.01"} for k, v in ST.items()]), hide_index=True)
        fig = go.Figure()
        for k, v in ST.items():
            d = v["auprc_diff"]; lo, hi = v["auprc_diff_ci"]; fig.add_trace(go.Scatter(x=[d], y=[f"TATT minus {k}"], mode="markers", marker=dict(color=MC[k], size=11), error_x=dict(type="data", symmetric=False, array=[hi - d], arrayminus=[d - lo]), showlegend=False))
        fig.add_vline(x=0, line=dict(color=RED, dash="dash")); fig.update_xaxes(title="Paired difference in adjusted AUPRC (95% bootstrap interval)"); show(fig, 260)
    with tabs[3]:
        st.markdown("**Time-in-stay control.** A score equal to the window's hours since admission uses no physiology. It outperforms every model on the test windows, so model comparisons must also be made within patients.")
        st.dataframe(pd.DataFrame({"Cohort": ["1,945 patients"] * 3 + ["4,927 patients"] * 2, "Model or control": ["Time-in-stay control", "TATT, absolute-time encoding", "TATT, window-relative encoding", "Time-in-stay control", "TATT, window-relative encoding (final)"],
                                   "AUROC": [0.918, 0.909, 0.594, round(R["controls"]["time_only"]["auroc"], 3), round(M["TATT"]["auroc"], 3)], "AUPRC (raw)": [0.668, 0.603, 0.209, round(R["controls"]["time_only"]["auprc"], 3), round(M["TATT"]["auprc"], 3)]}), hide_index=True)
        st.caption("The 1,945-patient rows come from the earlier development cohort and are diagnostic evidence for the choice of a window-relative time encoding.")
        st.markdown(f"**Within-septic test.** {W['n_pos']} positive windows versus {W['n_neg']} within-patient negative windows from the same septic patients.")
        st.dataframe(pd.DataFrame([{"Model": k, "AUROC": ci(W[k], "auroc"), "AUPRC": ci(W[k], "auprc")} for k in MODELS]), hide_index=True)
        fig = make_subplots(rows=1, cols=2, subplot_titles=["AUROC", "AUPRC"])
        for i, key in enumerate(("auroc", "auprc"), 1):
            fig.add_trace(go.Bar(x=MODELS, y=[W[k][key] for k in MODELS], marker_color=[MC[k] for k in MODELS], error_y=dict(type="data", symmetric=False, array=[W[k][key + "_ci"][1] - W[k][key] for k in MODELS], arrayminus=[W[k][key] - W[k][key + "_ci"][0] for k in MODELS]), showlegend=False), row=1, col=i)
            fig.add_hline(y=.5, line=dict(color=GREY, dash="dot"), row=1, col=i)
        fig.update_yaxes(range=[0, .85]); show(fig, 320)
    with tabs[4]:
        HZ, HW, AB = R["horizon"], R["horizon_within"], R["alarm_burden"]
        fig = go.Figure(); fig.add_vrect(x0=4, x1=6, fillcolor=NAVY, opacity=.08, line_width=0)
        for k in MODELS:
            xs = sorted(int(x) for x in HZ[k]); fig.add_trace(go.Scatter(x=xs, y=[HZ[k][str(x)] for x in xs], mode="lines+markers", name=k, line=dict(color=MC[k])))
        fig.add_hline(y=R["threshold_auprc"], line=dict(color=RED, dash="dash"), annotation_text="Clinical signal threshold 0.35"); fig.add_hline(y=PREV, line=dict(color=GREY, dash="dot"))
        fig.update_xaxes(title="Lead time before Sepsis-3 onset (h)"); fig.update_yaxes(title="AUPRC (adjusted to prevalence)", range=[0, .8]); show(fig, 340)
        st.dataframe(pd.DataFrame({"Model": MODELS, "Early prediction horizon (h)": [R["early_horizon"][k] if R["early_horizon"][k] is not None else "none above threshold" for k in MODELS],
                                   "Adjusted AUPRC at 4 h": [round(HZ[k]["4"], 3) for k in MODELS], "Adjusted AUPRC at 6 h": [round(HZ[k]["6"], 3) for k in MODELS]}), hide_index=True)
        st.caption("These curves are flat for all models. A signal driven by emerging physiology should weaken with lead time, so this analysis, which uses negatives from other patients, does not demonstrate genuine anticipation.")
        fig = go.Figure(); fig.add_vrect(x0=4, x1=6, fillcolor=NAVY, opacity=.08, line_width=0)
        for k in MODELS:
            xs = sorted(int(x) for x in HW[k]); fig.add_trace(go.Scatter(x=xs, y=[HW[k][str(x)] for x in xs], mode="lines+markers", name=k, line=dict(color=MC[k])))
        fig.add_hline(y=.5, line=dict(color=GREY, dash="dot")); fig.update_xaxes(title="Lead time before onset (h)"); fig.update_yaxes(title="Within-patient AUROC", range=[.4, .8]); show(fig, 320)
        st.caption(f"Positive windows at each lead time versus early baseline windows of the same {R['horizon_within_n']['patients']} septic test patients.")
        fig = go.Figure(go.Bar(x=MODELS, y=[AB[k]["false_alerts_per_1000h"] for k in MODELS], marker_color=[MC[k] for k in MODELS])); fig.update_yaxes(title="False alerts per 1,000 monitored hours"); show(fig, 300)
        st.caption(f"At the validation-selected thresholds, from hourly scoring of 60 test patients ({AB['TATT']['hours']:,} patient-hours). These alert volumes are too high for clinical use.")

# ============================================================================================ INTERPRETABILITY
elif page == "Interpretability":
    st.title("Interpretability evaluation (Objective iv)")
    IN = R["interpretability"]; tv = R["cohort"]["tokens_by_variable"]; tot = sum(tv.values()); att = IN["mean_attention_by_variable"]; rh = np.array(IN["spearman_rhos"])
    t1, t2, t3 = st.tabs(["Clinical concordance", "Severity correlation", "Case review"])
    with t1:
        k = st.columns(3); k[0].metric("True-positive alerts analysed", IN["n_true_positive_alerts"]); k[1].metric("Concordance rate", f"{IN['concordance_rate']:.1%}"); k[2].metric("Chance level", f"{IN['chance_rate']:.1%}")
        st.caption("Concordance: the three most-attended variables in the final six hours include lactate, MAP, respiratory rate or SpO2.")
        ks = [x for x in VN if tv[x] > 0]
        fig = go.Figure([go.Bar(y=ks, x=[tv[x] / tot for x in ks], orientation="h", name="Share of observation tokens", marker_color=LIGHT), go.Bar(y=ks, x=[att[x] for x in ks], orientation="h", name="Mean attention mass", marker_color=NAVY)])
        fig.update_layout(barmode="group"); fig.update_yaxes(autorange="reversed"); fig.update_xaxes(title="Share"); show(fig, 460)
        st.caption("Attention is not simply proportional to how often a variable is observed: laboratory tokens receive more attention than their frequency implies.")
    with t2:
        k = st.columns(3); k[0].metric("Patients", len(rh)); k[1].metric("Median Spearman rho", f"{np.median(rh):.2f}"); k[2].metric("Share with rho > 0", f"{np.mean(rh > 0):.0%}")
        fig = go.Figure(go.Histogram(x=rh, xbins=dict(start=-1, end=1, size=.1), marker_color=NAVY)); fig.add_vline(x=0, line=dict(color=RED, dash="dash")); fig.update_xaxes(title="Spearman rho: TATT risk vs SOFA proxy (12 h before onset)"); fig.update_yaxes(title="Patients"); show(fig, 320)
        st.caption("The SOFA proxy uses platelets, bilirubin, creatinine and MAP. The risk score does not track it consistently.")
    with t3:
        p = PATS[[label(x) for x in PATS].index(st.selectbox("Test patient", [label(x) for x in PATS], key="case_p"))]
        d0 = float(p["onset"] - 5) if (not np.isnan(p["onset"]) and p["onset"] - 5 >= D.MIN_PRED_TIME) else float(min(p["T"], 48.0))
        te = st.slider("Alert time (h)", float(D.MIN_PRED_TIME), float(p["T"]), d0, 0.5, key="case_t")
        score, va, w, imp = explain_at(p, te); c1, c2 = st.columns(2)
        with c1:
            o_ = np.argsort(-va); fig = go.Figure(go.Bar(x=[VN[i] for i in o_ if va[i] > 0], y=[va[i] for i in o_ if va[i] > 0], marker_color=NAVY)); fig.update_yaxes(title="Share of attention (final 6 h)"); show(fig, 320)
            st.caption(f"Risk score {score:.3f}")
        with c2:
            fig = go.Figure(go.Scatter(x=w["time"], y=imp, mode="markers", marker=dict(color=[RED if f == 0 else NAVY for f in w["flag"]], size=5))); fig.update_xaxes(title="Hours since window start"); fig.update_yaxes(title="Attention received"); show(fig, 320)
            st.caption("Red points are missing-laboratory mask tokens.")

# ============================================================================================ SIMULATOR
elif page == "Patient simulator":
    st.title("Patient simulator")
    st.markdown("Enter an ICU patient's state, upload observations, or replay a held-out test patient. The trained TATT returns a sepsis risk score for the 4-to-6-hour horizon.")
    note(f"Context: on the held-out test set this model reached AUROC {M['TATT']['auroc']:.3f} and was less discriminative than XGBoost ({M['XGBoost']['auroc']:.3f}); at the default threshold it produced about {R['alarm_burden']['TATT']['false_alerts_per_1000h']:.0f} false alerts per 1,000 monitored hours. Scores are rankings, not calibrated probabilities. Not for clinical use.", "warn")
    thr0 = float(np.clip(THR["TATT"], 0.01, 0.99))
    thr = st.sidebar.slider("Alert threshold", 0.01, 0.99, thr0, 0.01); st.sidebar.caption(f"Validation-F1 optimum: {thr0:.2f}")

    def render(p, t_eval, true_onset=None):
        score, va, w, imp = explain_at(p, t_eval)
        ts = np.array([t for t in np.arange(t_eval - 12, t_eval + 1e-9, 1.0) if t >= D.MIN_PRED_TIME]); traj = risk_at(p, ts) if len(ts) else np.zeros(0); alert = score >= thr
        g1, g2 = st.columns([1, 2])
        with g1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=score * 100, number=dict(suffix="%"), title=dict(text="Model risk score"),
                                         gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=RED if alert else NAVY), threshold=dict(line=dict(color="#14202B", width=3), value=thr * 100),
                                                    steps=[dict(range=[0, thr * 100], color="#EAEFF5"), dict(range=[thr * 100, 100], color="#FBEDEC")])))
            show(fig, 270)
            st.markdown('<div class="alert">Alert: risk score is above the selected threshold. In a real system this would prompt clinician review for suspected infection; it is not a diagnosis.</div>' if alert else '<div class="ok">No alert: risk score is below the selected threshold.</div>', unsafe_allow_html=True)
        with g2:
            fig = go.Figure()
            if len(ts): fig.add_trace(go.Scatter(x=ts, y=traj, mode="lines+markers", line=dict(color=NAVY, width=2), name="Risk score"))
            fig.add_hline(y=thr, line=dict(color=RED, dash="dash"), annotation_text="Threshold")
            if true_onset is not None and not np.isnan(true_onset): fig.add_vline(x=true_onset, line=dict(color=RED), annotation_text="Sepsis-3 onset")
            fig.update_xaxes(title="Hours from ICU admission"); fig.update_yaxes(title="Risk score", range=[0, 1.02]); show(fig, 310)
        pneg = float((SC["TATT"][Y == 0] <= score).mean()); ppos = float((SC["TATT"][Y == 1] <= score).mean())
        d1, d2 = st.columns([2, 3])
        with d1:
            st.markdown("**Where this score falls among real test windows**")
            st.markdown(f"The score is higher than **{pneg:.0%}** of negative windows and **{ppos:.0%}** of positive windows (4 to 6 h before onset) in the held-out test set. "
                        "Because the two distributions overlap heavily, a score in the middle of the range carries little information about whether sepsis will follow.")
        with d2:
            fig = go.Figure([go.Histogram(x=SC["TATT"][Y == 0], histnorm="probability density", nbinsx=30, name="Negative test windows", marker_color=BLUE, opacity=.6),
                             go.Histogram(x=SC["TATT"][Y == 1], histnorm="probability density", nbinsx=30, name="Positive test windows", marker_color=RED, opacity=.55)])
            fig.update_layout(barmode="overlay"); fig.add_vline(x=score, line=dict(color="#14202B", width=3), annotation_text="This patient"); fig.add_vline(x=thr, line=dict(color=RED, dash="dash"))
            fig.update_xaxes(title="TATT risk score"); fig.update_yaxes(title="Density"); show(fig, 250)
        c1, c2 = st.columns(2)
        with c1:
            o_ = [i for i in np.argsort(-va) if va[i] > 0][:10]
            fig = go.Figure(go.Bar(x=[va[i] for i in o_][::-1], y=[VN[i] for i in o_][::-1], orientation="h", marker_color=NAVY)); fig.update_xaxes(title="Share of attention (final 6 h)"); show(fig, 340)
        with c2:
            cur = {}
            for j in range(D.N_VARS):
                k = np.where((p["var"] == j) & (p["time"] <= t_eval))[0]
                if len(k): cur[VN[j]] = (float(p["val"][k[-1]]), t_eval - float(p["time"][k[-1]]))
            st.markdown("**Most recent value of each variable**")
            st.dataframe(pd.DataFrame([{"Variable": k, "Value": round(v[0], 2), "Unit": UN[k], "Hours ago": round(v[1], 2)} for k, v in cur.items()]), hide_index=True)

    tab_m, tab_u, tab_t = st.tabs(["Manual entry", "Upload observations", "Test-set patient"])
    with tab_m:
        SPEC = {"HR": (30.0, 200.0, 1.0), "MAP": (30.0, 140.0, 1.0), "SBP": (50.0, 220.0, 1.0), "DBP": (20.0, 120.0, 1.0), "RR": (6.0, 50.0, 1.0), "SpO2": (70.0, 100.0, 1.0),
                "Lactate": (0.4, 15.0, 0.1), "Creatinine": (0.3, 8.0, 0.1), "BUN": (4.0, 120.0, 1.0), "Bilirubin": (0.1, 20.0, 0.1), "Platelets": (10.0, 450.0, 5.0), "WBC": (1.0, 40.0, 0.5)}
        PRESETS = {
            "Stable": dict(HR=(78, 0), MAP=(86, 0), SBP=(124, 0), DBP=(64, 0), RR=(15, 0), SpO2=(97, 0), Lactate=(1.0, 0.0), Creatinine=(0.9, 0.0), BUN=(15, 0), Bilirubin=(0.7, 0.0), Platelets=(240, 0), WBC=(8.5, 0.0)),
            "Early deterioration": dict(HR=(104, 22), MAP=(72, -13), SBP=(104, -18), DBP=(55, -8), RR=(23, 6), SpO2=(94, -3), Lactate=(2.6, 1.2), Creatinine=(1.3, 0.3), BUN=(26, 6), Bilirubin=(1.3, 0.4), Platelets=(165, -40), WBC=(14.0, 4.0)),
            "Septic shock": dict(HR=(128, 40), MAP=(56, -28), SBP=(82, -36), DBP=(40, -18), RR=(30, 11), SpO2=(90, -6), Lactate=(5.2, 3.0), Creatinine=(2.6, 1.0), BUN=(48, 18), Bilirubin=(3.0, 1.6), Platelets=(70, -90), WBC=(21.0, 9.0))}

        def apply_preset():
            for n_, (v_, t_) in PRESETS[st.session_state["preset"]].items():
                st.session_state[f"v_{n_}"] = float(v_); st.session_state[f"tr_{n_}"] = float(t_)

        if "preset" not in st.session_state:
            st.session_state["preset"] = "Stable"; apply_preset()
        st.selectbox("Load a clinical scenario", list(PRESETS), key="preset", on_change=apply_preset)
        st.caption("Scenarios are illustrative, not patient data. The model separates septic from non-septic windows only weakly (see Evaluation), so these scenarios will not necessarily receive clearly different scores; compare each score with the test-set distributions shown below the result.")
        with st.form("manual_form"):
            s1, s2, s3 = st.columns(3)
            t_eval = s1.number_input("Hours since ICU admission", 24.0, 400.0, 48.0, 1.0, help="The model needs at least 24 hours of context.")
            interval = s2.slider("Vital-sign charting interval (minutes)", 5, 60, 60)
            seed = s3.number_input("Sampling seed", 0, 9999, 7)
            st.markdown("**Vital signs** (current value and change over the last 6 hours)")
            vin = {}
            for c, n in zip(st.columns(6), list(SPEC)[:6]):
                lo, hi, stp = SPEC[n]
                with c:
                    v = st.number_input(f"{n} ({UN[n]})", lo, hi, step=stp, key=f"v_{n}"); tr = st.number_input(f"{n} change", -hi, hi, step=stp, key=f"tr_{n}")
                vin[n] = {"val": v, "trend": tr}
            st.markdown("**Laboratory results** (latest value, change from previous draw, hours since the latest draw)")
            lin = {}
            for c, n in zip(st.columns(6), list(SPEC)[6:]):
                lo, hi, stp = SPEC[n]
                with c:
                    meas = st.checkbox(f"{n} measured", True, key=f"m_{n}")
                    v = st.number_input(f"{n} ({UN[n]})", lo, hi, step=stp, key=f"v_{n}"); tr = st.number_input(f"{n} change", -hi, hi, step=stp, key=f"tr_{n}"); ago = st.number_input(f"{n} hours ago", 0.0, 24.0, 2.0, 0.5, key=f"ago_{n}")
                lin[n] = {"meas": meas, "val": v, "trend": tr, "ago": ago}
            go_btn = st.form_submit_button("Run prediction", type="primary")

        def build(vin, lin):
            rng = np.random.default_rng(int(seed)); V, X, T = [], [], []
            for n, d in vin.items():
                j = D.VAR_IDX[n]; gaps = rng.exponential(interval / 60.0, int(24 / (interval / 60.0) * 2) + 5) + 0.03
                t = t_eval - np.cumsum(gaps) + gaps[0]; t = t[t > max(0.0, t_eval - 30)]; frac = np.clip((t_eval - t) / 6.0, 0, 1)
                v = np.clip(d["val"] - d["trend"] * frac + rng.normal(0, D._NOISE[n], len(t)) * 0.5, SPEC[n][0], SPEC[n][1]); V.append(np.full(len(t), j, np.int8)); X.append(v); T.append(t)
            for n, d in lin.items():
                if not d["meas"]: continue
                j = D.VAR_IDX[n]; t = np.array([t_eval - d["ago"] - 16, t_eval - d["ago"] - 8, t_eval - d["ago"]]); v = np.array([d["val"] - d["trend"], d["val"] - d["trend"], d["val"]]); ok = t >= 0
                V.append(np.full(int(ok.sum()), j, np.int8)); X.append(np.clip(v[ok], SPEC[n][0], SPEC[n][1])); T.append(t[ok])
            var, val, tm = np.concatenate(V), np.concatenate(X).astype(np.float32), np.concatenate(T).astype(np.float32); o = np.argsort(tm, kind="stable")
            return {"id": "manual", "T": t_eval, "onset": np.nan, "var": var[o], "val": val[o], "time": tm[o]}

        if go_btn or st.session_state.get("ran_manual"):
            st.session_state["ran_manual"] = True; render(build(vin, lin), float(t_eval))
            note("Observations are generated from the entered values and trends with irregular timing, so the trajectory reflects the entered trends and not recorded history.")
    with tab_u:
        st.markdown("Upload a long-format CSV with columns `variable, value, time_h` (hours from ICU admission). Variables: " + ", ".join(VN[:12]) + ".")
        ex = PATS[0]; k = (ex["time"] > 10) & (ex["time"] < 50)
        tmpl = pd.DataFrame({"variable": [VN[i] for i in ex["var"][k]], "value": ex["val"][k].round(3), "time_h": ex["time"][k].round(3)})
        st.download_button("Download CSV template", tmpl.to_csv(index=False).encode(), "observations_template.csv", "text/csv")
        up = st.file_uploader("Observations CSV", type="csv")
        if up is not None:
            try:
                df = pd.read_csv(up); df.columns = [c.strip().lower() for c in df.columns]
                assert {"variable", "value", "time_h"} <= set(df.columns), "Missing required columns: variable, value, time_h"
                bad = sorted(set(df["variable"]) - set(VN[:12])); assert not bad, f"Unknown variable names: {bad}"
                df = df.dropna(subset=["value", "time_h"]).sort_values("time_h")
                p = {"id": "upload", "T": float(df["time_h"].max()), "onset": np.nan, "var": df["variable"].map(D.VAR_IDX).to_numpy(np.int8), "val": df["value"].to_numpy(np.float32), "time": df["time_h"].to_numpy(np.float32)}
                assert p["T"] >= D.MIN_PRED_TIME, "At least 24 hours of data are required."
                te = st.slider("Evaluate at (hours from ICU admission)", float(D.MIN_PRED_TIME), float(p["T"]), float(p["T"]), 0.5, key="up_t"); render(p, float(te))
            except AssertionError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Could not read the file: {e}")
    with tab_t:
        p = PATS[[label(x) for x in PATS].index(st.selectbox("Held-out test patient", [label(x) for x in PATS], key="sim_p"))]
        d0 = float(p["onset"] - 5) if (not np.isnan(p["onset"]) and p["onset"] - 5 >= D.MIN_PRED_TIME) else float(min(p["T"], 48.0))
        te = st.slider("Evaluate at (hours from ICU admission)", float(D.MIN_PRED_TIME), float(p["T"]), d0, 0.5, key="sim_t")
        if not np.isnan(p["onset"]): st.caption(f"Sepsis-3 onset at {p['onset']:.1f} h; selected time is {p['onset'] - te:+.1f} h from onset.")
        render(p, float(te), p["onset"])

# ============================================================================================ METHODS
else:
    st.title("Methods and limitations")
    st.markdown("""
**Data.** PhysioNet/Computing in Cardiology Challenge 2019 records (open access), restricted to stays of at least 48 hours. Sepsis-3 onset was recovered from the supplied label (first positive label plus six hours). The split is by patient (70/15/15, stratified).

**Windows.** The model scores the 24 hours of observations before a prediction time (at most 256 most recent tokens). Positive windows end 4 to 6 hours before onset; negatives come from the earlier course of septic patients (at least 7 hours before onset) and from non-septic patients, at a 5:1 ratio.

**Model.** Observation tokens (variable, value, time) are embedded and combined with a sinusoidal encoding of time within the window. Two attention layers add the bias -alpha x time gap to the logits. Final-six-hour pooling feeds an MLP head. Training used Focal Loss (gamma 2, alpha_t 0.75) and AdamW with early stopping on validation AUPRC.

**Baselines.** XGBoost on 315 window summary features, and a two-layer bidirectional LSTM on an hourly grid with last-observation-carried-forward and mask channels.
""")
    st.subheader("Limitations")
    st.markdown("""
- One public database, no external validation; hourly data without intervention records, so the irregular-sampling premise is only partly exercised.
- Sepsis labels and onset times come from the database organisers.
- The TATT was significantly less discriminative than both baselines, and within-patient discrimination was weak for all models.
- A control using only time in the stay outperformed all models on case-control windows; a window-relative time encoding is used to avoid that shortcut.
- 534 septic training patients, one configuration, one seed.
- Scores are not calibrated, alert volumes at F1-optimal thresholds are far too high for clinical use, and attention weights are not validated explanations.
""")
    note("This dashboard is a research prototype that accompanies a thesis. It must not be used to make clinical decisions.", "warn")
