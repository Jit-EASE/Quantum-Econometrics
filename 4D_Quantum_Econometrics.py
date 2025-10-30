# Hyperfield_4D_Econometric.py
# ------------------------------------------------------------
# True  Simulator
# - 4D coordinate system with econometric-driven hyper-angle
# - Perspective/orthogonal projection from 4D -> 3D
# - Residuals, Fitted, or Combined colour source (user toggle)
# - OLS with optional robust SE, standardization, simulated Y targeting Adj R²
# - Streamlit + Plotly with animation (3D rotation + 4D precession)
# ------------------------------------------------------------

import io, math, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale
import os

# --- OpenAI Agentic Explanations ---
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # library may not be installed; we handle gracefully


st.set_page_config(page_title="4-Dimensional Quantum-Econometric HyperSpace - Concept", layout="wide")
# Environment key status for Agentic AI
HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

# ==============================
# Sidebar — Controls
# ==============================
st.sidebar.title("Controls")

seed = st.sidebar.number_input("Random Seed", 0, 999_999, 2025, step=1)
np.random.seed(seed)

# Geometry / particles
n_particles = st.sidebar.slider("Number of Particles", 500, 15000, 2000, step=100)
n_rings     = st.sidebar.slider("Number of Rings", 2, 48, 10, step=1)
max_radius  = st.sidebar.slider("Max Radius", 2.0, 50.0, 12.0, step=0.5)
ring_jitter = st.sidebar.slider("Radial Jitter (σ)", 0.0, 3.0, 0.5, step=0.05)

# Aesthetics
colorscale   = st.sidebar.selectbox("Colorscale", ["solar","viridis","plasma","cividis","magma","ice","haline","sunset","tempo"], index=0)
show_scale   = st.sidebar.checkbox("Show color scale", True)
bg_color     = st.sidebar.selectbox("Background", ["black","white","rgb(5,5,20)","rgb(10,10,10)"], index=0)
show_axes    = st.sidebar.checkbox("Show Axes", False)
size_min, size_max = st.sidebar.slider("Marker Size Range", 1, 20, (2, 8), step=1)


# Animation
st.sidebar.markdown("---")
st.sidebar.subheader("Animation")
animate      = st.sidebar.checkbox("Enable Animation", True)
frames_total = st.sidebar.slider("Frames", 10, 360, 120, step=5)
breathe_amp  = st.sidebar.slider("Breathing Amplitude (3D)", 0.00, 0.60, 0.12, step=0.01)
rot_speed    = st.sidebar.slider("Rotation Speed (3D)", 0.00, 1.00, 0.20, step=0.01)

# Canvas & Aspect controls
st.sidebar.markdown("---")
st.sidebar.subheader("Canvas & Aspect")
use_container = st.sidebar.checkbox("Fit to container width", True)
canvas_width  = st.sidebar.number_input("Canvas width (px)", 600, 4000, 1400, step=50)
canvas_height = st.sidebar.number_input("Canvas height (px)", 400, 3000, 800, step=50)
manual_aspect = st.sidebar.checkbox("Manual aspect ratio (x:y:z)", True)
aspect_x = st.sidebar.slider("Aspect X", 0.1, 4.0, 1.4, 0.1)
aspect_y = st.sidebar.slider("Aspect Y", 0.1, 4.0, 1.0, 0.1)
aspect_z = st.sidebar.slider("Aspect Z", 0.1, 4.0, 0.8, 0.1)

st.sidebar.subheader("Animation Timing")
frame_duration_ms = st.sidebar.slider("Frame duration (ms)", 10, 200, 60, step=5)

# 4D Projection Settings
st.sidebar.markdown("---")
st.sidebar.subheader("4D Projection Settings")
projection_intensity = st.sidebar.slider("Projection Intensity λ", 0.10, 5.00, 1.20, 0.10)
w_rotation_speed     = st.sidebar.slider("4D Rotation Speed ω₄", 0.00, 1.00, 0.15, 0.01)
color_source_4d      = st.sidebar.selectbox("Color Source (4D View)", ["Residuals","Fitted","Combined"], index=2)
orthogonal_proj      = st.sidebar.checkbox("Orthogonal Projection", value=False)

# Econometrics section
st.sidebar.markdown("---")
st.sidebar.subheader("Econometrics")
dep_var = st.sidebar.selectbox("Dependent Variable", ["z","y","x","r"], index=0)
pred_vars = st.sidebar.multiselect("Predictors (X)", ["x","y","z","r","theta","phi"], default=["x","y","r"])
use_robust = st.sidebar.checkbox("White Robust SE (HC1)", True)

# ---------------------------------------------------
# Estimator selector & advanced model controls
st.sidebar.markdown("---")
st.sidebar.subheader("Estimator")
estimator = st.sidebar.selectbox("Active model", ["OLS","QKR (quantum kernel)","QUBO selector","Compare (all)"], index=0)

# QKR hyperparameters
with st.sidebar.expander("QKR settings"):
    qkr_gamma = st.slider("γ (feature scale)", 0.05, 5.0, 0.8, 0.05)
    qkr_depth = st.slider("Depth (layers)", 1, 6, 3, 1)
    qkr_ridge = st.slider("Ridge α", 0.0, 5.0, 0.1, 0.05)
    qkr_seed  = st.number_input("Random seed", 0, 1_000_000, 4242, 1)

# QUBO-like subset selector
with st.sidebar.expander("QUBO selector settings"):
    qubo_lambda = st.slider("λ sparsity penalty", 0.0, 5.0, 1.0, 0.1)
    qubo_max_p  = st.slider("Max predictors (enumerate)", 2, 12, min(12, len(pred_vars)), 1)
    qubo_criterion = st.selectbox("Criterion", ["AIC","BIC"], index=0)

# Modeling options
st.sidebar.subheader("Modeling Options")
standardize   = st.sidebar.checkbox("Standardize predictors (z-score)", True)
synth_y       = st.sidebar.checkbox("Simulate Y to target Adj R²", True)
target_adj_r2 = st.sidebar.slider("Target Adj R²", 0.00, 0.95, 0.40, step=0.01)
lock_adj      = st.sidebar.checkbox("Lock Adj R² near target (auto-tune noise)", True)
beta_scale    = st.sidebar.number_input("simulated β scale", 0.0, 10.0, 1.0, step=0.1)
beta_seed     = st.sidebar.number_input("β Random Seed", 0, 1_000_000, 777, step=1)

show_hover = st.sidebar.checkbox("Show Hover Info", True)

st.sidebar.markdown("---")
st.sidebar.subheader("Agentic AI Explanations")
enable_ai = st.sidebar.checkbox("Enable OpenAI Explanations", value=False)
ai_model  = st.sidebar.selectbox("Model", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)
ai_max_tokens = st.sidebar.slider("Max tokens", 128, 2048, 512, step=64)

st.sidebar.caption("Tip: Toggle colour source to switch between Residuals, Fitted, or Combined energy.")

# ==============================
# Generate Base Field (3D)
# ==============================
theta = np.random.uniform(0, 2*np.pi, n_particles)
phi   = np.random.uniform(0, np.pi, n_particles)

base_rings = np.linspace(1, max_radius, n_rings)
radii = np.random.choice(base_rings, n_particles) + np.random.normal(0, ring_jitter, n_particles)

x = radii * np.sin(phi) * np.cos(theta)
y = radii * np.sin(phi) * np.sin(theta)
z = radii * np.cos(phi)
r = radii.copy()

# Sizes
sizes = np.random.uniform(size_min, size_max, n_particles)

# ==============================
# Econometrics (OLS)
# ==============================
feat_map = {"x": x, "y": y, "z": z, "r": r, "theta": theta, "phi": phi}

def build_features(pred_list, use_standardize=False):
    if len(pred_list) == 0:
        return None, None, None
    cols, mus, sigs = [], [], []
    for p in pred_list:
        v = feat_map[p].astype(float)
        if use_standardize:
            mu = float(v.mean())
            sd = float(v.std() + 1e-12)
            v = (v - mu) / sd
            mus.append(mu); sigs.append(sd)
        else:
            mus.append(0.0); sigs.append(1.0)
        cols.append(v)
    Xk = np.column_stack(cols)
    return Xk, np.array(mus), np.array(sigs)

def build_design(pred_list, use_standardize=False):
    Xk, _, _ = build_features(pred_list, use_standardize)
    if Xk is None:
        return None
    return np.column_stack([np.ones(Xk.shape[0]), Xk])  # add intercept

def _adj_r2_from_XY(Xmat: np.ndarray, Yvec: np.ndarray) -> float:
    try:
        XtX = Xmat.T @ Xmat
        XtX_inv = np.linalg.pinv(XtX)
        beta_tmp = XtX_inv @ (Xmat.T @ Yvec)
        yhat_tmp = Xmat @ beta_tmp
        resid_tmp = Yvec - yhat_tmp
        n_tmp, k_tmp = Xmat.shape
        ss_tot_tmp = ((Yvec - Yvec.mean())**2).sum()
        ss_res_tmp = (resid_tmp**2).sum()
        r2_tmp = 1 - ss_res_tmp / (ss_tot_tmp + 1e-12)
        adj_tmp = 1 - (1 - r2_tmp) * (n_tmp - 1) / max(1, (n_tmp - k_tmp))
        return float(adj_tmp)
    except Exception:
        return np.nan

# ==============================
# Metrics & model helpers
# ==============================

def compute_metrics(y_true, y_hat, n_params):
    resid = y_true - y_hat
    n = len(y_true)
    k = int(n_params)
    ss_tot = float(((y_true - y_true.mean())**2).sum())
    ss_res = float((resid**2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, (n - k))
    rmse = math.sqrt(max(ss_res / max(1, n), 0.0))
    sigma2 = max(ss_res / max(1, n), 1e-12)
    ll = -0.5 * n * (math.log(2*math.pi*sigma2) + 1)
    aic = 2*k - 2*ll
    bic = k*math.log(max(n,1)) - 2*ll
    return {"r2": r2, "adj_r2": adj_r2, "rmse": rmse, "aic": aic, "bic": bic}


def fit_qkr(Xk, Y, gamma=0.8, depth=3, ridge=0.1, seed=4242):
    """Quantum-inspired kernel ridge via stacked cos/sin random feature maps.
    Xk: (n, p) without intercept. Returns result dict akin to OLS block.
    """
    rng = np.random.default_rng(int(seed))
    n, p = Xk.shape
    feats = []
    for _ in range(int(depth)):
        W = rng.normal(0.0, gamma, size=(p, p))
        Z = Xk @ W
        feats.append(np.cos(Z))
        feats.append(np.sin(Z))
    Phi = np.concatenate(feats, axis=1)
    Phi_i = np.column_stack([np.ones(n), Phi])
    I = np.eye(Phi_i.shape[1])
    beta = np.linalg.pinv(Phi_i.T @ Phi_i + ridge * I) @ (Phi_i.T @ Y)
    yhat = Phi_i @ beta
    resid = Y - yhat
    mets = compute_metrics(Y, yhat, n_params=Phi_i.shape[1])
    return {
        "beta": beta,
        "yhat": yhat,
        "resid": resid,
        "r2": mets["r2"], "adj_r2": mets["adj_r2"],
        "rmse": mets["rmse"], "aic": mets["aic"], "bic": mets["bic"],
        "names": ["Intercept"] + [f"f{i}" for i in range(Phi_i.shape[1]-1)],
        "n": n, "k": Phi_i.shape[1], "robust": False,
        "_kind": "QKR"
    }


def fit_qubo_selector(Xk_full, pred_names, Y, lam=1.0, criterion="AIC", max_p=12):
    """Exhaustive subset selection up to max_p predictors minimizing AIC/BIC + λ|S|."""
    from itertools import combinations
    n, p = Xk_full.shape
    use_idx = list(range(min(int(max_p), p)))
    best = None
    for rsel in range(1, len(use_idx)+1):
        for combo in combinations(use_idx, rsel):
            Xs = Xk_full[:, combo]
            Xi = np.column_stack([np.ones(n), Xs])
            beta = np.linalg.pinv(Xi.T @ Xi) @ (Xi.T @ Y)
            yhat = Xi @ beta
            mets = compute_metrics(Y, yhat, n_params=Xi.shape[1])
            score = (mets["aic"] if criterion == "AIC" else mets["bic"]) + lam * len(combo)
            if (best is None) or (score < best["score"]):
                best = {
                    "beta": beta, "yhat": yhat, "resid": Y - yhat,
                    "r2": mets["r2"], "adj_r2": mets["adj_r2"],
                    "rmse": mets["rmse"], "aic": mets["aic"], "bic": mets["bic"],
                    "names": ["Intercept"] + [pred_names[i] for i in combo],
                    "n": n, "k": Xi.shape[1], "robust": False,
                    "score": score, "subset": [pred_names[i] for i in combo],
                    "_kind": "QUBO"
                }
    return best

# ==============================
# Agentic AI helpers
# ==============================
def _state_summary_for_prompt():
    # Build a concise, numeric summary of the current scene/econometrics
    summary = {
        "n_particles": int(n_particles),
        "n_rings": int(n_rings),
        "projection_intensity": float(projection_intensity),
        "w_rotation_speed": float(w_rotation_speed),
        "breathe_amp": float(breathe_amp),
        "rot_speed": float(rot_speed),
        "orthogonal_proj": bool(orthogonal_proj),
        "colorscale": str(colorscale),
        "color_source": str(color_source_4d),
    }
    if 'ols_results' in globals() and ols_results is not None:
        summary.update({
            "r2": float(ols_results.get("r2", float('nan'))),
            "adj_r2": float(ols_results.get("adj_r2", float('nan'))),
            "n": int(ols_results.get("n", 0)),
            "k": int(ols_results.get("k", 0)),
        })
        # basic residual stats
        try:
            resid_arr = np.asarray(ols_results.get("resid"))
            summary.update({
                "resid_mean": float(np.mean(resid_arr)),
                "resid_std": float(np.std(resid_arr)),
                "resid_p95": float(np.percentile(resid_arr, 95)),
                "resid_p05": float(np.percentile(resid_arr, 5)),
            })
        except Exception:
            pass
    return summary


def _call_openai_explain(model: str, max_tokens: int, narrative_goal: str = "Explain the current hyperfield succinctly for a mixed technical audience."):
    if OpenAI is None:
        return "OpenAI client not available. Please install the 'openai' package."
    # Read API key from environment (no UI collection)
    if not os.getenv("OPENAI_API_KEY"):
        return "OpenAI API key not found. Set OPENAI_API_KEY in your environment."
    try:
        client = OpenAI()  # reads key from env automatically
        summary = _state_summary_for_prompt()
        # Keep prompt compact and deterministic; no chain-of-thought disclosure
        user_msg = (
            "You are an econometrics+visual analytics assistant. "
            "Explain, without equations, what the current 4D hyperfield shows. "
            "Touch on: what colours encode (" + str(color_source_4d) + "), how 4D projection (λ) and precession (ω4) affect depth, and what R²/Adj R² imply. "
            "Be concise (150-220 words), professional, and actionable.\n\n"
            f"STATE: {summary}\n"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You generate crisp, non-speculative explanations without revealing internal reasoning."},
                      {"role": "user", "content": user_msg}],
            temperature=0.4,
            max_tokens=int(max_tokens),
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[AI error] {e}"

# Build X
X = build_design(pred_vars, use_standardize=standardize)

# Build/Select Y
if synth_y and X is not None and len(pred_vars) > 0:
    rng_beta = np.random.default_rng(int(beta_seed))
    Xk, _, _ = build_features(pred_vars, use_standardize=standardize)
    beta_vec = rng_beta.normal(0.0, beta_scale, size=Xk.shape[1])
    y_signal = Xk @ beta_vec
    var_s = float(np.var(y_signal))
    if var_s < 1e-12:
        noise = np.random.default_rng(12345).normal(0.0, 1.0, size=Xk.shape[0])
        Y = y_signal + noise
    else:
        R2_target = float(np.clip(target_adj_r2, 0.0, 0.95))
        var_e = var_s * (1.0 - R2_target) / max(1e-12, R2_target)
        # auto-tune to target Adj R² (±0.02)
        for _ in range(10):
            noise = rng_beta.normal(0.0, np.sqrt(max(var_e, 1e-18)), size=Xk.shape[0])
            Y_try = y_signal + noise
            if not lock_adj:
                Y = Y_try
                break
            adj_now = _adj_r2_from_XY(X, Y_try)
            if np.isnan(adj_now):
                Y = Y_try; break
            if abs(adj_now - target_adj_r2) <= 0.02:
                Y = Y_try; break
            var_e = var_e * 1.5 if adj_now > target_adj_r2 else var_e / 1.5
        else:
            Y = Y_try
    dep_label = f"simulated Y (Adj R²≈{_adj_r2_from_XY(X, Y):.2f}, target {target_adj_r2:.2f})"
else:
    Y = feat_map[dep_var].astype(float)
    dep_label = f"Observed {dep_var}"

# ==============================
# Model fitting: OLS, QKR, QUBO, and active selection
# ==============================
ols_results = None
if X is not None:
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ Y)
    yhat = X @ beta
    resid = Y - yhat
    n, k = X.shape
    sigma2 = (resid @ resid) / max(1, (n - k))
    var_beta = sigma2 * XtX_inv
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    var_beta_hc = XtX_inv @ S @ XtX_inv
    var_beta_hc *= n / max(1, (n - k))  # HC1

    se_ols = np.sqrt(np.diag(var_beta))
    se_hc1 = np.sqrt(np.diag(var_beta_hc))
    ss_tot = ((Y - Y.mean())**2).sum()
    ss_res = (resid**2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(1, (n - k))

    def pval_from_t(tval):
        return 2 * (1 - 0.5 * (1 + math.erf(abs(tval) / math.sqrt(2))))
    se_use = se_hc1 if use_robust else se_ols
    tvals = beta / np.where(se_use==0, np.nan, se_use)
    pvals = np.array([pval_from_t(tv) if np.isfinite(tv) else np.nan for tv in tvals])

    ols_results = {
        "beta": beta, "se": se_use, "t": tvals, "p": pvals,
        "r2": r2, "adj_r2": adj_r2, "yhat": yhat, "resid": resid,
        "names": ["Intercept"] + pred_vars, "n": n, "k": k, "robust": use_robust
    }
    # Attach additional metrics for comparison table compatibility
    try:
        _mets_ols = compute_metrics(Y, yhat, n_params=k)
        ols_results.update({
            "rmse": _mets_ols.get("rmse"),
            "aic":  _mets_ols.get("aic"),
            "bic":  _mets_ols.get("bic"),
        })
    except Exception:
        pass

# === Advanced models (QKR / QUBO) and active selection ===
qkr_results = None
qubo_results = None

Xk_for_adv, _, _ = build_features(pred_vars, use_standardize=standardize) if (pred_vars and X is not None) else (None, None, None)

if estimator in ("QKR (quantum kernel)", "Compare (all)") and Xk_for_adv is not None:
    qkr_results = fit_qkr(Xk_for_adv, Y, gamma=qkr_gamma, depth=qkr_depth, ridge=qkr_ridge, seed=qkr_seed)

if estimator in ("QUBO selector", "Compare (all)") and Xk_for_adv is not None:
    qubo_results = fit_qubo_selector(Xk_for_adv, pred_vars, Y, lam=qubo_lambda, criterion=qubo_criterion, max_p=qubo_max_p)

res_active = None
active_name = None
if estimator == "OLS":
    res_active = ols_results; active_name = "OLS"
elif estimator == "QKR (quantum kernel)":
    res_active = qkr_results or ols_results; active_name = "QKR"
elif estimator == "QUBO selector":
    res_active = qubo_results or ols_results; active_name = "QUBO"
else:  # Compare (all)
    candidates = [("OLS", ols_results), ("QKR", qkr_results), ("QUBO", qubo_results)]
    candidates = [(n, r) for n, r in candidates if r is not None]
    if candidates:
        active_name, res_active = max(candidates, key=lambda t: t[1]["adj_r2"])  # choose best Adj R² by default
    else:
        res_active = ols_results; active_name = "OLS"

# ==============================
# 4D Hyperfield Construction
# ==============================
# Econometric energy w: Combined by default = 0.7*resid + 0.3*(fitted - mean)
if 'res_active' in globals() and res_active is not None:
    resid = res_active["resid"]
    fitted = res_active["yhat"]
    w_combined = 0.7 * resid + 0.3 * (fitted - np.mean(fitted))
    if color_source_4d == "Residuals":
        color_vals = resid
    elif color_source_4d == "Fitted":
        color_vals = fitted
    else:
        color_vals = w_combined
else:
    w_combined = np.random.normal(0, 1, n_particles)
    color_vals = w_combined

# Hyper-angle α from w via logistic mapping
sigma_w = np.std(w_combined) + 1e-12
alpha = np.pi * (1.0 / (1.0 + np.exp(-w_combined / sigma_w)))

# 4D coords (x4, y4, z4, w4)
def project_4d_to_3d(r_, th_, ph_, alpha_, lam_, orthogonal=False):
    # 4D embedding
    x4 = r_ * np.sin(ph_) * np.cos(th_) * np.cos(alpha_)
    y4 = r_ * np.sin(ph_) * np.sin(th_) * np.cos(alpha_)
    z4 = r_ * np.cos(ph_) * np.cos(alpha_)
    w4 = r_ * np.sin(alpha_)
    # Projection
    R = np.max(r_)
    if orthogonal:
        pf = 1.0
    else:
        pf = 1.0 / (1.0 + lam_ * (w4 / R))
    return x4 * pf, y4 * pf, z4 * pf

# Base 3D projection from 4D
x_proj, y_proj, z_proj = project_4d_to_3d(radii, theta, phi, alpha, projection_intensity, orthogonal_proj)

# Colours (use scalar array + colorscale)
# (Per-point opacity arrays are not supported; keep a scalar or embed in RGBA if needed.)
# We'll use a constant opacity with varying colour scale.
opacity_scalar = 0.85

# ==============================
# Hover text
# ==============================
if show_hover:
    deg = 180.0/np.pi
    alpha_deg = (alpha * 180.0 / np.pi)
    if 'res_active' in globals() and res_active is not None:
        yhat = res_active["yhat"]
        resid = res_active["resid"]
        hover_text = [
            f"id:{i} | {dep_label} | r:{rv:.2f} | θ:{(th*deg):.1f}° | φ:{(ph*deg):.1f}° | α:{ad:.1f}° | yhat:{yh:.3f} | resid:{rs:.3f}"
            for i, (rv, th, ph, ad, yh, rs) in enumerate(zip(radii, theta, phi, alpha_deg, yhat, resid))
        ]
    else:
        hover_text = [
            f"id:{i} | r:{rv:.2f} | θ:{(th*deg):.1f}° | φ:{(ph*deg):.1f}° | α:{ad:.1f}°"
            for i, (rv, th, ph, ad) in enumerate(zip(radii, theta, phi, alpha_deg))
        ]
else:
    hover_text = None

# ==============================
# Build Figure + Frames
# ==============================
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x_proj, y=y_proj, z=z_proj,
    mode='markers',
    marker=dict(
        size=sizes,
        color=color_vals,
        colorscale=colorscale,
        opacity=opacity_scalar,
        showscale=show_scale
    ),
    text=hover_text,
    hoverinfo='text' if show_hover else 'skip'
))

# Animation frames: 3D rotation + 4D precession + breathing
frames = []
if animate:
    for t in range(frames_total):
        # 3D rotation about Z and breathing
        angle3d = rot_speed * 2*np.pi * (t / max(1, frames_total-1))
        pulse   = 1.0 + breathe_amp * np.sin(2*np.pi * (t / max(1, frames_total-1)))
        r_t     = radii * pulse
        # 4D precession (rotate α)
        alpha_t = alpha + w_rotation_speed * 2*np.pi * (t / max(1, frames_total-1))

        # Recompute projection with updated r and α
        x4 = r_t * np.sin(phi) * np.cos(theta) * np.cos(alpha_t)
        y4 = r_t * np.sin(phi) * np.sin(theta) * np.cos(alpha_t)
        z4 = r_t * np.cos(phi) * np.cos(alpha_t)
        w4 = r_t * np.sin(alpha_t)

        # Perspective / orthogonal factor
        Rm = np.max(r_t)
        if orthogonal_proj:
            pf = 1.0
        else:
            pf = 1.0 / (1.0 + projection_intensity * (w4 / Rm))

        # Apply 3D rotation in XY plane
        cp, sp = np.cos(angle3d), np.sin(angle3d)
        x_rot = (x4 * pf) * cp - (y4 * pf) * sp
        y_rot = (x4 * pf) * sp + (y4 * pf) * cp
        z_rot = (z4 * pf)

        frames.append(go.Frame(
            name=str(t),
            data=[go.Scatter3d(
                x=x_rot, y=y_rot, z=z_rot,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=color_vals,           # keep colour stable to econometric choice
                    colorscale=colorscale,
                    opacity=opacity_scalar,
                    showscale=show_scale
                ),
                text=hover_text,
                hoverinfo='text' if show_hover else 'skip'
            )]
        ))
    fig.frames = frames

# Optional nucleus for spatial reference
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers',
    marker=dict(size=max(18, int(size_max*2)), color='white', symbol='circle', opacity=0.95),
    hoverinfo="skip",
    showlegend=False
))

fig.update_layout(
    title="4-Dimensional Quantum-Econometric HyperSpace — Concept (Projected Hypersphere)",
    scene=dict(
        xaxis=dict(visible=show_axes, showgrid=False, zeroline=False),
        yaxis=dict(visible=show_axes, showgrid=False, zeroline=False),
        zaxis=dict(visible=show_axes, showgrid=False, zeroline=False),
        bgcolor=bg_color,
        aspectmode=('manual' if manual_aspect else 'data'),
        aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z)
    ),
    paper_bgcolor=bg_color if bg_color != "white" else "white",
    margin=dict(l=0, r=0, b=0, t=50),
    width=canvas_width,
    height=canvas_height,
    updatemenus=[dict(
        type='buttons', showactive=False,
        buttons=[
            dict(label='Play', method='animate',
                 args=[None, {"fromcurrent": True, "frame": {"duration": int(frame_duration_ms), "redraw": True}, "transition": {"duration": 0}}]),
            dict(label='Pause', method='animate',
                 args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}])
        ]
    )] if animate else []
)

# ==============================
# Streamlit Layout
# ==============================
st.title("4-Dimensional Quantum-Econometric HyperSpace — Concept")
st.caption("True 4D projection: econometric energy warps a hypersphere (α) and projects to 3D with perspective. Toggle colour source for Residuals / Fitted / Combined.")
st.caption("Developed by Shubhojit Bagchi | Inspired by 4 Dimensional Geometry (an extension of Euclidean Geometry) and Quantum Visualization")

if animate:
    st.caption(f"Anim: {frames_total} frames • 3D rot={rot_speed:.2f} • 3D breathe={breathe_amp:.2f} • 4D precession ω₄={w_rotation_speed:.2f} • λ={projection_intensity:.2f}")
else:
    st.caption("Animation disabled — static hyper-projection shown.")

# Agentic AI env banner (compact)
if enable_ai:
    if HAS_OPENAI:
        st.info("Agentic AI: Active")
    else:
        st.warning("Agentic AI: Inactive Set the environment variable to enable explanations.")

st.plotly_chart(fig, use_container_width=use_container, config={"displaylogo": False})

# === Model comparison panel ===
if estimator == "Compare (all)":
    rows = []
    for name, res in [("OLS", ols_results), ("QKR", qkr_results), ("QUBO", qubo_results)]:
        if res is None: continue
        rows.append({
            "Model": name,
            "Adj R²": res["adj_r2"],
            "R²": res["r2"],
            "RMSE": res["rmse"],
            "AIC": res["aic"],
            "BIC": res["bic"],
            "k": res["k"],
        })
    if rows:
        st.subheader("Model comparison")
        st.dataframe(pd.DataFrame(rows).sort_values(by=["Adj R²"], ascending=False), use_container_width=True)
        st.caption("Active model for colour/4D: " + (active_name or "OLS"))

# === Agentic AI Explanations Panel ===
if enable_ai:
    st.subheader("Agentic AI — Auto Explanation")
    colA, colB = st.columns([1,1])
    with colA:
        explain_btn = st.button("Explain current hyperfield")
    with colB:
        narr_goal = st.text_input("Narrative goal (optional)", "Explain insights for a policy+analytics audience.")
    if explain_btn:
        with st.spinner("Generating explanation..."):
            txt = _call_openai_explain(ai_model, ai_max_tokens, narrative_goal=narr_goal)
        st.write(txt)
    

# Econometrics summary (if available)
if ols_results is not None:
    st.subheader("Econometrics — OLS Summary")
    st.caption(f"Dependent: {dep_label} • Standardize: {'ON' if standardize else 'OFF'} • Robust SE: {'HC1' if use_robust else 'Classical'} • Colour: {color_source_4d} • Estimator: {active_name or estimator}")
    c1, c2, c3 = st.columns(3)
    c1.metric("n (rows)", f"{ols_results['n']}")
    c2.metric("k (params)", f"{ols_results['k']}")
    c3.metric("R² / Adj R²", f"{ols_results['r2']:.4f} / {ols_results['adj_r2']:.4f}")

    coef_df = pd.DataFrame({
        "term": ols_results["names"],
        "beta": ols_results["beta"],
        "se":   ols_results["se"],
        "t":    ols_results["t"],
        "p":    ols_results["p"],
    })
    st.dataframe(coef_df, use_container_width=True)
    st.download_button("Download Coefficients (CSV)", data=coef_df.to_csv(index=False).encode("utf-8"),
                       file_name="ols_coefficients.csv", mime="text/csv")

# Download 4D HTML
buf = io.StringIO()
fig.write_html(buf, include_plotlyjs="cdn", full_html=True)
st.download_button("Download 4D Interactive HTML", data=buf.getvalue().encode("utf-8"),
                   file_name="hyperfield_4d_econometric.html", mime="text/html")
