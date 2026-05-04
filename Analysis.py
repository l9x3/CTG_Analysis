"""
=================================================================
Federated Learning — Fetal Heart Rate (FHR) Estimation
Dataset : IIScFHSDB (PhysioNet)

Algorithms (paper-faithful implementations)
─────────────────────────────────────────────
1. FedAvg     – McMahan et al., AISTATS 2017
               Weighted model averaging; client fraction C;
               E local epochs of mini-batch SGD.

2. FedProx    – Li et al., MLSys 2020
               Adds proximal term (μ/2)‖w−w^t‖² to each
               client's local objective; same aggregation
               as FedAvg.

3. SCAFFOLD   – Karimireddy et al., ICML 2020
               Control-variate (variance reduction) correction
               to eliminate client drift.  Uses Option-II
               control-variate update from the paper:
                 c_i^{new} = c_i − c + Δ_i/(K·η)
               Server: c^{t+1} = c^t + (|S|/N)·Σ(c_i^{new}−c_i)

4. FedNova    – Wang et al., NeurIPS 2020
               Normalised averaging that eliminates objective
               inconsistency under heterogeneous local steps.
               Clients send (Δ_i, τ_i); server computes
               τ_eff = 1/Σ_k p_k/τ_k and aggregates
               w^{t+1} = w^t + η_g·τ_eff·Σ_k p_k·(Δ_k/τ_k).
               τ_k sampled uniformly in [τ_min, τ_max] per round
               to simulate computation heterogeneity.

5. FedCrit-HEA – Novel (this paper)
               Criticality-Aware FL with Hierarchical Edge
               Aggregation.  Three tiers:
               • Client → edge: DP local training (prox +
                 gradient clip + Gaussian noise) + privatised
                 class histogram via Laplace mechanism.
               • Edge server: size-weighted avg of client
                 updates; criticality ρ̄_m from histograms.
               • Global server: criticality-weighted aggregation
                 ω_m ∝ ρ̄_m · N_m, then w^{t+1} update.

Outputs
  convergence_summary.csv
  convergence_curve.csv
  privacy_utility.csv
  client_distribution.csv
  noniid_sensitivity.csv
  fl_results.png
=================================================================
"""

import os, copy, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import librosa

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────
DATA_DIR = "./iiscfhsdb_data"
OUT_DIR  = "./fl_outputs_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# ── GLOBAL FL SETTINGS ────────────────────────────────────────────────
N_ROUNDS        = 80
E               = 5          # local epochs (FedAvg / FedProx / SCAFFOLD)
ETA             = 0.01       # client learning rate η
ETA_G           = 1.0        # server learning rate η_g  (FedNova, SCAFFOLD)
FEDPROX_MU      = 0.1        # proximal coefficient μ
SCAFFOLD_OPTION = "II"       # control-variate update option (II is simpler)
FEDNOVA_TAU_MIN = 3          # min local steps (straggler simulation)
FEDNOVA_TAU_MAX = 10         # max local steps
BATCH           = 64
N_CLIENTS       = 60
CLIENT_FRAC     = 1.0        # fraction of clients sampled per round (C)
N_GROUPS        = 6          # edge servers for FedCrit-HEA
DP_CLIP         = 1.0        # gradient clip norm (C in the algorithm)
DP_SIGMA        = 0.3        # Gaussian noise σ for DP (default, ε≈10)
DP_EPS_H        = 1.0        # Laplace ε for histogram privatisation
DP_EPSILONS     = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
DIRICHLET_ALPHAS= [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
CTG_LABEL_COL   = "nsp"      # normalized (lowercase) label column name in CTG_Dataset.csv
CTG_EXCLUDE_COLS= ("class",) # normalized alternate label column to exclude from features
MIN_SAMPLES_PER_CLIENT = 5
SR              = 2000
SEGMENT_S       = 10
OVERLAP         = 0.5

# ══════════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════
def bandpass(y, lo=25, hi=100, fs=SR, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
    return filtfilt(b, a, y)

def autocorr_fhr(sig, fs=SR, lo=60, hi=200):
    lmin, lmax = int(fs*60/hi), int(fs*60/lo)
    ac = np.correlate(sig, sig, mode="full")[len(sig)-1:]
    ac /= (ac[0] + 1e-9)
    peaks, props = find_peaks(ac[lmin:lmax], height=0)
    if len(peaks) == 0: return np.nan
    return 60.0 * fs / (peaks[np.argmax(props["peak_heights"])] + lmin)

def extract_features(seg, fs=SR):
    from scipy.signal import hilbert
    f = []
    f.append(autocorr_fhr(seg, fs))
    env = np.abs(hilbert(seg))
    f += [env.mean(), env.std(), env.max(), float(pd.Series(env).kurtosis())]
    fft = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), 1/fs)
    sc = np.sum(freqs*fft)/(fft.sum()+1e-9)
    sb = np.sqrt(np.sum(((freqs-sc)**2)*fft)/(fft.sum()+1e-9))
    f += [sc, sb, np.mean(fft[(freqs>=1)&(freqs<=3)]**2)]
    mfcc = librosa.feature.mfcc(y=seg.astype(np.float32), sr=fs, n_mfcc=13)
    for row in mfcc: f += [row.mean(), row.std()]
    f += [librosa.feature.zero_crossing_rate(seg).mean(),
          librosa.feature.rms(y=seg).mean()]
    return np.array(f, dtype=np.float32)

def normalize_columns(columns):
    """Normalize column names for consistent matching."""
    return [c.strip().lower().replace("\ufeff", "").replace(" ", "_") for c in columns]

def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = normalize_columns(df.columns)
    sc = next((c for c in df.columns if "subject" in c or c=="id"), None)
    fc = next((c for c in df.columns if "fhr" in c or "heart_rate" in c or c=="hr"), None)
    if not sc or not fc: return {}
    return {f"subject_{int(r[sc]):02d}": float(r[fc])
            for _, r in df.iterrows() if pd.notna(r[fc])}

def load_ctg_dataset(csv_path, label_col, exclude_cols=None):
    """Load CTG dataset.

    Args:
        csv_path: Path to the CTG CSV file.
        label_col: Column name to use as the output label.
        exclude_cols: Optional iterable of additional label-like columns to exclude.

    Returns:
        Tuple of (X, y, feature_cols) where X is the feature matrix, y is the
        label array, and feature_cols is the list of feature column names.
    """
    df = pd.read_csv(csv_path)
    # Normalize column names: trim whitespace, lower-case, and strip BOM if present.
    df.columns = normalize_columns(df.columns)
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {csv_path}")
    y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(np.float32)
    if np.all(np.isnan(y)):
        raise ValueError(f"All '{label_col}' values are NaN in {csv_path}")
    exclude = {label_col}
    if exclude_cols:
        exclude.update(c.lower() for c in exclude_cols)
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    mask = ~np.isnan(y)
    return X[mask], y[mask], feature_cols

def standardize_features(X):
    """Median-impute NaNs and standardize features."""
    col_med = np.nanmedian(X, 0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = col_med[j]
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# ══════════════════════════════════════════════════════════════════════
#  MLP (pure NumPy)  Architecture: D → 64 → 32 → 1
# ══════════════════════════════════════════════════════════════════════
class MLP:
    def __init__(self, D, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/D),   (D,  64)).astype(np.float32)
        self.b1 = np.zeros(64,  dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2/64),  (64, 32)).astype(np.float32)
        self.b2 = np.zeros(32,  dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(2/32),  (32,  1)).astype(np.float32)
        self.b3 = np.zeros(1,   dtype=np.float32)

    def forward(self, X):
        self.X   = X
        self.z1  = X @ self.W1 + self.b1;  self.a1 = np.maximum(0, self.z1)
        self.z2  = self.a1 @ self.W2 + self.b2; self.a2 = np.maximum(0, self.z2)
        self.out = (self.a2 @ self.W3 + self.b3)
        return self.out.ravel()

    def predict(self, X):
        z1 = np.maximum(0, X @ self.W1 + self.b1)
        z2 = np.maximum(0, z1 @ self.W2 + self.b2)
        return (z2 @ self.W3 + self.b3).ravel()

    def gradients(self, X, y):
        """Compute and return gradients of MSE loss (does NOT update weights)."""
        n   = len(y)
        self.forward(X)
        err = (self.out.ravel() - y).reshape(-1,1) / n
        dW3 = self.a2.T @ err;        db3 = err.sum(0)
        dz2 = (err @ self.W3.T) * (self.z2 > 0)
        dW2 = self.a1.T @ dz2;        db2 = dz2.sum(0)
        dz1 = (dz2 @ self.W2.T) * (self.z1 > 0)
        dW1 = self.X.T @ dz1;         db1 = dz1.sum(0)
        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)

    def apply_update(self, delta, lr=1.0, clip=5.0):
        """Apply a pre-computed parameter update dict (gradient or correction)."""
        for k, v in delta.items():
            np.clip(v, -clip, clip, out=v)
            getattr(self, k).__isub__(lr * v)

    def get_weights(self):
        return {k: getattr(self, k).copy()
                for k in ("W1","b1","W2","b2","W3","b3")}

    def set_weights(self, w):
        for k, v in w.items(): getattr(self, k)[:] = v

# ══════════════════════════════════════════════════════════════════════
#  AGGREGATION HELPERS
# ══════════════════════════════════════════════════════════════════════
def weighted_avg(weight_list, sizes):
    """FedAvg-style weighted average of weight dicts."""
    tot = sum(sizes)
    return {k: sum(w[k]*s for w,s in zip(weight_list,sizes))/tot
            for k in weight_list[0]}

def uniform_avg(weight_list):
    """Simple uniform average (used in SCAFFOLD server step)."""
    n = len(weight_list)
    return {k: sum(w[k] for w in weight_list)/n for k in weight_list[0]}

def dict_add(a, b, scale_b=1.0):
    return {k: a[k] + scale_b * b[k] for k in a}

def dict_sub(a, b):
    return {k: a[k] - b[k] for k in a}

def dict_scale(d, s):
    return {k: v * s for k, v in d.items()}

def zeros_like(w):
    return {k: np.zeros_like(v) for k, v in w.items()}

def safe_mae(y_true, y_pred):
    y_pred = np.nan_to_num(y_pred, nan=float(np.nanmean(y_true)))
    return mean_absolute_error(y_true, y_pred)

def global_mae(model, client_data):
    preds, trues = [], []
    for X_c, y_c in client_data:
        if len(X_c) == 0: continue
        preds.append(model.predict(X_c)); trues.append(y_c)
    if not preds: return np.nan
    return safe_mae(np.concatenate(trues), np.concatenate(preds))

def client_maes(model, client_data):
    return [safe_mae(y_c, model.predict(X_c)) if len(X_c)>0 else np.nan
            for X_c, y_c in client_data]

# ══════════════════════════════════════════════════════════════════════
#  ALGORITHM 1 — FedAvg   (McMahan et al. 2017)
# ══════════════════════════════════════════════════════════════════════
def fedavg(client_data, n_rounds=N_ROUNDS, E=E, lr=ETA, C=CLIENT_FRAC, seed=0):
    """
    Algorithm:
      Server broadcasts w^t to sampled clients.
      Client k: w_{k,0}=w^t; run E epochs SGD; send w_{k,E}.
      Server:   w^{t+1} = Σ_k (n_k/n) w_{k,E}
    """
    np.random.seed(seed)
    D = client_data[0][0].shape[1]
    gm = MLP(D, seed); gw = gm.get_weights()
    active = [i for i,(X,y) in enumerate(client_data) if len(X)>=2]
    curve  = []

    for _ in range(n_rounds):
        sampled = np.random.choice(active, max(1,int(len(active)*C)), replace=False)
        lws, szs = [], []
        for k in sampled:
            X_c, y_c = client_data[k]
            lm = MLP(D); lm.set_weights(copy.deepcopy(gw))
            # E epochs of mini-batch SGD
            for _ in range(E):
                idx = np.random.permutation(len(X_c))
                for s in range(0, len(X_c), BATCH):
                    b = idx[s:s+BATCH]
                    g = lm.gradients(X_c[b], y_c[b])
                    lm.apply_update(g, lr)
            lws.append(lm.get_weights()); szs.append(len(X_c))
        gw = weighted_avg(lws, szs)
        gm.set_weights(gw)
        curve.append(global_mae(gm, client_data))

    return curve, client_maes(gm, client_data)


# ══════════════════════════════════════════════════════════════════════
#  ALGORITHM 2 — FedProx   (Li et al. 2020)
# ══════════════════════════════════════════════════════════════════════
def fedprox(client_data, n_rounds=N_ROUNDS, E=E, lr=ETA, mu=FEDPROX_MU,
            C=CLIENT_FRAC, seed=0):
    """
    Local objective: h_k(w;w^t) = F_k(w) + (μ/2)‖w−w^t‖²
    Gradient:        ∇h_k = ∇F_k(w) + μ(w−w^t)
    Aggregation:     same as FedAvg (weighted avg of w_{k,E})
    """
    np.random.seed(seed)
    D = client_data[0][0].shape[1]
    gm = MLP(D, seed); gw = gm.get_weights()
    active = [i for i,(X,y) in enumerate(client_data) if len(X)>=2]
    curve  = []

    for _ in range(n_rounds):
        sampled = np.random.choice(active, max(1,int(len(active)*C)), replace=False)
        lws, szs = [], []
        for k in sampled:
            X_c, y_c = client_data[k]
            lm = MLP(D); lm.set_weights(copy.deepcopy(gw))
            w_t = copy.deepcopy(gw)   # reference point for proximal term
            for _ in range(E):
                idx = np.random.permutation(len(X_c))
                for s in range(0, len(X_c), BATCH):
                    b = idx[s:s+BATCH]
                    g = lm.gradients(X_c[b], y_c[b])
                    # add proximal gradient: μ(w − w^t)
                    prox = {key: mu*(getattr(lm,key) - w_t[key]) for key in g}
                    g_total = dict_add(g, prox)
                    lm.apply_update(g_total, lr)
            lws.append(lm.get_weights()); szs.append(len(X_c))
        gw = weighted_avg(lws, szs)
        gm.set_weights(gw)
        curve.append(global_mae(gm, client_data))

    return curve, client_maes(gm, client_data)


# ══════════════════════════════════════════════════════════════════════
#  ALGORITHM 3 — SCAFFOLD   (Karimireddy et al. 2020, Option II)
# ══════════════════════════════════════════════════════════════════════
def scaffold(client_data, n_rounds=N_ROUNDS, K=E*8, lr=ETA, lr_g=ETA_G,
             C=CLIENT_FRAC, seed=0):
    """
    Control variates: server c; client c_i (all init to 0).

    Client i local step k:
      g_corrected = ∇F_i(y) − c_i + c
      y ← y − η·g_corrected

    After K steps (Option II):
      Δ_i = y_{i,K} − x^t
      c_i^new = c_i − c − Δ_i / (K·η)
      send (Δ_i, c_i^new − c_i) to server

    Server:
      x^{t+1} = x^t + (η_g/|S|)·Σ Δ_i      (uniform avg of deltas)
      c^{t+1} = c^t + (|S|/N)·Σ (c_i^new − c_i)
    """
    np.random.seed(seed)
    D = client_data[0][0].shape[1]
    gm = MLP(D, seed); gw = gm.get_weights()
    N  = len(client_data)
    # server and client control variates (init = 0)
    c  = zeros_like(gw)
    ci = [zeros_like(gw) for _ in range(N)]
    active = [i for i,(X,y) in enumerate(client_data) if len(X)>=2]
    curve  = []

    for _ in range(n_rounds):
        S = np.random.choice(active, max(1,int(len(active)*C)), replace=False)
        deltas, dc_list = [], []

        for k in S:
            X_c, y_c = client_data[k]
            lm = MLP(D); lm.set_weights(copy.deepcopy(gw))
            c_k = ci[k]

            steps = 0
            for _ in range(E):
                idx = np.random.permutation(len(X_c))
                for s in range(0, len(X_c), BATCH):
                    b = idx[s:s+BATCH]
                    g = lm.gradients(X_c[b], y_c[b])
                    # corrected gradient: g − c_i + c
                    correction = {key: -c_k[key] + c[key] for key in g}
                    g_corr = dict_add(g, correction)
                    lm.apply_update(g_corr, lr)
                    steps += 1

            K_actual = max(steps, 1)
            lw   = lm.get_weights()
            delta_k = dict_sub(lw, gw)                      # Δ_i = w_{k,K} − w^t

            # Option-II control-variate update
            # c_i^new = c_i − c − Δ_i / (K·η)
            c_k_new = {key: c_k[key] - c[key] - delta_k[key]/(K_actual*lr)
                       for key in gw}
            dc_k = dict_sub(c_k_new, c_k)                   # c_i^new − c_i

            deltas.append(delta_k)
            dc_list.append(dc_k)
            ci[k] = c_k_new   # update local control variate

        # Server model update (uniform avg of deltas)
        avg_delta = {key: np.mean([d[key] for d in deltas], axis=0) for key in gw}
        gw = dict_add(gw, avg_delta, scale_b=lr_g)

        # Server control-variate update
        avg_dc = {key: np.mean([d[key] for d in dc_list], axis=0) for key in gw}
        c = dict_add(c, avg_dc, scale_b=len(S)/N)

        gm.set_weights(gw)
        curve.append(global_mae(gm, client_data))

    return curve, client_maes(gm, client_data)


# ══════════════════════════════════════════════════════════════════════
#  ALGORITHM 4 — FedNova   (Wang et al. 2020)
# ══════════════════════════════════════════════════════════════════════
def fednova(client_data, n_rounds=N_ROUNDS, lr=ETA, lr_g=ETA_G,
            tau_min=FEDNOVA_TAU_MIN, tau_max=FEDNOVA_TAU_MAX,
            C=CLIENT_FRAC, seed=0):
    """
    Each client k performs τ_k local SGD steps (heterogeneous).
    τ_k ~ Uniform[τ_min, τ_max] to simulate computation heterogeneity.

    Normalised gradient direction:
      a_k = Δ_k / τ_k    where Δ_k = w_{k,τ_k} − w^t

    Effective local steps:
      τ_eff = 1 / Σ_k p_k / τ_k

    Server update:
      w^{t+1} = w^t + η_g · τ_eff · Σ_k p_k · a_k
               = w^t + η_g · τ_eff · Σ_k p_k · Δ_k/τ_k
    """
    np.random.seed(seed)
    D = client_data[0][0].shape[1]
    gm = MLP(D, seed); gw = gm.get_weights()
    sizes = np.array([len(y) for _,y in client_data], dtype=np.float64)
    active = [i for i,(X,y) in enumerate(client_data) if len(X)>=2]
    curve  = []

    for _ in range(n_rounds):
        S = np.random.choice(active, max(1,int(len(active)*C)), replace=False)
        a_list, p_list, tau_list = [], [], []

        for k in S:
            X_c, y_c = client_data[k]
            tau_k = int(np.random.randint(tau_min, tau_max+1))  # heterogeneous steps
            lm = MLP(D); lm.set_weights(copy.deepcopy(gw))

            step = 0
            done = False
            for _ in range(E * 10):  # enough epochs to cover tau_k
                if done: break
                idx = np.random.permutation(len(X_c))
                for s in range(0, len(X_c), BATCH):
                    if step >= tau_k: done = True; break
                    b = idx[s:s+BATCH]
                    g = lm.gradients(X_c[b], y_c[b])
                    lm.apply_update(g, lr)
                    step += 1

            delta_k = dict_sub(lm.get_weights(), gw)          # Δ_k
            a_k     = dict_scale(delta_k, 1.0/tau_k)          # normalised: Δ_k/τ_k
            a_list.append(a_k)
            p_list.append(sizes[k])
            tau_list.append(tau_k)

        p_arr  = np.array(p_list); p_arr /= p_arr.sum()        # p_k = n_k/n (over sampled)
        tau_arr = np.array(tau_list, dtype=np.float64)
        tau_eff = 1.0 / np.sum(p_arr / tau_arr)                # τ_eff

        # Σ_k p_k · a_k
        agg = {key: sum(p*a[key] for p,a in zip(p_arr,a_list)) for key in gw}
        gw  = dict_add(gw, agg, scale_b=lr_g * tau_eff)

        gm.set_weights(gw)
        curve.append(global_mae(gm, client_data))

    return curve, client_maes(gm, client_data)


# ══════════════════════════════════════════════════════════════════════
#  ALGORITHM 5 — FedCrit-HEA   (this paper)
# ══════════════════════════════════════════════════════════════════════
def _criticality(hist, eps_h=DP_EPS_H, n_bins=10):
    """
    Eq.(criticality): ρ = 1 − H_norm(h̃)
    H_norm = Shannon entropy / log(n_bins)  (in [0,1])
    hist: privatised class histogram (after Laplace noise)
    Returns ρ ∈ [0,1] where 1 = maximally non-IID (concentrated).
    """
    h = np.maximum(hist, 1e-9)
    h = h / h.sum()
    H = -np.sum(h * np.log(h + 1e-12))
    H_max = np.log(n_bins)
    return float(1.0 - H / H_max)

def _crit_weight(rho_list, N_list):
    """
    Eq.(crit_weight): ω_m = (ρ̄_m · N_m) / Σ (ρ̄_{m'} · N_{m'})
    """
    scores = np.array([r * n for r, n in zip(rho_list, N_list)], dtype=np.float64)
    scores = np.maximum(scores, 1e-9)
    return scores / scores.sum()

def fedcrit_hea(client_data, n_rounds=N_ROUNDS, E=E, lr=ETA,
                mu=FEDPROX_MU, clip=DP_CLIP, sigma=DP_SIGMA,
                eps_h=DP_EPS_H, n_groups=N_GROUPS,
                C=CLIENT_FRAC, seed=0, n_bins=10):
    """
    Hierarchical structure:
      Clients → M edge servers → global server

    Client (parallel):
      • FedProx local training (proximal term μ/2‖w−w^t‖²)
      • DP: clip Δ_k to norm C; add N(0, σ²C²I)
      • Class histogram h_k (binned FHR) + Lap(1/ε_h) noise
      • Send (Δ̃_k, h̃_k) to edge server

    Edge server m:
      • Δ̃_m^edge = Σ_{k∈G_m} n_k Δ̃_k / Σ n_k   (size-weighted avg)
      • ρ̄_m from {h̃_k}: group-level criticality
      • Send (Δ̃_m^edge, ρ̄_m) to global server

    Global server:
      • ω_m = criticality-based weight
      • w^{t+1} = w^t + Σ_m ω_m · Δ̃_m^edge
      • Convergence check: ‖w^{t+1}−w^t‖₂ < τ
    """
    np.random.seed(seed)
    D = client_data[0][0].shape[1]
    gm = MLP(D, seed); gw = gm.get_weights()

    # Partition clients into M equal groups (edge servers)
    active  = [i for i,(X,y) in enumerate(client_data) if len(X)>=2]
    groups  = np.array_split(active, n_groups)

    # Build FHR histograms for each client (true, not privatised)
    y_all_vals = np.concatenate([y for X,y in client_data if len(y)>0])
    bins = np.linspace(y_all_vals.min()-1e-3, y_all_vals.max()+1e-3, n_bins+1)

    def client_hist(y_c):
        h, _ = np.histogram(y_c, bins=bins, density=False)
        return h.astype(np.float64) / (h.sum() + 1e-9)

    curve = []

    for rnd in range(n_rounds):
        edge_deltas, edge_rhos, edge_Ns = [], [], []

        # ── Edge-server loop ────────────────────────────────
        for group in groups:
            # sample C fraction from this group
            sampled_g = group if C >= 1.0 else np.random.choice(
                group, max(1, int(len(group)*C)), replace=False)
            if len(sampled_g) == 0: continue

            client_deltas_dp, client_sizes, group_hists = [], [], []

            # ── Client local training ────────────────────────
            for k in sampled_g:
                X_c, y_c = client_data[k]
                lm = MLP(D); lm.set_weights(copy.deepcopy(gw))
                w_t = copy.deepcopy(gw)

                # FedProx local training
                for _ in range(E):
                    idx = np.random.permutation(len(X_c))
                    for s in range(0, len(X_c), BATCH):
                        b = idx[s:s+BATCH]
                        g = lm.gradients(X_c[b], y_c[b])
                        prox = {key: mu*(getattr(lm,key)-w_t[key]) for key in g}
                        lm.apply_update(dict_add(g, prox), lr)

                delta_k = dict_sub(lm.get_weights(), gw)      # Δ_k

                # DP: gradient clipping
                delta_norm = np.sqrt(sum(np.sum(v**2) for v in delta_k.values()))
                scale = min(1.0, clip / (delta_norm + 1e-9))
                hat_delta = dict_scale(delta_k, scale)         # Δ̂_k (clipped)

                # DP: Gaussian noise (Δ̃_k)
                tilde_delta = {key: hat_delta[key] + np.random.normal(
                    0, sigma*clip, hat_delta[key].shape).astype(np.float32)
                    for key in hat_delta}

                # Privatised class histogram: h̃_k = h_k + Lap(1/ε_h)
                h_k    = client_hist(y_c)
                h_tilde = h_k + np.random.laplace(0, 1.0/eps_h, h_k.shape)

                client_deltas_dp.append(tilde_delta)
                client_sizes.append(len(X_c))
                group_hists.append(h_tilde)

            # ── Edge aggregation ────────────────────────────
            szs = np.array(client_sizes, dtype=np.float64)
            # Size-weighted average of DP updates
            delta_edge = {key: np.sum([
                (szs[i]/szs.sum()) * client_deltas_dp[i][key]
                for i in range(len(client_deltas_dp))
            ], axis=0) for key in gw}

            # Group criticality from privatised histograms
            # ρ̄_m = size-weighted avg of per-client criticality
            rho_m = sum((szs[i]/szs.sum()) * _criticality(group_hists[i], eps_h)
                        for i in range(len(group_hists)))

            edge_deltas.append(delta_edge)
            edge_rhos.append(rho_m)
            edge_Ns.append(int(szs.sum()))

        # ── Global criticality-aware aggregation ─────────────
        omegas = _crit_weight(edge_rhos, edge_Ns)
        agg = {key: sum(omegas[m] * edge_deltas[m][key]
                        for m in range(len(edge_deltas)))
               for key in gw}

        new_gw = dict_add(gw, agg)
        # Convergence check ‖w^{t+1}−w^t‖₂ < τ
        delta_norm = np.sqrt(sum(np.sum((new_gw[k]-gw[k])**2) for k in gw))
        gw = new_gw
        gm.set_weights(gw)
        curve.append(global_mae(gm, client_data))

        if delta_norm < 1e-4:
            # Plateau — pad remaining rounds
            curve += [curve[-1]] * (n_rounds - len(curve))
            break

    return curve, client_maes(gm, client_data)


# ══════════════════════════════════════════════════════════════════════
#  BASELINES
# ══════════════════════════════════════════════════════════════════════
def centralised(client_data, n_rounds=N_ROUNDS, lr=ETA):
    """Upper bound: all data pooled."""
    X_all = np.vstack([X for X,y in client_data if len(X)>0])
    y_all = np.concatenate([y for X,y in client_data if len(y)>0])
    D = X_all.shape[1]; m = MLP(D, 42); curve = []
    for _ in range(n_rounds):
        idx = np.random.permutation(len(X_all))
        for s in range(0, len(X_all), BATCH):
            b = idx[s:s+BATCH]
            m.apply_update(m.gradients(X_all[b], y_all[b]), lr)
        curve.append(safe_mae(y_all, m.predict(X_all)))
    return curve


# ══════════════════════════════════════════════════════════════════════
#  NON-IID PARTITIONING  (Dirichlet)
# ══════════════════════════════════════════════════════════════════════
def dirichlet_partition(X, y, n_clients, alpha, seed=42):
    rng = np.random.default_rng(seed)
    bins = np.quantile(y, np.linspace(0,1,11))
    cls  = np.digitize(y, bins[1:-1])
    idx_per_client = [[] for _ in range(n_clients)]
    for c in np.unique(cls):
        ci  = np.where(cls==c)[0]
        props = rng.dirichlet([alpha]*n_clients)
        props = np.round(props * len(ci)).astype(int)
        props[-1] = len(ci) - props[:-1].sum()
        shuf = rng.permutation(ci); ptr = 0
        for kid, cnt in enumerate(props):
            idx_per_client[kid].extend(shuf[ptr:ptr+cnt].tolist()); ptr+=cnt
    D = X.shape[1]
    return [(X[idx], y[idx]) if idx else (np.empty((0,D)), np.array([]))
            for idx in idx_per_client]


# ══════════════════════════════════════════════════════════════════════
#  CONVERGENCE METRICS
# ══════════════════════════════════════════════════════════════════════
def conv_metrics(curve, pct=0.90):
    curve = np.array(curve)
    init  = curve[0]
    r90   = next((i+1 for i,v in enumerate(curve) if v <= init*pct), len(curve))
    conv  = len(curve)
    for i in range(5, len(curve)):
        if all(abs(curve[i-j]-curve[i-j-1])<0.1 for j in range(5)):
            conv = i-4; break
    mae50 = curve[49] if len(curve)>=50 else curve[-1]
    return r90, conv, round(float(mae50),3), round(float(curve[-1]),3)


# ══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════
METHOD_STYLE = {
    "FedAvg":       dict(color="#4FC3F7", lw=2.0, ls="-"),
    "FedProx":      dict(color="#81C784", lw=2.0, ls="--"),
    "SCAFFOLD":     dict(color="#FFB74D", lw=2.0, ls="-."),
    "FedNova":      dict(color="#CE93D8", lw=2.0, ls=":"),
    "FedCrit-HEA":  dict(color="#EF9A9A", lw=2.5, ls="-"),
    "Centralised":  dict(color="#80DEEA", lw=1.5, ls=(0,(3,1,1,1))),
}

def make_plots(curves, privacy_df, client_df, noniid_df, out):
    fig = plt.figure(figsize=(20,14))
    fig.patch.set_facecolor("#10131c")
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(2)]

    def _ax(ax, title, xl, yl, legend=True):
        ax.set_facecolor("#191d2b")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(xl, color="#9aa0b4", fontsize=10); ax.set_ylabel(yl, color="#9aa0b4", fontsize=10)
        ax.tick_params(colors="#9aa0b4", labelsize=9)
        [sp.set_edgecolor("#2d3148") for sp in ax.spines.values()]
        ax.grid(color="#1f2438", lw=0.7, ls="--")
        if legend:
            ax.legend(framealpha=0.25, labelcolor="white", facecolor="#191d2b",
                      edgecolor="#2d3148", fontsize=9, loc="upper right")

    # A — Convergence curves
    ax = axes[0]
    for name, curve in curves.items():
        st = METHOD_STYLE.get(name, dict(color="white", lw=1.5, ls="-"))
        ax.plot(range(1, len(curve)+1), curve, label=name, **st)
    _ax(ax, "A  Convergence Curves", "FL Round", "Global MAE (bpm)")

    # B — Privacy–utility
    ax = axes[1]
    ax.plot(privacy_df["epsilon"], privacy_df["final_mae"],
            color="#FFB74D", marker="o", ms=7, lw=2, label="DP-FedCrit-HEA")
    ax.axvline(1.0,  color="#ffffff40", ls=":", lw=1.2, label="ε=1 (strong)")
    ax.axvline(10.0, color="#ffffff25", ls=":", lw=1.2, label="ε=10 (moderate)")
    ax.set_xscale("log")
    _ax(ax, "B  Privacy–Utility Curve (FedCrit-HEA)", "Privacy Budget ε", "Final MAE (bpm)")

    # C — Client boxplots
    ax = axes[2]
    methods = [m for m in METHOD_STYLE if m in client_df["method"].unique()]
    data_by_m = [client_df[client_df["method"]==m]["mae"].dropna().values for m in methods]
    bp = ax.boxplot(data_by_m, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white",lw=2))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_STYLE[m]["color"]); patch.set_alpha(0.7)
    for w in bp["whiskers"]+bp["caps"]: w.set_color("#6b7194")
    for fl in bp["fliers"]: fl.set(marker="o", color="#6b7194", alpha=0.4, ms=3)
    ax.set_xticks(range(1, len(methods)+1))
    ax.set_xticklabels(methods, color="#9aa0b4", fontsize=8, rotation=20)
    _ax(ax, "C  Client-Level MAE Distribution", "Method", "Client MAE (bpm)", legend=False)

    # D — Non-IID sensitivity
    ax = axes[3]
    for m in ["FedAvg","FedProx","SCAFFOLD","FedNova","FedCrit-HEA"]:
        col = m.lower().replace("-","_").replace(" ","_")+"_mae"
        if col in noniid_df.columns:
            st = METHOD_STYLE.get(m, dict(color="white", lw=1.5, ls="-"))
            ax.plot(noniid_df["alpha"], noniid_df[col],
                    marker="s", ms=6, label=m, **st)
    ax.set_xscale("log"); ax.invert_xaxis()
    _ax(ax, "D  Non-IID Sensitivity (Dirichlet α)", "Dirichlet α  (← more non-IID)", "Final MAE (bpm)")

    fig.suptitle("Federated Learning for FHR Estimation — IIScFHSDB",
                 color="white", fontsize=16, fontweight="bold")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Plot saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("="*65)
    print("  Federated FHR Estimation — FedAvg / FedProx / SCAFFOLD /")
    print("  FedNova / FedCrit-HEA")
    print("="*65)

    # ── 1. Build client data ─────────────────────────────────────────
    ctg_path = os.path.join(os.path.dirname(__file__), "CTG_Dataset.csv")
    use_ctg = os.path.exists(ctg_path)
    rng = np.random.default_rng(42)
    if use_ctg:
        label_display = CTG_LABEL_COL.upper()
        X_all, y_all, feature_cols = load_ctg_dataset(
            ctg_path,
            label_col=CTG_LABEL_COL,
            exclude_cols=CTG_EXCLUDE_COLS,
        )
        if len(X_all) == 0:
            raise ValueError("CTG_Dataset.csv has no usable rows after filtering.")
        order = rng.permutation(len(X_all))
        X_all, y_all = X_all[order], y_all[order]
        X_all = standardize_features(X_all)
        max_clients = max(1, len(X_all) // MIN_SAMPLES_PER_CLIENT)
        n_clients = min(N_CLIENTS, max_clients)
        if n_clients < N_CLIENTS:
            print(f"Note: reducing clients from {N_CLIENTS} to {n_clients} to match sample count.")
        splits = np.array_split(np.arange(len(X_all)), n_clients)
        client_data = [(X_all[idx], y_all[idx]) for idx in splits]
        print(f"Using CTG_Dataset.csv (label={label_display}).")
        print(f"Samples: {len(X_all)}  |  Features: {len(feature_cols)}  |  Clients: {len(client_data)}")
        print(f"{label_display}: {y_all.min():.0f}–{y_all.max():.0f}  mean={y_all.mean():.2f}\n")
    else:
        n_clients = N_CLIENTS
        csv_path = os.path.join(DATA_DIR, "Records.csv")
        labels   = load_labels(csv_path) if os.path.exists(csv_path) else {}
        wav_paths = sorted([os.path.join(DATA_DIR, f"subject_{i:02d}.wav")
                            for i in range(1, n_clients+1)
                            if os.path.exists(os.path.join(DATA_DIR, f"subject_{i:02d}.wav"))])

        use_real = bool(wav_paths)
        if not use_real:
            print("\n⚠  Data not found — using synthetic demo data.\n"
                  "   Run fhr_pipeline.py first to download IIScFHSDB.\n")

        if use_real:
            raw = []
            for wp in wav_paths:
                sid = os.path.basename(wp).replace(".wav","")
                y_c, fs = librosa.load(wp, sr=SR, mono=True)
                y_c = bandpass(y_c); y_c /= (np.abs(y_c).max()+1e-9)
                seg_len = int(SEGMENT_S*SR); hop = int(seg_len*(1-OVERLAP))
                segs = [y_c[s:s+seg_len] for s in range(0,len(y_c)-seg_len+1,hop)]
                if not segs: raw.append((np.empty((0,36)), np.array([]))); continue
                X_c = np.vstack([extract_features(s) for s in segs])
                fhr = labels.get(sid, autocorr_fhr(y_c[:SR*30]))
                y_c_arr = np.full(len(X_c), fhr if not np.isnan(fhr) else 135., np.float32)
                raw.append((X_c, y_c_arr))
            D = 36
        else:
            raw = []
            for i in range(n_clients):
                fhr = rng.uniform(110, 165)
                n   = int(rng.integers(20, 55))
                X_c = rng.normal(0,1,(n,36)).astype(np.float32)
                X_c[:,0] = fhr + rng.normal(0,8,n)
                y_c = (np.full(n,fhr) + rng.normal(0,5,n)).astype(np.float32)
                raw.append((X_c, y_c))
            D = 36

        # Global standardisation
        X_all = np.vstack([X for X,y in raw if len(X)>0])
        y_all = np.concatenate([y for X,y in raw if len(y)>0])
        X_all = standardize_features(X_all)
        ptr=0; client_data=[]
        for X_c,y_c in raw:
            n=len(X_c)
            if n==0: client_data.append((np.empty((0,D)),np.array([]))); continue
            client_data.append((X_all[ptr:ptr+n], y_c)); ptr+=n
        n_clients = len(client_data)

        print(f"Clients: {len(client_data)}  |  Segments: {len(X_all)}")
        print(f"FHR: {y_all.min():.1f}–{y_all.max():.1f} bpm  mean={y_all.mean():.1f}\n")

    # ── 2. Run algorithms ────────────────────────────────────────────
    results = {}
    print("[2/7] FedAvg …");           results["FedAvg"]      = fedavg(client_data)
    print("[3/7] FedProx …");          results["FedProx"]     = fedprox(client_data)
    print("[4/7] SCAFFOLD …");         results["SCAFFOLD"]    = scaffold(client_data)
    print("[5/7] FedNova …");          results["FedNova"]     = fednova(client_data)
    print("[6/7] FedCrit-HEA …");      results["FedCrit-HEA"] = fedcrit_hea(client_data)
    print("[  ] Centralised baseline …"); results["Centralised"]= (centralised(client_data), None)

    curves = {k: (v[0] if isinstance(v,tuple) else v) for k,v in results.items()}
    c_maes = {k: (v[1] if isinstance(v,tuple) else None) for k,v in results.items()}

    # ── 3. Convergence summary ───────────────────────────────────────
    print("\n[6/7] Convergence summary …")
    rows = []
    for name, curve in curves.items():
        r90,conv,mae50,final = conv_metrics(curve)
        rows.append({"Method":name, "Round@90% (MAE↓)":r90,
                     "Conv. Round":conv, "MAE@50":mae50, "Final MAE":final})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT_DIR,"convergence_summary.csv"), index=False)
    print(summ.to_string(index=False))

    # convergence_curve.csv
    ccdf = pd.DataFrame({"Round": range(1, N_ROUNDS+1)})
    for name,curve in curves.items(): ccdf[name] = curve
    ccdf.to_csv(os.path.join(OUT_DIR,"convergence_curve.csv"), index=False)

    # ── 4. Privacy–utility sweep (DP within FedCrit-HEA) ────────────
    print("\n[7a/7] Privacy–utility sweep …")
    pu_rows = []
    for eps in DP_EPSILONS:
        sigma_dp = np.sqrt(2*np.log(1.25/1e-5)) / eps
        print(f"  ε={eps}  σ={sigma_dp:.3f}")
        c_pu, _ = fedcrit_hea(client_data, sigma=float(sigma_dp))
        pu_rows.append({"epsilon":eps, "dp_sigma":round(sigma_dp,4),
                        "final_mae":round(c_pu[-1],3),
                        "mae_at_50":round(c_pu[49] if len(c_pu)>=50 else c_pu[-1],3)})
    pu_df = pd.DataFrame(pu_rows)
    pu_df.to_csv(os.path.join(OUT_DIR,"privacy_utility.csv"), index=False)
    print(pu_df.to_string(index=False))

    # ── 5. Client distribution ───────────────────────────────────────
    cd_rows = []
    for name in ["FedAvg","FedProx","SCAFFOLD","FedNova","FedCrit-HEA"]:
        maes = c_maes.get(name)
        if maes is None: continue
        for cid, m in enumerate(maes):
            cd_rows.append({"client":f"C{cid+1:02d}","method":name,"mae":m})
    cd_df = pd.DataFrame(cd_rows)
    cd_df.to_csv(os.path.join(OUT_DIR,"client_distribution.csv"), index=False)

    # ── 6. Non-IID sensitivity ───────────────────────────────────────
    print("\n[7b/7] Non-IID sensitivity sweep …")
    ni_rows = []
    for alpha in DIRICHLET_ALPHAS:
        print(f"  α={alpha}")
        part = dirichlet_partition(X_all, y_all, n_clients, alpha)
        row  = {"alpha": alpha}
        for name, fn in [("FedAvg",fedavg),("FedProx",fedprox),
                         ("SCAFFOLD",scaffold),("FedNova",fednova),
                         ("FedCrit-HEA",fedcrit_hea)]:
            c_ni, _ = fn(part)
            col = name.lower().replace("-","_").replace(" ","_")+"_mae"
            row[col] = round(c_ni[-1], 3)
        ni_rows.append(row)
    ni_df = pd.DataFrame(ni_rows)
    ni_df.to_csv(os.path.join(OUT_DIR,"noniid_sensitivity.csv"), index=False)
    print(ni_df.to_string(index=False))

    # ── 7. Plot ──────────────────────────────────────────────────────
    make_plots(curves, pu_df, cd_df, ni_df,
               out=os.path.join(OUT_DIR,"fl_results.png"))

    print("\n✅  All outputs in:", OUT_DIR)
    for f in ["convergence_summary.csv","convergence_curve.csv",
              "privacy_utility.csv","client_distribution.csv",
              "noniid_sensitivity.csv","fl_results.png"]:
        print(f"   {f}")

if __name__ == "__main__":
    main()
