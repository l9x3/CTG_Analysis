"""
=================================================================
Federated Learning — CTG Fetal State Classification
Dataset : CTG_Dataset.csv (derived from CTG.xlsx)
Task    : 3-class classification  NSP 1=Normal 2=Suspect 3=Pathological

Algorithms
──────────────────────────────────────────────────────────────
1. FedAvg       McMahan et al., AISTATS 2017
2. FedProx      Li et al., MLSys 2020
3. SCAFFOLD     Karimireddy et al., ICML 2020  (Option-II CVs)
4. FedNova      Wang et al., NeurIPS 2020
5. FedCrit-HEA  Novel — Criticality-Aware + Hierarchical Edge Agg.

Outputs (./ctg_fl_outputs/)
──────────────────────────────────────────────────────────────
  convergence_summary.csv   Method | Round@90%(Acc↑) | Conv.Round | Acc@50 | Final Acc
  convergence_curve.csv     Accuracy per round for every method
  privacy_utility.csv       Final accuracy vs DP budget ε
  client_distribution.csv   Per-client accuracy  (boxplots)
  noniid_sensitivity.csv    Final accuracy vs Dirichlet α
  ctg_fl_results.png        4-panel publication figure
=================================================================
"""

import os, copy, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────
CTG_PATH = "./CTG_Dataset.csv"
OUT_DIR  = "./ctg_fl_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── FEATURES & TARGET ─────────────────────────────────────
FEATURES = [
    "LB","AC","FM","UC","ASTV","MSTV","ALTV","MLTV",
    "DL","DS","DP","DR","Width","Min","Max","Nmax",
    "Nzeros","Mode","Mean","Median","Variance","Tendency"
]
TARGET    = "NSP"       # 1=Normal, 2=Suspect, 3=Pathological
N_CLASSES = 3

# ── FL HYPER-PARAMETERS ────────────────────────────────────
N_ROUNDS         = 80
E                = 5            # local epochs
ETA              = 0.01         # client learning rate
ETA_G            = 1.0          # server learning rate (SCAFFOLD / FedNova)
FEDPROX_MU       = 0.1
FEDNOVA_TAU_MIN  = 3
FEDNOVA_TAU_MAX  = 10
BATCH            = 64
N_CLIENTS        = 10           # simulate 10 hospitals
CLIENT_FRAC      = 1.0
N_GROUPS         = 5            # edge servers for FedCrit-HEA
DP_CLIP          = 1.0
DP_SIGMA_DEFAULT = 0.3          # σ for DP (≈ε=10)
DP_EPS_H         = 1.0          # Laplace ε for histogram
DP_EPSILONS      = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
DIRICHLET_ALPHAS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]


# ══════════════════════════════════════════════════════════
#  1. DATA LOADING
# ══════════════════════════════════════════════════════════
def load_ctg(path=CTG_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET])
    X = df[FEATURES].values.astype(np.float32)
    y = (df[TARGET].values.astype(int) - 1)   # 0-indexed: 0,1,2
    # median imputation for any remaining NaN
    for j in range(X.shape[1]):
        col = X[:, j]
        col[np.isnan(col)] = np.nanmedian(col)
    return X, y


def iid_partition(X, y, n_clients, seed=42):
    """IID: random shuffle and split equally."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    splits = np.array_split(idx, n_clients)
    D = X.shape[1]
    return [(X[s], y[s]) if len(s) > 0
            else (np.empty((0, D)), np.array([], dtype=int))
            for s in splits]


def dirichlet_partition(X, y, n_clients, alpha, seed=42):
    """Non-IID Dirichlet partition per class."""
    rng = np.random.default_rng(seed)
    idx_per_client = [[] for _ in range(n_clients)]
    D = X.shape[1]
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        props = rng.dirichlet([alpha] * n_clients)
        counts = np.round(props * len(ci)).astype(int)
        counts[-1] = len(ci) - counts[:-1].sum()
        shuf = rng.permutation(ci); ptr = 0
        for kid, cnt in enumerate(counts):
            idx_per_client[kid].extend(shuf[ptr:ptr+cnt].tolist())
            ptr += cnt
    return [(X[idx], y[idx]) if idx
            else (np.empty((0, D)), np.array([], dtype=int))
            for idx in idx_per_client]


# ══════════════════════════════════════════════════════════
#  2. MLP  (pure NumPy, 3-class softmax)
#     Architecture : D → 128 → 64 → 3
# ══════════════════════════════════════════════════════════
class MLP:
    def __init__(self, D, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/D),    (D,   128)).astype(np.float32)
        self.b1 = np.zeros(128, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2/128),  (128,  64)).astype(np.float32)
        self.b2 = np.zeros(64,  dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(2/64),   (64,    3)).astype(np.float32)
        self.b3 = np.zeros(3,   dtype=np.float32)

    # ── forward ──────────────────────────────────────────
    def _forward(self, X):
        self.X   = X
        self.z1  = X  @ self.W1 + self.b1
        self.a1  = np.maximum(0, self.z1)          # ReLU
        self.z2  = self.a1 @ self.W2 + self.b2
        self.a2  = np.maximum(0, self.z2)          # ReLU
        self.z3  = self.a2 @ self.W3 + self.b3
        # stable softmax
        z3s      = self.z3 - self.z3.max(axis=1, keepdims=True)
        exp_z    = np.exp(z3s)
        self.prob = exp_z / exp_z.sum(axis=1, keepdims=True)
        return self.prob

    def predict_proba(self, X):
        z1 = np.maximum(0, X  @ self.W1 + self.b1)
        z2 = np.maximum(0, z1 @ self.W2 + self.b2)
        z3 = z2 @ self.W3 + self.b3
        z3 -= z3.max(axis=1, keepdims=True)
        e   = np.exp(z3)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # ── backward  (cross-entropy + optional proximal term) ─
    def gradients(self, X, y_int, prox_w=None, mu=0.0):
        n = len(y_int)
        self._forward(X)
        # cross-entropy gradient w.r.t. logits
        delta = self.prob.copy()
        delta[np.arange(n), y_int] -= 1
        delta /= n

        dW3 = self.a2.T @ delta;   db3 = delta.sum(0)
        dz2 = (delta @ self.W3.T) * (self.z2 > 0)
        dW2 = self.a1.T @ dz2;    db2 = dz2.sum(0)
        dz1 = (dz2 @ self.W2.T)   * (self.z1 > 0)
        dW1 = self.X.T  @ dz1;    db1 = dz1.sum(0)

        grads = dict(W1=dW1,b1=db1,W2=dW2,b2=db2,W3=dW3,b3=db3)

        # proximal penalty gradient: μ(w − w_ref)
        if mu > 0 and prox_w is not None:
            for k in grads:
                grads[k] = grads[k] + mu * (getattr(self, k) - prox_w[k])

        # gradient clipping
        for v in grads.values():
            np.clip(v, -5.0, 5.0, out=v)
        return grads

    def apply_update(self, grads, lr=1.0):
        for k, v in grads.items():
            getattr(self, k).__isub__(lr * v)

    def get_weights(self):
        return {k: getattr(self, k).copy()
                for k in ("W1","b1","W2","b2","W3","b3")}

    def set_weights(self, w):
        for k, v in w.items(): getattr(self, k)[:] = v


# ══════════════════════════════════════════════════════════
#  3. WEIGHT UTILITIES
# ══════════════════════════════════════════════════════════
def weighted_avg(wlist, sizes):
    tot = sum(sizes)
    return {k: sum(w[k]*s for w,s in zip(wlist,sizes))/tot for k in wlist[0]}

def dict_add(a, b, s=1.0):
    return {k: a[k] + s*b[k] for k in a}

def dict_sub(a, b):
    return {k: a[k] - b[k] for k in a}

def dict_scale(d, s):
    return {k: v*s for k,v in d.items()}

def zeros_like(w):
    return {k: np.zeros_like(v) for k,v in w.items()}

def safe_acc(y_true, y_pred):
    if len(y_true) == 0: return np.nan
    return float(accuracy_score(y_true, y_pred))

def safe_f1(y_true, y_pred):
    if len(y_true) == 0: return np.nan
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

def global_acc(model, client_data):
    yt, yp = [], []
    for X_c, y_c in client_data:
        if len(X_c) == 0: continue
        yt.append(y_c); yp.append(model.predict(X_c))
    if not yt: return np.nan
    return safe_acc(np.concatenate(yt), np.concatenate(yp))

def client_accs(model, client_data):
    return [safe_acc(y_c, model.predict(X_c)) if len(X_c)>0 else np.nan
            for X_c,y_c in client_data]


# ══════════════════════════════════════════════════════════
#  4a. FedAvg
# ══════════════════════════════════════════════════════════
def fedavg(client_data, n_rounds=N_ROUNDS, E=E, lr=ETA, C=CLIENT_FRAC, seed=0):
    np.random.seed(seed)
    D = client_data[0][0].shape[1]
    gm = MLP(D, seed); gw = gm.get_weights()
    active = [i for i,(X,y) in enumerate(client_data) if len(X)>=2]
    curve = []

    for _ in range(n_rounds):
        S = np.random.choice(active, max(1,int(len(active)*C)), replace=False)
        lws, szs = [], []
        for k in S:
            X_c,y_c = client_data[k]
            lm = MLP(D); lm.set_weights(copy.deepcopy(gw))
            for _ in range(E):
                idx = np.random.permutation(len(X_c))
                for s in range(0, len(X_c), BATCH):
                    b = idx[s:s+BATCH]
                    lm.apply_update(lm.gradients(X_c[b], y_c[b]), lr)
            lws.append(lm.get_weights()); szs.append(len(X_c))
        gw = weighted_avg(lws, szs); gm.set_weights(gw)
        curve.append(global_acc(gm, client_data))

    return curve, client_accs(gm, client_data)


# ══════════════════════════════════════════════════════════
#  4b. FedProx
# ══════════════════════════════════════════════════════════
def fedprox(client_data, n_rounds=N_ROUNDS, E=E, lr=ETA,
            mu=FEDPROX_MU, C=CLIENT_FRAC, seed=0):
    np.random.seed(seed)
    D = client_data[0][0].shape[1]
    gm = MLP(D, seed); gw = gm.get_weights()
    active = [i for i,(X,y) in enumerate(client_data) if len(X)>=2]
    curve = []

    for _ in range(n_rounds):
        S = np.random.choice(active, max(1,int(len(active)*C)), replace=False)
        lws, szs = [], []
        for k in S:
            X_c,y_c = client_data[k]
            lm = MLP(D); lm.set_weights(copy.deepcopy(gw))
            w_t = copy.deepcopy(gw)
            for _ in range(E):
                idx = np.random.permutation(len(X_c))
                for s in range(0, len(X_c), BATCH):
                    b = idx[s:s+BATCH]
                    lm.apply_update(lm.gradients(X_c[b], y_c[b], prox_w=w_t, mu=mu), lr)
            lws.append(lm.get_weights()); szs.append(len(X_c))
        gw = weighted_avg(lws, szs); gm.set_weights(gw)
        curve.append(global_acc(gm, client_data))

    return curve, client_accs(gm, client_data)


# ══════════════════════════════════════════════════════════
#  4c. SCAFFOLD  (Option-II control variates)
# ══════════════════════════════════════════════════════════
def scaffold(client_data, n_rounds=N_ROUNDS, E=E, lr=ETA,
             lr_g=ETA_G, C=CLIENT_FRAC, seed=0):
    np.random.seed(seed)
    D = client_data[0][0].shape[1]
    gm = MLP(D, seed); gw = gm.get_weights()
    N  = len(client_data)
    c  = zeros_like(gw)
    ci = [zeros_like(gw) for _ in range(N)]
    active = [i for i,(X,y) in enumerate(client_data) if len(X)>=2]
    curve  = []

    for _ in range(n_rounds):
        S = np.random.choice(active, max(1,int(len(active)*C)), replace=False)
        deltas, dc_list = [], []

        for k in S:
            X_c,y_c = client_data[k]
            lm = MLP(D); lm.set_weights(copy.deepcopy(gw))
            c_k = ci[k]; steps = 0

            for _ in range(E):
                idx = np.random.permutation(len(X_c))
                for s in range(0, len(X_c), BATCH):
                    b = idx[s:s+BATCH]
                    g = lm.gradients(X_c[b], y_c[b])
                    # SCAFFOLD correction: g − c_i + c
                    g_corr = {key: g[key] - c_k[key] + c[key] for key in g}
                    for v in g_corr.values(): np.clip(v, -5., 5., out=v)
                    lm.apply_update(g_corr, lr); steps += 1

            K = max(steps, 1)
            delta_k  = dict_sub(lm.get_weights(), gw)
            # Option-II: c_i^new = c_i − c − Δ_i/(K·η)
            c_k_new  = {key: c_k[key] - c[key] - delta_k[key]/(K*lr) for key in gw}
            dc_k     = dict_sub(c_k_new, c_k)
            deltas.append(delta_k); dc_list.append(dc_k); ci[k] = c_k_new

        avg_delta = {key: np.mean([d[key] for d in deltas], 0) for key in gw}
        gw = dict_add(gw, avg_delta, lr_g)
        avg_dc = {key: np.mean([d[key] for d in dc_list], 0) for d in gw}
        c  = dict_add(c, avg_dc, len(S)/N)
        gm.set_weights(gw)
        curve.append(global_acc(gm, client_data))

    return curve, client_accs(gm, client_data)


# ══════════════════════════════════════════════════════════
#  4d. FedNova  (normalised averaging)
# ══════════════════════════════════════════════════════════
def fednova(client_data, n_rounds=N_ROUNDS, lr=ETA, lr_g=ETA_G,
            tau_min=FEDNOVA_TAU_MIN, tau_max=FEDNOVA_TAU_MAX,
            C=CLIENT_FRAC, seed=0):
    np.random.seed(seed)
    D = client_data[0][0].shape[1]
    gm = MLP(D, seed); gw = gm.get_weights()
    sizes  = np.array([len(y) for _,y in client_data], dtype=np.float64)
    active = [i for i,(X,y) in enumerate(client_data) if len(X)>=2]
    curve  = []

    for _ in range(n_rounds):
        S = np.random.choice(active, max(1,int(len(active)*C)), replace=False)
        a_list, p_list, tau_list = [], [], []

        for k in S:
            X_c,y_c = client_data[k]
            tau_k = int(np.random.randint(tau_min, tau_max+1))
            lm = MLP(D); lm.set_weights(copy.deepcopy(gw))
            step = 0
            outer_break = False
            for _ in range(E*10):
                if outer_break: break
                idx = np.random.permutation(len(X_c))
                for s in range(0, len(X_c), BATCH):
                    if step >= tau_k: outer_break=True; break
                    b = idx[s:s+BATCH]
                    lm.apply_update(lm.gradients(X_c[b], y_c[b]), lr); step+=1

            delta_k = dict_sub(lm.get_weights(), gw)
            a_list.append(dict_scale(delta_k, 1.0/tau_k))
            p_list.append(sizes[k]); tau_list.append(tau_k)

        p_arr   = np.array(p_list); p_arr /= p_arr.sum()
        tau_arr = np.array(tau_list, dtype=np.float64)
        tau_eff = 1.0 / np.sum(p_arr / tau_arr)
        agg = {key: sum(p*a[key] for p,a in zip(p_arr,a_list)) for key in gw}
        gw  = dict_add(gw, agg, lr_g * tau_eff)
        gm.set_weights(gw)
        curve.append(global_acc(gm, client_data))

    return curve, client_accs(gm, client_data)


# ══════════════════════════════════════════════════════════
#  4e. FedCrit-HEA  (criticality-aware + hierarchical)
# ══════════════════════════════════════════════════════════
def _criticality(hist):
    h = np.maximum(hist, 1e-9); h /= h.sum()
    H = -np.sum(h * np.log(h + 1e-12))
    return float(1.0 - H / np.log(N_CLASSES))

def _crit_weights(rhos, Ns):
    scores = np.maximum([r*n for r,n in zip(rhos,Ns)], 1e-9)
    return scores / scores.sum()

def fedcrit_hea(client_data, n_rounds=N_ROUNDS, E=E, lr=ETA,
                mu=FEDPROX_MU, clip=DP_CLIP, sigma=DP_SIGMA_DEFAULT,
                eps_h=DP_EPS_H, n_groups=N_GROUPS,
                C=CLIENT_FRAC, seed=0):
    np.random.seed(seed)
    D = client_data[0][0].shape[1]
    gm = MLP(D, seed); gw = gm.get_weights()
    active = [i for i,(X,y) in enumerate(client_data) if len(X)>=2]
    groups = np.array_split(active, n_groups)
    curve  = []

    for rnd in range(n_rounds):
        edge_deltas, edge_rhos, edge_Ns = [], [], []

        for group in groups:
            if len(group) == 0: continue
            cli_deltas, cli_sizes, cli_hists = [], [], []

            for k in group:
                X_c,y_c = client_data[k]
                lm = MLP(D); lm.set_weights(copy.deepcopy(gw))
                w_t = copy.deepcopy(gw)

                # FedProx local training
                for _ in range(E):
                    idx = np.random.permutation(len(X_c))
                    for s in range(0, len(X_c), BATCH):
                        b = idx[s:s+BATCH]
                        lm.apply_update(lm.gradients(X_c[b], y_c[b], prox_w=w_t, mu=mu), lr)

                delta_k = dict_sub(lm.get_weights(), gw)

                # DP: clip
                dnorm = np.sqrt(sum(np.sum(v**2) for v in delta_k.values()))
                sc    = min(1.0, clip/(dnorm+1e-9))
                hat_d = dict_scale(delta_k, sc)

                # DP: Gaussian noise
                tilde_d = {key: hat_d[key] + np.random.normal(
                    0, sigma*clip, hat_d[key].shape).astype(np.float32)
                    for key in hat_d}

                # Privatised class histogram via Laplace
                h = np.bincount(y_c, minlength=N_CLASSES).astype(np.float64)
                h /= (h.sum()+1e-9)
                h += np.random.laplace(0, 1.0/eps_h, h.shape)

                cli_deltas.append(tilde_d); cli_sizes.append(len(X_c))
                cli_hists.append(h)

            szs = np.array(cli_sizes, dtype=np.float64);
            # edge aggregation: size-weighted
            d_edge = {key: np.sum([(szs[i]/szs.sum())*cli_deltas[i][key]
                                    for i in range(len(cli_deltas))], axis=0)
                      for key in gw}
            # group criticality
            rho_m = sum((szs[i]/szs.sum())*_criticality(cli_hists[i])
                        for i in range(len(cli_hists)))

            edge_deltas.append(d_edge); edge_rhos.append(rho_m)
            edge_Ns.append(int(szs.sum()))

        if not edge_deltas:
            curve.append(curve[-1] if curve else 0.0); continue

        # criticality-aware global aggregation
        omegas = _crit_weights(edge_rhos, edge_Ns)
        agg    = {key: sum(omegas[m]*edge_deltas[m][key]
                           for m in range(len(edge_deltas))) for key in gw}
        new_gw = dict_add(gw, agg)
        drift  = np.sqrt(sum(np.sum((new_gw[k]-gw[k])**2) for k in gw))
        gw = new_gw; gm.set_weights(gw)
        curve.append(global_acc(gm, client_data))
        if drift < 1e-4:
            curve += [curve[-1]]*(n_rounds-len(curve)); break

    return curve, client_accs(gm, client_data)


# ══════════════════════════════════════════════════════════
#  5. CENTRALISED BASELINE
# ══════════════════════════════════════════════════════════
def centralised(client_data, n_rounds=N_ROUNDS, lr=ETA):
    X_all = np.vstack([X for X,y in client_data if len(X)>0])
    y_all = np.concatenate([y for X,y in client_data if len(y)>0])
    D = X_all.shape[1]; m = MLP(D, 42); curve = []
    for _ in range(n_rounds):
        idx = np.random.permutation(len(X_all))
        for s in range(0, len(X_all), BATCH):
            b = idx[s:s+BATCH]
            m.apply_update(m.gradients(X_all[b], y_all[b]), lr)
        curve.append(safe_acc(y_all, m.predict(X_all)))
    return curve


# ══════════════════════════════════════════════════════��═══
#  6. CONVERGENCE METRICS
# ══════════════════════════════════════════════════════════
def conv_metrics(curve):
    """
    Round@90%: first round where (acc-acc_0)/(acc_max-acc_0) ≥ 0.90
    Conv.Round: first round after which acc changes <0.002 for 5 rounds
    Acc@50, Final Acc
    """
    curve = np.array(curve)
    acc0  = curve[0]; acc_max = curve.max()
    span  = acc_max - acc0
    tgt   = acc0 + 0.90 * span if span > 1e-4 else acc_max

    r90 = next((i+1 for i,v in enumerate(curve) if v >= tgt), len(curve))

    conv = len(curve)
    for i in range(5, len(curve)):
        if all(abs(curve[i-j]-curve[i-j-1]) < 0.002 for j in range(5)):
            conv = i-4; break

    a50   = curve[49] if len(curve)>=50 else curve[-1]
    final = curve[-1]
    return r90, conv, round(float(a50),4), round(float(final),4)


# ══════════════════════════════════════════════════════════
#  7. PLOTTING
# ══════════════════════════════════════════════════════════
STYLE = {
    "FedAvg":       dict(color="#4FC3F7", lw=2.0, ls="-"),
    "FedProx":      dict(color="#81C784", lw=2.0, ls="--"),
    "SCAFFOLD":     dict(color="#FFB74D", lw=2.0, ls="-.") ,
    "FedNova":      dict(color="#CE93D8", lw=2.0, ls=":"),
    "FedCrit-HEA":  dict(color="#EF9A9A", lw=2.5, ls="-"),
    "Centralised":  dict(color="#80DEEA", lw=1.5, ls=(0,(3,1,1,1))),
}

def make_plots(curves, privacy_df, client_df, noniid_df, out):
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#10131c")
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.34,
                            left=0.07, right=0.97, top=0.93, bottom=0.07)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    def _style(ax, title, xl, yl, legend=True):
        ax.set_facecolor("#191d2b")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(xl, color="#9aa0b4", fontsize=10)
        ax.set_ylabel(yl, color="#9aa0b4", fontsize=10)
        ax.tick_params(colors="#9aa0b4", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#2d3148")
        ax.grid(color="#1f2438", lw=0.7, ls="--")
        if legend:
            ax.legend(framealpha=0.25, labelcolor="white", facecolor="#191d2b",
                      edgecolor="#2d3148", fontsize=9, loc="lower right")

    # ── A: Convergence curves ─────────────────────────────
    ax = axes[0]
    for name, curve in curves.items():
        st = STYLE.get(name, dict(color="white",lw=1.5,ls="-"))
        ax.plot(range(1, len(curve)+1), curve, label=name, **st)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.2f}"))
    _style(ax, "A  Convergence Curves", "FL Round", "Global Accuracy")

    # ── B: Privacy–utility ────────────────────────────────
    ax = axes[1]
    ax.plot(privacy_df["epsilon"], privacy_df["final_acc"],
            color="#EF9A9A", marker="o", ms=7, lw=2.2, label="FedCrit-HEA (DP)")
    ax.axvline(1.0,  color="#ffffff40", ls=":", lw=1.2, label="ε=1 (strong)")
    ax.axvline(10.0, color="#ffffff25", ls=":", lw=1.2, label="ε=10 (moderate)")
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.3f}"))
    _style(ax, "B  Privacy–Utility Curve (FedCrit-HEA)", "Privacy Budget ε", "Final Accuracy")

    # ── C: Client-level boxplots ──────────────────────────
    ax = axes[2]
    methods = [m for m in STYLE if m in client_df["method"].unique()]
    data_bm = [client_df[client_df["method"]==m]["accuracy"].dropna().values
               for m in methods]
    bp = ax.boxplot(data_bm, patch_artist=True, widths=0.55,
                    medianprops=dict(color="white", lw=2))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(STYLE[m]["color"]); patch.set_alpha(0.7)
    for w in bp["whiskers"]+bp["caps"]: w.set_color("#6b7194")
    for fl in bp["fliers"]: fl.set(marker="o", color="#6b7194", alpha=0.4, ms=3)
    ax.set_xticks(range(1, len(methods)+1))
    ax.set_xticklabels(methods, color="#9aa0b4", fontsize=8, rotation=18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.2f}"))
    _style(ax, "C  Client-Level Accuracy Distribution", "Method", "Client Accuracy", legend=False)

    # ── D: Non-IID sensitivity ────────────────────────────
    ax = axes[3]
    for m in ["FedAvg","FedProx","SCAFFOLD","FedNova","FedCrit-HEA"]:
        col = m.lower().replace("-","_").replace(" ","_")+"_acc"
        if col in noniid_df.columns:
            st = STYLE.get(m, dict(color="white",lw=1.5,ls="-"))
            ax.plot(noniid_df["alpha"], noniid_df[col],
                    marker="s", ms=6, label=m, **st)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.3f}"))
    _style(ax, "D  Non-IID Sensitivity (Dirichlet α)",
           "Dirichlet α  (← more non-IID)", "Final Accuracy")

    fig.suptitle("Federated Learning for Fetal State Classification — CTG Dataset",
                 color="white", fontsize=16, fontweight="bold")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Plot saved → {out}")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    print("="*65)
    print("  Federated CTG Classification")
    print("  FedAvg / FedProx / SCAFFOLD / FedNova / FedCrit-HEA")
    print("="*65)

    # ── Load & scale data ─────────────────────────────────
    print("\n[1/7] Loading CTG dataset …")
    X_raw, y_raw = load_ctg()
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_raw).astype(np.float32)

    print(f"  Samples: {len(y_raw)} | Features: {X_raw.shape[1]}")
    for c,n in zip([0,1,2],np.bincount(y_raw)):
        label = ["Normal","Suspect","Pathological"][c]
        print(f"  Class {c} ({label}): {n} ({100*n/len(y_raw):.1f}%)")

    # ── IID partition for main experiments ────────────────
    client_data = iid_partition(X_sc, y_raw, N_CLIENTS)
    print(f"\n  Clients: {N_CLIENTS} | Avg samples/client: "
          f"{len(y_raw)//N_CLIENTS}")

    # ── Run algorithms ────────────────────────────────────
    results = {}
    print("\n[2/7] FedAvg …");          results["FedAvg"]       = fedavg(client_data)
    print("[3/7] FedProx …");          results["FedProx"]      = fedprox(client_data)
    print("[4/7] SCAFFOLD …");         results["SCAFFOLD"]     = scaffold(client_data)
    print("[5/7] FedNova …");          results["FedNova"]      = fednova(client_data)
    print("[6/7] FedCrit-HEA …");      results["FedCrit-HEA"]  = fedcrit_hea(client_data)
    print("     Centralised …");       results["Centralised"]  = (centralised(client_data), None)

    curves = {k: v[0] if isinstance(v,tuple) else v for k,v in results.items()}
    c_accs = {k: v[1] if isinstance(v,tuple) else None for k,v in results.items()}

    # ── Convergence summary ───────────────────────────────
    print("\n[6/7] Convergence summary …")
    rows = []
    for name, curve in curves.items():
        r90,conv,a50,final = conv_metrics(curve)
        rows.append({"Method":name, "Round@90% (Acc↑)":r90,
                     "Conv. Round":conv, "Acc@50":a50, "Final Acc":final})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT_DIR,"convergence_summary.csv"), index=False)
    print(summ.to_string(index=False))

    # convergence_curve.csv
    ccdf = pd.DataFrame({"Round": range(1, N_ROUNDS+1)})
    for name,curve in curves.items(): ccdf[name] = curve
    ccdf.to_csv(os.path.join(OUT_DIR,"convergence_curve.csv"), index=False)

    # ── Privacy–utility sweep (FedCrit-HEA) ──────────────
    print("\n[7a/7] Privacy–utility sweep …")
    pu_rows = []
    for eps in DP_EPSILONS:
        sigma_dp = np.sqrt(2*np.log(1.25/1e-5)) / eps
        print(f"  ε={eps}  σ={sigma_dp:.3f}")
        c_pu, _ = fedcrit_hea(client_data, sigma=float(sigma_dp))
        pu_rows.append({"epsilon":eps, "dp_sigma":round(sigma_dp,4),
                        "final_acc":   round(c_pu[-1],4),
                        "acc_at_50":   round(c_pu[49] if len(c_pu)>=50 else c_pu[-1],4)})
    pu_df = pd.DataFrame(pu_rows)
    pu_df.to_csv(os.path.join(OUT_DIR,"privacy_utility.csv"), index=False)
    print(pu_df.to_string(index=False))

    # ── Client distribution ───────────────────────────────
    cd_rows = []
    for name in ["FedAvg","FedProx","SCAFFOLD","FedNova","FedCrit-HEA"]:
        accs = c_accs.get(name)
        if accs is None: continue
        for cid, acc in enumerate(accs):
            cd_rows.append({"client":f"H{cid+1:02d}","method":name,"accuracy":acc})
    cd_df = pd.DataFrame(cd_rows)
    cd_df.to_csv(os.path.join(OUT_DIR,"client_distribution.csv"), index=False)

    # ── Non-IID sensitivity ───────────────────────────────
    print("\n[7b/7] Non-IID sensitivity sweep …")
    ni_rows = []
    for alpha in DIRICHLET_ALPHAS:
        print(f"  α={alpha}")
        part = dirichlet_partition(X_sc, y_raw, N_CLIENTS, alpha)
        row  = {"alpha": alpha}
        for name, fn in [("FedAvg",fedavg),("FedProx",fedprox),
                         ("SCAFFOLD",scaffold),("FedNova",fednova),
                         ("FedCrit-HEA",fedcrit_hea)]:
            c_ni, _ = fn(part)
            col = name.lower().replace("-","_")+"_acc"
            row[col] = round(c_ni[-1], 4)
        ni_rows.append(row)
    ni_df = pd.DataFrame(ni_rows)
    ni_df.to_csv(os.path.join(OUT_DIR,"noniid_sensitivity.csv"), index=False)
    print(ni_df.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────
    print("\nGenerating ctg_fl_results.png …")
    make_plots(curves, pu_df, cd_df, ni_df,
               out=os.path.join(OUT_DIR,"ctg_fl_results.png"))

    print("\n✅  All outputs saved to", OUT_DIR)
    for f in ["convergence_summary.csv","convergence_curve.csv",
              "privacy_utility.csv","client_distribution.csv",
              "noniid_sensitivity.csv","ctg_fl_results.png"]:
        print(f"   {f}")

if __name__ == "__main__":
    main()
