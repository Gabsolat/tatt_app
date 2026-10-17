"""
Data layer for the TATT sepsis project.

Implements, in plain numpy (no torch dependency):
  * the observation-triplet representation (variable, value, time)   [Sec 3.3.1]
  * a calibrated synthetic ICU cohort generator (offline fallback)
  * a loader for the open-access PhysioNet/CinC 2019 sepsis dataset
  * Sepsis-3 label shifting + negative-window sampling                [Sec 3.5.1]
  * window encoders for TATT (triplets), Bi-LSTM (hourly grid) and XGBoost (tabular)
  * a SOFA proxy used for the interpretability check                  [Sec 3.6.3 (ii)]
"""
import glob
import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Variable catalogue
# ----------------------------------------------------------------------------
VAR_NAMES = ["HR", "MAP", "SBP", "DBP", "RR", "SpO2",
             "Lactate", "Creatinine", "BUN", "Bilirubin", "Platelets", "WBC",
             "Vasopressor", "MechVent", "FluidBolus"]
N_VARS = len(VAR_NAMES)
VAR_IDX = {n: i for i, n in enumerate(VAR_NAMES)}
VITALS = list(range(0, 6))
LABS = list(range(6, 12))
INTERV = list(range(12, 15))
MODALITY = {**{i: "Vital" for i in VITALS}, **{i: "Lab" for i in LABS}, **{i: "Intervention" for i in INTERV}}
UNITS = {"HR": "bpm", "MAP": "mmHg", "SBP": "mmHg", "DBP": "mmHg", "RR": "breaths/min", "SpO2": "%",
         "Lactate": "mmol/L", "Creatinine": "mg/dL", "BUN": "mg/dL", "Bilirubin": "mg/dL",
         "Platelets": "x10^3/uL", "WBC": "x10^3/uL", "Vasopressor": "flag", "MechVent": "flag", "FluidBolus": "flag"}
_MEAN = np.array([85, 80, 120, 62, 18, 96, 1.8, 1.2, 22, 1.0, 200, 10, 0, 0, 0], dtype=np.float32)
_STD = np.array([18, 14, 22, 12, 5, 3.5, 1.6, 1.0, 16, 1.5, 90, 5, 1, 1, 1], dtype=np.float32)

MIN_PRED_TIME = 24.0   # windows start only after the first 24 h (Sec 3.5.1)


def norm(var, val):
    return (np.asarray(val, np.float32) - _MEAN[var]) / _STD[var]


# ----------------------------------------------------------------------------
# Synthetic cohort (offline fallback)
# ----------------------------------------------------------------------------
_EFFECT = {"HR": 35, "MAP": -22, "SBP": -30, "DBP": -14, "RR": 9, "SpO2": -4.0, "Lactate": 3.5,
           "Creatinine": 1.2, "BUN": 14, "Bilirubin": 2.0, "Platelets": -90, "WBC": 7}
_NOISE = {"HR": 3.5, "MAP": 4, "SBP": 6, "DBP": 4, "RR": 1.5, "SpO2": 0.8, "Lactate": 0.15,
          "Creatinine": 0.05, "BUN": 1.5, "Bilirubin": 0.1, "Platelets": 12, "WBC": 0.8}
_LAG = {"Creatinine": 3.0, "BUN": 3.0, "Bilirubin": 3.0, "Platelets": 3.0}
_BOUNDS = {"HR": (30, 190), "MAP": (30, 140), "SBP": (50, 220), "DBP": (20, 120), "RR": (6, 50),
           "SpO2": (70, 100), "Lactate": (0.4, 15), "Creatinine": (0.3, 8), "BUN": (4, 120),
           "Bilirubin": (0.1, 20), "Platelets": (10, 450), "WBC": (1, 40)}


def _severity_fn(rng, septic, T, onset):
    if septic:
        g = rng.uniform(0.55, 1.25)
        lead = rng.uniform(7.0, 13.0)

        def s(t):
            x = np.clip((np.asarray(t) - (onset - lead)) / lead, 0, 1.4)
            return g * (np.exp(3 * x) - 1) / (np.exp(3) - 1)
        return s
    bumps = []
    if rng.random() < 0.30:   # SIRS-like benign excursions -> hard negatives
        for _ in range(rng.integers(1, 3)):
            bumps.append((rng.uniform(10, T - 5), rng.uniform(1.5, 4), rng.uniform(0.25, 0.6)))

    def s(t):
        t = np.asarray(t, float)
        out = np.zeros_like(t)
        for c, w, h in bumps:
            out += h * np.exp(-((t - c) / w) ** 2)
        return out
    return s


def _simulate_patient(rng, pid, septic):
    T = float(rng.uniform(48, 120))
    onset = float(rng.uniform(30, T - 1)) if septic else np.nan
    sev = _severity_fn(rng, septic, T, onset)
    base = {"HR": rng.normal(80, 10), "MAP": rng.normal(85, 8), "RR": rng.normal(16, 2),
            "SpO2": rng.normal(97, 1.2), "Lactate": float(np.exp(rng.normal(np.log(1.0), 0.3))),
            "Creatinine": float(np.clip(rng.normal(0.9, 0.25), 0.4, 2.5)) + (1.3 if rng.random() < 0.08 else 0),
            "BUN": rng.normal(16, 5), "Bilirubin": float(np.clip(rng.normal(0.7, 0.25), 0.2, 2)),
            "Platelets": rng.normal(230, 50), "WBC": rng.normal(9, 2.5)}
    base["SBP"] = base["MAP"] + rng.normal(38, 6)
    base["DBP"] = base["MAP"] - rng.normal(20, 4)
    wander = rng.uniform(0, 6.28, 2)

    def sample(name, t):
        lag = _LAG.get(name, 0.0)
        s = sev(t - lag) + 0.04 * np.sin(t / 7.0 + wander[0])
        v = base[name] + _EFFECT[name] * s
        if name == "Lactate":
            v = base[name] * (1 + 3.0 * np.clip(sev(t), 0, None)) + rng.normal(0, _NOISE[name], len(t))
        else:
            v = v + rng.normal(0, _NOISE[name], len(t))
        lo, hi = _BOUNDS[name]
        return np.clip(v, lo, hi)

    V, X, TT = [], [], []

    def add(j, t, x):
        V.append(np.full(len(t), j, np.int8)); X.append(np.asarray(x, np.float32)); TT.append(np.asarray(t, np.float32))

    for j in VITALS:
        name = VAR_NAMES[j]
        gaps = rng.exponential(0.55, int(T / 0.2)) + 0.08
        t = np.cumsum(gaps) + rng.uniform(0, 0.5)
        t = t[t < T]
        if septic:   # informative sampling: denser charting while deteriorating
            g2 = rng.exponential(0.30, int(T / 0.1)) + 0.05
            t2 = onset - 8 + np.cumsum(g2)
            t = np.sort(np.concatenate([t, t2[(t2 < T) & (t2 > 0)]]))
        keep = rng.random(len(t)) > 0.05     # 5% randomly missing
        t = t[keep]
        add(j, t, sample(name, t))
    for j in LABS:
        name = VAR_NAMES[j]
        t, cur = [], rng.uniform(0, 4)
        while cur < T:
            t.append(cur)
            cur += rng.uniform(6, 14)
        t = np.array(t)
        if name == "Lactate" and septic:
            extra = np.arange(onset - 6 + rng.uniform(0, 2), T, rng.uniform(2, 3.5))
            t = np.sort(np.concatenate([t, extra[extra < T]]))
        add(j, t, sample(name, t))
    # interventions (events: 1 = start / bolus, 0 = stop)
    if septic and rng.random() < 0.85:
        st = float(np.clip(onset + rng.uniform(-1, 1.5), 0, T - 0.1))
        en = min(T, st + rng.uniform(6, 24))
        add(VAR_IDX["Vasopressor"], [st, en], [1, 0])
    elif (not septic) and rng.random() < 0.04:
        st = rng.uniform(5, T - 8)
        add(VAR_IDX["Vasopressor"], [st, st + rng.uniform(4, 10)], [1, 0])
    if rng.random() < 0.15:
        st = rng.uniform(0, T / 2)
        add(VAR_IDX["MechVent"], [st, min(T, st + rng.uniform(10, 40))], [1, 0])
    if septic and rng.random() < 0.6:
        add(VAR_IDX["FluidBolus"], [max(0.5, onset - rng.uniform(0, 3))], [1])
    elif (not septic) and rng.random() < 0.10:
        n = int(rng.integers(1, 3))
        add(VAR_IDX["FluidBolus"], np.sort(rng.uniform(2, T - 1, n)), np.ones(n))
    var = np.concatenate(V); val = np.concatenate(X); tm = np.concatenate(TT)
    o = np.argsort(tm, kind="stable")
    return {"id": pid, "T": T, "onset": onset, "var": var[o], "val": val[o], "time": tm[o]}


def generate_synthetic_cohort(n_patients=2500, prevalence=0.142, seed=0):
    rng = np.random.default_rng(seed)
    n_sep = int(round(n_patients * prevalence))
    flags = np.array([True] * n_sep + [False] * (n_patients - n_sep))
    rng.shuffle(flags)
    return [_simulate_patient(rng, f"S{i:05d}", bool(f)) for i, f in enumerate(flags)]


# ----------------------------------------------------------------------------
# PhysioNet/CinC 2019 loader (open access, no credentialing required)
# ----------------------------------------------------------------------------
_PN_MAP = {"HR": "HR", "MAP": "MAP", "SBP": "SBP", "DBP": "DBP", "RR": "Resp", "SpO2": "O2Sat",
           "Lactate": "Lactate", "Creatinine": "Creatinine", "BUN": "BUN",
           "Bilirubin": "Bilirubin_total", "Platelets": "Platelets", "WBC": "WBC"}


def load_physionet2019(root, min_stay=48, limit=None):
    files = sorted(glob.glob(os.path.join(root, "**", "*.psv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .psv files under {root}. Run download_data.py first.")
    out = []
    for f in files[:limit]:
        df = pd.read_csv(f, sep="|")
        if len(df) < min_stay:
            continue
        t_all = df["ICULOS"].to_numpy(float)
        V, X, TT = [], [], []
        for name, col in _PN_MAP.items():
            x = df[col].to_numpy(float)
            m = ~np.isnan(x)
            V.append(np.full(m.sum(), VAR_IDX[name], np.int8)); X.append(x[m].astype(np.float32)); TT.append(t_all[m].astype(np.float32))
        var, val, tm = np.concatenate(V), np.concatenate(X), np.concatenate(TT)
        o = np.argsort(tm, kind="stable")
        lab = df["SepsisLabel"].to_numpy()
        # In the 2019 challenge SepsisLabel turns 1 six hours BEFORE clinical onset.
        onset = float(t_all[np.argmax(lab > 0)] + 6.0) if lab.max() > 0 else np.nan
        out.append({"id": os.path.splitext(os.path.basename(f))[0], "T": float(t_all[-1]), "onset": onset,
                    "var": var[o], "val": val[o], "time": tm[o]})
    return out


# ----------------------------------------------------------------------------
# Splitting and window sampling
# ----------------------------------------------------------------------------
def patient_split(patients, seed=0, frac=(0.70, 0.15, 0.15)):
    """Patient-level split stratified on sepsis status (Sec 3.2.5)."""
    rng = np.random.default_rng(seed)
    idx = {"train": [], "val": [], "test": []}
    for flag in (True, False):
        ids = [i for i, p in enumerate(patients) if (not np.isnan(p["onset"])) == flag]
        rng.shuffle(ids)
        a = int(frac[0] * len(ids)); b = a + int(frac[1] * len(ids))
        idx["train"] += ids[:a]; idx["val"] += ids[a:b]; idx["test"] += ids[b:]
    return idx


def build_samples(patients, ids, rng, neg_ratio=5, pos_step=0.5, hard_neg_per_pos=1, lead=(4.0, 6.0), buffer=1.0):
    """Label shifting (Eq 3.5) + negative-window sampling (Sec 3.5.1).
    Returns list of (patient_index, t_end, label)."""
    S = []
    sep = [i for i in ids if not np.isnan(patients[i]["onset"])]
    non = [i for i in ids if np.isnan(patients[i]["onset"])]
    n_pos = 0
    for i in sep:
        Ts = patients[i]["onset"]
        n_pp = int(round((lead[1] - lead[0]) / pos_step)) + 1
        ts = np.sort(rng.uniform(Ts - lead[1], Ts - lead[0], n_pp))   # continuous: avoids hourly-grid fingerprint
        ts = ts[ts >= MIN_PRED_TIME]
        for t in ts:
            S.append((i, float(t), 1)); n_pos += 1
        hi = Ts - lead[1] - buffer     # explicit exclusion buffer: t <= Ts - 7
        if hi > MIN_PRED_TIME and len(ts):
            for t in rng.uniform(MIN_PRED_TIME, hi, hard_neg_per_pos * len(ts)):
                S.append((i, float(t), 0))
    have = sum(1 for s in S if s[2] == 0)
    remaining = max(0, neg_ratio * n_pos - have)
    if non and remaining:
        per = remaining / len(non)
        for i in non:
            k = int(per) + int(rng.random() < per - int(per))
            for t in rng.uniform(MIN_PRED_TIME, patients[i]["T"], k):
                S.append((i, float(t), 0))
    return S


def horizon_samples(patients, ids, base_samples, rng, h, neg_ratio=5):
    """Positives at lead time h hours before onset; negatives drawn from the base sample set."""
    pos = []
    for i in ids:
        Ts = patients[i]["onset"]
        t = Ts - h + rng.uniform(-0.5, 0.5)    # jitter so window ends are not tied to the data's sampling grid
        if not np.isnan(Ts) and t >= MIN_PRED_TIME:
            pos.append((i, float(t), 1))
    negs = [s for s in base_samples if s[2] == 0]
    k = min(len(negs), neg_ratio * len(pos))
    sel = rng.choice(len(negs), k, replace=False) if k else []
    return pos + [negs[j] for j in sel]


# ----------------------------------------------------------------------------
# Window encoders
# ----------------------------------------------------------------------------
def _byvar(p):
    if "_bv" not in p:
        p["_bv"] = [(p["time"][p["var"] == j], p["val"][p["var"] == j]) for j in range(N_VARS)]
    return p["_bv"]


def encode_window(p, t_end, W=24.0, max_len=256, pool_h=6.0):
    """Triplet tokens (variable, value, time) for [t_end - W, t_end] + missing-lab mask tokens."""
    tm = p["time"]
    lo = np.searchsorted(tm, t_end - W, "left"); hi = np.searchsorted(tm, t_end, "right")
    if hi - lo > max_len:
        lo = hi - max_len
    v = p["var"][lo:hi].astype(np.int64); x = p["val"][lo:hi]; t = tm[lo:hi].astype(np.float32)
    xn = norm(v, x); flag = np.ones(len(v), np.float32)
    pool = t >= (t_end - pool_h)
    t = (t - (t_end - W)).astype(np.float32)          # hours since window start: removes time-in-stay as a shortcut
    present = np.zeros(N_VARS, bool); present[v] = True
    miss = [j for j in LABS if not present[j]]       # informative absence (Sec 3.3.3)
    if miss:
        v = np.concatenate([v, np.array(miss, np.int64)])
        xn = np.concatenate([xn, np.zeros(len(miss), np.float32)])
        t = np.concatenate([t, np.full(len(miss), W, np.float32)])
        flag = np.concatenate([flag, np.zeros(len(miss), np.float32)])
        pool = np.concatenate([pool, np.ones(len(miss), bool)])
    return {"var": v, "val": xn.astype(np.float32), "flag": flag, "time": t, "pool": pool}


def encode_grid(p, t_end, W=24.0):
    """Hourly grid with LOCF (vitals/labs), zero-fill (interventions) and a mask channel (Bi-LSTM baseline)."""
    n = int(W); X = np.zeros((n, 2 * N_VARS), np.float32)
    ends = t_end - W + np.arange(n) + 1.0
    bv = _byvar(p)
    for j in range(N_VARS):
        tj, xj = bv[j]
        if len(tj) == 0:
            continue
        ks = np.searchsorted(tj, ends, "right")
        valid = ks > 0
        last_t = np.where(valid, tj[np.maximum(ks - 1, 0)], -1e9)
        in_bin = valid & (last_t > ends - 1.0)
        vals = norm(j, xj[np.maximum(ks - 1, 0)])
        if j in INTERV:
            X[:, j] = np.where(in_bin, vals, 0.0)
        else:
            X[:, j] = np.where(valid, vals, 0.0)
        X[:, N_VARS + j] = in_bin.astype(np.float32)
    return X


_WINS = [(1.0, 0.0), (4.0, 1.0), (24.0, 4.0)]   # non-overlapping: (t-1,t], (t-4,t-1], (t-24,t-4]


def tabular_feature_names():
    names = []
    for n in VAR_NAMES:
        for w, _ in _WINS:
            names += [f"{n}_w{int(w)}_{s}" for s in ("mean", "std", "min", "max", "last", "since", "missing")]
    return names


def tabular_features(p, t_end):
    bv = _byvar(p); f = []
    for j in range(N_VARS):
        tj, xj = bv[j]
        k = np.searchsorted(tj, t_end, "right")
        since = float(t_end - tj[k - 1]) if k > 0 else 48.0
        last = float(xj[k - 1]) if k > 0 else np.nan
        for w, off in _WINS:
            a, b = t_end - w, t_end - off
            lo = np.searchsorted(tj, a, "right"); hi = np.searchsorted(tj, b, "right")
            seg = xj[lo:hi]
            if len(seg):
                f += [seg.mean(), seg.std(), seg.min(), seg.max(), last, since, 0.0]
            else:
                f += [np.nan, np.nan, np.nan, np.nan, last, since, 1.0]
    return np.array(f, np.float32)


# ----------------------------------------------------------------------------
# SOFA proxy (platelets, bilirubin, creatinine, MAP, vasopressor) - Table 3.2
# ----------------------------------------------------------------------------
def sofa_proxy(p, t):
    bv = _byvar(p)

    def last(name):
        tj, xj = bv[VAR_IDX[name]]
        k = np.searchsorted(tj, t, "right")
        return float(xj[k - 1]) if k > 0 else np.nan
    s = 0
    plt, bil, cr, mp, vp = last("Platelets"), last("Bilirubin"), last("Creatinine"), last("MAP"), last("Vasopressor")
    if not np.isnan(plt): s += 0 if plt >= 150 else 1 if plt >= 100 else 2 if plt >= 50 else 3 if plt >= 20 else 4
    if not np.isnan(bil): s += 0 if bil < 1.2 else 1 if bil < 2 else 2 if bil < 6 else 3 if bil < 12 else 4
    if not np.isnan(cr): s += 0 if cr < 1.2 else 1 if cr < 2 else 2 if cr < 3.5 else 3 if cr < 5 else 4
    if not np.isnan(mp): s += 1 if mp < 70 else 0
    if not np.isnan(vp) and vp > 0: s += 3
    return s


def strip_cache(p):
    return {k: v for k, v in p.items() if k != "_bv"}
