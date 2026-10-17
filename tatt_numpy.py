"""Pure-NumPy inference for the Time-Aware Temporal Transformer (mirrors models.TATT in eval mode)."""
import json
import numpy as np
from scipy.special import erf, expit


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def layer_norm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def softplus(x):
    return np.log1p(np.exp(x))


def tape(t, d):
    """Time-aware positional encoding (Eq 3.2-3.3). t: (B, L) hours."""
    k = np.arange(0, d, 2, dtype=np.float64)
    ang = t[..., None].astype(np.float64) / np.power(10000.0, k / d)
    pe = np.zeros(t.shape + (d,), np.float64); pe[..., 0::2] = np.sin(ang); pe[..., 1::2] = np.cos(ang)
    return pe


def collate(windows):
    """Pad a list of encode_window() dicts into batch arrays."""
    B = len(windows); L = max(len(w["var"]) for w in windows)
    var = np.zeros((B, L), np.int64); val = np.zeros((B, L)); flag = np.zeros((B, L)); tm = np.zeros((B, L))
    pad = np.zeros((B, L), bool); pool = np.zeros((B, L), bool)
    for i, w in enumerate(windows):
        n = len(w["var"]); var[i, :n] = w["var"]; val[i, :n] = w["val"]; flag[i, :n] = w["flag"]; tm[i, :n] = w["time"]; pad[i, :n] = True; pool[i, :n] = w["pool"]
    return {"var": var, "val": val, "flag": flag, "time": tm, "pad": pad, "pool": pool}


class TATTNumpy:
    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=False)
        self.cfg = json.loads(str(z["__cfg_json"])); self.p = {k: z[k].astype(np.float64) for k in z.files if k != "__cfg_json"}
        self.d = self.cfg["d_model"]; self.h = self.cfg["n_heads"]; self.dk = self.d // self.h
        self.n_layers = self.cfg["n_layers"]; self.window = float(self.cfg["window"]); self.max_len = int(self.cfg["max_len"])

    def alphas(self):
        return [float(softplus(self.p[f"blocks.{i}.attn.raw_alpha"])) for i in range(self.n_layers)]

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def _embed(self, b):
        P = self.p; x = np.concatenate([P["var_emb.weight"][b["var"]], b["val"][..., None], b["flag"][..., None]], -1)
        return x @ P["embed.weight"].T + P["embed.bias"] + tape(b["time"], self.d)

    def _attn(self, i, x, b, alpha=None):
        P = self.p; pre = f"blocks.{i}."; B, L, d = x.shape
        qkv = (x @ P[pre + "attn.qkv.weight"].T + P[pre + "attn.qkv.bias"]).reshape(B, L, 3, self.h, self.dk).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        a = float(softplus(P[pre + "attn.raw_alpha"])) if alpha is None else float(alpha)
        dT = np.abs(b["time"][:, :, None] - b["time"][:, None, :])
        logits = logits - a * dT[:, None]
        logits = np.where(b["pad"][:, None, None, :], logits, -np.inf)
        logits = logits - logits.max(-1, keepdims=True); att = np.exp(logits); att /= att.sum(-1, keepdims=True)
        out = (att @ v).transpose(0, 2, 1, 3).reshape(B, L, d) @ P[pre + "attn.out.weight"].T + P[pre + "attn.out.bias"]
        return out, att

    def _block(self, i, x, b, alpha=None):
        P = self.p; pre = f"blocks.{i}."
        a, att = self._attn(i, x, b, alpha)
        x = layer_norm(x + a, P[pre + "n1.weight"], P[pre + "n1.bias"])
        f = gelu(x @ P[pre + "ff.0.weight"].T + P[pre + "ff.0.bias"]) @ P[pre + "ff.3.weight"].T + P[pre + "ff.3.bias"]
        return layer_norm(x + f, P[pre + "n2.weight"], P[pre + "n2.bias"]), att

    def predict(self, windows, return_attn=False):
        """Probabilities for a list of windows. With return_attn: also per-variable attention shares and token importance."""
        b = collate(windows); x = self._embed(b); att = None
        for i in range(self.n_layers): x, att = self._block(i, x, b)
        pm = (b["pool"] & b["pad"]).astype(float); pm = np.where(pm.sum(1, keepdims=True) > 0, pm, b["pad"].astype(float))
        z = (x * pm[..., None]).sum(1) / pm.sum(1, keepdims=True)
        P = self.p; h = gelu(z @ P["head.net.0.weight"].T + P["head.net.0.bias"]); logit = (h @ P["head.net.3.weight"].T + P["head.net.3.bias"])[:, 0]
        prob = expit(logit)
        if not return_attn: return prob
        a = att.mean(1); imp = (a * pm[:, :, None]).sum(1) / pm.sum(1, keepdims=True)
        m = (b["pool"] & b["pad"]) * imp; va = np.zeros((len(windows), 15))
        for i in range(len(windows)): np.add.at(va[i], b["var"][i], m[i])
        va = va / np.clip(va.sum(1, keepdims=True), 1e-9, None)
        return prob, va, imp, b

    def first_layer_attention(self, window, alpha=None):
        b = collate([window]); x = self._embed(b); _, att = self._block(0, x, b, alpha)
        return att.mean(1)[0]
