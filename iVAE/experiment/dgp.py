"""人工データ生成(DGP)。iVAE論文(Khemakhem+2020)のToy実験と同じ構成。

真の潜在源 s は補助変数u(セグメント番号)ごとに分散が変わる(=非定常性)。
これをnonlinear-ICAの識別性を買うための補助変数として使う。
s を非線形・可逆なMLPで混ぜて観測データ x を作る(x_dim = s_dim)。
"""
import numpy as np


def leaky_relu(x, slope=0.2):
    return np.where(x > 0, x, slope * x)


def make_mixing_mlp(d, n_layers=2, seed=0):
    """d次元→d次元の非線形・可逆な混合。各層の重みはランダム直交行列(QR分解で作る)。
    直交行列は常によく条件付けられている(条件数=1)ので、層を重ねても数値的に
    退化しない。活性化はleaky_relu(slope!=0なら全単射なので、各層が可逆=全体も可逆)。"""
    rng = np.random.default_rng(seed)
    weights = []
    for _ in range(n_layers):
        a = rng.normal(0, 1, size=(d, d))
        q, _ = np.linalg.qr(a)
        weights.append(q)
    return weights


def mix(s, weights, slope=0.2):
    h = s
    for i, w in enumerate(weights):
        h = h @ w.T
        if i < len(weights) - 1:
            h = leaky_relu(h, slope)
    return h


def generate_data(n_segments=20, n_per_segment=1000, d=2, n_layers=2, sigma_x=0.1, seed=0):
    """
    u = セグメント番号(0 ... n_segments-1)
    s | u ~ N(0, diag(seg_vars[u]))    (セグメントごとに各次元の分散が変わる)
    x = mix(s) + noise

    Returns: x (N,d), s (N,d) 真の潜在源, u (N,) セグメント番号, seg_vars (n_segments,d)
    """
    rng = np.random.default_rng(seed)
    n_total = n_segments * n_per_segment
    seg_vars = rng.uniform(0.5, 3.0, size=(n_segments, d))

    u = np.repeat(np.arange(n_segments), n_per_segment)
    s = np.zeros((n_total, d))
    for seg in range(n_segments):
        idx = (u == seg)
        s[idx] = rng.normal(0, np.sqrt(seg_vars[seg]), size=(n_per_segment, d))

    weights = make_mixing_mlp(d, n_layers=n_layers, seed=seed + 1)
    x = mix(s, weights) + rng.normal(0, sigma_x, size=(n_total, d))
    return x, s, u, seg_vars


def one_hot(u, n_segments):
    out = np.zeros((len(u), n_segments))
    out[np.arange(len(u)), u] = 1.0
    return out
