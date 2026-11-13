"""人工データ生成(DGP)。CEVAE/experiment/dgp.py の1次元版(generate_toy_1d)に、
補助変数u(セグメント)を追加したもの。

u(セグメント番号) → Z(uごとに分散が変わる、真の潜在交絡因子) → X(proxy)
                                                    ↓
                                                    T(処置)
                                                    ↓
                                            Z, T → Y(結果)

uはZより先に(Zとは無関係に)決まっている外生変数で、iVAE/experiment/dgp.py の
セグメント構造と同じ役割。X・T・Yの生成式自体はCEVAE/experiment/dgp.pyのgenerate_toy_1dと同じ。
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_toy_1d_u(N=20000, d=8, n_segments=20, gamma=1.5, theta=2.0, alpha=1.0, beta=1.0,
                       delta_nl=0.8, sigma_x=3.5, sigma_y=1.0, link_scale=1.5, seed=0):
    """潜在交絡因子Zが1次元、かつuごとにZの分散が変わるDGP。

    u ~ Uniform({0, ..., n_segments-1})
    Z | u ~ N(0, seg_var[u])   (segment_varはUniform(0.5, 3.0)からセグメントごとに生成)
    X = tanh(link_scale*Z)*a + noise  (CEVAE/dgp.pyのgenerate_toy_1dと同じ非線形proxy)
    T | Z ~ Bernoulli(sigmoid(gamma*Z))
    tau(Z) = alpha + beta*Z + delta_nl*(Z^2 - 1)
    Y0 = theta*Z + noise,  Y1 = Y0 + tau(Z)
    """
    rng = np.random.default_rng(seed)

    seg_var = rng.uniform(0.5, 3.0, size=n_segments)
    u = rng.integers(0, n_segments, size=N)
    Z = rng.normal(0, np.sqrt(seg_var[u]))

    g_z = np.tanh(link_scale * Z)
    a = rng.normal(1.0, 1.0, size=d)
    X = g_z[:, None] * a[None, :] + rng.normal(0, sigma_x, size=(N, d))
    T = rng.binomial(1, sigmoid(gamma * Z))
    tau = alpha + beta * Z + delta_nl * (Z ** 2 - 1)
    base = theta * Z
    Y0 = base + rng.normal(0, sigma_y, size=N)
    Y1 = base + tau + rng.normal(0, sigma_y, size=N)
    Y = np.where(T == 1, Y1, Y0)
    return X, T, Y, Y0, Y1, tau, Z, u


def one_hot(u, n_segments):
    out = np.zeros((len(u), n_segments))
    out[np.arange(len(u)), u] = 1.0
    return out
