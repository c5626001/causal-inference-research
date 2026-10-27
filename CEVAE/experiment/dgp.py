"""人工データ生成(DGP: Data Generating Process)。

1次元版(v4で使った `generate_toy_v2` 相当)と2次元版(v5で使った `generate_toy_v5` 相当)の
2つを用意している。潜在交絡因子Zの次元数が違うだけで、どちらも
「Zが共変量Xを生み、Zが処置Tと結果Yの両方に影響する」という同じ構造。
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_toy_1d(N=20000, d=8, gamma=1.5, theta=2.0, alpha=1.0, beta=1.0, delta_nl=0.8,
                     sigma_x=3.5, sigma_y=1.0, link_scale=1.5, seed=0):
    """潜在交絡因子Zが1次元のDGP(v4で使用)。

    Z ~ N(0,1)
    X = tanh(link_scale*Z)*a + noise  (非線形リンクでproxyを難しくしている)
    T | Z ~ Bernoulli(sigmoid(gamma*Z))
    tau(Z) = alpha + beta*Z + delta_nl*(Z^2 - 1)  (異質性に非線形項あり)
    Y0 = theta*Z + noise,  Y1 = Y0 + tau(Z)
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, size=N)
    g_z = np.tanh(link_scale * Z)
    a = rng.normal(1.0, 1.0, size=d)
    X = g_z[:, None] * a[None, :] + rng.normal(0, sigma_x, size=(N, d))
    T = rng.binomial(1, sigmoid(gamma * Z))
    tau = alpha + beta * Z + delta_nl * (Z ** 2 - 1)
    base = theta * Z
    Y0 = base + rng.normal(0, sigma_y, size=N)
    Y1 = base + tau + rng.normal(0, sigma_y, size=N)
    Y = np.where(T == 1, Y1, Y0)
    return X, T, Y, Y0, Y1, tau, Z


def generate_toy_2d(N=20000, d=8, sigma_x=3.5, sigma_y=1.0, seed=0):
    """潜在交絡因子Zが2次元のDGP(v5で使用。v3のDGPを流用)。

    Z = [Z1, Z2] ~ N(0, I2)
    X = tanh(Z @ A) + noise,  A ∈ R^{2×d} ~ N(1,1)
    T | Z ~ Bernoulli(sigmoid(1.5*Z1 + 0.5*Z2))
    tau(Z) = 1.0 + Z1 + 0.5*Z2 + 0.8*(Z1^2 + Z2^2 - 2)
    Y0 = 2.0*Z1 + 1.0*Z2 + noise,  Y1 = Y0 + tau(Z)
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, size=(N, 2))
    A = rng.normal(1.0, 1.0, size=(2, d))
    X = np.tanh(Z @ A) + rng.normal(0, sigma_x, size=(N, d))
    w_T = np.array([1.5, 0.5])
    T = rng.binomial(1, sigmoid(Z @ w_T))
    w_tau = np.array([1.0, 0.5])
    tau = 1.0 + Z @ w_tau + 0.8 * (np.sum(Z ** 2, axis=1) - 2)
    w_Y = np.array([2.0, 1.0])
    base = Z @ w_Y
    Y0 = base + rng.normal(0, sigma_y, size=N)
    Y1 = base + tau + rng.normal(0, sigma_y, size=N)
    Y = np.where(T == 1, Y1, Y0)
    return X, T, Y, Y0, Y1, tau, Z
