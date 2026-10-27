"""人工データ生成: 卒論7章(数値実験)の状況1〜6のDGP(データ生成過程)。

各状況のパラメータ(次元d, 処置割合e(x), 応答関数mu0/mu1)は卒論本文7.1節の数式に準拠。
ただし以下2点は元の卒論から修正している(詳細はREADME参照):

- 状況3・4のd: 本文は「d=15」と書いているが、同じ式内でbeta0, beta1の次元が
  Unif([1,30]^20)と20次元になっており矛盾している。実際に実行されたコード
  (卒研フォルダ内の元ノートブック)もd=20で一貫していたため、d=20をデフォルトにした。
  SITUATIONSの下の行を書き換えれば d=15 にも戻せる。
- 状況5・6のmu1: 元ノートブックには cos(2*x4*x5) とあったが、卒論本文の数式は
  cos(x4*x5) (係数2なし)だったため、本文に合わせて係数2を外した。

SITUATIONS の値を書き換えれば、次元・処置割合・サンプル数は自由に変更できる。
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class SituationConfig:
    d: int
    e_const: float          # e(x) = P(W=1|x) (定数)
    structure: Literal["shared", "separate", "nonlinear"]
    N: int = 100_000        # サンプル数(本文: train/testそれぞれ50,000 = 合計100,000)


# ここを書き換えれば各状況のパラメータを自由に変更できる
SITUATIONS: dict[int, SituationConfig] = {
    1: SituationConfig(d=5, e_const=0.5, structure="shared"),
    2: SituationConfig(d=5, e_const=0.1, structure="shared"),
    3: SituationConfig(d=20, e_const=0.5, structure="separate"),   # 本文表記はd=15(矛盾あり、上部docstring参照)
    4: SituationConfig(d=20, e_const=0.1, structure="separate"),   # 同上
    5: SituationConfig(d=10, e_const=0.5, structure="nonlinear"),
    6: SituationConfig(d=10, e_const=0.1, structure="nonlinear"),
}


def make_ar1_cov(d: int, rho: float = 0.5) -> np.ndarray:
    """AR(1)型の相関行列 Sigma_p (卒論7.1節 式1)"""
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def sample_X(N: int, d: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """共変量 X ~ N(0, Sigma_p) を生成"""
    Sigma = make_ar1_cov(d, rho)
    L = np.linalg.cholesky(Sigma)
    Z = rng.normal(size=(N, d))
    return Z @ L.T


def sigmoid_feature(x: np.ndarray) -> np.ndarray:
    """状況5・6で使うシグモイド型の特徴量変換 varsigma(x) = 2 / (1 + exp(-12(x - 1/2)))"""
    return 2.0 / (1.0 + np.exp(-12.0 * (x - 0.5)))


def simulate_situation(
    situation: int,
    rho: float = 0.5,
    seed: int = 42,
    seed_w: int = 123,
    config: SituationConfig | None = None,
):
    """状況1〜6のいずれかについて (X, tau_true, Y0, Y1, W, Y) を生成する。

    config を渡せば SITUATIONS の値を使わず独自パラメータで生成できる。
    """
    cfg = config or SITUATIONS.get(situation)
    if cfg is None:
        raise ValueError(f"situation must be 1-6 (or pass config explicitly), got {situation}")

    rng = np.random.default_rng(seed)
    X = sample_X(cfg.N, cfg.d, rho, rng)

    if cfg.structure == "shared":
        # 状況1・2: CATE関数が0 (mu1 = mu0)
        beta = rng.uniform(1, 30, size=cfg.d)
        mu0 = X @ beta
        mu1 = mu0.copy()
    elif cfg.structure == "separate":
        # 状況3・4: mu0, mu1が別係数(共通構造が存在しない)
        beta0 = rng.uniform(1, 30, size=cfg.d)
        beta1 = rng.uniform(1, 30, size=cfg.d)
        mu0 = X @ beta0
        mu1 = X @ beta1
    elif cfg.structure == "nonlinear":
        # 状況5・6: 複雑な非線形構造
        s1 = sigmoid_feature(X[:, 0])
        s2 = sigmoid_feature(X[:, 1])
        mu0 = -0.5 * s1 * s2
        mu1 = 5 * s1 * s2 + 3 * np.sin(X[:, 2]) + 0.5 * np.cos(X[:, 3] * X[:, 4])
    else:
        raise ValueError(f"unknown structure: {cfg.structure}")

    tau_true = mu1 - mu0
    eps0 = rng.normal(0, 1, size=cfg.N)
    eps1 = rng.normal(0, 1, size=cfg.N)
    Y0 = mu0 + eps0
    Y1 = mu1 + eps1

    rng_w = np.random.default_rng(seed_w)
    W = rng_w.binomial(1, cfg.e_const, size=cfg.N)
    Y = np.where(W == 1, Y1, Y0)

    return X, tau_true, Y0, Y1, W, Y


def make_df(X, W, Y, tau_true, Y0, Y1):
    """学習・評価用にDataFrameへまとめる"""
    import pandas as pd

    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
    df["W"] = W
    df["Y"] = Y
    df["Y0"] = Y0
    df["Y1"] = Y1
    df["tau"] = tau_true

    col_order = ["tau", "W", "Y", "Y0", "Y1"] + [f"X{i+1}" for i in range(X.shape[1])]
    return df[col_order]
