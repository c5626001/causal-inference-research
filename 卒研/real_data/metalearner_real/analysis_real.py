"""実データ版の結果分析(卒論7.2節のRMSE/MAEに相当する「正解」が実データには無いため、
推定された処置効果(CATE)の分布を見る/ブートストラップでATEの信頼区間を見る、という
分析方法になっている)。
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_cate_boxplot(tau_by_learner: dict, title: str = "CATE Boxplot by Meta-learner"):
    """学習器ごとに推定されたtauの箱ひげ図を描く

    tau_by_learner は {"S-learner (XGB)": tau配列, ...} のような辞書。
    """
    labels = list(tau_by_learner.keys())
    data = [np.asarray(tau)[~np.isnan(tau)] for tau in tau_by_learner.values()]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.axhline(0, linewidth=1)
    plt.ylabel("Estimated CATE")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()


def plot_cate_violin(tau_by_learner: dict, title: str = "CATE Violin Plot by Meta-learner"):
    """学習器ごとに推定されたtauのバイオリンプロットを描く"""
    labels = list(tau_by_learner.keys())
    data = [np.asarray(tau)[~np.isnan(tau)] for tau in tau_by_learner.values()]

    plt.violinplot(dataset=data, showmeans=False, showmedians=True, showextrema=True)
    plt.axhline(0, linewidth=1)
    plt.ylabel("Estimated CATE")
    plt.title(title)
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels, rotation=25, ha="right")
    plt.tight_layout()
    plt.show()


def bootstrap_ate(tau, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """tau(個体ごとの推定処置効果)をブートストラップして、平均処置効果(ATE)と
    その95%信頼区間を求める。

    戻り値: (平均, 信頼区間下限, 信頼区間上限, 0が区間に含まれないか(=有意かどうか))
    """
    tau = np.asarray(tau, dtype=float)
    tau = tau[~np.isnan(tau)]
    n = len(tau)
    if n == 0:
        return np.nan, np.nan, np.nan, False

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = tau[idx].mean(axis=1)

    mean = tau.mean()
    low, high = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    significant = not (low <= 0 <= high)
    return mean, low, high, significant


def print_ate_summary(tau_by_learner: dict, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """学習器ごとにATE・信頼区間・有意性をまとめて表示する"""
    for name, tau in tau_by_learner.items():
        mean, low, high, significant = bootstrap_ate(tau, n_boot=n_boot, alpha=alpha, seed=seed)
        print(f"{name}")
        print(f"  平均(ATE) : {mean:.6f}")
        print(f"  {int((1 - alpha) * 100)}%CI     : [{low:.6f}, {high:.6f}]")
        print(f"  有意({int(alpha * 100)}%)  : {significant}")
        print("-" * 40)
