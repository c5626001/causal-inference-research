"""評価指標: MCC(Mean Correlation Coefficient)。

推定した潜在変数 z と真の潜在源 s は、次元の並び順や符号が入れ替わっていても
「情報として復元できていれば」よい(置換・符号を除いた識別性)。そこで各次元ペアの
相関を全部計算し、最適な対応付け(線形assignment)をしたうえで平均相関を取る。
iVAE論文の標準評価指標。1に近いほど真の潜在源をよく復元できている。
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def mean_correlation_coefficient(z_est, s_true):
    d = z_est.shape[1]
    corr = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            corr[i, j] = np.corrcoef(z_est[:, i], s_true[:, j])[0, 1]
    row_idx, col_idx = linear_sum_assignment(-np.abs(corr))
    return np.abs(corr[row_idx, col_idx]).mean(), (row_idx, col_idx, corr)


def reconstruction_mse(nets, x, u_onehot=None):
    """観測xの再構成誤差。MCCと条件を揃えるため、サンプリングせずエンコーダの事後平均
    mu_qだけをデコーダに通した決定的な再構成 x_hat = Decoder(mu_q(x)) で評価する。
    """
    with torch.no_grad():
        mu_q, _ = nets["encoder"](x, u_onehot)
        x_hat = nets["decoder"](mu_q)
        return ((x - x_hat) ** 2).sum(dim=-1).mean().item()
