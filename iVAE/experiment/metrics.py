"""評価指標: MCC(Mean Correlation Coefficient)。

推定した潜在変数 z と真の潜在源 s は、次元の並び順や符号が入れ替わっていても
「情報として復元できていれば」よい(置換・符号を除いた識別性)。そこで各次元ペアの
相関を全部計算し、最適な対応付け(線形assignment)をしたうえで平均相関を取る。
iVAE論文の標準評価指標。1に近いほど真の潜在源をよく復元できている。
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def mean_correlation_coefficient(z_est, s_true):
    d = z_est.shape[1]
    corr = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            corr[i, j] = np.corrcoef(z_est[:, i], s_true[:, j])[0, 1]
    row_idx, col_idx = linear_sum_assignment(-np.abs(corr))
    return np.abs(corr[row_idx, col_idx]).mean(), (row_idx, col_idx, corr)
