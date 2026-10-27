"""CEVAEの「崩壊」(posterior collapse: zが交絡因子ではなく処置Tの予測器になってしまう現象)を
検出するための診断指標。

v4(1次元Z)で見つけた指標と、v5(2次元Z)で見つけた指標の両方を載せている。
詳しい発見の経緯は `CEVAE/CEVAE_まとめ.md` を参照。

真のZ・真のtauを使う指標(z_corr_general, z_recovery_score, sqrt_pehe)は
「答え合わせ専用」。本番の実データでは使えない(正解を知らないため)。
それ以外(compute_y_mse_diag, z_info_signals, compute_y_mse)は観測データだけで
計算できる「選択信号」で、本番でも使える候補。
"""
import numpy as np
import torch
from sklearn.linear_model import LinearRegression


def _get_mu_z(nets, x, t, y):
    with torch.no_grad():
        mu_z, _ = nets["encoder_z"](x, t, y)
    return mu_z.numpy()


# ---------------------------------------------------------------- 答え合わせ専用(真のZ・tauを使う)
def z_corr_general(nets, x, t, y, Z_true):
    """z復元度(v4で使用): 真のZの各次元に最も相関するmu_zの次元を選んで、その絶対相関の平均を取る。

    Zが1次元のときはシンプルな相関係数とほぼ同じ。Zが多次元だとmu_zとZの次元の対応が
    入れ替わっていても評価できるように、各真の次元ごとに最も合うmu_z次元を探す。
    """
    mz = _get_mu_z(nets, x, t, y)
    if Z_true.ndim == 1:
        Z_true = Z_true[:, None]
    z_dim = mz.shape[1]
    scores = [
        max(abs(np.corrcoef(mz[:, k], Z_true[:, j])[0, 1]) for k in range(z_dim))
        for j in range(Z_true.shape[1])
    ]
    return float(np.mean(scores))


def z_recovery_score(nets, x, t, y, Z_true):
    """z復元度(v5で使用): 真のZの各列をmu_z(全次元)から線形回帰し、決定係数R^2の平均を返す。

    z_corr_general は「1対1に対応するmu_z次元」を探すが、Zが2次元以上だと
    mu_zの中でZの情報が複数次元に混ざって(回転して)入っていてもおかしくない。
    z_recovery_scoreは回転に対して不変なので、2次元以上のZではこちらの方が適切。
    """
    mz = _get_mu_z(nets, x, t, y)
    Z_true = Z_true if Z_true.ndim > 1 else Z_true[:, None]
    scores = []
    for j in range(Z_true.shape[1]):
        reg = LinearRegression().fit(mz, Z_true[:, j])
        scores.append(max(reg.score(mz, Z_true[:, j]), 0.0))
    return float(np.mean(scores))


# ---------------------------------------------------------------- 観測データだけで計算できる信号
def compute_y_mse_diag(nets, x_obs, t_obs, y_obs_np):
    """mse_diff(v4で発見): 真のtau・Zを一切使わない崩壊検出信号。

    mse_q: 補助NW q(y|x,t)(zを介さない)の予測誤差。
    mse_p: テスト時フロー(zを推定してからdecoder_yで予測)の予測誤差。
    mse_diff = mse_p - mse_q。崩壊run(zが無情報)ではmse_pが悪化し、mse_diffが正に大きくなる。
    """
    mse_q, mse_p = _compute_mse_q_p(nets, x_obs, t_obs, y_obs_np)
    return mse_q, mse_p, mse_p - mse_q


def compute_y_mse(nets, x_obs, t_obs, y_obs_np):
    """mse_ratio(v5で発見): mse_diffを比率にしたもの。mse_ratio = mse_p / mse_q。

    1に近いほど健全、大きいほど崩壊疑い。差より比率の方が、DGPのスケールが変わっても
    閾値の意味が安定するため、v5ではこちらを採用した。
    """
    mse_q, mse_p = _compute_mse_q_p(nets, x_obs, t_obs, y_obs_np)
    mse_ratio = mse_p / mse_q if mse_q > 0 else float("inf")
    return mse_q, mse_p, mse_ratio


def _compute_mse_q_p(nets, x_obs, t_obs, y_obs_np):
    with torch.no_grad():
        y0_q, y1_q = nets["encoder_y"](x_obs)
        y_q_hat = torch.where(t_obs == 1, y1_q, y0_q)

        t_hat = (torch.sigmoid(nets["encoder_t"](x_obs)) > 0.5).float()
        y0_for_z, y1_for_z = nets["encoder_y"](x_obs)
        y_hat_for_z = torch.where(t_hat == 1, y1_for_z, y0_for_z)
        z_mu, _ = nets["encoder_z"](x_obs, t_hat, y_hat_for_z)
        y0_p, y1_p = nets["decoder_y"](z_mu)
        y_p_hat = torch.where(t_obs == 1, y1_p, y0_p)

    mse_q = float(np.mean((y_obs_np - y_q_hat.numpy()) ** 2))
    mse_p = float(np.mean((y_obs_np - y_p_hat.numpy()) ** 2))
    return mse_q, mse_p


def z_info_signals(nets, x_obs, t_obs, y_obs):
    """z情報量診断(v4で発見): 観測されたt, y(学習時に実際に使える値)だけを使う。

    z_var: mu_zの分散。崩壊するとzがほぼ一定値になり分散が小さくなる。
    z_t_corr: |corr(mu_z, T)|。崩壊runはzがTの予測器になるので、この値が上がる。
    z_y_corr: |corr(mu_z, Y)|。健全runはzがY(交絡)の情報を持つので、この値が高いはず。
    """
    with torch.no_grad():
        mu_z, _ = nets["encoder_z"](x_obs, t_obs, y_obs)
    mz = mu_z.numpy().reshape(-1)
    z_var = float(np.var(mz))
    z_t_corr = float(abs(np.corrcoef(mz, t_obs.numpy())[0, 1]))
    z_y_corr = float(abs(np.corrcoef(mz, y_obs.numpy())[0, 1]))
    return z_var, z_t_corr, z_y_corr


def guard_threshold(mse_ratio_series, k=3.0, min_abs=1.5):
    """mse_ratioの分布から「崩壊」を弾く閾値を決める(v5で使用)。

    相対ルール(中央値からの頑健な外れ値検出、MAD基準)を主に、目安として絶対値1.5を下限に使う。
    """
    med = mse_ratio_series.median()
    mad = (mse_ratio_series - med).abs().median()
    robust = med + k * 1.4826 * mad
    return max(robust, min_abs)


# ---------------------------------------------------------------- 信号の良し悪しを評価する(共通)
def confusion_at_threshold(df, signal_col, threshold, direction="above", true_col="collapsed"):
    """ある信号・ある閾値で「崩壊」を予測したときの混同行列を返す。

    direction='above' なら signal >= threshold を崩壊予測、'below' なら signal <= threshold。
    """
    true_collapsed = df[true_col]
    pred_collapsed = df[signal_col] >= threshold if direction == "above" else df[signal_col] <= threshold
    tp = int((pred_collapsed & true_collapsed).sum())
    fn = int((~pred_collapsed & true_collapsed).sum())
    fp = int((pred_collapsed & ~true_collapsed).sum())
    tn = int((~pred_collapsed & ~true_collapsed).sum())
    return {
        "signal": signal_col, "direction": direction, "threshold": threshold,
        "崩壊を正しく検出(TP)": tp, "崩壊を見逃す(FN)": fn,
        "成功を誤って除外(FP)": fp, "成功を正しく残す(TN)": tn, "errors": fn + fp,
    }


def best_threshold(df, signal_col, true_col="collapsed"):
    """真のラベルに対して最も分離が良い閾値・方向を全探索で選ぶ(信号自体の分離能力を評価する用)"""
    values = np.sort(df[signal_col].unique())
    candidates = (values[:-1] + values[1:]) / 2 if len(values) > 1 else values
    best = None
    for direction in ["above", "below"]:
        for th in candidates:
            res = confusion_at_threshold(df, signal_col, th, direction, true_col)
            if best is None or res["errors"] < best["errors"]:
                best = res
    return best
