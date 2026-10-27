"""評価指標: RMSE, MAE (卒論7.2節)"""
import numpy as np


def RMSE(tau_true, tau_pred) -> float:
    """RMSE = sqrt(E[(tau_hat(x) - tau(x))^2])"""
    return float(np.sqrt(np.mean((np.asarray(tau_true) - np.asarray(tau_pred)) ** 2)))


def MAE(tau_true, tau_pred) -> float:
    """MAE = E[|tau_hat(x) - tau(x)|]"""
    return float(np.mean(np.abs(np.asarray(tau_true) - np.asarray(tau_pred))))
