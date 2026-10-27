"""SHAP値によるITE(個体処置効果)への寄与度分析 — 実データ版。

元の8章ノートブックでは X-learner(XGBoost)のtau0モデル・tau1モデルそれぞれの
SHAP値を、傾向スコアで重み付けして合成していた。

卒論コードとの違い(修正点): 元のコードは
    shap_values_ITE_x = (1-e)*shap_values_x1 - e*shap_values_x0   (引き算)
となっていたが、処置効果 tau 自体は
    tau = tau0 * P(W=0|x) + tau1 * P(W=1|x)   (足し算)
で計算している。SHAP値は出力に対して線形に分解できるので、tauの計算式と
揃えるなら足し算が正しいはず。ここでは
    shap_values_ITE = P(W=0|x)*shap_values_tau0 + P(W=1|x)*shap_values_tau1
と、足し算に修正している。
"""
import shap


def compute_ite_shap(tau0_model, tau1_model, prop_model, test_df, treatment_col: str, outcome_col: str):
    """X-learnerのtau0・tau1モデルからITEのSHAP値を計算する

    戻り値: (shap_values_ite, X_test) — shap.summary_plotにそのまま渡せる
    """
    X_test = test_df.drop(columns=[treatment_col, outcome_col])

    explainer_tau0 = shap.TreeExplainer(tau0_model)
    explainer_tau1 = shap.TreeExplainer(tau1_model)
    shap_values_tau0 = explainer_tau0.shap_values(X_test)
    shap_values_tau1 = explainer_tau1.shap_values(X_test)

    prob_untreated = prop_model.predict_proba(X_test)[:, 0].reshape(-1, 1)  # P(W=0|x)
    shap_values_ite = prob_untreated * shap_values_tau0 + (1 - prob_untreated) * shap_values_tau1

    return shap_values_ite, X_test


def plot_ite_shap(shap_values_ite, X_test):
    shap.summary_plot(shap_values_ite, X_test)
