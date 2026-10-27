"""SHAP値によるITE(個体処置効果)への寄与度分析。

S-learner(W込みの単一モデル)について、W=1で予測したときのSHAP値とW=0で
予測したときのSHAP値の差分を「ITEへの寄与度」とみなす(卒論7.3節の手法)。
"""
import shap


def compute_ite_shap(s_learner_model, X_test, n_sample: int = 2000, random_state: int = 0):
    """S-learnerモデル1つを対象に、ITEのSHAP値を計算する。

    戻り値: (shap_values_ite, X_sampled_w1)
      X_sampled_w1 は shap.summary_plot にそのまま渡せる(W=1側の特徴量)
    """
    n_sample = min(n_sample, len(X_test))
    idx = X_test.sample(n_sample, random_state=random_state).index

    X0 = X_test.loc[idx].copy()
    X0["W"] = 0
    X1 = X_test.loc[idx].copy()
    X1["W"] = 1

    explainer = shap.TreeExplainer(s_learner_model)
    shap_values_0 = explainer.shap_values(X0)
    shap_values_1 = explainer.shap_values(X1)
    shap_values_ite = shap_values_1 - shap_values_0

    return shap_values_ite, X1


def plot_ite_shap(shap_values_ite, X_sampled_w1, max_display: int = 21):
    """ITEのSHAP summary plotを表示する"""
    shap.summary_plot(shap_values_ite, X_sampled_w1, max_display=max_display)
