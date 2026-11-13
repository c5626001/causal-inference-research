"""比較対象のベースライン: S-learner(XGBoost)。

xlearner_baseline.pyと同じ考え方で、Wも特徴量に混ぜた1本のモデルを学習する
(卒研/synthetic/metalearner/learners.pyのS-learnerを参照)。CEVAE-iVAE(共有encoder)の
設計がS-learner型であることと対応させて比較するためのベースライン。
"""
from sklearn.linear_model import LogisticRegression

from xlearner_baseline import search_xgb


def run_slearner(df_train, df_test, proxy_cols, n_iter=50, cv=5, seed=0):
    """S-learnerを学習し、テストデータに対するtau(処置効果)の予測を返す

    戻り値: (tau_pred, models)  models = (model,)
    """
    Xw_train = df_train[proxy_cols].copy()
    Xw_train["T"] = df_train["T"].values

    model = search_xgb(Xw_train, df_train["Y"], n_iter=n_iter, cv=cv, seed=seed)

    tau_pred = predict_slearner((model,), df_test, proxy_cols)
    return tau_pred, (model,)


def refit_slearner_with_fixed_params(models, df_train, proxy_cols):
    """run_slearnerで探索したハイパーパラメータはそのまま使い、新しいdata_seedのデータで学習し直す"""
    model_fixed, = models

    Xw_train = df_train[proxy_cols].copy()
    Xw_train["T"] = df_train["T"].values

    model = model_fixed.__class__(**model_fixed.get_params()).fit(Xw_train, df_train["Y"])
    return (model,)


def predict_slearner(models, df_test, proxy_cols):
    model, = models
    Xw_test_0 = df_test[proxy_cols].copy()
    Xw_test_0["T"] = 0
    Xw_test_1 = df_test[proxy_cols].copy()
    Xw_test_1["T"] = 1

    mu0 = model.predict(Xw_test_0)
    mu1 = model.predict(Xw_test_1)
    return mu1 - mu0
