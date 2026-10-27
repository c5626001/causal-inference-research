"""比較対象のベースライン: X-learner(XGBoost)。

CEVAEが「潜在交絡因子を考慮しない普通の因果推論手法」にどれだけ勝てるかを見るための
比較対象。卒研で使ったX-learnerと同じ考え方(卒研/synthetic/metalearner/learners.pyのX-learnerを参照)。
"""
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

XGB_PARAM_DIST = {
    "n_estimators": randint(300, 2500),
    "learning_rate": uniform(0.01, 0.19),
    "max_depth": randint(2, 9),
    "min_child_weight": randint(1, 10),
    "subsample": uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.7, 0.3),
    "reg_lambda": uniform(0.0, 10.0),
}


def search_xgb(X_in, y_in, n_iter=50, cv=5, seed=0, n_jobs=1):
    """XGBoostのハイパーパラメータをランダムサーチで探して、一番良いモデルを返す

    n_jobs=1がデフォルト: PyTorch(CEVAE)とjoblibの並列化(n_jobs=-1)を同じプロセスで
    併用すると、macOSで子プロセスがデッドロックすることがあるため。もしCEVAEのコードを
    一切使わないスクリプトで速度を上げたい場合は n_jobs=-1 を指定してよい。
    """
    search = RandomizedSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=seed, verbosity=0),
        param_distributions=XGB_PARAM_DIST,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        random_state=seed,
    )
    search.fit(X_in, y_in)
    return search.best_estimator_


def run_xlearner(df_train, df_test, proxy_cols, n_iter=50, cv=5, seed=0):
    """X-learnerを学習し、テストデータに対するtau(処置効果)の予測を返す

    戻り値: (tau_pred, models)  models = (model_0, model_1, tau0_model, tau1_model, prop_model)
    """
    df0 = df_train[df_train["T"] == 0]
    df1 = df_train[df_train["T"] == 1]

    model_0 = search_xgb(df0[proxy_cols], df0["Y"], n_iter=n_iter, cv=cv, seed=seed)
    model_1 = search_xgb(df1[proxy_cols], df1["Y"], n_iter=n_iter, cv=cv, seed=seed)

    D0 = model_1.predict(df0[proxy_cols]) - df0["Y"].values
    D1 = df1["Y"].values - model_0.predict(df1[proxy_cols])

    tau0_model = search_xgb(df0[proxy_cols], D0, n_iter=n_iter, cv=cv, seed=seed)
    tau1_model = search_xgb(df1[proxy_cols], D1, n_iter=n_iter, cv=cv, seed=seed)

    prop_model = LogisticRegression(max_iter=1000).fit(df_train[proxy_cols], df_train["T"])
    propensity_0 = prop_model.predict_proba(df_test[proxy_cols])[:, 0]  # P(T=0|x)

    tau_pred = propensity_0 * tau0_model.predict(df_test[proxy_cols]) \
        + (1 - propensity_0) * tau1_model.predict(df_test[proxy_cols])

    return tau_pred, (model_0, model_1, tau0_model, tau1_model, prop_model)


def refit_xlearner_with_fixed_params(models, df_train, proxy_cols):
    """run_xlearnerで探索したハイパーパラメータはそのまま使い、新しいdata_seedのデータで学習し直す

    (ハイパラ探索は重いので、data_seed=0で1回だけ探索したパラメータを他のseedでも使い回す)
    """
    model_0_fixed, model_1_fixed, tau0_fixed, tau1_fixed, _ = models

    df0 = df_train[df_train["T"] == 0]
    df1 = df_train[df_train["T"] == 1]

    model_0 = model_0_fixed.__class__(**model_0_fixed.get_params()).fit(df0[proxy_cols], df0["Y"])
    model_1 = model_1_fixed.__class__(**model_1_fixed.get_params()).fit(df1[proxy_cols], df1["Y"])

    D0 = model_1.predict(df0[proxy_cols]) - df0["Y"].values
    D1 = df1["Y"].values - model_0.predict(df1[proxy_cols])

    tau0_model = tau0_fixed.__class__(**tau0_fixed.get_params()).fit(df0[proxy_cols], D0)
    tau1_model = tau1_fixed.__class__(**tau1_fixed.get_params()).fit(df1[proxy_cols], D1)

    prop_model = LogisticRegression(max_iter=1000).fit(df_train[proxy_cols], df_train["T"])

    return model_0, model_1, tau0_model, tau1_model, prop_model


def predict_xlearner(models, df_test, proxy_cols):
    _, _, tau0_model, tau1_model, prop_model = models
    propensity_0 = prop_model.predict_proba(df_test[proxy_cols])[:, 0]
    tau_pred = propensity_0 * tau0_model.predict(df_test[proxy_cols]) \
        + (1 - propensity_0) * tau1_model.predict(df_test[proxy_cols])
    return tau_pred
