"""S-learner / T-learner / X-learner: 実データ(クーポンキャンペーン)版。

人工データ版(`卒研/synthetic/metalearner/learners.py`)とほぼ同じ流れだが、
W=0群とW=1群でデータ数が大きく違う(W=1=クーポン配布群の方が少ない)ため、
ハイパーパラメータの探索範囲をW=0用・W=1用で分けている
(W=1側は過学習を抑えるため正則化を強めにしている)。
"""
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from metalearner_real.data_real import OUTCOME_COL, TREATMENT_COL

# ---- S-learner用(W=0, W=1まとめて学習するので、ある程度自由な探索範囲) ----
XGB_PARAM_DISTRIBUTIONS_S = {
    "n_estimators": randint(600, 5000),
    "learning_rate": loguniform(0.01, 0.2),
    "max_depth": randint(2, 9),
    "min_child_weight": randint(1, 20),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma": loguniform(1e-8, 5.0),
    "reg_alpha": loguniform(1e-8, 10.0),
    "reg_lambda": loguniform(1e-3, 100.0),
}

RF_PARAM_DISTRIBUTIONS_S = {
    "max_depth": [None] + list(range(3, 25)),
    "min_samples_split": randint(2, 40),
    "min_samples_leaf": randint(1, 30),
    "max_features": ["sqrt", "log2", uniform(0.3, 0.7)],
}

# ---- T/X-learner用、W=0群(データが多いので少し自由) ----
XGB_PARAM_DISTRIBUTIONS_W0 = {
    "n_estimators": randint(500, 3500),
    "learning_rate": loguniform(0.01, 0.15),
    "max_depth": randint(2, 8),
    "min_child_weight": randint(3, 40),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma": loguniform(1e-8, 5.0),
    "reg_alpha": loguniform(1e-6, 10.0),
    "reg_lambda": loguniform(1e-2, 200.0),
}

RF_PARAM_DISTRIBUTIONS_W0 = {
    "max_depth": [None] + list(range(3, 20)),
    "min_samples_split": randint(2, 30),
    "min_samples_leaf": randint(3, 40),
    "max_features": ["sqrt", "log2", uniform(0.3, 0.6)],
}

# ---- T/X-learner用、W=1群(データが少ないので正則化を強める) ----
XGB_PARAM_DISTRIBUTIONS_W1 = {
    "n_estimators": randint(400, 3000),
    "learning_rate": loguniform(0.01, 0.12),
    "max_depth": randint(2, 6),              # 深さ抑えめ
    "min_child_weight": randint(10, 80),     # 強め
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma": loguniform(1e-6, 10.0),
    "reg_alpha": loguniform(1e-5, 20.0),
    "reg_lambda": loguniform(1e-2, 500.0),
}

RF_PARAM_DISTRIBUTIONS_W1 = {
    "max_depth": list(range(2, 12)),         # 深さ抑えめ
    "min_samples_split": randint(2, 40),
    "min_samples_leaf": randint(10, 80),     # 葉を大きめ
    "max_features": ["sqrt", "log2", uniform(0.3, 0.6)],
}


def search_xgb(X, y, param_distributions, n_iter=100, cv=3, random_state=0):
    """XGBoostのハイパーパラメータをランダムサーチで探して、一番良いモデルを返す"""
    model = XGBRegressor(objective="reg:squarederror")
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
    )
    search.fit(X, y)
    return search.best_estimator_


def search_rf(X, y, param_distributions, n_iter=100, cv=3, random_state=0):
    """ランダムフォレストのハイパーパラメータをランダムサーチで探して、一番良いモデルを返す

    探索自体は木の本数を少なめ(400本)にして行い、見つかったパラメータで
    木の本数を1200本に増やして学習し直す。
    """
    base_model = RandomForestRegressor(n_estimators=400, bootstrap=True, random_state=random_state)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
    )
    search.fit(X, y)

    best_model = search.best_estimator_.set_params(n_estimators=1200, n_jobs=-1)
    best_model.fit(X, y)
    return best_model


# ---------------------------------------------------------------- S-learner
def fit_s_learner(train_df, cv, n_iter=100):
    """S-learner: 処置Wも特徴量に含めた1つのモデルを学習する(XGBoostとRFの両方)

    cv には data_real.load_and_split が返す period基準の時系列CV分割を渡す。
    """
    X_train = train_df.drop(columns=OUTCOME_COL)
    y_train = train_df[OUTCOME_COL]

    model_xgb = search_xgb(X_train, y_train, XGB_PARAM_DISTRIBUTIONS_S, n_iter=n_iter, cv=cv)
    model_rf = search_rf(X_train, y_train, RF_PARAM_DISTRIBUTIONS_S, n_iter=n_iter, cv=cv)
    return model_xgb, model_rf


def predict_s_learner(model, test_df):
    """W=0とW=1それぞれで予測し、その差を処置効果(tau)とする"""
    X_test = test_df.drop(columns=OUTCOME_COL)
    X_test_w0 = X_test.copy()
    X_test_w0[TREATMENT_COL] = 0
    X_test_w1 = X_test.copy()
    X_test_w1[TREATMENT_COL] = 1

    mu0 = model.predict(X_test_w0)
    mu1 = model.predict(X_test_w1)
    tau = mu1 - mu0
    return tau, mu0, mu1


# ---------------------------------------------------------------- T-learner
def fit_t_learner(train_df, n_iter=100):
    """T-learner: W=0群・W=1群それぞれに別モデルを学習する"""
    train_0 = train_df[train_df[TREATMENT_COL] == 0]
    train_1 = train_df[train_df[TREATMENT_COL] == 1]

    X_train_0 = train_0.drop(columns=[TREATMENT_COL, OUTCOME_COL])
    y_train_0 = train_0[OUTCOME_COL]
    X_train_1 = train_1.drop(columns=[TREATMENT_COL, OUTCOME_COL])
    y_train_1 = train_1[OUTCOME_COL]

    model_xgb_0 = search_xgb(X_train_0, y_train_0, XGB_PARAM_DISTRIBUTIONS_W0, n_iter=n_iter)
    model_xgb_1 = search_xgb(X_train_1, y_train_1, XGB_PARAM_DISTRIBUTIONS_W1, n_iter=n_iter)
    model_rf_0 = search_rf(X_train_0, y_train_0, RF_PARAM_DISTRIBUTIONS_W0, n_iter=n_iter)
    model_rf_1 = search_rf(X_train_1, y_train_1, RF_PARAM_DISTRIBUTIONS_W1, n_iter=n_iter)

    return model_xgb_0, model_xgb_1, model_rf_0, model_rf_1


def predict_t_learner(model_0, model_1, test_df):
    """対照群用モデル・処置群用モデルそれぞれで予測し、差を処置効果(tau)とする"""
    X_test = test_df.drop(columns=[TREATMENT_COL, OUTCOME_COL])
    mu0 = model_0.predict(X_test)
    mu1 = model_1.predict(X_test)
    tau = mu1 - mu0
    return tau, mu0, mu1


# ---------------------------------------------------------------- X-learner
def fit_x_learner(train_df, t_model_0, t_model_1, search_fn=search_xgb, n_iter=100):
    """X-learner: T-learnerのモデルを使って、個体ごとの処置効果を擬似的に作り、それを再学習する

    t_model_0, t_model_1 には T-learner で学習したモデル(XGBoost同士、または
    RF同士のペア)を渡す。search_fn=search_rf を指定すればRF版になる。
    """
    train_0 = train_df[train_df[TREATMENT_COL] == 0]
    train_1 = train_df[train_df[TREATMENT_COL] == 1]

    X_train_0 = train_0.drop(columns=[TREATMENT_COL, OUTCOME_COL])
    y_train_0 = train_0[OUTCOME_COL]
    X_train_1 = train_1.drop(columns=[TREATMENT_COL, OUTCOME_COL])
    y_train_1 = train_1[OUTCOME_COL]

    # D0: 対照群の人が「クーポンをもらっていたら」どうなっていたかの推定値 - 実際の値
    D0 = t_model_1.predict(X_train_0) - y_train_0
    # D1: 処置群の人の実際の値 - 「クーポンをもらっていなかったら」の推定値
    D1 = y_train_1 - t_model_0.predict(X_train_1)

    param_dist_w0 = XGB_PARAM_DISTRIBUTIONS_W0 if search_fn is search_xgb else RF_PARAM_DISTRIBUTIONS_W0
    param_dist_w1 = XGB_PARAM_DISTRIBUTIONS_W1 if search_fn is search_xgb else RF_PARAM_DISTRIBUTIONS_W1

    tau0_model = search_fn(X_train_0, D0, param_dist_w0, n_iter=n_iter)
    tau1_model = search_fn(X_train_1, D1, param_dist_w1, n_iter=n_iter)

    prop_model = LogisticRegression(max_iter=1000)
    X_train_all = train_df.drop(columns=[TREATMENT_COL, OUTCOME_COL])
    prop_model.fit(X_train_all, train_df[TREATMENT_COL])

    return tau0_model, tau1_model, prop_model


def predict_x_learner(tau0_model, tau1_model, prop_model, test_df):
    """tau0, tau1を傾向スコアで重み付けして最終的なtauを出す

      tau = tau0 * P(W=0|x) + tau1 * P(W=1|x)
    """
    X_test = test_df.drop(columns=[TREATMENT_COL, OUTCOME_COL])
    tau0 = tau0_model.predict(X_test)
    tau1 = tau1_model.predict(X_test)
    propensity = prop_model.predict_proba(X_test)[:, 0]  # P(W=0|x)
    tau = tau0 * propensity + tau1 * (1 - propensity)
    return tau, tau0, tau1, propensity
