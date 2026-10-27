"""S-learner / T-learner / X-learner の学習・予測ロジック。

卒論7章付録のコード(状況ごとに全く同じ処理をコピペしていた)を、関数として
まとめ直したもの。中身の流れは元のコードとほぼ同じで、状況が変わっても
同じ関数を呼べるようにしただけ。ハイパーパラメータ探索範囲は付録(p.45)に準拠。
"""
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

# ここを書き換えればハイパーパラメータ探索範囲を変更できる(卒論付録p.45準拠)
XGB_PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(300, 2500),
    "learning_rate": uniform(0.01, 0.19),
    "max_depth": randint(2, 9),
    "min_child_weight": randint(1, 10),
    "subsample": uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.7, 0.3),
    "reg_lambda": uniform(0.0, 10.0),
}

RF_PARAM_DISTRIBUTIONS = {
    "max_depth": randint(4, 14),
    "max_features": uniform(0.3, 0.7),
    "min_samples_split": randint(10, 120),
    "min_samples_leaf": randint(5, 30),
}


def search_xgb(X, y, n_iter=50, cv=5, random_state=0):
    """XGBoostのハイパーパラメータをランダムサーチで探して、一番良いモデルを返す"""
    model = XGBRegressor(objective="reg:squarederror")
    search = RandomizedSearchCV(
        model,
        param_distributions=XGB_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
    )
    search.fit(X, y)
    return search.best_estimator_


def search_rf(X, y, n_iter=30, cv=5, random_state=0):
    """ランダムフォレストのハイパーパラメータをランダムサーチで探して、一番良いモデルを返す

    探索自体は木の本数を少なめ(400本)にして行い、見つかったパラメータで
    木の本数を1200本に増やして学習し直す(卒論付録に準拠)。
    """
    base_model = RandomForestRegressor(n_estimators=400, bootstrap=True, random_state=random_state)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=RF_PARAM_DISTRIBUTIONS,
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
def fit_s_learner(X_train, W_train, Y_train, **search_kwargs):
    """S-learner: Wも特徴量に入れた1つのモデルを学習する(XGBoostとRFの両方)"""
    Xw_train = X_train.copy()
    Xw_train["W"] = W_train

    model_xgb = search_xgb(Xw_train, Y_train, **search_kwargs)
    model_rf = search_rf(Xw_train, Y_train, **search_kwargs)
    return model_xgb, model_rf


def predict_s_learner(model, X_test):
    """W=0とW=1それぞれで予測し、その差を処置効果(tau)とする"""
    X_test_w0 = X_test.copy()
    X_test_w0["W"] = 0
    X_test_w1 = X_test.copy()
    X_test_w1["W"] = 1

    mu0 = model.predict(X_test_w0)
    mu1 = model.predict(X_test_w1)
    tau = mu1 - mu0
    return tau, mu0, mu1


# ---------------------------------------------------------------- T-learner
def fit_t_learner(X_train, W_train, Y_train, **search_kwargs):
    """T-learner: 処置群(W=1)・対照群(W=0)で別々のモデルを学習する"""
    X_train_0 = X_train[W_train == 0]
    Y_train_0 = Y_train[W_train == 0]
    X_train_1 = X_train[W_train == 1]
    Y_train_1 = Y_train[W_train == 1]

    model_xgb_0 = search_xgb(X_train_0, Y_train_0, **search_kwargs)
    model_xgb_1 = search_xgb(X_train_1, Y_train_1, **search_kwargs)
    model_rf_0 = search_rf(X_train_0, Y_train_0, **search_kwargs)
    model_rf_1 = search_rf(X_train_1, Y_train_1, **search_kwargs)

    return model_xgb_0, model_xgb_1, model_rf_0, model_rf_1


def predict_t_learner(model_0, model_1, X_test):
    """対照群用モデル・処置群用モデルそれぞれで予測し、差を処置効果(tau)とする"""
    mu0 = model_0.predict(X_test)
    mu1 = model_1.predict(X_test)
    tau = mu1 - mu0
    return tau, mu0, mu1


# ---------------------------------------------------------------- X-learner
def fit_x_learner(X_train, W_train, Y_train, t_model_0, t_model_1, search_fn=search_xgb, **search_kwargs):
    """X-learner: T-learnerのモデルを使って、個体ごとの処置効果を擬似的に作り、それを再学習する

    t_model_0, t_model_1 には T-learner で学習したモデル(1つの学習器だけ、
    例えばXGBoost同士)を渡す。RF版が欲しい場合は t_model_0/1 をRFのモデルにして、
    search_fn=search_rf を指定して呼び出す(XGB版・RF版で2回呼ぶ想定)。
    """
    X_train_0 = X_train[W_train == 0]
    Y_train_0 = Y_train[W_train == 0]
    X_train_1 = X_train[W_train == 1]
    Y_train_1 = Y_train[W_train == 1]

    # D0: 対照群の人が「処置を受けていたら」どうなっていたかの推定値 - 実際の値
    D0 = t_model_1.predict(X_train_0) - Y_train_0
    # D1: 処置群の人の実際の値 - 「処置を受けていなかったら」の推定値
    D1 = Y_train_1 - t_model_0.predict(X_train_1)

    tau0_model = search_fn(X_train_0, D0, **search_kwargs)
    tau1_model = search_fn(X_train_1, D1, **search_kwargs)

    prop_model = LogisticRegression()
    prop_model.fit(X_train, W_train)

    return tau0_model, tau1_model, prop_model


def predict_x_learner(tau0_model, tau1_model, prop_model, X_test):
    """tau0, tau1を傾向スコアで重み付けして最終的なtauを出す

    卒論の元コードに合わせ、重みには P(W=0|x) (predict_probaの0列目)を使っている:
      tau = tau0 * P(W=0|x) + tau1 * P(W=1|x)
    """
    tau0 = tau0_model.predict(X_test)
    tau1 = tau1_model.predict(X_test)
    propensity = prop_model.predict_proba(X_test)[:, 0]  # P(W=0|x)
    tau = tau0 * propensity + tau1 * (1 - propensity)
    return tau, tau0, tau1, propensity
