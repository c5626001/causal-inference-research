# 2025年度卒業論文「Meta-learnerによる因果推論」付録(第7章のコード)より文字起こし
#
# 出典: 去年_2025年度卒業論文.pdf p.44-49「付録 第7章のコード」
# 状況1のコードを掲載。他の状況(状況2〜4)も関数や次元などの違いはあるが、
# meta-learnerの部分に関してはおおむね同じ、と原論文に記載あり。
# 元のPDFはコードが画像(スクリーンショット)として貼られていたため、全文を手で書き起こしたもの。
# そのため実行には元の学習済みオブジェクト(result_s_2等)が一部不足しており、単体では動かない箇所がある。

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, randint
from sklearn.base import clone
from xgboost import XGBRegressor
import shap
from sklearn.model_selection import StratifiedKFold

# データ生成過程
rng = np.random.default_rng(42)


def make_ar1_cov(d: int, rho: float = 0.5) -> np.ndarray:
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def sample_X(N: int, d: int, rho: float = 0.5) -> np.ndarray:
    Sigma = make_ar1_cov(d, rho)
    L = np.linalg.cholesky(Sigma)
    Z = rng.normal(size=(N, d))
    X = Z @ L.T
    return X


def simulate_cate_zero(N=100000, d=5, rho=0.5):

    X = sample_X(N, d, rho)

    # 応答関数の設定
    beta = rng.uniform(1, 30, size=d)
    mu0 = X @ beta
    mu1 = mu0.copy()
    tau_true = mu1 - mu0

    eps0 = rng.normal(0, 1, size=N)
    eps1 = rng.normal(0, 1, size=N)
    Y0 = mu0 + eps0
    Y1 = mu1 + eps1

    return X, tau_true, Y0, Y1


def assign_W_and_observe(X, Y0, Y1, e_const=0.5, seed_w=0):

    rng_w = np.random.default_rng(seed_w)
    N = X.shape[0]

    W = rng_w.binomial(1, e_const, size=N)

    Y = np.where(W == 1, Y1, Y0)
    return W, Y


# データの生成
X_base, tau_base, Y0_base, Y1_base = simulate_cate_zero(
    N=100000, d=5, rho=0.5
)
W05, Y_05 = assign_W_and_observe(X_base, Y0_base, Y1_base, e_const=0.5, seed_w=123)
X1, W1, Y_1, tau_1, Y0_1, Y1_1 = X_base, W05, Y_05, tau_base, Y0_base, Y1_base

# ランダムフォレストのハイパーパラメータ設定
rf_model = RandomForestRegressor(
    n_estimators=400,
    bootstrap=True,
    n_jobs=1,
    random_state=0,
    oob_score=False
)

param_distributions_rf = {
    "max_depth": randint(4, 14),
    "max_features": uniform(0.3, 0.7),
    "min_samples_split": randint(10, 120),
    "min_samples_leaf": randint(5, 30),
}

param_distributions_rf_2 = {
    "max_depth": randint(8, 18),
    "max_features": uniform(0.3, 0.7),
    "min_samples_split": randint(10, 120),
    "min_samples_leaf": randint(5, 30),
}

param_distributions_rf_e = {
    "max_depth": randint(4, 14),
    "max_features": uniform(0.3, 0.7),
    "min_samples_split": randint(5, 50),
    "min_samples_leaf": randint(1, 15),
}


# 評価指標
def RMSE(ITE_true, ITE_pred):
    return np.sqrt(np.mean((ITE_true - ITE_pred) ** 2))


def MAE(ITE_true, ITE_pred):
    MAE = np.mean(np.abs(ITE_true - ITE_pred))
    return MAE


def make_df(X, W, Y, tau_true, Y0, Y1):
    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])

    df["W"] = W
    df["Y"] = Y
    df["Y0"] = Y0
    df["Y1"] = Y1
    df["tau"] = tau_true

    col_order = ["tau", "W", "Y", "Y0", "Y1"] + [f"X{i+1}" for i in range(X.shape[1])]
    df = df[col_order]
    return df


# XGBoostのハイパーパラメータ設定
param_distributions = {
    "n_estimators": randint(300, 2500),
    "learning_rate": uniform(0.01, 0.19),
    "max_depth": randint(2, 9),
    "min_child_weight": randint(1, 10),
    "subsample": uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.7, 0.3),
    "reg_lambda": uniform(0.0, 10.0),
}

xgb_model = XGBRegressor(objective="reg:squarederror")

df1 = make_df(X1, W1, Y_1, tau_1, Y0_1, Y1_1)

# データの分割
df1_train, df1_test = train_test_split(df1, test_size=0.5, random_state=0, stratify=df1['W'])

df1_x_train = df1_train.drop(['tau', 'Y0', 'Y1', 'Y'], axis=True)
df1_x_test = df1_test.drop(['tau', 'Y0', 'Y1', 'Y'], axis=True)

df1_y_train = df1_train['Y']
df1_y_test = df1_test['Y']

# cvのための分割
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv_splits = list(skf.split(df1_x_train, df1_x_train["W"]))

# XGBoostの設定
random_search_s = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_distributions,
    n_iter=50,  # 探索回数
    cv=cv_splits,  # 5-fold CV
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
    random_state=0
)

random_search_t = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_distributions,
    n_iter=50,  # 探索回数
    cv=5,  # 5-fold CV
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
    random_state=0
)

random_search_t0 = clone(random_search_t)
random_search_t1 = clone(random_search_t)
random_search_x0 = clone(random_search_t)
random_search_x1 = clone(random_search_t)

random_search_rf_s = RandomizedSearchCV(
    rf_model,
    param_distributions=param_distributions_rf,
    n_iter=30,
    cv=cv_splits,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
    random_state=0
)

random_search_rf_t = RandomizedSearchCV(
    rf_model,
    param_distributions=param_distributions_rf,
    n_iter=30,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
    random_state=0
)

random_search_rf_t0 = clone(random_search_rf_t)
random_search_rf_t1 = clone(random_search_rf_t)
random_search_rf_x0 = clone(random_search_rf_t)
random_search_rf_x1 = clone(random_search_rf_t)

# ===== S-learner =====
# XGBoostのハイパーパラメータ探索
random_search_s.fit(df1_x_train, df1_y_train)
s_learner_XGB = random_search_s.best_estimator_

# ランダムフォレストのハイパーパラメータ探索
random_search_rf_s.fit(df1_x_train, df1_y_train)
s_learner_rf = random_search_rf_s.best_estimator_.set_params(n_estimators=1200, n_jobs=-1)
s_learner_rf.fit(df1_x_train, df1_y_train)

results_df1 = pd.DataFrame(random_search_s.cv_results_)

# W=0のデータとW=1のデータを作成
df1_test_s0 = df1_x_test.copy()
df1_test_s0['W'] = 0
df1_test_s1 = df1_x_test.copy()
df1_test_s1['W'] = 1

result_s = df1_test.copy()
result_s = result_s[['W', 'tau']]

# テストデータに対して予測
result_s['mu0_XGB'] = s_learner_XGB.predict(df1_test_s0)
result_s['mu1_XGB'] = s_learner_XGB.predict(df1_test_s1)
result_s['tau_XGB'] = result_s['mu1_XGB'] - result_s['mu0_XGB']

result_s['mu0_rf'] = s_learner_rf.predict(df1_test_s0)
result_s['mu1_rf'] = s_learner_rf.predict(df1_test_s1)

# 処置効果の算出
result_s['tau_rf'] = result_s['mu1_rf'] - result_s['mu0_rf']

# ===== T-learner =====
# W=0のデータとW=1のデータに分割
df1_1_train = df1_train.copy()
df1_1_test = df1_test.copy()

df1_1_train_0 = df1_1_train[df1_1_train['W'] == 0]
df1_1_train_1 = df1_1_train[df1_1_train['W'] == 1]
x_train_t0 = df1_1_train_0.drop(['W', 'tau', 'Y0', 'Y1', 'Y'], axis=True)
x_train_t1 = df1_1_train_1.drop(['W', 'tau', 'Y0', 'Y1', 'Y'], axis=True)

df1_1_test_0 = df1_1_test[df1_1_test['W'] == 0]
df1_1_test_1 = df1_1_test[df1_1_test['W'] == 1]
x_test_t0 = df1_1_test_0.drop(['W', 'tau', 'Y0', 'Y1', 'Y'], axis=True)
x_test_t1 = df1_1_test_1.drop(['W', 'tau', 'Y0', 'Y1', 'Y'], axis=True)

y_train_t0 = df1_1_train_0['Y']
y_train_t1 = df1_1_train_1['Y']

# ランダムフォレストのハイパーパラメータ探索
random_search_rf_t0.fit(x_train_t0, y_train_t0)
random_search_rf_t1.fit(x_train_t1, y_train_t1)
t0_learner_rf = random_search_rf_t0.best_estimator_.set_params(n_estimators=1200, n_jobs=-1)
t1_learner_rf = random_search_rf_t1.best_estimator_.set_params(n_estimators=1200, n_jobs=-1)
t0_learner_rf.fit(x_train_t0, y_train_t0)
t1_learner_rf.fit(x_train_t1, y_train_t1)

# XGBoostのハイパーパラメータ探索
random_search_t0.fit(x_train_t0, y_train_t0)
random_search_t1.fit(x_train_t1, y_train_t1)
results_t0 = pd.DataFrame(random_search_t0.cv_results_)
results_t1 = pd.DataFrame(random_search_t1.cv_results_)
t_learner_XGB0 = random_search_t0.best_estimator_
t_learner_XGB1 = random_search_t1.best_estimator_

t_tau = df1_1_test.copy()
t_tau = t_tau['tau']
test_t = df1_1_test.copy()
test_t = test_t.drop(['tau', 'W', 'Y', 'Y1', 'Y0'], axis=True)

# テストデータに対して予測
result_t = test_t.copy()
result_t['mu0_rf'] = t0_learner_rf.predict(test_t)
result_t['mu1_rf'] = t1_learner_rf.predict(test_t)
result_t['mu0_XGB'] = t_learner_XGB0.predict(test_t)
result_t['mu1_XGB'] = t_learner_XGB1.predict(test_t)

# 処置効果の算出
result_t['tau_rf'] = result_t['mu1_rf'] - result_t['mu0_rf']
result_t['tau_XGB'] = result_t['mu1_XGB'] - result_t['mu0_XGB']
result_t = result_t[['mu0_rf', 'mu1_rf', 'mu0_XGB', 'mu1_XGB', 'tau_rf', 'tau_XGB']]
result_t = pd.concat([result_t, t_tau], axis=1)

# ===== X-learner =====
# D0用のデータの作成
x_train_x0 = x_train_t0.copy()
x_train_x0['mu1_rf'] = t1_learner_rf.predict(x_train_t0)
x_train_x0['mu1_XGB'] = t_learner_XGB1.predict(x_train_t0)
x_train_x0['Y'] = df1_1_train_0['Y']
x_train_x0['D0_rf'] = x_train_x0['mu1_rf'] - x_train_x0['Y']
x_train_x0['D0_XGB'] = x_train_x0['mu1_XGB'] - x_train_x0['Y']

# D1用のデータ作成
x_train_x1 = x_train_t1.copy()
x_train_x1['mu0_rf'] = t0_learner_rf.predict(x_train_t1)
x_train_x1['mu0_XGB'] = t_learner_XGB0.predict(x_train_t1)
x_train_x1['Y'] = df1_1_train_1['Y']
x_train_x1['D1_rf'] = x_train_x1['Y'] - x_train_x1['mu0_rf']
x_train_x1['D1_XGB'] = x_train_x1['Y'] - x_train_x1['mu0_XGB']

# D0とD1のハイパーパラメータ予測(XGBoost)
random_search_x0.fit(x_train_x0[['X1', 'X2', 'X3', 'X4', 'X5']], x_train_x0['D0_XGB'])
random_search_x1.fit(x_train_x1[['X1', 'X2', 'X3', 'X4', 'X5']], x_train_x1['D1_XGB'])
results_x0 = pd.DataFrame(random_search_x0.cv_results_)
results_x1 = pd.DataFrame(random_search_x1.cv_results_)
model_tau0_XGB = random_search_x0.best_estimator_
model_tau1_XGB = random_search_x1.best_estimator_

# D0とD1のハイパーパラメータ予測(ランダムフォレスト)
random_search_rf_x0.fit(x_train_x0[['X1', 'X2', 'X3', 'X4', 'X5']], x_train_x0['D0_rf'])
random_search_rf_x1.fit(x_train_x1[['X1', 'X2', 'X3', 'X4', 'X5']], x_train_x1['D1_rf'])
x0_learner_rf = random_search_rf_x0.best_estimator_.set_params(n_estimators=1200, n_jobs=-1)
x1_learner_rf = random_search_rf_x1.best_estimator_.set_params(n_estimators=1200, n_jobs=-1)
x0_learner_rf.fit(x_train_x0[['X1', 'X2', 'X3', 'X4', 'X5']], x_train_x0['D0_rf'])
x1_learner_rf.fit(x_train_x1[['X1', 'X2', 'X3', 'X4', 'X5']], x_train_x1['D1_rf'])

# 傾向スコア算出モデルの作成
prop_model = LogisticRegression()
prop_model.fit(df1_train[['X1', 'X2', 'X3', 'X4', 'X5']], df1_train['W'])
propensity_score = prop_model.predict_proba(df1_train[['X1', 'X2', 'X3', 'X4', 'X5']])[:, 1]

test_x = test_t.copy()

# 処置効果の予測
test_x['tau0_rf'] = x0_learner_rf.predict(test_t)
test_x['tau1_rf'] = x1_learner_rf.predict(test_t)
test_x['tau0_XGB'] = model_tau0_XGB.predict(test_t)
test_x['tau1_XGB'] = model_tau1_XGB.predict(test_t)
test_x['propensity'] = prop_model.predict_proba(test_t)[:, 0]

test_x['tau_rf'] = test_x['tau0_rf'] * test_x['propensity'] + test_x['tau1_rf'] * (1 - test_x['propensity'])
test_x['tau_XGB'] = test_x['tau0_XGB'] * test_x['propensity'] + test_x['tau1_XGB'] * (1 - test_x['propensity'])

result_x = test_x[['tau0_rf', 'tau1_rf', 'tau0_XGB', 'tau1_XGB', 'propensity', 'tau_rf', 'tau_XGB']]
result_x['tau'] = t_tau

# 結果の表示
print(RMSE(result_s['tau'], result_s['tau_rf']))
print(RMSE(result_s['tau'], result_s['tau_XGB']))
print(RMSE(result_t['tau'], result_t['tau_rf']))
print(RMSE(result_t['tau'], result_t['tau_XGB']))
print(RMSE(result_x['tau'], result_x['tau_rf']))
print(RMSE(result_x['tau'], result_x['tau_XGB']))

# 状況2(result_s_2等)は本付録には未掲載。原論文には他状況も構造は同様と記載あり。
# print(RMSE(result_s_2['tau'], result_s_2['tau_rf']))
# print(RMSE(result_s_2['tau'], result_s_2['tau_XGB']))
# print(RMSE(result_t_2['tau'], result_t_2['tau_rf']))
# print(RMSE(result_t_2['tau'], result_t_2['tau_XGB']))
# print(RMSE(result_x_2['tau'], result_x_2['tau_rf']))
# print(RMSE(result_x_2['tau'], result_x_2['tau_XGB']))

# ===== SHAP値の計算 =====
explainer_s = shap.TreeExplainer(s_learner_rf)
n = 2000
idx = df1_test_s0.sample(n, random_state=0).index
shap_test_s0 = df1_test_s0.loc[idx].copy()
shap_test_s1 = df1_test_s1.loc[idx].copy()
shap_values_s0 = explainer_s.shap_values(shap_test_s0)
shap_values_s1 = explainer_s.shap_values(shap_test_s1)
shap_values_ITE = shap_values_s1 - shap_values_s0

shap.summary_plot(shap_values_ITE, shap_test_s1, max_display=21)
