"""実データ(クーポンキャンペーン)の読み込みと前処理。

卒論8章で使ったデータセット(couponsretailer.csv)を、学習に使える形に整える。
処置(W) = coupons_Any (クーポンを配布したかどうか)
結果(Y)  = dailyspending (日次支出額)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# 学習に使う列。これ以外の列(customer_id、個別クーポンの種類別の列など)は使わない。
USE_COLUMNS = [
    "period",
    "age_range",
    "married",
    "rented",
    "family_size",
    "income_bracket",
    "dailyspending_preperiod",
    "coupons_Any_preperiod",
    "coupons_Any_redeemed_preperiod",
    "coupons_Any",       # 処置 W
    "dailyspending",     # 結果 Y
]

TREATMENT_COL = "coupons_Any"
OUTCOME_COL = "dailyspending"

TRAIN_PERIOD_RANGE = (1, 25)   # この範囲のperiodを訓練データにする
TEST_PERIOD_RANGE = (26, 33)   # この範囲のperiodをテストデータにする


def load_and_split(csv_path: str, n_cv_splits: int = 5):
    """CSVを読み込み、欠損行を除いたうえで訓練/テストに分割する(period基準)

    戻り値: (train_df, test_df, cv_splits)
      train_df, test_df は period列を落とした最終的な特徴量テーブル。
      cv_splits は period列が残っている間に作った時系列CV分割(S-learnerのハイパラ探索に使う)。
    """
    df = pd.read_csv(csv_path)
    df = df[USE_COLUMNS].dropna().copy()

    train_df = df[df["period"].between(*TRAIN_PERIOD_RANGE)].copy()
    test_df = df[df["period"].between(*TEST_PERIOD_RANGE)].copy()

    # period列があるうちに時系列CVを作る
    cv_splits = make_period_timeseries_cv(train_df, period_col="period", n_splits=n_cv_splits)

    train_df = train_df.drop(columns="period")
    test_df = test_df.drop(columns="period")

    return train_df, test_df, cv_splits


def make_period_timeseries_cv(df: pd.DataFrame, period_col: str = "period", n_splits: int = 5):
    """period単位でTimeSeriesSplitする(RandomizedSearchCVのcv=にそのまま渡せる)

    通常のランダムなK-foldだと未来のperiodのデータで過去を予測する形になり
    時系列として不自然なため、period(時間)の前後関係を保ったまま分割する。
    """
    df = df.reset_index(drop=True)
    periods = df[period_col].to_numpy()
    uniq_periods = np.sort(np.unique(periods))

    tss = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_p_idx, valid_p_idx in tss.split(uniq_periods):
        train_periods = uniq_periods[train_p_idx]
        valid_periods = uniq_periods[valid_p_idx]
        train_idx = np.where(np.isin(periods, train_periods))[0]
        valid_idx = np.where(np.isin(periods, valid_periods))[0]
        splits.append((train_idx, valid_idx))

    return splits
