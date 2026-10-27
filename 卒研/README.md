# 卒研 — Meta-learnerによる因果推論(2025年度卒業論文)

S-learner・T-learner・X-learnerの3手法で、CATE(条件付き平均処置効果)を推定する。
指導: 山中卓准教授。

## フォルダ構成

- [`meta_learner_scenario1.py`](meta_learner_scenario1.py) — 卒論本体の付録(第7章のコード)をそのまま文字起こししたもの。参考用に残してある。実際に動かすなら下の`synthetic/`・`real_data/`を使うこと。
- [`synthetic/`](synthetic) — 人工データでの数値実験(卒論7章)。S/T/X-learnerの性能を6つの状況(線形/非線形、効果ゼロ/あり、処置群バランス/不均衡)で比較する。
- [`real_data/`](real_data) — 実データでの応用(卒論8章)。クーポンキャンペーンデータに適用し、CATEを推定する。

## 動かし方

```bash
pip install -r ../requirements.txt
cd synthetic && jupyter notebook run_synthetic.ipynb
cd ../real_data && jupyter notebook run_real_data.ipynb
```

どちらのノートブックも、上から実行すれば結果(RMSE/MAE、SHAPの図など)が表示される。
計算量が重いので、まず試したい場合はノートブック内の`N`や`N_ITER`を小さくするとよい。

## 卒論からの修正点(文字起こし・整理の過程で見つけたもの)

元の卒論・コードと、今回整理し直したコードで異なる箇所。詳しい理由は各コードのコメント・docstringを参照。

| 箇所 | 元の卒論・コード | 今回のコード | 理由 |
| --- | --- | --- | --- |
| 状況3・4の次元d | 本文表記は d=15 | d=20(デフォルト、書き換え可) | 本文中のβの次元表記(20)や元ノートブックのコードはd=20で一貫していたため、d=15は本文の誤記と判断 |
| 状況5・6のμ1 | 元ノートブックは `cos(2*x4*x5)` | `cos(x4*x5)`(係数2なし) | 卒論本文の数式通りに修正 |
| 実データ版のSHAP | `(1-e)*shap1 - e*shap0`(引き算) | `e*shap_tau0 + (1-e)*shap_tau1`(足し算) | tauの計算式(`tau0*P(W=0) + tau1*P(W=1)`)と整合させるため |

いずれも元のコードを`git log`やこのREADMEで追えるようにしてあるので、発表等で聞かれた場合はここを参照。
