# causal-inference-research

青山学院大学 山中卓研究室での因果推論の研究コードをまとめたリポジトリ。

## 構成

- [`卒研/`](卒研) — 2025年度卒業論文「Meta-learnerによる因果推論」
- [`CEVAE/`](CEVAE) — 2026年度(M1)の研究「潜在交絡因子を含む因果推論」(CEVAE/iVAEベース)。卒研の発展版。
  整理されたコードは [`CEVAE/experiment/`](CEVAE/experiment) にある。

詳しくは各フォルダのREADMEを参照。

## 動作環境

Python 3 + numpy, pandas, scikit-learn, xgboost, shap, scipy, matplotlib, torch。
`pip install -r requirements.txt` で一通り入る。
