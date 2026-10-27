# causal-inference-research

青山学院大学 山中卓研究室での因果推論の研究コードをまとめたリポジトリ。

## 構成

2026年度(M1)の研究テーマ「潜在交絡因子を含む因果推論」は、前期にCEVAEで取り組み、
後期にその限界(posterior collapse=識別不可能性)をiVAEで構造的に解消する方向に進んでいる。

- [`卒研/`](卒研) — 2025年度卒業論文「Meta-learnerによる因果推論」
- [`CEVAE/`](CEVAE) — 2026年度(M1)前期の研究。CEVAEで潜在交絡因子Zの推定に挑戦。
  整理されたコードは [`CEVAE/experiment/`](CEVAE/experiment) にある。
- [`iVAE/`](iVAE) — 2026年度(M1)後期の研究。CEVAEの崩壊を補助変数つきprior(iVAE)で解消する方向。
  整理されたコードは [`iVAE/experiment/`](iVAE/experiment) にある。

詳しくは各フォルダのREADMEを参照。

## 動作環境

Python 3 + numpy, pandas, scikit-learn, xgboost, shap, scipy, matplotlib, torch。
`pip install -r requirements.txt` で一通り入る。
