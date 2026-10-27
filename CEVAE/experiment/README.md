# CEVAE実験: 潜在交絡因子を含む因果推論(2026年度 M1前期研究)

CEVAEが抱える崩壊(posterior collapse)の問題を構造的に解消する方向の実験は
[`iVAE/experiment/`](../../iVAE/experiment) で行っている(M1後期研究)。

CEVAE(Causal Effect Variational Autoencoder)が、潜在交絡因子Zがある状況で
普通の因果推論手法(X-learner)よりITEを正確に推定できるかを検証する。

元になった生ノートブック(`CEVAE実験.ipynb` 〜 `_v5.ipynb`, `CEVAE診断_seed3.ipynb`)は
探索過程のメモ書きでGitHubには含めていない(ローカルの`claude_code/CEVAE実験/`に残っている)。
ここではその中で一番重要な2つの実験(v4=1次元Z, v5=2次元Z)を整理し直して、再実行できる形にしてある。

## フォルダ構成

- `cevae_model.py` — CEVAE本体(v4/v5共通)
- `xlearner_baseline.py` — 比較対象のX-learner(共通)
- `dgp.py` — 人工データ生成(1次元版・2次元版)
- `diagnostics.py` — 崩壊(posterior collapse)検出の診断指標
- `run_1d_experiment.ipynb` — 元v4相当: 潜在交絡因子Zが1次元の場合の実験
- `run_2d_experiment.ipynb` — 元v5相当: Zが2次元の場合の実験
- `compare_1d_vs_2d.ipynb` — 「1次元のときのCEVAE vs X-learner」「2次元のときのCEVAE vs X-learner」を
  横に並べて見せる(1次元と2次元そのものを比較しているわけではない。発表スライド用の数値・図を作る)
- `results/` — 各実験ノートブックが出力したCSVを保存する場所

## 動かし方

1. `run_1d_experiment.ipynb` を上から実行 → `results/1d_main_results.csv` などができる
2. `run_2d_experiment.ipynb` を上から実行 → `results/2d_main_results.csv` ができる
3. `compare_1d_vs_2d.ipynb` を実行 → 「1次元ならCEVAEが勝てるが、2次元だと勝てない」の比較表・図が出る

計算量の注意: 元の設定(卒論相当のハイパラ探索回数・seed数)はそれぞれ数十分かかる。
各ノートブック冒頭の `N_ITER_*` や `N_DATA_SEEDS` 等を小さくすれば、動作確認だけなら数分で終わる。

## 既知の注意点(環境依存の問題)

**`torch`を先にimportしてから`XGBRegressor.fit()`を呼ぶと、環境によってはクラッシュする**
(PyTorchとXGBoostが使うOpenMPライブラリの競合)。そのため各ノートブックでは
`xlearner_baseline`(XGBoost関連)を先にimportし、`torch`はその後にimportしている。
自分でセルを足す場合もこの順番を守ること。

また、PyTorch(CEVAE)とjoblibの並列化(`n_jobs=-1`)を同じプロセスで併用すると、
macOSでは子プロセスがデッドロックすることがあったため、`xlearner_baseline.search_xgb`の
デフォルトは`n_jobs=1`にしてある。
