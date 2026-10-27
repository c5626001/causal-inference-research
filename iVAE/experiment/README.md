# iVAE実験: VAE vs iVAEの恒等性比較(2026年度 M1後期研究)

CEVAEの崩壊(posterior collapse)は、VAEの事前分布p(z)がuを見ない(固定)ことに由来する
非識別性が原因、というのが[`CEVAE/experiment/`](../../CEVAE/experiment)側の結論。
その修正案がiVAE(事前分布をp(z\|u)に変える)。ここではまずCEVAE(因果推論)を離れて、
「VAEとiVAEのどちらが真の潜在変数を復元できるか」という**恒等性そのもの**を
人工データで確認する。確認できたら、その先にCEVAEへの拡張(別フォルダ予定)を進める。

参考文献: Khemakhem, Kingma, Monti, Hyvärinen (2020) "Variational Autoencoders and
Nonlinear ICA: A Unifying Framework"。`claude_code/iVAE/refs/Khemakhem2020_iVAE.pdf`
にローカルコピーあり。ここの実験は同論文 5.1節・付録H.1の設定に沿っている。

## やっていること

1. **データ生成**(`dgp.py`): 真の潜在源 s はセグメント(補助変数u)ごとに各次元の分散が
   変わる(非定常性)。これを非線形・可逆な4層→簡略化して2層のleaky_relu MLP(重みは
   直交行列、条件数=1で数値的に安定)で混ぜて観測 x を作る。小さい観測ノイズを加える
   (論文と同じ σ²=0.01)。
2. **モデル**(`models.py`): VAE(prior固定 N(0,I))とiVAE(prior p(z\|u)、uから出す
   prior_netで条件付け)。encoder/decoderの構造はVAE/iVAEで共通、違いはpriorだけ。
3. **ハイパラ探索**: 論文はVAE・iVAEそれぞれについて層数・hidden幅・学習率を探索し、
   「各モデルのベストな設定」同士を比較している(そうしないとVAE側の力不足とzの
   非識別性を混同してしまう)。ここでも同じ理由で、hidden幅{64,128}×層数{2,3}を軽く
   探索し(複雑にしすぎない範囲)、検証データでのELBO(真のsを使わない、教師なしの基準)
   が一番良い設定を選んでから比較する(`models.search_best_config`)。
4. **評価**: MCC(Mean Correlation Coefficient)。復元したzと真のsの間で、置換・符号を
   揃えた上での相関。iVAE論文の標準評価指標。

## 結果(2026-10-27、d=3, 25segments×1000, 5 data_seed)

`run_vae_vs_ivae.ipynb` を参照。**iVAEが一貫してVAEを上回る**(複数seedで安定)。
ただし論文のFigure 2ほど劇的な差(iVAE 0.95+ vs VAE大きく低下)ではなく、効果量は
小さめ(数ポイント差)。理由は付録H.1と比べてハイパラ探索の範囲を絞っている
(論文は層数3〜6・hidden10〜200まで探索)ため。「複雑にしすぎない」方針を優先した
結果なので、方向性の確認としては十分だが、数値そのものは論文の再現ではない。

## フォルダ構成

- `dgp.py` — 人工データ生成(セグメント型非定常ソース+非線形混合)
- `models.py` — VAE/iVAE本体、ハイパラ探索(`search_best_config`)
- `metrics.py` — MCC(Mean Correlation Coefficient)
- `run_vae_vs_ivae.ipynb` — VAE vs iVAEの比較実験本体
- `results/` — 出力CSVを保存する場所

## 動かし方

`run_vae_vs_ivae.ipynb` を上から実行 → `results/vae_vs_ivae_mcc.csv` ができる。
冒頭の `D`, `N_SEGMENTS`, `N_PER_SEGMENT`, `HIDDEN_OPTIONS`, `N_LAYERS_OPTIONS` 等を
変えれば設定を変えられる。元の設定(d=3, 5 data_seed, 探索4パターン×2モデル×5seed)は
数十分かかる。小さくすれば動作確認だけなら数分で終わる。
