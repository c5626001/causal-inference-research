# CEVAE まとめ（参照ノート）

> Causal Effect Variational Autoencoder（Louizos et al., NeurIPS 2017,
> "Causal Effect Inference with Deep Latent-Variable Models"）の自分用リファレンス。
> 間野透太 / 山中卓研。CEVAEの話題が出たらまずこれを参照する。
> 実験設計は同フォルダの [設計書.md](設計書.md) に分離。

最終更新：2026-06-27

---

## 1. 一言でいうと

**未観測の交絡因子 Z を、観測できる代理変数 X（proxy）からVAEで復元し、
その Z を使って因果効果（ITE / CATE）を推定する手法。**

キモは「**X は Z の原因ではなく、Z のノイズ込みの観測代理（proxy）**」という点。
ここを外すと CEVAE の意義そのものが消えるので最重要。

---

## 2. 因果構造（DAG）

```
        Z   (潜在交絡。未観測)
      / | \
     ↓  ↓  ↓
    X   T → Y
```

- Z → X : X は Z の **proxy**（複数のノイジーな測定）
- Z → T : 交絡（処置の割り当てが Z に依存）
- Z → Y : 交絡（結果が Z に依存）
- T → Y : 推定したい因果効果
- 観測できるのは **X, T, Y のみ**。Z は未知。

---

## 3. 生成モデル（Decoder, p）

VAEのデコーダ側で次を学習する：

```
p(z)      = N(0, I)
p(x | z)                       # proxy の再構成（連続ならGaussian, 二値ならBernoulli）
p(t | z)  = Bernoulli(sigmoid(f1(z)))
p(y | t,z)                     # t と z から結果を再構成
```

- y が連続なら `p(y|t,z) = N(μ, σ²)`。**論文では分散σ²は固定**、平均μだけNNで学習 ⇒ 実質MSE。
- y が二値なら Bernoulli ⇒ BCE。

---

## 4. 推論モデル（Encoder, q）

エンコーダで `q(z | x, t, y)` を推定する。TARNet と同じ「t で分岐」構造：

```
g1(x) → 共有表現 h
        ├─ g2  (t=0 側ヘッド) → (μ_z, σ²_z)
        └─ g3  (t=1 側ヘッド) → (μ_z, σ²_z)
実際には観測された t に対応するヘッドだけを使い、z をサンプリング（reparameterization）。
```

---

## 5. ELBO（目的関数）

```
ELBO = E_q[ log p(x, t | z) + log p(y | t, z) ]  −  KL( q(z|x,t,y) || p(z) )

第1項 log p(x,t|z) = log p(x|z) + log p(t|z)   … x の再構成 + t の再構成
第2項 log p(y|t,z)                              … z,t から y を再構成
第3項 KL                                        … q を N(0,I) に近づける正則化
```

学習は `−ELBO` を最小化。

---

## 6. 補助ネットワーク（テスト時に必須）

**問題**：テスト時は x しか観測できない。t, y が未知なので `q(z|x,t,y)` が直接使えない。

**解決**：補助NWを足して同時学習する。
```
q(t | x)        # x から処置を予測
q(y | x, t)     # x, t から結果を予測
```
学習時、ELBOに `log q(t=t*|x) + log q(y=y*|x,t)` を加算して同時学習。

**テスト時のITE推定フロー：**
```
x → t̂ = q(t|x) → ŷ = q(y|x,t̂) → z ~ q(z|x,t̂,ŷ)
→ y0 = p(y|t=0,z), y1 = p(y|t=1,z)
→ ITE_pred = y1 − y0
```

---

## 7. 誤差関数の対応

| 変数 | 型 | 損失 |
|---|---|---|
| t | 二値 | Binary Cross Entropy |
| y | 二値 | Binary Cross Entropy |
| y | 連続 | 負の対数尤度（分散固定 ⇒ 実質MSE） |
| z | — | KL項で N(0,I) に正則化 |

---

## 8. ★重要な注意点・落とし穴（発表で刺されないために）

1. **X は proxy であって交絡因子そのものではない。**
   ふつうの手法（S/T/X-learner, TARNet）は X で調整するが、X≠Z なので
   **残差交絡（residual confounding）が残りバイアスする。** CEVAE はここを Z 復元で埋める。

2. **CEVAE は万能ではない。**
   Rissanen & Marttinen (NeurIPS 2021) "A Critical Look at the Consistency of
   Causal Estimation with Deep Latent Variable Models" が、CEVAE は Z を安定に
   復元できず**ベースラインに負けることも多い**と報告。過大主張は禁物。

3. **効くのは「proxy が中程度」のスイートスポット。**
   - proxy が少ない/ノイズ過多 → Z を復元できず優位消失
   - proxy が多く高情報 → ふつうの手法も X≈Z で調整でき差が消える
   - 中間 → CEVAE が最も光る

4. **学習が不安定。** posterior collapse 対策に KLアニーリング、複数シードで平均±SD報告。

5. **公平性**：全モデルが見るのは X のみに統一。真の Z は Oracle 以外に渡さない。

---

## 9. 評価指標（人工データなら真ITEが分かる）

```
PEHE   = (1/N) Σ (ITE_pred − ITE_true)²
√PEHE  = sqrt(PEHE)              ← 因果推論の標準報告値
MAE    = (1/N) Σ |ITE_pred − ITE_true|
ε_ATE  = |mean(ITE_pred) − mean(ITE_true)|
```
標準ペアは **√PEHE と ε_ATE**。

---

## 10. 実装の選択肢

- **A. PyTorchフルスクラッチ**：Encoder/Decoder/補助NW/ELBOを自前。約150〜250行。
  スライドと1対1で対応・研究の土台になるが学習がデリケート。→ **推奨（主役）**
- **B. Pyro組み込み**：`pyro.contrib.cevae.CEVAE`。安定・高速だがブラックボックス。→ 裏取り用

---

## 11. 関連メモ・リンク

- 実験設計の詳細 → [設計書.md](設計書.md)
- 自分の研究（M1）：潜在交絡因子を含む因果推論。CEVAE/VAEベース。去年のMeta-learner因果推論の発展。
- 去年の卒論（Meta-learner / S,T,X-learner / SHAP）コードはベースラインに流用可能。

---

## 12. 都度更新するポイント（更新したらここを直す）

- [ ] 実験を回したら「実際の√PEHE結果・効いた設定」を 8項 / 設計書に追記
- [ ] CEVAEの式・理解が更新されたら 3〜7項を直す
- [ ] 新しい関連論文（CEVAE批判・改良）を読んだら 8項に追記
- [ ] 実装で詰まった安定化トリックが見つかったら 8-4 に追記
