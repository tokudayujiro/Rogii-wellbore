# exp003: GR シーケンス整合（Viterbi 復号）+ carry_forward シュリンク

## 目的

geosteering の本命シグナル＝「横坑井 GR が typewell の GR(TVT) のどの深度に一致するか」を
**系列全体で整合**させて TVT を復号する。exp002 の点ごと最近傍 GR マッチ（非一意で無力）を、
なめらかさ制約と始点固定の **Viterbi 復号**で置き換える。

## 手法（`src/rogii/decode.py`）

- hidden state = TVT[i] を grid 離散化（`last_tvt ± radius`, 分解能 `step`）。
- 発光コスト: `(gr_typewell(TVT) − gr_horizontal_smoothed[i])²`。
- 遷移コスト: `lam·(ΔTVT)²`（なめらかさ。prior=stay）。窓 `window` で探索制限。
- **GR キャリブレーション**: PS 既知区間（TVT も GR も既知）で
  `gr_h ≈ a·gr_tw(TVT)+b` を最小二乗推定し、横坑井↔typewell の GR スケール差を補正（slide 9）。
- 始点 `TVT[PS-1]` を grid 中心に固定して Viterbi 復号。
- **CF へシュリンク**: `TVT = tvt0 + alpha·(viterbi − tvt0)`。

## 検証

坑井単位 GroupKFold(5)。**alpha は各 fold の train 側のみで選択**し valid で評価（リーク防止）。

## 結果（GroupKFold 5, CV RMSE ft）

| モデル | CV RMSE | 備考 |
|--------|--------:|------|
| carry_forward | 12.814 | exp001 ベスト |
| viterbi (alpha=1, 生) | 13.831 | 生の復号は GR ノイズに引かれ CF に劣後 |
| **blend (alpha≈0.2)** | **12.764** | **全 5 fold で CF を上回る** |

fold 別 blend vs cf: (12.579/12.594), (12.379/12.428), (12.481/12.505), (13.332/13.381),
(13.050/13.160) — **5/5 で blend が勝ち**。選択 alpha = [0.2, 0.2, 0.2, 0.2, 0.15]、final=0.2。

ハイパラ: `lam=20, radius=12, step=0.5, window=8, gr_win=15, calib_k=400`。

提出: `outputs/submissions/exp003_viterbi.csv`。

## 学び（重要）

- **生の Viterbi 復号は CF に負ける**（13.83）。GR(TVT) は非一意で、系列整合だけでは
  ノイズに引かれて誤った深度に流れる。
- **CF へのシュリンク（alpha≈0.2）で初めて CF を超える**（12.764, 全 fold 一貫）。
  復号の方向（地層の上下トレンド）には弱い実シグナルがあり、その 2 割だけ採用すると
  フラット仮定にわずかな改善を上乗せできる。
- ただし **改善幅は ~0.05 ft（約 0.4%）と小さい**。CV→LB の乖離（exp001 で +3 ft）を
  考えると LB で消える可能性あり。実 LB で確認する。

## 次の改善案（exp004 方針）

- 発光を **GR 波形の形状/勾配のクロス相関**にして非一意性を低減（絶対値マッチからの脱却）。
- **offset well（近傍坑井）の dip を空間補間**して遷移 prior に入れる（stay からの改善）。
- alpha を**距離・misfit 依存**にする（PS 近傍は信頼、遠方は CF 寄り）。
- 復号 TVT を**特徴として LGB に渡す**（exp002 の枠組みと統合）。

## 記録

- MLflow: experiment `rogii-wellbore` / run `exp003_viterbi_decode`。
- 再現: `uv run python experiments/exp003_viterbi_decode/train.py`
- 提出 kernel: `kaggle/submission/rogii_submit.py`（`MODEL="viterbi_blend"`）。
  実験出力 `exp003_viterbi.csv` と**完全一致を検証済み**。
