# exp002: PS オフセット直接回帰 + typewell GR–TVT 相関特徴

## 目的

exp001 の学び（増分 ΔTVT の積分はドリフトが累積して carry_forward に劣後）を受け、
**ターゲットを PS からのオフセット `y[i] = TVT[i] − TVT[PS-1]` に変えて直接回帰**する。
あわせて geosteering の本命シグナルである **typewell の GR(TVT) との相関**を特徴化する。

## 定式化

- ターゲット: `y[i] = TVT[i] − TVT[PS-1]`（PS からのオフセット）。
  予測 TVT は `tvt0 + ŷ`。最低でも `ŷ=0`（= carry_forward）を学べる定式化。
- typewell 特徴（`src/rogii/features.py::_typewell_match`）:
  - `match_off` — last_tvt ±40 ft の TVT グリッドで、平滑化した横坑井 GR に最も近い
    typewell GR を与える TVT を選び、`整合TVT − last_tvt` を返す。
  - `match_misfit` — その整合点の GR 残差（信頼度の代理）。

## 検証

坑井単位 GroupKFold(5)。各 valid 坑井で実 PS を使い PS 以降の TVT を復元して RMSE。

## 結果（GroupKFold 5, CV RMSE ft）

| モデル | CV RMSE | 備考 |
|--------|--------:|------|
| **carry_forward** | **12.81** | 依然ベスト（exp001 と同値） |
| lgb_offset | 13.60 | exp001 の lgb_delta(16.48) から大幅改善も CF に未達 |

fold 別: cf=[12.59, 12.43, 12.51, 13.38, 13.16] / lgb=[13.24, 12.94, 13.43, 14.19, 14.19]。

提出: `outputs/submissions/exp002_offset.csv`（参考）。
**現状ベストは引き続き `exp001_carryforward.csv`（CV 12.81）。**

## 学び（重要）

- **オフセット直接回帰は増分積分より明確に良い**（16.48 → 13.60）。ドリフト累積は解消した。
- **だが CF には勝てない**。特徴重要度を見ると、モデルは
  `last_slope` / `z` / `known_gr_mean,std` / `x_off,y_off` など **坑井ごとの定数**に依存しており、
  これらは未知坑井へ汎化せず、結局フラット仮定（CF）にノイズを足して悪化させている。
- **typewell マッチ特徴はほぼ効いていない**（`match_off`=186, `match_misfit`=55 と最下位群）。
  原因は素朴な「点ごとの最近傍 GR」マッチが **非一意**（同じ GR を与える TVT が多数）で、
  geosteering の本質である **GR 波形のシーケンス整合**を捉えられていないこと。

## 次の改善案（exp003 方針）

- **GR 波形のシーケンス整合で TVT を復号する**（本命）。
  TVT を離散グリッド化し、各点の発光コスト `|gr_typewell(TVT) − GR_h[i]|` と、
  なめらかさの遷移コスト（`|ΔTVT − 幾何prior|` を罰則）を組む **Viterbi/DTW 復号**。
  PS の既知 TVT を始点に固定。これは「GR が typewell のどの深度に一致するか」を
  系列全体で整合させる定式化で、点ごとマッチの非一意性を解消する。
  Felzenszwalb の一般化距離変換を使えば 1 ステップ O(G) で計算可能。
- 復号 TVT を**特徴**として LGB に渡す（生予測 + 残差学習）案も併用。
- 距離別の重み付け・後段補正（PS から遠いほど誤差増）。

## 記録

- MLflow: experiment `rogii-wellbore` / run `exp002_offset_typewell`（`mlruns/`, `mlflow.db`）。
- 再現: `uv run python experiments/exp002_offset_typewell/train.py`
- リーク（test=train 同梱）は exploit しない方針。
