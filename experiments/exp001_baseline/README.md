# exp001: ベースライン + LightGBM 増分モデル

## 目的

TVT 予測の基準値を確立する。naive ベースラインと、汎化する初手モデルを比較。

## 定式化

- ターゲットを **TVT の 1 ステップ増分 `d[i] = TVT[i] - TVT[i-1]`** に変換。
- 既知の `TVT[PS-1]`（= `TVT_input` 最終値）から `d` を累積して TVT を復元。
- これは geosteering における「各点の地層 dip を推定し積分する」操作に対応。

## 比較モデル

| モデル | 内容 |
|--------|------|
| `carry_forward` | `d=0`（PS の TVT を定数外挿。地層 flat 仮定） |
| `linear` | `d = last_slope × dMD`（PS 直前 100 点の dip を一定外挿） |
| `lgb_delta` | LightGBM で `d` を回帰（GR・GR勾配/rolling・幾何・PS距離・既知dip 等 15 特徴） |

## 検証

- 坑井単位 **GroupKFold(5)**（同一坑井が train/valid に跨らない）。
- 各 valid 坑井で実 PS を使い、PS 以降の TVT 予測を復元して RMSE。
- 本番（test 3 坑井）と同じ「前半既知・後半予測」設定を再現。

## 特徴量（test 時に利用可能なもののみ）

GR（線形補間）, GR 勾配, GR rolling 平均/標準偏差, 既知区間平均からの GR 偏差,
dMD, 傾斜 dZ/dMD, Z, PS からの距離/インデックス, PS 直前 dip(last_slope),
既知区間 GR 平均/標準偏差, PS からの X/Y オフセット。

> TVT は PS 以降を特徴量に使わない（リーク防止）。

## 結果（GroupKFold 5, CV RMSE ft）

| モデル | CV RMSE | 備考 |
|--------|--------:|------|
| **carry_forward** | **12.81** | 最良。PS の TVT を定数外挿 |
| lgb_delta | 16.48 | 増分予測の積分でドリフト累積 → CF に劣後 |
| linear | 59.38 | PS 直前 dip の一定外挿は overshoot し大幅劣化 |

提出: `outputs/submissions/exp001_carryforward.csv`（現状ベスト）, `exp001_lgb.csv`（参考）。

## 学び（重要）

- **ラテラル区間は geosteering で地層内に保たれ TVT がほぼ一定** → carry_forward が強い基準。
- **増分(ΔTVT)を積分する定式化は誤差が累積（ランダムウォーク的ドリフト）**し、自明な定数外挿に負ける。
  ΔTVT が極小（~1e-3 ft/step）で、3000+ ステップ積分すると微小バイアスが致命的。
- **landing 直後の dip を一定外挿（linear）は overshoot**。地層は着地後フラット化する。

## 次の改善案（exp002 方針）

- **増分積分をやめ、`TVT − TVT[PS-1]`（PS からのオフセット）を直接回帰**。最低でも 0（=CF）を学べ、
  そこから GR シグナルで上積みできる定式化にする。
- **typewell GR–TVT 相関**を明示的に特徴化（横坑井 GR を typewell の GR(TVT) にマッチング）。本命の物理シグナル。
- 近傍坑井（offset well）の dip を空間補間して事前情報に。
- PS からの距離で誤差が増える想定 → 距離別の重み付け・後段補正。

## 記録

- MLflow: experiment `rogii-wellbore` / run `exp001_baseline`（`mlruns/`）。
- 提出: `outputs/submissions/exp001_lgb.csv`。
- リーク（test=train 同梱）は exploit しない方針。詳細は `docs/agent/data-catalog.md`。
