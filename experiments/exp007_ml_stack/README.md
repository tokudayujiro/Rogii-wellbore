# exp007: PF 残差スタッキング（PF + LGB）

## 目的

参考解法（LB 7.168）は PF と ML スタックを blend している。ここでは PF が取り切れない
**系統誤差**を、test 時に利用可能な幾何/GR 特徴＋PF/beam 出力から LGB で予測し、
`PF + shrink·resid` で補正する（非リーク）。

## 手法

1. **キャッシュ**（`build_cache.py`）: 全 773 train 坑井で PF(scale8, 64seed/300part) と
   beam を計算し、予測対象行の特徴量＋真 TVT を parquet 化（3.78M 行）。PF は学習なしの
   坑井独立処理なのでリークなし。
2. **学習**（`train.py`）: ターゲット `resid = TVT − PF_pred`。坑井単位 GroupKFold(5) の
   OOF で評価。特徴 19 個（z, dist_from_ps, incl, GR 系, x/y_off, pf_off, beam_off,
   n_eval, z_span, known_gr_*, last_slope, gr_sig）。
3. 最終予測 = `PF_pred + shrink·resid`（shrink を CV で選択）。

## 結果（全 773 坑井 GroupKFold OOF, per-well RMSE ft）

| モデル | RMSE |
|--------|-----:|
| carry_forward | 12.812 |
| PF (raw scale8) | 8.419 |
| PF α0.9（=exp004 提出） | 8.183 |
| **PF + 0.5·LGB残差** | **8.121** |

shrink=0.5 が最良（0.25:8.151 / 0.75:8.287 / 1.0:8.606）。

特徴重要度トップ: `z, x_off, y_off, dist_from_ps, z_span, known_gr_mean, last_slope, n_eval`
→ **空間/幾何が支配的**。LGB は「offset well の dip 的な空間補正」を学習しており、
プロジェクト概要の「近傍坑井の地質傾斜は似る」という事前情報に対応する。

## 学び

- **ML 残差スタッキングは PF を一貫して改善**（8.419→8.121、α0.9 比でも −0.06）。
  CV は全 773 OOF なのでサブサンプル評価より信頼できる。
- 効いているのは GR ではなく**空間特徴**（z/x/y/dist）。PF が GR で取り切れない
  地層の空間トレンドを LGB が補完している＝両者が相補的。
- ただし改善幅は中程度。7.168 との残差は、参考解法のさらなる多段（複数 PF 変種・
  beam の選択的 blend・CatBoost 併用・投影後処理）の積み上げ分と思われる。

## デプロイ

- 学習済み LGB を Kaggle Dataset `tokudayujiro/rogii-wellbore-stack-lgb` として配布。
- kernel `MODEL="stack"`（v7）: PF(128/500)+beam → 特徴量（build_cache と一致）→ LGB 残差
  → `PF + 0.5·resid`。`dataset_sources` に上記を追加済み。
- LB は SUBMISSIONS.md 参照。

## 記録

- MLflow: run `exp007_ml_stack`。
- 再現: `build_cache.py`（PF/特徴キャッシュ, 約75分）→ `train.py`（LGB+CV）。
- 依存追加: `numba`（PF）, `pyarrow`（parquet）, `joblib`（既存）, `lightgbm`（既存）。
