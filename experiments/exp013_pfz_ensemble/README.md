# exp013: 第2トラッカー pf_z のアンサンブル

## 目的
参考NBの Z速度連成PF（pf_z）を非リーク実装し、既存 PF/hybrid と組み合わせて改善する。

## pf_z とは（src/rogii/pf.py）
- 状態 = **TVT 直接**（PF本体は TVT+Z）、速度 = dTVT/dMD。
- **幾何 dip 速度 prior**: 既知区間で `dTVT/dMD ~ beta·(dZ/dMD)+icpt` を回帰し、粒子速度が
  この幾何予測から外れると罰則。坑井の傾きから期待される地層変化を事前情報にする。
- **raw + 平滑 GR の二重尤度**（窓5）。

## 結果（全773坑井 GroupKFold OOF, per-well RMSE ft）

| | RMSE |
|---|--:|
| hybrid (base) | 8.133 |
| exp009 (hybrid + rich残差) | 7.971 |
| **exp013 (+ pfz_off 特徴, shrink0.5)** | **7.890** |

## 学び
- **pf_z を base にブレンドしても raw は改善しない**（w=1.0=hybrid が最良, 0.6で9.38に悪化）。
  全773坑井では hybrid base が強く、pf_z 直接ブレンドは劣化。
- **だが pf_z 予測を残差モデルの特徴 `pfz_off` にすると改善**（7.971→7.890）。CatBoost が
  「pf_z と hybrid の差」から系統誤差を学べる。100坑井ゲートで PF+0.3pf_z が効いたのは
  サブサンプル特有で、全体では特徴化が正解だった。
- shrink が 0.4→0.5 に上がり残差の信頼度向上。

## デプロイ
kernel v10（`MODEL="combined"`）が pf_z も計算し `pfz_off` を特徴に追加、pfz_ensemble.joblib を
優先ロード。Kaggle Dataset に同梱。LB は SUBMISSIONS.md。
