# exp014: 3トラッカー不一致特徴 + LGB/CatBoost blend

## 目的
exp013(7.890) に PF/pf_z/beam の不一致・pf_z 軌跡派生を特徴追加。3トラッカーの食い違いは
局所的な不確実性の代理で、残差モデルが「base が外れやすい箇所」を学べる。

## 追加特徴
pf_pfz_diff, pfz_beam_diff, tracker_spread(3者std), pfz_slope, pfz_curv（+ exp013 の pfz_off）

## 結果（全773坑井 GroupKFold OOF, per-well RMSE ft）

| | RMSE |
|---|--:|
| exp013 (cat, pfz_off) | 7.890 |
| cat*0.6 | 7.834 |
| **lgb+cat*0.6** | **7.828** |

CV 系譜: 7.971(exp009) → 7.890(exp013) → **7.828(exp014)**。carry_forward 12.81 比 **−39%**。

## デプロイ
kernel v11（`MODEL="combined"`）が pf_z + 3トラッカー特徴を計算し、LGB+CatBoost 平均残差を
hybrid に shrink0.6 で加算。`tracker.joblib` を最優先ロード（Kaggle Dataset 同梱）。LB は SUBMISSIONS.md。