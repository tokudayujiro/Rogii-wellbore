# exp008: 合体モデル（selector hybrid + CatBoost/LGB 残差）

## 目的

これまでの最良要素を結合して CV を更新する:
- **base** = per-well selector hybrid（PF scale 切替 + beam 混合 + CF hold; exp006, LB 8.711）
- **補正** = LGB / CatBoost が base 残差 `TVT − hybrid` を幾何/GR/空間特徴から予測（exp007 発展）
- **final** = `hybrid + shrink·resid`

## 方法

`build_cache.py`（全 PF scale 版）で `exp007_ml_rows_v2.parquet`（pf3/5/8/12 + beam + 特徴 +
真 TVT, 3.78M 行）を作成。`train.py` でキャッシュから selector hybrid を再構成し、残差を
LGB / CatBoost で学習（坑井単位 GroupKFold(5) OOF）。shrink と learner を CV で選択。

## 結果（全 773 坑井 GroupKFold OOF, per-well RMSE ft）

| モデル | RMSE |
|--------|-----:|
| carry_forward | 12.812 |
| PF (raw scale8) | 8.419 |
| PF α0.9（exp004, LB 8.895） | 8.183 |
| hybrid（exp006, LB 8.711） | 8.133 |
| stack pf8+LGB（exp007） | 8.121 |
| **hybrid + 0.25·CatBoost残差（本実験）** | **8.027** |

(参考: hybrid+lgb+cat*0.25=8.034, hybrid+lgb*0.25=8.056)

## 学び

- **CatBoost 残差が LGB 残差よりわずかに良い**（8.027 vs 8.056 @shrink0.25）。木の種類が異なり
  PF/hybrid の残す系統誤差をうまく拾う。LGB+Cat 平均はこの場合 cat 単独に劣後。
- **base を pf8 から hybrid に変えるだけで 8.121→8.027** に寄与。selector の per-well 最適化と
  残差補正が相補的。
- 全 OOF で exp001 以降の最良。**carry_forward 12.81 → 8.03（−37%）**。

## デプロイ

- `combined.joblib`（lgb + cat + features + shrink=0.25 + learner=cat + base=hybrid）を
  Kaggle Dataset `tokudayujiro/rogii-wellbore-stack-lgb` に追加。
- kernel `MODEL="combined"`（v8）: PF(128/500) 1 回 → pf8/hybrid/beam/特徴 → CatBoost 残差
  → `hybrid + 0.25·resid`。LB は SUBMISSIONS.md 参照。

## 記録

- MLflow: run `exp008_combined`。依存追加: `catboost`。
- 再現: build_cache.py（v2, 約77分）→ train.py（LGB+Cat OOF）。
