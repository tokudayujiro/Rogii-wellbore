# exp009: 残差スタックの特徴量拡充

## 目的

exp008（hybrid + CatBoost 残差, CV 8.027）の残差学習に、**PF を再計算せずキャッシュ列から
派生できる特徴**を追加して改善する。

## 追加特徴（9 個, `add_features`）

- `pf_beam_diff = pf_off − beam_off`（PF と beam の不一致 = 不確実性の代理）
- `dist2`（距離²）, `frac_idx`（坑井内相対位置 idx/n_eval）, `abs_incl`
- `pf_slope` / `pf_curv`（PF 予測の局所勾配・曲率 = ドリフト傾向）
- `z_off`（eval 先頭からの Z 変位）, `gr_rmean_long` / `gr_rstd_long`（窓 101）

## 結果（全 773 坑井 GroupKFold OOF, per-well RMSE ft）

| モデル | RMSE |
|--------|-----:|
| hybrid（base） | 8.133 |
| exp008 hybrid+CatBoost*0.25（19 特徴） | 8.027 |
| **exp009 hybrid+CatBoost*0.4（28 特徴）** | **7.971** |

CatBoost 残差の重要特徴トップ: `known_gr_mean, z_span, last_slope, n_eval, z, x_off,
known_gr_std, gr_sig, y_off, pf_beam_diff`。追加した `pf_beam_diff` が 10 位に入り、
shrink も 0.25→0.4 に上がった（残差の信頼度が増した）。

## デプロイ

- `combined_rich.joblib`（lgb+cat+28特徴+shrink0.4+learner=cat+base=hybrid）を
  Kaggle Dataset に追加。kernel `MODEL="combined"`（v9）は `add_rich_features` で同特徴を
  再現し `load_combined_model` が rich 版を優先ロード。LB は SUBMISSIONS.md 参照。

## 記録

- MLflow: run `exp009_rich_features`。PF 再計算不要のため数分で完了（v2 キャッシュ流用）。
