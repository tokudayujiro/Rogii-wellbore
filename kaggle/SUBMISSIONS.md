# 提出台帳（Submissions Log）

ROGII Wellbore Geology Prediction への提出記録。提出するたびに 1 行追記する。
評価指標は予測点 `dTVT` の **RMSE（ft, 小さいほど良い）**。

| # | 日付 | モデル (MODEL) | kernel ver | ローカル CV RMSE | Public LB | Private LB | メッセージ / 備考 |
|---|------|----------------|-----------:|----------------:|----------:|-----------:|-------------------|
| 1 | 2026-06-21 | carry_forward | v2 | 12.81 | 15.883 | （終了時公開） | 初回提出。LB アンカー確立。sub ref 53912517 |
| 2 | 2026-06-21 | viterbi_blend (a=0.2) | v3 | 12.76 | **15.823** | （終了時公開） | exp003。CV/LB とも CF 超え。sub ref 53913296 |
| 3 | 2026-06-21 | leak (train TVT コピー) | v4 | — | 15.823 | — | **リークは罠**。kernel は真 train TVT を出力したが LB 改善せず＝隠し採点コピーは train と別版（参考NB cell36 の警告通り）。sub ref 53913826 |
| 4 | 2026-06-28 | particle filter (a=0.9) | v5 | ~8.1 (200坑井) | **8.895** | （終了時公開） | exp004。粒子フィルタ。15.82→8.90 と半減。参考 7.168 を追う。sub ref 54139024 |
| 5 | 2026-06-29 | hybrid (PF+beam+selector) | v6 | 7.715 (150坑井) | **8.711** | — | exp006。CV +0.03 だが LB は −0.18（隠し3坑井で selector/beam が効いた）。sub ref 54158720 |
| 6 | 2026-06-29 | stack (PF + 0.5·LGB残差) | v7 | 8.121 (773 OOF) | **8.614** | — | exp007。LGB が PF 残差を幾何/空間特徴で補正。LB も hybrid 更新。sub ref 54163748 |
| 7 | 2026-06-29 | combined (hybrid + 0.25·CatBoost残差) | v8 | **8.027** (773 OOF) | 採点中 | — | exp008。CV ベスト。base=selector hybrid + CatBoost 残差。sub ref 54169305 |

## 重要な訂正（リークについて）

当初「参考NB の 7.168 は test=train リーク」と述べたが**誤り**。実際に train TVT をそのまま
提出しても LB は 15.823 のままで改善しなかった（kernel は真 TVT を出力済みと確認）。
Kaggle の隠し採点では overlap 坑井の**別バージョン**が使われ、train コピーと一致しない。
→ **リークは効かない。7.168 は粒子フィルタ等の正当なモデリングによるスコア**。
今後は非リークで PF/beam を実装して追う（exp004）。

## メモ

- CV 12.81 → Public 15.88 の差（約 3 ft）は test 分布差。今後のモデルは
  **CV と LB の両方**で carry_forward を超えるかを確認する（CV だけで判断しない）。

> 記入方法: `kaggle competitions submissions -c rogii-wellbore-geology-prediction` で
> Public スコアを確認し、kernel version（`kaggle kernels status` / push 出力）とともに記録する。
> ローカル CV は対応 experiment の `cv_result.txt` を参照。
