# 指標定義

## 公式評価指標: RMSE of dTVT

| 項目 | 内容 |
|------|------|
| 定義 | 予測対象点（PS 点以降）について `dTVT = manualTVT − predictedTVT` を計算し、その RMSE を取る |
| 数式 | `RMSE = sqrt( mean( (TVT_true − TVT_pred)^2 ) )`（対象は全 test 坑井の PS 以降の全点をまとめて） |
| 単位 | feet |
| 対象 | submission の各 `id`（= 各坑井 PS 点以降の 1 ft 刻みの点）|
| 方向 | 小さいほど良い |

## ローカル検証（CV）方針

test が 3 坑井しかないため、**train 坑井で PS 点を人工的に設定して**本番と同じ設定を再現する。

1. 各 train 坑井について、`TVT_input` の非 NaN 範囲（実際の PS）を参考に、PS 点を決める（本番同様に前半を既知、後半を予測対象に）。
2. PS 以降の予測に対し上記 RMSE を計算。
3. 坑井単位の GroupKFold（同一坑井がtrain/validに跨らない）で汎化性能を見る。
4. 近傍坑井情報を使う場合は、空間的リークに注意（valid 坑井の近傍に train 坑井が入ること自体は本番再現として許容、ただし valid 坑井の PS 以降 TVT は使わない）。

## ベースライン指標（比較対象）

- **Carry-forward**: PS 点の TVT をそのまま定数で外挿（地質が flat の仮定）。最弱ベースライン。
- **Linear extrapolation**: PS 直前の TVT 勾配（dTVT/dMD）を線形外挿。
- **GR correlation**: 横坑井 GR を typewell GR–TVT にマッチングして TVT を割り当て（本命）。

各実験で上記ベースライン比の改善を `outputs/reports/` に記録する。
