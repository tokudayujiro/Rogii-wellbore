# プロジェクト概要

## 目的

Kaggle コンペ **ROGII - Wellbore Geology Prediction** の解法開発。

横坑井（horizontal well）の各測定点（1 ft 刻み）について、地質の真垂直深度 **TVT (True Vertical Thickness)** を予測する。TVT は「坑井がいまどの地層を貫いているか」を表す地質量であり、地質ステアリング（geosteering）の中核指標。

## タスクの本質

- 各横坑井には 1 本の縦坑井（**typewell**）が対応づけられている。
- typewell では TVT と GR（ガンマ線）の対応関係（= その地域の地層の標準応答）が全区間で既知。
- 横坑井では GR・XYZ・MD が全区間で取得できるが、TVT は **Prediction Start (PS) 点まで**しか分からない（`TVT_input` 列として提供）。
- **PS 点以降の TVT を、横坑井の GR シグネチャを typewell の GR–TVT 関係に相関させて推定する。**
- 直感: 横坑井の GR 波形が typewell のどの深度（TVT）の GR 波形に一致するかを探すと、その点の TVT が分かる。坑井が地層を上昇/下降すると GR シグネチャが typewell 上を移動する（pptx slide 5–9）。

## 補助情報

- 近傍坑井（offset well）の地質傾斜（dip）は似る傾向 → XYZ・方位を使った空間的な事前情報が効く（slide 12–13）。
- 横坑井の GR は typewell の GR より高分解能。PS 前の高分解能 GR と既知 TVT を相関の足がかりに使うと良い（slide 9）。

## データ規模

- train: 773 坑井（各 horizontal_well.csv + typewell.csv + 参照用 .png）
- test: 3 坑井（horizontal_well.csv + typewell.csv、TVT と地質サーフェス列は除去済み）
- submission: 14,151 行（test 3 坑井の PS 点以降の各点）

## 評価

予測対象点の `dTVT = manualTVT − predictedTVT` の **RMSE**（単位: feet）。詳細は [metrics-and-definitions.md](metrics-and-definitions.md)。

## スコープ外

- リアルタイム/オンライン geosteering 運用は対象外（バッチ予測のみ）。
- 物理シミュレーションによる地層モデリングは対象外（データ駆動で解く）。

## 重要な前提

- 単位はすべて feet。横坑井は MD（測長深度）昇順に並ぶ。
- `id = <wellid>_<行インデックス>`（horizontal_well.csv の 0 始まり行番号）。
- PS 点 = `TVT_input` が非 NaN である最終行の次の行。それ以降が予測対象。
