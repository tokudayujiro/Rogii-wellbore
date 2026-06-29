# EDA: データ全体像

- train 坑井数: **773**
- 横坑井 行数: 中央値 6576 (min 2058 / max 12141)
- MD ステップ: 中央値 1.000 ft（≒1 ft 刻み）
- GR NaN 率: 中央値 27.704% / 95%点 60.195%

## PS 点（予測対象の難しさ）
- ps_frac（既知区間の割合）: 中央値 0.260（= 坑井の約 26% が既知、残りを予測）
- PS 以降の |TVT−TVT@PS| 最大: 中央値 20.4 ft / 95%点 48.1 ft
  → carry-forward（PS の TVT を定数外挿）の誤差スケールの目安。

## TVT レンジ
- 横坑井 TVT レンジ: 中央値 758.2 ft
- typewell TVT レンジ: 中央値 907.5 ft

## サンプル坑井の可視化
- ![000d7d20](../figures/eda/example_000d7d20.png)
- ![00bbac68](../figures/eda/example_00bbac68.png)
- ![00e12e8b](../figures/eda/example_00e12e8b.png)

![分布](../figures/eda/dataset_distributions.png)

## 所見（次アクション）
- GR は typewell の GR–TVT 関係への相関キー。PS 以降は GR シグネチャ照合が本命。
- carry-forward / 線形外挿をまずベースライン化し、RMSE の基準値を確定する。
- ps_frac が小さい（既知が短い）坑井ほど難しい想定 → CV を坑井単位で層化。