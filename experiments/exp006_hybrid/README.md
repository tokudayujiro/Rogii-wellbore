# exp006: ハイブリッド（PF + beam tracker + per-well セレクタ）

## 目的

参考解法（LB 7.168）の「第2トラッカー（beam search）」と「per-well セレクタ」を非リークで
取り込み、exp004 の単一 PF（LB 8.895）を上回るハイブリッドを作る。

## 構成（`src/rogii/beam.py`, `src/rogii/ensemble.py`）

- **beam tracker**: typewell の GR(TVT) 曲線上をインデックス移動して横坑井 GR に整合
  （Z 非依存）。PF（pos=TVT+Z の慣性追従）と系統が異なり多様性を出す。14 設定平均。
- **selector**: 坑井を `n_eval`（予測点数）と `z_span`（予測区間 Z 幅）でビン分けし、
  ビンごとに「PF scale / beam 混合率 / carry_forward への hold」を切替（参考の定数。
  n_eval/z_span は test でも得られるのでリークなし）。

## 検証（train 150 坑井サブサンプル, RMSE ft）

| モデル | RMSE |
|--------|-----:|
| carry_forward | 12.61 |
| beam ensemble 単体 | 12.38 |
| PF single (scale8, α0.9) | 7.749 |
| **HYBRID (selector)** | **7.715** |

## 学び

- **beam 単体は弱い**（12.38、CF とほぼ同等）。Z 幾何を使わず GR のみで typewell 上を
  辿るため、PF（Z 連成 + 慣性）に大きく劣る。
- **selector が beam を 5–20% しか混ぜない**ため、ハイブリッドの上積みは **+0.03 ft と僅少**。
  PF が支配的で、beam の多様性効果は限定的。
- → 単一 PF からの本質的な伸びには、やはり **ML スタッキング**（参考は PF:stack=0.7:0.3）が
  必要と考えられる。beam/selector だけでは 7.168 には届きにくい。

## 提出

- kernel `MODEL="hybrid"`（v6）。LB スコアは SUBMISSIONS.md 参照。
- CV では PF より僅かに良い（never worse）ため低リスクで採用。

## 次の改善案（exp007 方針）

- **LGB/CatBoost スタッキング**: PF/beam 出力＋幾何・GR 特徴で残差学習し PF と blend。
  学習済みモデルは Kaggle Dataset 経由で kernel に読み込む構成にする。

## 記録

- `src/rogii/beam.py`（beam_search, beam_ensemble）, `src/rogii/ensemble.py`（selector, hybrid_predict）。
- 検証は exp004/005 と同じ 150 坑井サブサンプル（seed=42）。
