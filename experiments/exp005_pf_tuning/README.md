# exp005: PF の後処理・チューニング探索（単一PFの天井確認）

## 目的

exp004 の PF（LB 8.895）に対し、追加計算を増やさず詰められる要素を探索:
**scale 選択 / savgol 平滑化 / CF blend(alpha) / GR キャリブレーション**。

## 方法

`pf_allseeds` で 1 回 PF を回した結果（per-seed preds+liks）から複数 scale を合成し、
savgol・alpha のグリッドを評価（train サブサンプル, 並列）。calibration は別途 A/B。

## 結果（train サブサンプル, RMSE ft）

| 項目 | 結論 |
|------|------|
| **scale** | **8 が最良**（3/5/12 は劣る）。exp004 と一致 |
| **alpha (CF blend)** | **0.9 が最良**。exp004 と一致 |
| **savgol 平滑化** | ほぼ無効（7.7459 vs 無平滑 7.7489、差 ~0.003 = ノイズ）→ 不採用 |
| **GR calibration** | **大幅悪化**（120坑井: 8.37 → 16.61）→ 不採用 |

best: `scale=8, sg=61, alpha=0.9` = 7.746（150坑井）。savgol の寄与は誤差範囲。

## 学び

- **単一 PF は exp004 の設定（scale=8, alpha=0.9）でほぼ天井**。後処理では伸びない。
- **GR calibration は逆効果**: 既知区間（狭い TVT 範囲）で fit した線形補正が予測区間へ
  外挿されて GR–TVT 関係を歪める。生 GR 比較（+ σ 推定）が正しい。
- → 7.168 に届くには**別トラッカー（beam search）との blend**や **ML スタッキング**など、
  単一 PF の外側の多コンポーネント化が必要（参考解法の構成）。

## 次の改善案（exp006 方針）

- **beam search トラッカー**（参考 cell 5）を実装し PF と blend（多様性で底上げ）。
- PF 出力＋幾何/GR 特徴で **LGB/CatBoost スタッキング**（参考は PF:stack = 0.7:0.3）。
- per-well の scale 選択（参考の selector）。

## 記録

- MLflow: run `exp005_pf_tuning`。再現: `uv run python experiments/exp005_pf_tuning/train.py`
- `src/rogii/pf.py` に `pf_allseeds` / `combine_scale` / `calibrate` 引数を追加。
- 提出 kernel は exp004 設定（scale=8, alpha=0.9, calibrate=False）を維持。
