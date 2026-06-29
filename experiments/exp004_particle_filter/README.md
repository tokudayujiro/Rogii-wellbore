# exp004: 尤度重み付き粒子フィルタ（PF）による TVT トラッキング

## 目的

参考解法（LB 7.168）の中核＝**粒子フィルタによる geosteering トラッキング**を、非リークで
再実装する（`src/rogii/pf.py`）。exp003 の Viterbi はフラット仮定からの微修正に留まったが、
PF は**地層 dip の傾きトレンドを慣性付きで追従**し GR 尤度で補正するため、桁違いに改善する。

## 手法（`src/rogii/pf.py`）

状態を **`pos = TVT + Z`**（地層マーカーの絶対位置）とし、各粒子が `pos` と `rate`（dip 速度）を持つ。

- **初期化**: `pos₀ = TVT[PS-1] + Z[PS-1]`、`rate₀` = 既知末尾 30 点の `(ΔTVT+ΔZ)/ΔMD` 中央値
  （PS 直前の実 dip）。発光ノイズ `σ` = 既知区間の `GR_h − GR_tw(TVT)` の std（[10,60]）。
- **遷移**（MD ステップ dm ごと）: `rate ← MOM·rate + VN·ε`、`pos ← pos + rate·dm + PN·ε`。
- **尤度**: `exp(−0.5·((GR_h − GR_typewell(pos−Z))/σ)²)` で粒子を重み付け、neff 低下時に
  systematic リサンプル（roughening 付き）。
- **多シード合成**: 同じ PF を `n_seeds` 回（異なる乱数）回し、各シードの総対数尤度で
  `softmax(loglik / scale)` 合成。GR と最も整合するシードを選びドリフトに強くする。

ハイパラ（参考準拠）: `MOM=0.998, VN=0.002, PN=0.005, resamp=0.5, init_spr=4.5, scale=8`。

## 検証

PF は坑井ごと独立・学習なし（決定的）なので fold 不要。**train から 200 坑井をサブサンプル**
して平均 RMSE を評価（速度のため `n_seeds=64, n_particles=300`。最終提出は 128/500）。
並列は joblib threading（numba `nogil`）。

## 結果（200 坑井サブサンプル, RMSE ft）

| モデル | RMSE | 備考 |
|--------|-----:|------|
| carry_forward | 13.61 | 基準 |
| **PF (scale=8)** | **8.36** | CF 比 −38%。exp003 viterbi(12.76) を大幅更新 |
| PF blend (α=0.9) | **8.11** | CF へ 1 割シュリンクで微改善 |

速度: 2.70 s/well（並列 ×4, 64seed/300part）。

## 学び

- **PF は exp001–003 を桁違いに上回る**（13.6 → 8.1）。鍵は「dip rate を慣性で追従」＋
  「GR 尤度補正」＋「多シード尤度重み付け」。フラット/微修正の発想から脱却できた。
- 参考解法の **7.168 はリークでなく正当な PF スコア**だった（SUBMISSIONS.md の訂正参照）。
  単一 PF で 8.1 まで来ており、7.168 との差は ensemble/beam/stack/selector の上積み分。

## 次の改善案（exp005 方針）

- **scale の per-well 選択**（参考の selector）。坑井特性（n_eval, z_span）で温度を切替。
- **beam search アンサンブル**（参考 cell 5）と PF の blend。
- 複数 PF 変種（ANCC-anchored / Z-velocity-coupled）の平均。
- PF 出力を特徴に **LGB/CatBoost スタッキング**（参考の最終段）。

## 記録

- MLflow: experiment `rogii-wellbore` / run `exp004_particle_filter`。
- 再現: `uv run python experiments/exp004_particle_filter/train.py`
- 提出 kernel: `kaggle/submission/rogii_submit.py`（`MODEL="pf"`、`PF_ALPHA=0.9`）。
- 依存追加: `numba`（uv 管理。Kaggle 標準イメージにも存在）。
