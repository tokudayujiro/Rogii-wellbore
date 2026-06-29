# データカタログ

すべて `data/raw/` 配下（不変・コミット禁止）。単位は feet。

## ファイル構成

| パス | 内容 |
|------|------|
| `train/<wellid>__horizontal_well.csv` | 横坑井データ（773 坑井）。学習用は TVT・地質サーフェス列あり |
| `train/<wellid>__typewell.csv` | 対応する縦坑井データ（773 坑井） |
| `train/<wellid>.png` | TVT プロットの参照画像（773 枚、EDA 用） |
| `test/<wellid>__horizontal_well.csv` | 横坑井データ（3 坑井）。TVT・地質サーフェス列は除去済み |
| `test/<wellid>__typewell.csv` | 対応する縦坑井データ（3 坑井）。Geology 列なし |
| `sample_submission.csv` | 提出フォーマット（`id`, `tvt`）。14,151 行 |
| `AI_wellbore_geology_prediction_task_en.pptx` | 公式タスク説明（28 MB） |

## horizontal_well.csv の列

| 列 | 説明 | train | test |
|----|------|:----:|:----:|
| `MD` | 測長深度（坑井長）。昇順 | ✓ | ✓ |
| `X`, `Y`, `Z` | 各点の座標 | ✓ | ✓ |
| `GR` | ガンマ線（NaN を含みうる） | ✓ | ✓ |
| `TVT_input` | PS 点までの既知 TVT（PS 以降は NaN） | ✓ | ✓ |
| `TVT` | **予測対象**。坑井の地質（真垂直深度） | ✓ | ✗ |
| `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` | 各地質層の上端深度（地質サーフェス） | ✓ | ✗ |

## typewell.csv の列

| 列 | 説明 | train | test |
|----|------|:----:|:----:|
| `TVT` | 真垂直深度（縦坑井の深度）。全区間既知 | ✓ | ✓ |
| `GR` | 各点のガンマ線 | ✓ | ✓ |
| `Geology` | 地質層名（train でも多くが空） | ✓ | ✗ |

## 予測対象と id 規約

- `id = <wellid>_<行インデックス>`。行インデックスは horizontal_well.csv の **0 始まり**行番号。
- 予測対象は **PS 点以降**の行。PS 点 = `TVT_input` が非 NaN の最終行の次。
- 例: `000d7d20` は 5,278 行、`TVT_input` 非 NaN が先頭 1,442 行 → 予測対象は index 1442〜5277（3,836 点）。

## ⚠️ 既知のリーク（2026-06-14 発見）

- test の 3 坑井（`000d7d20`, `00bbac68`, `00e12e8b`）は **train にも同 id で存在**し、train 側は `TVT` が全行埋まっている。
- MD・GR は train/test で完全一致（test は `TVT` と地質サーフェス列を除去しただけ）。
- 従って `train/<id>__horizontal_well.csv` の `TVT` をそのまま提出すれば理論上 RMSE=0。
- **方針**: このリークは exploit せず、GR 相関ベースの汎化モデルを本筋として開発する（記録のみ）。
  CV は train 坑井を GroupKFold し、各坑井の実 PS を使って本番設定を再現する。

## 注意（Hard Rules 関連）

- `data/raw/` は不変。前処理結果は `data/interim/` `data/processed/` へ書き出す。
- raw データ・画像・大きな CSV はコミットしない（`.gitkeep` と md のみ）。
