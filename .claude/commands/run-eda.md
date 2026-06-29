---
description: データセットに対する EDA を実装して実行する
---

EDA（探索的データ分析）を以下の方針で実装してください。

引数:
$ARGUMENTS
（推奨フォーマット: `dataset_path=<入力データのパス> topic=<分析テーマ>`）

`CLAUDE.md`、関連スキル（特に `safe-data-handling`, `path-and-io`, `dataframe-polars`, `visualization`, `notebook-workflow`, `python-style`）、および `docs/agent/*` を遵守すること。

## 期待される作業

1. raw データを**不変の入力**として読み込む。
2. 再利用可能な EDA コードを `src/analysis_project/` に作成する。
3. EDA を実行するスクリプトを `scripts/` に作成する。
4. サマリーテーブルを `outputs/tables/` に保存する。
5. 図を `outputs/figures/` に保存する。
6. データ取り扱い / パス / Python スタイル / DataFrame 操作 / 可視化のリポジトリ規約に従う。
7. 可能なら EDA スクリプトと品質チェックを実行する。

## 最後に日本語で報告

- 作成 / 変更したファイル
- 実行したコマンド
- 生成された成果物
- 主な発見
- 残課題
