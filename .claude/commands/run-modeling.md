---
description: シンプルな予測モデルを実装・評価する
---

予測モデリングのワークフローを以下の方針で実装してください。

引数:
$ARGUMENTS
（推奨フォーマット: `dataset_path=<...> target=<目的変数> task=<タスク説明>`）

`CLAUDE.md`、関連スキル（特に `statistical-ml-review`, `dataframe-polars`, `python-style`, `safe-data-handling`, `path-and-io`, `visualization`）、および `docs/agent/*` を遵守すること。

## 期待される作業

1. 特徴量エンジニアリングのコードを `src/analysis_project/` に作成する。
2. モデリングのコードを `src/analysis_project/` に作成する。
3. 学習・評価を実行するスクリプトを `scripts/` に作成する。
4. 少なくとも 1 つのベースラインモデルと 1 つのシンプルな ML モデルを比較する。
5. train/validation 分割を使う。
6. ターゲットリークを確認する。
7. 評価指標を `outputs/tables/` に保存する。
8. 関連する図を `outputs/figures/` に保存する。
9. 特徴量エンジニアリングのテストを追加または更新する。
10. 可能ならテストと品質チェックを実行する。

## 最後に日本語で報告

- 作成 / 変更したファイル
- 使用した特徴量
- ベースラインの結果
- モデルの結果
- 最良モデル
- 制約・限界
- 実行したコマンド
- 残課題
