# ROGII Wellbore Geology Prediction — 解法 / データサイエンステンプレート

Kaggle「ROGII - Wellbore Geology Prediction」の TVT 予測解法（`src/rogii`, `experiments/`,
`kaggle/`）と、その土台となる分析テンプレートを含むリポジトリ。

> **解法の手法・成績は [kaggle/SUBMISSIONS.md](kaggle/SUBMISSIONS.md) と各 `experiments/expNNN/README.md` を参照。**
> 現行ベスト: 粒子フィルタ + beam + per-well selector + LGB/CatBoost 残差スタッキング（非リーク, CV ~7.9 ft）。

### クレジット / コンプライアンス
- PF / beam / selector の発想はコンペ公開ノートブック（"ROGII SUPER SOLUTION", romantamrazov 他）を**参考**にし、**非リークで独自に再実装**。第三者ノート自体は再配布しない（repo 非収録）。
- 過去に test=train リークのプローブ提出を 1 回だけ実施（ユーザー許可・効果なし・撤回済み）。詳細と方針は [kaggle/SUBMISSIONS.md](kaggle/SUBMISSIONS.md) のコンプライアンス注記を参照。生データ・認証情報は非コミット。

---

**Claude Code** および GitHub Copilot / Copilot Agent Mode / Copilot Cloud Agent と連携し、安全かつ一貫したデータ分析作業を行うためのリポジトリテンプレートです。

> 📘 **各スキルが何をしているかの日本語解説は [docs/skills-overview.ja.md](docs/skills-overview.ja.md) を参照してください。**

## Claude Code で使う

このリポジトリには Claude Code 向けの設定が `.claude/` 配下に同梱されています。

- `CLAUDE.md` — Claude 用ルーター（`AGENTS.md` の Claude 向け版）
- `.claude/skills/` — 10 個のスキル（タスクに応じて Claude が動的に読み込む）
- `.claude/commands/` — 7 個のスラッシュコマンド（`/plan-analysis`, `/run-eda`, `/run-modeling`, `/review-sql`, `/summarize-analysis`, `/prepare-pr`, `/update-agent-docs`）

```bash
# プロジェクトディレクトリで Claude Code を起動するだけで自動認識される
claude
```

主な使い方:

```text
# 分析計画を作りたい
/plan-analysis 売上の月次トレンドを分析したい

# EDA を実装・実行
/run-eda dataset_path=data/raw/titanic/train.csv topic=生存率の要因分析

# SQL レビュー
/review-sql queries/monthly_active_users.sql

# PR 説明文を作る
/prepare-pr
```



## セットアップ

### 前提条件

- Python 3.11
- [uv](https://docs.astral.sh/uv/) がインストール済みであること

### インストール

```bash
uv sync
```

### 主要コマンド

```bash
uv sync                                              # 依存関係のインストール・同期
uv run pytest                                         # テスト実行
uv run ruff check .                                   # リント
uv run ruff format .                                  # フォーマット
uv run mypy src                                       # 型チェック
uv run python scripts/check_no_raw_data_commit.py     # rawデータのコミットチェック
uv run python scripts/check_no_sensitive_patterns.py   # 秘密情報パターンの検出
uv run python scripts/validate_agent_docs.py           # エージェント文書の検証
bash scripts/run_quality_checks.sh                     # 全品質チェック一括実行
```

## ディレクトリ構成

```
.
├── AGENTS.md                          # 汎用エージェント用ルーター（Copilot 等）
├── CLAUDE.md                          # Claude Code 用ルーター
├── .claude/
│   ├── skills/                        # Claude Skills（10個）
│   └── commands/                      # スラッシュコマンド（7個）
├── .github/
│   ├── copilot-instructions.md        # Copilot共通指示（薄いファイル）
│   ├── workflows/ci.yml               # GitHub Actions CI
│   ├── instructions/                  # パス別補助指示
│   ├── prompts/                       # 再利用プロンプト
│   └── skills/                        # 作業別スキル
├── docs/
│   ├── agent/                         # プロジェクト固有ドキュメント
│   └── skills-overview.ja.md          # スキル一覧（日本語解説）
├── data/
│   ├── raw/                           # 元データ（不変・gitignore対象）
│   ├── external/                      # 外部データ（不変・gitignore対象）
│   ├── interim/                       # 中間加工データ
│   └── processed/                     # 最終加工データ
├── notebooks/                         # 分析用Notebook
├── outputs/
│   ├── figures/                       # グラフ・図
│   ├── tables/                        # 集計テーブル
│   └── reports/                       # レポート
├── scripts/                           # CI・検証スクリプト
├── src/analysis_project/              # 再利用可能なPythonモジュール
└── tests/                             # テスト
```

## Copilot指示体系の設計

### AGENTS.md はルーター、skills は作業別手順、docs/agent はプロジェクト固有知識

このリポジトリでは、エージェント向けの指示を3層に分割しています：

1. **`.github/copilot-instructions.md`** — 薄い共通指示。全タスクで必要な最小限のルールと、詳細な指示への案内。
2. **`AGENTS.md`** — エージェント用ルーター。タスクの種類に応じて適切なskillファイルへ誘導する。
3. **`.github/skills/*/SKILL.md`** — 作業別の詳細手順。Python、SQL、データ処理、可視化など。
4. **`docs/agent/*`** — プロジェクト固有の知識。データカタログ、指標定義、分析ワークフローなど。
5. **`.github/instructions/*.instructions.md`** — パス別の補助指示。ファイルの種類に応じた自動適用ルール。
6. **`.github/prompts/*.prompt.md`** — 再利用可能なプロンプト。分析計画、SQLレビュー、レポート作成など。

この設計により、全部入りの巨大な指示ファイルを避け、トークン効率よく必要な情報だけを参照できます。

### 利用可能なスキル

各スキルの**詳しい挙動・ルール・コード例**は [docs/skills-overview.ja.md](docs/skills-overview.ja.md) に日本語でまとめてあります。

| スキル | ひとことで言うと | 詳細 |
|--------|------------------|------|
| `python-project-ops` | `uv` で依存・テスト・lint を回す手順 | [→](docs/skills-overview.ja.md#1-python-project-ops) |
| `safe-data-handling` | raw データ・個人情報の事故防止 | [→](docs/skills-overview.ja.md#2-safe-data-handling) |
| `sql-analysis` | SQL の書き方とレビュー観点 | [→](docs/skills-overview.ja.md#3-sql-analysis) |
| `python-style` | 型ヒント・docstring・日本語コメント | [→](docs/skills-overview.ja.md#4-python-style) |
| `dataframe-polars` | Polars 優先の DataFrame 操作 | [→](docs/skills-overview.ja.md#5-dataframe-polars) |
| `visualization` | matplotlib / seaborn の品質ルール | [→](docs/skills-overview.ja.md#6-visualization) |
| `path-and-io` | パス操作と I/O の作法 | [→](docs/skills-overview.ja.md#7-path-and-io) |
| `notebook-workflow` | Notebook の構造と運用 | [→](docs/skills-overview.ja.md#8-notebook-workflow) |
| `statistical-ml-review` | 統計・ML 分析の前提整理 | [→](docs/skills-overview.ja.md#9-statistical-ml-review) |
| `analysis-reporting` | 分析結果の報告テンプレ | [→](docs/skills-overview.ja.md#10-analysis-reporting) |

スキル本体（Claude 用 / Copilot 用）:
- Claude Code: `.claude/skills/<name>/SKILL.md`
- GitHub Copilot: `.github/skills/<name>/SKILL.md`

### 利用可能なプロンプト / スラッシュコマンド

| 名称 | 用途 | Claude Code |
|------|------|-------------|
| `plan-analysis` | 分析計画の作成 | `/plan-analysis` |
| `run-eda` | EDA の実装・実行 | `/run-eda` |
| `run-modeling` | モデリングの実装・評価 | `/run-modeling` |
| `review-sql` | SQL のレビュー | `/review-sql` |
| `summarize-analysis` | 分析結果の要約 | `/summarize-analysis` |
| `prepare-pr` | PR 説明文の作成 | `/prepare-pr` |
| `update-agent-docs` | エージェント文書の更新 | `/update-agent-docs` |

実体: Claude 用は `.claude/commands/<name>.md`、Copilot 用は `.github/prompts/<name>.prompt.md`。

## データの安全性ルール

- `data/raw/` と `data/external/` は不変として扱い、直接変更しない。
- rawデータ、認証情報、APIキー、トークン、顧客レベルのレコードをコミットしない。
- 加工データは `data/interim/` や `data/processed/` に出力する。
- 図表やレポートは `outputs/` に出力する。
- `.env` ファイルは `.gitignore` でコミット対象外。

## パッケージ管理

- **uv のみを使用**する。pip、conda、poetry は使わない。
- 依存関係の追加: `uv add <package>`
- 開発依存の追加: `uv add --group dev <package>`
