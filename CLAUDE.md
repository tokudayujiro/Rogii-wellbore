# CLAUDE.md

このリポジトリは **Python 3.11 のデータサイエンス / 分析プロジェクト** です。
このファイルは Claude Code 用のルーターであり、共通の最重要ルールと、詳細指示への入口を提供します。

> 元は `AGENTS.md`（GitHub Copilot 等の汎用 AI エージェント向け）です。Claude Code では本ファイルが優先されます。両者の内容は同期させてください。

## Hard Rules（常に適用）

- raw データ、認証情報、API キー、トークン、顧客レベルのレコードを**絶対にコミットしない**。
- raw データを直接変更・上書き・削除・再生成**しない**。
- 変更は小さく、レビュー可能な単位にする。
- 自明でない解析判断の前に、前提条件を明示する。
- データの意味が不明瞭なら質問する。
- Python 依存管理は **uv のみ**。pip / conda / poetry / pipenv は禁止。

## How Claude Should Work in This Repo

### スキルの動的読み込み
作業内容に応じて、下記ルーティング表に従い `.claude/skills/<name>/SKILL.md` を読んでから着手すること。
SKILL.md の冒頭 frontmatter（`name` / `description`）を見て、合致するものを選ぶ。

### プロジェクト固有情報
データセット、指標、ワークフロー等の固有知識は `docs/agent/*` を参照。

### スラッシュコマンド
よく使う作業は `.claude/commands/` に定義済み:
- `/plan-analysis` — 分析計画の作成
- `/run-eda` — EDA の実装と実行
- `/run-modeling` — モデリング一連
- `/review-sql` — SQL レビュー
- `/summarize-analysis` — 分析結果の要約
- `/prepare-pr` — PR 説明文の作成
- `/update-agent-docs` — エージェント文書の更新

### 出力言語
- ユーザー向けレポート・要約・PR 説明文・コード内コメントは **日本語**。
- コード識別子・コミットメッセージ件名・コマンド・パスは英語のまま。

## Routing Table（タスク → スキル）

| タスク | スキル |
|--------|--------|
| 依存関係 / テスト / lint / 型チェック / notebook 実行 | [python-project-ops](.claude/skills/python-project-ops/SKILL.md) |
| データファイルの読み書き・移動 | [safe-data-handling](.claude/skills/safe-data-handling/SKILL.md) + [path-and-io](.claude/skills/path-and-io/SKILL.md) |
| SQL の作成 / レビュー | [sql-analysis](.claude/skills/sql-analysis/SKILL.md) |
| Python コードの作成 / レビュー | [python-style](.claude/skills/python-style/SKILL.md) |
| DataFrame 操作 | [dataframe-polars](.claude/skills/dataframe-polars/SKILL.md) |
| グラフ・可視化 | [visualization](.claude/skills/visualization/SKILL.md) |
| Notebook の作成 / 編集 | [notebook-workflow](.claude/skills/notebook-workflow/SKILL.md) |
| 統計 / ML 分析 | [statistical-ml-review](.claude/skills/statistical-ml-review/SKILL.md) |
| 分析結果の要約・報告 | [analysis-reporting](.claude/skills/analysis-reporting/SKILL.md) |
| ファイルパスと I/O | [path-and-io](.claude/skills/path-and-io/SKILL.md) |

## Path-Specific Rules（ファイル種別ごとの自動適用ルール）

| 対象 | ルール |
|------|--------|
| `**/*.py` | python-style / dataframe-polars / path-and-io を遵守 |
| `**/*.sql` | sql-analysis を遵守。`SELECT *` 禁止、CTE 推奨、破壊的 DML はユーザー明示要求時のみ |
| `**/*.ipynb` | notebook-workflow / safe-data-handling を遵守。restartable に保つ |
| `data/**` | `data/raw/` / `data/external/` は不変。`.gitkeep` と markdown 以外コミット禁止 |
| `README.md`, `docs/**/*.md` | 日本語で記述。コード・コマンド・パスは英語のまま |

## Project Context（`docs/agent/`）

| ドキュメント | 目的 |
|--------------|------|
| [project-overview.md](docs/agent/project-overview.md) | プロジェクトの目的とスコープ |
| [repository-structure.md](docs/agent/repository-structure.md) | ディレクトリ構成 |
| [data-catalog.md](docs/agent/data-catalog.md) | データセット一覧と定義 |
| [metrics-and-definitions.md](docs/agent/metrics-and-definitions.md) | 指標定義 |
| [analysis-workflow.md](docs/agent/analysis-workflow.md) | 分析ワークフロー |
| [statistical-and-ml-guidelines.md](docs/agent/statistical-and-ml-guidelines.md) | 統計・ML ガイドライン |
| [validation-and-testing.md](docs/agent/validation-and-testing.md) | テスト・検証方針 |
| [reporting-guidelines.md](docs/agent/reporting-guidelines.md) | 報告テンプレート |
| [security-and-privacy.md](docs/agent/security-and-privacy.md) | セキュリティ・プライバシー |
| [agent-behavior.md](docs/agent/agent-behavior.md) | エージェント行動指針 |

## Common Commands

```bash
uv sync                                              # 依存同期
uv run pytest                                         # テスト
uv run ruff check .                                   # lint
uv run ruff format .                                  # フォーマット
uv run mypy src                                       # 型チェック
uv run python scripts/check_no_raw_data_commit.py     # raw データ混入チェック
uv run python scripts/check_no_sensitive_patterns.py  # 秘密情報チェック
uv run python scripts/validate_agent_docs.py          # エージェント文書検証
bash scripts/run_quality_checks.sh                    # 全チェック一括
```
