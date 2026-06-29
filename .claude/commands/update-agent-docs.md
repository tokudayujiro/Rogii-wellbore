---
description: 規約変更時に CLAUDE.md / skills / docs/agent を更新する
---

リポジトリの規約が変わった際、関連するエージェント文書を更新してください。

追加指定:
$ARGUMENTS

## 手順

1. `CLAUDE.md` のルーティング表が最新か確認する。
2. `AGENTS.md`（GitHub Copilot 等で使用）も同期させる必要があるか確認する。
3. `.claude/skills/*/SKILL.md` の更新要否を確認する。
4. `docs/agent/*.md` の更新要否を確認する。

## ルール

- `CLAUDE.md` / `AGENTS.md` はルーターとして**薄く**保つ。
- 詳細ルールを複数ファイルに重複させない。
- コピーするのではなく、スキルファイルへのリンクで参照する。
- 変更後に `uv run python scripts/validate_agent_docs.py` を実行する。
