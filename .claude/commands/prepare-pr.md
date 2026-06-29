---
description: 現在の変更内容をもとに PR 説明文を作成する
---

現在の変更内容に対する Pull Request の説明文を作成してください。

追加指定:
$ARGUMENTS

## 含める項目

1. **変更内容**: 何が変わったか / 修正されたファイル
2. **変更理由**: なぜこの変更を行ったか
3. **検証コマンド**:
   ```bash
   uv run pytest
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src
   ```
4. **リスク**: 想定される副作用やリスク
5. **影響ファイル**: 触れたファイルの一覧

## 形式

**日本語**で記述、簡潔にレビュアーフレンドリーに。
