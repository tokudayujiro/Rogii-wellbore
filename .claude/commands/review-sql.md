---
description: SQL を正確性・安全性の観点でレビューする
---

`.claude/skills/sql-analysis/SKILL.md` のチェックリストに沿って、提供された SQL をレビューしてください。

対象 SQL（パス or 直接貼り付け）:
$ARGUMENTS

## 確認項目

- `SELECT *` の使用（明示的なカラム指定にすべき）
- 大きなテーブルでの日付フィルタ欠如
- Join のカーディナリティ問題 (1:1, 1:N, M:N)
- Join の前後での行数検証
- 破壊的ステートメント (DROP, TRUNCATE, DELETE, UPDATE)
- 指標定義の曖昧さ
- NULL の取り扱い
- 重複リスク
- 暗黙的な CROSS JOIN

## フィードバック形式

行番号への具体的な参照と修正案を **日本語**で提示してください。
