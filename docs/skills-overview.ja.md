# スキル一覧（日本語解説）

このドキュメントは、本リポジトリに収録されている 10 個の Claude / Copilot 用スキルが
**何をしているか・いつ起動するか** を日本語で俯瞰するためのものです。

スキルの本体は以下の 2 箇所に同じ内容が置かれています:

- `.claude/skills/<name>/SKILL.md` — Claude Code 用
- `.github/skills/<name>/SKILL.md` — GitHub Copilot 用（原本）

---

## クイックリファレンス

| # | スキル名 | ひとことで言うと | いつ起動するか |
|---|----------|------------------|----------------|
| 1 | [python-project-ops](#1-python-project-ops) | `uv` で依存・テスト・lint を回す手順 | 依存追加 / pytest / ruff / mypy / notebook 実行 |
| 2 | [safe-data-handling](#2-safe-data-handling) | データ取り扱いの安全ガード | data/ 配下を読む・書く・移動するとき |
| 3 | [sql-analysis](#3-sql-analysis) | SQL の書き方とレビュー観点 | SELECT / JOIN / 集計 / 検証クエリ |
| 4 | [python-style](#4-python-style) | 型ヒント・docstring・コメント規約 | Python ファイルを書く・直すとき |
| 5 | [dataframe-polars](#5-dataframe-polars) | Polars 優先の DataFrame 操作 | DataFrame の読込・filter / join / 集計 |
| 6 | [visualization](#6-visualization) | matplotlib / seaborn の品質ルール | 図表の作成・保存 |
| 7 | [path-and-io](#7-path-and-io) | パス操作と I/O の作法 | ファイル読み書き・パス組み立て |
| 8 | [notebook-workflow](#8-notebook-workflow) | Notebook の構造と運用 | `.ipynb` 作成・編集・実行 |
| 9 | [statistical-ml-review](#9-statistical-ml-review) | 統計・ML 分析の前提整理 | 仮説検定 / A/B / 学習・評価 |
| 10 | [analysis-reporting](#10-analysis-reporting) | 分析結果の報告テンプレ | 結論・事実・解釈をまとめる |

---

## 1. python-project-ops

**目的:** Python プロジェクトの運用コマンドを統一する。

**主要ルール:**
- 依存管理は **`uv` のみ**。pip / conda / poetry / pipenv は禁止。
- `requirements.txt` を手で作らない・編集しない。
- 依存追加は `uv add <pkg>`、開発依存は `uv add --group dev <pkg>`。
- 変更後は `pyproject.toml` と `uv.lock` の diff を確認。

**よく使うコマンド:**
```bash
uv sync                                            # 依存同期
uv run pytest                                       # テスト
uv run ruff check . / uv run ruff format .          # lint / format
uv run mypy src                                     # 型チェック
uv run papermill in.ipynb out.ipynb                 # notebook 実行
bash scripts/run_quality_checks.sh                  # 全部入り
```

---

## 2. safe-data-handling

**目的:** raw データ / 個人情報の事故を防ぐ最重要ガード。

**ハードルール:**
- raw データ・認証情報・APIキー・トークン・顧客レコードを**絶対にコミットしない**。
- `data/raw/` と `data/external/` は **immutable**（直接変更・削除・再生成しない）。
- 派生データは `data/interim/` / `data/processed/` / `outputs/` へ書き出す。

**ディレクトリの役割:**

| ディレクトリ | 役割 | 変更可? |
|--------------|------|---------|
| `data/raw/` | オリジナルのソースデータ | × |
| `data/external/` | 外部参照データ | × |
| `data/interim/` | 中間変換 | ○ |
| `data/processed/` | 最終加工データ | ○ |
| `outputs/` | 図表・テーブル・レポート | ○ |

---

## 3. sql-analysis

**目的:** SQL の品質と安全性を担保する。

**主要ルール:**
- **明示的なカラム指定**（`SELECT *` は探索時のみ）。
- **CTE** を使って読みやすく分割する。
- 大きなファクトテーブルには**日付フィルタ**を必ず入れる。
- **JOIN 前後で行数を検証**してファンアウト / データロスを検知。
- 暗黙的な CROSS JOIN を避ける。
- `DROP` / `TRUNCATE` / `DELETE` / `UPDATE` は**ユーザーが明示的に依頼した時のみ**。

**レビューチェックリスト（抜粋）:**
- 行の粒度（grain）は明確か
- 日付範囲は明示的か
- NULL の取り扱いは決めてあるか
- 重複の可能性を検討したか
- JOIN のカーディナリティを検証したか

---

## 4. python-style

**目的:** Python コードの一貫したスタイルを保つ。

**主要ルール:**
- **型ヒント**を公開関数の signature に必ず付ける。
- **Google スタイルの docstring**（Purpose / Args / Returns / Raises / Examples / Assumptions）。
- **インラインコメントは日本語で書く**。
- `pathlib.Path` を使う（[path-and-io](#7-path-and-io) 参照）。
- `ruff` 設定（pyproject.toml）に従う。

---

## 5. dataframe-polars

**目的:** DataFrame 操作の第一選択を統一する。

**主要ルール:**
- **`polars` を優先**。`pandas` は外部依存などで仕方なく使う場合のみ最小限。
- 可能なら **LazyFrame**（`scan_parquet` / `scan_csv`）で遅延実行。
- JOIN 前後で**行数を確認**し、ファンアウトを検知する。
- 変換は再現可能・スクリプタブルに保つ。

**典型コード:**
```python
import polars as pl

lf = pl.scan_parquet("data/raw/events.parquet")
result = (
    lf.filter(pl.col("event_date") >= "2024-01-01")
    .group_by("user_id")
    .agg(pl.col("event_type").count().alias("event_count"))
    .collect()
)
```

---

## 6. visualization

**目的:** ぐちゃぐちゃ・誤読を招くグラフを作らせない。

**主要ルール:**
- 冒頭で `sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)` を 1 回呼ぶ。
- 日本語を含むなら `japanize_matplotlib` を import（または CJK フォントを指定）。
- `plt.figure()` のステートフル API は禁止。**`fig, ax = plt.subplots()`** で明示作成。
- `constrained_layout=True` を優先（`tight_layout()` より）。
- カラーマップ: カテゴリは `muted` / `Set2` / `colorblind`、連続値は `viridis` / `cividis` / `mako`、発散は `coolwarm` / `RdBu` / `vlag`。**`jet` と `rainbow` は禁止**。
- 棒グラフは `ax.set_ylim(bottom=0)`、線・散布図は強制的に 0 起点にしない。
- 保存先は `outputs/figures/`、`dpi=150`（印刷は 300）、保存後に `plt.close(fig)`。

---

## 7. path-and-io

**目的:** パス操作のばらつきと事故を防ぐ。

**主要ルール:**
- すべて **`pathlib.Path`**。絶対パスをハードコードしない。
- リポジトリルートからの相対 or 設定済みディレクトリを優先。
- `src/analysis_project/paths.py` のユーティリティを使う:
  - `get_repo_root()` — リポジトリルート
  - `data_dir()` — `data/`
  - `outputs_dir()` — `outputs/`
  - `ensure_parent_dir(path)` — 親ディレクトリを作って path を返す
- 出力先が `data/raw/` / `data/external/` 配下になっていないか確認。
- 出力ファイル名には日付や run ID を含める。既存出力を勝手に上書きしない。

---

## 8. notebook-workflow

**目的:** Notebook を「読み物」と「再現可能な処理」両立で運用する。

**主要ルール:**
- Notebook は**探索とコミュニケーション**用。再利用ロジックは `src/analysis_project/` に切り出す。
- **Restart & Run All で動くこと**を保つ（隠れた状態に依存しない）。
- 秘密情報・顧客データを出力に残さない。
- 命名は `NNN_short_description.ipynb`（例: `001_data_exploration.ipynb`）。
- 自動実行は **papermill** を使う。

**標準構造:**
1. ヘッダー（タイトル / 作成者 / 作成日 / 目的）
2. import まとめ
3. 設定（パス / 定数 / パラメータ）
4. 分析セル
5. まとめ（発見と次のアクション）

---

## 9. statistical-ml-review

**目的:** 統計 / ML 作業の前提を明示し、リーク・誤解釈を防ぐ。

**分析前に明文化すべき項目:**
- 目的変数 / 分析単位 / 期間 / サンプル定義 / 除外条件
- 仮定 / リーク（leakage）リスク / 評価指標 / ベースライン

**実験・因果:**
- **相関と因果を明確に区別**する。
- 信頼区間・不確実性を併記する。
- データが支持する範囲を超えて主張しない。

**ML:**
- train / validation / test の明確な分離。
- 未来情報やターゲットリークの確認。
- 前処理（encoding / scaling / imputation / FE）を文書化。
- まず**シンプルなベースライン**と比較する。
- 制約・限界（データ品質・サンプル偏り・汎化）を必ず記述。

---

## 10. analysis-reporting

**目的:** 分析結果の報告フォーマットを統一する。

**言語:** 原則**日本語**。

**標準構造:**
1. **結論** — 最初に主要な発見を置く
2. **事実** — データから読み取れる客観事実
3. **仮定** — 分析中に置いた前提
4. **解釈** — 含意と示唆
5. **制約・注意点** — 制約・バイアス・注意

**必須コンテキスト:** データ期間 / フィルタ条件 / サンプルサイズ / 指標定義

**再現性メモ:** 入力データパス / クエリ・スクリプトパス / 出力パス / 実行コマンド

---

## スキル間の関係

```
[Hard Rules @ CLAUDE.md]
        │
        ├─ python-project-ops    （環境・コマンド）
        │
        ├─ safe-data-handling    ─┐
        │                         │ ファイル I/O のたびに連動
        ├─ path-and-io           ─┘
        │
        ├─ python-style          ─┐
        │                         │ Python ファイル編集時
        ├─ dataframe-polars      ─┤
        │                         │
        ├─ visualization        ─┘  （図表を書くとき）
        │
        ├─ notebook-workflow     （.ipynb のとき）
        │
        ├─ sql-analysis          （.sql のとき）
        │
        ├─ statistical-ml-review （統計・ML 作業）
        │
        └─ analysis-reporting    （まとめを書くとき）
```

## プロンプト（スラッシュコマンド）との対応

| コマンド | 主に呼び出すスキル |
|----------|--------------------|
| `/plan-analysis` | statistical-ml-review, analysis-reporting |
| `/run-eda` | safe-data-handling, dataframe-polars, visualization, notebook-workflow |
| `/run-modeling` | statistical-ml-review, dataframe-polars, python-style |
| `/review-sql` | sql-analysis |
| `/summarize-analysis` | analysis-reporting |
| `/prepare-pr` | python-project-ops |
| `/update-agent-docs` | （メタ作業：本ファイルの保守） |
