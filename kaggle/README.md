# Kaggle 提出（Code Competition）

ROGII Wellbore Geology Prediction は **Code Competition** です。`outputs/submissions/*.csv`
を直接アップロードすることは**できません**。Kaggle 上で kernel（ノートブック/スクリプト）を
実行し、その出力 `submission.csv` を提出します。

## 構成

| ファイル | 役割 |
|----------|------|
| `submission/rogii_submit.py` | 提出本体（script kernel）。自己完結・pandas/numpy のみ依存 |
| `submission/kernel-metadata.json` | kernel 設定（コンペデータを source に指定、internet off） |
| `../scripts/kaggle_submit.sh` | push→実行待ち→提出のワンコマンド |
| `SUBMISSIONS.md` | 提出台帳（モデル / kernel version / LB スコア） |

## 提出フロー

```bash
# 0) （任意）ローカルで出力を検証
uv run python kaggle/submission/rogii_submit.py data/raw   # -> ./submission.csv

# 1) ワンコマンド提出（push → 実行完了待ち → submit）
bash scripts/kaggle_submit.sh "exp001 carry_forward baseline"

# 2) 結果確認
uv run kaggle competitions submissions -c rogii-wellbore-geology-prediction

# 3) SUBMISSIONS.md にスコアを追記
```

手動でやる場合:

```bash
uv run kaggle kernels push -p kaggle/submission
uv run kaggle kernels status tokudayujiro/rogii-wellbore-submit   # complete を待つ
uv run kaggle competitions submit -c rogii-wellbore-geology-prediction \
  -k tokudayujiro/rogii-wellbore-submit -f submission.csv -m "メッセージ"
```

## モデルを差し替えるとき（exp003+）

1. `submission/rogii_submit.py` の `MODEL` を変更し、対応する `predict_*` を実装。
2. **軽いロジック**（carry_forward 等、学習不要）はそのまま kernel 内で完結できる。
3. **重いモデル**（lgb 等、学習が必要）は kernel 内学習だと再現性/時間が不安定。
   学習済みモデルを **Kaggle Dataset** にアップロードし、`kernel-metadata.json` の
   `dataset_sources` に追加して kernel 内で読み込む方式に拡張する。
4. ローカルで `python kaggle/submission/rogii_submit.py data/raw` が通ることを確認してから push。
5. 提出後に `SUBMISSIONS.md` を更新。

## 注意

- 提出は外向き操作。`kaggle_submit.sh` 実行前に `MODEL` を確認すること。
- `submission.csv` はリポジトリにコミットしない（`.gitignore` 済み想定。生成物）。
- raw データ・認証情報はコミットしない（リポジトリ共通ルール）。
