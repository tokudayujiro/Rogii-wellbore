#!/usr/bin/env bash
# ROGII Wellbore 提出ワンコマンド: kernel を push → 完了待ち → 提出。
#
# 前提:
#   - ~/.kaggle/kaggle.json に有効なトークン
#   - コンペ規約に同意済み
#   - kaggle/submission/ に kernel-metadata.json と rogii_submit.py がある
#
# 使い方:
#   bash scripts/kaggle_submit.sh "提出メッセージ"
#
# 注意: 実際に LB へ提出する外向き操作。実行前に kaggle/submission/rogii_submit.py の
#       MODEL が意図したものか確認すること。提出後は kaggle/SUBMISSIONS.md に記録する。
set -euo pipefail

# Windows(cp932) で kaggle CLI が UTF-8 の日本語コメントを読めず落ちるのを防ぐ
export PYTHONUTF8=1

COMP="rogii-wellbore-geology-prediction"
KERNEL="tokudayujiro/rogii-wellbore-submit"
KDIR="kaggle/submission"
MSG="${1:?提出メッセージを指定してください: bash scripts/kaggle_submit.sh \"...\"}"

echo "[1/3] push kernel ($KERNEL) ..."
push_out="$(uv run kaggle kernels push -p "$KDIR" 2>&1 | grep -v Warning)"
echo "$push_out"
# "Kernel version N successfully pushed" から版番号を取得
VERSION="$(echo "$push_out" | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' | head -1)"
[ -n "$VERSION" ] || { echo "版番号の取得に失敗"; exit 1; }
echo "  -> kernel version $VERSION"

echo "[2/3] wait for kernel run to complete ..."
# push 直後は実行中。COMPLETE になるまでポーリング（status は大文字 COMPLETE/ERROR）。
for _ in $(seq 1 80); do
  status="$(uv run kaggle kernels status "$KERNEL" 2>&1 | grep -v Warning || true)"
  echo "  $status"
  case "$status" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCEL*) echo "kernel run failed (ログ: kaggle kernels output $KERNEL)"; exit 1 ;;
  esac
  sleep 15
done

echo "[3/3] submit kernel output to competition (version=$VERSION) ..."
# code competition は -k（kernel）+ -v（version）+ -f（出力ファイル名）が必須。
uv run kaggle competitions submit -c "$COMP" -k "$KERNEL" -v "$VERSION" -f submission.csv -m "$MSG"

echo "done. 結果確認: uv run kaggle competitions submissions -c $COMP"
echo "→ kaggle/SUBMISSIONS.md にスコアを記録してください。"
