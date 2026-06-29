#!/usr/bin/env bash
# Kaggle competition data downloader for ROGII Wellbore Geology Prediction.
#
# 前提:
#   1. ~/.kaggle/kaggle.json に有効な API トークンがあること
#      (https://www.kaggle.com/settings → "Create New Token")
#   2. コンペページで規約に同意済みであること
#      ("I Understand and Accept" をクリック)
#
# 使い方: bash scripts/download_data.sh
set -euo pipefail

COMP="rogii-wellbore-geology-prediction"
RAW_DIR="data/raw"

mkdir -p "${RAW_DIR}"

echo "[1/2] downloading competition files -> ${RAW_DIR}"
uv run kaggle competitions download -c "${COMP}" -p "${RAW_DIR}"

echo "[2/2] extracting zip archives"
for z in "${RAW_DIR}"/*.zip; do
  [ -e "$z" ] || continue
  echo "  unzip $z"
  uv run python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$z" "${RAW_DIR}"
  rm -f "$z"
done

echo "done. files in ${RAW_DIR}:"
ls -la "${RAW_DIR}"
