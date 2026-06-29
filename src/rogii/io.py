"""データ入出力ユーティリティ。

raw データ（`data/raw/`）は不変。読み取り専用で扱う。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def project_root() -> Path:
    """リポジトリルート（このファイルから 2 つ上の src の親）を返す。"""
    return Path(__file__).resolve().parents[2]


def raw_dir() -> Path:
    return project_root() / "data" / "raw"


def list_well_ids(split: str) -> list[str]:
    """split('train'|'test') の坑井 id 一覧を返す。"""
    d = raw_dir() / split
    ids = sorted(p.name.split("__")[0] for p in d.glob("*__horizontal_well.csv"))
    return ids


def load_horizontal(split: str, well_id: str) -> pl.DataFrame:
    """横坑井 CSV を読み込む（MD 昇順を保証）。"""
    path = raw_dir() / split / f"{well_id}__horizontal_well.csv"
    return pl.read_csv(path).sort("MD")


def load_typewell(split: str, well_id: str) -> pl.DataFrame:
    """対応する縦坑井（typewell）CSV を読み込む（TVT 昇順）。"""
    path = raw_dir() / split / f"{well_id}__typewell.csv"
    return pl.read_csv(path).sort("TVT")


def ps_index(hw: pl.DataFrame) -> int:
    """Prediction Start (PS) の行インデックスを返す。

    PS = TVT_input が非 NaN である最終行の次の行インデックス。
    予測対象は index >= ps_index の行。
    """
    mask = hw["TVT_input"].is_not_null()
    n_known = int(mask.sum())
    return n_known
