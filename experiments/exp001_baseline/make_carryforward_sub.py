"""carry_forward（CV 最良）の提出ファイルを生成する。

PS 以降の TVT を PS-1 の既知 TVT で一定とする。
実行: uv run python experiments/exp001_baseline/make_carryforward_sub.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rogii import io  # noqa: E402


def main() -> None:
    out = io.project_root() / "outputs" / "submissions" / "exp001_carryforward.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for wid in io.list_well_ids("test"):
        hw = io.load_horizontal("test", wid)
        ps = io.ps_index(hw)
        ti = hw["TVT_input"].to_numpy().astype(float)
        known = ti[~np.isnan(ti)]
        tvt0 = float(known[-1])
        for i in range(ps, hw.height):
            rows.append({"id": f"{wid}_{i}", "tvt": tvt0})
    sub = pl.DataFrame(rows)
    sub.write_csv(out)
    print(f"submission -> {out} ({sub.height} rows)")


if __name__ == "__main__":
    main()
