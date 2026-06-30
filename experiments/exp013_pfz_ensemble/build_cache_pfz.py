"""exp013: Z 速度連成 PF（pf_z, scale8）を全 train 坑井でキャッシュ。

v2 キャッシュ（pf/beam/特徴）に後で merge する用に、(well, idx_from_ps, pfz_pred, pfz_off)
だけを出力する。pf_z はゲートテストで PF と blend すると base を改善した（9.72→9.38）。

出力: data/interim/exp013_pfz.parquet
実行: uv run python experiments/exp013_pfz_ensemble/build_cache_pfz.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rogii import io, pf  # noqa: E402

PF_KW = dict(n_seeds=64, n_particles=300)
N_JOBS = 4
OUT = io.project_root() / "data" / "interim" / "exp013_pfz.parquet"


def build_rows(wid: str) -> pd.DataFrame | None:
    hw = io.load_horizontal("train", wid).to_pandas()
    tw = io.load_typewell("train", wid).to_pandas()
    ps = int(hw["TVT_input"].notna().sum())
    n = hw.shape[0]
    if ps >= n:
        return None
    preds, liks, ev_idx = pf.pf_z_allseeds(hw, tw, **PF_KW)
    if preds.size == 0 or not np.array_equal(ev_idx, np.arange(ps, n)):
        return None
    pfz = pf.combine_scale(preds, liks, 8.0)
    last = float(hw["TVT_input"].dropna().iloc[-1])
    return pd.DataFrame(
        {
            "well": wid,
            "idx_from_ps": np.arange(ps, n) - ps,
            "pfz_pred": pfz,
            "pfz_off": pfz - last,
        }
    )


def main() -> None:
    wids = io.list_well_ids("train")
    hw0 = io.load_horizontal("train", wids[0]).to_pandas()
    tw0 = io.load_typewell("train", wids[0]).to_pandas()
    pf.lik_pf_z(hw0, tw0, n_seeds=4, n_particles=64)  # warmup

    print(f"pf_z cache for {len(wids)} wells (parallel x{N_JOBS}) ...")
    t0 = time.time()
    dfs = Parallel(n_jobs=N_JOBS, backend="threading")(delayed(build_rows)(w) for w in wids)
    dfs = [d for d in dfs if d is not None]
    out = pd.concat(dfs, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    print(
        f"done in {time.time() - t0:.0f}s -> {OUT} ({len(out)} rows, {out['well'].nunique()} wells)"
    )


if __name__ == "__main__":
    main()
