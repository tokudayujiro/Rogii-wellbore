"""exp007: PF/beam 予測 + 特徴量を全 train 坑井でキャッシュ（残差スタッキング用）。

PF はフィットしないので坑井ごと独立に計算（リークなし）。予測対象行（i>=ps）について
PF/beam 予測と test 時に利用可能な特徴量、真 TVT を 1 つの parquet に保存する。

出力: data/interim/exp007_ml_rows.parquet
実行: uv run python experiments/exp007_ml_stack/build_cache.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rogii import beam as beam_mod  # noqa: E402
from rogii import io, pf  # noqa: E402

PF_KW = dict(n_seeds=64, n_particles=300)
N_JOBS = 4
OUT = io.project_root() / "data" / "interim" / "exp007_ml_rows_v2.parquet"
SCALES = (3.0, 5.0, 8.0, 12.0)


def _roll(a: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(a)
    m = s.rolling(win, center=True, min_periods=1).mean().to_numpy()
    sd = s.rolling(win, center=True, min_periods=1).std().fillna(0.0).to_numpy()
    return m, sd


def build_rows(wid: str) -> pd.DataFrame | None:
    hw = io.load_horizontal("train", wid).to_pandas()
    tw = io.load_typewell("train", wid).to_pandas()
    ps = int(hw["TVT_input"].notna().sum())
    n = hw.shape[0]
    if ps >= n:
        return None
    tvt = hw["TVT"].to_numpy(float)
    md = hw["MD"].to_numpy(float)
    z = hw["Z"].to_numpy(float)
    x = hw["X"].to_numpy(float)
    y = hw["Y"].to_numpy(float)
    gr = pd.Series(hw["GR"]).interpolate(limit_direction="both").to_numpy(float)
    ti = hw["TVT_input"].to_numpy(float)
    last = float(ti[~np.isnan(ti)][-1])

    # PF / beam
    preds, liks, ev_idx = pf.pf_allseeds(hw, tw, **PF_KW)
    if preds.size == 0 or not np.array_equal(ev_idx, np.arange(ps, n)):
        return None
    pf_by_scale = {sc: pf.combine_scale(preds, liks, sc) for sc in SCALES}
    pf_pred = pf_by_scale[8.0]
    beam_ev, beam_idx = beam_mod.beam_ensemble(hw, tw)
    if beam_ev.size != ev_idx.size:
        beam_ev = pf_pred.copy()

    # 既知区間統計
    known_gr_mean = float(np.mean(gr[:ps]))
    known_gr_std = float(np.std(gr[:ps]))
    k0 = max(0, ps - 100)
    last_slope = (
        float((tvt[ps - 1] - tvt[k0]) / (md[ps - 1] - md[k0]))
        if (md[ps - 1] - md[k0]) != 0
        else 0.0
    )
    tw_s = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    gr_sig = 30.0
    if len(tw_s) >= 2:
        tw_at_k = np.interp(ti[:ps][~np.isnan(ti[:ps])], tw_s["TVT"], tw_s["GR"])
        gr_sig = float(np.clip(np.std(gr[:ps][: len(tw_at_k)] - tw_at_k), 10.0, 60.0))
    z_eval = z[ps:]
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval))

    gr_rm, gr_rs = _roll(gr, 25)
    dmd = np.concatenate([[1.0], np.diff(md)])
    dz = np.concatenate([[0.0], np.diff(z)])
    incl = np.divide(dz, dmd, out=np.zeros_like(dz), where=dmd != 0)

    sl = slice(ps, n)
    df = pd.DataFrame(
        {
            "well": wid,
            "tvt": tvt[sl],
            "last": last,
            "pf_pred": pf_pred,
            "pf3": pf_by_scale[3.0],
            "pf5": pf_by_scale[5.0],
            "pf12": pf_by_scale[12.0],
            "beam_pred": beam_ev,
            "z": z[sl],
            "dist_from_ps": md[sl] - md[ps - 1],
            "idx_from_ps": np.arange(ps, n) - ps,
            "dmd": dmd[sl],
            "incl": incl[sl],
            "gr": gr[sl],
            "gr_rmean": gr_rm[sl],
            "gr_rstd": gr_rs[sl],
            "gr_dev_known": gr[sl] - known_gr_mean,
            "x_off": x[sl] - x[ps - 1],
            "y_off": y[sl] - y[ps - 1],
            "pf_off": pf_pred - last,
            "beam_off": beam_ev - last,
            "n_eval": float(n - ps),
            "z_span": z_span,
            "known_gr_mean": known_gr_mean,
            "known_gr_std": known_gr_std,
            "last_slope": last_slope,
            "gr_sig": gr_sig,
        }
    )
    return df


def main() -> None:
    wids = io.list_well_ids("train")
    # JIT warmup
    hw0 = io.load_horizontal("train", wids[0]).to_pandas()
    tw0 = io.load_typewell("train", wids[0]).to_pandas()
    pf.lik_pf(hw0, tw0, n_seeds=4, n_particles=64)

    print(f"building ML rows for {len(wids)} wells (parallel x{N_JOBS}) ...")
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
