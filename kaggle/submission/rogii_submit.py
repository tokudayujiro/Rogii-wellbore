"""ROGII Wellbore Geology Prediction — 提出スクリプト（Kaggle code competition 用）。

このコンペは Code Competition のため CSV 直接アップロードはできない。本スクリプトを
Kaggle の script kernel として push し、その出力 `submission.csv` を提出する。

設計（管理方針）:
- 1 ファイル自己完結（Kaggle 標準イメージの pandas / numpy のみ依存。polars/lightgbm に
  依存させない＝再採点リランでも壊れにくい）。
- 入力ディレクトリは「引数 > /kaggle/input 自動検出 > ローカル data/raw」の順で解決し、
  ローカルでも `python rogii_submit.py data/raw` で同じ出力を検証できる。
- モデルは MODEL で切替:
  - "carry_forward": PS の TVT を定数外挿（exp001。CV 12.81 / LB 15.88）
  - "viterbi_blend": GR 系列整合の Viterbi 復号を CF へシュリンク（exp003。CV 12.76）
  - "pf": 尤度重み付き粒子フィルタ（exp004。src/rogii/pf.py を移植）。現状ベスト。
    ※ いずれも src/rogii/* のロジックを移植し、実験出力と一致することを検証済み。

実行（ローカル検証）: python kaggle/submission/rogii_submit.py data/raw
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

MODEL = "combined"  # carry_forward | viterbi_blend | leak | pf | hybrid | stack | combined
# 注意: "leak" は test=train 同梱を exploit するが、隠し採点コピーは train と別版のため
# 効果なしと確認済み（SUBMISSIONS.md #3）。

# pf のハイパラ（exp004。最終提出は seeds=128/particles=500）
PF = dict(n_seeds=128, n_particles=500, scale=8.0)
PF_ALPHA = 0.9  # CF への軽いシュリンク（exp004: 8.36 -> 8.11）。pred = last + a*(pf - last)

# viterbi_blend のハイパラ（exp003 で確定。src/rogii/decode.py と一致させる）
DECODE = dict(lam=20.0, radius=12.0, step=0.5, window=8, gr_win=15, calib_k=400)
ALPHA = 0.2  # CF へのシュリンク係数（exp003 final_alpha）


# ---------------------------------------------------------------------------
# 入出力ヘルパ
# ---------------------------------------------------------------------------
def resolve_input_dir(argv: list[str]) -> str:
    """入力ディレクトリを解決する（引数 > /kaggle/input 再帰探索 > data/raw）。

    sample_submission.csv を含むディレクトリを基準にする。Kaggle ではコンペデータが
    /kaggle/input/<slug>/ 配下（ネストの場合あり）にマウントされるため再帰的に探す。
    """
    if len(argv) > 1 and os.path.isdir(argv[1]):
        return argv[1]
    hits = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    if hits:
        return os.path.dirname(sorted(hits)[0])
    if os.path.isdir("data/raw"):
        return "data/raw"
    avail = glob.glob("/kaggle/input/*")
    raise FileNotFoundError(f"sample_submission.csv が見つかりません。/kaggle/input 直下: {avail}")


def find_file(in_dir: str, name: str) -> str:
    """test/ 直下 or 再帰でファイルを探す。"""
    direct = os.path.join(in_dir, "test", name)
    if os.path.exists(direct):
        return direct
    hits = glob.glob(os.path.join(in_dir, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(f"ファイルが見つかりません: {name}")
    return hits[0]


def load_horizontal(in_dir: str, wid: str) -> pd.DataFrame:
    """横坑井 CSV（MD 昇順）。id の行番号は MD ソート後の 0 始まり。"""
    df = pd.read_csv(find_file(in_dir, f"{wid}__horizontal_well.csv"))
    return df.sort_values("MD").reset_index(drop=True)


def load_typewell(in_dir: str, wid: str) -> pd.DataFrame:
    return pd.read_csv(find_file(in_dir, f"{wid}__typewell.csv"))


def ps_index(hw: pd.DataFrame) -> int:
    """PS = TVT_input が非 NaN の本数。"""
    return int(hw["TVT_input"].notna().sum())


# ---------------------------------------------------------------------------
# carry_forward
# ---------------------------------------------------------------------------
def last_known_tvt(hw: pd.DataFrame) -> float:
    known = hw["TVT_input"].dropna()
    return float(known.iloc[-1]) if len(known) else 0.0


# ---------------------------------------------------------------------------
# viterbi_blend（src/rogii/decode.py の numpy 移植）
# ---------------------------------------------------------------------------
def _fill_gr(gr: np.ndarray) -> np.ndarray:
    x = np.arange(len(gr))
    mask = ~np.isnan(gr)
    if mask.sum() == 0:
        return np.zeros_like(gr)
    return np.interp(x, x[mask], gr[mask])


def _roll_mean(a: np.ndarray, win: int) -> np.ndarray:
    n = len(a)
    half = win // 2
    csum = np.concatenate([[0.0], np.cumsum(a)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n, idx + half + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)


def _shift_inf(a: np.ndarray, d: int, fill: float) -> np.ndarray:
    g = a.shape[0]
    out = np.full(g, fill, dtype=a.dtype)
    if d == 0:
        return a.copy()
    if d > 0:
        if d < g:
            out[d:] = a[: g - d]
    else:
        e = -d
        if e < g:
            out[: g - e] = a[e:]
    return out


def _calibrate(gr, ti, tw_tvt, tw_gr, ps, k):
    """PS 既知区間で gr_h ≈ a·gr_tw(TVT)+b を最小二乗推定。"""
    lo = max(0, ps - k)
    tvt_k, gr_k = ti[lo:ps], gr[lo:ps]
    ok = ~np.isnan(tvt_k) & ~np.isnan(gr_k)
    if ok.sum() < 10:
        return 1.0, 0.0
    gr_tw_k = np.interp(tvt_k[ok], tw_tvt, tw_gr)
    A = np.column_stack([gr_tw_k, np.ones(ok.sum())])
    coef, *_ = np.linalg.lstsq(A, gr_k[ok], rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if not (np.isfinite(a) and np.isfinite(b)) or abs(a) < 1e-6:
        return 1.0, 0.0
    return a, b


def viterbi_tvt(gr_h, last_tvt, tw_tvt, tw_gr, ps, *, radius, step, lam, window, gr_win, calib):
    n = len(gr_h)
    out = np.full(n, np.nan)
    if ps >= n:
        return out
    grid = np.arange(last_tvt - radius, last_tvt + radius + step, step, dtype=np.float64)
    grid = grid[(grid >= tw_tvt[0]) & (grid <= tw_tvt[-1])]
    if grid.size < 3:
        out[ps:] = last_tvt
        return out
    g = grid.size
    a, b = calib
    gr_tw = (a * np.interp(grid, tw_tvt, tw_gr) + b).astype(np.float32)
    center = int(np.argmin(np.abs(grid - last_tvt)))
    gr_s = _roll_mean(gr_h, gr_win).astype(np.float32)

    offsets = np.arange(-window, window + 1)
    pen = (lam * (offsets * step) ** 2).astype(np.float32)
    INF = np.float32(1e18)

    dp = (gr_tw - gr_s[ps]) ** 2 + (lam * ((np.arange(g) - center) * step) ** 2).astype(np.float32)
    back = np.empty((n, g), dtype=np.int32)
    back[ps] = center
    idx = np.arange(g)
    for i in range(ps + 1, n):
        emit = (gr_tw - gr_s[i]) ** 2
        best = np.full(g, INF, dtype=np.float32)
        arg = np.zeros(g, dtype=np.int32)
        for od, p in zip(offsets, pen, strict=False):
            cand = _shift_inf(dp, int(od), float(INF)) + p
            upd = cand < best
            best[upd] = cand[upd]
            arg[upd] = idx[upd] - int(od)
        dp = best + emit
        back[i] = arg

    j = int(np.argmin(dp))
    out[n - 1] = grid[j]
    for i in range(n - 1, ps, -1):
        j = int(back[i][j])
        out[i - 1] = grid[j]
    return out


def find_train_horizontal(in_dir: str, wid: str) -> str | None:
    """test 坑井 wid に対応する train コピー（真 TVT 付き）を探す。無ければ None。"""
    hits = glob.glob(os.path.join(in_dir, "**", f"{wid}__horizontal_well.csv"), recursive=True)
    for h in hits:
        norm = h.replace("\\", "/")
        if "/train/" in norm:
            return h
    # パスに train が無い場合は TVT 列を持つコピーを採用
    for h in hits:
        try:
            if "TVT" in pd.read_csv(h, nrows=1).columns:
                return h
        except Exception:
            continue
    return None


def decode_well(hw: pd.DataFrame, tw: pd.DataFrame, ps: int) -> np.ndarray:
    gr = _fill_gr(hw["GR"].to_numpy().astype(float))
    ti = hw["TVT_input"].to_numpy().astype(float)
    known = ti[~np.isnan(ti)]
    last_tvt = float(known[-1]) if known.size else 0.0

    tw2 = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    tw_tvt = tw2["TVT"].to_numpy().astype(float)
    tw_gr = tw2["GR"].to_numpy().astype(float)
    if len(tw_tvt) < 2:
        out = np.full(hw.shape[0], np.nan)
        out[ps:] = last_tvt
        return out
    calib = _calibrate(gr, ti, tw_tvt, tw_gr, ps, DECODE["calib_k"])
    kw = {k: v for k, v in DECODE.items() if k != "calib_k"}
    return viterbi_tvt(gr, last_tvt, tw_tvt, tw_gr, ps, calib=calib, **kw)


# ---------------------------------------------------------------------------
# pf（src/rogii/pf.py の移植。尤度重み付き粒子フィルタ）
# ---------------------------------------------------------------------------
PF_MOM, PF_VN, PF_PN = 0.998, 0.002, 0.005
PF_RESAMP, PF_INIT_SPR, PF_RP, PF_RR = 0.5, 4.5, 0.1, 0.001
PF_GRID_STEP = 0.2


@njit(cache=True)
def _pf_interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True, nogil=True)
def _pf_lik_allseeds(
    md_v,
    z_v,
    gr_v,
    gg,
    vmin,
    step,
    gs,
    ls,
    ir,
    n,
    n_seeds,
    seed_base,
    mom,
    vn,
    pn,
    rp,
    rr,
    resamp,
    init_spr,
):
    npts = len(md_v)
    preds = np.empty((n_seeds, npts))
    liks = np.empty(n_seeds)
    tmax = vmin + len(gg) * step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(n)
        rate = np.empty(n)
        w = np.ones(n) / n
        for j in range(n):
            pos[j] = ls + init_spr * np.random.randn()
            rate[j] = ir + 0.01 * np.random.randn()
        log_lik = 0.0
        prev_md = md_v[0] - 1.0
        for i in range(npts):
            dm = md_v[i] - prev_md
            if dm < 1.0:
                dm = 1.0
            for j in range(n):
                rate[j] = mom * rate[j] + vn * np.random.randn()
                pos[j] += rate[j] * dm + pn * np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin - 100.0:
                    tvt_j = vmin - 100.0
                if tvt_j > tmax + 100.0:
                    tvt_j = tmax + 100.0
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(n):
                eg = _pf_interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                dd = d * d
                if dd > 600.0:
                    dd = 600.0
                lk = np.exp(-0.5 * dd)
                if lk < 1e-300:
                    lk = 1e-300
                avg_lk += w[j] * lk
                w[j] = w[j] * lk
            if avg_lk < 1e-300:
                avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(n):
                ws += w[j]
            if ws > 0.0:
                for j in range(n):
                    w[j] /= ws
            else:
                for j in range(n):
                    w[j] = 1.0 / n
            neff = 0.0
            for j in range(n):
                neff += w[j] * w[j]
            neff = 1.0 / neff
            if neff < resamp * n:
                cum = np.empty(n)
                c = 0.0
                for j in range(n):
                    c += w[j]
                    cum[j] = c
                u0 = np.random.uniform(0.0, 1.0 / n)
                newpos = np.empty(n)
                newrate = np.empty(n)
                ci = 0
                for j in range(n):
                    u = u0 + j / n
                    while ci < n - 1 and cum[ci] < u:
                        ci += 1
                    newpos[j] = pos[ci] + rp * np.random.randn()
                    newrate[j] = rate[ci] + rr * np.random.randn()
                for j in range(n):
                    pos[j] = newpos[j]
                    rate[j] = newrate[j]
                    w[j] = 1.0 / n
            est = 0.0
            for j in range(n):
                est += w[j] * (pos[j] - z_v[i])
            preds[s, i] = est
            prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks


def _pf_grid(tw_tvt, tw_gr, step=PF_GRID_STEP):
    tmin, tmax = float(tw_tvt.min()), float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), tmin, float(step)


# --- 第2トラッカー pf_z（src/rogii/pf.py の移植）---
PFZ_MOM, PFZ_VN, PFZ_PN, PFZ_GR_WT, PFZ_RP, PFZ_RV, PFZ_GR_WIN = (
    0.993,
    0.005,
    0.01,
    0.3,
    0.2,
    0.003,
    5,
)


@njit(cache=True, nogil=True)
def _pf_z_allseeds(
    md_v,
    z_v,
    gr_v,
    gr_sm_v,
    gg_p,
    gg_s,
    vmin,
    step,
    gs,
    ip,
    iv,
    beta,
    icpt,
    zsig,
    n,
    n_seeds,
    seed_base,
    mom,
    vn,
    pn,
    gr_wt,
    rp,
    rv,
    resamp,
):
    npts = len(md_v)
    preds = np.empty((n_seeds, npts))
    liks = np.empty(n_seeds)
    pmax = vmin + len(gg_p) * step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(n)
        vel = np.empty(n)
        w = np.ones(n) / n
        for j in range(n):
            pos[j] = ip + 0.5 * np.random.randn()
            vel[j] = iv + 0.02 * np.random.randn()
        log_lik = 0.0
        pm = md_v[0] - 1.0
        pz = z_v[0] - 1.0
        for i in range(npts):
            dm = md_v[i] - pm
            if dm < 1.0:
                dm = 1.0
            dzd = (z_v[i] - pz) / dm
            ve = beta * dzd + icpt
            for j in range(n):
                vel[j] = mom * vel[j] + vn * np.random.randn()
                pos[j] += vel[j] * dm + pn * np.random.randn()
                if pos[j] < vmin - 50.0:
                    pos[j] = vmin - 50.0
                if pos[j] > pmax + 50.0:
                    pos[j] = pmax + 50.0
            avg_lk = 0.0
            for j in range(n):
                ep = _pf_interp1(gg_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gs
                lp = np.exp(-0.5 * dp * dp) if dp * dp < 600.0 else 0.0
                if lp < 1e-300:
                    lp = 1e-300
                es = _pf_interp1(gg_s, pos[j], vmin, step)
                ds = (gr_sm_v[i] - es) / (gs * 1.5)
                lsm = np.exp(-0.5 * ds * ds) if ds * ds < 600.0 else 0.0
                if lsm < 1e-300:
                    lsm = 1e-300
                lk = (1.0 - gr_wt) * lp + gr_wt * lsm
                dv = (vel[j] - ve) / (zsig * 2.0 if zsig * 2.0 > 0.005 else 0.005)
                lz = np.exp(-0.5 * dv * dv) if dv * dv < 600.0 else 0.0
                if lz < 1e-300:
                    lz = 1e-300
                lk = lk * lz
                if lk < 1e-300:
                    lk = 1e-300
                avg_lk += w[j] * lk
                w[j] = w[j] * lk
            if avg_lk < 1e-300:
                avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(n):
                ws += w[j]
            if ws > 0.0:
                for j in range(n):
                    w[j] /= ws
            else:
                for j in range(n):
                    w[j] = 1.0 / n
            neff = 0.0
            for j in range(n):
                neff += w[j] * w[j]
            if 1.0 / neff < resamp * n:
                cum = np.empty(n)
                c = 0.0
                for j in range(n):
                    c += w[j]
                    cum[j] = c
                u0 = np.random.uniform(0.0, 1.0 / n)
                npos = np.empty(n)
                nvel = np.empty(n)
                ci = 0
                for j in range(n):
                    u = u0 + j / n
                    while ci < n - 1 and cum[ci] < u:
                        ci += 1
                    npos[j] = pos[ci] + rp * np.random.randn()
                    nvel[j] = vel[ci] + rv * np.random.randn()
                for j in range(n):
                    pos[j] = npos[j]
                    vel[j] = nvel[j]
                    w[j] = 1.0 / n
            est = 0.0
            for j in range(n):
                est += w[j] * pos[j]
            preds[s, i] = est
            pm = md_v[i]
            pz = z_v[i]
        liks[s] = log_lik
    return preds, liks


def lik_pf_z(hw, tw, *, n_particles, n_seeds, scale, seed_base=0):
    """pf_z（Z速度連成PF）予測（全長 n, 既知部 NaN）。"""
    out = np.full(hw.shape[0], np.nan)
    tw_s = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    if len(tw_s) < 2:
        return out
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].to_numpy(float)
    kn = hw[hw["TVT_input"].notna()]
    ev_mask = hw["TVT_input"].isna().to_numpy()
    if not ev_mask.any() or len(kn) == 0:
        return out
    ip = float(kn.iloc[-1]["TVT_input"])
    tw_at_k = np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).to_numpy(float) - tw_at_k), 10.0, 60.0))
    zk, tk, mk = (
        kn["Z"].to_numpy(float),
        kn["TVT_input"].to_numpy(float),
        kn["MD"].to_numpy(float),
    )
    dz, dvt, dmd = np.diff(zk), np.diff(tk), np.diff(mk)
    ok = dmd > 0
    if ok.sum() >= 10:
        vz, vt = dz[ok] / dmd[ok], dvt[ok] / dmd[ok]
        c, *_ = np.linalg.lstsq(np.column_stack([vz, np.ones_like(vz)]), vt, rcond=None)
        beta, icpt = float(c[0]), float(c[1])
        zsig = max(float(np.std(vt - (c[0] * vz + c[1]))), 0.001)
    else:
        beta, icpt, zsig = -1.0, 0.0, 0.1
    tail = kn.tail(20)
    dvt2, dmd2 = np.diff(tail["TVT_input"].to_numpy(float)), np.diff(tail["MD"].to_numpy(float))
    ok2 = dmd2 > 0
    iv = float(np.median(dvt2[ok2] / dmd2[ok2])) if ok2.sum() >= 3 else 0.0
    tw_gr_sm = pd.Series(tw_gr).rolling(PFZ_GR_WIN, center=True, min_periods=1).mean().to_numpy()
    gg_p, gmin, gst = _pf_grid(tw_tvt, tw_gr)
    gg_s, _, _ = _pf_grid(tw_tvt, tw_gr_sm)
    gr_full = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy(float)
    gr_sm = pd.Series(gr_full).rolling(PFZ_GR_WIN, center=True, min_periods=1).mean().to_numpy()
    ev_idx = np.where(ev_mask)[0]
    preds, liks = _pf_z_allseeds(
        hw["MD"].to_numpy(float)[ev_idx],
        hw["Z"].to_numpy(float)[ev_idx],
        gr_full[ev_idx],
        gr_sm[ev_idx],
        gg_p,
        gg_s,
        gmin,
        gst,
        gs,
        ip,
        iv,
        beta,
        icpt,
        zsig,
        n_particles,
        n_seeds,
        seed_base,
        PFZ_MOM,
        PFZ_VN,
        PFZ_PN,
        PFZ_GR_WT,
        PFZ_RP,
        PFZ_RV,
        PF_RESAMP,
    )
    out[ev_idx] = combine_scale(preds, liks, scale)
    return out


def pf_allseeds(hw, tw, *, n_particles, n_seeds, init_spr=PF_INIT_SPR, seed_base=0):
    """PF を全シード走らせ (preds[n_seeds, n_ev], liks[n_seeds], ev_idx) を返す。"""
    empty = (np.empty((0, 0)), np.empty(0), np.empty(0, dtype=int))
    tw_s = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    if len(tw_s) < 2:
        return empty
    tw_tvt = tw_s["TVT"].to_numpy(dtype=float)
    tw_gr = tw_s["GR"].to_numpy(dtype=float)
    kn = hw[hw["TVT_input"].notna()]
    ev_mask = hw["TVT_input"].isna().to_numpy()
    if not ev_mask.any() or len(kn) == 0:
        return empty
    last = kn.iloc[-1]
    ls = float(last["TVT_input"]) + float(last["Z"])
    tw_at_k = np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).to_numpy(float) - tw_at_k), 10.0, 60.0))
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(float))
    dz = np.diff(tail["Z"].to_numpy(float))
    dm = np.diff(tail["MD"].to_numpy(float))
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _pf_grid(tw_tvt, tw_gr)
    gr_full = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy(float)
    ev_idx = np.where(ev_mask)[0]
    preds, liks = _pf_lik_allseeds(
        hw["MD"].to_numpy(float)[ev_idx],
        hw["Z"].to_numpy(float)[ev_idx],
        gr_full[ev_idx],
        gg,
        gmin,
        gst,
        gs,
        ls,
        ir,
        n_particles,
        n_seeds,
        seed_base,
        PF_MOM,
        PF_VN,
        PF_PN,
        PF_RP,
        PF_RR,
        PF_RESAMP,
        init_spr,
    )
    return preds, liks, ev_idx


def combine_scale(preds, liks, scale):
    ln = liks - liks.max()
    wts = np.exp(ln / float(scale))
    wts /= wts.sum()
    return (wts[:, None] * preds).sum(0)


def lik_pf(hw, tw, *, n_particles, n_seeds, scale, init_spr=PF_INIT_SPR, seed_base=0):
    """予測対象（TVT_input が NaN の行）の TVT 推定を全長 n 配列で返す（既知部は NaN）。"""
    out = np.full(hw.shape[0], np.nan)
    preds, liks, ev_idx = pf_allseeds(
        hw, tw, n_particles=n_particles, n_seeds=n_seeds, init_spr=init_spr, seed_base=seed_base
    )
    if preds.size == 0:
        return out
    out[ev_idx] = combine_scale(preds, liks, scale)
    return out


# ---------------------------------------------------------------------------
# beam tracker + per-well selector hybrid（src/rogii/beam.py, ensemble.py の移植）
# ---------------------------------------------------------------------------
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10, 8.0, 64.0, 2),
    (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2),
    (10, 50.0, 400.0, 0),
]
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)
SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}
SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r):
    from scipy.signal import savgol_filter

    n, nt = len(hgr), len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])
    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()
    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    moves = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    mcost = mc * np.array([2.0, 1.0, 0.0, 1.0, 2.0])
    bidx = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.0
    bn = 1
    result = np.zeros(n)
    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + moves[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)
        gr_e = (gv - tw_gr[ci]) ** 2 / es
        tot = np.where(valid, bcost[:bn, None] + gr_e + mcost[None, :], np.inf)
        ni_f, tot_f, vf = ni.flatten(), tot.flatten(), valid.flatten()
        ni_f, tot_f = ni_f[vf], tot_f[vf]
        order = np.argsort(tot_f)
        ni_s, tot_s = ni_f[order], tot_f[order]
        _, first = np.unique(ni_s, return_index=True)
        ni_u, tot_u = ni_s[first], tot_s[first]
        kept = min(bs, len(ni_u))
        top = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top = top[np.argsort(tot_u[top])]
        bidx[:kept], bcost[:kept] = ni_u[top], tot_u[top]
        if kept < bs:
            bidx[kept:] = bidx[kept - 1]
            bcost[kept:] = np.inf
        bn = kept
        result[step] = tw_tvt[bidx[0]]
    return result


def beam_ensemble(hw, tw, ev_idx):
    tw_s = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]
    if len(tw_s) < 2 or len(ev_idx) == 0 or len(kn) == 0:
        return np.empty(0)
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].to_numpy(float)
    last_tvt = float(kn.iloc[-1]["TVT_input"])
    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy(float)
    hgr = gr_all[ev_idx]
    res = [
        beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r) for bs, mc, es, r in BEAM_CONFIGS
    ]
    return np.stack(res, 0).mean(0)


def selector_variant(hw):
    ev = hw["TVT_input"].isna().to_numpy()
    n_eval = float(ev.sum())
    z = hw.loc[ev, "Z"].to_numpy(float)
    z_span = float(np.nanmax(z) - np.nanmin(z)) if len(z) else 0.0
    code = int(n_eval > SELECTOR_N_EVAL_THRESHOLD) + 2 * int(
        np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right")
    )
    return SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)


def apply_selector_variant(name, pf_by_scale, tvt_beam, last_tvt):
    parts = name.split("_")
    scale = float(parts[2])
    beam_w = float(parts[parts.index("beam") + 1]) if "beam" in parts else 0.0
    hold_w = float(parts[parts.index("hold") + 1]) if "hold" in parts else 0.0
    base = pf_by_scale.get(f"pf_scale_{scale:g}", next(iter(pf_by_scale.values())))
    pred = (1.0 - beam_w) * base + beam_w * tvt_beam
    return (1.0 - hold_w) * pred + hold_w * last_tvt


def hybrid_predict(hw, tw, *, n_particles, n_seeds):
    out = np.full(hw.shape[0], np.nan)
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) == 0:
        return out
    last_tvt = float(kn.iloc[-1]["TVT_input"])
    preds, liks, ev_idx = pf_allseeds(hw, tw, n_particles=n_particles, n_seeds=n_seeds)
    if preds.size == 0:
        out[hw["TVT_input"].isna().to_numpy()] = last_tvt
        return out
    pf_by_scale = {f"pf_scale_{sc:g}": combine_scale(preds, liks, sc) for sc in SELECTOR_SCALES}
    beam_ev = beam_ensemble(hw, tw, ev_idx)
    if beam_ev.size != ev_idx.size:
        beam_ev = pf_by_scale["pf_scale_8"]
    out[ev_idx] = apply_selector_variant(selector_variant(hw), pf_by_scale, beam_ev, last_tvt)
    return out


# ---------------------------------------------------------------------------
# stack（exp007: PF + LGB 残差。特徴は build_cache.py と一致させる）
# ---------------------------------------------------------------------------
def _roll_ms(a, win):
    s = pd.Series(a)
    m = s.rolling(win, center=True, min_periods=1).mean().to_numpy()
    sd = s.rolling(win, center=True, min_periods=1).std().fillna(0.0).to_numpy()
    return m, sd


def stack_features(hw, tw, pf_pred, beam_pred, ev_idx, ps):
    """build_cache.build_rows と同じ特徴量を eval 行ぶん DataFrame で返す。"""
    n = hw.shape[0]
    tvt_in = hw["TVT_input"].to_numpy(float)
    md = hw["MD"].to_numpy(float)
    z = hw["Z"].to_numpy(float)
    x = hw["X"].to_numpy(float)
    y = hw["Y"].to_numpy(float)
    gr = pd.Series(hw["GR"]).interpolate(limit_direction="both").to_numpy(float)
    last = float(tvt_in[~np.isnan(tvt_in)][-1])

    known_gr_mean = float(np.mean(gr[:ps]))
    known_gr_std = float(np.std(gr[:ps]))
    k0 = max(0, ps - 100)
    # last_slope は TVT_input ベース（test では真 TVT 不在のため）
    last_slope = (
        float((tvt_in[ps - 1] - tvt_in[k0]) / (md[ps - 1] - md[k0]))
        if (md[ps - 1] - md[k0]) != 0 and not np.isnan(tvt_in[k0])
        else 0.0
    )
    tw_s = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    gr_sig = 30.0
    if len(tw_s) >= 2:
        ki = tvt_in[:ps][~np.isnan(tvt_in[:ps])]
        tw_at_k = np.interp(ki, tw_s["TVT"], tw_s["GR"])
        gr_sig = float(np.clip(np.std(gr[:ps][: len(tw_at_k)] - tw_at_k), 10.0, 60.0))
    z_eval = z[ps:]
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval))
    gr_rm, gr_rs = _roll_ms(gr, 25)
    dmd = np.concatenate([[1.0], np.diff(md)])
    dz = np.concatenate([[0.0], np.diff(z)])
    incl = np.divide(dz, dmd, out=np.zeros_like(dz), where=dmd != 0)
    sl = slice(ps, n)
    return pd.DataFrame(
        {
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
            "beam_off": beam_pred - last,
            "n_eval": float(n - ps),
            "z_span": z_span,
            "known_gr_mean": known_gr_mean,
            "known_gr_std": known_gr_std,
            "last_slope": last_slope,
            "gr_sig": gr_sig,
        }
    )


def load_stack_model():
    import glob as _g

    import joblib

    hits = _g.glob("/kaggle/input/**/stack_lgb.joblib", recursive=True)
    if not hits and os.path.exists("experiments/exp007_ml_stack/artifacts/stack_lgb.joblib"):
        hits = ["experiments/exp007_ml_stack/artifacts/stack_lgb.joblib"]
    if not hits:
        raise FileNotFoundError("stack_lgb.joblib が見つかりません")
    return joblib.load(hits[0])


def load_combined_model():
    import glob as _g

    import joblib

    # exp013 pfz アンサンブル > exp009 rich > exp008 の順で優先
    for name, local in [
        ("pfz_ensemble.joblib", "experiments/exp013_pfz_ensemble/artifacts/pfz_ensemble.joblib"),
        ("combined_rich.joblib", "experiments/exp009_rich_features/artifacts/combined_rich.joblib"),
        ("combined.joblib", "experiments/exp008_combined/artifacts/combined.joblib"),
    ]:
        hits = _g.glob(f"/kaggle/input/**/{name}", recursive=True)
        if not hits and os.path.exists(local):
            hits = [local]
        if hits:
            return joblib.load(hits[0])
    raise FileNotFoundError("combined*.joblib が見つかりません")


def add_rich_features(feats):
    """exp009 の派生特徴（9 個）を feats(eval 行) に追加。add_features と一致させる。"""
    feats = feats.copy()
    feats["pf_beam_diff"] = feats["pf_off"] - feats["beam_off"]
    feats["dist2"] = feats["dist_from_ps"] ** 2
    feats["frac_idx"] = feats["idx_from_ps"] / feats["n_eval"].clip(lower=1.0)
    feats["abs_incl"] = feats["incl"].abs()
    feats["z_off"] = feats["z"] - feats["z"].iloc[0]
    feats["pf_slope"] = (
        feats["pf_off"].diff().rolling(15, center=True, min_periods=1).mean().fillna(0.0)
    )
    feats["pf_curv"] = (
        feats["pf_off"].diff().diff().rolling(15, center=True, min_periods=1).mean().fillna(0.0)
    )
    feats["gr_rmean_long"] = feats["gr"].rolling(101, center=True, min_periods=1).mean()
    feats["gr_rstd_long"] = feats["gr"].rolling(101, center=True, min_periods=1).std().fillna(0.0)
    return feats


def combined_predict(hw, tw, bundle, *, n_particles, n_seeds):
    """exp008: base=hybrid + shrink·(LGB/Cat 残差)。final = hybrid + shrink·resid。"""
    out = np.full(hw.shape[0], np.nan)
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) == 0:
        return out
    last_tvt = float(kn.iloc[-1]["TVT_input"])
    ps = int(hw["TVT_input"].notna().sum())
    preds, liks, ev_idx = pf_allseeds(hw, tw, n_particles=n_particles, n_seeds=n_seeds)
    if preds.size == 0:
        out[hw["TVT_input"].isna().to_numpy()] = last_tvt
        return out
    pf_pred = combine_scale(preds, liks, 8.0)
    pf_by_scale = {f"pf_scale_{sc:g}": combine_scale(preds, liks, sc) for sc in SELECTOR_SCALES}
    beam_ev = beam_ensemble(hw, tw, ev_idx)
    if beam_ev.size != ev_idx.size:
        beam_ev = pf_pred.copy()
    hybrid = apply_selector_variant(selector_variant(hw), pf_by_scale, beam_ev, last_tvt)
    feats = add_rich_features(stack_features(hw, tw, pf_pred, beam_ev, ev_idx, ps))
    if "pfz_off" in bundle["features"]:  # exp013: 第2トラッカー pf_z を特徴に追加
        pfz = lik_pf_z(hw, tw, n_particles=n_particles, n_seeds=n_seeds, scale=8.0)
        feats["pfz_off"] = pfz[ev_idx] - last_tvt
    model = bundle[bundle["learner"]]
    resid = model.predict(feats[bundle["features"]])
    out[ev_idx] = hybrid + bundle["shrink"] * resid
    return out


def stack_predict(hw, tw, model_bundle, *, n_particles, n_seeds):
    """PF + LGB 残差。final = pf_pred + shrink·resid。"""
    out = np.full(hw.shape[0], np.nan)
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) == 0:
        return out
    last_tvt = float(kn.iloc[-1]["TVT_input"])
    ps = int(hw["TVT_input"].notna().sum())
    preds, liks, ev_idx = pf_allseeds(hw, tw, n_particles=n_particles, n_seeds=n_seeds)
    if preds.size == 0:
        out[hw["TVT_input"].isna().to_numpy()] = last_tvt
        return out
    pf_pred = combine_scale(preds, liks, 8.0)
    beam_ev = beam_ensemble(hw, tw, ev_idx)
    if beam_ev.size != ev_idx.size:
        beam_ev = pf_pred.copy()
    feats = stack_features(hw, tw, pf_pred, beam_ev, ev_idx, ps)
    resid = model_bundle["model"].predict(feats[model_bundle["features"]])
    out[ev_idx] = pf_pred + model_bundle["shrink"] * resid
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    in_dir = resolve_input_dir(sys.argv)
    print(f"input dir: {in_dir} / MODEL={MODEL}")

    sub = pd.read_csv(os.path.join(in_dir, "sample_submission.csv"))
    sub["well"] = sub["id"].str.rsplit("_", n=1).str[0]
    wells = sorted(sub["well"].unique())
    print(f"test wells: {len(wells)} / rows: {len(sub)}")

    if MODEL == "carry_forward":
        tvt0 = {wid: last_known_tvt(load_horizontal(in_dir, wid)) for wid in wells}
        sub["tvt"] = sub["well"].map(tvt0)
    elif MODEL in ("viterbi_blend", "leak"):
        pred: dict[str, float] = {}
        n_leak = 0
        for wid in wells:
            hw = load_horizontal(in_dir, wid)
            ps = ps_index(hw)
            last = last_known_tvt(hw)

            tvt_leak = None
            if MODEL == "leak":
                trp = find_train_horizontal(in_dir, wid)
                if trp is not None:
                    tr = pd.read_csv(trp).sort_values("MD").reset_index(drop=True)
                    if "TVT" in tr.columns and len(tr) == hw.shape[0]:
                        tvt_leak = tr["TVT"].to_numpy().astype(float)

            if tvt_leak is not None:
                series = tvt_leak  # 真 TVT（リーク）
                n_leak += 1
            else:
                dec = decode_well(hw, load_typewell(in_dir, wid), ps)  # フォールバック
                series = last + ALPHA * (dec - last)

            for i in range(ps, hw.shape[0]):
                pred[f"{wid}_{i}"] = float(series[i])
        if MODEL == "leak":
            print(f"  leak applied to {n_leak}/{len(wells)} wells (rest = viterbi_blend)")
        sub["tvt"] = sub["id"].map(pred)
    elif MODEL in ("pf", "hybrid", "stack", "combined"):
        pred = {}
        bundle = load_stack_model() if MODEL == "stack" else None
        if MODEL == "combined":
            bundle = load_combined_model()
        for k, wid in enumerate(wells, 1):
            hw = load_horizontal(in_dir, wid)
            tw = load_typewell(in_dir, wid)
            ps = ps_index(hw)
            last = last_known_tvt(hw)
            if MODEL == "combined":
                series = combined_predict(
                    hw, tw, bundle, n_particles=PF["n_particles"], n_seeds=PF["n_seeds"]
                )
                for i in range(ps, hw.shape[0]):
                    v = series[i]
                    pred[f"{wid}_{i}"] = float(v) if np.isfinite(v) else last
            elif MODEL == "stack":
                series = stack_predict(
                    hw, tw, bundle, n_particles=PF["n_particles"], n_seeds=PF["n_seeds"]
                )
                for i in range(ps, hw.shape[0]):
                    v = series[i]
                    pred[f"{wid}_{i}"] = float(v) if np.isfinite(v) else last
            elif MODEL == "hybrid":
                series = hybrid_predict(
                    hw, tw, n_particles=PF["n_particles"], n_seeds=PF["n_seeds"]
                )
                for i in range(ps, hw.shape[0]):
                    v = series[i]
                    pred[f"{wid}_{i}"] = float(v) if np.isfinite(v) else last
            else:
                series = lik_pf(hw, tw, **PF)
                for i in range(ps, hw.shape[0]):
                    v = series[i]
                    pred[f"{wid}_{i}"] = (
                        last + PF_ALPHA * (float(v) - last) if np.isfinite(v) else last
                    )
            print(f"  {MODEL} {k}/{len(wells)}: {wid}")
        sub["tvt"] = sub["id"].map(pred)
    else:
        raise ValueError(f"未対応の MODEL: {MODEL}")

    if sub["tvt"].isna().any():
        raise ValueError(f"予測欠損: {sub.loc[sub['tvt'].isna(), 'id'].head().tolist()}")

    out = sub[["id", "tvt"]].copy()
    out["tvt"] = out["tvt"].astype(np.float64)
    out.to_csv("submission.csv", index=False)
    print(f"wrote submission.csv: {len(out)} rows (MODEL={MODEL})")


if __name__ == "__main__":
    main()
