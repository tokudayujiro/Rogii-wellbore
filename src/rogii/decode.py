"""GR シーケンス整合による TVT 復号（geosteering の HMM/Viterbi 定式化）。

横坑井の GR 波形が typewell の GR(TVT) 関係のどの深度に一致するかを、系列全体で
整合させて各点の TVT を復号する。点ごとの最近傍 GR マッチ（exp002）の非一意性を、
なめらかさ制約（隣接点の TVT 変化を罰則）と PS 既知 TVT の始点固定で解消する。

定式化（hidden=TVT[i] を grid 離散化）:
- 発光コスト E_i(s) = (gr_typewell(s) - gr_horizontal_smoothed[i])^2
- 遷移コスト      T(s, s') = lam * (s - s')^2   （TVT のなめらかさ。prior=stay）
- 始点 TVT[PS-1] は既知 → grid 中心に固定。
- Viterbi で総コスト最小経路を復号。窓幅 W で探索を制限（O(n·G·W)）。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from rogii.features import _fill_gr, _roll_mean_std


def _shift_inf(a: np.ndarray, d: int, fill: float) -> np.ndarray:
    """a_shifted[j] = a[j-d]（範囲外は fill）。"""
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


def viterbi_tvt(
    gr_h: np.ndarray,
    last_tvt: float,
    tw: pl.DataFrame,
    ps: int,
    *,
    radius: float = 60.0,
    step: float = 0.5,
    lam: float = 20.0,
    window: int = 8,
    gr_win: int = 15,
    calib: tuple[float, float] = (1.0, 0.0),
) -> np.ndarray:
    """横坑井 GR を typewell に整合させて TVT を復号する。

    Returns 全長 n の TVT 配列（i<ps は NaN、i>=ps が復号値）。typewell が使えない場合は
    carry_forward（last_tvt 一定）にフォールバック。
    """
    n = len(gr_h)
    out = np.full(n, np.nan)
    if ps >= n:
        return out

    tw_tvt = tw["TVT"].to_numpy().astype(float)
    tw_gr = tw["GR"].to_numpy().astype(float)
    m = ~np.isnan(tw_tvt) & ~np.isnan(tw_gr)
    if m.sum() < 2 or not np.isfinite(last_tvt):
        out[ps:] = last_tvt
        return out
    order = np.argsort(tw_tvt[m])
    tw_tvt, tw_gr = tw_tvt[m][order], tw_gr[m][order]

    grid = np.arange(last_tvt - radius, last_tvt + radius + step, step, dtype=np.float64)
    grid = grid[(grid >= tw_tvt[0]) & (grid <= tw_tvt[-1])]
    if grid.size < 3:
        out[ps:] = last_tvt
        return out
    g = grid.size
    a, b = calib  # 横坑井スケールに合わせた typewell GR の線形補正
    gr_tw = (a * np.interp(grid, tw_tvt, tw_gr) + b).astype(np.float32)  # (G,)
    center = int(np.argmin(np.abs(grid - last_tvt)))

    gr_s = _roll_mean_std(gr_h, gr_win)[0].astype(np.float32)

    offsets = np.arange(-window, window + 1)
    pen = (lam * (offsets * step) ** 2).astype(np.float32)  # 遷移コスト（ft^2）
    INF = np.float32(1e18)

    # i=ps: 中心からの遷移 + 発光
    dp = (gr_tw - gr_s[ps]) ** 2 + (lam * ((np.arange(g) - center) * step) ** 2).astype(np.float32)
    back = np.empty((n, g), dtype=np.int32)
    back[ps] = center

    idx = np.arange(g)
    for i in range(ps + 1, n):
        emit = (gr_tw - gr_s[i]) ** 2
        best = np.full(g, INF, dtype=np.float32)
        arg = np.zeros(g, dtype=np.int32)
        for od, p in zip(offsets, pen, strict=True):
            cand = _shift_inf(dp, int(od), float(INF)) + p  # dp[j-od] + pen
            upd = cand < best
            best[upd] = cand[upd]
            arg[upd] = idx[upd] - int(od)  # 前状態 k = j - od
        dp = best + emit
        back[i] = arg

    # 後退（最終状態から経路復元）
    j = int(np.argmin(dp))
    out[n - 1] = grid[j]
    for i in range(n - 1, ps, -1):
        j = int(back[i][j])
        out[i - 1] = grid[j]
    return out


def _calibrate(
    gr: np.ndarray, ti: np.ndarray, tw: pl.DataFrame, ps: int, k: int
) -> tuple[float, float]:
    """PS 既知区間で横坑井 GR を typewell GR(TVT) に線形整合する係数 (a, b)。

    既知区間では TVT(=TVT_input) と GR の両方が分かるので、typewell の GR(TVT) との
    スケール/オフセット差を gr_h ≈ a·gr_tw(TVT)+b で最小二乗推定する（slide 9 の足がかり）。
    """
    tw_tvt = tw["TVT"].to_numpy().astype(float)
    tw_gr = tw["GR"].to_numpy().astype(float)
    m = ~np.isnan(tw_tvt) & ~np.isnan(tw_gr)
    if m.sum() < 2:
        return 1.0, 0.0
    order = np.argsort(tw_tvt[m])
    tw_tvt, tw_gr = tw_tvt[m][order], tw_gr[m][order]

    lo = max(0, ps - k)
    tvt_k = ti[lo:ps]
    gr_k = gr[lo:ps]
    ok = ~np.isnan(tvt_k) & ~np.isnan(gr_k)
    if ok.sum() < 10:
        return 1.0, 0.0
    gr_tw_k = np.interp(tvt_k[ok], tw_tvt, tw_gr)
    # gr_k ≈ a·gr_tw_k + b
    A = np.column_stack([gr_tw_k, np.ones(ok.sum())])
    coef, *_ = np.linalg.lstsq(A, gr_k[ok], rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if not (np.isfinite(a) and np.isfinite(b)) or abs(a) < 1e-6:
        return 1.0, 0.0
    return a, b


def decode_well(
    hw: pl.DataFrame, tw: pl.DataFrame, ps: int, *, calib_k: int = 400, **kw
) -> np.ndarray:
    """1 坑井の復号 TVT（全長 n、i>=ps が予測）。"""
    gr = _fill_gr(hw["GR"].to_numpy().astype(float))
    ti = hw["TVT_input"].to_numpy().astype(float)
    known = ti[~np.isnan(ti)]
    last_tvt = float(known[-1]) if known.size else 0.0
    calib = _calibrate(gr, ti, tw, ps, calib_k)
    return viterbi_tvt(gr, last_tvt, tw, ps, calib=calib, **kw)
