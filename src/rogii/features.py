"""特徴量生成。

定式化: TVT の 1 ステップ増分 d[i] = TVT[i] - TVT[i-1] を予測し、
既知の TVT[PS-1] から累積して TVT を復元する（geosteering の dip 推定に相当）。

テスト時に利用可能な情報のみを特徴量にする:
- GR は全区間既知（未来側の GR を使う rolling も可）
- 幾何（MD, X, Y, Z）は全区間既知
- TVT は PS 点まで（= TVT_input）のみ。PS 以降の TVT は特徴量に使わない
"""

from __future__ import annotations

import numpy as np
import polars as pl

KNOWN_SLOPE_K = 100  # PS 直前の dip 推定に使う点数
GR_WIN = 25  # GR rolling 窓
MATCH_RADIUS = 40.0  # typewell GR マッチングの探索半径（ft）
MATCH_STEP = 0.5  # マッチング探索の TVT 分解能（ft）


def _fill_gr(gr: np.ndarray) -> np.ndarray:
    """GR の NaN を線形補間 + 端は最近傍で埋める。"""
    x = np.arange(len(gr))
    mask = ~np.isnan(gr)
    if mask.sum() == 0:
        return np.zeros_like(gr)
    return np.interp(x, x[mask], gr[mask])


def _roll_mean_std(a: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """中心化窓の rolling mean / std を O(n) で計算（端は窓を縮める）。"""
    n = len(a)
    half = win // 2
    csum = np.concatenate([[0.0], np.cumsum(a)])
    csum2 = np.concatenate([[0.0], np.cumsum(a * a)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n, idx + half + 1)
    cnt = (hi - lo).astype(float)
    s = csum[hi] - csum[lo]
    s2 = csum2[hi] - csum2[lo]
    mean = s / cnt
    var = np.maximum(0.0, s2 / cnt - mean**2)
    return mean, np.sqrt(var)


def build_well_features(hw: pl.DataFrame, ps: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """1 坑井の (特徴行列 X, 増分ターゲット d, 列名) を返す。

    全行ぶん返す（行 i は TVT[i]-TVT[i-1]）。利用側で i>=ps を選択する。
    train では d は実値、test では NaN。
    """
    md = hw["MD"].to_numpy().astype(float)
    z = hw["Z"].to_numpy().astype(float)
    x = hw["X"].to_numpy().astype(float)
    y = hw["Y"].to_numpy().astype(float)
    gr = _fill_gr(hw["GR"].to_numpy().astype(float))
    n = len(md)

    tvt = hw["TVT"].to_numpy().astype(float) if "TVT" in hw.columns else np.full(n, np.nan)
    d = np.empty(n)
    d[0] = np.nan
    d[1:] = np.diff(tvt)

    dmd = np.empty(n)
    dmd[0] = np.nan
    dmd[1:] = np.diff(md)
    dz = np.empty(n)
    dz[0] = np.nan
    dz[1:] = np.diff(z)
    incl = dz / dmd  # 鉛直方向の傾き（dip の幾何成分）

    gr_grad = np.empty(n)
    gr_grad[0] = 0.0
    gr_grad[1:] = np.diff(gr)
    gr_rmean, gr_rstd = _roll_mean_std(gr, GR_WIN)

    # PS 直前の既知 dip（TVT 勾配）。PS 以降一定値として broadcast。
    ps_md = md[ps - 1] if ps >= 1 else md[0]
    k0 = max(0, ps - KNOWN_SLOPE_K)
    if ps - 1 > k0 and (md[ps - 1] - md[k0]) != 0:
        last_slope = (tvt[ps - 1] - tvt[k0]) / (md[ps - 1] - md[k0])
    else:
        last_slope = 0.0
    last_slope = float(np.nan_to_num(last_slope))

    known_gr_mean = float(np.mean(gr[:ps])) if ps >= 1 else float(np.mean(gr))
    known_gr_std = float(np.std(gr[:ps])) if ps >= 1 else float(np.std(gr))

    dist_from_ps = md - ps_md
    idx_from_ps = np.arange(n) - ps

    feats = {
        "gr": gr,
        "gr_grad": gr_grad,
        "gr_rmean": gr_rmean,
        "gr_rstd": gr_rstd,
        "gr_dev_known": gr - known_gr_mean,  # 既知区間平均からの偏差
        "dmd": np.nan_to_num(dmd, nan=1.0),
        "incl": np.nan_to_num(incl),
        "z": z,
        "dist_from_ps": dist_from_ps,
        "idx_from_ps": idx_from_ps.astype(float),
        "last_slope": np.full(n, last_slope),
        "known_gr_mean": np.full(n, known_gr_mean),
        "known_gr_std": np.full(n, known_gr_std),
        "x_off": x - x[ps - 1] if ps >= 1 else x - x[0],
        "y_off": y - y[ps - 1] if ps >= 1 else y - y[0],
    }
    cols = list(feats.keys())
    X = np.column_stack([feats[c] for c in cols])
    return X, d, cols


def _typewell_match(
    gr_h: np.ndarray, last_tvt: float, tw: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """横坑井 GR を typewell の GR(TVT) に相関させ、各点の整合 TVT を推定する。

    last_tvt を中心に ±MATCH_RADIUS の TVT グリッドを張り、各横坑井点の（平滑化）
    GR に最も近い typewell GR を与える TVT を選ぶ。geosteering の GR 相関の素朴版。

    Returns
    -------
    match_off : 各点の「整合 TVT − last_tvt」（GR から推定した PS からのオフセット）
    misfit : その整合点での GR 残差（小さいほど信頼できる）
    """
    n = len(gr_h)
    tw_tvt = tw["TVT"].to_numpy().astype(float)
    tw_gr = tw["GR"].to_numpy().astype(float)
    m = ~np.isnan(tw_tvt) & ~np.isnan(tw_gr)
    if m.sum() < 2 or not np.isfinite(last_tvt):
        return np.zeros(n), np.full(n, np.nan)
    tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
    order = np.argsort(tw_tvt)
    tw_tvt, tw_gr = tw_tvt[order], tw_gr[order]

    # last_tvt 周辺の TVT グリッドと、その点での typewell GR
    grid = np.arange(last_tvt - MATCH_RADIUS, last_tvt + MATCH_RADIUS + MATCH_STEP, MATCH_STEP)
    grid = grid[(grid >= tw_tvt[0]) & (grid <= tw_tvt[-1])]
    if grid.size == 0:
        return np.zeros(n), np.full(n, np.nan)
    grid_gr = np.interp(grid, tw_tvt, tw_gr)  # (G,)

    # 平滑化 GR を使い |grid_gr - gr| を最小化する TVT を各点で選ぶ
    gr_s, _ = _roll_mean_std(gr_h, GR_WIN)
    diff = np.abs(grid_gr[None, :] - gr_s[:, None])  # (n, G)
    j = np.argmin(diff, axis=1)
    match_tvt = grid[j]
    misfit = diff[np.arange(n), j]
    return match_tvt - last_tvt, misfit


def build_well_features_v2(
    hw: pl.DataFrame, tw: pl.DataFrame, ps: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """exp002 用: PS からのオフセット `TVT − TVT[PS-1]` を直接回帰する特徴量。

    exp001 の増分(ΔTVT)積分はドリフトが累積し carry_forward に劣後した。本関数は
    オフセットを直接ターゲットにし、最低でも 0(=carry_forward) を学べる定式化にする。
    さらに typewell GR–TVT 相関（geosteering の本命シグナル）を特徴に加える。

    Returns (X, y_offset, cols)。y_offset[i] = TVT[i] - TVT[PS-1]（train のみ実値）。
    test では y は NaN。利用側は i>=ps を選択する。
    """
    X1, _d, cols1 = build_well_features(hw, ps)
    gr = _fill_gr(hw["GR"].to_numpy().astype(float))
    n = X1.shape[0]

    tvt = hw["TVT"].to_numpy().astype(float) if "TVT" in hw.columns else np.full(n, np.nan)
    last_tvt = float(tvt[ps - 1]) if ps >= 1 and not np.isnan(tvt[ps - 1]) else np.nan
    if not np.isfinite(last_tvt):
        # test 等で TVT 不在: TVT_input の最終既知値を使う
        ti = hw["TVT_input"].to_numpy().astype(float)
        ti = ti[~np.isnan(ti)]
        last_tvt = float(ti[-1]) if ti.size else 0.0

    match_off, misfit = _typewell_match(gr, last_tvt, tw)

    y = tvt - last_tvt  # PS からのオフセット

    extra = {
        "match_off": np.nan_to_num(match_off),
        "match_misfit": np.nan_to_num(
            misfit, nan=float(np.nanmax(misfit)) if np.isfinite(np.nanmax(misfit)) else 0.0
        ),
    }
    cols = cols1 + list(extra.keys())
    X = np.column_stack([X1] + [extra[c] for c in extra])
    return X, y, cols
