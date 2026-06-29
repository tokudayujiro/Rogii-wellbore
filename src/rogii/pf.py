"""尤度重み付き粒子フィルタによる TVT トラッキング（geosteering の本命手法）。

参考解法（LB 7.168）の中核を非リークで再実装。Viterbi（exp003）との違い:
- 状態を `pos = TVT + Z`（地層マーカーの絶対位置）とし、**dip rate を慣性付きで追従**する。
  PS 直前の既知 dip から rate を初期化するので、フラット仮定でなく「地層の傾きトレンド」を辿る。
- GR の Gaussian 尤度 `exp(-0.5·((GR_h − GR_typewell(TVT))/σ)²)` で粒子を重み付け・リサンプル。
- **多シード化**: 同じ PF を n_seeds 回まわし、各シードの総対数尤度で softmax 合成（温度 scale）。
  GR と整合するシードを選び、ドリフトに強い推定にする。

状態遷移（MD ステップ dm ごと）:
    rate ← MOM·rate + VN·N(0,1);  pos ← pos + rate·dm + PN·N(0,1)
    TVT  = pos − Z
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

# 既定ハイパラ（参考解法準拠）
N_PARTICLES = 500
N_SEEDS = 128
MOM = 0.998  # dip rate の慣性
VN = 0.002  # rate のプロセスノイズ
PN = 0.005  # pos のプロセスノイズ
RESAMP = 0.5  # neff < RESAMP·N でリサンプル
INIT_SPR = 4.5  # pos 初期ばらつき
RP = 0.1  # リサンプル時の pos roughening
RR = 0.001  # リサンプル時の rate roughening
GRID_STEP = 0.2  # typewell GR グリッド分解能
SCALE = 8.0  # シード合成の softmax 温度


@njit(cache=True)
def _interp1(grid: np.ndarray, v: float, vmin: float, step: float) -> float:
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
    """全シードの予測 (n_seeds, n) と総対数尤度 (n_seeds,) を返す。"""
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
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
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


def _grid(tw_tvt: np.ndarray, tw_gr: np.ndarray, step: float = GRID_STEP):
    tmin, tmax = float(tw_tvt.min()), float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), tmin, float(step)


def pf_allseeds(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    n_particles: int = N_PARTICLES,
    n_seeds: int = N_SEEDS,
    init_spr: float = INIT_SPR,
    seed_base: int = 0,
    calibrate: bool = False,
    mom: float = MOM,
    vn: float = VN,
    pn: float = PN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PF を全シード走らせ (preds[n_seeds, n_ev], liks[n_seeds], ev_idx) を返す。

    1 回の計算結果から複数 scale を合成できる（scale チューニング用）。typewell 等が
    使えない場合は空配列を返す。calibrate=True で既知区間から GR_tw を横坑井スケールへ
    線形補正（gr_h ≈ a·gr_tw+b）してから尤度評価する。
    """
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
    ls = float(last["TVT_input"]) + float(last["Z"])  # 初期 pos = 既知 TVT + Z

    # 既知区間から GR_tw → 横坑井 GR の線形補正（任意）
    tw_at_k = np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gr_k = kn["GR"].fillna(0).to_numpy(float)
    if calibrate:
        ok = np.isfinite(tw_at_k) & np.isfinite(gr_k)
        if ok.sum() >= 10:
            A = np.column_stack([tw_at_k[ok], np.ones(int(ok.sum()))])
            coef, *_ = np.linalg.lstsq(A, gr_k[ok], rcond=None)
            a, b = float(coef[0]), float(coef[1])
            if np.isfinite(a) and np.isfinite(b) and abs(a) > 1e-6:
                tw_gr = a * tw_gr + b
                tw_at_k = a * tw_at_k + b

    # 発光ノイズ σ: 既知区間の (GR_h − GR_tw(TVT)) の std（[10,60] にクリップ）
    gs = float(np.clip(np.nanstd(gr_k - tw_at_k), 10.0, 60.0))

    # 初期 dip rate = 既知末尾 30 点の (ΔTVT+ΔZ)/ΔMD の中央値
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(float))
    dz = np.diff(tail["Z"].to_numpy(float))
    dm = np.diff(tail["MD"].to_numpy(float))
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    gg, gmin, gst = _grid(tw_tvt, tw_gr)
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
        mom,
        vn,
        pn,
        RP,
        RR,
        RESAMP,
        init_spr,
    )
    return preds, liks, ev_idx


def combine_scale(preds: np.ndarray, liks: np.ndarray, scale: float) -> np.ndarray:
    """シード対数尤度の softmax(温度 scale) で preds を合成。"""
    ln = liks - liks.max()
    wts = np.exp(ln / float(scale))
    wts /= wts.sum()
    return (wts[:, None] * preds).sum(0)


def lik_pf(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    n_particles: int = N_PARTICLES,
    n_seeds: int = N_SEEDS,
    scale: float = SCALE,
    init_spr: float = INIT_SPR,
    seed_base: int = 0,
    mom: float = MOM,
    vn: float = VN,
    pn: float = PN,
) -> np.ndarray:
    """予測対象（TVT_input が NaN の行）の TVT 推定を全長 n 配列で返す（既知部は NaN）。

    hw 列: MD, Z, GR, TVT_input。tw 列: TVT, GR。
    """
    out = np.full(hw.shape[0], np.nan)
    preds, liks, ev_idx = pf_allseeds(
        hw,
        tw,
        n_particles=n_particles,
        n_seeds=n_seeds,
        init_spr=init_spr,
        seed_base=seed_base,
        mom=mom,
        vn=vn,
        pn=pn,
    )
    if preds.size == 0:
        return out
    out[ev_idx] = combine_scale(preds, liks, scale)
    return out
