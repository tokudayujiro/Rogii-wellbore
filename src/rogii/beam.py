"""beam search トラッカー（PF とは別系統の GR 整合）。

PF が `pos=TVT+Z` の慣性追従なのに対し、beam は **typewell の GR(TVT) 曲線上を
インデックス移動**して横坑井 GR に整合させる（Z 非依存）。系統が異なるため PF と
blend するとアンサンブル効果が出る（参考解法 LB 7.168 の第2トラッカー）。

参考解法の BEAM_CONFIGS（14 設定）を平均してロバスト化する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# (beam_size, move_cost, emit_scale, savgol_radius)
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


def beam_search(
    hgr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    last_tvt: float,
    bs: int = 10,
    mc: float = 20.0,
    es: float = 144.0,
    r: int = 2,
) -> np.ndarray:
    """横坑井 GR 系列 hgr を typewell GR(TVT) に beam search で整合し TVT を返す。"""
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
        tot = bcost[:bn, None] + gr_e + mcost[None, :]
        tot = np.where(valid, tot, np.inf)

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


def beam_ensemble(hw: pd.DataFrame, tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """BEAM_CONFIGS を平均した beam 予測（ev 行）と ev_idx を返す。"""
    tw_s = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    ev_mask = hw["TVT_input"].isna().to_numpy()
    ev_idx = np.where(ev_mask)[0]
    kn = hw[hw["TVT_input"].notna()]
    if len(tw_s) < 2 or len(ev_idx) == 0 or len(kn) == 0:
        return np.empty(0), ev_idx
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].to_numpy(float)
    last_tvt = float(kn.iloc[-1]["TVT_input"])
    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy(float)
    hgr = gr_all[ev_idx]
    res = [
        beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r) for bs, mc, es, r in BEAM_CONFIGS
    ]
    return np.stack(res, 0).mean(0), ev_idx
