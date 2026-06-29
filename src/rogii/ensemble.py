"""PF + beam + carry_forward の per-well セレクタ・ハイブリッド（参考解法 LB 7.168 準拠）。

坑井ごとに `n_eval`（予測点数）と `z_span`（予測区間の Z 幅）でビン分けし、ビンに応じて
「どの PF scale を使うか / beam をどれだけ混ぜるか / carry_forward へどれだけ寄せるか(hold)」
を切り替える。閾値・変種は参考解法が 773 坑井 OOF で調整した定数（n_eval/z_span は test でも
取得可能なのでリークなし）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rogii import beam as beam_mod
from rogii import pf as pf_mod

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


def selector_well_code(hw: pd.DataFrame) -> tuple[int, str, float, float]:
    ev = hw["TVT_input"].isna().to_numpy()
    n_eval = float(ev.sum())
    z = hw.loc[ev, "Z"].to_numpy(float)
    z_span = float(np.nanmax(z) - np.nanmin(z)) if len(z) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right"))
    code = n_bin + 2 * z_bin
    return code, SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT), n_eval, z_span


def parse_selector_variant(name: str) -> tuple[float, float, float]:
    parts = name.split("_")
    scale = float(parts[2])
    beam_w = float(parts[parts.index("beam") + 1]) if "beam" in parts else 0.0
    hold_w = float(parts[parts.index("hold") + 1]) if "hold" in parts else 0.0
    return scale, beam_w, hold_w


def apply_selector_variant(
    name: str, pf_by_scale: dict[str, np.ndarray], tvt_beam: np.ndarray, last_tvt: float
) -> np.ndarray:
    scale, beam_w, hold_w = parse_selector_variant(name)
    base = pf_by_scale.get(f"pf_scale_{scale:g}")
    if base is None:
        base = next(iter(pf_by_scale.values()))
    pred = (1.0 - beam_w) * base + beam_w * tvt_beam
    pred = (1.0 - hold_w) * pred + hold_w * last_tvt
    return pred


def hybrid_from_cache(df):
    """キャッシュ（pf3/pf5/pf_pred(=pf8)/pf12, beam_pred, n_eval, z_span, last）から
    selector hybrid を行ごとに再構成して返す（experiments 用ユーティリティ）。"""
    import numpy as _np

    out = _np.empty(len(df))
    scale_col = {3.0: "pf3", 5.0: "pf5", 8.0: "pf_pred", 12.0: "pf12"}
    for _, idx in df.groupby("well", sort=False).groups.items():
        s = df.loc[idx]
        n_eval = float(s["n_eval"].iloc[0])
        z_span = float(s["z_span"].iloc[0])
        last = float(s["last"].iloc[0])
        code = int(n_eval > SELECTOR_N_EVAL_THRESHOLD) + 2 * int(
            _np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right")
        )
        name = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
        scale, beam_w, hold_w = parse_selector_variant(name)
        base = s[scale_col[scale]].to_numpy()
        pred = (1 - beam_w) * base + beam_w * s["beam_pred"].to_numpy()
        pred = (1 - hold_w) * pred + hold_w * last
        out[df.index.get_indexer(idx)] = pred
    return out


def hybrid_predict(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    n_particles: int = pf_mod.N_PARTICLES,
    n_seeds: int = pf_mod.N_SEEDS,
) -> np.ndarray:
    """PF(multi-scale) + beam + selector のハイブリッド予測（全長 n、既知部 NaN）。"""
    out = np.full(hw.shape[0], np.nan)
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) == 0:
        return out
    last_tvt = float(kn.iloc[-1]["TVT_input"])

    preds, liks, ev_idx = pf_mod.pf_allseeds(hw, tw, n_particles=n_particles, n_seeds=n_seeds)
    if preds.size == 0:
        out[hw["TVT_input"].isna().to_numpy()] = last_tvt
        return out
    pf_by_scale = {
        f"pf_scale_{sc:g}": pf_mod.combine_scale(preds, liks, sc) for sc in SELECTOR_SCALES
    }
    beam_ev, beam_idx = beam_mod.beam_ensemble(hw, tw)
    if beam_ev.size == 0 or not np.array_equal(beam_idx, ev_idx):
        beam_ev = pf_by_scale["pf_scale_8"]  # フォールバック（beam 不可）

    _, variant, _, _ = selector_well_code(hw)
    out[ev_idx] = apply_selector_variant(variant, pf_by_scale, beam_ev, last_tvt)
    return out
