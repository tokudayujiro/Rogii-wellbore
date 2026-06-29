"""exp011: PF プロセスノイズ/慣性のサブサンプル探索（base PF 底上げ狙い）。

PF の MOM/VN/PN を変えて raw PF(scale8) の平均 RMSE を 80 坑井で比較。base が良くなれば
残差スタックも含め全段に波及する。lik_pf に mom/vn/pn を明示渡し（スレッド安全）。

実行: uv run python experiments/exp011_pf_params/sweep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rogii import io, pf  # noqa: E402

N_EVAL = 80
SEED = 7
N_JOBS = 4
CONFIGS = {
    "baseline(.998/.002/.005)": dict(mom=0.998, vn=0.002, pn=0.005),
    "vn0.001": dict(mom=0.998, vn=0.001, pn=0.005),
    "vn0.004": dict(mom=0.998, vn=0.004, pn=0.005),
    "pn0.003": dict(mom=0.998, vn=0.002, pn=0.003),
    "pn0.010": dict(mom=0.998, vn=0.002, pn=0.010),
    "mom0.995": dict(mom=0.995, vn=0.002, pn=0.005),
    "mom0.999": dict(mom=0.999, vn=0.002, pn=0.005),
    "vn0.003/pn0.008": dict(mom=0.998, vn=0.003, pn=0.008),
}


def rmse(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def eval_cfg(hw, tw, ps, tvt, cfg):
    pred = pf.lik_pf(hw, tw, n_seeds=64, n_particles=300, scale=8.0, **cfg)
    return rmse(pred[ps:], tvt[ps:])


def main() -> None:
    rng = np.random.default_rng(SEED)
    wids = sorted(rng.choice(io.list_well_ids("train"), size=N_EVAL, replace=False).tolist())
    data = []
    for w in wids:
        hw = io.load_horizontal("train", w).to_pandas()
        tw = io.load_typewell("train", w).to_pandas()
        ps = int(hw["TVT_input"].notna().sum())
        data.append((hw, tw, ps, hw["TVT"].to_numpy(float)))
    pf.lik_pf(data[0][0], data[0][1], n_seeds=4, n_particles=64)  # warmup

    print(f"PF param sweep on {len(wids)} wells")
    results = {}
    for name, cfg in CONFIGS.items():
        t0 = time.time()
        scores = Parallel(n_jobs=N_JOBS, backend="threading")(
            delayed(eval_cfg)(hw, tw, ps, tvt, cfg) for hw, tw, ps, tvt in data
        )
        results[name] = float(np.nanmean(scores))
        print(f"  {name:26s} = {results[name]:.3f}  ({time.time() - t0:.0f}s)")

    best = min(results, key=results.get)
    print(f"\nBEST: {best} = {results[best]:.3f}")
    (Path(__file__).parent / "cv_result.txt").write_text(
        "\n".join(f"{k}: {v:.4f}" for k, v in sorted(results.items(), key=lambda t: t[1]))
        + f"\nBEST: {best}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
