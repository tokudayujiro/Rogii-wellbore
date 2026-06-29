"""exp005: PF の後処理チューニング（scale 選択 + savgol 平滑化 + CF blend）。

exp004 の PF（LB 8.895）をベースに、**1 回の PF 計算結果から複数 scale を合成**し、
savgol 平滑化と CF への blend を組み合わせて最良設定を探す（追加計算をほぼ増やさない）。

検証: train サブサンプル N_EVAL 坑井で平均 RMSE。並列 joblib threading。
グリッド: scale × savgol 窓 × alpha。

実行: uv run python experiments/exp005_pf_tuning/train.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rogii import io, pf  # noqa: E402

EXP_NAME = "rogii-wellbore"
RUN_NAME = "exp005_pf_tuning"
N_EVAL = 150
PF_KW = dict(n_seeds=64, n_particles=300)
N_JOBS = 4
SEED = 42

SCALES = [3.0, 5.0, 8.0, 12.0]
SG_WINDOWS = [1, 7, 15, 31, 61]  # 1 = 平滑化なし
ALPHAS = [0.8, 0.9, 1.0]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def eval_well(wid: str) -> dict | None:
    """PF を 1 回走らせ、各 scale の pred_off と true_off を返す。"""
    hw = io.load_horizontal("train", wid).to_pandas()
    tw = io.load_typewell("train", wid).to_pandas()
    ps = int(hw["TVT_input"].notna().sum())
    tvt = hw["TVT"].to_numpy(dtype=float)
    last = float(hw["TVT_input"].dropna().iloc[-1])
    preds, liks, ev_idx = pf.pf_allseeds(hw, tw, **PF_KW)
    if preds.size == 0:
        return None
    true_off = tvt[ev_idx] - last
    by_scale = {sc: pf.combine_scale(preds, liks, sc) - last for sc in SCALES}
    cf = rmse(np.full(tvt.shape, last)[ps:], tvt[ps:])
    return dict(by_scale=by_scale, true_off=true_off, cf=cf)


def apply_sg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) <= win:
        return x
    w = win if win % 2 == 1 else win + 1
    return savgol_filter(x, w, 2)


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(io.project_root() / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXP_NAME)

    wids_all = io.list_well_ids("train")
    rng = np.random.default_rng(SEED)
    wids = sorted(rng.choice(wids_all, size=min(N_EVAL, len(wids_all)), replace=False).tolist())

    hw0 = io.load_horizontal("train", wids[0]).to_pandas()
    tw0 = io.load_typewell("train", wids[0]).to_pandas()
    pf.lik_pf(hw0, tw0, n_seeds=4, n_particles=64)  # JIT warmup

    print(f"PF on {len(wids)} wells (parallel x{N_JOBS}) ...")
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS, backend="threading")(delayed(eval_well)(w) for w in wids)
    res = [r for r in res if r is not None]
    print(f"done in {time.time() - t0:.0f}s ({len(res)} wells)")

    cf_cv = float(np.nanmean([r["cf"] for r in res]))

    def grid_rmse(scale: float, sg: int, alpha: float) -> float:
        per = []
        for r in res:
            po = apply_sg(r["by_scale"][scale], sg)
            per.append(rmse(alpha * po, r["true_off"]))
        return float(np.nanmean(per))

    results = []
    for sc, sg, al in itertools.product(SCALES, SG_WINDOWS, ALPHAS):
        results.append((grid_rmse(sc, sg, al), sc, sg, al))
    results.sort()

    best_r, best_sc, best_sg, best_al = results[0]
    print(f"\ncarry_forward = {cf_cv:.3f}")
    print("top 10 configs (rmse, scale, sg_win, alpha):")
    for r, sc, sg, al in results[:10]:
        print(f"  {r:.3f}  scale={sc} sg={sg} alpha={al}")

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({f"pf_{k}": v for k, v in PF_KW.items()})
        mlflow.log_param("n_eval", len(res))
        mlflow.log_metric("cv_rmse_carry_forward", cf_cv)
        mlflow.log_metric("cv_rmse_best", best_r)
        mlflow.log_param("best_scale", best_sc)
        mlflow.log_param("best_sg_win", best_sg)
        mlflow.log_param("best_alpha", best_al)

    (Path(__file__).parent / "cv_result.txt").write_text(
        f"n_eval: {len(res)}\ncarry_forward: {cf_cv:.4f}\nbest_rmse: {best_r:.4f}\n"
        f"best_scale: {best_sc}\nbest_sg_win: {best_sg}\nbest_alpha: {best_al}\n"
        + "top10:\n"
        + "\n".join(f"  {r:.4f} scale={sc} sg={sg} alpha={al}" for r, sc, sg, al in results[:10])
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
