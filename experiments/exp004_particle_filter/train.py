"""exp004: 尤度重み付き粒子フィルタ（PF）による TVT トラッキング。

参考解法（LB 7.168）の中核を非リークで再実装（`src/rogii/pf.py`）。exp003 の Viterbi は
フラット仮定からの微修正にとどまったが、PF は **dip rate を慣性付きで追従**し GR 尤度で
補正するため、地層の傾きトレンドを辿れる。

検証:
- PF は坑井ごと独立・学習なし（決定的）なので fold 不要。**train のサブサンプルで平均 RMSE**
  を評価（速度のため n_seeds=64/n_particles=300。最終提出は 128/500）。
- 比較: carry_forward / PF(scale=8) / PF を CF へ blend。
- 並列: joblib threading（numba nogil）。

実行: uv run python experiments/exp004_particle_filter/train.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlflow
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rogii import io, pf  # noqa: E402

EXP_NAME = "rogii-wellbore"
RUN_NAME = "exp004_particle_filter"
N_EVAL = 200  # 検証に使う train 坑井数（サブサンプル）
EVAL_KW = dict(n_seeds=64, n_particles=300, scale=8.0)
N_JOBS = 4
SEED = 42
ALPHA_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def eval_well(wid: str) -> tuple[np.ndarray, np.ndarray, float]:
    """1 坑井を PF 推定。(pf_off, true_off, cf_rmse) を返す（i>=ps）。"""
    hw = io.load_horizontal("train", wid).to_pandas()
    tw = io.load_typewell("train", wid).to_pandas()
    ps = int(hw["TVT_input"].notna().sum())
    tvt = hw["TVT"].to_numpy(dtype=float)
    last = float(hw["TVT_input"].dropna().iloc[-1])
    pred = pf.lik_pf(hw, tw, **EVAL_KW)
    cf = rmse(np.full(tvt.shape, last)[ps:], tvt[ps:])
    return pred[ps:] - last, tvt[ps:] - last, cf


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(io.project_root() / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXP_NAME)

    wids_all = io.list_well_ids("train")
    rng = np.random.default_rng(SEED)
    wids = sorted(rng.choice(wids_all, size=min(N_EVAL, len(wids_all)), replace=False).tolist())

    # JIT ウォームアップ（計測から除外）
    hw0 = io.load_horizontal("train", wids[0]).to_pandas()
    tw0 = io.load_typewell("train", wids[0]).to_pandas()
    pf.lik_pf(hw0, tw0, n_seeds=4, n_particles=64)

    print(f"evaluating PF on {len(wids)} wells (parallel x{N_JOBS}) ...")
    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, backend="threading")(delayed(eval_well)(wid) for wid in wids)
    dt = time.time() - t0
    print(f"done in {dt:.0f}s ({dt / len(wids):.2f}s/well)")

    cf_rmses = np.array([r[2] for r in results])

    def blend_rmse(alpha: float) -> float:
        per = [rmse(alpha * po, to) for po, to, _ in results]
        return float(np.nanmean(per))

    cf_cv = float(np.nanmean(cf_rmses))
    pf_cv = blend_rmse(1.0)
    best_a = min(ALPHA_GRID, key=blend_rmse)
    blend_cv = blend_rmse(float(best_a))

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({f"pf_{k}": v for k, v in EVAL_KW.items()})
        mlflow.log_param("n_eval", len(wids))
        mlflow.log_param("seed", SEED)
        mlflow.log_metric("cv_rmse_carry_forward", cf_cv)
        mlflow.log_metric("cv_rmse_pf", pf_cv)
        mlflow.log_metric("cv_rmse_blend", blend_cv)
        mlflow.log_param("best_alpha", float(best_a))
        mlflow.log_metric("eval_sec_per_well", dt / len(wids))

    print(f"CV RMSE (ft, {len(wids)} wells subsample):")
    print(f"  carry_forward = {cf_cv:.3f}")
    print(f"  PF(scale=8)   = {pf_cv:.3f}")
    print(f"  PF blend(a={best_a}) = {blend_cv:.3f}")

    (Path(__file__).parent / "cv_result.txt").write_text(
        f"n_eval: {len(wids)}\ncarry_forward: {cf_cv:.4f}\npf: {pf_cv:.4f}\n"
        f"blend: {blend_cv:.4f}\nbest_alpha: {best_a}\nsec_per_well: {dt / len(wids):.2f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
