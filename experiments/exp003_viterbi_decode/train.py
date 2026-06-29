"""exp003: GR シーケンス整合（Viterbi 復号）+ carry_forward へのシュリンク。

exp001/002 の学び:
- carry_forward(CV 12.81 / LB 15.88)が強い。増分積分もオフセット回帰も未達。

本実験の方針（geosteering の本命）:
- 横坑井 GR を typewell の GR(TVT) に系列整合させて TVT を Viterbi 復号
  （`src/rogii/decode.py`）。PS 既知区間で GR を typewell にスケール整合（calib）し、
  始点 TVT[PS-1] を固定、なめらかさ（lam）と探索半径（radius）で拘束。
- 生の復号はノイズに引かれて CF に劣後するため、**CF へシュリンク**:
  `TVT = tvt0 + alpha·(viterbi − tvt0)`。alpha は fold 内のみで選択しリークを防ぐ。

- CV: 坑井単位 GroupKFold(5)。各 fold で alpha を train fold から選び valid fold で評価。
- 比較: carry_forward / viterbi(alpha=1) / blend(alpha*)。
- 記録: MLflow（experiment="rogii-wellbore"）。
- 出力: outputs/submissions/exp003_viterbi.csv

実行: uv run python experiments/exp003_viterbi_decode/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rogii import (
    decode,  # noqa: E402
    io,  # noqa: E402
)

EXP_NAME = "rogii-wellbore"
RUN_NAME = "exp003_viterbi_decode"
N_FOLDS = 5
DECODE_KW = dict(lam=20.0, radius=12.0, step=0.5, window=8, gr_win=15, calib_k=400)
ALPHA_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def well_rmse_alpha(pred_off: np.ndarray, true_off: np.ndarray, alpha: float) -> float:
    """blend 後 RMSE: error = alpha·pred_off − true_off。"""
    e = alpha * pred_off - true_off
    m = ~np.isnan(e)
    return float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else np.nan


def decode_train(wids: list[str]) -> dict:
    """各 train 坑井を復号し、PS 以降の (pred_off, true_off) を保存。"""
    out = {}
    for k, wid in enumerate(wids, 1):
        hw = io.load_horizontal("train", wid)
        tw = io.load_typewell("train", wid)
        ps = io.ps_index(hw)
        tvt = hw["TVT"].to_numpy().astype(float)
        ti = hw["TVT_input"].to_numpy().astype(float)
        last = float(ti[~np.isnan(ti)][-1])
        pred = decode.decode_well(hw, tw, ps, **DECODE_KW)
        out[wid] = dict(
            pred_off=(pred[ps:] - last),
            true_off=(tvt[ps:] - last),
            cf=rmse(np.full(tvt.shape, last)[ps:], tvt[ps:]),
        )
        if k % 100 == 0:
            print(f"  decoded {k}/{len(wids)}")
    return out


def best_alpha(dec: dict, wids: list[str]) -> tuple[float, float]:
    """wids 集合で平均 RMSE を最小化する alpha と、その RMSE。"""
    best_a, best_r = 0.0, np.inf
    for a in ALPHA_GRID:
        r = float(
            np.nanmean([well_rmse_alpha(dec[w]["pred_off"], dec[w]["true_off"], a) for w in wids])
        )
        if r < best_r:
            best_a, best_r = float(a), r
    return best_a, best_r


def main() -> None:
    sub_dir = io.project_root() / "outputs" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(io.project_root() / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXP_NAME)

    print("decoding train (Viterbi) ...")
    wids = io.list_well_ids("train")
    dec = decode_train(wids)

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = np.arange(len(wids))
    cf_folds, vit_folds, blend_folds, chosen_alphas = [], [], [], []

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({f"decode_{k}": v for k, v in DECODE_KW.items()})
        mlflow.log_param("n_folds", N_FOLDS)

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(wids, groups=groups), 1):
            tr_w = [wids[i] for i in tr_idx]
            va_w = [wids[i] for i in va_idx]
            a_star, _ = best_alpha(dec, tr_w)  # alpha は train fold のみで選択（リーク防止）
            cf = np.nanmean([dec[w]["cf"] for w in va_w])
            vit = np.nanmean(
                [well_rmse_alpha(dec[w]["pred_off"], dec[w]["true_off"], 1.0) for w in va_w]
            )
            bl = np.nanmean(
                [well_rmse_alpha(dec[w]["pred_off"], dec[w]["true_off"], a_star) for w in va_w]
            )
            cf_folds.append(float(cf))
            vit_folds.append(float(vit))
            blend_folds.append(float(bl))
            chosen_alphas.append(a_star)
            print(f"fold{fold}: cf={cf:.3f} viterbi={vit:.3f} blend(a={a_star})={bl:.3f}")

        cv = {
            "carry_forward": float(np.mean(cf_folds)),
            "viterbi": float(np.mean(vit_folds)),
            "blend": float(np.mean(blend_folds)),
        }
        for k, v in cv.items():
            mlflow.log_metric(f"cv_rmse_{k}", v)
        mlflow.log_param("alpha_per_fold", ",".join(map(str, chosen_alphas)))
        print("CV RMSE (ft):", {k: round(v, 3) for k, v in cv.items()})
        print("alpha/fold:", chosen_alphas)

        # 最終 alpha は全 train で選択 → test 復号 → 提出
        final_alpha, _ = best_alpha(dec, wids)
        mlflow.log_param("final_alpha", final_alpha)
        print(f"final alpha (all train) = {final_alpha}")

        rows = []
        for wid in io.list_well_ids("test"):
            hw = io.load_horizontal("test", wid)
            tw = io.load_typewell("test", wid)
            ps = io.ps_index(hw)
            ti = hw["TVT_input"].to_numpy().astype(float)
            last = float(ti[~np.isnan(ti)][-1])
            pred = decode.decode_well(hw, tw, ps, **DECODE_KW)
            blended = last + final_alpha * (pred - last)
            for i in range(ps, hw.height):
                rows.append({"id": f"{wid}_{i}", "tvt": float(blended[i])})
        sub = pl.DataFrame(rows)
        sub_path = sub_dir / "exp003_viterbi.csv"
        sub.write_csv(sub_path)
        mlflow.log_artifact(str(sub_path))
        mlflow.log_metric("submission_rows", sub.height)
        print(f"submission -> {sub_path} ({sub.height} rows)")

    (Path(__file__).parent / "cv_result.txt").write_text(
        "\n".join(f"{k}: {v:.4f}" for k, v in cv.items())
        + f"\nfinal_alpha: {final_alpha}\nalpha_per_fold: {chosen_alphas}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
