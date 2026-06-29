"""exp007: PF 残差を LGB で学習し PF に上乗せ（スタッキング）。

参考解法は PF と ML スタックを blend している。ここでは PF が取り切れない系統誤差を、
test 時に利用可能な幾何/GR 特徴 + PF/beam 出力から LGB で予測し、`PF + shrink·resid` で補正。

- ターゲット: resid = TVT − PF_pred（予測対象行）。
- CV: 坑井単位 GroupKFold(5)。OOF で per-well RMSE を評価（リーク防止）。
- 比較: carry_forward / PF(scale8) / PF blend(α0.9) / PF + LGB残差(shrink)。

前提: experiments/exp007_ml_stack/build_cache.py で parquet を作成済み。
実行: uv run python experiments/exp007_ml_stack/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rogii import io  # noqa: E402

EXP_NAME = "rogii-wellbore"
RUN_NAME = "exp007_ml_stack"
CACHE = io.project_root() / "data" / "interim" / "exp007_ml_rows.parquet"
ART_DIR = Path(__file__).parent / "artifacts"
N_FOLDS = 5

FEATURES = [
    "z",
    "dist_from_ps",
    "idx_from_ps",
    "dmd",
    "incl",
    "gr",
    "gr_rmean",
    "gr_rstd",
    "gr_dev_known",
    "x_off",
    "y_off",
    "pf_off",
    "beam_off",
    "n_eval",
    "z_span",
    "known_gr_mean",
    "known_gr_std",
    "last_slope",
    "gr_sig",
]
LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    n_estimators=600,
    verbose=-1,
    n_jobs=-1,
)
SHRINKS = [0.0, 0.25, 0.5, 0.75, 1.0]


def per_well_rmse(df: pd.DataFrame, pred_col: str) -> float:
    g = df.groupby("well", sort=False)
    return float(
        np.mean(
            [np.sqrt(np.mean((s[pred_col].to_numpy() - s["tvt"].to_numpy()) ** 2)) for _, s in g]
        )
    )


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(io.project_root() / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXP_NAME)
    df = pd.read_parquet(CACHE)
    print(f"loaded {len(df)} rows / {df['well'].nunique()} wells")

    df["resid"] = df["tvt"] - df["pf_pred"]
    wells = df["well"].to_numpy()
    X = df[FEATURES]
    y = df["resid"].to_numpy()

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.zeros(len(df))
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=wells), 1):
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict(X.iloc[va])
        print(f"  fold{fold} done")

    df["pf_alpha09"] = df["last"] + 0.9 * (df["pf_pred"] - df["last"])
    base = {
        "carry_forward": per_well_rmse(df.assign(cf=df["last"]), "cf"),
        "pf": per_well_rmse(df, "pf_pred"),
        "pf_alpha09": per_well_rmse(df, "pf_alpha09"),
    }
    stack = {}
    for sh in SHRINKS:
        df["stk"] = df["pf_pred"] + sh * oof
        stack[sh] = per_well_rmse(df, "stk")
    best_sh = min(stack, key=stack.get)

    print("\nCV per-well RMSE (ft):")
    for k, v in base.items():
        print(f"  {k:14s} = {v:.3f}")
    for sh, v in stack.items():
        print(f"  pf+resid*{sh:<4} = {v:.3f}{'  <-- best' if sh == best_sh else ''}")

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({f"lgb_{k}": v for k, v in LGB_PARAMS.items()})
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("best_shrink", best_sh)
        for k, v in base.items():
            mlflow.log_metric(f"cv_rmse_{k}", v)
        mlflow.log_metric("cv_rmse_stack_best", stack[best_sh])

    # 全データで最終 LGB を学習し保存（shrink とともに）。kernel へは Dataset 経由で配布。
    ART_DIR.mkdir(parents=True, exist_ok=True)
    final = lgb.LGBMRegressor(**LGB_PARAMS)
    final.fit(X, y)
    joblib.dump(
        {"model": final, "features": FEATURES, "shrink": float(best_sh)},
        ART_DIR / "stack_lgb.joblib",
    )
    imp = sorted(zip(FEATURES, final.feature_importances_, strict=False), key=lambda t: -t[1])
    (Path(__file__).parent / "cv_result.txt").write_text(
        "\n".join(f"{k}: {v:.4f}" for k, v in base.items())
        + "\n"
        + "\n".join(f"pf+resid*{sh}: {v:.4f}" for sh, v in stack.items())
        + f"\nbest_shrink: {best_sh}\nbest_stack: {stack[best_sh]:.4f}\n"
        + "feat_importance:\n"
        + "\n".join(f"  {c}: {v}" for c, v in imp)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nsaved final model -> {ART_DIR / 'stack_lgb.joblib'} (shrink={best_sh})")
    print("feature importance:", imp[:8])


if __name__ == "__main__":
    main()
