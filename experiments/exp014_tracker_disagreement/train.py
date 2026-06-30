"""exp014: 3トラッカー不一致特徴で残差をさらに改善（PF再計算なし）。

exp013（hybrid + CatBoost残差 + pfz_off, CV 7.890）に、PF/pf_z/beam の不一致や pf_z 軌跡の
派生を特徴として追加。3 トラッカーの食い違い = 局所的な不確実性の代理で、残差モデルが
「どこで base が外れやすいか」を学べる。LGB/CatBoost と blend も比較。

追加特徴: pf_pfz_diff, pfz_beam_diff, tracker_spread, pfz_slope, pfz_curv
前提: exp007_ml_rows_v2.parquet + exp013_pfz.parquet
実行: uv run python experiments/exp014_tracker_disagreement/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rogii import ensemble as ens  # noqa: E402
from rogii import io  # noqa: E402

BASE = [
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
NEW = [
    "pf_beam_diff",
    "dist2",
    "frac_idx",
    "pf_slope",
    "pf_curv",
    "z_off",
    "abs_incl",
    "gr_rmean_long",
    "gr_rstd_long",
]
TRK = ["pfz_off", "pf_pfz_diff", "pfz_beam_diff", "tracker_spread", "pfz_slope", "pfz_curv"]
LGB_P = dict(
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
CAT_P = dict(
    loss_function="RMSE",
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    n_estimators=600,
    random_seed=42,
    verbose=0,
)
SHRINKS = [0.4, 0.5, 0.6]


def per_well_rmse(df, col):
    return float(
        np.mean(
            [
                np.sqrt(np.mean((s[col].to_numpy() - s["tvt"].to_numpy()) ** 2))
                for _, s in df.groupby("well", sort=False)
            ]
        )
    )


def add_feats(df):
    df = df.copy()
    g = df.groupby("well", sort=False)
    df["pf_beam_diff"] = df["pf_off"] - df["beam_off"]
    df["dist2"] = df["dist_from_ps"] ** 2
    df["frac_idx"] = df["idx_from_ps"] / df["n_eval"].clip(lower=1.0)
    df["abs_incl"] = df["incl"].abs()
    df["z_off"] = df["z"] - g["z"].transform("first")
    df["pf_slope"] = (
        g["pf_pred"]
        .transform(lambda s: s.diff().rolling(15, center=True, min_periods=1).mean())
        .fillna(0.0)
    )
    df["pf_curv"] = (
        g["pf_pred"]
        .transform(lambda s: s.diff().diff().rolling(15, center=True, min_periods=1).mean())
        .fillna(0.0)
    )
    df["gr_rmean_long"] = g["gr"].transform(
        lambda s: s.rolling(101, center=True, min_periods=1).mean()
    )
    df["gr_rstd_long"] = g["gr"].transform(
        lambda s: s.rolling(101, center=True, min_periods=1).std().fillna(0.0)
    )
    # 3 トラッカー不一致
    df["pf_pfz_diff"] = df["pf_off"] - df["pfz_off"]
    df["pfz_beam_diff"] = df["pfz_off"] - df["beam_off"]
    df["tracker_spread"] = df[["pf_off", "pfz_off", "beam_off"]].std(axis=1)
    df["pfz_slope"] = (
        g["pfz_pred"]
        .transform(lambda s: s.diff().rolling(15, center=True, min_periods=1).mean())
        .fillna(0.0)
    )
    df["pfz_curv"] = (
        g["pfz_pred"]
        .transform(lambda s: s.diff().diff().rolling(15, center=True, min_periods=1).mean())
        .fillna(0.0)
    )
    return df


def oof(model_fn, X, y, groups):
    o = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups=groups):
        m = model_fn()
        m.fit(X.iloc[tr], y[tr])
        o[va] = m.predict(X.iloc[va])
    return o


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(io.project_root() / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("rogii-wellbore")
    inter = io.project_root() / "data" / "interim"
    df = pd.read_parquet(inter / "exp007_ml_rows_v2.parquet").merge(
        pd.read_parquet(inter / "exp013_pfz.parquet"), on=["well", "idx_from_ps"], how="left"
    )
    df["pfz_pred"] = df["pfz_pred"].fillna(df["pf_pred"])
    df["pfz_off"] = df["pfz_off"].fillna(df["pf_off"])
    df = add_feats(df).reset_index(drop=True)
    df["hybrid"] = ens.hybrid_from_cache(df)
    feats = BASE + NEW + TRK
    X = df[feats]
    y = (df["tvt"] - df["hybrid"]).to_numpy()
    wells = df["well"].to_numpy()
    print(f"{len(df)} rows, {len(feats)} features")

    print("LGB OOF ...")
    oof_lgb = oof(lambda: lgb.LGBMRegressor(**LGB_P), X, y, wells)
    print("CatBoost OOF ...")
    oof_cat = oof(lambda: CatBoostRegressor(**CAT_P), X, y, wells)

    res = {"exp013_ref(7.890)": 7.890}
    for nm, o in [("lgb", oof_lgb), ("cat", oof_cat), ("lgb+cat", 0.5 * (oof_lgb + oof_cat))]:
        for sh in SHRINKS:
            df["p"] = df["hybrid"] + sh * o
            res[f"{nm}*{sh}"] = per_well_rmse(df, "p")
    best = min(res, key=res.get)
    print("\nresults (CV per-well RMSE):")
    for k in sorted(res, key=res.get)[:10]:
        print(f"  {k:14s} = {res[k]:.4f}{'  <-- best' if k == best else ''}")

    with mlflow.start_run(run_name="exp014_tracker_disagreement"):
        mlflow.log_param("features", ",".join(feats))
        mlflow.log_metric("cv_best", res[best])
        mlflow.log_param("best", best)

    learner = best.split("*")[0]
    shrink = float(best.split("*")[1])
    final_lgb = lgb.LGBMRegressor(**LGB_P).fit(X, y)
    final_cat = CatBoostRegressor(**CAT_P).fit(X, y)
    art = Path(__file__).parent / "artifacts"
    art.mkdir(exist_ok=True)
    joblib.dump(
        {
            "lgb": final_lgb,
            "cat": final_cat,
            "features": feats,
            "shrink": shrink,
            "learner": learner if learner in ("lgb", "cat") else "cat",
            "blend": learner == "lgb+cat",
            "base": "hybrid",
        },
        art / "tracker.joblib",
    )
    imp = sorted(zip(feats, final_cat.feature_importances_, strict=False), key=lambda t: -t[1])
    (Path(__file__).parent / "cv_result.txt").write_text(
        "\n".join(f"{k}: {v:.4f}" for k, v in sorted(res.items(), key=lambda t: t[1]))
        + f"\nBEST: {best} = {res[best]:.4f}\n"
        + "cat_importance:\n"
        + "\n".join(f"  {c}: {v:.1f}" for c, v in imp[:15]),
        encoding="utf-8",
    )
    print(f"saved -> {art / 'tracker.joblib'}; cat top:", [c for c, _ in imp[:8]])


if __name__ == "__main__":
    main()
