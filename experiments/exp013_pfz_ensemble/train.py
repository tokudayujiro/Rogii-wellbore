"""exp013: PF + Z速度連成PF アンサンブルを base に、残差スタッキング。

ゲートテスト: 0.7·PF + 0.3·PF_z が PF 単体を上回った（base 多様性）。ここでは
hybrid と pf_z をブレンドした base に exp009 の rich 残差（CatBoost）を載せて CV を測る。

前提: exp007_ml_rows_v2.parquet（pf/beam/特徴） + exp013_pfz.parquet（pf_z）
実行: uv run python experiments/exp013_pfz_ensemble/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
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
CAT = dict(
    loss_function="RMSE",
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    n_estimators=600,
    random_seed=42,
    verbose=0,
)
W_GRID = [0.6, 0.7, 0.8, 0.9, 1.0]  # base = w*hybrid + (1-w)*pf_z
SHRINKS = [0.25, 0.4, 0.5]


def add_features(df):
    df = df.copy()
    df["pf_beam_diff"] = df["pf_off"] - df["beam_off"]
    df["dist2"] = df["dist_from_ps"] ** 2
    df["frac_idx"] = df["idx_from_ps"] / df["n_eval"].clip(lower=1.0)
    df["abs_incl"] = df["incl"].abs()
    g = df.groupby("well", sort=False)
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
    return df


def per_well_rmse(df, col):
    return float(
        np.mean(
            [
                np.sqrt(np.mean((s[col].to_numpy() - s["tvt"].to_numpy()) ** 2))
                for _, s in df.groupby("well", sort=False)
            ]
        )
    )


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(io.project_root() / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("rogii-wellbore")
    inter = io.project_root() / "data" / "interim"
    df = pd.read_parquet(inter / "exp007_ml_rows_v2.parquet")
    pfz = pd.read_parquet(inter / "exp013_pfz.parquet")
    df = df.merge(pfz, on=["well", "idx_from_ps"], how="left")
    if df["pfz_pred"].isna().any():
        df["pfz_pred"] = df["pfz_pred"].fillna(df["pf_pred"])
        df["pfz_off"] = df["pfz_off"].fillna(df["pf_off"])
    df = add_features(df).reset_index(drop=True)
    df["hybrid"] = ens.hybrid_from_cache(df)

    # base 候補: w*hybrid + (1-w)*pf_z の raw RMSE
    print("raw base RMSE:")
    raw = {}
    for w in W_GRID:
        df["base"] = w * df["hybrid"] + (1 - w) * df["pfz_pred"]
        raw[w] = per_well_rmse(df, "base")
        print(f"  w={w}: {raw[w]:.3f}")
    best_w = min(raw, key=raw.get)
    df["base"] = best_w * df["hybrid"] + (1 - best_w) * df["pfz_pred"]
    print(f"best raw base: w={best_w} -> {raw[best_w]:.3f}")

    feats = BASE + NEW + ["pfz_off"]
    X = df[feats]
    y = (df["tvt"] - df["base"]).to_numpy()
    wells = df["well"].to_numpy()

    print("CatBoost residual OOF on ensemble base ...")
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups=wells):
        m = CatBoostRegressor(**CAT)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict(X.iloc[va])

    res = {
        "hybrid_only": per_well_rmse(df.assign(p=df["hybrid"]), "p"),
        "ens_base_raw": raw[best_w],
    }
    for sh in SHRINKS:
        df["p"] = df["base"] + sh * oof
        res[f"ens+resid*{sh}"] = per_well_rmse(df, "p")
    best = min(res, key=res.get)
    print("\nresults (exp009 best was 7.971):")
    for k in sorted(res, key=res.get):
        print(f"  {k:16s} = {res[k]:.4f}{'  <-- best' if k == best else ''}")

    with mlflow.start_run(run_name="exp013_pfz_ensemble"):
        mlflow.log_param("best_w", best_w)
        mlflow.log_metric("cv_best", res[best])
        mlflow.log_param("best_combo", best)

    # 最終モデル保存（ベスト shrink）
    best_sh = float(best.split("*")[1]) if "*" in best else 0.4
    final = CatBoostRegressor(**CAT).fit(X, y)
    art = Path(__file__).parent / "artifacts"
    art.mkdir(exist_ok=True)
    joblib.dump(
        {
            "cat": final,
            "features": feats,
            "shrink": best_sh,
            "w": float(best_w),
            "base": "hybrid+pfz",
            "learner": "cat",
        },
        art / "pfz_ensemble.joblib",
    )
    (Path(__file__).parent / "cv_result.txt").write_text(
        f"best_w: {best_w}\n"
        + "\n".join(f"{k}: {v:.4f}" for k, v in sorted(res.items(), key=lambda t: t[1]))
        + f"\nBEST: {best} = {res[best]:.4f}\nbest_shrink: {best_sh}\n",
        encoding="utf-8",
    )
    print(f"saved -> {art / 'pfz_ensemble.joblib'} (w={best_w}, shrink={best_sh})")


if __name__ == "__main__":
    main()
