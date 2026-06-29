"""exp012: 投影後処理（per-well 低次多項式平滑）の検証。

参考解法は最終予測に低次 PROJECTION 後処理（degree4, blend0.75 で CV 改善を主張）を入れる。
ここでは exp009 の最終予測（hybrid + 0.4·CatBoost残差, rich特徴）に対し、坑井ごとに
idx の低次多項式を当ててブレンドし、CV が下がるか検証する。

前提: data/interim/exp007_ml_rows_v2.parquet
実行: uv run python experiments/exp012_projection_pp/train.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rogii import ensemble as ens  # noqa: E402
from rogii import io  # noqa: E402

# exp009 と同じ特徴
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
CAT_PARAMS = dict(
    loss_function="RMSE",
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    n_estimators=600,
    random_seed=42,
    verbose=0,
)
SHRINK = 0.4
DEGREES = [2, 3, 4]
BLENDS = [0.0, 0.25, 0.5, 0.75]


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


def project(df, col, degree, blend):
    """坑井ごとに idx_from_ps の degree 次多項式で col を平滑化しブレンド。"""
    out = np.empty(len(df))
    for _, idx in df.groupby("well", sort=False).groups.items():
        s = df.loc[idx]
        x = s["idx_from_ps"].to_numpy(float)
        yv = s[col].to_numpy(float)
        pos = df.index.get_indexer(idx)
        if len(x) <= degree + 1:
            out[pos] = yv
            continue
        xc = (x - x.mean()) / (x.std() + 1e-9)
        coef = np.polyfit(xc, yv, degree)
        smooth = np.polyval(coef, xc)
        out[pos] = (1 - blend) * yv + blend * smooth
    return out


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(io.project_root() / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("rogii-wellbore")
    df = pd.read_parquet(io.project_root() / "data/interim/exp007_ml_rows_v2.parquet").reset_index(
        drop=True
    )
    df = add_features(df)
    df["hybrid"] = ens.hybrid_from_cache(df)
    feats = BASE + NEW
    y = (df["tvt"] - df["hybrid"]).to_numpy()
    wells = df["well"].to_numpy()
    X = df[feats]

    print("CatBoost residual OOF ...")
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups=wells):
        m = CatBoostRegressor(**CAT_PARAMS)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict(X.iloc[va])
    df["final"] = df["hybrid"] + SHRINK * oof
    base_rmse = per_well_rmse(df, "final")
    print(f"exp009 final (no PP) = {base_rmse:.4f}")

    results = {"no_pp": base_rmse}
    for d, b in itertools.product(DEGREES, BLENDS):
        if b == 0.0:
            continue
        df["pp"] = project(df, "final", d, b)
        results[f"deg{d}_blend{b}"] = per_well_rmse(df, "pp")
    best = min(results, key=results.get)
    print("\nprojection PP results:")
    for k in sorted(results, key=results.get):
        print(f"  {k:16s} = {results[k]:.4f}{'  <-- best' if k == best else ''}")

    with mlflow.start_run(run_name="exp012_projection_pp"):
        mlflow.log_metric("cv_no_pp", base_rmse)
        mlflow.log_metric("cv_best_pp", results[best])
        mlflow.log_param("best_pp", best)

    (Path(__file__).parent / "cv_result.txt").write_text(
        "\n".join(f"{k}: {v:.4f}" for k, v in sorted(results.items(), key=lambda t: t[1]))
        + f"\nBEST: {best}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
