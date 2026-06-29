"""exp009: 残差モデルの特徴量拡充（PF 再計算なし、キャッシュ派生で高速）。

exp008（hybrid + CatBoost 残差, CV 8.027）の残差学習に、キャッシュ列から派生する
新特徴を追加して改善を狙う:
- pf_beam_diff = pf_off − beam_off（2 トラッカー不一致 = 不確実性の代理）
- dist2 / idx2 / frac_idx（距離の非線形・坑井内相対位置）
- pf_slope / pf_curv（PF 予測の局所勾配・曲率: ドリフト傾向）
- z_off（eval 先頭からの Z 変位）, abs_incl, gr_rmean_long, gr_rstd_long

base=hybrid 固定、残差を LGB / CatBoost で学習し shrink を CV 選択。

前提: data/interim/exp007_ml_rows_v2.parquet
実行: uv run python experiments/exp009_rich_features/train.py
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

EXP_NAME = "rogii-wellbore"
RUN_NAME = "exp009_rich_features"
CACHE = io.project_root() / "data" / "interim" / "exp007_ml_rows_v2.parquet"
ART_DIR = Path(__file__).parent / "artifacts"
N_FOLDS = 5

BASE_FEATURES = [
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
NEW_FEATURES = [
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
CAT_PARAMS = dict(
    loss_function="RMSE",
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    n_estimators=600,
    random_seed=42,
    verbose=0,
)
SHRINKS = [0.1, 0.25, 0.4, 0.5]


def per_well_rmse(df, col):
    return float(
        np.mean(
            [
                np.sqrt(np.mean((s[col].to_numpy() - s["tvt"].to_numpy()) ** 2))
                for _, s in df.groupby("well", sort=False)
            ]
        )
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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


def oof(model_fn, X, y, groups):
    o = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(X, y, groups=groups):
        m = model_fn()
        m.fit(X.iloc[tr], y[tr])
        o[va] = m.predict(X.iloc[va])
    return o


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(io.project_root() / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXP_NAME)
    df = pd.read_parquet(CACHE).reset_index(drop=True)
    df = add_features(df)
    df["hybrid"] = ens.hybrid_from_cache(df)
    feats = BASE_FEATURES + NEW_FEATURES
    print(f"loaded {len(df)} rows, {len(feats)} features")

    wells = df["well"].to_numpy()
    X = df[feats]
    y = (df["tvt"] - df["hybrid"]).to_numpy()

    print("LGB OOF ...")
    oof_lgb = oof(lambda: lgb.LGBMRegressor(**LGB_PARAMS), X, y, wells)
    print("CatBoost OOF ...")
    oof_cat = oof(lambda: CatBoostRegressor(**CAT_PARAMS), X, y, wells)

    base_hyb = per_well_rmse(df, "hybrid")
    combos = {}
    for name, o in [("lgb", oof_lgb), ("cat", oof_cat), ("lgb+cat", 0.5 * (oof_lgb + oof_cat))]:
        for sh in SHRINKS:
            df["p"] = df["hybrid"] + sh * o
            combos[f"{name}*{sh}"] = per_well_rmse(df, "p")
    best = min(combos, key=combos.get)

    print(f"\nhybrid base = {base_hyb:.3f}  (exp008 best combo was 8.027)")
    for k in sorted(combos, key=combos.get)[:8]:
        print(f"  hybrid+{k:12s} = {combos[k]:.3f}")
    print(f"BEST: hybrid+{best} = {combos[best]:.3f}")

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_param("features", ",".join(feats))
        mlflow.log_metric("cv_rmse_hybrid", base_hyb)
        mlflow.log_metric("cv_rmse_best", combos[best])
        mlflow.log_param("best_combo", best)

    learner = best.split("*")[0]
    shrink = float(best.split("*")[1])
    ART_DIR.mkdir(parents=True, exist_ok=True)
    final_lgb = lgb.LGBMRegressor(**LGB_PARAMS).fit(X, y)
    final_cat = CatBoostRegressor(**CAT_PARAMS).fit(X, y)
    joblib.dump(
        {
            "lgb": final_lgb,
            "cat": final_cat,
            "features": feats,
            "shrink": shrink,
            "learner": learner,
            "base": "hybrid",
        },
        ART_DIR / "combined_rich.joblib",
    )
    imp = sorted(zip(feats, final_cat.feature_importances_, strict=False), key=lambda t: -t[1])
    (Path(__file__).parent / "cv_result.txt").write_text(
        f"hybrid: {base_hyb:.4f}\n"
        + "\n".join(f"{k}: {v:.4f}" for k, v in sorted(combos.items(), key=lambda t: t[1])[:10])
        + f"\nBEST: hybrid+{best} = {combos[best]:.4f}\n"
        + "cat_importance:\n"
        + "\n".join(f"  {c}: {v:.1f}" for c, v in imp),
        encoding="utf-8",
    )
    print(f"saved -> {ART_DIR / 'combined_rich.joblib'}")
    print("cat importance top:", [c for c, _ in imp[:10]])


if __name__ == "__main__":
    main()
