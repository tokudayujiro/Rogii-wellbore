"""exp008: 合体モデル（hybrid selector + LGB残差, CatBoost 多様性も評価）。

これまでの最良要素を結合:
- base = per-well selector hybrid（PF scale 切替 + beam 混合 + CF hold; exp006, LB 8.711）
- 補正 = LGB が base 残差 `TVT − base` を幾何/GR/空間特徴で予測（exp007 の発展）
- final = base + shrink·resid

参考用に base=pf8 の残差（exp007 相当）や CatBoost 残差・LGB+Cat 平均も比較する。

前提: build_cache.py（全 PF scale 版）で exp007_ml_rows_v2.parquet を作成済み。
実行: uv run python experiments/exp008_combined/train.py
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
RUN_NAME = "exp008_combined"
CACHE = io.project_root() / "data" / "interim" / "exp007_ml_rows_v2.parquet"
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
CAT_PARAMS = dict(
    loss_function="RMSE",
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    n_estimators=600,
    random_seed=42,
    verbose=0,
)
SHRINKS = [0.0, 0.25, 0.5, 0.75, 1.0]


def per_well_rmse(df: pd.DataFrame, col: str) -> float:
    return float(
        np.mean(
            [
                np.sqrt(np.mean((s[col].to_numpy() - s["tvt"].to_numpy()) ** 2))
                for _, s in df.groupby("well", sort=False)
            ]
        )
    )


def hybrid_from_cache(df: pd.DataFrame) -> np.ndarray:
    """キャッシュの pf3/5/8/12・beam・n_eval・z_span から selector hybrid を再構成。"""
    out = np.empty(len(df))
    scale_col = {3.0: "pf3", 5.0: "pf5", 8.0: "pf_pred", 12.0: "pf12"}
    for _, idx in df.groupby("well", sort=False).groups.items():
        s = df.loc[idx]
        n_eval = float(s["n_eval"].iloc[0])
        z_span = float(s["z_span"].iloc[0])
        last = float(s["last"].iloc[0])
        code = int(n_eval > ens.SELECTOR_N_EVAL_THRESHOLD) + 2 * int(
            np.searchsorted(ens.SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right")
        )
        name = ens.SELECTOR_BIN_VARIANTS.get(code, ens.SELECTOR_GLOBAL_VARIANT)
        scale, beam_w, hold_w = ens.parse_selector_variant(name)
        base = s[scale_col[scale]].to_numpy()
        pred = (1 - beam_w) * base + beam_w * s["beam_pred"].to_numpy()
        pred = (1 - hold_w) * pred + hold_w * last
        out[df.index.get_indexer(idx)] = pred
    return out


def oof_predict(model_fn, X, y, groups):
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=N_FOLDS)
    for tr, va in gkf.split(X, y, groups=groups):
        m = model_fn()
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict(X.iloc[va])
    return oof


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(io.project_root() / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXP_NAME)
    df = pd.read_parquet(CACHE).reset_index(drop=True)
    print(f"loaded {len(df)} rows / {df['well'].nunique()} wells")

    df["hybrid"] = hybrid_from_cache(df)
    df["pf_alpha09"] = df["last"] + 0.9 * (df["pf_pred"] - df["last"])
    wells = df["well"].to_numpy()
    X = df[FEATURES]

    # 残差ターゲット（base = hybrid）
    y_hyb = (df["tvt"] - df["hybrid"]).to_numpy()
    print("LGB residual (base=hybrid) OOF ...")
    oof_lgb = oof_predict(lambda: lgb.LGBMRegressor(**LGB_PARAMS), X, y_hyb, wells)
    print("CatBoost residual (base=hybrid) OOF ...")
    oof_cat = oof_predict(lambda: CatBoostRegressor(**CAT_PARAMS), X, y_hyb, wells)
    oof_ens = 0.5 * (oof_lgb + oof_cat)

    base = {
        "carry_forward": per_well_rmse(df.assign(c=df["last"]), "c"),
        "pf8": per_well_rmse(df, "pf_pred"),
        "pf_alpha09": per_well_rmse(df, "pf_alpha09"),
        "hybrid": per_well_rmse(df, "hybrid"),
    }
    combos = {}
    for name, oof in [("lgb", oof_lgb), ("cat", oof_cat), ("lgb+cat", oof_ens)]:
        for sh in SHRINKS:
            df["p"] = df["hybrid"] + sh * oof
            combos[f"hybrid+{name}*{sh}"] = per_well_rmse(df, "p")
    best_key = min(combos, key=combos.get)

    print("\nCV per-well RMSE (ft):")
    for k, v in base.items():
        print(f"  {k:14s} = {v:.3f}")
    for k in sorted(combos, key=combos.get)[:8]:
        print(f"  {k:20s} = {combos[k]:.3f}")
    print(f"BEST: {best_key} = {combos[best_key]:.3f}")

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_param("features", ",".join(FEATURES))
        for k, v in base.items():
            mlflow.log_metric(f"cv_rmse_{k}", v)
        mlflow.log_metric("cv_rmse_best_combo", combos[best_key])
        mlflow.log_param("best_combo", best_key)

    # 最終モデル保存（base=hybrid, LGB+Cat 残差, best shrink）。kernel は Dataset で読む。
    best_learner = best_key.split("+")[1].split("*")[0]
    best_shrink = float(best_key.split("*")[1])
    ART_DIR.mkdir(parents=True, exist_ok=True)
    final_lgb = lgb.LGBMRegressor(**LGB_PARAMS).fit(X, y_hyb)
    final_cat = CatBoostRegressor(**CAT_PARAMS).fit(X, y_hyb)
    joblib.dump(
        {
            "lgb": final_lgb,
            "cat": final_cat,
            "features": FEATURES,
            "shrink": best_shrink,
            "learner": best_learner,
            "base": "hybrid",
        },
        ART_DIR / "combined.joblib",
    )
    (Path(__file__).parent / "cv_result.txt").write_text(
        "\n".join(f"{k}: {v:.4f}" for k, v in base.items())
        + "\n"
        + "\n".join(f"{k}: {v:.4f}" for k, v in sorted(combos.items(), key=lambda t: t[1])[:10])
        + f"\nBEST: {best_key} = {combos[best_key]:.4f}\n",
        encoding="utf-8",
    )
    print(
        f"\nsaved -> {ART_DIR / 'combined.joblib'} (learner={best_learner}, shrink={best_shrink})"
    )


if __name__ == "__main__":
    main()
