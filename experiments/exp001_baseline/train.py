"""exp001: ベースライン + LightGBM 増分(dTVT)モデル。

- 定式化: d[i]=TVT[i]-TVT[i-1] を予測し PS から累積復元
- CV: 坑井単位 GroupKFold(5)、各坑井の実 PS で本番設定を再現
- 比較: carry_forward / linear / lgb_delta
- 記録: MLflow（experiment="rogii-wellbore"）
- 出力: outputs/submissions/exp001_lgb.csv

実行: uv run python experiments/exp001_baseline/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rogii import features as F  # noqa: E402
from rogii import io  # noqa: E402

EXP_NAME = "rogii-wellbore"
RUN_NAME = "exp001_baseline"
N_FOLDS = 5
LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    n_estimators=400,
    verbose=-1,
)


def known_tvt_at_ps(hw: pl.DataFrame, ps: int) -> float:
    """PS-1 の TVT（= TVT_input の最終既知値）。"""
    ti = hw["TVT_input"].to_numpy().astype(float)
    known = ti[: ps if ps >= 1 else 1]
    known = known[~np.isnan(known)]
    return float(known[-1]) if known.size else 0.0


def reconstruct(tvt0: float, deltas: np.ndarray, ps: int, n: int) -> np.ndarray:
    """PS から deltas を累積して TVT を復元（i>=ps の区間）。"""
    out = np.full(n, np.nan)
    cur = tvt0
    for i in range(ps, n):
        cur = cur + deltas[i]
        out[i] = cur
    return out


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def load_split(split: str) -> dict:
    """split の全坑井を読み、坑井ごとに必要配列をまとめて返す。"""
    wells = {}
    for wid in io.list_well_ids(split):
        hw = io.load_horizontal(split, wid)
        ps = io.ps_index(hw)
        X, d, cols = F.build_well_features(hw, ps)
        wells[wid] = dict(
            X=X,
            d=d,
            cols=cols,
            ps=ps,
            n=hw.height,
            md=hw["MD"].to_numpy().astype(float),
            dmd=np.concatenate([[1.0], np.diff(hw["MD"].to_numpy().astype(float))]),
            tvt=hw["TVT"].to_numpy().astype(float) if "TVT" in hw.columns else None,
            tvt0=known_tvt_at_ps(hw, ps),
            last_slope=float(X[0, cols.index("last_slope")]),
        )
    return wells


def stack_rows(wells: dict, wids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """i>=ps の行を縦結合して (X, d) を返す（train 用、d 非 NaN のみ）。"""
    Xs, ds = [], []
    for wid in wids:
        w = wells[wid]
        sl = slice(w["ps"], w["n"])
        X = w["X"][sl]
        d = w["d"][sl]
        m = ~np.isnan(d)
        Xs.append(X[m])
        ds.append(d[m])
    return np.vstack(Xs), np.concatenate(ds)


def well_rmse(w: dict, deltas: np.ndarray, kind: str) -> float:
    """1 坑井の予測 TVT と真値の RMSE（i>=ps）。"""
    pred = reconstruct(w["tvt0"], deltas, w["ps"], w["n"])
    return rmse(pred[w["ps"] :], w["tvt"][w["ps"] :])


def make_deltas(w: dict, model: lgb.LGBMRegressor | None, kind: str) -> np.ndarray:
    n = w["n"]
    if kind == "carry_forward":
        return np.zeros(n)
    if kind == "linear":
        return w["last_slope"] * w["dmd"]
    # lgb
    return model.predict(w["X"])


def main() -> None:
    sub_dir = io.project_root() / "outputs" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    db = (io.project_root() / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    mlflow.set_experiment(EXP_NAME)

    print("loading train ...")
    train = load_split("train")
    wids = list(train.keys())
    cols = train[wids[0]]["cols"]

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = np.arange(len(wids))
    fold_scores = {"carry_forward": [], "linear": [], "lgb_delta": []}

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({f"lgb_{k}": v for k, v in LGB_PARAMS.items()})
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("features", ",".join(cols))
        mlflow.log_param("known_slope_k", F.KNOWN_SLOPE_K)

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(wids, groups=groups), 1):
            tr_w = [wids[i] for i in tr_idx]
            va_w = [wids[i] for i in va_idx]
            Xtr, ytr = stack_rows(train, tr_w)
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(Xtr, ytr)

            for kind in fold_scores:
                m = model if kind == "lgb_delta" else None
                scores = [well_rmse(train[w], make_deltas(train[w], m, kind), kind) for w in va_w]
                fold_scores[kind].append(float(np.nanmean(scores)))
            print(
                f"fold{fold}: cf={fold_scores['carry_forward'][-1]:.2f} "
                f"lin={fold_scores['linear'][-1]:.2f} lgb={fold_scores['lgb_delta'][-1]:.2f}"
            )

        cv = {}
        for kind, sc in fold_scores.items():
            cv[kind] = float(np.mean(sc))
            mlflow.log_metric(f"cv_rmse_{kind}", cv[kind])
        print("CV RMSE (ft):", {k: round(v, 3) for k, v in cv.items()})

        # 全 train で再学習 → test 推論 → 提出
        print("refit on all train + predict test ...")
        Xall, yall = stack_rows(train, wids)
        final = lgb.LGBMRegressor(**LGB_PARAMS)
        final.fit(Xall, yall)

        imp = sorted(zip(cols, final.feature_importances_, strict=False), key=lambda t: -t[1])
        mlflow.log_text(
            "\n".join(f"{c}\t{v}" for c, v in imp), "feature_importance.txt"
        )

        test = load_split("test")
        rows = []
        for wid, w in test.items():
            deltas = final.predict(w["X"])
            pred = reconstruct(w["tvt0"], deltas, w["ps"], w["n"])
            for i in range(w["ps"], w["n"]):
                rows.append({"id": f"{wid}_{i}", "tvt": float(pred[i])})
        sub = pl.DataFrame(rows)
        sub_path = sub_dir / "exp001_lgb.csv"
        sub.write_csv(sub_path)
        mlflow.log_artifact(str(sub_path))
        mlflow.log_metric("submission_rows", sub.height)
        print(f"submission -> {sub_path} ({sub.height} rows)")

    # README 用に CV を保存
    (Path(__file__).parent / "cv_result.txt").write_text(
        "\n".join(f"{k}: {v:.4f}" for k, v in cv.items()), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
