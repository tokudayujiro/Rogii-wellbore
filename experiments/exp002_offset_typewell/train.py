"""exp002: PS からのオフセット直接回帰 + typewell GR–TVT 相関特徴。

exp001 の学び:
- 増分(ΔTVT)を積分する定式化はドリフトが累積し carry_forward(12.81)に劣後。

本実験の方針:
- ターゲットを `y[i] = TVT[i] - TVT[PS-1]`（PS からのオフセット）に変更し直接回帰。
  予測 TVT は `tvt0 + ŷ`。最低でも ŷ=0(=carry_forward) を学べる。
- typewell の GR(TVT) に横坑井 GR を相関させた `match_off` / `match_misfit` を特徴に追加。

- CV: 坑井単位 GroupKFold(5)、各坑井の実 PS で本番設定を再現。
- 比較: carry_forward / lgb_offset。
- 記録: MLflow（experiment="rogii-wellbore"）。
- 出力: outputs/submissions/exp002_offset.csv

実行: uv run python experiments/exp002_offset_typewell/train.py
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
RUN_NAME = "exp002_offset_typewell"
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


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def last_known_tvt(hw: pl.DataFrame, ps: int) -> float:
    ti = hw["TVT_input"].to_numpy().astype(float)
    known = ti[: ps if ps >= 1 else 1]
    known = known[~np.isnan(known)]
    return float(known[-1]) if known.size else 0.0


def load_split(split: str) -> dict:
    """split の全坑井を読み、オフセット定式化の配列をまとめて返す。"""
    wells = {}
    for wid in io.list_well_ids(split):
        hw = io.load_horizontal(split, wid)
        tw = io.load_typewell(split, wid)
        ps = io.ps_index(hw)
        X, y, cols = F.build_well_features_v2(hw, tw, ps)
        wells[wid] = dict(
            X=X,
            y=y,  # TVT - tvt0（train のみ実値）
            cols=cols,
            ps=ps,
            n=hw.height,
            tvt=hw["TVT"].to_numpy().astype(float) if "TVT" in hw.columns else None,
            tvt0=last_known_tvt(hw, ps),
        )
    return wells


def stack_rows(wells: dict, wids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """i>=ps の行を縦結合（train 用、y 非 NaN のみ）。"""
    Xs, ys = [], []
    for wid in wids:
        w = wells[wid]
        sl = slice(w["ps"], w["n"])
        X = w["X"][sl]
        y = w["y"][sl]
        m = ~np.isnan(y)
        Xs.append(X[m])
        ys.append(y[m])
    return np.vstack(Xs), np.concatenate(ys)


def predict_tvt(w: dict, model: lgb.LGBMRegressor | None, kind: str) -> np.ndarray:
    """坑井の予測 TVT 配列（全行、利用側で i>=ps を選択）。"""
    if kind == "carry_forward":
        return np.full(w["n"], w["tvt0"])
    off = model.predict(w["X"])
    return w["tvt0"] + off


def well_rmse(w: dict, model: lgb.LGBMRegressor | None, kind: str) -> float:
    pred = predict_tvt(w, model, kind)
    return rmse(pred[w["ps"] :], w["tvt"][w["ps"] :])


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
    fold_scores = {"carry_forward": [], "lgb_offset": []}

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({f"lgb_{k}": v for k, v in LGB_PARAMS.items()})
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("features", ",".join(cols))
        mlflow.log_param("match_radius", F.MATCH_RADIUS)
        mlflow.log_param("formulation", "offset(TVT-TVT[PS-1])")

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(wids, groups=groups), 1):
            tr_w = [wids[i] for i in tr_idx]
            va_w = [wids[i] for i in va_idx]
            Xtr, ytr = stack_rows(train, tr_w)
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(Xtr, ytr)

            for kind in fold_scores:
                m = model if kind == "lgb_offset" else None
                scores = [well_rmse(train[w], m, kind) for w in va_w]
                fold_scores[kind].append(float(np.nanmean(scores)))
            print(
                f"fold{fold}: cf={fold_scores['carry_forward'][-1]:.2f} "
                f"lgb={fold_scores['lgb_offset'][-1]:.2f}"
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
        mlflow.log_text("\n".join(f"{c}\t{v}" for c, v in imp), "feature_importance.txt")

        test = load_split("test")
        rows = []
        for wid, w in test.items():
            pred = predict_tvt(w, final, "lgb_offset")
            for i in range(w["ps"], w["n"]):
                rows.append({"id": f"{wid}_{i}", "tvt": float(pred[i])})
        sub = pl.DataFrame(rows)
        sub_path = sub_dir / "exp002_offset.csv"
        sub.write_csv(sub_path)
        mlflow.log_artifact(str(sub_path))
        mlflow.log_metric("submission_rows", sub.height)
        print(f"submission -> {sub_path} ({sub.height} rows)")

    (Path(__file__).parent / "cv_result.txt").write_text(
        "\n".join(f"{k}: {v:.4f}" for k, v in cv.items()), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
