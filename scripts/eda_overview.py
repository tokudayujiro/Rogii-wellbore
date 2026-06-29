"""EDA: データ全体像の把握とサンプル坑井の可視化。

実行: uv run python scripts/eda_overview.py
出力: outputs/figures/eda/*.png, outputs/reports/eda_overview.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rogii import io  # noqa: E402

FIG_DIR = io.project_root() / "outputs" / "figures" / "eda"
REPORT = io.project_root() / "outputs" / "reports" / "eda_overview.md"


def collect_train_stats() -> pl.DataFrame:
    """全 train 坑井の要約統計を 1 行/坑井で集計する。"""
    rows = []
    ids = io.list_well_ids("train")
    for wid in ids:
        hw = io.load_horizontal("train", wid)
        tw = io.load_typewell("train", wid)
        n = hw.height
        ps = io.ps_index(hw)
        tvt = hw["TVT"].to_numpy()
        md = hw["MD"].to_numpy()
        # PS 以降の TVT 変動（carry-forward の難しさの目安）
        post = tvt[ps:] if ps < n else np.array([])
        tvt_at_ps = tvt[ps - 1] if 0 < ps <= n else np.nan
        post_drift = float(np.nanmax(np.abs(post - tvt_at_ps))) if post.size else 0.0
        rows.append(
            {
                "well_id": wid,
                "n_rows": n,
                "ps_index": ps,
                "ps_frac": ps / n if n else np.nan,
                "md_step": float(np.median(np.diff(md))) if n > 1 else np.nan,
                "gr_nan_rate": float(hw["GR"].is_null().mean()),
                "tvt_min": float(np.nanmin(tvt)),
                "tvt_max": float(np.nanmax(tvt)),
                "tvt_range": float(np.nanmax(tvt) - np.nanmin(tvt)),
                "post_ps_drift": post_drift,
                "tw_rows": tw.height,
                "tw_tvt_range": float(tw["TVT"].max() - tw["TVT"].min()),
            }
        )
    return pl.DataFrame(rows)


def plot_distributions(stats: pl.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(stats["ps_frac"].drop_nulls().to_numpy(), bins=40, color="steelblue")
    axes[0, 0].set_title("PS position (known fraction = ps_index/n_rows)")
    axes[0, 0].set_xlabel("ps_frac")

    axes[0, 1].hist(stats["post_ps_drift"].to_numpy(), bins=40, color="indianred")
    axes[0, 1].set_title("max |TVT - TVT@PS| after PS (carry-forward error scale)")
    axes[0, 1].set_xlabel("feet")

    axes[1, 0].hist(stats["gr_nan_rate"].to_numpy(), bins=40, color="seagreen")
    axes[1, 0].set_title("GR NaN rate per well")
    axes[1, 0].set_xlabel("nan rate")

    axes[1, 1].hist(stats["n_rows"].to_numpy(), bins=40, color="slateblue")
    axes[1, 1].set_title("horizontal well row count")
    axes[1, 1].set_xlabel("n_rows")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "dataset_distributions.png", dpi=110)
    plt.close(fig)


def plot_example_well(wid: str) -> None:
    hw = io.load_horizontal("train", wid)
    tw = io.load_typewell("train", wid)
    ps = io.ps_index(hw)
    md = hw["MD"].to_numpy()
    tvt = hw["TVT"].to_numpy()
    gr = hw["GR"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(md, tvt, lw=0.8)
    if 0 < ps <= len(md):
        axes[0].axvline(md[ps - 1], color="red", ls="--", label="PS")
        axes[0].legend()
    axes[0].set_title(f"{wid}: MD vs TVT")
    axes[0].set_xlabel("MD (ft)")
    axes[0].set_ylabel("TVT (ft)")

    axes[1].plot(md, gr, lw=0.6, color="darkgreen")
    if 0 < ps <= len(md):
        axes[1].axvline(md[ps - 1], color="red", ls="--")
    axes[1].set_title(f"{wid}: MD vs GR")
    axes[1].set_xlabel("MD (ft)")
    axes[1].set_ylabel("GR")

    axes[2].plot(tw["GR"].to_numpy(), tw["TVT"].to_numpy(), lw=0.8, color="black")
    axes[2].invert_yaxis()
    axes[2].set_title(f"{wid}: typewell GR vs TVT")
    axes[2].set_xlabel("GR")
    axes[2].set_ylabel("TVT (ft)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"example_{wid}.png", dpi=110)
    plt.close(fig)


def write_report(stats: pl.DataFrame, examples: list[str]) -> None:
    def q(col: str, p: float) -> float:
        return float(stats[col].quantile(p))

    lines = [
        "# EDA: データ全体像",
        "",
        f"- train 坑井数: **{stats.height}**",
        f"- 横坑井 行数: 中央値 {int(stats['n_rows'].median())} "
        f"(min {int(stats['n_rows'].min())} / max {int(stats['n_rows'].max())})",
        f"- MD ステップ: 中央値 {stats['md_step'].median():.3f} ft（≒1 ft 刻み）",
        f"- GR NaN 率: 中央値 {stats['gr_nan_rate'].median():.3%} "
        f"/ 95%点 {q('gr_nan_rate', 0.95):.3%}",
        "",
        "## PS 点（予測対象の難しさ）",
        f"- ps_frac（既知区間の割合）: 中央値 {stats['ps_frac'].median():.3f}"
        f"（= 坑井の約 {stats['ps_frac'].median() * 100:.0f}% が既知、残りを予測）",
        f"- PS 以降の |TVT−TVT@PS| 最大: 中央値 {stats['post_ps_drift'].median():.1f} ft "
        f"/ 95%点 {q('post_ps_drift', 0.95):.1f} ft",
        "  → carry-forward（PS の TVT を定数外挿）の誤差スケールの目安。",
        "",
        "## TVT レンジ",
        f"- 横坑井 TVT レンジ: 中央値 {stats['tvt_range'].median():.1f} ft",
        f"- typewell TVT レンジ: 中央値 {stats['tw_tvt_range'].median():.1f} ft",
        "",
        "## サンプル坑井の可視化",
        *[f"- ![{w}](../figures/eda/example_{w}.png)" for w in examples],
        "",
        "![分布](../figures/eda/dataset_distributions.png)",
        "",
        "## 所見（次アクション）",
        "- GR は typewell の GR–TVT 関係への相関キー。PS 以降は GR シグネチャ照合が本命。",
        "- carry-forward / 線形外挿をまずベースライン化し、RMSE の基準値を確定する。",
        "- ps_frac が小さい（既知が短い）坑井ほど難しい想定 → CV を坑井単位で層化。",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    stats = collect_train_stats()
    stats.write_csv(io.project_root() / "data" / "interim" / "train_well_stats.csv")

    plot_distributions(stats)
    examples = io.list_well_ids("train")[:3]
    for w in examples:
        plot_example_well(w)

    write_report(stats, examples)
    print(f"wells={stats.height}")
    for c in ["n_rows", "ps_frac", "post_ps_drift", "gr_nan_rate"]:
        print(f"  {c:14s} median={stats[c].median():.4f} p95={stats[c].quantile(0.95):.4f}")
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
