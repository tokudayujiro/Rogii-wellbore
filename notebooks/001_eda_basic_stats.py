# %% [markdown]
# # 001 EDA: 基本統計量と可視化
# - **作成者**: yujirotokuda
# - **作成日**: 2026-06-14
# - **目的**: rogii-wellbore データ（横坑井 / 縦坑井）の構造・分布・欠損・相関を把握し、
#   TVT 予測モデル設計の前提を固める。
#
# 対象データ: `data/raw/train/` の 773 坑井（`<id>__horizontal_well.csv` / `__typewell.csv`）。
# raw データは不変・読み取り専用で扱う（`src/rogii/io.py` 経由でロード）。

# %% [markdown]
# ## 0. データの全体像（何のデータか）
#
# これは **geosteering（地質ステアリング）= 横坑井がいま地質的にどの深さを掘っているかを当てる** タスク。
# 1 つの坑井（well_id, 例 `000d7d20`）につき、以下が 1 セットで存在する。
#
# | ファイル | 何のデータか | 役割 |
# |----------|------------|------|
# | `<id>__horizontal_well.csv` | **横坑井**（水平に掘った坑井）を MD 刻みで測ったログ。1 行 = 1 測定点 | **主データ／予測対象**。`TVT` を当てる |
# | `<id>__typewell.csv`        | **縦坑井（typewell）**＝その地域の標準地質プロファイル（深さごとの GR） | **基準データ**。横坑井の GR をこれに照合する |
# | `<id>.png`                  | 上 2 つを重ねた参照画像 | 目視確認用 |
# | `sample_submission.csv`     | 提出フォーマット（`id`, `tvt`） | 答えの書式 |
#
# ### 横坑井 `horizontal_well.csv` の列
# | 列 | 意味 | 使える？ |
# |----|------|----------|
# | `MD` | 測長深度（坑井に沿った長さ, feet）。昇順 | ✓ 全区間既知 |
# | `X`, `Y`, `Z` | 各測定点の座標（Z は標高方向） | ✓ 全区間既知 |
# | `GR` | ガンマ線。地層を見分ける主シグナル（NaN あり） | ✓ 全区間既知（要補間） |
# | `TVT_input` | PS 点までの「既知の TVT」。PS 以降は NaN | ✓ 入力に使える |
# | `TVT` | **予測ターゲット**＝真垂直深度（地質基準の深さ） | ✗ PS 以降は未知（これを当てる） |
# | `ANCC` 〜 `BUDA` | 地質サーフェス（地層上端深度）。train のみ | △ test に無い→モデル入力には使えない |
#
# ### 縦坑井 `typewell.csv` の列
# | 列 | 意味 |
# |----|------|
# | `TVT` | 真垂直深度（全区間既知。横坑井の TVT と同じ物差し） |
# | `GR` | その深さでのガンマ線（地質の「お手本」プロファイル） |
# | `Geology` | 地層名（train でも多くが空） |
#
# ### キーと予測対象
# - 全ファイルは **`well_id` で紐づく**（横坑井・縦坑井・PNG が 1 セット）。
# - **PS 点** = `TVT_input` が埋まっている最終行の次。予測対象は **PS 点以降の `TVT`**。
# - `MD` は坑井に沿った距離、`TVT` は地質基準の垂直深度。横坑井は水平なので
#   「MD は進むが TVT はほぼ一定〜緩やかに上下」という関係になる（§5 の図参照）。

# %%
# --- imports（最初のセルに集約） ---
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import font_manager

# src レイアウトのパッケージを import できるよう保険でパスを通す（restartable に保つ）
_SRC = Path.cwd().parents[0] / "src" if (Path.cwd().name == "notebooks") else Path.cwd() / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rogii import io  # noqa: E402

# テーマを先に適用してから日本語フォントを設定する
# （sns.set_theme が font.family を上書きするため、順序が重要）。
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

# 日本語フォント有効化: japanize_matplotlib が使えれば利用、
# 不可（distutils 未提供の環境など）なら同梱の CJK フォントへフォールバック。
try:
    import japanize_matplotlib  # noqa: F401
except Exception:
    _available = {f.name for f in font_manager.fontManager.ttflist}
    for _font in ("Yu Gothic", "Meiryo", "Noto Sans JP", "MS Gothic", "BIZ UDGothic"):
        if _font in _available:
            plt.rcParams["font.family"] = _font
            break
    plt.rcParams["axes.unicode_minus"] = False

# %%
# --- config / paths ---
SPLIT = "train"
FIG_DIR = io.project_root() / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

well_ids = io.list_well_ids(SPLIT)
print(f"{SPLIT} 坑井数: {len(well_ids)}")
print(f"先頭5件: {well_ids[:5]}")

# %% [markdown]
# ## 1. データ概要
# 横坑井 CSV の列・行数・PS 点（既知 TVT の末尾）を 1 坑井で確認する。

# %%
sample_id = well_ids[0]
hw = io.load_horizontal(SPLIT, sample_id)  # 横坑井（主データ。TVT を予測する対象）
tw = io.load_typewell(SPLIT, sample_id)    # 縦坑井（基準データ。GR-TVT のお手本）
ps = io.ps_index(hw)                       # PS 点 = 既知 TVT の末尾の次（ここ以降が予測対象）

print(f"坑井 id: {sample_id}")
print(f"horizontal: {hw.shape[0]} 行 x {hw.shape[1]} 列  列={hw.columns}")
print(f"typewell  : {tw.shape[0]} 行 x {tw.shape[1]} 列  列={tw.columns}")
print(f"PS index = {ps}（既知 TVT={ps} 点 / 予測対象={hw.shape[0] - ps} 点）")
hw.head()

# %% [markdown]
# ## 2. 全坑井の基本統計量
# 全 773 坑井を結合し、数値列の記述統計とサイズ分布を見る。
# raw は変更せず、メモリ上で結合するだけ。

# %%
# 全坑井を結合（well_id 列を付与）。MD・GR・TVT など数値列を一括で集計する。
frames = []
sizes = []
for wid in well_ids:
    df = io.load_horizontal(SPLIT, wid)
    n = df.shape[0]
    ps_i = io.ps_index(df)
    sizes.append({"well_id": wid, "n_rows": n, "ps": ps_i, "n_target": n - ps_i})
    # 空列が坑井ごとに String 推論されるため、データ列は Float64 に統一する
    df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False) for c in df.columns)
    frames.append(df.with_columns(pl.lit(wid).alias("well_id")))

hw_all = pl.concat(frames)
sizes_df = pl.DataFrame(sizes)
print(f"結合後: {hw_all.shape[0]:,} 行 x {hw_all.shape[1]} 列")

# %%
# 結合した横坑井データの中身を確認（全 773 坑井ぶんの測定点を縦に積んだもの）
hw_all
# %%
# 数値列の記述統計量（polars describe）
# count=非NULL数, null_count=欠損数, mean/std/min/25%/50%/75%/max を列ごとに見る。
# TVT_input は PS 以降が NULL なので count が他列より大きく減る点に注目。
num_cols = ["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input",
            "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
num_cols = [c for c in num_cols if c in hw_all.columns]
hw_all.select(num_cols).describe()

# %%
# 坑井サイズの統計（行数・PS・予測対象点数）
sizes_df.select(["n_rows", "ps", "n_target"]).describe()

# %% [markdown]
# ## 3. 欠損値の確認
# GR は NaN を含みうる。TVT_input は PS 以降が NaN（仕様どおり）。

# %%
null_counts = hw_all.select(
    [pl.col(c).is_null().sum().alias(c) for c in hw_all.columns if c != "well_id"]
)
null_rate = hw_all.select(
    [(pl.col(c).is_null().mean() * 100).round(2).alias(c)
     for c in hw_all.columns if c != "well_id"]
)
print("欠損件数:")
print(null_counts.to_pandas().T.rename(columns={0: "n_null"}))
print("\n欠損率(%):")
print(null_rate.to_pandas().T.rename(columns={0: "null_rate_%"}))

# %% [markdown]
# ## 4. 可視化
# 以降の図は `outputs/figures/` にも保存する。

# %%
# 4-1. 主要数値列のヒストグラム（GR / TVT / MD / Z）
plot_cols = [c for c in ["GR", "TVT", "MD", "Z"] if c in hw_all.columns]
pdf = hw_all.select(plot_cols).to_pandas()

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for ax, col in zip(axes.ravel(), plot_cols):
    sns.histplot(pdf[col].dropna(), bins=60, ax=ax, kde=False)
    ax.set_title(f"{col} の分布")
    ax.set_xlabel(col)
    ax.set_ylabel("頻度")
fig.suptitle(f"主要数値列の分布（{SPLIT}, {len(well_ids)} 坑井, n={len(pdf):,}）", fontsize=18)
fig.savefig(FIG_DIR / "eda_hist_numeric.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

# %%
# 4-2. 坑井ごとの行数・予測対象点数の分布
spdf = sizes_df.to_pandas()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
sns.histplot(spdf["n_rows"], bins=40, ax=axes[0])
axes[0].set_title("坑井あたりの行数")
axes[0].set_xlabel("行数")
axes[0].set_ylabel("坑井数")
sns.histplot(spdf["n_target"], bins=40, ax=axes[1], color="tab:orange")
axes[1].set_title("坑井あたりの予測対象点数（PS 以降）")
axes[1].set_xlabel("予測対象点数")
axes[1].set_ylabel("坑井数")
fig.suptitle(f"坑井サイズの分布（{len(well_ids)} 坑井）", fontsize=18)
fig.savefig(FIG_DIR / "eda_well_sizes.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

# %%
# 4-3. 既知割合（PS / n_rows）の分布 — どれだけ既知 TVT があるか
known_ratio = (spdf["ps"] / spdf["n_rows"]).rename("known_ratio")
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
sns.histplot(known_ratio, bins=40, ax=ax)
ax.set_title(f"既知 TVT 割合 (PS / 全行) の分布  中央値={known_ratio.median():.2f}")
ax.set_xlabel("既知割合")
ax.set_ylabel("坑井数")
fig.savefig(FIG_DIR / "eda_known_ratio.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

# %%
# 4-4. 数値列の相関ヒートマップ（地質サーフェスは座標と強相関のはず）
corr_cols = [c for c in ["MD", "X", "Y", "Z", "GR", "TVT",
                         "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
             if c in hw_all.columns]
corr = hw_all.select(corr_cols).to_pandas().corr()
fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("数値列の相関（Pearson）")
fig.savefig(FIG_DIR / "eda_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

# %%
# 4-5. GR と TVT の関係 = このタスクの肝。
#   「GR（測れる）から TVT（当てたい）をどこまで言えるか」を見るための散布図。
#   点が多い（数百万）ので 1% にサンプリングして描画する。
gt = hw_all.select(["GR", "TVT"]).drop_nulls().to_pandas()
gt_s = gt.sample(frac=0.01, random_state=42) if len(gt) > 50_000 else gt
fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
sns.scatterplot(data=gt_s, x="GR", y="TVT", s=8, alpha=0.3, ax=ax)
ax.set_title(f"GR vs TVT（1% サンプル, n={len(gt_s):,}）")
ax.set_xlabel("GR (ガンマ線)")
ax.set_ylabel("TVT (真垂直深度, feet)")
fig.savefig(FIG_DIR / "eda_gr_vs_tvt.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

# %% [markdown]
# ## 5. 1 坑井の軌跡プロファイル
# 単一坑井で MD に沿った TVT 軌跡（既知 / 予測対象を色分け）と GR プロファイルを確認する。

# %%
md = hw["MD"].to_numpy()
tvt = hw["TVT"].to_numpy()
gr = hw["GR"].to_numpy()

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
axes[0].plot(md[:ps], tvt[:ps], color="tab:blue", label="既知 (TVT_input)")
axes[0].plot(md[ps:], tvt[ps:], color="tab:red", label="予測対象 (PS 以降)")
axes[0].axvline(md[ps], color="gray", ls="--", lw=1)
axes[0].invert_yaxis()  # 深度は下向きが大きい
axes[0].set_ylabel("TVT (feet)")
axes[0].set_title(f"坑井 {sample_id} の TVT 軌跡")
axes[0].legend(loc="upper right")

axes[1].plot(md, gr, color="tab:green", lw=0.8)
axes[1].axvline(md[ps], color="gray", ls="--", lw=1, label="PS 点")
axes[1].set_xlabel("MD (測長深度, feet)")
axes[1].set_ylabel("GR")
axes[1].set_title("GR プロファイル")
axes[1].legend(loc="upper right")
fig.savefig(FIG_DIR / f"eda_well_profile_{sample_id}.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

# %% [markdown]
# ## 6. まとめ・次のステップ
# - 坑井数 773、列構成・PS 点・予測対象点数を確認した。
# - GR は NaN を含む（特徴量化では補間が必要）。TVT_input は PS 以降が仕様どおり欠損。
# - 地質サーフェス列（ANCC 等）は座標 Z と強相関の想定 → 相関ヒートマップで確認。
# - 次: GR 相関ベースの増分（dip）予測モデルを GroupKFold（坑井単位）で検証する。
#   （※ data-catalog.md 記載の test リークは exploit しない方針）

# %%
# %%
