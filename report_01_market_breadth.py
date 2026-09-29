"""生成模块 1「全 A 温度计」PDF 报告。

报告是一页横向总览：顶部为当日 KPI，中部为 6 张广度/成交额图，底部为四行
固定格式的当日读数。它只回答“市场内部是否支持价格信号”，不生成买卖指令。

示例：

    python report_01_market_breadth.py \
        --input data/market_breadth.csv \
        --output data/pdf/report_01_market_breadth.pdf \
        --date 20260924
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator


REQUIRED_COLUMNS = {
    "trade_date",
    "sample_count",
    "total_amount_100m",
    "up_count",
    "down_count",
    "flat_count",
    "up_ratio",
    "new_high20_count",
    "new_low20_count",
    "above_ma20_ratio",
    "above_ma60_ratio",
    "above_ma120_ratio",
    "up_amount_ratio",
    "amount_ratio_20",
    "breadth_ma60_change_pp",
}

TEXT = "#243447"
MUTED = "#6B7A90"
GRID = "#DCE3EC"
PANEL = "#FFFFFF"
PAGE = "#F5F7FA"
RED = "#D94C4C"  # A 股约定：红色代表上涨/偏强
GREEN = "#2E9B66"  # A 股约定：绿色代表下跌/偏弱
BLUE = "#4C78A8"
ORANGE = "#E08E3C"
PURPLE = "#8064A2"
FONT_CANDIDATES = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"


def configure_font() -> None:
    """尽量选择 macOS 中文字体；没有中文字体时仍允许脚本生成 PDF。"""
    for path in FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            break
        except Exception:  # pragma: no cover - 由本机字体环境决定
            continue
    plt.rcParams["axes.unicode_minus"] = False


def load_data(path: str | Path) -> pd.DataFrame:
    """读取市场广度 CSV，并检查模块 1 的完整字段。"""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV 缺少模块 1 字段：{sorted(missing)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    if frame["trade_date"].isna().any():
        raise ValueError("CSV 中存在无法解析的 trade_date")
    numeric = REQUIRED_COLUMNS - {"trade_date"}
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("trade_date").drop_duplicates("trade_date").reset_index(drop=True)


def choose_asof(frame: pd.DataFrame, date_arg: str | None) -> pd.Timestamp:
    if not date_arg:
        return frame["trade_date"].max()
    try:
        requested = pd.to_datetime(date_arg, format="%Y%m%d")
    except ValueError as exc:
        raise ValueError("--date 必须使用 YYYYMMDD 格式") from exc
    available = frame.loc[frame["trade_date"] == requested, "trade_date"]
    if available.empty:
        raise ValueError(f"CSV 中没有交易日：{date_arg}")
    return requested


def history_until(
    frame: pd.DataFrame, asof: pd.Timestamp, lookback: int | None
) -> pd.DataFrame:
    result = frame.loc[frame["trade_date"] <= asof].copy()
    if result.empty:
        raise ValueError(f"{asof:%Y-%m-%d} 之前没有可用数据")
    if lookback is not None:
        if lookback < 20:
            raise ValueError("--lookback 至少为 20")
        result = result.tail(lookback)
    return result


def _pct(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}%}"


def _number(value: object, digits: int = 0) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def _amount(value: object) -> str:
    if pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 10_000:
        return f"{value / 10_000:.2f} 万亿"
    return f"{value:,.0f} 亿"


def _status_color(value: object, kind: str) -> str:
    if pd.isna(value):
        return MUTED
    value = float(value)
    if kind == "amount_ratio":
        return GREEN if value >= 1.0 else RED
    if kind in {"ratio", "ma120"}:
        return GREEN if value >= 0.5 else RED
    return GREEN


def add_kpi(
    figure: Figure,
    position: tuple[float, float, float, float],
    label: str,
    value: str,
    color: str,
) -> None:
    ax = figure.add_axes(position)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#E3E8EF")
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.08, 0.76, label, transform=ax.transAxes, fontsize=8.3, color=MUTED)
    ax.text(0.08, 0.28, value, transform=ax.transAxes, fontsize=14, fontweight="bold", color=color)


def style_axis(ax: Axes, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=TEXT, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    ax.tick_params(axis="both", colors=MUTED, labelsize=7, length=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))


def percent_axis(ax: Axes, ymin: float = 0.0, ymax: float = 1.0) -> None:
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))


def draw_amount_chart(ax: Axes, history: pd.DataFrame) -> None:
    style_axis(ax, "图 1  两市成交额 + 20 日量比")
    dates = history["trade_date"]
    ax.bar(dates, history["total_amount_100m"], width=1.8, color=BLUE, alpha=0.72, label="成交额（亿元）")
    ax.set_ylabel("亿元", fontsize=7, color=MUTED)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    twin = ax.twinx()
    twin.plot(dates, history["amount_ratio_20"], color=ORANGE, linewidth=1.7, label="量比")
    for level, color in ((0.8, "#C7A36A"), (1.0, "#7C8796"), (1.2, "#C7A36A")):
        twin.axhline(level, color=color, linestyle="--" if level != 1.0 else "-", linewidth=0.8)
    max_ratio = history["amount_ratio_20"].max(skipna=True)
    if pd.isna(max_ratio):
        max_ratio = 1.5
    twin.set_ylim(0, max(1.5, float(max_ratio) * 1.12))
    twin.tick_params(axis="y", colors=MUTED, labelsize=7)
    twin.set_ylabel("量比", fontsize=7, color=MUTED)
    ax.legend(loc="upper left", fontsize=6.5, frameon=False)
    twin.legend(loc="upper right", fontsize=6.5, frameon=False)


def draw_updown_chart(ax: Axes, history: pd.DataFrame) -> None:
    style_axis(ax, "图 2  涨跌家数 + 上涨占比")
    dates = history["trade_date"]
    ax.bar(dates, history["up_count"], width=1.8, color=RED, alpha=0.82, label="上涨")
    ax.bar(dates, history["down_count"], width=1.8, bottom=history["up_count"], color=GREEN, alpha=0.82, label="下跌")
    ax.bar(
        dates,
        history["flat_count"],
        width=1.8,
        bottom=history["up_count"] + history["down_count"],
        color="#AAB4C0",
        alpha=0.75,
        label="平盘",
    )
    twin = ax.twinx()
    twin.plot(dates, history["up_ratio"], color=ORANGE, linewidth=1.7, label="上涨占比")
    for level in (0.3, 0.5, 0.7):
        twin.axhline(level, color=GRID, linestyle="--", linewidth=0.7)
    percent_axis(twin)
    twin.tick_params(axis="y", colors=MUTED, labelsize=7)
    twin.set_ylabel("上涨占比", fontsize=7, color=MUTED)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.legend(loc="upper left", fontsize=6.2, frameon=False, ncol=3)
    twin.legend(loc="upper right", fontsize=6.5, frameon=False)


def draw_price_volume_chart(ax: Axes, history: pd.DataFrame) -> None:
    style_axis(ax, "图 3  价量广度对照")
    ax.plot(history["trade_date"], history["up_ratio"], color=RED, linewidth=1.8, label="上涨家数占比")
    ax.plot(history["trade_date"], history["up_amount_ratio"], color=BLUE, linewidth=1.8, label="上涨成交额占比")
    ax.fill_between(
        history["trade_date"],
        history["up_ratio"].to_numpy(dtype=float),
        history["up_amount_ratio"].to_numpy(dtype=float),
        where=(history["up_ratio"].to_numpy(dtype=float) >= history["up_amount_ratio"].to_numpy(dtype=float)),
        color=RED,
        alpha=0.08,
    )
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color=GRID, linestyle=":", linewidth=0.8)
    percent_axis(ax)
    ax.legend(loc="upper left", fontsize=6.8, frameon=False)


def draw_new_high_low_chart(ax: Axes, history: pd.DataFrame) -> None:
    style_axis(ax, "图 4  20 日新高 - 新低")
    valid = (history["new_high20_count"].fillna(0) + history["new_low20_count"].fillna(0)) > 0
    plotted = history.loc[valid]
    if plotted.empty:
        ax.text(0.5, 0.5, "暖机期：暂时没有有效新高/新低数据", ha="center", va="center", color=MUTED, fontsize=8, transform=ax.transAxes)
        ax.set_xticks([])
        return
    dates = plotted["trade_date"]
    ax.bar(dates, plotted["new_high20_count"], width=1.8, color=RED, alpha=0.72, label="新高")
    ax.bar(dates, -plotted["new_low20_count"], width=1.8, color=GREEN, alpha=0.72, label="新低")
    net = plotted["new_high20_count"] - plotted["new_low20_count"]
    twin = ax.twinx()
    twin.plot(dates, net, color=PURPLE, linewidth=1.5, label="NH - NL")
    twin.axhline(0, color=GRID, linewidth=0.8)
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    twin.tick_params(axis="y", colors=MUTED, labelsize=7)
    ax.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=2)
    twin.legend(loc="upper right", fontsize=6.5, frameon=False)


def draw_ma_breadth_chart(ax: Axes, history: pd.DataFrame) -> None:
    style_axis(ax, "图 5  MA 广度三线")
    styles = (
        ("above_ma20_ratio", "MA20", RED),
        ("above_ma60_ratio", "MA60", BLUE),
        ("above_ma120_ratio", "MA120", PURPLE),
    )
    for column, label, color in styles:
        ax.plot(history["trade_date"], history[column], linewidth=1.7, color=color, label=label)
    for level in (0.3, 0.5, 0.7):
        ax.axhline(level, color=GRID, linestyle="--", linewidth=0.7)
    percent_axis(ax)
    ax.legend(loc="upper left", fontsize=6.8, frameon=False, ncol=3)
    ax.text(0.99, 0.04, "空白 = 样本外暖机，不补 0", ha="right", va="bottom", transform=ax.transAxes, fontsize=6.5, color=MUTED)


def draw_ma_change_chart(ax: Axes, history: pd.DataFrame) -> None:
    style_axis(ax, "图 6  MA60 广度日变化")
    change = history["breadth_ma60_change_pp"]
    colors = [RED if value >= 0 else GREEN for value in change.fillna(0)]
    ax.bar(history["trade_date"], change, width=1.8, color=colors, alpha=0.78)
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.axhspan(-3, 3, color=GRID, alpha=0.23, label="±3pp")
    ax.set_ylabel("百分点", fontsize=7, color=MUTED)
    ax.legend(loc="upper left", fontsize=6.5, frameon=False)


def _daily_readout(row: pd.Series) -> list[tuple[str, str, str]]:
    amount_ratio = row["amount_ratio_20"]
    up_ratio = row["up_ratio"]
    up_amount_ratio = row["up_amount_ratio"]
    high = row["new_high20_count"]
    low = row["new_low20_count"]
    ma120 = row["above_ma120_ratio"]

    if pd.isna(amount_ratio):
        volume = "量能：数据不足（20 日基准尚未形成）"
        volume_color = MUTED
    elif amount_ratio < 0.9:
        volume = f"量能：缩量，量比 {_number(amount_ratio, 2)}"
        volume_color = RED
    elif amount_ratio > 1.2:
        volume = f"量能：放量，量比 {_number(amount_ratio, 2)}"
        volume_color = GREEN
    else:
        volume = f"量能：平量，量比 {_number(amount_ratio, 2)}"
        volume_color = MUTED

    if pd.isna(up_ratio) or pd.isna(up_amount_ratio):
        internal = "内部：上涨家数或成交额占比缺失"
        internal_color = MUTED
    elif up_ratio < 0.35 and up_amount_ratio < 0.35:
        internal = "内部：普跌，家数和成交额占比都弱"
        internal_color = RED
    elif up_ratio < 0.35 and up_amount_ratio >= up_ratio + 0.10:
        internal = "内部：权重撑盘，少数大市值方向支撑成交"
        internal_color = RED
    elif up_ratio >= 0.60 and up_amount_ratio < up_ratio - 0.10:
        internal = "内部：小票闹，普涨但成交额没有同步扩散"
        internal_color = MUTED
    elif up_ratio >= 0.60 and up_amount_ratio >= 0.60:
        internal = "内部：普涨，家数和成交额占比同步改善"
        internal_color = GREEN
    else:
        internal = "内部：分化，尚未形成一致性扩散"
        internal_color = MUTED

    if pd.isna(high) or pd.isna(low):
        structure = "结构：新高/新低数据不足"
        structure_color = MUTED
    elif low > high:
        structure = f"结构：新低占优，NH/NL {_number(high, 0)}/{_number(low, 0)}"
        structure_color = RED
    elif high > low:
        structure = f"结构：新高占优，NH/NL {_number(high, 0)}/{_number(low, 0)}"
        structure_color = GREEN
    else:
        structure = f"结构：新高新低接近，NH/NL {_number(high, 0)}/{_number(low, 0)}"
        structure_color = MUTED

    weak = (
        (not pd.isna(amount_ratio) and amount_ratio < 0.9)
        or (not pd.isna(up_ratio) and up_ratio < 0.35)
        or (not pd.isna(low) and not pd.isna(high) and low > high)
        or (not pd.isna(ma120) and ma120 < 0.35)
    )
    if weak:
        implication = "对 v3 的含义：全 A 过滤器偏弱；若宽基价信号仍为 Risk-On，标记为背离，执行层不因单一价格信号扩张仓位。"
        implication_color = RED
    else:
        implication = "对 v3 的含义：全 A 过滤器未显示明显背离，仍需结合宽基价格和行业轮动确认。"
        implication_color = GREEN
    return [
        (volume, volume_color, ""),
        (internal, internal_color, ""),
        (structure, structure_color, ""),
        (implication, implication_color, ""),
    ]


def add_readout(figure: Figure, row: pd.Series, asof: pd.Timestamp) -> None:
    ax = figure.add_axes((0.035, 0.035, 0.93, 0.17))
    ax.set_facecolor(PANEL)
    ax.axis("off")
    ax.text(
        0.015,
        0.87,
        f"当日读数 · {asof:%Y-%m-%d} · 样本 {int(row['sample_count']):,} 只",
        fontsize=9.5,
        fontweight="bold",
        color=TEXT,
        transform=ax.transAxes,
    )
    for index, (text, color, _) in enumerate(_daily_readout(row)):
        y = 0.66 - index * 0.19
        ax.text(0.02, y, text, fontsize=8.3, color=color, va="center", transform=ax.transAxes)
        if index < 3:
            ax.plot([0.015, 0.985], [y - 0.085, y - 0.085], color="#EEF1F5", linewidth=0.6, transform=ax.transAxes)


def build_report(frame: pd.DataFrame, asof: pd.Timestamp, lookback: int | None) -> Figure:
    configure_font()
    row = frame.loc[frame["trade_date"] == asof].iloc[0]
    history = history_until(frame, asof, lookback)
    fig = plt.figure(figsize=(16, 9), facecolor=PAGE)
    fig.text(0.035, 0.972, f"全 A 温度计 · {asof:%Y-%m-%d}", fontsize=20, fontweight="bold", color=TEXT, va="top")
    fig.text(
        0.035,
        0.943,
        "市场广度与成交额过滤页 | 只回答内部是否支持宽基价格信号，不直接生成买卖指令",
        fontsize=8.5,
        color=MUTED,
        va="top",
    )

    kpis = [
        ("成交额", _amount(row["total_amount_100m"]), TEXT),
        ("量比", _number(row["amount_ratio_20"], 2), _status_color(row["amount_ratio_20"], "amount_ratio")),
        ("上涨家数比", _pct(row["up_ratio"]), _status_color(row["up_ratio"], "ratio")),
        ("上涨成交额比", _pct(row["up_amount_ratio"]), _status_color(row["up_amount_ratio"], "ratio")),
        ("20 日新高", _number(row["new_high20_count"]), GREEN if row["new_high20_count"] >= row["new_low20_count"] else RED),
        ("20 日新低", _number(row["new_low20_count"]), RED if row["new_low20_count"] > row["new_high20_count"] else GREEN),
        ("站上 MA20", _pct(row["above_ma20_ratio"]), _status_color(row["above_ma20_ratio"], "ratio")),
        ("站上 MA60 / MA120", f"{_pct(row['above_ma60_ratio'])} / {_pct(row['above_ma120_ratio'])}", _status_color(row["above_ma120_ratio"], "ma120")),
    ]
    left, gap, width = 0.035, 0.008, 0.108
    for index, (label, value, color) in enumerate(kpis):
        add_kpi(fig, (left + index * (width + gap), 0.855, width, 0.068), label, value, color)

    axes = [
        fig.add_axes((0.035, 0.565, 0.295, 0.255)),
        fig.add_axes((0.352, 0.565, 0.295, 0.255)),
        fig.add_axes((0.669, 0.565, 0.295, 0.255)),
        fig.add_axes((0.035, 0.285, 0.295, 0.235)),
        fig.add_axes((0.352, 0.285, 0.295, 0.235)),
        fig.add_axes((0.669, 0.285, 0.295, 0.235)),
    ]
    draw_amount_chart(axes[0], history)
    draw_updown_chart(axes[1], history)
    draw_price_volume_chart(axes[2], history)
    draw_new_high_low_chart(axes[3], history)
    draw_ma_breadth_chart(axes[4], history)
    draw_ma_change_chart(axes[5], history)
    add_readout(fig, row, asof)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DATA_DIR / "market_breadth.csv"), help="模块 1 输出的 CSV")
    parser.add_argument("--output", default=str(PDF_DIR / "report_01_market_breadth.pdf"))
    parser.add_argument("--date", help="报告交易日 YYYYMMDD，默认使用 CSV 最新交易日")
    parser.add_argument("--lookback", type=int, default=120, help="图表回看交易日数量，默认 120")
    args = parser.parse_args()

    frame = load_data(args.input)
    asof = choose_asof(frame, args.date)
    figure = build_report(frame, asof, args.lookback)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    print(f"已生成报告：{output}（交易日 {asof:%Y-%m-%d}）")


if __name__ == "__main__":
    main()
