#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模块 2：申万行业热力图与轮动，生成横向 PDF 报告。

数据源：同目录 industry_rotation.csv，完全使用上游已计算字段；不请求 Tushare/AKShare。

示例：
    python report_02_industry_rotation.py
    python report_02_industry_rotation.py --date 20260928 \
        --input data/industry_rotation.csv \
        --output data/pdf/report_02_industry_rotation.pdf \
        --summary-json data/report_02_industry_rotation_summary.json

报告两页：
    1. 31 个申万 2021 一级行业流通市值加权面积热力图、短期强弱与量能；
    2. 20 日行业轮动四象限（若数据不足则明确标为待形成）、5 日相对强弱，
       二级/三级行业分层扫描榜单。

重要限制：源 CSV 未提供上下级行业关系、行业内成分股广度，因此不臆造
二、三级树形归属，也不把行业涨跌家数当作成分股上涨家数。`excess_*`
的基准由上游脚本定义，本程序不会擅自宣称一定是 A500。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


# 固定申万 2021 版一级行业索引代码。不可简单把所有 801xxx 当作一级行业：
# 原文件还包含二级行业、申万宽基及风格指数。
SW2021_L1 = {
    "801010": "农林牧渔", "801030": "基础化工", "801040": "钢铁",
    "801050": "有色金属", "801080": "电子", "801110": "家用电器",
    "801120": "食品饮料", "801130": "纺织服饰", "801140": "轻工制造",
    "801150": "医药生物", "801160": "公用事业", "801170": "交通运输",
    "801180": "房地产", "801200": "商贸零售", "801210": "社会服务",
    "801230": "综合", "801710": "建筑材料", "801720": "建筑装饰",
    "801730": "电力设备", "801740": "国防军工", "801750": "计算机",
    "801760": "传媒", "801770": "通信", "801780": "银行",
    "801790": "非银金融", "801880": "汽车", "801890": "机械设备",
    "801950": "煤炭", "801960": "石油石化", "801970": "环保",
    "801980": "美容护理",
}

REQUIRED_COLUMNS = {
    "ts_code", "trade_date", "name", "ret_1d", "ret_5d", "amount_ratio_20d",
    "float_mv", "excess_5d", "excess_20d", "momentum_change_20d", "quadrant",
}
OPTIONAL_NUMERIC = (
    "ret_20d", "ret_60d", "amount_ratio_5d", "excess_60d", "amount", "pct_change",
)

TEXT = "#253548"
MUTED = "#718097"
LINE = "#DEE6EE"
BG = "#F5F7FA"
RED = "#C84B56"       # A 股：红涨
GREEN = "#238A66"     # A 股：绿跌
ORANGE = "#DB8B3C"
BLUE = "#4879AC"
PURPLE = "#8A65AC"

FONT_PATHS = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"


def configure_font() -> None:
    for path in FONT_PATHS:
        if Path(path).exists():
            try:
                fm.fontManager.addfont(path)
                plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
                break
            except Exception:
                continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"行业数据文件不存在：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"ts_code": str}, low_memory=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV 缺少模块 2 必需字段：{sorted(missing)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    if frame["trade_date"].isna().any():
        raise ValueError("CSV 中存在不能解析的 trade_date")
    frame["ts_code"] = frame["ts_code"].str.strip().str.upper()
    frame["name"] = frame["name"].fillna("").astype(str).str.strip()
    numeric = (REQUIRED_COLUMNS - {"ts_code", "trade_date", "name", "quadrant"}) | (
        set(OPTIONAL_NUMERIC) & set(frame.columns)
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    duplicate = frame.duplicated(["trade_date", "ts_code"], keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, ["trade_date", "ts_code"]].head(5).to_dict("records")
        raise ValueError(f"同一日期行业代码存在重复，请先检查上游数据：{sample}")
    if frame.empty:
        raise ValueError("行业数据文件为空")
    return frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def choose_date(frame: pd.DataFrame, date_arg: str | None) -> pd.Timestamp:
    if date_arg:
        try:
            date = pd.to_datetime(date_arg, format="%Y%m%d", errors="raise")
        except ValueError as exc:
            raise ValueError("--date 应使用 YYYYMMDD，例如 20260928") from exc
    else:
        date = frame["trade_date"].max()
    if not (frame["trade_date"] == date).any():
        raise ValueError(f"CSV 中不存在交易日：{date:%Y-%m-%d}")
    return date


def split_levels(day: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    codes = day["ts_code"].str.extract(r"^(\d{6})\.SI$", expand=False).fillna("")
    l1 = day.loc[codes.isin(SW2021_L1)].copy()
    l1["code6"] = codes.loc[l1.index].to_numpy()
    # 剔除申万宽基、风格指数，避免将 801001/801811 等误识别成二级行业。
    is_l2_code = codes.str.match(r"^801\d{3}$") & ~codes.isin(SW2021_L1)
    is_general = day["name"].str.startswith("申万") | day["name"].str.endswith("指数")
    l2 = day.loc[is_l2_code & ~is_general].copy()
    l3 = day.loc[codes.str.match(r"^85\d{4}$")].copy()
    if l1.empty:
        raise ValueError("当前日期没有匹配申万2021版一级行业代码，不能生成行业热力图")
    missing_l1 = set(SW2021_L1) - set(l1["code6"])
    if missing_l1:
        print(f"警告：当日只获取到 {len(l1)}/31 个一级行业；缺失代码：{sorted(missing_l1)}")
    renamed = l1[l1.apply(lambda r: r["name"] != SW2021_L1[r["code6"]], axis=1)]
    if not renamed.empty:
        print("提示：CSV 与静态申万2021一级行业名称不完全一致，请核对分类版本")
    return l1, l2, l3


def fmt_pct(value: Any, digits: int = 2, signed: bool = True) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):+.{digits}%}" if signed else f"{float(value):.{digits}%}"


def fmt_ratio(value: Any) -> str:
    return "—" if pd.isna(value) else f"{float(value):.2f}×"


def safe_median(series: pd.Series) -> float:
    return float(series.median()) if series.notna().any() else float("nan")


def panel(ax: Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.025, 0.955, title, fontsize=11, color=TEXT, fontweight="bold", va="top", transform=ax.transAxes)
    if subtitle:
        ax.text(0.025, 0.865, subtitle, fontsize=7.2, color=MUTED, va="top", transform=ax.transAxes)


def base_page(date: pd.Timestamp, page: int, title: str, subtitle: str) -> Figure:
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(0.035, 0.975, title, fontsize=19, fontweight="bold", color=TEXT, va="top")
    fig.text(0.035, 0.935, subtitle, fontsize=8.4, color=MUTED, va="top")
    fig.text(0.964, 0.975, f"{date:%Y-%m-%d}", ha="right", fontsize=10, color=TEXT, va="top")
    fig.text(0.035, 0.019, "drcc Quant Engine  |  模块 02：申万行业轮动  |  仅展示日线统计，不生成交易指令", fontsize=7.1, color=MUTED)
    fig.text(0.965, 0.019, f"{page} / 2", ha="right", fontsize=7.1, color=MUTED)
    return fig


def add_kpis(fig: Figure, items: list[tuple[str, str, str]]) -> None:
    x0, gap = 0.035, 0.009
    width = (0.93 - (len(items) - 1) * gap) / len(items)
    for i, (label, value, color) in enumerate(items):
        ax = fig.add_axes((x0 + i * (width + gap), 0.828, width, 0.075))
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(LINE)
            spine.set_linewidth(0.7)
        ax.text(0.055, 0.76, label, fontsize=8.1, color=MUTED, transform=ax.transAxes)
        ax.text(0.055, 0.22, value, fontsize=14, color=color, fontweight="bold", transform=ax.transAxes)


def heat_color(value: float) -> str:
    """A 股颜色习惯：负收益为绿，正收益为红；按固定范围饱和以便跨日比较。"""
    if not np.isfinite(value):
        return "#AEB8C5"
    stops = [(-0.06, (21, 108, 81)), (-0.03, (42, 152, 107)),
             (-0.005, (181, 222, 201)), (0, (236, 240, 244)),
             (0.005, (240, 184, 177)), (0.03, (205, 80, 87)),
             (0.06, (162, 38, 55))]
    rgb = [int(np.interp(float(value), [s[0] for s in stops], [s[1][j] for s in stops])) for j in range(3)]
    return "#" + "".join(f"{v:02x}" for v in rgb)


def foreground(bg: str) -> str:
    rgb = np.array([int(bg[i:i+2], 16) / 255 for i in (1, 3, 5)])
    brightness = float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))
    return "#FFFFFF" if brightness < 0.56 else TEXT


def _row_worst(sizes: list[float], side: float) -> float:
    if not sizes or min(sizes) <= 0 or side <= 0:
        return float("inf")
    total = sum(sizes)
    return max(side * side * max(sizes) / total**2, total**2 / (side * side * min(sizes)))


def squarify_tiles(weights: list[float], x: float, y: float, w: float, h: float) -> list[tuple[float, float, float, float]]:
    """无 squarify 包依赖的矩形树图布局；输入顺序与输出矩形一一对应。"""
    if not weights or w <= 0 or h <= 0:
        return []
    weight_sum = sum(weights)
    areas = [max(float(v), 0) / weight_sum * w * h for v in weights]
    rectangles: list[tuple[float, float, float, float]] = []
    row: list[float] = []

    def layout(sizes: list[float], xx: float, yy: float, ww: float, hh: float) -> tuple[float, float, float, float]:
        area = sum(sizes)
        if ww >= hh:
            band = area / hh if hh else 0
            cy = yy
            for val in sizes:
                rh = val / band if band else 0
                rectangles.append((xx, cy, band, rh))
                cy += rh
            return xx + band, yy, max(ww - band, 0), hh
        band = area / ww if ww else 0
        cx = xx
        for val in sizes:
            rw = val / band if band else 0
            rectangles.append((cx, yy, rw, band))
            cx += rw
        return xx, yy + band, ww, max(hh - band, 0)

    for a in areas:
        if not row or _row_worst(row + [a], min(w, h)) <= _row_worst(row, min(w, h)):
            row.append(a)
        else:
            x, y, w, h = layout(row, x, y, w, h)
            row = [a]
    if row:
        layout(row, x, y, w, h)
    return rectangles


def draw_treemap(ax: Axes, sectors: pd.DataFrame) -> bool:
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4.7)
    ax.set_aspect("auto")
    ax.axis("off")
    arranged = sectors.sort_values("float_mv", ascending=False).reset_index(drop=True)
    raw = arranged["float_mv"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    equal_weight = not (np.isfinite(raw).all() and raw.sum() > 0 and (raw > 0).all())
    if equal_weight:
        raw = np.ones(len(arranged), dtype=float)
    blocks = squarify_tiles(raw.tolist(), 0, 0, 16, 4.7)
    for (_, row), (x, y, w, h) in zip(arranged.iterrows(), blocks):
        color = heat_color(float(row["ret_1d"]))
        rectangle = patches.Rectangle((x + 0.015, y + 0.015), max(w - 0.03, 0), max(h - 0.03, 0),
                                      facecolor=color, edgecolor="white", linewidth=0.7)
        ax.add_patch(rectangle)
        area = w * h
        if area < 0.12 or min(w, h) < 0.22:
            continue
        fg = foreground(color)
        if area >= 0.78 and min(w, h) >= 0.48:
            fs = min(11.6, 6.9 + area * 0.46)
            ax.text(x + w/2, y + h/2 + min(h*.12, .20), row["name"], ha="center", va="center",
                    fontsize=fs, color=fg, fontweight="bold", clip_on=True, clip_path=rectangle)
            ax.text(x + w/2, y + h/2 - min(h*.15, .24), fmt_pct(row["ret_1d"]), ha="center", va="center",
                    fontsize=max(7, fs - 1), color=fg, fontweight="bold", clip_on=True, clip_path=rectangle)
        elif area >= 0.25 and min(w, h) >= 0.29:
            ax.text(x + w/2, y + h/2, f"{row['name']}\n{fmt_pct(row['ret_1d'], 1)}", ha="center", va="center",
                    fontsize=6.4, color=fg, clip_on=True, clip_path=rectangle)
        else:
            ax.text(x + w/2, y + h/2, row["name"], ha="center", va="center",
                    fontsize=6.0, color=fg, clip_on=True, clip_path=rectangle)
    return equal_weight


def draw_ranked_bars(ax: Axes, data: pd.DataFrame, field: str, n: int, title: str,
                     percent: bool = True, reverse: bool = False) -> None:
    panel(ax, title)
    rows = data.dropna(subset=[field]).sort_values(field, ascending=reverse).head(n).copy()
    if rows.empty:
        ax.text(.5, .46, "尚无有效数据", ha="center", color=MUTED, transform=ax.transAxes)
        return
    # 从上到下排序；以 0 为基准，正负统一采用 A 股红涨绿跌。
    vals = rows[field].to_numpy(dtype=float)
    ys = np.arange(len(rows))
    xx = vals * 100 if percent else vals
    xlim = max(float(np.max(np.abs(xx))) * 1.20, 0.1 if percent else .15)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-.8, len(rows) + .35)
    ax.axvline(0, color=LINE, linewidth=.9)
    ax.barh(ys, xx, height=.57, color=[RED if v >= 0 else GREEN for v in vals], alpha=.92)
    for y, (_, row) in zip(ys, rows.iterrows()):
        ax.text(.025, (y + .95)/(len(rows) + 1.15), str(row["name"]), ha="left", va="center",
                color=TEXT, fontsize=7.8, transform=ax.transAxes,
                bbox={"facecolor":"white", "edgecolor":"none", "alpha":.91, "pad":1.0})
        val = row[field]
        ax.text(.975, (y + .95)/(len(rows) + 1.15), fmt_pct(val) if percent else fmt_ratio(val),
                ha="right", va="center", color=RED if val >= 0 else GREEN, fontsize=7.8,
                transform=ax.transAxes, bbox={"facecolor":"white", "edgecolor":"none", "alpha":.93, "pad":1.0})
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def draw_text_rank(ax: Axes, data: pd.DataFrame, field: str, count: int, heading: str,
                   descending: bool = True, suffix: str = "pct") -> None:
    panel(ax, heading)
    rows = data.dropna(subset=[field]).sort_values(field, ascending=not descending).head(count)
    if rows.empty:
        ax.text(.5, .45, "无有效记录", ha="center", color=MUTED, fontsize=9, transform=ax.transAxes)
        return
    for i, (_, row) in enumerate(rows.iterrows()):
        yy = .81 - .116 * i
        v = row[field]
        label = fmt_ratio(v) if suffix == "ratio" else fmt_pct(v)
        ax.text(.045, yy, f"{i+1:>2}.  {row['name']}", fontsize=8, color=TEXT, va="center", transform=ax.transAxes)
        ax.text(.94, yy, label, fontsize=8.3, color=(RED if v >= 0 else GREEN) if suffix == "pct" else TEXT,
                ha="right", va="center", fontweight="bold", transform=ax.transAxes)
        if i < len(rows)-1:
            ax.plot([.045,.94], [yy-.059, yy-.059], color="#EFF2F6", lw=.65, transform=ax.transAxes)


def data_count(l1: pd.DataFrame, key: str) -> int:
    return int(l1[key].notna().sum()) if key in l1 else 0


def first_page(day: pd.DataFrame, l1: pd.DataFrame, date: pd.Timestamp) -> Figure:
    fig = base_page(date, 1, "申万行业热力图 · 市场结构", "申万2021一级行业 | 方块面积 = 流通市值占比；颜色 = 当日涨跌；均来自上游 CSV")
    up = int((l1["ret_1d"] > 0).sum())
    down = int((l1["ret_1d"] < 0).sum())
    vr = safe_median(l1["amount_ratio_20d"])
    add_kpis(fig, [
        ("一级行业覆盖", f"{len(l1)} / 31", TEXT),
        ("上涨 / 下跌", f"{up} / {down}", RED if up > down else GREEN),
        ("行业涨跌中位数", fmt_pct(safe_median(l1["ret_1d"])), TEXT),
        ("行业20日量比中位数", fmt_ratio(vr), ORANGE),
        ("5日相对强势行业", str(int((l1["excess_5d"] > 0).sum())) + " / " + str(data_count(l1,"excess_5d")), BLUE),
    ])
    hm = fig.add_axes((.035, .349, .93, .455))
    equal = draw_treemap(hm, l1)
    fig.text(.036, .329, "-6%及以下", color=GREEN, fontsize=7.2, va="center")
    for k, v in enumerate(np.linspace(-.06, .06, 25)):
        fig.add_artist(patches.Rectangle((.38 + k * .010, .322), .010, .013, transform=fig.transFigure,
                                         facecolor=heat_color(v), edgecolor="none"))
    fig.text(.372, .328, "绿跌", fontsize=7, color=MUTED, ha="right")
    fig.text(.505, .328, "0", fontsize=7, color=MUTED, ha="center")
    fig.text(.640, .328, "红涨", fontsize=7, color=MUTED, ha="left")
    fig.text(.67, .329, "+6%及以上", color=RED, fontsize=7.2, va="center")
    if equal:
        fig.text(.963, .329, "市值缺失：面积临时等权", color=ORANGE, fontsize=7.3, ha="right")
    p1 = fig.add_axes((.035, .061, .297, .243))
    p2 = fig.add_axes((.35, .061, .297, .243))
    p3 = fig.add_axes((.665, .061, .30, .243))
    draw_text_rank(p1, l1, "ret_1d", 5, "当日表现靠前 · 一级", True)
    draw_text_rank(p2, l1, "ret_1d", 5, "当日表现靠后 · 一级", False)
    draw_text_rank(p3, l1, "amount_ratio_20d", 5, "20日成交额比率靠前 · 一级", True, "ratio")
    fig.text(.036, .045, "注：热力图面积不是成交额；量比表示今日成交额/上游20日基准。一级行业不包含申万宽基/风格指数。", fontsize=7.1, color=MUTED)
    return fig


def quadrant_state(l1: pd.DataFrame) -> pd.DataFrame:
    result = l1.dropna(subset=["excess_20d", "momentum_change_20d"]).copy()
    if result.empty:
        return result
    # 只有两项指标同时有效时才派生四象限；不修改上游 quadrant 字段。
    result["derived_quadrant"] = np.select(
        [(result["excess_20d"] >= 0) & (result["momentum_change_20d"] >= 0),
         (result["excess_20d"] < 0) & (result["momentum_change_20d"] >= 0),
         (result["excess_20d"] >= 0) & (result["momentum_change_20d"] < 0)],
        ["领先且改善", "落后但改善", "领先但减弱"], default="落后且减弱")
    return result


def draw_quadrants(ax: Axes, l1: pd.DataFrame) -> None:
    panel(ax, "20日行业轮动四象限", "横轴：20日相对超额；纵轴：20日动能变化（上游口径）")
    valid = quadrant_state(l1)
    if valid.empty:
        ax.text(.5, .58, "20日轮动指标尚未形成", ha="center", fontsize=14, fontweight="bold", color=TEXT, transform=ax.transAxes)
        ax.text(.5, .42, "excess_20d / momentum_change_20d 均缺失", ha="center", fontsize=9, color=MUTED, transform=ax.transAxes)
        ax.text(.5, .29, "保留空白，不用 5 日指标替代 20 日象限", ha="center", fontsize=8.2, color=ORANGE, transform=ax.transAxes)
        return
    plot = ax.inset_axes((.10, .13, .82, .68))
    xx = valid["excess_20d"] * 100
    yy = valid["momentum_change_20d"] * 100
    span_x = max(xx.abs().max() * 1.25, 1.0)
    span_y = max(yy.abs().max() * 1.25, 1.0)
    plot.set_xlim(-span_x, span_x); plot.set_ylim(-span_y, span_y)
    plot.axhline(0, lw=.9, color=LINE); plot.axvline(0, lw=.9, color=LINE)
    plot.grid(alpha=.13)
    colors = {"领先且改善": RED, "落后但改善": ORANGE, "领先但减弱": BLUE, "落后且减弱": GREEN}
    for _, r in valid.iterrows():
        q = r["derived_quadrant"]
        plot.scatter(r["excess_20d"]*100, r["momentum_change_20d"]*100, s=32,
                     color=colors[q], alpha=.87)
    # 极端坐标标签避免 31 条同时标注而重叠。
    focus = valid.assign(distance=abs(xx.to_numpy())+abs(yy.to_numpy())).nlargest(7, "distance")
    for _, r in focus.iterrows():
        plot.annotate(r["name"], (r["excess_20d"]*100, r["momentum_change_20d"]*100),
                      fontsize=6.9, xytext=(3,3), textcoords="offset points", color=TEXT)
    plot.tick_params(labelsize=7, colors=MUTED)
    plot.set_xlabel("20日相对超额（百分点）", fontsize=7, color=MUTED)
    plot.set_ylabel("20日动能变化（百分点）", fontsize=7, color=MUTED)
    ax.text(.03,.045, f"有效样本 {len(valid)}/{len(l1)}；象限按两项指标正负派生，不覆盖上游 quadrant", fontsize=7.1, color=MUTED, transform=ax.transAxes)


def draw_excess_chart(ax: Axes, l1: pd.DataFrame) -> None:
    panel(ax, "5日相对强弱 · 一级行业", "展示超额靠前 / 靠后各6个；excess_5d 基准按上游口径")
    all_values = l1.dropna(subset=["excess_5d"]).sort_values("excess_5d")
    values = pd.concat([all_values.head(6), all_values.tail(6)]).drop_duplicates("ts_code").sort_values("excess_5d")
    if values.empty:
        ax.text(.5,.5,"5日相对强弱缺失",color=MUTED,ha="center",transform=ax.transAxes)
        return
    subax = ax.inset_axes((.24,.15,.70,.68))
    vals = values["excess_5d"].to_numpy()*100
    indices = np.arange(len(vals))
    subax.barh(indices, vals, height=.72, color=[RED if v >= 0 else GREEN for v in vals])
    subax.axvline(0,color=LINE,lw=.9)
    subax.set_yticks(indices, values["name"],fontsize=6.7)
    subax.tick_params(axis="x",labelsize=7,colors=MUTED)
    subax.tick_params(axis="y",length=0,colors=TEXT)
    subax.grid(axis="x",alpha=.14)
    for spine in subax.spines.values(): spine.set_visible(False)
    subax.set_xlabel("5日超额收益（百分点）",fontsize=7,color=MUTED)
    ax.text(.03,.038,f"全体正超额 {int((all_values['excess_5d']>0).sum())} / {len(all_values)} 个行业",fontsize=7.5,color=MUTED,transform=ax.transAxes)


def second_page(day: pd.DataFrame, l1: pd.DataFrame, l2: pd.DataFrame, l3: pd.DataFrame, date: pd.Timestamp) -> Figure:
    fig = base_page(date, 2, "行业轮动 · 二三级行业扫描", "仅使用行业 CSV 预计算收益、相对超额及成交额比率；不从代码推断二三级上下级归属")
    q = quadrant_state(l1)
    valid20 = len(q)
    add_kpis(fig, [
        ("一级 / 二级 / 三级", f"{len(l1)} / {len(l2)} / {len(l3)}", TEXT),
        ("5日收益中位数", fmt_pct(safe_median(l1["ret_5d"])), TEXT),
        ("5日超额中位数", fmt_pct(safe_median(l1["excess_5d"])), BLUE),
        ("20日象限有效覆盖", f"{valid20} / {len(l1)}", RED if valid20 == len(l1) else ORANGE),
    ])
    qa = fig.add_axes((.035,.455,.448,.350))
    ea = fig.add_axes((.499,.455,.466,.350))
    draw_quadrants(qa,l1)
    draw_excess_chart(ea,l1)
    ps = [fig.add_axes((.035,.075,.217,.355)), fig.add_axes((.267,.075,.217,.355)),
          fig.add_axes((.499,.075,.217,.355)), fig.add_axes((.731,.075,.234,.355))]
    draw_text_rank(ps[0], l2, "excess_5d", 6, "二级：5日超额靠前", True)
    draw_text_rank(ps[1], l2, "excess_5d", 6, "二级：5日超额靠后", False)
    draw_text_rank(ps[2], l3, "excess_5d", 6, "三级：5日超额靠前", True)
    draw_text_rank(ps[3], l3, "excess_5d", 6, "三级：5日超额靠后", False)
    fig.text(.036,.050,"局限：没有行业成分股与上下级映射，无法输出行业内个股广度或可信的三级嵌套树图；本页只展示已知层级的独立榜单。",fontsize=7.1,color=MUTED)
    return fig


def build_summary(day: pd.DataFrame, l1: pd.DataFrame, l2: pd.DataFrame, l3: pd.DataFrame, date: pd.Timestamp) -> dict[str, Any]:
    result: dict[str, Any] = {
        "trade_date": date.strftime("%Y-%m-%d"),
        "source_rows": len(day),
        "level_counts": {"level1":len(l1), "level2":len(l2), "level3":len(l3)},
        "industry_up_count": int((l1["ret_1d"] > 0).sum()),
        "industry_down_count": int((l1["ret_1d"] < 0).sum()),
        "median_ret_1d": safe_median(l1["ret_1d"]),
        "median_ret_5d": safe_median(l1["ret_5d"]),
        "median_amount_ratio_20d": safe_median(l1["amount_ratio_20d"]),
        "median_excess_5d": safe_median(l1["excess_5d"]),
        "valid_quadrant_count": len(quadrant_state(l1)),
        "excess_benchmark": "按上游 industry_rotation.csv 计算口径；本报告不作推定",
        "caveats": ["无行业内成分股广度数据", "无二三级父子映射，未构建三级嵌套图"],
    }
    # NaN 不属于严格 JSON；空值用 null。
    for key, value in list(result.items()):
        if isinstance(value, float) and not math.isfinite(value):
            result[key] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(DATA_DIR / "industry_rotation.csv"), help="上游行业 CSV，默认 data/industry_rotation.csv")
    parser.add_argument("--output", default=str(PDF_DIR / "report_02_industry_rotation.pdf"), help="生成的两页 PDF")
    parser.add_argument("--date", help="报告交易日 YYYYMMDD；默认源 CSV 中最新日期")
    parser.add_argument("--summary-json", default=str(DATA_DIR / "report_02_industry_rotation_summary.json"), help="输出供总控程序使用的 JSON 摘要")
    args = parser.parse_args()

    frame = load_data(Path(args.input))
    date = choose_date(frame, args.date)
    day = frame.loc[frame["trade_date"] == date].copy()
    l1, l2, l3 = split_levels(day)
    configure_font()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for figure in (first_page(day,l1,date), second_page(day,l1,l2,l3,date)):
            pdf.savefig(figure, facecolor=figure.get_facecolor(), bbox_inches="tight")
            plt.close(figure)
    if args.summary_json:
        dst = Path(args.summary_json)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(build_summary(day,l1,l2,l3,date),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        print(f"摘要 JSON：{dst}")
    print(f"已生成：{out}（交易日 {date:%Y-%m-%d}，一级 {len(l1)} / 二级 {len(l2)} / 三级 {len(l3)}，有效象限 {len(quadrant_state(l1))}/{len(l1)}）")


if __name__ == "__main__":
    main()
