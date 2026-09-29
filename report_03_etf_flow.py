#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模块 03：ETF资金与交易质量 | 日线监控PDF（独立报告生成器）。

沿用模块01/02的本地CSV -> PDF/JSON方式；不连接Tushare、不生成交易指令。
所有输入文件默认为本脚本同目录，只有ETF日线为必需。可选文件缺失时明确展示缺口。

示例：
    python report_03_etf_flow.py
    python report_03_etf_flow.py --date 20260928 --input etf_daily.csv \\
        --shares etf_share.csv --nav etf_nav.csv --index index_daily.csv \\
        --map etf_watchlist.csv --summary-json output/data/report_03_etf_flow.json

文件接口：
* etf_daily.csv (必需): ts_code,trade_date,close,amount；可含name,pct_chg,pre_close。
  Tushare fund_daily 的 amount 单位为千元、pct_chg 为百分数；也支持
  amount_100m（亿元）或 amount_yuan（元）。默认 amount 按千元读取。
* etf_share.csv (可选): ts_code,trade_date,fd_share（万份，Tushare fund_share）。
  也支持 shares_10k（万份）或 shares_units（份）；不存在不自行推算。
* etf_nav.csv (可选): ts_code,nav_date,unit_nav（元/份，Tushare fund_nav）。
* index_daily.csv (可选): ts_code,trade_date,close；pct_chg可选，按百分数读取。
* etf_watchlist.csv (可选): ts_code,name,index_code,category,focus。
  ETF与标的指数必须通过显式映射；不会从ETF简称猜测指数。

核心口径：
1. 量能比率 = 当日成交额 / 之前20个该ETF交易记录的平均成交额（不含当日）。
2. 5/20日份额变化要求报告日和第5/20个前序ETF交易日均有披露记录；
   不把缺失/延迟披露视为零申赎，也不以前值冒充当日。
3. 同日ETF收盘价/正式单位净值 - 1 为收盘折溢价参考，不是盘中IOPV溢价。
   净值日期晚于/早于报告日时不计算当日折溢价。
4. ETF-指数收益差是“价格收益差”，不是严格跟踪误差；ETF未复权价格在
   分红/拆分等事件附近会失真，QDII、跨境ETF的净值时间口径应另行复核。
5. 成交额是二级市场交易，不是基金净流入；份额变化 × 当日单位净值只是
   申赎量级的近似金额，绝不标注为真实资金流入。
6. 只有存在可用日线时才出报告，不会用上游市场/行业CSV伪造ETF数据。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

BASE = Path(__file__).resolve().parent
TEXT, MUTED, LINE, BG = "#253548", "#718097", "#DEE6EE", "#F5F7FA"
RED, GREEN, BLUE, ORANGE, PURPLE = "#C84B56", "#238A66", "#4879AC", "#DB8B3C", "#8A65AC"
FONT_PATHS = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)


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


def read_optional(path: Path | None, label: str) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        print(f"提示：{label}未提供（{path.name}），相应指标将显示缺失")
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"ts_code": str, "index_code": str}, low_memory=False)
    if frame.empty:
        print(f"提示：{label}为空（{path.name}），相应指标将显示缺失")
        return None
    return frame


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")


def normalize_code(frame: pd.DataFrame, code_col: str = "ts_code") -> None:
    frame[code_col] = frame[code_col].astype("string").str.strip().str.upper()
    if frame[code_col].isna().any() or frame[code_col].eq("").any():
        raise ValueError(f"{code_col}存在空值")


def normalize_date(frame: pd.DataFrame, date_col: str, label: str) -> None:
    raw = frame[date_col].astype("string").str.strip()
    compact = raw.str.fullmatch(r"\d{8}")
    parsed = pd.to_datetime(raw.where(compact), format="%Y%m%d", errors="coerce")
    parsed = parsed.where(compact, pd.to_datetime(raw.where(~compact), errors="coerce"))
    frame[date_col] = parsed
    if frame[date_col].isna().any():
        raise ValueError(f"{label}存在不可解析日期 {date_col}")


def assert_unique(frame: pd.DataFrame, cols: list[str], label: str) -> None:
    dup = frame.duplicated(cols, keep=False)
    if dup.any():
        pairs = frame.loc[dup, cols].head(3).to_dict("records")
        raise ValueError(f"{label}的键 {cols} 出现重复，请先修复上游数据：{pairs}")


def numeric(frame: pd.DataFrame, fields: tuple[str, ...]) -> None:
    for field in fields:
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")


def read_daily(path: Path, amount_unit: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"缺少ETF日线：{path}。当前附件只有市场广度/行业数据，不能据此生成ETF报告；"
            "请由上游另行导出etf_daily.csv，字段见脚本开头。"
        )
    frame = read_optional(path, "ETF日线")
    if frame is None:
        raise ValueError("ETF日线为空")
    require_columns(frame, {"ts_code", "trade_date", "close"}, "ETF日线")
    if not any(c in frame for c in ("amount_100m", "amount_yuan", "amount")):
        raise ValueError("ETF日线必须提供 amount（单位由--amount-unit指定）或 amount_100m / amount_yuan")
    normalize_code(frame)
    normalize_date(frame, "trade_date", "ETF日线")
    assert_unique(frame, ["ts_code", "trade_date"], "ETF日线")
    numeric(frame, ("close", "pre_close", "pct_chg", "pct_change", "ret_1d", "amount", "amount_yuan", "amount_100m"))
    if (frame["close"].dropna() <= 0).any() or frame["close"].isna().any():
        raise ValueError("ETF日线close必须为有效正数")
    if "amount_100m" in frame:
        frame["amount_100m"] = frame["amount_100m"].astype(float)
    elif "amount_yuan" in frame:
        frame["amount_100m"] = frame["amount_yuan"] / 1e8
    else:
        frame["amount_100m"] = frame["amount"] * {"thousand": 1e-5, "ten_thousand": 1e-4, "yuan": 1e-8}[amount_unit]
    if frame["amount_100m"].isna().any() or (frame["amount_100m"] < 0).any():
        raise ValueError("ETF成交额存在缺失或负数，不能计算量比")
    frame = frame.sort_values(["ts_code", "trade_date"]).copy()
    # 单日回报优先使用原数据商的除息参考价口径；无此字段再使用收盘价简单变化。
    frame["simple_ret_1d"] = frame.groupby("ts_code")["close"].pct_change(fill_method=None)
    if "pct_chg" in frame:
        frame["ret_1d"] = frame["pct_chg"] / 100
    elif "pct_change" in frame:
        frame["ret_1d"] = frame["pct_change"] / 100
    elif "ret_1d" not in frame:
        frame["ret_1d"] = frame["simple_ret_1d"]
    frame["ret_1d"] = frame["ret_1d"].where(frame["ret_1d"].notna(), frame["simple_ret_1d"])
    group = frame.groupby("ts_code", group_keys=False)
    for days in (5, 20):
        # 需要完整前N个该ETF交易日；不包含报告日，防止放量时抬高基准。
        base = group["amount_100m"].transform(lambda s: s.shift(1).rolling(days, min_periods=days).mean())
        frame[f"amount_ratio_{days}d"] = (frame["amount_100m"] / base).replace([np.inf, -np.inf], np.nan)
        frame[f"price_ret_{days}d"] = group["close"].pct_change(days, fill_method=None)
    frame["session_num"] = group.cumcount()
    frame["window_start_5d"] = group["trade_date"].shift(5)
    if "name" not in frame:
        frame["name"] = frame["ts_code"]
    frame["name"] = frame["name"].fillna(frame["ts_code"]).astype(str)
    return frame.reset_index(drop=True)


def read_watchlist(path: Path | None) -> pd.DataFrame | None:
    frame = read_optional(path, "ETF观察映射")
    if frame is None:
        return None
    require_columns(frame, {"ts_code"}, "ETF观察映射")
    normalize_code(frame)
    assert_unique(frame, ["ts_code"], "ETF观察映射")
    if "index_code" in frame:
        frame["index_code"] = frame["index_code"].fillna("").astype(str).str.strip().str.upper()
    else:
        frame["index_code"] = ""
    for key in ("name", "category"):
        if key not in frame:
            frame[key] = ""
        frame[key] = frame[key].fillna("").astype(str).str.strip()
    if "focus" not in frame:
        frame["focus"] = 0
    frame["focus"] = frame["focus"].astype(str).str.lower().isin(("1", "true", "yes", "y", "是"))
    return frame[["ts_code", "name", "category", "focus", "index_code"]].copy()


def read_share(path: Path | None) -> pd.DataFrame | None:
    frame = read_optional(path, "ETF份额")
    if frame is None:
        return None
    require_columns(frame, {"ts_code", "trade_date"}, "ETF份额")
    normalize_code(frame)
    normalize_date(frame, "trade_date", "ETF份额")
    assert_unique(frame, ["ts_code", "trade_date"], "ETF份额")
    numeric(frame, ("fd_share", "shares_10k", "shares_units"))
    if "fd_share" in frame:
        frame["shares_10k"] = frame["fd_share"]
    elif "shares_10k" not in frame:
        if "shares_units" not in frame:
            raise ValueError("ETF份额缺少fd_share（万份）/shares_10k（万份）/shares_units（份）")
        frame["shares_10k"] = frame["shares_units"] / 10000
    if (frame["shares_10k"].dropna() < 0).any():
        raise ValueError("ETF份额存在负值")
    return frame[["ts_code", "trade_date", "shares_10k"]].sort_values(["ts_code", "trade_date"])


def read_nav(path: Path | None) -> pd.DataFrame | None:
    frame = read_optional(path, "ETF单位净值")
    if frame is None:
        return None
    require_columns(frame, {"ts_code", "nav_date", "unit_nav"}, "ETF单位净值")
    normalize_code(frame)
    normalize_date(frame, "nav_date", "ETF单位净值")
    numeric(frame, ("unit_nav",))
    if "ann_date" in frame:
        normalize_date(frame, "ann_date", "ETF净值公告")
        # 保留公告日期，后续按报告日期先过滤，再选择最后一个可知版本，避免历史重算使用未来修订。
        assert_unique(frame, ["ts_code", "nav_date", "ann_date"], "ETF单位净值公告")
    else:
        assert_unique(frame, ["ts_code", "nav_date"], "ETF单位净值")
    if (frame["unit_nav"].dropna() <= 0).any():
        raise ValueError("ETF单位净值需为正数")
    fields = ["ts_code", "nav_date", "unit_nav"] + (["ann_date"] if "ann_date" in frame else [])
    return frame[fields].sort_values(["ts_code", "nav_date"])


def read_index(path: Path | None) -> pd.DataFrame | None:
    frame = read_optional(path, "跟踪指数日线")
    if frame is None:
        return None
    require_columns(frame, {"ts_code", "trade_date", "close"}, "指数日线")
    normalize_code(frame)
    normalize_date(frame, "trade_date", "指数日线")
    assert_unique(frame, ["ts_code", "trade_date"], "指数日线")
    numeric(frame, ("close", "pct_chg", "pct_change", "ret_1d"))
    if (frame["close"].dropna() <= 0).any():
        raise ValueError("指数日线close存在非正值")
    frame = frame.sort_values(["ts_code", "trade_date"]).copy()
    group = frame.groupby("ts_code", group_keys=False)
    frame["simple_ret_1d"] = group["close"].pct_change(fill_method=None)
    if "pct_chg" in frame:
        frame["index_ret_1d"] = frame["pct_chg"] / 100
    elif "pct_change" in frame:
        frame["index_ret_1d"] = frame["pct_change"] / 100
    elif "ret_1d" in frame:
        frame["index_ret_1d"] = frame["ret_1d"]
    else:
        frame["index_ret_1d"] = frame["simple_ret_1d"]
    frame["index_ret_1d"] = frame["index_ret_1d"].where(frame["index_ret_1d"].notna(), frame["simple_ret_1d"])
    for days in (5, 20):
        frame[f"index_ret_{days}d"] = group["close"].pct_change(days, fill_method=None)
    return frame[["ts_code", "trade_date", "close", "index_ret_1d", "index_ret_5d", "index_ret_20d"]]


def build_metrics(
    daily: pd.DataFrame, shares: pd.DataFrame | None, nav: pd.DataFrame | None,
    index: pd.DataFrame | None, watch: pd.DataFrame | None, date: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = daily.loc[daily["trade_date"] <= date].copy()
    day = hist.loc[hist["trade_date"] == date].copy()
    if day.empty:
        raise ValueError(f"ETF日线中没有报告交易日 {date:%Y-%m-%d}；拒绝使用上一日充当今日数据")
    if watch is not None:
        day = day.merge(watch, on="ts_code", how="left", suffixes=("", "_map"), validate="one_to_one")
        day["name"] = day["name_map"].where(day["name_map"].fillna("").str.len() > 0, day["name"])
        day = day.drop(columns=["name_map"])
        # 若日线本身带有同名列，显式映射优先；不留下 *_map 影子列。
        for c in ("category", "focus", "index_code"):
            mapped = c + "_map"
            if mapped in day:
                if c == "focus":
                    day[c] = day[mapped].where(day[mapped].notna(), day[c])
                else:
                    mask = day[mapped].fillna("").astype(str).str.strip().ne("")
                    day[c] = day[mapped].where(mask, day[c])
                day = day.drop(columns=[mapped])
    for col, default in (("category", ""), ("focus", False), ("index_code", "")):
        if col not in day:
            day[col] = default
        day[col] = day[col].fillna(default)
    # 只对同一日期且具有直接映射的指数计算收益差，避免把不同指数误关联。
    if index is not None:
        ondate = index.loc[index["trade_date"] == date].rename(columns={"ts_code": "index_code", "close": "index_close_today"})
        day = day.merge(ondate, on=["index_code", "trade_date"], how="left", validate="many_to_one")
        # 用ETF起点日的“同日”指数收盘价衡量5日收益，避免两者交易日历错位。
        prev = index[["ts_code", "trade_date", "close"]].rename(
            columns={"ts_code": "index_code", "trade_date": "window_start_5d", "close": "index_close_start_5d"})
        day = day.merge(prev, on=["index_code", "window_start_5d"], how="left", validate="many_to_one")
        day["index_ret_5d"] = day["index_close_today"] / day["index_close_start_5d"] - 1
        day["index_ret_5d"] = day["index_ret_5d"].replace([np.inf,-np.inf],np.nan)
    for col in ("index_ret_1d", "index_ret_5d", "index_ret_20d"):
        if col not in day:
            day[col] = np.nan
    day["price_index_gap_1d"] = day["ret_1d"] - day["index_ret_1d"]
    day["price_index_gap_5d"] = day["price_ret_5d"] - day["index_ret_5d"]
    # 精确匹配报告日：缺失不填补；时间/披露滞后由上游负责人复查。
    if nav is not None:
        n = nav.copy()
        if "ann_date" in n:
            n = n.loc[n["ann_date"] <= date].sort_values("ann_date").drop_duplicates(
                ["ts_code", "nav_date"], keep="last")
        n = n.rename(columns={"nav_date": "trade_date"})
        day = day.merge(n, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    if "unit_nav" not in day:
        day["unit_nav"] = np.nan
    day["premium"] = day["close"] / day["unit_nav"] - 1
    day["premium"] = day["premium"].replace([np.inf, -np.inf], np.nan)
    day["share_today_10k"] = np.nan
    day["share_change_5d_pct"] = np.nan
    day["share_change_20d_pct"] = np.nan
    day["share_delta_5d_10k"] = np.nan
    if shares is not None:
        share_idx = shares.set_index(["ts_code", "trade_date"])["shares_10k"]
        hist_by_code = {code: sub.reset_index(drop=True) for code, sub in hist.groupby("ts_code")}
        for ix, r in day.iterrows():
            code = r["ts_code"]
            seq = hist_by_code[code]
            end = int(seq.index[seq["trade_date"] == date][-1])
            current = share_idx.get((code, date), np.nan)
            if pd.isna(current):
                continue
            day.at[ix, "share_today_10k"] = float(current)
            for n in (5, 20):
                if end < n:
                    continue
                earlier = seq.iloc[end - n]["trade_date"]
                old = share_idx.get((code, earlier), np.nan)
                if pd.notna(old) and old > 0:
                    day.at[ix, f"share_change_{n}d_pct"] = float(current / old - 1)
                    if n == 5:
                        day.at[ix, "share_delta_5d_10k"] = float(current - old)
    day["subscription_proxy_5d_100m"] = (day["share_delta_5d_10k"] * 10000 * day["unit_nav"] / 1e8)
    # 资料质量覆盖：不对缺失份额/净值进行填补。
    day["has_nav_today"] = day["unit_nav"].notna()
    day["has_share_today"] = day["share_today_10k"].notna()
    day["has_index_today"] = day["index_ret_1d"].notna()
    return day.sort_values(["focus", "amount_100m"], ascending=[False, False]).reset_index(drop=True), hist


def choose_day(daily: pd.DataFrame, value: str | None) -> pd.Timestamp:
    if not value:
        return daily["trade_date"].max()
    date = pd.to_datetime(value, format="%Y%m%d", errors="coerce")
    if pd.isna(date):
        raise ValueError("--date必须是YYYYMMDD")
    return date


def fmt_num(value: Any, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{float(value):,.{digits}f}"


def fmt_pct(value: Any, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{float(value):+.{digits}%}"


def fmt_ratio(value: Any) -> str:
    return "—" if pd.isna(value) else f"{float(value):.2f}×"


def color_change(value: Any) -> str:
    return MUTED if pd.isna(value) else (RED if value > 0 else GREEN if value < 0 else TEXT)


def setup_ax(ax: Axes, title: str, note: str = "") -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(.03, .95, title, transform=ax.transAxes, va="top", fontsize=10.8, fontweight="bold", color=TEXT)
    if note:
        ax.text(.03, .865, note, transform=ax.transAxes, va="top", fontsize=7.1, color=MUTED)


def page(date: pd.Timestamp, title: str, subtitle: str, number: int, label: str = "") -> Figure:
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(.035,.973,title,fontsize=18.5,fontweight="bold",color=TEXT,va="top")
    fig.text(.035,.933,subtitle,fontsize=8.2,color=MUTED,va="top")
    fig.text(.966,.973,f"{date:%Y-%m-%d}",ha="right",fontsize=10,color=TEXT,va="top")
    fig.text(.035,.018,"drcc Quant Engine  |  模块 03：ETF资金与交易质量  |  收盘日线，不生成买卖指令",fontsize=7.0,color=MUTED)
    fig.text(.965,.018,f"{number} / 2",ha="right",fontsize=7.0,color=MUTED)
    if label:
        fig.text(.5,.50,label,ha="center",va="center",fontsize=40,alpha=.12,rotation=19,fontweight="bold",color=ORANGE)
    return fig


def kpi_cards(fig: Figure, cards: list[tuple[str, str, str]]) -> None:
    n = len(cards); gap = .010; x0 = .035
    w = (.93 - (n-1)*gap)/n
    for i,(label,value,color) in enumerate(cards):
        ax=fig.add_axes((x0+i*(w+gap),.828,w,.075)); ax.set_facecolor("white")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor(LINE); s.set_linewidth(.8)
        ax.text(.05,.75,label,transform=ax.transAxes,color=MUTED,fontsize=8)
        ax.text(.05,.20,value,transform=ax.transAxes,color=color,fontsize=14,fontweight="bold")


def draw_table(fig: Figure, current: pd.DataFrame, max_items: int) -> None:
    ax=fig.add_axes((.035,.482,.93,.322)); ax.set_facecolor("white"); ax.axis("off")
    ax.text(.016,.956,"重点ETF交易质量总览",transform=ax.transAxes,color=TEXT,fontsize=11,fontweight="bold",va="top")
    headers=[("ETF / 代码",.016),("当日涨幅",.278),("成交(亿)",.384),("20日量比",.501),
             ("份额5日",.613),("收盘折溢",.716),("指数1日",.821),("价差1日",.916)]
    for s,x in headers: ax.text(x,.844,s,transform=ax.transAxes,fontsize=7.5,color=MUTED,va="center",ha="right" if x>.02 else "left")
    ax.plot([.014,.984],[.803,.803],transform=ax.transAxes,color=LINE,lw=.8)
    sample=current.head(max_items)
    for rowidx,(_,r) in enumerate(sample.iterrows()):
        y=.765-rowidx*.082
        if rowidx%2: ax.add_patch(plt.Rectangle((.012,y-.037),.973,.077,transform=ax.transAxes,facecolor=BG,edgecolor="none"))
        name=str(r["name"]).strip()
        if len(name)>12: name=name[:11]+"…"
        ax.text(.016,y,f"{name}  {r['ts_code']}",transform=ax.transAxes,fontsize=7.7,color=TEXT,va="center")
        values=[r["ret_1d"],r["amount_100m"],r["amount_ratio_20d"],r["share_change_5d_pct"],r["premium"],r["index_ret_1d"],r["price_index_gap_1d"]]
        formats=[fmt_pct(values[0]),fmt_num(values[1]),fmt_ratio(values[2]),fmt_pct(values[3]),fmt_pct(values[4]),fmt_pct(values[5]),fmt_pct(values[6])]
        xs=[.278,.384,.501,.613,.716,.821,.916]
        for j,(text,x) in enumerate(zip(formats,xs)):
            ax.text(x,y,text,transform=ax.transAxes,fontsize=7.7,color=color_change(values[j]) if j in (0,3,4,5,6) else TEXT,ha="right",va="center")
    if len(sample)==0: ax.text(.5,.52,"报告日没有ETF记录",color=MUTED,ha="center",transform=ax.transAxes)
    ax.text(.016,.030,"— 为上游缺失或窗口尚未形成；表中ETF按观察标志优先、其后按成交额排序。",color=MUTED,fontsize=7.0,transform=ax.transAxes)


def small_chart(ax: Axes, title: str, note: str, sub: pd.DataFrame, key: str, fmt: str, sort: bool = True) -> None:
    setup_ax(ax,title,note)
    valid=sub.dropna(subset=[key]).head(10)
    if valid.empty:
        ax.text(.5,.51,"当日暂无可用数据",transform=ax.transAxes,ha="center",color=MUTED,fontsize=11)
        return
    if sort: valid=valid.sort_values(key)
    inner=ax.inset_axes((.36,.13,.59,.62))
    vals=valid[key].to_numpy(dtype=float)
    names=[str(n)[:10] for n in valid["name"]]
    colors=[color_change(v) if fmt in ("percent","percent_point") else BLUE for v in vals]
    inner.barh(np.arange(len(vals)),vals,color=colors,height=.65)
    inner.set_yticks(np.arange(len(vals)),names,fontsize=7)
    inner.tick_params(axis="y",length=0,colors=TEXT)
    inner.tick_params(axis="x",labelsize=7,colors=MUTED)
    inner.grid(axis="x",alpha=.18)
    for spine in inner.spines.values(): spine.set_visible(False)
    if fmt=="percent": inner.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x*100:+.1f}%"))
    if fmt=="percent_point": inner.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x*100:+.1f}pp"))
    if fmt=="ratio": inner.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:.1f}×"))
    if fmt in ("percent","percent_point"): inner.axvline(0,color=LINE,lw=.9)


def draw_amount_history(ax: Axes, current: pd.DataFrame, hist: pd.DataFrame, date: pd.Timestamp) -> None:
    top=current.iloc[0]
    code=top["ts_code"]
    setup_ax(ax,f"成交额历史 · {str(top['name'])[:12]}",f"{code} | 当日成交额/之前20日均额")
    sub=hist.loc[(hist["ts_code"]==code)&(hist["trade_date"]<=date)].tail(45)
    if sub.empty:
        return
    inner=ax.inset_axes((.12,.13,.83,.60))
    inner.bar(sub["trade_date"],sub["amount_100m"],width=1.6,color=BLUE,alpha=.70)
    base=sub["amount_100m"]/sub["amount_ratio_20d"]
    inner.plot(sub["trade_date"],base,color=ORANGE,lw=1.3,label="前20日均额")
    inner.set_ylabel("亿元",fontsize=7,color=MUTED)
    inner.tick_params(labelsize=7,colors=MUTED); inner.grid(axis="y",alpha=.15)
    inner.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5))
    inner.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    for spine in inner.spines.values(): spine.set_visible(False)
    inner.legend(loc="upper left",fontsize=6.5,frameon=False)


def first_page(current: pd.DataFrame, hist: pd.DataFrame, date: pd.Timestamp, top: int, label: str) -> Figure:
    fig=page(date,"ETF资金与交易质量 · 收盘监控","ETF二级成交、份额申赎、正式净值和基准指数分开计算；没有的数据不补值。",1,label)
    kpi_cards(fig,[
        ("样本ETF数量",str(len(current)),TEXT),
        ("样本成交额合计",f"{current['amount_100m'].sum():,.1f}亿",TEXT),
        ("20日量比覆盖",f"{current['amount_ratio_20d'].notna().sum()}/{len(current)}",BLUE),
        ("当日份额可用",f"{current['has_share_today'].sum()}/{len(current)}",GREEN),
        ("当日净值可用",f"{current['has_nav_today'].sum()}/{len(current)}",ORANGE),
        ("当日指数可用",f"{current['has_index_today'].sum()}/{len(current)}",PURPLE),
    ])
    draw_table(fig,current,min(top,8))
    panels=[fig.add_axes((.035,.076,.298,.378)),fig.add_axes((.350,.076,.298,.378)),fig.add_axes((.665,.076,.300,.378))]
    draw_amount_history(panels[0],current,hist,date)
    small_chart(panels[1],"ETF成交额相对强度","当日/前20交易日均额，1×为基准",current,"amount_ratio_20d","ratio")
    small_chart(panels[2],"ETF与指数单日价格收益差","同日行情；不等于跟踪误差",current,"price_index_gap_1d","percent_point")
    return fig


def quality_panel(fig: Figure, current: pd.DataFrame) -> None:
    ax=fig.add_axes((.035,.078,.930,.210)); ax.set_facecolor("white"); ax.axis("off")
    ax.text(.018,.91,"数据质量与口径提示",fontsize=10.8,fontweight="bold",color=TEXT,transform=ax.transAxes)
    total=len(current)
    share5=int(current["share_change_5d_pct"].notna().sum())
    premium=int(current["premium"].notna().sum())
    ix=int(current["price_index_gap_1d"].notna().sum())
    msg=(
        f"份额5日有效 {share5}/{total}；同日折溢价有效 {premium}/{total}；同日指数价格差有效 {ix}/{total}。",
        "份额变化按两个确切交易日披露值计算；不将缺失数据作零流入，也不将ETF二级成交视为净流入。",
        "折溢价＝收盘价/同日正式单位净值-1，不是盘中IOPV；跨境QDII需注意净值时间与汇率口径。",
        "ETF-指数为未复权收盘价的价格收益差；若发生分红、拆分或跟踪口径不同，应单独复核。",
        "5日申赎量级估算＝5日份额变化×当日净值，仅供参考；若有公告日字段，排除报告日期后的净值公告。",
    )
    for i,line in enumerate(msg):
        ax.text(.019,.72-i*.145,line,fontsize=7.9,color=TEXT if i==0 else MUTED,transform=ax.transAxes,va="top")


def second_page(current: pd.DataFrame, date: pd.Timestamp, label: str) -> Figure:
    fig=page(date,"ETF申赎与价格偏离 · 数据校验","本页仅反映官方日线/份额/净值可验证的部分，延迟披露不允许自动填补。",2,label)
    kpi_cards(fig,[
        ("有效5日份额变化",f"{current['share_change_5d_pct'].notna().sum()}/{len(current)}",BLUE),
        ("有效20日份额变化",f"{current['share_change_20d_pct'].notna().sum()}/{len(current)}",BLUE),
        ("有效净申赎量级估算",f"{current['subscription_proxy_5d_100m'].notna().sum()}/{len(current)}",ORANGE),
        ("同日折溢价",f"{current['premium'].notna().sum()}/{len(current)}",PURPLE),
        ("5日价格收益差",f"{current['price_index_gap_5d'].notna().sum()}/{len(current)}",GREEN),
    ])
    panels=[fig.add_axes((.035,.315,.298,.490)),fig.add_axes((.350,.315,.298,.490)),fig.add_axes((.665,.315,.300,.490))]
    small_chart(panels[0],"5个交易日ETF份额变化","必须有当前日与5交易日前份额；否则留空",current,"share_change_5d_pct","percent")
    small_chart(panels[1],"同日ETF收盘折溢价","与同日正式单位净值比较，非盘中IOPV",current,"premium","percent")
    small_chart(panels[2],"5日ETF与指数价格收益差","历史价格未复权；不是严格跟踪误差",current,"price_index_gap_5d","percent_point")
    quality_panel(fig,current)
    return fig


def strict_json(item: Any) -> Any:
    if isinstance(item,(pd.Timestamp,)):
        return item.strftime("%Y-%m-%d")
    if isinstance(item,(np.integer,)):
        return int(item)
    if isinstance(item,(np.floating,float)):
        return float(item) if math.isfinite(float(item)) else None
    if isinstance(item,(np.bool_,)):
        return bool(item)
    if pd.isna(item):
        return None
    return item


def build_summary(current: pd.DataFrame, date: pd.Timestamp, label: str) -> dict[str,Any]:
    columns=("ts_code","name","category","index_code","ret_1d","amount_100m","amount_ratio_20d",
             "share_today_10k","share_change_5d_pct","share_change_20d_pct","subscription_proxy_5d_100m",
             "unit_nav","premium","index_ret_1d","price_index_gap_1d","price_index_gap_5d")
    rows=[{key:strict_json(row[key]) for key in columns} for _,row in current.iterrows()]
    return {
        "trade_date":date.strftime("%Y-%m-%d"),"label":label,"sample_count":len(current),
        "total_amount_100m":float(current["amount_100m"].sum()),
        "coverage":{"volume_ratio20":int(current["amount_ratio_20d"].notna().sum()),
                    "shares_today":int(current["has_share_today"].sum()),
                    "share_delta5":int(current["share_change_5d_pct"].notna().sum()),
                    "nav_today":int(current["has_nav_today"].sum()),
                    "index_today":int(current["has_index_today"].sum())},
        "caveats":["成交额不等于净申购","份额和净值仅同日匹配，缺失不回填",
                   "收盘折溢价不是盘中IOPV","ETF-指数为价格收益差，不是严格跟踪误差"],
        "etfs":rows,
    }


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",default=str(BASE/"etf_daily.csv"),help="必需：ETF日线CSV")
    parser.add_argument("--shares",default=str(BASE/"etf_share.csv"),help="可选：ETF份额CSV")
    parser.add_argument("--nav",default=str(BASE/"etf_nav.csv"),help="可选：ETF单位净值CSV")
    parser.add_argument("--index",default=str(BASE/"index_daily.csv"),help="可选：指数日线CSV")
    parser.add_argument("--map",default=str(BASE/"etf_watchlist.csv"),help="可选：手工ETF-指数映射CSV")
    parser.add_argument("--amount-unit",choices=("thousand","ten_thousand","yuan"),default="thousand",
                        help="仅用于原始amount字段：默认Tushare fund_daily千元")
    parser.add_argument("--date",help="报告交易日YYYYMMDD；默认最新ETF交易日")
    parser.add_argument("--output",default="output/pdf/report_03_etf_flow.pdf")
    parser.add_argument("--summary-json",help="可选：输出可由总控脚本使用的JSON摘要")
    parser.add_argument("--top",type=int,default=8,help="首页重点ETF数量，上限8")
    parser.add_argument("--label",default="",help="可选大字水印，例如：模拟数据·仅测试排版")
    args=parser.parse_args()
    daily=read_daily(Path(args.input),args.amount_unit)
    date=choose_day(daily,args.date)
    share=read_share(Path(args.shares))
    nav=read_nav(Path(args.nav))
    index=read_index(Path(args.index))
    watch=read_watchlist(Path(args.map))
    current,hist=build_metrics(daily,share,nav,index,watch,date)
    configure_font()
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    with PdfPages(output) as pdf:
        for fig in (first_page(current,hist,date,max(1,min(args.top,8)),args.label),second_page(current,date,args.label)):
            pdf.savefig(fig,facecolor=fig.get_facecolor(),bbox_inches="tight")
            plt.close(fig)
    if args.summary_json:
        dest=Path(args.summary_json);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(build_summary(current,date,args.label),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        print(f"摘要JSON：{dest}")
    print(f"已生成：{output}（{date:%Y-%m-%d}，ETF {len(current)}只；份额 {int(current['has_share_today'].sum())}，净值 {int(current['has_nav_today'].sum())}，指数 {int(current['has_index_today'].sum())}）")


if __name__=="__main__":
    main()
