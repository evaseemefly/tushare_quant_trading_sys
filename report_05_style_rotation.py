#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模块05：市场风格轮动与相对强弱（Python 3.9+）。

数据全部为**日线收盘价**，不需要分钟级接口。价格指数对价格指数，
RS = (本标的收盘点位 / 对照标的收盘点位)；以图表区间第一日归一化为100。
RS的20/60日变化 = RS(t)/RS(t-n) - 1，按共同交易日对齐。绝对收益另行展示。
RS上升只代表相对跑赢，不代表本标的绝对上涨，也不是买入指令。

首次执行（与config.yaml放同目录）：
    python report_05_style_rotation.py --fetch --date 20260928
每日：
    python report_05_style_rotation.py --fetch
仅根据已缓存CSV出图：
    python report_05_style_rotation.py

优先使用/兼容同目录：
    style_05_index_daily.csv   模块05指数日线缓存；--fetch生成
    style_05_sw_daily.csv      模块05申万行业日线缓存；--fetch生成
    index_04_daily.csv        模块04已经下载的指数日线（只读）
    index_daily.csv           模块03已有的指数日线（只读）
    industry_rotation.csv     模块02已有的申万行业当日快照（只读）
    market_breadth.csv        提供默认目标交易日（只读）

新建文件：
    output/pdf/report_05_style_rotation.pdf
    output/data/report_05_style_rotation.json
    output/data/style_05_signals.csv

该程序不改变原v3持仓、不会自动下单；数据缺失及交易日不匹配时显示缺失。
红利使用价格指数，不代表含分红的总收益；请勿将不同时点或来源强拼为同一天。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# 代码逐一固定，不从相似名称推测或替换数据。
INDEX_CODES = {
    "CSI1000": ("000852.CSI", "中证1000"),
    "CSI300": ("000300.SH", "沪深300"),
    "STAR50": ("000688.SH", "科创50"),
    "A500": ("000510.SH", "中证A500"),
    "DIVIDEND": ("000922.CSI", "中证红利"),
}
SW_CODES = {
    "SEMI": ("801081.SI", "申万半导体"),
    "GOLD": ("850531.SI", "申万黄金Ⅲ"),
    "NONFERROUS": ("801050.SI", "申万有色金属"),
    "UTILITY": ("801160.SI", "申万公用事业"),
}
# 清晰指定分子/分母；原指数代码不通过ETF交易代码替代。
PAIRS = [
    ("small_vs_large", "中证1000 / 沪深300", "CSI1000", "CSI300", "大盘与小盘"),
    ("growth_vs_core", "科创50 / A500", "STAR50", "A500", "成长与宽基"),
    ("dividend_vs_core", "中证红利 / A500", "DIVIDEND", "A500", "红利与宽基"),
    ("semiconductor_vs_core", "半导体 / A500", "SEMI", "A500", "科技行业"),
    ("gold_vs_core", "黄金行业 / A500", "GOLD", "A500", "黄金股方向"),
    ("nonferrous_vs_core", "有色金属 / A500", "NONFERROUS", "A500", "周期行业"),
    ("utility_vs_core", "公用事业 / A500", "UTILITY", "A500", "防御行业"),
]
ALL_CODES = {k: v[0] for k, v in {**INDEX_CODES, **SW_CODES}.items()}
CODE_LABELS = {v[0]: v[1] for v in {**INDEX_CODES, **SW_CODES}.values()}
FONT_PATHS = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)
COLORS = {
    "text": "#26374B", "muted": "#6E7E93", "bg": "#F5F7FB", "panel": "#FFFFFF",
    "grid": "#DDE5EE", "red": "#C74D58", "green": "#268C6C",
    "blue": "#477BAF", "orange": "#DE933E", "purple": "#896CB1",
}
MIN_COMMON_DAYS = 61  # 必须有t-60日，才显示60日相对强弱
MAX_DISPLAY_DAYS = 180


def configure_font() -> None:
    for font in FONT_PATHS:
        if Path(font).exists():
            try:
                fm.fontManager.addfont(font)
                plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
                break
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42


def parse_date(value: Any) -> pd.Timestamp:
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if len(s) == 8 and s.isdigit():
        dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    else:
        dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        raise ValueError("无法识别交易日期：%r" % value)
    return dt.normalize()


def ymd(value: Any) -> str:
    return parse_date(value).strftime("%Y%m%d")


def latest_market_date(root: Path) -> Optional[pd.Timestamp]:
    p = root / "market_breadth.csv"
    if not p.exists():
        return None
    x = pd.read_csv(p, usecols=["trade_date"], dtype=str, encoding="utf-8-sig")
    return max(x["trade_date"].dropna().map(parse_date)) if not x.empty else None


def _read_series_file(path: Path, allowed: set, source_label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ts_code", "trade_date", "close", "source"])
    raw = pd.read_csv(path, dtype={"ts_code": str}, encoding="utf-8-sig", low_memory=False)
    need = {"ts_code", "trade_date", "close"}
    if not need.issubset(raw.columns):
        raise ValueError("%s 缺少字段 %s" % (path.name, sorted(need - set(raw.columns))))
    data = raw[["ts_code", "trade_date", "close"]].copy()
    data["ts_code"] = data["ts_code"].astype(str).str.upper().str.strip()
    data = data[data["ts_code"].isin(allowed)].copy()
    if data.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "close", "source"])
    data["trade_date"] = data["trade_date"].map(parse_date)
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data[data["close"].gt(0) & np.isfinite(data["close"])].copy()
    if data.duplicated(["ts_code", "trade_date"]).any():
        dup = data.loc[data.duplicated(["ts_code", "trade_date"], keep=False), ["ts_code", "trade_date"]]
        raise ValueError("%s 同一代码与交易日有重复：%s" % (path.name, dup.head(3).to_dict("records")))
    data["source"] = source_label
    return data


def load_prices(root: Path) -> Tuple[pd.DataFrame, List[str]]:
    """只读合并优先级：独立缓存 > 模块04/03 > 模块02当日快照。"""
    parts, notices = [], []
    items = [
        ("style_05_index_daily.csv", set(x[0] for x in INDEX_CODES.values())),
        ("style_05_sw_daily.csv", set(x[0] for x in SW_CODES.values())),
        ("index_04_daily.csv", set(x[0] for x in INDEX_CODES.values())),
        ("index_daily.csv", set(x[0] for x in INDEX_CODES.values())),
        ("industry_rotation.csv", set(x[0] for x in SW_CODES.values())),
    ]
    for filename, allowed in items:
        p = root / filename
        if p.exists():
            part = _read_series_file(p, allowed, filename)
            notices.append("%s：%s条" % (filename, len(part)))
            if not part.empty:
                parts.append(part)
    if not parts:
        raise FileNotFoundError(
            "未找到可用的指数/行业历史价格；执行 --fetch 从Tushare下载。"
            "只有industry_rotation.csv单日快照不足以计算20/60日轮动。"
        )
    merged = pd.concat(parts, ignore_index=True)
    # 最靠前的缓存优先；相同键而不同来源出现显著差异时输出告警，但不擅自平均。
    merged["priority"] = merged["source"].map({p[0]: i for i, p in enumerate(items)})
    merged = merged.sort_values(["ts_code", "trade_date", "priority"])
    dup = merged[merged.duplicated(["ts_code", "trade_date"], keep=False)]
    if not dup.empty:
        rng = dup.groupby(["ts_code", "trade_date"])["close"].agg(["min", "max"])
        conflicts = rng[(rng["max"] / rng["min"] - 1).gt(0.005)]
        if not conflicts.empty:
            notices.append("发现%s组跨文件收盘价差异>0.5%%；采用模块05缓存优先，不进行平均。" % len(conflicts))
    merged = merged.drop_duplicates(["ts_code", "trade_date"], keep="first")
    return merged.sort_values(["ts_code", "trade_date"])[["ts_code", "trade_date", "close", "source"]], notices


def _return(values: pd.Series, n: int) -> Optional[float]:
    if len(values) <= n:
        return None
    v0, v1 = float(values.iloc[-n-1]), float(values.iloc[-1])
    if v0 <= 0 or v1 <= 0:
        return None
    return (v1 / v0 - 1) * 100.0


def _slope(values: pd.Series, step: int) -> Optional[float]:
    return _return(values, step)


def trend_name(slope: Optional[float], tolerance: float = 0.15) -> str:
    if slope is None:
        return "暖机中"
    if slope > tolerance:
        return "上行"
    if slope < -tolerance:
        return "下行"
    return "走平"


def relative_state(diff: Optional[float], slope: Optional[float], current: float, ma: Optional[float]) -> str:
    """仅描述RS本身的象限，不输出买卖建议。"""
    if diff is None or slope is None or ma is None:
        return "数据不足"
    if current >= ma and slope > 0.15:
        return "领先且改善"
    if current < ma and slope > 0.15:
        return "落后但改善"
    if current >= ma and slope < -0.15:
        return "领先但减弱"
    if current < ma and slope < -0.15:
        return "落后且减弱"
    return "相对动能平缓"


def compute_pair(data: pd.DataFrame, row: tuple, asof: pd.Timestamp) -> Dict[str, Any]:
    key, label, nk, bk, group = row
    nc, bc = ALL_CODES[nk], ALL_CODES[bk]
    n = data[(data["ts_code"] == nc) & (data["trade_date"] <= asof)][["trade_date", "close"]]
    b = data[(data["ts_code"] == bc) & (data["trade_date"] <= asof)][["trade_date", "close"]]
    m = n.rename(columns={"close": "numerator"}).merge(
        b.rename(columns={"close": "benchmark"}), on="trade_date", how="inner"
    ).sort_values("trade_date").reset_index(drop=True)
    res: Dict[str, Any] = {
        "id": key, "name": label, "group": group,
        "numerator_code": nc, "benchmark_code": bc,
        "asof": asof.strftime("%Y-%m-%d"),
        "matched_days": len(m), "status": "数据不足", "state": "数据不足",
        "rs_5d_pct": None, "rs_20d_pct": None, "rs_60d_pct": None,
        "num_20d_pct": None, "benchmark_20d_pct": None,
        "num_60d_pct": None, "benchmark_60d_pct": None,
        "rs_ma20_slope_5d_pct": None, "rs_ma60_slope_10d_pct": None,
        "trend20": "暖机中", "trend60": "暖机中",
        "latest_num": None, "latest_benchmark": None,
        "history": pd.DataFrame(),
    }
    if m.empty:
        res["status"] = "缺少共同交易日"
        return res
    last = m["trade_date"].iloc[-1]
    res["latest_common_date"] = last.strftime("%Y-%m-%d")
    if last != asof:
        res["status"] = "数据滞后（最新共同交易日%s）" % last.strftime("%Y-%m-%d")
        return res
    if len(m) < 2:
        res["status"] = "历史不足"
        return res
    ratio = m["numerator"] / m["benchmark"]
    m["rs"] = 100.0 * ratio / ratio.iloc[0]
    m["rs_ma20"] = m["rs"].rolling(20, min_periods=20).mean()
    m["rs_ma60"] = m["rs"].rolling(60, min_periods=60).mean()
    # 统一共同交易日且仅以 t 时及之前数据滚动计算，未使用未来数据。
    res["latest_num"] = float(m["numerator"].iloc[-1])
    res["latest_benchmark"] = float(m["benchmark"].iloc[-1])
    for days in (5, 20, 60):
        res["rs_%sd_pct" % days] = _return(m["rs"], days)
    for days in (20, 60):
        res["num_%sd_pct" % days] = _return(m["numerator"], days)
        res["benchmark_%sd_pct" % days] = _return(m["benchmark"], days)
    a, z = m["rs_ma20"].dropna(), m["rs_ma60"].dropna()
    res["rs_ma20_slope_5d_pct"] = _slope(a, 5)
    res["rs_ma60_slope_10d_pct"] = _slope(z, 10)
    res["trend20"] = trend_name(res["rs_ma20_slope_5d_pct"])
    res["trend60"] = trend_name(res["rs_ma60_slope_10d_pct"])
    ma = float(m["rs_ma20"].iloc[-1]) if np.isfinite(m["rs_ma20"].iloc[-1]) else None
    res["state"] = relative_state(res["rs_20d_pct"], res["rs_ma20_slope_5d_pct"], float(m["rs"].iloc[-1]), ma)
    res["status"] = "可用（20/60日）" if len(m) >= MIN_COMMON_DAYS else ("可用（20日，60日暖机）" if len(m) >= 21 else "仅短期，历史不足")
    res["history"] = m
    return res


def _fmt(v: Optional[float], suffix: str = "%", precision: int = 1) -> str:
    return ("%+.*f%s" % (precision, float(v), suffix)) if v is not None and np.isfinite(v) else "-"


def _pct_color(value: Optional[float]) -> str:
    return COLORS["red"] if value is not None and value > 0 else (COLORS["green"] if value is not None and value < 0 else COLORS["muted"])


def _fig() -> plt.Figure:
    configure_font()
    return plt.figure(figsize=(16, 9), facecolor=COLORS["bg"])


def _header(fig: plt.Figure, asof: pd.Timestamp, section: str, subtitle: str, page: int, demo: bool = False) -> None:
    fig.text(.04, .969, "市场风格轮动 · %s" % asof.strftime("%Y-%m-%d"), fontsize=19, fontweight="bold", color=COLORS["text"], va="top")
    fig.text(.04, .913, section, fontsize=11, fontweight="bold", color=COLORS["blue"])
    fig.text(.04, .884, subtitle, fontsize=8.5, color=COLORS["muted"])
    fig.text(.96, .962, "MODULE 05  /  %s" % page, fontsize=9, color=COLORS["muted"], ha="right", va="top")
    if demo:
        fig.text(.5, .49, "模拟数据 · 仅用于排版验证", ha="center", va="center", fontsize=25, color="#A8B3C3", rotation=23, alpha=.36)
    fig.text(.04, .014, "相对强弱 ≠ 绝对收益。指数均按价格点位比较，不含红利现金再投资；不直接生成交易指令。", fontsize=7.5, color=COLORS["muted"])


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(COLORS["panel"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], lw=.65)
    ax.tick_params(labelsize=7, length=0, colors=COLORS["muted"])
    ax.set_axisbelow(True)


def _panel(fig: plt.Figure, result: Dict[str, Any], xywh: tuple) -> None:
    box = fig.add_axes(xywh)
    _style(box)
    box.set_xticks([]); box.set_yticks([])
    for s in box.spines.values():
        s.set_visible(True); s.set_color(COLORS["grid"]); s.set_linewidth(.8)
    box.text(.035, .94, result["name"], transform=box.transAxes, fontsize=11, fontweight="bold", va="top", color=COLORS["text"])
    box.text(.965, .94, result["state"], transform=box.transAxes, fontsize=8, ha="right", va="top", color=COLORS["muted"])
    h: pd.DataFrame = result["history"]
    if h.empty:
        box.text(.5, .52, result["status"], fontsize=11, ha="center", va="center", color=COLORS["muted"], transform=box.transAxes)
        box.text(.035, .12, "需要相同交易日的两条指数/行业收盘价序列", fontsize=7, color=COLORS["muted"], transform=box.transAxes)
        return
    # 内部绘图区：用figure固定坐标，避免嵌套axes/中文标签挤压。
    x, y, w, hh = xywh
    ax = fig.add_axes((x+.027*w, y+.28*hh, .93*w, .48*hh))
    _style(ax)
    plot = h.tail(MAX_DISPLAY_DAYS)
    dates = plot["trade_date"]
    ax.plot(dates, plot["rs"], color=COLORS["blue"], lw=1.75, label="相对强弱指数")
    ax.plot(dates, plot["rs_ma20"], color=COLORS["orange"], lw=1.4, label="RS MA20")
    ax.plot(dates, plot["rs_ma60"], color=COLORS["purple"], lw=1.4, label="RS MA60")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.00), fontsize=6.5, frameon=False, ncol=3, borderaxespad=0)
    # 当前RS在20/60日和分子/分母的绝对收益分开列明。
    box.text(.035, .17,
             "RS 5/20/60日：%s  /  %s  /  %s" % (_fmt(result["rs_5d_pct"]), _fmt(result["rs_20d_pct"]), _fmt(result["rs_60d_pct"])),
             fontsize=8.1, color=COLORS["text"], transform=box.transAxes)
    box.text(.035, .07,
             "分子20日 %s · 分母20日 %s  |  RS均线20/60日：%s / %s" %
             (_fmt(result["num_20d_pct"]), _fmt(result["benchmark_20d_pct"]), result["trend20"], result["trend60"]),
             fontsize=7.4, color=COLORS["muted"], transform=box.transAxes)


def _rows(results: List[Dict[str, Any]], only: List[str]) -> List[Dict[str, Any]]:
    return [r for r in results if r["id"] in only]


def _table(fig: plt.Figure, results: List[Dict[str, Any]], loc: tuple, mini: bool = False) -> None:
    ax = fig.add_axes(loc)
    _style(ax); ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color(COLORS["grid"])
    colx = [.026, .35, .49, .63, .77, .88] if not mini else [.026, .42, .58, .72, .85, .95]
    header = ["观察组合", "RS20日", "RS60日", "MA20趋势", "MA60趋势", "数据状态"]
    for xpos, h in zip(colx, header):
        ax.text(xpos, .88, h, transform=ax.transAxes, fontsize=7.7, fontweight="bold", color=COLORS["muted"], ha="right" if xpos > .9 else "left")
    for i, r in enumerate(results):
        y = .66 - i*.16
        if y < .04: break
        values = [r["name"], _fmt(r["rs_20d_pct"]), _fmt(r["rs_60d_pct"]), r["trend20"], r["trend60"], r["status"]]
        for j, (xpos, val) in enumerate(zip(colx, values)):
            col = _pct_color(r["rs_20d_pct"] if j == 1 else r["rs_60d_pct"]) if j in (1, 2) else COLORS["text"] if j == 0 else COLORS["muted"]
            ax.text(xpos, y, str(val), fontsize=7.4 if j == 5 else 8.0, transform=ax.transAxes, va="center", color=col, ha="right" if xpos > .9 else "left")
        ax.plot([.02, .98], [y-.075]*2, color=COLORS["grid"], linewidth=.55, transform=ax.transAxes)


def build_pdf(results: List[Dict[str, Any]], asof: pd.Timestamp, outpath: Path, notes: List[str], demo: bool = False) -> int:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    core = results[:4]
    sector = results[4:]
    with PdfPages(outpath) as pdf:
        fig = _fig()
        _header(fig, asof, "核心风格 | 大盘、小盘、成长、红利、科技", "RS = 分子/分母，归一化为100；20/60日曲线为相对强弱的滚动均线。", 1, demo)
        for r, loc in zip(core, [(.04,.56,.44,.30),(.52,.56,.44,.30),(.04,.235,.44,.30),(.52,.235,.44,.30)]):
            _panel(fig, r, loc)
        _table(fig, core, (.04,.043,.92,.17))
        pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)
        fig = _fig()
        _header(fig, asof, "行业风格 | 黄金、有色、公用事业", "申万行业指数与A500价格指数相比较；行业涨跌幅不等于对应ETF真实收益。", 2, demo)
        for r, loc in zip(sector, [(.04,.56,.44,.30),(.52,.56,.44,.30),(.04,.235,.44,.30)]):
            _panel(fig, r, loc)
        info = fig.add_axes((.52,.235,.44,.30)); _style(info); info.axis("off")
        for s in info.spines.values(): s.set_visible(True); s.set_color(COLORS["grid"])
        info.text(.04,.90,"使用与风险提示",fontsize=11,fontweight="bold",color=COLORS["text"],transform=info.transAxes)
        lines = [
            "① 相对强弱上升可能只是跌得更少；请同时看分子绝对收益。",
            "② 日线指标不显示盘中冲高回落和ETF折溢价。",
            "③ 红利使用价格指数，未纳入现金分红再投资。",
            "④ MA20/60趋势需足够共同交易日；缺值不补零。",
            "⑤ 5/20/60日指相隔5/20/60个共同交易日。",
            "⑥ 本报告只提供原v3模型的辅助过滤，不修改仓位。",
        ]
        for i, line in enumerate(lines): info.text(.04,.75-i*.115,line,fontsize=7.6,color=COLORS["muted"],transform=info.transAxes)
        _table(fig, sector, (.04,.043,.92,.17))
        pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)
    return 2


def _json_ready(result: Dict[str, Any]) -> Dict[str, Any]:
    item = {k:v for k,v in result.items() if k != "history"}
    for k,v in list(item.items()):
        if isinstance(v, (float, np.floating)) and not np.isfinite(v): item[k] = None
        if isinstance(v, np.integer): item[k] = int(v)
    return item


def save_summary(results: List[Dict[str, Any]], asof: pd.Timestamp, root: Path, notes: List[str]) -> None:
    p = root / "output/data"
    p.mkdir(parents=True, exist_ok=True)
    payload = {
        "module": "05_style_rotation", "trade_date": asof.strftime("%Y-%m-%d"),
        "formula": "RS=close_numerator/close_benchmark; RS_N_day_change=RS(t)/RS(t-N)-1, same trading days",
        "dividend_caveat": "price-index comparison; excludes reinvested cash dividends",
        "market_notice": "relative strength does not imply positive absolute return",
        "input_status": notes,
        "pairs": [_json_ready(r) for r in results],
    }
    (p/"report_05_style_rotation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(payload["pairs"]).to_csv(p/"style_05_signals.csv", index=False, encoding="utf-8-sig")


def _token(root: Path) -> str:
    env = os.environ.get("TUSHARE_TOKEN", "").strip()
    if env: return env
    f = root / "config.yaml"
    if f.exists():
        try: import yaml
        except ImportError as exc: raise RuntimeError("请在py39环境安装PyYAML：python -m pip install pyyaml") from exc
        cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        token = str((cfg.get("keys") or {}).get("tushare") or "").strip()
        if token: return token
    raise RuntimeError("找不到Tushare Token；请在同目录config.yaml填写 keys.tushare 或设置环境变量TUSHARE_TOKEN")


def _client(root: Path):
    try: import tushare as ts
    except ImportError as exc: raise RuntimeError("当前环境缺少tushare：python -m pip install tushare") from exc
    try: return ts.pro_api(_token(root))
    except Exception as exc: raise RuntimeError("Tushare客户端初始化失败（未输出Token）：%s" % type(exc).__name__) from exc


def _fetch_one(pro, endpoint: str, code: str, begin: str, end: str, pause: float) -> pd.DataFrame:
    current = parse_date(begin)
    finish = parse_date(end)
    pieces: List[pd.DataFrame] = []
    while current <= finish:
        stop = min(finish, current + timedelta(days=299))
        args = {"ts_code": code, "start_date": ymd(current), "end_date": ymd(stop)}
        err = None
        for attempt in range(3):
            try:
                x = getattr(pro, endpoint)(**args)
                if not isinstance(x, pd.DataFrame): raise TypeError("非DataFrame返回")
                if not x.empty:
                    missing = {"ts_code", "trade_date", "close"} - set(x.columns)
                    if missing: raise ValueError("接口缺少字段 %s" % sorted(missing))
                    pieces.append(x[["ts_code", "trade_date", "close"]].copy())
                err = None
                break
            except Exception as exc:
                err = exc
                if attempt < 2: time.sleep(max(.05, pause)*(attempt+1))
        if err is not None:
            raise RuntimeError("%s %s %s-%s抓取失败：%s" % (endpoint,code,args["start_date"],args["end_date"],str(err))) from err
        current = stop + timedelta(days=1)
        time.sleep(max(0.0,pause))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["ts_code","trade_date","close"])


def _write_cache(path: Path, incoming: pd.DataFrame, allowed: set) -> int:
    old = _read_series_file(path, allowed, path.name)
    if incoming.empty:
        return len(old)
    fresh = incoming.copy()
    fresh["ts_code"] = fresh["ts_code"].astype(str).str.upper().str.strip()
    fresh = fresh[fresh["ts_code"].isin(allowed)].copy()
    fresh["trade_date"] = fresh["trade_date"].map(parse_date)
    fresh["close"] = pd.to_numeric(fresh["close"],errors="coerce")
    fresh = fresh[fresh["close"].gt(0) & np.isfinite(fresh["close"])].copy()
    if fresh.empty: return len(old)
    merged = pd.concat([old[["ts_code","trade_date","close"]],fresh[["ts_code","trade_date","close"]]],ignore_index=True)
    merged = merged.drop_duplicates(["ts_code","trade_date"],keep="last").sort_values(["ts_code","trade_date"])
    out = merged.copy()
    out["trade_date"] = out["trade_date"].dt.strftime("%Y%m%d")
    temp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(temp,index=False,encoding="utf-8-sig")
    os.replace(temp,path)
    return len(out)


def fetch_data(root: Path, asof: pd.Timestamp, args, pro=None) -> List[str]:
    pro = pro if pro is not None else _client(root)
    all_errors: List[str] = []
    end = ymd(asof)
    bootstrap = ymd(args.start_date) if args.start_date else ymd(asof - timedelta(days=args.bootstrap_days))
    for endpoint, mapping, filename in (("index_daily", INDEX_CODES, "style_05_index_daily.csv"),
                                         ("sw_daily", SW_CODES, "style_05_sw_daily.csv")):
        path = root / filename
        old = _read_series_file(path, set(x[0] for x in mapping.values()), filename)
        additions = []
        for _, (code,name) in mapping.items():
            existing = old.loc[old["ts_code"].eq(code), "trade_date"] if not old.empty else pd.Series(dtype="datetime64[ns]")
            begin = bootstrap
            if not args.full_refresh and not existing.empty:
                begin = max(parse_date(bootstrap), existing.max()-timedelta(days=args.overlap_days)).strftime("%Y%m%d")
            if begin > end: continue
            print("  %s %s: %s 至 %s" % (endpoint, name, begin, end))
            try:
                part = _fetch_one(pro,endpoint,code,begin,end,args.pause)
                if part.empty:
                    print("    数据为空：%s（不填造历史）" % code)
                else:
                    additions.append(part)
            except Exception as exc:
                all_errors.append(str(exc))
                print("    抓取失败：%s" % str(exc),file=sys.stderr)
        incoming = pd.concat(additions, ignore_index=True) if additions else pd.DataFrame(columns=["ts_code","trade_date","close"])
        n = _write_cache(path, incoming, set(x[0] for x in mapping.values()))
        print("  %s 已保存 %d 条" % (filename,n))
    return all_errors


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--directory", default=str(ROOT), help="同目录CSV/config.yaml所在目录")
    p.add_argument("--date", help="目标收盘交易日YYYYMMDD；默认使用market_breadth.csv最新日期")
    p.add_argument("--fetch", action="store_true", help="先用config.yaml增量补齐指数与申万行业历史日线")
    p.add_argument("--start-date", help="首次回溯起点YYYYMMDD")
    p.add_argument("--bootstrap-days", type=int, default=720, help="首次回溯自然日，默认720")
    p.add_argument("--overlap-days", type=int, default=14, help="每次增量回溯自然日，默认14")
    p.add_argument("--full-refresh", action="store_true", help="忽略缓存断点，重抓起点以后的历史")
    p.add_argument("--pause", type=float, default=.25, help="接口调用之间暂停秒数")
    p.add_argument("--output", help="PDF输出路径；默认output/pdf/report_05_style_rotation.pdf")
    p.add_argument("--demo-watermark", action="store_true", help="只在离线模拟数据验证时添加水印")
    return p


def run(args, pro=None) -> int:
    root = Path(args.directory).expanduser().resolve()
    asof = parse_date(args.date) if args.date else latest_market_date(root)
    if asof is None:
        raise ValueError("未指定--date，且market_breadth.csv不存在或为空；请传--date YYYYMMDD")
    if args.bootstrap_days < 100 or args.overlap_days < 0 or args.pause < 0:
        raise ValueError("bootstrap-days至少100，overlap-days/pause不得为负")
    errs: List[str] = []
    if args.fetch:
        errs = fetch_data(root,asof,args,pro=pro)
    data, notes = load_prices(root)
    results = [compute_pair(data,row,asof) for row in PAIRS]
    valid = sum(r["status"].startswith("可用") for r in results)
    if valid == 0:
        notes.append("全部配对没有足够、且截至目标日同步的历史数据；PDF将显示真实缺失状态。")
    if errs:
        notes.append("本轮有%d个下载窗口失败；已缓存历史保留，缺口没有填造。"%len(errs))
    out = Path(args.output).expanduser() if args.output else root/"output/pdf/report_05_style_rotation.pdf"
    n = build_pdf(results,asof,out,notes,demo=args.demo_watermark)
    save_summary(results,asof,root,notes)
    print("模块05 %s：%d/%d组合有历史信号；PDF %d页：%s"%(asof.strftime("%Y-%m-%d"),valid,len(results),n,out))
    for r in results:
        print("  %s | %s | RS20 %s | RS60 %s | %s" % (r["name"],r["status"],_fmt(r["rs_20d_pct"]),_fmt(r["rs_60d_pct"]),r["state"]))
    if errs: print("提示：有未解决的下载失败，详见前述日志；报告缺失项保持空白。",file=sys.stderr)
    return 0


def main() -> None:
    try: status = run(parser().parse_args())
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print("ERROR: %s" % exc,file=sys.stderr); status=1
    sys.exit(status)


if __name__ == "__main__":
    main()
