#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模块04：指数估值与历史位置（Python 3.9+）

用途：5个指数的价格历史位置 vs PE(TTM)/PB/股息率3年与5年历史分位。

一键首次运行（读取同目录 config.yaml 的 keys.tushare）：
  python report_04_index_valuation.py --fetch --date 20260928
每日更新：
  python report_04_index_valuation.py --fetch
只生成报告（已有数据）：
  python report_04_index_valuation.py

数据（同目录）：
  index_04_daily.csv       指数日线。若不存在，可借用模块3的 index_daily.csv。
  index_04_valuation.csv   Tushare index_dailybasic 实际返回的历史估值（仅能取得的指数）。
  index_04_valuation_extra.csv 可选，其他官方/可信数据源导出的完整历史估值。
  index_04_coverage.csv    可选，科创50 TTM正盈利成分权重覆盖率（需来源/口径）。

index_04_valuation_extra.csv 必须含：
  ts_code,trade_date,pe_ttm,pb,dividend_yield,source,methodology
  可选：pe,earnings_coverage_pct,coverage_methodology
  dividend_yield 单位为百分数，如3.50表示3.50%，不能填0.035。
  不允许将行业估值、个股PE平均、ETF价格或价格分位冒充指数估值；
  不允许把不同口径的历史估值拼接为同一历史序列。

报告输出：output/pdf/report_04_index_valuation.pdf
JSON输出：output/data/report_04_index_valuation.json

说明：官方 index_dailybasic 文档的覆盖叙述与样例不同（样例包含沪深300），
对每个指数实际尝试并保存返回数据；无数据则保留空值。股息率不在该接口
公开字段中，必须由外部有来源的指数级数据补充。
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"
# symbol 作为稳定ID；TS后缀由index_basic查验后确定（绝不从名称模糊选同名指数）
INDEXES = [
    ("000300", "沪深300", "000300.SH"),
    ("000510", "中证A500", "000510.SH"),
    ("000688", "科创50", "000688.SH"),
    ("399006", "创业板指", "399006.SZ"),
    ("000922", "中证红利", "000922.CSI"),
]
NAMES = {x[0]: x[1] for x in INDEXES}
COLORS = {"text":"#253548","muted":"#718097","line":"#DEE6EE","bg":"#F5F7FA",
          "red":"#C84B56","green":"#238A66","blue":"#4879AC","orange":"#DB8B3C"}
FONT_PATHS = ("/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
              "/Library/Fonts/Arial Unicode.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/noto/wqy/wqy-microhei.ttc")
METRICS = ("pe_ttm", "pb", "dividend_yield")
VALUATION_COLS = ["ts_code", "trade_date", "pe_ttm", "pe", "pb", "dividend_yield", "source", "methodology"]
PRICE_COLS = ["ts_code", "trade_date", "close", "pct_chg", "source"]
DATE_COLS = ["trade_date"]


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


def parse_date(value: Any) -> pd.Timestamp:
    raw = str(value).strip()
    if re.fullmatch(r"\d{8}(?:\.0)?", raw):
        raw = raw[:8]
        dt = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    else:
        dt = pd.to_datetime(raw, errors="coerce")
    if pd.isna(dt):
        raise ValueError("无效日期: %r" % value)
    return dt.normalize()


def daystr(value: Any) -> str:
    return parse_date(value).strftime("%Y%m%d")


def symbol(value: Any) -> str:
    s = str(value).strip().upper()
    s = s.split(".")[0]
    if s not in NAMES:
        return ""
    return s


def load_csv(path: Path, label: str, required: List[str], optional: bool = True) -> pd.DataFrame:
    if not path.exists():
        if optional:
            return pd.DataFrame()
        raise FileNotFoundError("缺少%s：%s。首次使用请运行 --fetch；行业CSV不能代替指数估值。" % (label, path))
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"ts_code":str}, low_memory=False)
    if df.empty:
        return pd.DataFrame()
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError("%s字段缺失：%s (%s)" % (label, sorted(missing), path.name))
    df = df.copy()
    df["trade_date"] = df["trade_date"].map(parse_date)
    df["symbol"] = df["ts_code"].map(symbol)
    df = df[df["symbol"].ne("")].copy()
    if df.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("%s相同指数+日期存在重复：%s" % (label, path.name))
    for col in ("close", "pct_chg", "pe_ttm", "pe", "pb", "dividend_yield", "earnings_coverage_pct", "matched_weight_pct"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def load_prices(root: Path) -> pd.DataFrame:
    primary = root / "index_04_daily.csv"
    fallback = root / "index_daily.csv"
    pieces = []
    # 优先本模块数据；老模块3的指数表仅作不覆盖已有记录的兜底
    for p in (fallback, primary):
        x = load_csv(p, "指数价格", ["ts_code", "trade_date", "close"])
        if not x.empty:
            pieces.append(x)
    if not pieces:
        raise FileNotFoundError("缺少指数日线。请运行 python report_04_index_valuation.py --fetch")
    out = pd.concat(pieces, ignore_index=True, sort=False).drop_duplicates(["symbol", "trade_date"], keep="last")
    out = out.sort_values(["symbol", "trade_date"])
    if out["close"].isna().any() or (out["close"] <= 0).any():
        raise ValueError("指数日线close必须为正且非空")
    return out


def load_valuations(root: Path) -> pd.DataFrame:
    base = load_csv(root / "index_04_valuation.csv", "指数估值", ["ts_code", "trade_date"])
    extra = load_csv(root / "index_04_valuation_extra.csv", "外部历史估值",
                     ["ts_code", "trade_date", "source", "methodology"])
    if not extra.empty:
        if extra["source"].fillna("").astype(str).str.strip().eq("").any() or extra["methodology"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError("外部估值必须逐行填写 source 与 methodology，避免不同口径混用")
    if base.empty and extra.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "symbol"] + list(METRICS) + ["source", "methodology"])
    for x in (base, extra):
        if not x.empty:
            for c in METRICS:
                if c not in x:
                    x[c] = np.nan
            if "source" not in x:
                x["source"] = "Tushare:index_dailybasic"
            if "methodology" not in x:
                x["methodology"] = "Tushare官方指数口径"
    # 按指标融合：同日外部源只补基础源缺失指标，绝不悄悄覆盖已有指标。
    # 对每条指标的来源按独立字段保留，以防“总体来源”掩盖混合口径。
    keys = set()
    for x in (base, extra):
        if not x.empty:
            keys.update((r.symbol, r.trade_date) for r in x.itertuples())
    b = {} if base.empty else {(r.symbol, r.trade_date):r for r in base.itertuples(index=False)}
    e = {} if extra.empty else {(r.symbol, r.trade_date):r for r in extra.itertuples(index=False)}
    records = []
    for key in sorted(keys):
        primary, supplemental = b.get(key), e.get(key)
        row = {"symbol":key[0], "trade_date":key[1], "ts_code": (getattr(primary, "ts_code", "") or getattr(supplemental, "ts_code", ""))}
        for m in METRICS:
            value = getattr(primary, m, np.nan) if primary is not None else np.nan
            src = getattr(primary, "source", "") if primary is not None else ""
            method = getattr(primary, "methodology", "") if primary is not None else ""
            if not pd.notna(value):
                value = getattr(supplemental, m, np.nan) if supplemental is not None else np.nan
                src = getattr(supplemental, "source", "") if supplemental is not None else ""
                method = getattr(supplemental, "methodology", "") if supplemental is not None else ""
            row[m] = value
            row[m + "_source"] = str(src or "")
            row[m + "_methodology"] = str(method or "")
        records.append(row)
    out = pd.DataFrame.from_records(records)
    for m in METRICS:
        out[m] = pd.to_numeric(out[m], errors="coerce")
        if m in ("pe_ttm", "pb"):
            out.loc[out[m] <= 0, m] = np.nan  # PE非正不参与正市盈率分位
        else:
            out.loc[out[m] < 0, m] = np.nan
    if (out["dividend_yield"].dropna() > 100).any():
        raise ValueError("股息率单位应为百分数且≤100，如3.50，不是350")
    return out.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def load_coverage(root: Path) -> pd.DataFrame:
    x = load_csv(root / "index_04_coverage.csv", "盈利覆盖率",
                 ["ts_code", "trade_date", "earnings_coverage_pct", "source", "methodology"])
    # 外部估值里也允许逐日附带盈利覆盖率，方便同一份官方数据一次导入。
    p = root / "index_04_valuation_extra.csv"
    if p.exists():
        y = load_csv(p, "外部历史估值", ["ts_code", "trade_date", "source", "methodology"])
        if not y.empty and "earnings_coverage_pct" in y:
            y = y[y["earnings_coverage_pct"].notna()].copy()
            if not y.empty:
                if "coverage_methodology" in y:
                    y["methodology"] = y["coverage_methodology"].fillna(y["methodology"])
                x = pd.concat(([y] if x.empty else [y, x]), ignore_index=True, sort=False)
                x = x.drop_duplicates(["symbol", "trade_date"], keep="last")
    if not x.empty:
        if ((x["earnings_coverage_pct"] < 0) | (x["earnings_coverage_pct"] > 100)).any():
            raise ValueError("earnings_coverage_pct使用0-100百分数单位")
        if x["source"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError("盈利覆盖率必须有source来源")
    return x


def latest_market_day(root: Path) -> Optional[str]:
    p = root / "market_breadth.csv"
    if not p.exists():
        return None
    x = pd.read_csv(p, encoding="utf-8-sig", usecols=["trade_date"])
    return daystr(x["trade_date"].dropna().iloc[-1]) if not x.empty else None


def percentile(history: pd.Series, current: float) -> Optional[float]:
    if not np.isfinite(current) or len(history) == 0:
        return None
    values = history.to_numpy(dtype=float)
    return round(100.0 * (np.count_nonzero(values < current) + 0.5 * np.count_nonzero(values == current)) / len(values), 1)


def period_stats(df: pd.DataFrame, field: str, anchor: pd.Timestamp, years: int, current: float,
                 source: Optional[str] = None, methodology: Optional[str] = None) -> Dict[str, Any]:
    """同一源同一方法的样本才纳入估值分位，避免方法变更产生伪历史。"""
    since = anchor - pd.DateOffset(years=years)
    hist = df[(df["trade_date"] >= since) & (df["trade_date"] <= anchor)].copy()
    if source is not None:
        hist = hist[hist[field + "_source"].eq(source) & hist[field + "_methodology"].eq(methodology)]
    hist = hist[pd.to_numeric(hist[field], errors="coerce").notna()]
    n = len(hist)
    # 3年/5年完整自然时长+最低样本数双约束，不能把新指数2年历史叫3年分位
    required = int(years * 252 * 0.78)
    cover_ok = n >= required and not hist.empty and hist["trade_date"].min() <= since + pd.Timedelta(days=35)
    if not cover_ok:
        return {"percentile":None, "n":n, "required":required, "complete":False}
    values = hist[field].astype(float)
    return {"percentile":percentile(values, current), "n":n, "required":required, "complete":True,
            "p25":round(float(values.quantile(.25)),3), "median":round(float(values.median()),3),
            "p75":round(float(values.quantile(.75)),3)}


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def assess(prices: pd.DataFrame, vals: pd.DataFrame, cov: pd.DataFrame, asof: pd.Timestamp) -> List[Dict[str, Any]]:
    result = []
    for sym, name, default_code in INDEXES:
        p = prices[(prices.symbol == sym) & (prices.trade_date <= asof)].sort_values("trade_date")
        item: Dict[str,Any] = {"symbol":sym, "name":name, "ts_code":default_code, "report_date":asof.strftime("%Y-%m-%d")}
        item["price"] = {"value":None, "date":None, "status":"缺失", "pct3":None,"pct5":None}
        item["metrics"] = {}
        item["coverage"] = {"value":None,"date":None,"methodology":None,"status":"未提供"}
        if not p.empty:
            current_p = p.iloc[-1]
            px = float(current_p["close"])
            item["ts_code"] = str(current_p.ts_code)
            is_current = current_p.trade_date == asof
            item["price"] = {"value":round(px,4), "date":current_p.trade_date.strftime("%Y-%m-%d"),
                             "status":"当日" if is_current else "滞后", "pct3":period_stats(p,"close",current_p.trade_date,3,px),
                             "pct5":period_stats(p,"close",current_p.trade_date,5,px)}
        v = vals[(vals.symbol == sym) & (vals.trade_date <= asof)]
        for metric in METRICS:
            valid = v[v[metric].notna()].sort_values("trade_date") if not v.empty else pd.DataFrame()
            row = valid.iloc[-1] if not valid.empty else None
            item_m = {"value":None,"date":None,"source":None,"methodology":None,"status":"缺失",
                      "pct3":None,"pct5":None}
            if row is not None:
                anchor = row.trade_date
                val = float(row[metric])
                src = row[metric+"_source"]
                method = row[metric+"_methodology"]
                gap = int((asof - anchor).days)
                status = "当日" if gap == 0 else ("滞后" if gap <= 7 else "过期")
                item_m.update({"value":round(val,4),"date":anchor.strftime("%Y-%m-%d"),"source":src,
                               "methodology":method,"status":status,
                               "pct3":period_stats(valid,metric,anchor,3,val,src,method),
                               "pct5":period_stats(valid,metric,anchor,5,val,src,method)})
            item["metrics"][metric] = item_m
        if not cov.empty:
            current_cov = cov[(cov.symbol == sym) & (cov.trade_date <= asof)].sort_values("trade_date")
            if not current_cov.empty:
                z = current_cov.iloc[-1]
                item["coverage"] = {"value":safe_float(z["earnings_coverage_pct"]),
                                    "date":z.trade_date.strftime("%Y-%m-%d"),
                                    "source":str(z["source"]), "methodology":str(z["methodology"]),
                                    "status":"当日" if z.trade_date == asof else "滞后"}
        result.append(item)
    return result


def pct_str(d: Any) -> str:
    if not isinstance(d, dict) or d.get("percentile") is None:
        return "-"
    return "%.1f%%" % d["percentile"]


def fmt(x: Any, digits: int = 2, suffix: str = "") -> str:
    return ("%%.%df%%s" % digits) % (float(x), suffix) if x is not None else "-"


def cell(ax, x: float, y: float, value: str, size: float = 9.0, color: str = "text", weight: str = "normal", ha: str = "center") -> None:
    ax.text(x, y, value, ha=ha, va="center", fontsize=size, color=COLORS.get(color,color), fontweight=weight, transform=ax.transAxes)


def page_overview(items: List[Dict[str,Any]], asof: pd.Timestamp):
    fig = plt.figure(figsize=(16,9), facecolor=COLORS["bg"])
    fig.text(.04,.953,"指数估值与历史位置",fontsize=23,fontweight="bold",color=COLORS["text"])
    fig.text(.04,.912,"%s  |  价格分位与估值分位并列，低价不等于低估  |  只用于趋势模型辅助" % asof.strftime("%Y-%m-%d"),fontsize=10,color=COLORS["muted"])
    ax = fig.add_axes([.035,.295,.93,.575])
    ax.set_facecolor("white"); ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    headings = ["指数","价格","价格3年/5年","PE(TTM)","PE 3年/5年","PB","PB 3年/5年","股息率","股息3年/5年","有效盈利覆盖"]
    xs = [.065,.16,.273,.395,.505,.596,.692,.783,.867,.955]
    widths = []
    for x,h in zip(xs,headings):cell(ax,x,.92,h,size=8.5,color="muted",weight="bold")
    ax.plot([.02,.98],[.86,.86],transform=ax.transAxes,color=COLORS["line"],lw=1)
    for i, item in enumerate(items):
        yy = .78 - i*.156
        if i%2==0:
            ax.axhspan(yy-.073, yy+.073,xmin=.016,xmax=.985,color="#F5F7FA",zorder=0)
        px = item["price"]
        cells = [item["name"],fmt(px["value"],2),pct_str(px["pct3"])+" / "+pct_str(px["pct5"])]
        for key in METRICS:
            m = item["metrics"][key]
            unit = "%" if key=="dividend_yield" else "x"
            cells += [fmt(m["value"],2,unit),pct_str(m["pct3"])+" / "+pct_str(m["pct5"])]
        c = item["coverage"]
        cells.append(fmt(c["value"],1,"%") if c["value"] is not None else "-")
        # order built: 价格、PE、PB、股息 then 覆盖
        for j,(x,text) in enumerate(zip(xs,cells)):
            cell(ax,x,yy,text,size=8.65 if j not in (2,4,6,8) else 7.9,
                 color="text" if (j==0 or text!="-") else "muted",weight="bold" if j==0 else "normal")
        # 当日估值状态提示；显示每个指数最重要的估值截止日期
        p = item["metrics"]["pb"]
        if p["date"] is not None and p["status"]!="当日":
            cell(ax,xs[0],yy-.050,"PB截至"+p["date"][5:],size=6.5,color="orange")
        if px["status"]!="当日":
            cell(ax,xs[1],yy-.050,"价格未更新",size=6.5,color="orange")
    fig.text(.042,.246,"数据状态",fontsize=11,fontweight="bold",color=COLORS["text"])
    status=[]
    cn={"pe_ttm":"PE", "pb":"PB", "dividend_yield":"股息率"}
    for item in items:
        unavailable=[x for x in METRICS if item["metrics"][x]["value"] is None]
        bad=[x for x in METRICS if item["metrics"][x]["status"] in ("滞后","过期")]
        status.append("%s：%s%s" % (item["name"],"/".join(cn[v] for v in unavailable)+"未提供" if unavailable else "指标已接入", ";滞后"+"/".join(cn[v] for v in bad) if bad else ""))
    for i,s in enumerate(status):
        fig.text(.048+(i%2)*.47,.207-(i//2)*.038,s,fontsize=9.2,color=COLORS["muted"])
    fig.text(.04,.049,"口径：分位=过去同一数据源、同一方法的历史样本位置（中位秩）；需覆盖完整3/5自然年且样本充足。",fontsize=9,color=COLORS["muted"])
    fig.text(.04,.027,"PE非正值不参与PE正值分位；股息率为指数级年度百分数。缺失/滞后数据不自动补值，不生成交易指令。",fontsize=9,color=COLORS["muted"])
    return fig


def chart(ax, data: pd.DataFrame, field: str, title: str, asof: pd.Timestamp, color: str = "blue", source: Optional[str] = None, method: Optional[str] = None):
    ax.set_facecolor("white")
    for spine in ("top","right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left","bottom"):
        ax.spines[spine].set_color(COLORS["line"])
    ax.tick_params(axis="both", labelsize=7, colors=COLORS["muted"],length=2)
    ax.grid(axis="y",color=COLORS["line"],alpha=.5)
    ax.set_title(title,loc="left",fontsize=10,color=COLORS["text"],fontweight="bold")
    use = data[(data.trade_date <= asof) & (data.trade_date >= asof - pd.DateOffset(years=5))].copy()
    if source is not None and not use.empty:
        use = use[use[field+"_source"].eq(source) & use[field+"_methodology"].eq(method)]
    use = use[use[field].notna()] if field in use.columns else pd.DataFrame()
    if use.empty:
        ax.text(.5,.5,"缺少同口径历史",ha="center",va="center",transform=ax.transAxes,color=COLORS["muted"],fontsize=10)
        ax.set_xticks([]);ax.set_yticks([]);return
    ax.plot(use.trade_date,use[field],lw=1.65,color=COLORS[color])
    ax.scatter(use.trade_date.iloc[-1],use[field].iloc[-1],s=22,color=COLORS[color],zorder=5)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax.margins(x=.02)


def page_charts(items: List[Dict[str,Any]], prices: pd.DataFrame, vals: pd.DataFrame, asof: pd.Timestamp,
                subset: List[Dict[str,Any]], page_no: int):
    fig = plt.figure(figsize=(16,9),facecolor=COLORS["bg"])
    fig.text(.04,.953,"价格与估值的历史变化 | %s" % asof.strftime("%Y-%m-%d"),fontsize=18,fontweight="bold",color=COLORS["text"])
    fig.text(.04,.918,"左：价格  |  中：PE(TTM)  |  右：PB ；同一条估值曲线只使用相同来源/口径。",fontsize=9.5,color=COLORS["muted"])
    top, row_h = .835, .248
    for i,item in enumerate(subset):
        y = top-i*row_h
        fig.text(.042,y+.019,item["name"]+"  "+item["ts_code"],fontsize=11,fontweight="bold",color=COLORS["text"])
        px=prices[prices.symbol.eq(item["symbol"])]
        va=vals[vals.symbol.eq(item["symbol"])]
        for j,(key,title) in enumerate((("close","价格（点）"),("pe_ttm","PE(TTM)") ,("pb","PB"))):
            ax=fig.add_axes((.045+j*.315,y-.168,.285,.145))
            if key=="close":
                chart(ax,px,key,title,asof,"blue")
            else:
                m=item["metrics"][key]
                chart(ax,va,key,title,asof,"orange" if key=="pe_ttm" else "green",m["source"],m["methodology"])
                if m["date"] and m["status"]!="当日":
                    ax.text(.98,.94,"截至 "+m["date"][5:],ha="right",va="top",transform=ax.transAxes,fontsize=7,color=COLORS["orange"])

    fig.text(.04,.037,"行业轮动CSV中的申万行业PE/PB不是宽基指数PE/PB；历史不足时分位留空。第%d页" % page_no,fontsize=8.5,color=COLORS["muted"])
    return fig


def make_report(items: List[Dict[str,Any]], prices: pd.DataFrame, vals: pd.DataFrame, asof: pd.Timestamp,
                outfile: Path, demo: bool = False) -> None:
    configure_font()
    outfile.parent.mkdir(parents=True,exist_ok=True)
    with PdfPages(outfile) as pdf:
        figures = [page_overview(items,asof),page_charts(items,prices,vals,asof,items[:3],2),
                   page_charts(items,prices,vals,asof,items[3:],3)]
        for fig in figures:
            if demo:
                fig.text(.5,.5,"演示模拟数据 · 非真实行情",ha="center",va="center",fontsize=39,
                         color=COLORS["orange"],alpha=.15,rotation=22,fontweight="bold")
                fig.text(.963,.956,"DEMO / NOT REAL DATA",ha="right",fontsize=10,color=COLORS["orange"])
            pdf.savefig(fig,facecolor=fig.get_facecolor())
            plt.close(fig)


def read_config_token(root:Path) -> Optional[str]:
    env=os.environ.get("TUSHARE_TOKEN","").strip()
    if env:
        return env
    candidates = (root / "config.yaml", ROOT / "config.yaml", root.parent / "config.yaml")
    p = next((candidate for candidate in candidates if candidate.exists()), None)
    if p is None:return None
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("检测到config.yaml但未安装PyYAML；运行 pip install pyyaml") from exc
    obj=yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return str((obj.get("keys") or {}).get("tushare") or "").strip() or None


def create_client(root:Path):
    try:import tushare as ts
    except ImportError as exc:raise RuntimeError("缺少tushare库，请安装 pip install tushare") from exc
    token=read_config_token(root)
    try:return ts.pro_api(token) if token else ts.pro_api()
    except Exception as exc:raise RuntimeError("Tushare初始化失败；请检查同目录config.yaml的keys.tushare") from exc


def api_retry(pro, method: str, **kwargs) -> pd.DataFrame:
    last=None
    for attempt in range(3):
        try:
            ans=getattr(pro,method)(**kwargs)
            if ans is None:return pd.DataFrame()
            if not isinstance(ans,pd.DataFrame):raise TypeError("Tushare返回非DataFrame")
            return ans.copy()
        except Exception as exc:
            last=exc
            if attempt<2:time.sleep(.7*(attempt+1))
    raise RuntimeError("%s %s失败（3次）：%s" % (method,kwargs,last))


def resolve_code(pro, sym: str, preferred: str, name: str) -> str:
    """只接受确切代码和名称；避免000922与上证红利混淆。"""
    try:
        basic=api_retry(pro,"index_basic",symbol=sym)
    except Exception as exc:
        print("  提示：无法核查%s指数代码，沿用显式预设%s：%s" % (name,preferred,exc))
        return preferred
    if basic.empty or "ts_code" not in basic or "name" not in basic:
        print("  提示：index_basic未返回%s，沿用预设%s" % (sym,preferred));return preferred
    basic=basic[basic.ts_code.astype(str).str.split(".").str[0].eq(sym)]
    exact=basic[basic.name.astype(str).str.replace(" ","",regex=False).eq(name)]
    if len(exact)==1:return str(exact.iloc[0].ts_code)
    if preferred in basic.ts_code.astype(str).values:return preferred
    print("  注意：无法唯一确定%s指数交易所后缀；沿用预设%s，下载为空时将保留缺失。" % (name,preferred))
    return preferred


def chunk_dates(start: str, end: str, days:int=480):
    t=parse_date(start);e=parse_date(end)
    while t<=e:
        last=min(e,t+pd.Timedelta(days=days-1))
        yield t.strftime("%Y%m%d"),last.strftime("%Y%m%d")
        t=last+pd.Timedelta(days=1)


def save_merge(path:Path, incoming:List[pd.DataFrame], required:List[str]) -> None:
    usable=[x for x in incoming if x is not None and not x.empty]
    if not usable:return
    old=pd.read_csv(path,encoding="utf-8-sig",dtype={"ts_code":str}) if path.exists() else pd.DataFrame()
    parts=([old] if not old.empty else [])+usable
    out=pd.concat(parts,ignore_index=True,sort=False)
    if not set(required).issubset(out.columns):raise ValueError("%s缺少字段%s" % (path.name,set(required)-set(out.columns)))
    out["trade_date"]=out["trade_date"].map(daystr)
    out["symbol"]=out["ts_code"].map(symbol)
    out=out[out.symbol.ne("")].drop_duplicates(["symbol","trade_date"],keep="last").sort_values(["symbol","trade_date"])
    out=out.drop(columns=["symbol"])
    tmp=path.with_suffix(path.suffix+".tmp")
    out.to_csv(tmp,index=False,encoding="utf-8-sig")
    os.replace(tmp,path)
    print("  已保存%s：%d行" % (path.name,len(out)))


def fetch_coverage_snapshot(pro, root: Path, target: str) -> None:
    """以最新可得权重及当日PE(TTM)>0估算科创50有效盈利权重，不伪称官方指数PE。"""
    code="000688.SH"
    start=daystr(parse_date(target)-pd.Timedelta(days=70))
    try:
        w=api_retry(pro,"index_weight",index_code=code,start_date=start,end_date=target)
        b=api_retry(pro,"daily_basic",trade_date=target,fields="ts_code,trade_date,pe_ttm")
        if w.empty or b.empty or not {"con_code","trade_date","weight"}.issubset(w.columns) or not {"ts_code","pe_ttm"}.issubset(b.columns):
            print("  提示：科创50盈利覆盖缺少权重/成分股PE，不填写估算值")
            return
        w=w.copy();w["trade_date"]=w.trade_date.map(daystr)
        w=w[w.trade_date.le(target)]
        if w.empty:return
        weight_date=w.trade_date.max()
        if (parse_date(target)-parse_date(weight_date)).days>45:
            print("  提示：科创50成分权重披露太旧（%s），不计算盈利覆盖" % weight_date)
            return
        w=w[w.trade_date.eq(weight_date)].drop_duplicates("con_code",keep="last")
        w["weight"]=pd.to_numeric(w["weight"],errors="coerce")
        w=w.dropna(subset=["weight"])
        den=w.weight.sum()
        if not 90<=den<=110:
            print("  提示：科创50最新权重总和异常，未计算盈利覆盖");return
        b=b.copy();b["pe_ttm"]=pd.to_numeric(b["pe_ttm"],errors="coerce")
        b=b.drop_duplicates("ts_code",keep="last")
        joined=w.merge(b[["ts_code","pe_ttm"]],left_on="con_code",right_on="ts_code",how="left")
        matched=(joined.loc[joined.pe_ttm.notna(),"weight"].sum()/den)*100
        # PE(TTM)为正可作为盈利贡献覆盖的代理，但不等同严格公司盈利披露率。
        positive=(joined.loc[joined.pe_ttm.gt(0),"weight"].sum()/den)*100
        if matched<85:
            print("  提示：科创50个股PE覆盖权重不足85%%，未输出盈利覆盖（实测%.1f%%）" % matched)
            return
        row=pd.DataFrame([{"ts_code":code,"trade_date":target,"earnings_coverage_pct":round(positive,3),
                           "matched_weight_pct":round(matched,3),"weight_date":weight_date,
                           "source":"Tushare:index_weight+daily_basic",
                           "methodology":"最近披露权重(%s)下PE(TTM)>0成分权重/全部成分权重，代理指标" % weight_date}])
        save_merge(root/"index_04_coverage.csv",[row],["ts_code","trade_date"])
    except Exception as exc:
        print("  提示：盈利覆盖率可选计算失败，不影响主报告：%s" % exc)

def fetch_data(root:Path, target:str, bootstrap:str, overlap:int, pro=None, coverage:bool=True) -> None:
    if pro is None:pro=create_client(root)
    statefile=root/"index_04_fetch_state.json"
    state=json.loads(statefile.read_text(encoding="utf-8")) if statefile.exists() else {}
    price_in, val_in = [], []
    for sym,name,pref in INDEXES:
        code=resolve_code(pro,sym,pref,name)
        st=state.get(sym,{})
        prior=st.get("price_success_end")
        begin=bootstrap if prior is None else max(bootstrap,daystr(parse_date(prior)-pd.Timedelta(days=overlap)))
        had_error=False; fetched_any=False
        print("[价格] %s %s  %s~%s" % (name,code,begin,target))
        for lo,hi in chunk_dates(begin,target):
            try:
                df=api_retry(pro,"index_daily",ts_code=code,start_date=lo,end_date=hi)
                if not df.empty:
                    if not {"ts_code","trade_date","close"}.issubset(df.columns):raise ValueError("index_daily返回字段不完整")
                    df["source"]="Tushare:index_daily"
                    price_in.append(df);fetched_any=True
            except Exception as exc:
                had_error=True;print("  ERROR: %s；已有历史不会清除" % exc)
            time.sleep(.2)
        if not had_error and fetched_any:st["price_success_end"]=target
        elif not fetched_any:print("  提示：%s无价格返回，检查交易所代码或是否有行情" % code)
        # 尝试每个指数；非覆盖品种可能正常返回空表，禁止代填
        prior_v=st.get("valuation_success_end")
        # 历史从未返回的指数，初次全量探测后仅检查近期，避免每天重抓5年空接口。
        if prior_v:
            vbegin=max(bootstrap,daystr(parse_date(prior_v)-pd.Timedelta(days=overlap)))
        elif st.get("valuation_checked_end"):
            vbegin=max(bootstrap,daystr(parse_date(target)-pd.Timedelta(days=60)))
        else:
            vbegin=bootstrap
        anyv=False;v_error=False
        for lo,hi in chunk_dates(vbegin,target,days=750):
            try:
                x=api_retry(pro,"index_dailybasic",ts_code=code,start_date=lo,end_date=hi)
                if not x.empty:
                    if not {"ts_code","trade_date"}.issubset(x.columns):raise ValueError("index_dailybasic返回缺少主键")
                    z=pd.DataFrame({"ts_code":x.ts_code,"trade_date":x.trade_date,
                                    "pe_ttm":x["pe_ttm"] if "pe_ttm" in x else np.nan,
                                    "pe":x["pe"] if "pe" in x else np.nan,
                                    "pb":x["pb"] if "pb" in x else np.nan,
                                    "dividend_yield":np.nan,
                                    "source":"Tushare:index_dailybasic", "methodology":"Tushare官方指数口径"})
                    val_in.append(z);anyv=True
            except Exception as exc:
                v_error=True;print("  提示：%s的index_dailybasic不可用：%s" % (name,exc))
                break
            time.sleep(.2)
        if not v_error:st["valuation_checked_end"]=target
        if anyv and not v_error:st["valuation_success_end"]=target
        if not anyv:print("  注意：%s暂未获得官方指数估值；不会使用股票PE简单平均替代。" % name)
        state[sym]=st
    save_merge(root/"index_04_daily.csv",price_in,["ts_code","trade_date","close"])
    save_merge(root/"index_04_valuation.csv",val_in,["ts_code","trade_date"])
    if coverage:fetch_coverage_snapshot(pro,root,target)
    tmp=statefile.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    os.replace(tmp,statefile)


def cli():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--directory",default=str(DATA_DIR),help="CSV/数据输出目录，默认项目 data 目录")
    ap.add_argument("--date",help="报告预期交易日YYYYMMDD，默认market_breadth.csv最新日期")
    ap.add_argument("--fetch",action="store_true",help="先根据同目录config.yaml从Tushare增量下载指数日线与可用指数估值")
    ap.add_argument("--start-date",help="首次回溯起点，默认报告日前5年+约45天")
    ap.add_argument("--overlap-days",type=int,default=14)
    ap.add_argument("--skip-coverage",action="store_true",help="不尝试获取科创50最新成分盈利覆盖率")
    ap.add_argument("--full-refresh",action="store_true",help="在--fetch时忽略增量断点，按--start-date重新获取历史")
    ap.add_argument("--output",help="PDF输出路径")
    ap.add_argument("--summary-json",help="JSON输出路径")
    return ap.parse_args()


def main():
    args=cli()
    root=Path(args.directory).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    target=daystr(args.date) if args.date else (latest_market_day(root) or datetime.now().strftime("%Y%m%d"))
    target_dt=parse_date(target)
    bootstrap=daystr(args.start_date) if args.start_date else daystr(target_dt-pd.DateOffset(years=5)-pd.Timedelta(days=45))
    if bootstrap>target:raise ValueError("--start-date不可晚于--date")
    if args.overlap_days<1:raise ValueError("--overlap-days至少1")
    if args.fetch:
        if args.full_refresh:
            state_path=root/"index_04_fetch_state.json"
            if state_path.exists():state_path.unlink()
        fetch_data(root,target,bootstrap,args.overlap_days,coverage=not args.skip_coverage)
    prices=load_prices(root)
    vals=load_valuations(root)
    cov=load_coverage(root)
    items=assess(prices,vals,cov,target_dt)
    if all(x["price"]["value"] is None for x in items):
        raise ValueError("指定日期之前五大指数均无真实价格记录，拒绝生成空报告")
    out=Path(args.output) if args.output else root/"pdf/report_04_index_valuation.pdf"
    js=Path(args.summary_json) if args.summary_json else root/"report_04_index_valuation.json"
    make_report(items,prices,vals,target_dt,out)
    js.parent.mkdir(parents=True,exist_ok=True)
    js.write_text(json.dumps({"trade_date":target_dt.strftime("%Y-%m-%d"),"indices":items,
                              "note":"来源/方法跨期不拼接；分位不足3/5年留空；不生成交易指令"},ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print("报告：%s" % out)
    print("摘要：%s" % js)
    for i in items:
        print("%s 价格=%s  PE=%s PB=%s 股息=%s" % (i["name"],i["price"]["value"],
              i["metrics"]["pe_ttm"]["value"],i["metrics"]["pb"]["value"],i["metrics"]["dividend_yield"]["value"]))


if __name__=="__main__":
    try:main()
    except (RuntimeError,ValueError,FileNotFoundError,KeyError) as exc:
        print("ERROR: %s" % exc,file=sys.stderr)
        sys.exit(1)
