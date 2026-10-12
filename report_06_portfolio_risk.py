#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模块 06：组合风险与交易条件（Python 3.9+）。

模块 01-05 负责市场环境、行业、ETF、估值与风格；模块 06 是组合级风险约束层。
它不重新计算单资产技术信号，而是把已有“模型仓位 / 执行仓位 / 冷静期”与
实际持仓、人工风险上限、组合因子暴露和资产重叠放在同一张风险表中。

核心原则
--------
1. 模型允许仓位 != 实际愿意承担仓位；人工上限只能收紧，不能放宽执行仓位。
2. 各资产分别满足加仓条件，不代表组合仍有风险容量；组合因子硬上限再次约束。
3. 不从资产名称猜测成分/行业重叠。科技成长、主题、红利等穿透暴露必须显式输入。
4. 两两重叠矩阵只做“重复风险指示”，避免多资产重复计算后机械扣减总权益仓位。
5. 缺失数据保持缺失；不使用上一日仓位、净值或暴露值伪造报告日数据。
6. 输出“最终允许加仓额度”是风险容量，不是自动交易指令。

输入文件（默认均位于 data/）
-------------------------
portfolio_positions.csv [必需]
    asset_id,asset_name,market_value,budget_weight,manual_cap_position
    可选：theme,asset_class,enabled

    market_value           当前实际持仓市值（统一币种/单位）
    budget_weight          该资产模型100%仓位对应“整个组合净值”的预算权重
    manual_cap_position    人工愿意承担的该资产预算仓位上限

portfolio_signals.csv [必需]
    trade_date,asset_id,model_position,execution_position
    可选：cooldown_days,signal,execution_status

    model_position / execution_position 接受0-1或0-100。
    本模块不重算原 v3 信号，只读取已形成的模型/执行仓位。

portfolio_exposure_map.csv [必需]
    asset_id,<动态风险因子列...>

    例如：domestic_equity, tech_growth, dividend_defense, hk_dividend,
          theme_chip, theme_broad_growth
    每个暴露系数范围0-1（也接受0-100）。
    当前因子敞口 = SUM(资产当前组合权重 × 因子暴露系数)。

portfolio_risk_limits.csv [必需]
    risk_item,label,limit_type,soft_limit,hard_limit

    limit_type=max：最大暴露约束；min：最低配置提示。
    特殊 risk_item：
      portfolio_drawdown  组合当前回撤阈值；达到hard_limit后冻结新增风险仓位。
      single_theme_max    对所有 theme_ 开头且未单独配置的因子应用统一上限。

portfolio_nav.csv [必需]
    trade_date,nav
    nav 为整个被监控组合总净值/总资产价值；用于实际权重和组合回撤。

portfolio_overlap_matrix.csv [可选]
    asset_a,asset_b,overlap
    overlap 取0-1（也接受0-100），由上游真实成分/行业重叠显式提供。
    重复风险指示 = min(weight_a, weight_b) × overlap。

输出
----
data/pdf/report_06_portfolio_risk.pdf
data/report_06_portfolio_risk.json

示例
----
python report_06_portfolio_risk.py --date 20260924

首次可生成输入模板：
python report_06_portfolio_risk.py --write-templates
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"

TEXT = "#26374B"
MUTED = "#6E7E93"
BG = "#F5F7FB"
PANEL = "#FFFFFF"
GRID = "#DDE5EE"
RED = "#C74D58"
GREEN = "#268C6C"
BLUE = "#477BAF"
ORANGE = "#DE933E"
PURPLE = "#896CB1"

FONT_PATHS = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)

POSITION_REQUIRED = {
    "asset_id", "asset_name", "market_value", "budget_weight", "manual_cap_position"
}
SIGNAL_REQUIRED = {
    "trade_date", "asset_id", "model_position", "execution_position"
}
LIMIT_REQUIRED = {
    "risk_item", "label", "limit_type", "soft_limit", "hard_limit"
}
NAV_REQUIRED = {"trade_date", "nav"}


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
    return pd.Timestamp(dt).normalize()


def require_columns(frame: pd.DataFrame, required: set, label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("%s 缺少字段：%s" % (label, sorted(missing)))


def assert_unique(frame: pd.DataFrame, cols: List[str], label: str) -> None:
    dup = frame.duplicated(cols, keep=False)
    if dup.any():
        sample = frame.loc[dup, cols].head(5).to_dict("records")
        raise ValueError("%s 键 %s 出现重复：%s" % (label, cols, sample))


def normalize_id(frame: pd.DataFrame, col: str = "asset_id") -> None:
    frame[col] = frame[col].astype("string").str.strip()
    if frame[col].isna().any() or frame[col].eq("").any():
        raise ValueError("%s 存在空值" % col)


def parse_fraction_series(series: pd.Series, label: str, allow_nan: bool = False) -> pd.Series:
    """接受0-1或0-100；绝对值>1.5时按百分数除以100。"""
    values = pd.to_numeric(series, errors="coerce")
    if not allow_nan and values.isna().any():
        raise ValueError("%s 存在非数字或缺失值" % label)
    mask = values.abs() > 1.5
    values = values.where(~mask, values / 100.0)
    bad = values.dropna().loc[(values.dropna() < 0) | (values.dropna() > 1)]
    if not bad.empty:
        raise ValueError("%s 必须在0-1或0-100范围内：%s" % (label, bad.head(3).tolist()))
    return values.astype(float)


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("缺少%s：%s" % (label, path))
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if frame.empty:
        raise ValueError("%s 为空：%s" % (label, path))
    return frame


def read_positions(path: Path) -> pd.DataFrame:
    frame = read_csv(path, "组合实际持仓")
    require_columns(frame, POSITION_REQUIRED, "组合实际持仓")
    normalize_id(frame)
    assert_unique(frame, ["asset_id"], "组合实际持仓")
    frame["asset_name"] = frame["asset_name"].fillna(frame["asset_id"]).astype(str).str.strip()
    frame["market_value"] = pd.to_numeric(frame["market_value"], errors="coerce")
    if frame["market_value"].isna().any() or (frame["market_value"] < 0).any():
        raise ValueError("market_value 必须为非负数字")
    frame["budget_weight"] = parse_fraction_series(frame["budget_weight"], "budget_weight")
    frame["manual_cap_position"] = parse_fraction_series(
        frame["manual_cap_position"], "manual_cap_position"
    )
    for col, default in (("theme", ""), ("asset_class", ""), ("enabled", True)):
        if col not in frame:
            frame[col] = default
    if frame["enabled"].dtype != bool:
        frame["enabled"] = (
            frame["enabled"].astype(str).str.strip().str.lower()
            .isin(("1", "true", "yes", "y", "是", "启用"))
        )
    return frame.reset_index(drop=True)


def read_signals(path: Path) -> pd.DataFrame:
    frame = read_csv(path, "模型/执行仓位")
    require_columns(frame, SIGNAL_REQUIRED, "模型/执行仓位")
    normalize_id(frame)
    frame["trade_date"] = frame["trade_date"].map(parse_date)
    assert_unique(frame, ["trade_date", "asset_id"], "模型/执行仓位")
    frame["model_position"] = parse_fraction_series(frame["model_position"], "model_position")
    frame["execution_position"] = parse_fraction_series(
        frame["execution_position"], "execution_position"
    )
    if "cooldown_days" not in frame:
        frame["cooldown_days"] = 0
    frame["cooldown_days"] = pd.to_numeric(frame["cooldown_days"], errors="coerce").fillna(0).astype(int)
    for col in ("signal", "execution_status"):
        if col not in frame:
            frame[col] = ""
        frame[col] = frame[col].fillna("").astype(str).str.strip()
    return frame.sort_values(["trade_date", "asset_id"]).reset_index(drop=True)


def read_exposure_map(path: Path, positions: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    frame = read_csv(path, "组合风险因子映射")
    if "asset_id" not in frame:
        raise ValueError("组合风险因子映射必须包含 asset_id")
    normalize_id(frame)
    assert_unique(frame, ["asset_id"], "组合风险因子映射")
    factors = [c for c in frame.columns if c != "asset_id"]
    if not factors:
        raise ValueError("组合风险因子映射至少需要一个风险因子列")
    for factor in factors:
        frame[factor] = parse_fraction_series(frame[factor], "暴露因子 %s" % factor)
    missing = sorted(set(positions["asset_id"]) - set(frame["asset_id"]))
    if missing:
        raise ValueError("以下持仓资产未定义风险因子暴露：%s" % ", ".join(missing))
    return frame, factors


def read_limits(path: Path) -> pd.DataFrame:
    frame = read_csv(path, "组合风险上限")
    require_columns(frame, LIMIT_REQUIRED, "组合风险上限")
    frame["risk_item"] = frame["risk_item"].astype(str).str.strip()
    frame["label"] = frame["label"].fillna(frame["risk_item"]).astype(str).str.strip()
    frame["limit_type"] = frame["limit_type"].astype(str).str.strip().str.lower()
    if not frame["limit_type"].isin(("max", "min")).all():
        bad = frame.loc[~frame["limit_type"].isin(("max", "min")), "limit_type"].unique().tolist()
        raise ValueError("limit_type 只允许 max/min，发现：%s" % bad)
    frame["soft_limit"] = parse_fraction_series(frame["soft_limit"], "soft_limit", allow_nan=True)
    frame["hard_limit"] = parse_fraction_series(frame["hard_limit"], "hard_limit", allow_nan=True)
    assert_unique(frame, ["risk_item"], "组合风险上限")
    for _, row in frame.iterrows():
        soft, hard, kind = row["soft_limit"], row["hard_limit"], row["limit_type"]
        if kind == "max" and pd.notna(soft) and pd.notna(hard) and soft > hard:
            raise ValueError("%s 为 max 时 soft_limit 不应高于 hard_limit" % row["risk_item"])
        if kind == "min" and pd.notna(soft) and pd.notna(hard) and soft < hard:
            raise ValueError("%s 为 min 时 soft_limit 应 >= hard_limit" % row["risk_item"])
    return frame


def read_nav(path: Path) -> pd.DataFrame:
    frame = read_csv(path, "组合净值")
    require_columns(frame, NAV_REQUIRED, "组合净值")
    frame["trade_date"] = frame["trade_date"].map(parse_date)
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    if frame["nav"].isna().any() or (frame["nav"] <= 0).any():
        raise ValueError("组合净值 nav 必须为正数")
    assert_unique(frame, ["trade_date"], "组合净值")
    return frame.sort_values("trade_date").reset_index(drop=True)


def read_overlap(path: Optional[Path], positions: pd.DataFrame) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        print("提示：未提供 portfolio_overlap_matrix.csv；两两重叠指标显示缺失")
        return None
    frame = read_csv(path, "两两重叠矩阵")
    require_columns(frame, {"asset_a", "asset_b", "overlap"}, "两两重叠矩阵")
    for col in ("asset_a", "asset_b"):
        frame[col] = frame[col].astype(str).str.strip()
    if (frame["asset_a"] == frame["asset_b"]).any():
        raise ValueError("两两重叠矩阵不应包含资产自身")
    frame["overlap"] = parse_fraction_series(frame["overlap"], "overlap")
    known = set(positions["asset_id"])
    unknown = sorted((set(frame["asset_a"]) | set(frame["asset_b"])) - known)
    if unknown:
        raise ValueError("重叠矩阵包含未知 asset_id：%s" % unknown)
    key = frame.apply(lambda r: "||".join(sorted((r["asset_a"], r["asset_b"]))), axis=1)
    if key.duplicated().any():
        raise ValueError("重叠矩阵存在重复无向资产对（A-B 与 B-A 视为同一对）")
    return frame.reset_index(drop=True)


def choose_asof(signals: pd.DataFrame, requested: Optional[str]) -> pd.Timestamp:
    if requested:
        return parse_date(requested)
    return signals["trade_date"].max()


def choose_nav_row(nav: pd.DataFrame, asof: pd.Timestamp, max_lag_days: int) -> Tuple[pd.Series, int]:
    eligible = nav.loc[nav["trade_date"] <= asof]
    if eligible.empty:
        raise ValueError("组合净值在报告日前没有记录")
    row = eligible.iloc[-1]
    lag = int((asof - row["trade_date"]).days)
    if lag > max_lag_days:
        raise ValueError(
            "组合净值最新记录 %s 距报告日 %d 天，超过 --nav-max-lag=%d"
            % (row["trade_date"].strftime("%Y-%m-%d"), lag, max_lag_days)
        )
    if lag:
        print("提示：报告日没有组合净值，使用 %s（滞后%d天）" % (row["trade_date"].strftime("%Y-%m-%d"), lag))
    return row, lag


def limits_dict(limits: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in limits.iterrows():
        out[str(row["risk_item"])] = {
            "label": row["label"],
            "limit_type": row["limit_type"],
            "soft_limit": row["soft_limit"],
            "hard_limit": row["hard_limit"],
        }
    return out


def factor_limit(factor: str, lookup: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if factor in lookup:
        return lookup[factor]
    if factor.startswith("theme_") and "single_theme_max" in lookup:
        cfg = dict(lookup["single_theme_max"])
        cfg["label"] = factor.replace("theme_", "主题·")
        return cfg
    return None


def classify(value: float, cfg: Optional[Dict[str, Any]]) -> str:
    if cfg is None or pd.isna(value):
        return "未设限"
    soft, hard, kind = cfg["soft_limit"], cfg["hard_limit"], cfg["limit_type"]
    if kind == "max":
        if pd.notna(hard) and value >= hard - 1e-12:
            return "硬上限"
        if pd.notna(soft) and value >= soft - 1e-12:
            return "预警"
        return "正常"
    if pd.notna(hard) and value <= hard + 1e-12:
        return "严重不足"
    if pd.notna(soft) and value <= soft + 1e-12:
        return "不足"
    return "正常"


def status_color(status: str) -> str:
    if status in ("硬上限", "严重不足", "超目标"):
        return RED
    if status in ("预警", "不足", "受组合约束"):
        return ORANGE
    if status in ("正常", "可加仓"):
        return GREEN
    return MUTED


def assemble_assets(
    positions: pd.DataFrame,
    signals: pd.DataFrame,
    exposure_map: pd.DataFrame,
    factors: List[str],
    asof: pd.Timestamp,
    portfolio_nav: float,
) -> pd.DataFrame:
    today = signals.loc[signals["trade_date"] == asof].copy()
    enabled_ids = set(positions.loc[positions["enabled"], "asset_id"])
    missing = sorted(enabled_ids - set(today["asset_id"]))
    if missing:
        raise ValueError("报告日缺少以下启用资产的模型/执行仓位：%s" % missing)

    merged = positions.merge(
        today[[
            "asset_id", "model_position", "execution_position", "cooldown_days",
            "signal", "execution_status"
        ]],
        on="asset_id", how="left", validate="one_to_one",
    )
    for col in ("model_position", "execution_position"):
        merged[col] = merged[col].fillna(0.0)
    merged["cooldown_days"] = merged["cooldown_days"].fillna(0).astype(int)
    merged["signal"] = merged["signal"].fillna("")
    merged["execution_status"] = merged["execution_status"].fillna("")

    merged = merged.merge(exposure_map, on="asset_id", how="left", validate="one_to_one")
    if merged[factors].isna().any().any():
        raise ValueError("风险因子映射合并后出现缺失值")

    merged["current_weight"] = merged["market_value"] / portfolio_nav
    merged["model_target_weight"] = merged["budget_weight"] * merged["model_position"]
    merged["execution_target_weight"] = merged["budget_weight"] * merged["execution_position"]
    merged["policy_position"] = np.minimum(
        merged["execution_position"], merged["manual_cap_position"]
    )
    merged["policy_target_weight"] = merged["budget_weight"] * merged["policy_position"]
    merged["raw_add_weight"] = (
        merged["policy_target_weight"] - merged["current_weight"]
    ).clip(lower=0.0)
    merged["over_policy_weight"] = (
        merged["current_weight"] - merged["policy_target_weight"]
    ).clip(lower=0.0)
    return merged


def factor_exposure(assets: pd.DataFrame, factors: List[str], weight_col: str) -> Dict[str, float]:
    return {factor: float((assets[weight_col] * assets[factor]).sum()) for factor in factors}


def calculate_drawdown(nav: pd.DataFrame, asof: pd.Timestamp) -> Tuple[float, float, pd.DataFrame]:
    hist = nav.loc[nav["trade_date"] <= asof].copy()
    if hist.empty:
        return np.nan, np.nan, hist
    hist["peak"] = hist["nav"].cummax()
    hist["drawdown"] = hist["nav"] / hist["peak"] - 1.0
    return float(hist.iloc[-1]["drawdown"]), float(hist["drawdown"].min()), hist


def build_factor_table(
    assets: pd.DataFrame, factors: List[str], lookup: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    current = factor_exposure(assets, factors, "current_weight")
    model = factor_exposure(assets, factors, "model_target_weight")
    execution = factor_exposure(assets, factors, "execution_target_weight")
    policy = factor_exposure(assets, factors, "policy_target_weight")
    rows = []
    for factor in factors:
        cfg = factor_limit(factor, lookup)
        rows.append({
            "risk_item": factor,
            "label": cfg["label"] if cfg else factor,
            "current": current[factor],
            "model": model[factor],
            "execution": execution[factor],
            "policy": policy[factor],
            "limit_type": cfg["limit_type"] if cfg else "",
            "soft_limit": cfg["soft_limit"] if cfg else np.nan,
            "hard_limit": cfg["hard_limit"] if cfg else np.nan,
            "status": classify(current[factor], cfg),
        })
    return pd.DataFrame(rows)


def apply_portfolio_caps(
    assets: pd.DataFrame,
    factors: List[str],
    lookup: Dict[str, Dict[str, Any]],
    current_drawdown: float,
    portfolio_nav: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """把人工约束后的原始加仓空间进一步压缩到组合硬上限以内。"""
    result = assets.copy()
    result["allowed_add_weight"] = result["raw_add_weight"].astype(float)
    result["asset_status"] = "持有/观察"
    result.loc[result["over_policy_weight"] > 1e-10, "asset_status"] = "超目标"
    notes: List[str] = []

    draw_cfg = lookup.get("portfolio_drawdown")
    if draw_cfg and draw_cfg["limit_type"] == "max" and pd.notna(draw_cfg["hard_limit"]):
        dd_abs = abs(min(float(current_drawdown), 0.0))
        if dd_abs >= float(draw_cfg["hard_limit"]) - 1e-12:
            result["allowed_add_weight"] = 0.0
            result["allowed_add_value"] = 0.0
            result.loc[result["over_policy_weight"] <= 1e-10, "asset_status"] = "受组合约束"
            notes.append(
                "当前回撤 %.1f%% 已达到/超过硬阈值 %.1f%%：冻结全部新增风险仓位"
                % (dd_abs * 100, float(draw_cfg["hard_limit"]) * 100)
            )
            return result, notes

    current_factor = factor_exposure(result, factors, "current_weight")
    hard_limits: Dict[str, float] = {}
    for factor in factors:
        cfg = factor_limit(factor, lookup)
        if cfg and cfg["limit_type"] == "max" and pd.notna(cfg["hard_limit"]):
            hard_limits[factor] = float(cfg["hard_limit"])

    # 多因子交叉暴露时，逐因子比例收紧，迭代到可行区域。
    for _ in range(20):
        changed = False
        for factor, hard in hard_limits.items():
            capacity = max(hard - current_factor.get(factor, 0.0), 0.0)
            demand_series = result["allowed_add_weight"] * result[factor]
            demand = float(demand_series.sum())
            if demand <= capacity + 1e-12:
                continue
            affected = result[factor] > 0
            if not affected.any():
                continue
            scale = 0.0 if demand <= 0 else min(max(capacity / demand, 0.0), 1.0)
            result.loc[affected, "allowed_add_weight"] *= scale
            changed = True
            notes.append(
                "%s：新增因子需求 %.1f%% > 剩余额度 %.1f%%，相关资产新增额度按 %.1f%% 收紧"
                % (factor, demand * 100, capacity * 100, scale * 100)
            )
        if not changed:
            break

    result["allowed_add_value"] = result["allowed_add_weight"] * portfolio_nav
    constrained = (
        (result["raw_add_weight"] > 1e-10)
        & (result["allowed_add_weight"] + 1e-10 < result["raw_add_weight"])
    )
    result.loc[constrained, "asset_status"] = "受组合约束"
    addable = (result["allowed_add_weight"] > 1e-10) & ~constrained
    result.loc[addable, "asset_status"] = "可加仓"
    return result, notes


def calculate_overlap(
    overlap: Optional[pd.DataFrame], assets: pd.DataFrame
) -> Tuple[pd.DataFrame, float]:
    if overlap is None or overlap.empty:
        return pd.DataFrame(), np.nan
    weights = assets.set_index("asset_id")["current_weight"].to_dict()
    names = assets.set_index("asset_id")["asset_name"].to_dict()
    rows = []
    for _, r in overlap.iterrows():
        a, b, ov = r["asset_a"], r["asset_b"], float(r["overlap"])
        duplicate = min(float(weights.get(a, 0.0)), float(weights.get(b, 0.0))) * ov
        rows.append({
            "asset_a": a,
            "asset_b": b,
            "name_a": names.get(a, a),
            "name_b": names.get(b, b),
            "overlap": ov,
            "duplication_weight": duplicate,
        })
    detail = pd.DataFrame(rows).sort_values("duplication_weight", ascending=False).reset_index(drop=True)
    total = float(detail["duplication_weight"].sum()) if not detail.empty else 0.0
    return detail, total


def build_warnings(
    factor_table: pd.DataFrame,
    assets: pd.DataFrame,
    current_drawdown: float,
    lookup: Dict[str, Dict[str, Any]],
    overlap_dup: float,
) -> List[str]:
    warnings: List[str] = []
    for _, row in factor_table.iterrows():
        if row["status"] in ("硬上限", "严重不足", "预警", "不足"):
            warnings.append("%s：当前 %.1f%%，状态 %s" % (row["label"], row["current"] * 100, row["status"]))
    draw_cfg = lookup.get("portfolio_drawdown")
    if draw_cfg:
        dd_abs = abs(min(current_drawdown, 0.0))
        state = classify(dd_abs, draw_cfg)
        if state != "正常":
            warnings.append("组合回撤：当前 %.1f%%，状态 %s" % (dd_abs * 100, state))
    for _, row in assets.loc[assets["over_policy_weight"] > 1e-10].iterrows():
        warnings.append(
            "%s：实际 %.1f%% > 人工约束后目标 %.1f%%，新增额度为0"
            % (row["asset_name"], row["current_weight"] * 100, row["policy_target_weight"] * 100)
        )
    if pd.notna(overlap_dup) and overlap_dup > 0:
        warnings.append(
            "两两重叠重复风险指示 %.1f%%；可能多重计数，仅用于集中度监控" % (overlap_dup * 100)
        )
    if not warnings:
        warnings.append("当前未触发组合预警/硬上限；仍需结合模块01-05与人工判断。")
    return warnings


def pct(value: Any, digits: int = 1) -> str:
    return "—" if pd.isna(value) else ("%%.%df%%%%" % digits) % (float(value) * 100)


def money(value: Any) -> str:
    if pd.isna(value):
        return "—"
    v = float(value)
    if abs(v) >= 1e8:
        return "%.2f亿" % (v / 1e8)
    if abs(v) >= 1e4:
        return "%.1f万" % (v / 1e4)
    return "%0.0f" % v


def panel(ax: Axes, title: str, note: str = "") -> None:
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)
        s.set_linewidth(.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(.025, .96, title, transform=ax.transAxes, va="top", fontsize=10.5, fontweight="bold", color=TEXT)
    if note:
        ax.text(.025, .895, note, transform=ax.transAxes, va="top", fontsize=7.0, color=MUTED)


def figure(asof: pd.Timestamp, title: str, subtitle: str, page_no: int) -> Figure:
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(.035, .973, title, fontsize=18.2, fontweight="bold", color=TEXT, va="top")
    fig.text(.035, .933, subtitle, fontsize=8.2, color=MUTED, va="top")
    fig.text(.966, .973, asof.strftime("%Y-%m-%d"), ha="right", fontsize=10, color=TEXT, va="top")
    fig.text(.035, .018, "drcc Quant Engine  |  模块 06：组合风险与交易条件  |  风险约束层，不直接生成买卖指令", fontsize=7.0, color=MUTED)
    fig.text(.965, .018, "%d / 2" % page_no, ha="right", fontsize=7.0, color=MUTED)
    return fig


def kpi_cards(fig: Figure, cards: List[Tuple[str, str, str]]) -> None:
    n, gap, x0 = len(cards), .009, .035
    width = (.93 - (n - 1) * gap) / n
    for i, (label, value, color) in enumerate(cards):
        ax = fig.add_axes((x0 + i * (width + gap), .828, width, .075))
        ax.set_facecolor(PANEL)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(GRID); s.set_linewidth(.8)
        ax.text(.05, .75, label, transform=ax.transAxes, color=MUTED, fontsize=7.4)
        ax.text(.05, .20, value, transform=ax.transAxes, color=color, fontsize=12.8, fontweight="bold")


def find_factor(factor_table: pd.DataFrame, risk_item: str) -> Tuple[float, str]:
    row = factor_table.loc[factor_table["risk_item"] == risk_item]
    if row.empty:
        return np.nan, "未设限"
    return float(row.iloc[0]["current"]), str(row.iloc[0]["status"])


def draw_factor_table(fig: Figure, factors: pd.DataFrame) -> None:
    ax = fig.add_axes((.035, .355, .58, .445))
    panel(ax, "组合穿透风险表", "当前/模型/执行/人工后目标均为整个组合净值口径")
    headers = [("风险项目", .025, "left"), ("当前", .35, "right"), ("模型", .465, "right"),
               ("执行", .58, "right"), ("人工后", .695, "right"), ("硬线", .81, "right"), ("状态", .955, "right")]
    for label, x, ha in headers:
        ax.text(x, .815, label, transform=ax.transAxes, fontsize=7.4, color=MUTED, ha=ha, va="center")
    ax.plot([.02, .97], [.775, .775], transform=ax.transAxes, color=GRID, lw=.8)
    rank = {"硬上限": 0, "严重不足": 0, "预警": 1, "不足": 1, "正常": 2, "未设限": 3}
    sample = factors.copy()
    sample["_rank"] = sample["status"].map(rank).fillna(4)
    sample = sample.sort_values(["_rank", "current"], ascending=[True, False]).head(9)
    for idx, (_, row) in enumerate(sample.iterrows()):
        y = .735 - idx * .073
        if idx % 2:
            ax.add_patch(plt.Rectangle((.018, y-.032), .955, .064, transform=ax.transAxes, facecolor=BG, edgecolor="none"))
        values = [
            (str(row["label"])[:20], .025, "left", TEXT),
            (pct(row["current"]), .35, "right", TEXT),
            (pct(row["model"]), .465, "right", BLUE),
            (pct(row["execution"]), .58, "right", PURPLE),
            (pct(row["policy"]), .695, "right", ORANGE),
            (pct(row["hard_limit"]) if pd.notna(row["hard_limit"]) else "—", .81, "right", MUTED),
            (row["status"], .955, "right", status_color(row["status"])),
        ]
        for text, x, ha, color in values:
            ax.text(x, y, text, transform=ax.transAxes, fontsize=7.2, color=color, ha=ha, va="center")


def draw_factor_bars(ax: Axes, factors: pd.DataFrame) -> None:
    panel(ax, "风险因子敞口对照", "实线=当前；并列查看模型、执行与人工约束后目标")
    sub = factors.sort_values("current", ascending=False).head(7).iloc[::-1]
    if sub.empty:
        ax.text(.5, .5, "无风险因子", transform=ax.transAxes, ha="center", color=MUTED)
        return
    inner = ax.inset_axes((.30, .12, .66, .67))
    y = np.arange(len(sub)); h = .17
    inner.barh(y-1.5*h, sub["current"], height=h, color=TEXT, alpha=.86, label="当前")
    inner.barh(y-.5*h, sub["model"], height=h, color=BLUE, alpha=.72, label="模型")
    inner.barh(y+.5*h, sub["execution"], height=h, color=PURPLE, alpha=.72, label="执行")
    inner.barh(y+1.5*h, sub["policy"], height=h, color=ORANGE, alpha=.78, label="人工后")
    inner.set_yticks(y, [str(x)[:11] for x in sub["label"]], fontsize=7)
    inner.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%.0f%%" % (x*100)))
    inner.tick_params(axis="x", labelsize=7, colors=MUTED)
    inner.tick_params(axis="y", length=0, colors=TEXT)
    inner.grid(axis="x", alpha=.15)
    for s in inner.spines.values(): s.set_visible(False)
    inner.legend(loc="lower right", fontsize=6.2, frameon=False, ncol=2)


def draw_warnings(ax: Axes, warnings: List[str], cap_notes: List[str]) -> None:
    panel(ax, "风险闸门与组合提示", "硬上限会压缩未来加仓额度；不自动生成卖出指令")
    lines: List[Tuple[str, str]] = []
    for item in warnings[:5]:
        color = RED if ("硬" in item or "超目标" in item or "冻结" in item) else ORANGE
        lines.append(("• " + item, color))
    for item in cap_notes[:3]:
        lines.append(("• " + item, PURPLE))
    for i, (line, color) in enumerate(lines[:8]):
        ax.text(.035, .78 - i*.095, line, transform=ax.transAxes, fontsize=7.15, color=color, va="top")


def first_page(
    asof: pd.Timestamp,
    nav_value: float,
    current_dd: float,
    factor_table: pd.DataFrame,
    assets: pd.DataFrame,
    warnings: List[str],
    cap_notes: List[str],
    overlap_dup: float,
) -> Figure:
    fig = figure(asof, "组合风险与交易条件 · 总览", "将各资产独立信号穿透到组合层；人工风险上限优先于新增风险信号。", 1)
    domestic, domestic_status = find_factor(factor_table, "domestic_equity")
    tech, tech_status = find_factor(factor_table, "tech_growth")
    dividend, dividend_status = find_factor(factor_table, "dividend_defense")
    kpi_cards(fig, [
        ("组合净值", money(nav_value), TEXT),
        ("国内权益总敞口", pct(domestic), status_color(domestic_status)),
        ("科技成长风险", pct(tech), status_color(tech_status)),
        ("红利防御仓位", pct(dividend), status_color(dividend_status)),
        ("当前组合回撤", pct(current_dd), RED if current_dd <= -0.10 else TEXT),
        ("最终可加仓", "%s / %s" % (pct(assets["allowed_add_weight"].sum()), money(assets["allowed_add_value"].sum())), BLUE),
        ("两两重叠指示", pct(overlap_dup), PURPLE if pd.notna(overlap_dup) else MUTED),
    ])
    draw_factor_table(fig, factor_table)
    draw_factor_bars(fig.add_axes((.635, .525, .330, .275)), factor_table)
    draw_warnings(fig.add_axes((.635, .080, .330, .420)), warnings, cap_notes)
    return fig


def draw_asset_table(fig: Figure, assets: pd.DataFrame) -> None:
    ax = fig.add_axes((.035, .510, .930, .290))
    panel(ax, "单资产：模型仓位 → 执行仓位 → 人工上限 → 组合允许加仓", "执行仓位保留原模型冷静期/分段执行；模块06只做组合风险收紧。")
    headers = [("资产", .020, "left"), ("实际", .285, "right"), ("模型目标", .390, "right"),
               ("执行目标", .500, "right"), ("人工后目标", .620, "right"), ("原始可加", .735, "right"),
               ("最终可加", .850, "right"), ("状态", .970, "right")]
    for label, x, ha in headers:
        ax.text(x, .775, label, transform=ax.transAxes, fontsize=7.3, color=MUTED, ha=ha)
    ax.plot([.018, .975], [.735, .735], transform=ax.transAxes, color=GRID, lw=.8)
    sample = assets.sort_values("current_weight", ascending=False).head(10)
    for idx, (_, row) in enumerate(sample.iterrows()):
        y = .690 - idx*.060
        if idx % 2:
            ax.add_patch(plt.Rectangle((.016, y-.025), .960, .050, transform=ax.transAxes, facecolor=BG, edgecolor="none"))
        name = str(row["asset_name"])
        if len(name) > 16: name = name[:15] + "…"
        cool = " · 冷静%d日" % int(row["cooldown_days"]) if int(row["cooldown_days"]) > 0 else ""
        ax.text(.020, y, name + cool, transform=ax.transAxes, fontsize=7.1, color=TEXT, va="center")
        vals = [
            (pct(row["current_weight"]), .285, TEXT),
            (pct(row["model_target_weight"]), .390, BLUE),
            (pct(row["execution_target_weight"]), .500, PURPLE),
            (pct(row["policy_target_weight"]), .620, ORANGE),
            (pct(row["raw_add_weight"]), .735, TEXT),
            (pct(row["allowed_add_weight"]), .850, BLUE),
            (row["asset_status"], .970, status_color(row["asset_status"])),
        ]
        for text, x, color in vals:
            ax.text(x, y, text, transform=ax.transAxes, fontsize=7.1, color=color, ha="right", va="center")


def draw_drawdown(ax: Axes, nav_hist: pd.DataFrame) -> None:
    panel(ax, "组合净值与回撤", "当前回撤相对历史净值高点；hard阈值可冻结新增风险")
    if nav_hist.empty:
        ax.text(.5, .5, "缺少净值历史", transform=ax.transAxes, ha="center", color=MUTED)
        return
    inner = ax.inset_axes((.10, .20, .86, .62))
    inner.fill_between(nav_hist["trade_date"], nav_hist["drawdown"].to_numpy(dtype=float), 0, color=RED, alpha=.18)
    inner.plot(nav_hist["trade_date"], nav_hist["drawdown"], color=RED, lw=1.3)
    inner.axhline(0, color=GRID, lw=.8)
    inner.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%.0f%%" % (x*100)))
    inner.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    inner.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    inner.tick_params(labelsize=6.5, colors=MUTED)
    inner.grid(axis="y", alpha=.15)
    for s in inner.spines.values(): s.set_visible(False)


def draw_overlap(ax: Axes, overlap: Optional[pd.DataFrame], assets: pd.DataFrame) -> None:
    panel(ax, "资产重叠风险", "仅使用显式重叠系数；不从名称猜成分重合")
    if overlap is None or overlap.empty:
        ax.text(.5, .52, "未提供两两重叠矩阵", transform=ax.transAxes, ha="center", color=MUTED)
        ax.text(.5, .40, "因子穿透仍正常工作", transform=ax.transAxes, ha="center", color=MUTED, fontsize=7)
        return
    ids = [x for x in assets["asset_id"].tolist() if x in (set(overlap["asset_a"]) | set(overlap["asset_b"]))][:8]
    if len(ids) < 2:
        ax.text(.5, .5, "有效重叠资产不足2只", transform=ax.transAxes, ha="center", color=MUTED)
        return
    matrix = pd.DataFrame(np.nan, index=ids, columns=ids)
    np.fill_diagonal(matrix.values, 1.0)
    for _, r in overlap.iterrows():
        if r["asset_a"] in matrix.index and r["asset_b"] in matrix.columns:
            matrix.loc[r["asset_a"], r["asset_b"]] = matrix.loc[r["asset_b"], r["asset_a"]] = r["overlap"]
    inner = ax.inset_axes((.21, .16, .70, .65))
    inner.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="Blues")
    names = assets.set_index("asset_id")["asset_name"].to_dict()
    labels = [str(names.get(x, x))[:7] for x in ids]
    inner.set_xticks(range(len(ids)), labels, rotation=45, ha="right", fontsize=6.2)
    inner.set_yticks(range(len(ids)), labels, fontsize=6.2)
    inner.tick_params(length=0)
    for i in range(len(ids)):
        for j in range(len(ids)):
            v = matrix.iloc[i, j]
            if pd.notna(v):
                inner.text(j, i, "%.0f%%" % (v*100), ha="center", va="center", fontsize=5.7, color=TEXT)
    for s in inner.spines.values(): s.set_visible(False)


def draw_overlap_pairs(ax: Axes, detail: pd.DataFrame) -> None:
    panel(ax, "重叠贡献靠前", "重复风险=min(两资产实际权重)×重叠系数；多资产间可能重复计数")
    if detail.empty:
        ax.text(.5, .5, "暂无可计算重叠对", transform=ax.transAxes, ha="center", color=MUTED)
        return
    for i, (_, row) in enumerate(detail.head(6).iterrows()):
        y = .77 - i*.115
        name = "%s × %s" % (str(row["name_a"])[:8], str(row["name_b"])[:8])
        ax.text(.04, y, name, transform=ax.transAxes, fontsize=7.2, color=TEXT, va="center")
        ax.text(.67, y, "重合 %.0f%%" % (row["overlap"]*100), transform=ax.transAxes, fontsize=6.8, color=MUTED, ha="right")
        ax.text(.95, y, pct(row["duplication_weight"]), transform=ax.transAxes, fontsize=7.1, color=PURPLE, ha="right")


def second_page(
    asof: pd.Timestamp,
    assets: pd.DataFrame,
    nav_hist: pd.DataFrame,
    overlap: Optional[pd.DataFrame],
    overlap_detail: pd.DataFrame,
) -> Figure:
    fig = figure(asof, "组合交易条件 · 资产约束与重叠", "最终可加仓额度是风险容量，不是自动交易指令；超目标只提示，不自动卖出。", 2)
    draw_asset_table(fig, assets)
    draw_drawdown(fig.add_axes((.035, .080, .300, .400)), nav_hist)
    draw_overlap(fig.add_axes((.350, .080, .300, .400)), overlap, assets)
    draw_overlap_pairs(fig.add_axes((.665, .080, .300, .400)), overlap_detail)
    return fig


def strict_json(value: Any) -> Any:
    if isinstance(value, pd.Timestamp): return value.strftime("%Y-%m-%d")
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_): return bool(value)
    if pd.isna(value): return None
    return value


def records(frame: pd.DataFrame, cols: List[str]) -> List[Dict[str, Any]]:
    return [
        {col: strict_json(row[col]) for col in cols if col in row.index}
        for _, row in frame.iterrows()
    ]


def build_summary(
    asof: pd.Timestamp,
    nav_row: pd.Series,
    current_dd: float,
    max_dd: float,
    factor_table: pd.DataFrame,
    assets: pd.DataFrame,
    overlap_dup: float,
    warnings: List[str],
    cap_notes: List[str],
) -> Dict[str, Any]:
    return {
        "module": "06_portfolio_risk",
        "trade_date": asof.strftime("%Y-%m-%d"),
        "portfolio_nav_date": nav_row["trade_date"].strftime("%Y-%m-%d"),
        "portfolio_nav": float(nav_row["nav"]),
        "current_drawdown": float(current_dd),
        "max_drawdown_to_date": float(max_dd),
        "allowed_add_weight_total": float(assets["allowed_add_weight"].sum()),
        "allowed_add_value_total": float(assets["allowed_add_value"].sum()),
        "pairwise_overlap_duplication_indicator": strict_json(overlap_dup),
        "warnings": warnings,
        "portfolio_cap_notes": cap_notes,
        "factors": records(factor_table, [
            "risk_item", "label", "current", "model", "execution", "policy",
            "limit_type", "soft_limit", "hard_limit", "status"
        ]),
        "assets": records(assets, [
            "asset_id", "asset_name", "market_value", "budget_weight",
            "model_position", "execution_position", "manual_cap_position",
            "current_weight", "model_target_weight", "execution_target_weight",
            "policy_target_weight", "raw_add_weight", "allowed_add_weight",
            "allowed_add_value", "over_policy_weight", "cooldown_days",
            "signal", "execution_status", "asset_status"
        ]),
        "caveats": [
            "最终可加仓额度是风险容量，不是自动买入指令",
            "人工单资产上限与组合风险上限只能收紧新增风险",
            "风险因子暴露和两两重叠必须由上游显式提供，不从名称猜测",
            "两两重叠重复风险指标可能多重计数，不从总权益敞口中机械扣减",
        ],
    }


def write_templates(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    positions = pd.DataFrame([
        ["CSI300", "沪深300", 0, .15, .80, "宽基", "A股", 1],
        ["A500", "中证A500", 0, .15, .80, "宽基", "A股", 1],
        ["CHINEXT", "创业板指", 0, .08, .60, "成长", "A股", 1],
        ["STAR50", "科创50", 0, .08, .60, "科技成长", "A股", 1],
        ["CHIP", "芯片ETF", 0, .05, .50, "芯片", "A股", 1],
        ["DIVIDEND", "国内红利", 0, .12, 1.00, "红利", "A股", 1],
        ["HKDIV", "港股低波红利", 0, .10, 1.00, "红利", "港股", 1],
    ], columns=["asset_id", "asset_name", "market_value", "budget_weight", "manual_cap_position", "theme", "asset_class", "enabled"])
    positions.to_csv(directory/"portfolio_positions.csv", index=False, encoding="utf-8-sig")

    signals = pd.DataFrame([
        ["20260924", "CSI300", 1.00, .80, 1, "Risk-On", "第1日预警"],
        ["20260924", "A500", 1.00, .80, 0, "Risk-On", "第1日预警"],
        ["20260924", "CHINEXT", .80, .80, 0, "Neutral", "一致"],
        ["20260924", "STAR50", .80, .80, 0, "Neutral", "一致"],
        ["20260924", "CHIP", .80, .80, 0, "Neutral", "一致"],
        ["20260924", "DIVIDEND", .80, .80, 0, "Neutral", "示例"],
        ["20260924", "HKDIV", .80, .80, 0, "Neutral", "示例"],
    ], columns=["trade_date", "asset_id", "model_position", "execution_position", "cooldown_days", "signal", "execution_status"])
    signals.to_csv(directory/"portfolio_signals.csv", index=False, encoding="utf-8-sig")

    # 暴露模板故意填入演示值；必须由真实成分/行业权重或人工认可映射替换后再用于正式报告。
    exposure = pd.DataFrame([
        ["CSI300", 1.0, .30, .05, 0.0, .10],
        ["A500", 1.0, .40, .05, 0.0, .15],
        ["CHINEXT", 1.0, .85, 0.0, 0.0, .30],
        ["STAR50", 1.0, .95, 0.0, 0.0, .65],
        ["CHIP", 1.0, 1.00, 0.0, 0.0, 1.00],
        ["DIVIDEND", 1.0, .10, 1.0, 0.0, .02],
        ["HKDIV", 0.0, .10, 1.0, 1.0, .02],
    ], columns=["asset_id", "domestic_equity", "tech_growth", "dividend_defense", "hk_dividend", "theme_chip"])
    exposure.to_csv(directory/"portfolio_exposure_map.csv", index=False, encoding="utf-8-sig")

    limits = pd.DataFrame([
        ["domestic_equity", "国内权益总敞口", "max", .70, .80],
        ["tech_growth", "科技成长风险", "max", .30, .35],
        ["dividend_defense", "红利防御仓位", "min", .10, .05],
        ["single_theme_max", "单一主题上限", "max", .15, .20],
        ["portfolio_drawdown", "组合当前回撤", "max", .12, .15],
    ], columns=["risk_item", "label", "limit_type", "soft_limit", "hard_limit"])
    limits.to_csv(directory/"portfolio_risk_limits.csv", index=False, encoding="utf-8-sig")

    nav = pd.DataFrame([
        ["20260920", 1000000], ["20260921", 1010000], ["20260922", 1005000],
        ["20260923", 995000], ["20260924", 990000],
    ], columns=["trade_date", "nav"])
    nav.to_csv(directory/"portfolio_nav.csv", index=False, encoding="utf-8-sig")

    overlap = pd.DataFrame([
        ["CSI300", "A500", .55], ["A500", "STAR50", .18], ["A500", "CHIP", .12], ["STAR50", "CHIP", .38]
    ], columns=["asset_a", "asset_b", "overlap"])
    overlap.to_csv(directory/"portfolio_overlap_matrix.csv", index=False, encoding="utf-8-sig")

    note = """模块06输入模板已生成。\n\n重要：模板中的风险因子暴露和重叠系数只是演示格式，不是正式投资参数。正式使用前必须由真实成分权重/行业暴露或人工认可映射替换。\n\n`budget_weight` 是该资产模型100%仓位对应整个组合净值的预算权重；`manual_cap_position` 是预算口径的人工上限。\n"""
    (directory/"README_report_06_inputs.txt").write_text(note, encoding="utf-8")
    print("已生成模块06输入模板：%s" % directory)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--directory", default=str(DATA_DIR), help="输入CSV、JSON与PDF的根目录，默认项目 data 目录")
    p.add_argument("--date", help="报告交易日YYYYMMDD；默认signals最新交易日")
    p.add_argument("--positions", help="实际持仓CSV；默认 <directory>/portfolio_positions.csv")
    p.add_argument("--signals", help="模型/执行仓位CSV；默认 <directory>/portfolio_signals.csv")
    p.add_argument("--exposure-map", dest="exposure_map", help="风险因子映射CSV；默认 <directory>/portfolio_exposure_map.csv")
    p.add_argument("--limits", help="风险上限CSV；默认 <directory>/portfolio_risk_limits.csv")
    p.add_argument("--nav", help="组合净值CSV；默认 <directory>/portfolio_nav.csv")
    p.add_argument("--overlap", help="可选重叠矩阵CSV；默认 <directory>/portfolio_overlap_matrix.csv")
    p.add_argument("--nav-max-lag", type=int, default=3, help="组合净值允许滞后自然日数，默认3")
    p.add_argument("--output", help="PDF输出；默认 <directory>/pdf/report_06_portfolio_risk.pdf")
    p.add_argument("--summary-json", help="JSON输出；默认 <directory>/report_06_portfolio_risk.json")
    p.add_argument("--write-templates", action="store_true", help="在directory生成模块06输入模板后退出")
    return p


def run(args) -> int:
    root = Path(args.directory).expanduser().resolve()
    if args.write_templates:
        write_templates(root)
        return 0

    positions_path = Path(args.positions).expanduser() if args.positions else root/"portfolio_positions.csv"
    signals_path = Path(args.signals).expanduser() if args.signals else root/"portfolio_signals.csv"
    exposure_path = Path(args.exposure_map).expanduser() if args.exposure_map else root/"portfolio_exposure_map.csv"
    limits_path = Path(args.limits).expanduser() if args.limits else root/"portfolio_risk_limits.csv"
    nav_path = Path(args.nav).expanduser() if args.nav else root/"portfolio_nav.csv"
    overlap_path = Path(args.overlap).expanduser() if args.overlap else root/"portfolio_overlap_matrix.csv"

    positions = read_positions(positions_path)
    signals = read_signals(signals_path)
    exposure_map, factors = read_exposure_map(exposure_path, positions)
    limits = read_limits(limits_path)
    nav = read_nav(nav_path)
    overlap = read_overlap(overlap_path, positions)

    asof = choose_asof(signals, args.date)
    nav_row, _ = choose_nav_row(nav, asof, args.nav_max_lag)
    portfolio_nav = float(nav_row["nav"])
    if positions["market_value"].sum() > portfolio_nav * 1.001:
        raise ValueError("持仓市值合计超过组合净值；请检查币种/单位/nav口径")

    assets = assemble_assets(positions, signals, exposure_map, factors, asof, portfolio_nav)
    current_dd, max_dd, nav_hist = calculate_drawdown(nav, asof)
    lookup = limits_dict(limits)
    factor_table = build_factor_table(assets, factors, lookup)
    assets, cap_notes = apply_portfolio_caps(assets, factors, lookup, current_dd, portfolio_nav)
    overlap_detail, overlap_dup = calculate_overlap(overlap, assets)
    warnings = build_warnings(factor_table, assets, current_dd, lookup, overlap_dup)

    configure_font()
    out = Path(args.output).expanduser() if args.output else root/"pdf/report_06_portfolio_risk.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for fig in (
            first_page(asof, portfolio_nav, current_dd, factor_table, assets, warnings, cap_notes, overlap_dup),
            second_page(asof, assets, nav_hist, overlap, overlap_detail),
        ):
            pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)

    summary = build_summary(asof, nav_row, current_dd, max_dd, factor_table, assets, overlap_dup, warnings, cap_notes)
    summary_path = Path(args.summary_json).expanduser() if args.summary_json else root/"report_06_portfolio_risk.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    print(
        "模块06 %s：资产%d只；当前持仓%.1f%%；最终可加%.1f%% / %s；PDF 2页：%s"
        % (asof.strftime("%Y-%m-%d"), len(assets), assets["current_weight"].sum()*100,
           assets["allowed_add_weight"].sum()*100, money(assets["allowed_add_value"].sum()), out)
    )
    for item in warnings[:6]:
        print("  - %s" % item)
    return 0


def main() -> None:
    try:
        status = run(parser().parse_args())
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        status = 1
    sys.exit(status)


if __name__ == "__main__":
    main()
