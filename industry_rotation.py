#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模块 2：申万行业日线增量归档及行业轮动计算（Python 3.9+）。

数据源：Tushare sw_daily + index_daily；Token 来自同目录 config.yaml 的 keys.tushare。

修复重点：
  1. sw_daily 按 5 个自然日短窗口获取，避免单次 4000 行上限造成静默截断。
  2. 保存 sw_daily_history.csv 和 a500_daily_history.csv 原始历史；快照仍输出
     industry_rotation.csv，兼容原 report_02_industry_rotation.py。
  3. 保存 industry_rotation_history.csv 完整指标历史；使用增量重叠回溯，
     对基准指数已有交易日的缺口重新请求；下载或覆盖不完整时不写坏旧缓存。
  4. 20 日量能和 20/60 日相对收益，严格使用完整有效历史计算，不以 0 补缺。

首次运行：
    python industry_rotation.py --start-date 20250101 --end-date 20260928 --full-refresh
以后盘后：
    python industry_rotation.py --end-date 20260928
注意：这是一套日频数据处理与研究标签，不生成买卖指令。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"
DEFAULT_BENCHMARK = "000510.SH"  # 中证A500，可从 benchmark.a500_ts_code 覆盖
SW_WINDOW_DAYS = 5                  # 每窗至多5个自然日，避免 >4000 行截断
BENCH_WINDOW_DAYS = 180              # 指数日线每窗不足200条交易记录
SW_API_CAP = 4000                    # 申万接口公布的单次行数上限
QUADRANT_ORDER = ["领先且改善", "落后但改善", "领先但减弱", "落后且减弱", "数据不足"]
RAW_SW_COLS = ("ts_code", "trade_date", "close")
RAW_BENCH_COLS = ("trade_date", "close")


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise ValueError("config.yaml 必须是 YAML mapping")
    return config


def _tushare_token(config: dict[str, Any]) -> str:
    keys = config.get("keys", {})
    if not isinstance(keys, dict) or not keys.get("tushare"):
        raise ValueError("config.yaml 中未找到 keys.tushare")
    return str(keys["tushare"]).strip()


def create_pro(config_path: str | Path = DEFAULT_CONFIG):
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("请先安装依赖：pip install tushare pandas pyyaml") from exc
    return ts.pro_api(_tushare_token(load_config(config_path)))


def normalize_date(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("交易日期不能为缺失值")
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    try:
        parsed = pd.to_datetime(text, format="%Y%m%d" if len(text) == 8 and text.isdigit() else None, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"无法解析交易日期：{value!r}") from exc
    return parsed.strftime("%Y%m%d")


def date_chunks(start_date: str, end_date: str, days: int) -> Iterable[tuple[str, str]]:
    start = datetime.strptime(normalize_date(start_date), "%Y%m%d").date()
    end = datetime.strptime(normalize_date(end_date), "%Y%m%d").date()
    if start > end:
        return
    if days < 1:
        raise ValueError("分块天数必须大于零")
    now = start
    while now <= end:
        stop = min(end, now + timedelta(days=days - 1))
        yield now.strftime("%Y%m%d"), stop.strftime("%Y%m%d")
        now = stop + timedelta(days=1)


def _api(pro: Any, endpoint: str, *, attempts: int = 3, pause: float = .4, **kwargs) -> pd.DataFrame:
    for retry in range(attempts):
        try:
            value = getattr(pro, endpoint)(**kwargs)
            if value is None:
                return pd.DataFrame()
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"{endpoint} 返回值不是 DataFrame")
            return value.copy()
        except Exception as exc:
            if retry == attempts - 1:
                raise RuntimeError(f"{endpoint} 请求失败（{kwargs}）：{exc}") from exc
            print(f"  重试 {endpoint} {retry + 1}/{attempts}: {type(exc).__name__}")
            time.sleep(pause * (retry + 1))
    raise AssertionError("unreachable")


def _standardize(frame: pd.DataFrame, *, industry: bool) -> pd.DataFrame:
    expected = RAW_SW_COLS if industry else RAW_BENCH_COLS
    if frame.empty:
        return pd.DataFrame(columns=(["ts_code", "trade_date", "close"] if industry else ["ts_code", "trade_date", "close"]))
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ValueError(f"接口/缓存缺少关键列：{sorted(missing)}")
    out = frame.copy()
    out["trade_date"] = out["trade_date"].map(normalize_date)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.loc[out["close"].gt(0) & out["close"].notna()].copy()
    if "ts_code" in out:
        out["ts_code"] = out["ts_code"].astype(str).str.strip().str.upper()
    for col in ("amount", "pct_change", "pct_chg", "float_mv", "total_mv", "pe", "pb"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    keys = ["ts_code", "trade_date"] if industry else ["trade_date"]
    out = out.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    return out


def read_cache(path: Path, *, industry: bool) -> pd.DataFrame:
    if not path.exists():
        return _standardize(pd.DataFrame(), industry=industry)
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype={"ts_code": str, "trade_date": str}, low_memory=False)
    return _standardize(raw, industry=industry)


def merge_data(old: pd.DataFrame, new: pd.DataFrame, *, industry: bool) -> pd.DataFrame:
    if old.empty:
        return _standardize(new, industry=industry)
    if new.empty:
        return _standardize(old, industry=industry)
    return _standardize(pd.concat([old, new], ignore_index=True, sort=False), industry=industry)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(temp, index=False, encoding="utf-8-sig", na_rep="")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def fetch_benchmark_history(pro: Any, ts_code: str, start_date: str, end_date: str,
                            *, pause: float = .4) -> pd.DataFrame:
    parts = []
    for lo, hi in date_chunks(start_date, end_date, BENCH_WINDOW_DAYS):
        part = _api(pro, "index_daily", ts_code=ts_code, start_date=lo, end_date=hi, pause=pause)
        if not part.empty:
            if "ts_code" not in part:
                part["ts_code"] = ts_code
            parts.append(part)
        time.sleep(pause)
    return _standardize(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), industry=False)


def _expected_days(benchmark: pd.DataFrame, lo: str, hi: str) -> set[str]:
    return set(benchmark.loc[benchmark["trade_date"].between(lo, hi), "trade_date"])


def fetch_industry_history(pro: Any, start_date: str, end_date: str,
                           *, benchmark: Optional[pd.DataFrame] = None,
                           old: Optional[pd.DataFrame] = None,
                           overlap_days: int = 14, full_refresh: bool = False,
                           min_per_day: int = 250, pause: float = .4) -> pd.DataFrame:
    """按短窗口获取；未覆盖基准交易日或疑似部分返回时整体报错，不提交坏数据。"""
    old = old if old is not None else _standardize(pd.DataFrame(), industry=True)
    old_counts = old.groupby("trade_date")["ts_code"].nunique().to_dict() if not old.empty else {}
    latest = max(old_counts) if old_counts else None
    overlap_start = ((datetime.strptime(latest, "%Y%m%d") - timedelta(days=overlap_days)).strftime("%Y%m%d")
                     if latest else start_date)
    parts = []
    requests = 0
    for lo, hi in date_chunks(start_date, end_date, SW_WINDOW_DAYS):
        known = _expected_days(benchmark, lo, hi) if benchmark is not None and not benchmark.empty else set()
        # 已覆盖且不在最近重叠回溯内的窗口可跳过；如果只有一部分日期，则整窗修补。
        complete = bool(known) and all(old_counts.get(d, 0) >= min_per_day for d in known)
        if not full_refresh and complete and hi < overlap_start:
            continue
        if benchmark is not None and not benchmark.empty and not known:
            continue  # 法定休市日，无须请求
        part = _api(pro, "sw_daily", start_date=lo, end_date=hi, pause=pause)
        requests += 1
        if len(part) >= SW_API_CAP:
            raise RuntimeError(f"申万窗口 {lo}-{hi} 返回{len(part)}条，可能触发接口上限；请缩短窗口")
        part = _standardize(part, industry=True)
        got = set(part["trade_date"]) if not part.empty else set()
        missing = known - got
        if missing:
            raise RuntimeError(f"申万行情窗口 {lo}-{hi} 缺少交易日 {sorted(missing)}；不覆盖旧缓存。"
                               "若刚收盘，请待申万接口更新后重试。")
        if min_per_day:
            per_day = part.groupby("trade_date")["ts_code"].nunique()
            baseline = list(old_counts.values()) + per_day.tolist()
            dynamic_floor = int(float(pd.Series(baseline).median()) * .80) if baseline else 0
            floor = max(min_per_day, dynamic_floor)
            thin = per_day.loc[per_day < floor]
            if not thin.empty:
                raise RuntimeError(f"申万行情疑似缺行：{thin.to_dict()}，检查下限 {floor}；不覆盖旧缓存")
        if not part.empty:
            parts.append(part)
        time.sleep(pause)
    print(f"申万短窗口请求 {requests} 次；补回记录 {sum(map(len, parts)):,} 条")
    return _standardize(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), industry=True)


def _return(series: pd.Series, periods: int) -> pd.Series:
    return series / series.shift(periods) - 1


def _quadrant(row: pd.Series) -> str:
    relative = row.get("excess_20d")
    momentum = row.get("momentum_change_20d")
    if pd.isna(relative) or pd.isna(momentum):
        return "数据不足"
    if relative >= 0 and momentum >= 0:
        return "领先且改善"
    if relative < 0 and momentum >= 0:
        return "落后但改善"
    if relative >= 0 and momentum < 0:
        return "领先但减弱"
    return "落后且减弱"


def calculate_rotation(industry: pd.DataFrame, benchmark: pd.DataFrame,
                       *, latest_only: bool = True) -> pd.DataFrame:
    """返回最新交易日快照，或所有历史日期的完整指标序列。"""
    if industry.empty:
        return pd.DataFrame()
    df = _standardize(industry, industry=True)
    if benchmark is None or benchmark.empty:
        bench = pd.DataFrame(columns=["trade_date", "bench_close"])
    else:
        bench = _standardize(benchmark, industry=False)[["trade_date", "close"]].rename(columns={"close": "bench_close"})
    df = df.merge(bench, on="trade_date", how="left", validate="many_to_one")
    g = df.groupby("ts_code", sort=False)
    for w in (1, 5, 20, 60):
        df[f"ret_{w}d"] = g["close"].transform(lambda s, n=w: _return(s, n))
    for w in (5, 20):
        if "amount" in df:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            past = g["amount"].shift(1)
            avg = past.groupby(df["ts_code"]).transform(lambda s, n=w: s.rolling(n, min_periods=n).mean())
            df[f"amount_ratio_{w}d"] = df["amount"] / avg.where(avg.ne(0))
        else:
            df[f"amount_ratio_{w}d"] = float("nan")
    # 基准在每一个行业实际存在的共同交易日上计算，避免行业缺日带来的错位。
    for w in (5, 20, 60):
        bench_return = g["bench_close"].transform(lambda s, n=w: _return(s, n))
        df[f"excess_{w}d"] = df[f"ret_{w}d"] - bench_return
    df["momentum_change_20d"] = df.groupby("ts_code", sort=False)["excess_20d"].transform(
        lambda s: s - s.shift(20)
    )
    df["quadrant"] = df.apply(_quadrant, axis=1)
    df = df.drop(columns=["bench_close"])
    if latest_only:
        latest = df["trade_date"].max()
        df = df.loc[df["trade_date"].eq(latest)].copy()
        df["quadrant"] = pd.Categorical(df["quadrant"], categories=QUADRANT_ORDER, ordered=True)
        df = df.sort_values(["quadrant", "excess_20d"], ascending=[True, False])
    return df.reset_index(drop=True)


def _outpath(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def run(start_date: str | None = None, end_date: str | None = None,
        config_path: str | Path = DEFAULT_CONFIG, *, root: Path = ROOT,
        output: str | Path = "industry_rotation.csv",
        history_output: str | Path = "industry_rotation_history.csv",
        sw_cache: str | Path = "sw_daily_history.csv",
        benchmark_cache: str | Path = "a500_daily_history.csv",
        overlap_days: int = 14, full_refresh: bool = False,
        min_per_day: int = 250, pause: float = .4, pro: Any = None,
        sync_style05_cache: bool = False) -> pd.DataFrame:
    root = Path(root).resolve()
    end = normalize_date(end_date or date.today().strftime("%Y%m%d"))
    start = normalize_date(start_date or (datetime.strptime(end, "%Y%m%d") - timedelta(days=420)).strftime("%Y%m%d"))
    if start > end:
        raise ValueError("--start-date 不能晚于 --end-date")
    if overlap_days < 1 or min_per_day < 0:
        raise ValueError("--overlap-days 应 >=1，--min-per-day 应 >=0")
    config = load_config(config_path)
    benchmark_code = config.get("benchmark", {}).get("a500_ts_code", DEFAULT_BENCHMARK)
    pro = pro if pro is not None else create_pro(config_path)
    sw_path = _outpath(root, sw_cache)
    bench_path = _outpath(root, benchmark_cache)
    old_sw = read_cache(sw_path, industry=True)
    old_bench = read_cache(bench_path, industry=False)
    print(f"行业数据窗口 {start} 至 {end}；基准 {benchmark_code}；旧缓存 {len(old_sw):,} 条")
    # 指数日线每次只需很少数据，按请求窗口刷新，避免旧缓存遗漏交易日。
    new_bench = fetch_benchmark_history(pro, benchmark_code, start, end, pause=pause)
    if new_bench.empty and old_bench.empty:
        raise RuntimeError("A500基准没有返回任何日线；不生成不完整的相对强弱数据")
    benchmark = merge_data(old_bench, new_bench, industry=False)
    new_sw = fetch_industry_history(pro, start, end, benchmark=benchmark, old=old_sw,
                                   overlap_days=overlap_days, full_refresh=full_refresh,
                                   min_per_day=min_per_day, pause=pause)
    industry = merge_data(old_sw, new_sw, industry=True)
    if industry.empty:
        raise RuntimeError("申万行情无有效历史，停止写入")
    expected = set(benchmark.loc[benchmark["trade_date"].between(start, end), "trade_date"])
    actual = set(industry.loc[industry["trade_date"].between(start, end), "trade_date"])
    missing = expected - actual
    if missing:
        raise RuntimeError(f"申万历史仍缺失{len(missing)}个基准交易日：{sorted(missing)[:8]}；旧缓存未覆盖")
    if min_per_day:
        counts = industry.loc[industry["trade_date"].between(start, end)].groupby("trade_date")["ts_code"].nunique()
        # 在静态下限外同时使用全窗口中位数，避免439个行业的正常日被截成300条仍通过。
        typical = float(counts.median()) if not counts.empty else 0.0
        floor = max(min_per_day, int(typical * 0.80))
        incomplete = counts[counts < floor]
        if not incomplete.empty:
            raise RuntimeError(f"申万历史存在疑似缺行交易日：{incomplete.head(8).to_dict()}，动态下限 {floor}；旧缓存未覆盖")
    all_metrics = calculate_rotation(industry, benchmark, latest_only=False)
    latest_bench = benchmark.loc[benchmark["trade_date"].le(end), "trade_date"].max()
    latest_sw = industry.loc[industry["trade_date"].le(end), "trade_date"].max()
    if pd.isna(latest_sw):
        raise RuntimeError("目标日期前没有可用申万交易日")
    if latest_bench != latest_sw:
        raise RuntimeError(f"申万数据最新 {latest_sw}，但基准最新 {latest_bench}；不生成可能过期的快照")
    snapshot = all_metrics.loc[all_metrics["trade_date"].eq(latest_sw)].copy()
    snapshot["quadrant"] = pd.Categorical(snapshot["quadrant"], categories=QUADRANT_ORDER, ordered=True)
    snapshot = snapshot.sort_values(["quadrant", "excess_20d"], ascending=[True, False]).reset_index(drop=True)
    # report_02 使用 pd.to_datetime() 读取快照日期；必须保留 ISO 格式，
    # 否则 YYYYMMDD 被 pandas 自动读为整数后会错误解析成1970年的纳秒时间戳。
    all_metrics["trade_date"] = pd.to_datetime(all_metrics["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    snapshot["trade_date"] = pd.to_datetime(snapshot["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    # 所有请求/完整性检查通过后才写磁盘，避免网络异常覆盖既有历史。
    atomic_csv(benchmark, bench_path)
    atomic_csv(industry, sw_path)
    atomic_csv(all_metrics, _outpath(root, history_output))
    atomic_csv(snapshot, _outpath(root, output))
    if sync_style05_cache:
        # 可选同步给模块05自己的申万缓存，不修改其指数缓存或交易模型。
        style_path = _outpath(root, "style_05_sw_daily.csv")
        style_old = read_cache(style_path, industry=True)
        style_merged = merge_data(style_old, industry, industry=True)
        atomic_csv(style_merged, style_path)
        print(f"已同步模块05行业缓存 style_05_sw_daily.csv：{len(style_merged):,} 条")
    print(f"已保存 sw_daily_history.csv {len(industry):,} 条、industry_rotation_history.csv {len(all_metrics):,} 条")
    print(f"已输出 industry_rotation.csv：{len(snapshot)} 个行业，交易日 {latest_sw}；"
          f"有效20日 {snapshot['ret_20d'].notna().sum()}、60日 {snapshot['ret_60d'].notna().sum()}、"
          f"动能 {snapshot['momentum_change_20d'].notna().sum()}")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-date", help="YYYYMMDD；默认截止日前420自然日")
    parser.add_argument("--end-date", help="YYYYMMDD；默认本机日期，建议盘后显式传入")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--directory", default=str(ROOT), help="CSV输出目录，默认脚本同目录")
    parser.add_argument("--output", default="industry_rotation.csv")
    parser.add_argument("--history-output", default="industry_rotation_history.csv")
    parser.add_argument("--sw-cache", default="sw_daily_history.csv")
    parser.add_argument("--benchmark-cache", default="a500_daily_history.csv")
    parser.add_argument("--overlap-days", type=int, default=14, help="增量重取最近14自然日")
    parser.add_argument("--full-refresh", action="store_true", help="重拉日期范围内全部申万日线，不丢弃范围外旧记录")
    parser.add_argument("--min-per-day", type=int, default=250, help="每日行业条数警戒线；0禁用")
    parser.add_argument("--pause", type=float, default=.4, help="API调用间隔秒数")
    parser.add_argument("--sync-style05-cache", action="store_true", help="可选：把申万历史同步到模块05的行业缓存")
    args = parser.parse_args()
    try:
        run(args.start_date, args.end_date, args.config, root=Path(args.directory),
            output=args.output, history_output=args.history_output, sw_cache=args.sw_cache,
            benchmark_cache=args.benchmark_cache, overlap_days=args.overlap_days,
            full_refresh=args.full_refresh, min_per_day=args.min_per_day, pause=args.pause,
            sync_style05_cache=args.sync_style05_cache)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
