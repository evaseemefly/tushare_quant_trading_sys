"""模块 1：全市场成交额与市场广度。

脚本从 Tushare ``daily`` 拉取 A 股日线，并按交易日计算：

* 全市场成交额（亿元）及相对过去 20 个交易日的量能；
* 上涨、下跌、平盘家数和上涨成交额占比；
* 20 日新高/新低家数；
* 收盘价高于 MA20、MA60、MA120 的股票占比。

这些结果是观察指标，不是自动买卖信号。示例：

    python market_breadth.py --start-date 20260101 --end-date 20260928

Tushare ``daily.amount`` 的单位为千元，本脚本同时输出亿元口径。
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
PAGE_SIZE = 5_000
DATE_CHUNK_DAYS = 15  # 避免单个日期窗口的 offset 接近 Tushare 的 100000 上限。


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """读取 YAML 配置，不在日志中打印 token。"""
    with Path(config_path).open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须是 YAML mapping：{config_path}")
    return config


def _tushare_token(config: dict[str, Any]) -> str:
    keys = config.get("keys", {})
    if not isinstance(keys, dict) or not keys.get("tushare"):
        raise ValueError("config.yaml 中未找到 keys.tushare")
    return str(keys["tushare"]).strip()


def create_pro(config_path: str | Path = DEFAULT_CONFIG):
    """创建 Tushare Pro 客户端；导入延迟到调用时，便于离线测试计算函数。"""
    try:
        import tushare as ts
    except ImportError as exc:  # pragma: no cover - 由运行环境决定
        raise RuntimeError("请先安装依赖：pip install tushare pandas pyyaml") from exc
    return ts.pro_api(_tushare_token(load_config(config_path)))


def _date_chunks(
    start_date: str, end_date: str, chunk_days: int = DATE_CHUNK_DAYS
) -> list[tuple[str, str]]:
    """将日期区间拆成小窗口，规避 Tushare offset 不能超过 100000。"""
    try:
        start = datetime.strptime(str(start_date), "%Y%m%d").date()
        end = datetime.strptime(str(end_date), "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError("日期必须使用 YYYYMMDD 格式") from exc
    if start > end:
        raise ValueError("start_date 不能晚于 end_date")
    if chunk_days <= 0:
        raise ValueError("chunk_days 必须为正数")

    chunks: list[tuple[str, str]] = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunks.append((current.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        current = chunk_end + timedelta(days=1)
    return chunks


def fetch_daily(pro: Any, start_date: str, end_date: str) -> pd.DataFrame:
    """分日期窗口、分页获取股票日线。

    Tushare 单次查询虽然支持 ``limit``/``offset``，但 offset 不能大于
    100000。直接查询几个月的全市场日线很容易触发该限制，因此先按 15
    个自然日拆窗，再在每个窗口内分页。
    """
    frames: list[pd.DataFrame] = []
    for window_start, window_end in _date_chunks(start_date, end_date):
        offset = 0
        while True:
            batch = pro.daily(
                start_date=window_start,
                end_date=window_end,
                fields="ts_code,trade_date,close,pct_chg,amount",
                limit=PAGE_SIZE,
                offset=offset,
            )
            if batch is None or batch.empty:
                break
            frames.append(batch)
            if len(batch) < PAGE_SIZE:
                break
            offset += len(batch)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(["ts_code", "trade_date"])
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    for column in ("close", "pct_chg", "amount"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def _safe_ratio(numerator: float, denominator: float) -> float | pd.NA:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return pd.NA
    return numerator / denominator


def calculate_breadth(daily: pd.DataFrame) -> pd.DataFrame:
    """从股票日线计算按交易日聚合的市场广度表。

    新高、新低和成交额比率只使用当日之前的数据，避免把当日价格/成交额
    泄漏进自己的比较基准。数据不足对应的指标保留为空值。
    """
    required = {"ts_code", "trade_date", "close", "pct_chg", "amount"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"daily 缺少字段：{sorted(missing)}")
    if daily.empty:
        return pd.DataFrame()

    df = daily[list(required)].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    for column in ("close", "pct_chg", "amount"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = (
        df.dropna(subset=["ts_code", "trade_date", "close"])
        .drop_duplicates(["ts_code", "trade_date"])
        .sort_values(["ts_code", "trade_date"])
    )
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("ts_code", sort=False)["close"]
    df["ma20"] = grouped.transform(lambda values: values.rolling(20, min_periods=20).mean())
    df["ma60"] = grouped.transform(lambda values: values.rolling(60, min_periods=60).mean())
    df["ma120"] = grouped.transform(lambda values: values.rolling(120, min_periods=120).mean())
    prior = grouped.shift(1)
    df["prior_high20"] = prior.groupby(df["ts_code"]).transform(
        lambda values: values.rolling(20, min_periods=20).max()
    )
    df["prior_low20"] = prior.groupby(df["ts_code"]).transform(
        lambda values: values.rolling(20, min_periods=20).min()
    )
    df["is_up"] = df["pct_chg"] > 0
    df["is_down"] = df["pct_chg"] < 0
    df["is_flat"] = df["pct_chg"] == 0
    # 均线尚未形成时不计入分母，而不是把缺失值当作“低于均线”。
    df["above_ma20"] = df["close"].gt(df["ma20"]).where(df["ma20"].notna())
    df["above_ma60"] = df["close"].gt(df["ma60"]).where(df["ma60"].notna())
    df["above_ma120"] = df["close"].gt(df["ma120"]).where(df["ma120"].notna())
    df["new_high20"] = df["prior_high20"].notna() & (df["close"] > df["prior_high20"])
    df["new_low20"] = df["prior_low20"].notna() & (df["close"] < df["prior_low20"])

    records: list[dict[str, Any]] = []
    for trade_date, day in df.groupby("trade_date", sort=True):
        valid = day["close"].notna()
        valid_count = int(valid.sum())
        amount = day["amount"].sum(min_count=1)
        up_amount = day.loc[day["is_up"] & day["amount"].notna(), "amount"].sum()
        records.append(
            {
                "trade_date": trade_date,
                "sample_count": valid_count,
                "total_amount_100m": amount / 100_000 if pd.notna(amount) else pd.NA,
                "up_count": int(day["is_up"].sum()),
                "down_count": int(day["is_down"].sum()),
                "flat_count": int(day["is_flat"].sum()),
                "up_ratio": _safe_ratio(float(day["is_up"].sum()), valid_count),
                "new_high20_count": int(day["new_high20"].sum()),
                "new_low20_count": int(day["new_low20"].sum()),
                "above_ma20_ratio": day["above_ma20"].mean(),
                "above_ma60_ratio": day["above_ma60"].mean(),
                "above_ma120_ratio": day["above_ma120"].mean(),
                "up_amount_ratio": _safe_ratio(float(up_amount), float(amount))
                if pd.notna(amount)
                else pd.NA,
            }
        )

    result = pd.DataFrame.from_records(records).sort_values("trade_date").reset_index(drop=True)
    amount_avg = result["total_amount_100m"].shift(1).rolling(20, min_periods=5).mean()
    result["amount_ratio_20"] = result["total_amount_100m"] / amount_avg
    result["breadth_ma60_change_pp"] = result["above_ma60_ratio"].diff() * 100
    return result


def run(
    start_date: str | None = None,
    end_date: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
) -> pd.DataFrame:
    end = end_date or date.today().strftime("%Y%m%d")
    start = start_date or (date.today() - timedelta(days=240)).strftime("%Y%m%d")
    return calculate_breadth(fetch_daily(create_pro(config_path), start, end))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", help="结束日期 YYYYMMDD")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default="market_breadth.csv")
    args = parser.parse_args()
    result = run(args.start_date, args.end_date, args.config)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"已输出 {len(result)} 个交易日：{args.output}")


if __name__ == "__main__":
    main()
