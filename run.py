#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日更新数据并生成 5 份量化监测报告。

在项目目录执行：
    python run.py

也可以指定报告交易日：
    python run.py --date 20260928
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"
CONFIG_PATH = ROOT / "config.yaml"

# 允许从任意当前目录执行本脚本，同时复用各模块独立运行时的实现。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fetch_03_etf_data
import industry_rotation
import market_breadth
import report_01_market_breadth
import report_02_industry_rotation
import report_03_etf_flow
import report_04_index_valuation
import report_05_style_rotation


def _run_cli(main: Callable[[], object], argv: Sequence[str], label: str) -> None:
    """用指定参数调用现有 CLI main，并将非零退出转换为流程错误。"""
    previous_argv = sys.argv
    try:
        sys.argv = [label, *argv]
        try:
            main()
        except SystemExit as exc:
            code = exc.code
            if code not in (None, 0):
                raise RuntimeError(f"{label} 退出码为 {code}") from exc
    finally:
        sys.argv = previous_argv


def _latest_breadth_date(path: Path) -> str:
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"市场广度数据未生成：{path}")
    frame = pd.read_csv(path, usecols=["trade_date"], dtype={"trade_date": str}, encoding="utf-8-sig")
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
    if dates.empty:
        raise RuntimeError(f"市场广度数据中没有有效交易日：{path}")
    return dates.max().strftime("%Y%m%d")


def _load_pdf_tools() -> tuple[Any, Any]:
    """加载 PDF 拼接库，尽早提示缺失依赖，避免抓取完数据后才失败。"""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter
        except ImportError as exc:
            raise RuntimeError("合并日报需要 pypdf，请先运行：python -m pip install pypdf") from exc
    return PdfReader, PdfWriter


def _merge_reports(report_date: str, pdf_paths: Sequence[Path], PdfReader: Any, PdfWriter: Any) -> Path:
    """按列表顺序拼接 PDF，并以交易日命名输出；写入成功后再替换正式文件。"""
    output = PDF_DIR / f"quant_monitor_report_{report_date}.pdf"
    temp_output = output.with_name(f"{output.stem}.tmp.pdf")
    writer = PdfWriter()
    streams = []
    try:
        for pdf_path in pdf_paths:
            if not pdf_path.is_file():
                raise FileNotFoundError(f"缺少待合并的报告：{pdf_path}")
            stream = pdf_path.open("rb")
            streams.append(stream)
            reader = PdfReader(stream)
            if reader.is_encrypted:
                raise RuntimeError(f"报告已加密，无法合并：{pdf_path}")
            if not reader.pages:
                raise RuntimeError(f"报告没有页面，无法合并：{pdf_path}")
            for page in reader.pages:
                writer.add_page(page)

        writer.add_metadata({
            "/Title": f"量化监测报告 {report_date}",
            "/Subject": "市场广度、行业轮动、ETF、指数估值与市场风格",
        })
        with temp_output.open("wb") as output_stream:
            writer.write(output_stream)
        temp_output.replace(output)
    except Exception:
        temp_output.unlink(missing_ok=True)
        raise
    finally:
        for stream in streams:
            stream.close()
    return output


def run_daily(report_date: str | None = None) -> str:
    """按依赖顺序更新数据，生成五份报告，返回实际报告交易日。"""
    config_path = CONFIG_PATH
    if report_date:
        try:
            datetime.strptime(report_date, "%Y%m%d")
        except ValueError as exc:
            raise ValueError("--date 必须是 YYYYMMDD 格式") from exc
    pdf_reader, pdf_writer = _load_pdf_tools()
    report_pdfs = [
        PDF_DIR / "report_01_market_breadth.pdf",
        PDF_DIR / "report_02_industry_rotation.pdf",
        PDF_DIR / "report_03_etf_flow.pdf",
        PDF_DIR / "report_04_index_valuation.pdf",
        PDF_DIR / "report_05_style_rotation.pdf",
    ]

    # 1. 市场广度：获取全市场日线并保存报告数据源。
    print("\n[1/5] 更新市场广度数据")
    breadth = market_breadth.run(end_date=report_date, config_path=config_path)
    breadth_path = DATA_DIR / "market_breadth.csv"
    breadth_path.parent.mkdir(parents=True, exist_ok=True)
    breadth.to_csv(breadth_path, index=False, encoding="utf-8-sig")
    asof = report_date or _latest_breadth_date(breadth_path)
    print(f"报告交易日：{asof}")

    # 2. 行业轮动：保存行业缓存，并同步给模块 5 复用。
    print("\n[2/5] 更新申万行业轮动数据")
    industry_rotation.run(
        end_date=asof,
        config_path=config_path,
        root=DATA_DIR,
        sync_style05_cache=True,
    )

    # 3. 报告 1、2：使用刚更新的数据并锁定同一个交易日。
    print("\n[3/5] 生成市场广度与行业轮动报告")
    _run_cli(
        report_01_market_breadth.main,
        ["--input", str(breadth_path), "--output", str(report_pdfs[0]), "--date", asof],
        "report_01_market_breadth.py",
    )
    _run_cli(
        report_02_industry_rotation.main,
        ["--input", str(DATA_DIR / "industry_rotation.csv"),
         "--output", str(report_pdfs[1]), "--date", asof,
         "--summary-json", str(DATA_DIR / "report_02_industry_rotation_summary.json")],
        "report_02_industry_rotation.py",
    )

    # 4. ETF：先抓数据，再调用报告模块生成 PDF 和摘要。
    print("\n[4/5] 更新 ETF 数据并生成 ETF 报告")
    etf_args = fetch_03_etf_data.make_parser().parse_args([
        "--directory", str(DATA_DIR), "--config", str(config_path), "--date", asof,
    ])
    status = fetch_03_etf_data.run(etf_args)
    if status != 0:
        raise RuntimeError(f"ETF 数据更新失败，退出码为 {status}")
    _run_cli(
        report_03_etf_flow.main,
        ["--input", str(DATA_DIR / "etf_daily.csv"),
         "--shares", str(DATA_DIR / "etf_share.csv"),
         "--nav", str(DATA_DIR / "etf_nav.csv"),
         "--index", str(DATA_DIR / "index_daily.csv"),
         "--map", str(DATA_DIR / "etf_watchlist.csv"), "--date", asof,
         "--output", str(report_pdfs[2]),
         "--summary-json", str(DATA_DIR / "report_03_etf_flow.json")],
        "report_03_etf_flow.py",
    )

    # 5. 估值与风格脚本都内置了数据增量抓取入口。
    print("\n[5/5] 更新指数估值、风格数据并生成报告")
    _run_cli(
        report_04_index_valuation.main,
        ["--directory", str(DATA_DIR), "--fetch", "--date", asof,
         "--output", str(report_pdfs[3])],
        "report_04_index_valuation.py",
    )
    style_args = report_05_style_rotation.parser().parse_args([
        "--directory", str(DATA_DIR), "--fetch", "--date", asof,
        "--output", str(report_pdfs[4]),
    ])
    status = report_05_style_rotation.run(style_args)
    if status != 0:
        raise RuntimeError(f"风格轮动报告生成失败，退出码为 {status}")

    combined_pdf = _merge_reports(asof, report_pdfs, pdf_reader, pdf_writer)
    print(f"\n全部完成：交易日 {asof}；完整报告：{combined_pdf}")
    return asof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="报告交易日 YYYYMMDD；默认自动使用市场广度数据的最新交易日")
    args = parser.parse_args()
    try:
        run_daily(args.date)
    except Exception as exc:
        print(f"\n运行失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
