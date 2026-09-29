#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模块03配套：Tushare ETF日频数据增量采集与报告调用（Python 3.9+）。

首次运行：将本文件、report_03_etf_flow.py、etf_watchlist.csv放在同一目录；
默认从脚本同目录 config.yaml 的 keys.tushare 读取密钥；也支持 TUSHARE_TOKEN 环境变量。
密钥优先级：环境变量 > config.yaml > SDK已保存的本地凭据。
运行：python fetch_03_etf_data.py --build-report。

只抓选定ETF，不使用8000积分的etf_basic/etf_share_size；对指数映射严格手工维护。
份额、净值、指数是可选数据，即使其中一项API失败，仍保留历史并生成日线报告。
--date 指定预期交易日；默认读取同目录market_breadth.csv中的最后交易日。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SPECS = {
    'etf_daily.csv': ('ts_code','trade_date'),
    'etf_share.csv': ('ts_code','trade_date'),
    'etf_nav.csv': ('ts_code','nav_date','ann_date'),
    'index_daily.csv': ('ts_code','trade_date'),
}
DAILY_COLS = ['ts_code','trade_date','open','high','low','close','pre_close','change','pct_chg','vol','amount']
SHARE_COLS = ['ts_code','trade_date','fd_share']
NAV_COLS = ['ts_code','ann_date','nav_date','unit_nav']
INDEX_COLS = ['ts_code','trade_date','open','high','low','close','pre_close','change','pct_chg','vol','amount']


def date_str(raw: object) -> str:
    value = str(raw).strip().replace('-', '').replace('/', '')
    if value.endswith('.0') and value[:-2].isdigit():
        value = value[:-2]
    return datetime.strptime(value, '%Y%m%d').strftime('%Y%m%d')


def last_market_day(root: Path) -> str | None:
    path = root / 'market_breadth.csv'
    if not path.exists():
        return None
    table = pd.read_csv(path, usecols=['trade_date'], dtype=str, encoding='utf-8-sig')
    return max(map(date_str, table['trade_date'].dropna())) if not table.empty else None


def watchlist(root: Path) -> pd.DataFrame:
    path = root / 'etf_watchlist.csv'
    if not path.exists():
        raise FileNotFoundError('缺少 etf_watchlist.csv；请使用配套模板填写需要追踪的ETF代码，勿直接以行业代码代替。')
    w = pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    if 'ts_code' not in w:
        raise ValueError('etf_watchlist.csv 缺少 ts_code')
    for col in ('name','index_code','category','focus'):
        if col not in w: w[col] = ''
    w = w[['ts_code','name','index_code','category','focus']].copy()
    for col in w.columns: w[col] = w[col].astype(str).str.strip()
    w['ts_code'] = w['ts_code'].str.upper()
    w['index_code'] = w['index_code'].str.upper()
    w = w.loc[w['ts_code'].ne('')].copy()
    if w.empty or w['ts_code'].duplicated().any():
        raise ValueError('ETF观察表为空或ETF代码重复')
    if not w['ts_code'].str.fullmatch(r'\d{6}\.(SH|SZ)').all():
        raise ValueError('ETF代码需为 6位数字.SH 或 .SZ；请不要填指数代码')
    for code in w['index_code'].loc[w['index_code'].ne('')]:
        if not __import__('re').fullmatch(r'[A-Z0-9]+\.[A-Z]+', code):
            raise ValueError(f'跟踪指数 {code!r} 格式错误；不确定时留空')
    return w


def client_from_env(root: Path = ROOT, config: str = 'config.yaml'):
    """按 环境变量 > 项目 config.yaml > Tushare SDK 本地凭据 初始化。

    config 相对路径相对于 --directory（默认本脚本所在目录），不输出密钥。
    """
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError('当前py39环境未安装tushare，请先运行 python -m pip install -U tushare') from exc

    token = os.environ.get('TUSHARE_TOKEN', '').strip()
    source = 'TUSHARE_TOKEN 环境变量' if token else ''
    if not token:
        config_path = Path(config).expanduser()
        if not config_path.is_absolute():
            config_path = root / config_path
        if config_path.is_file():
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError('读取 config.yaml 需要 PyYAML；请运行 python -m pip install pyyaml') from exc
            try:
                cfg = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
            except Exception:
                # 避免第三方解析异常把密钥所在行带入日志。
                raise RuntimeError(f'配置文件 {config_path.name} 解析失败，请检查 YAML 格式。') from None
            if not isinstance(cfg, dict) or not isinstance(cfg.get('keys'), dict):
                raise RuntimeError(f'{config_path.name} 缺少 keys 配置块。')
            token = str(cfg['keys'].get('tushare') or '').strip()
            if not token:
                raise RuntimeError(f'{config_path.name} 缺少 keys.tushare；请在本地配置有效 Token。')
            source = f'{config_path.name} → keys.tushare'

    try:
        client = ts.pro_api(token) if token else ts.pro_api()
    except Exception as exc:
        raise RuntimeError(f'Tushare 初始化失败（凭据来源：{source or "SDK本地凭据"}；异常类型：{type(exc).__name__}）。') from None
    print(f'Tushare 凭据来源：{source or "SDK本地凭据"}（Token 不显示）')
    return client


def api_call(pro, endpoint: str, *, attempts: int = 3, pause: float = 0.7, **kwargs) -> pd.DataFrame:
    for attempt in range(1, attempts+1):
        try:
            result = getattr(pro, endpoint)(**kwargs)
            if result is None:
                return pd.DataFrame()
            if not isinstance(result, pd.DataFrame):
                raise TypeError(f'{endpoint} 非DataFrame返回值')
            return result.copy()
        except Exception as exc:
            if attempt == attempts:
                raise RuntimeError(f'{endpoint} {kwargs} 连续{attempts}次失败：{exc}') from exc
            print(f'  重试 {endpoint} ({attempt}/{attempts}): {exc}')
            time.sleep(pause * attempt)
    raise RuntimeError('unreachable')


def read_old(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    frame = pd.read_csv(path, dtype={'ts_code':str,'index_code':str},encoding='utf-8-sig',low_memory=False)
    if 'ts_code' in frame:
        frame['ts_code'] = frame['ts_code'].astype(str).str.upper().str.strip()
    for c in ('trade_date','nav_date','ann_date'):
        if c in frame:
            frame[c] = frame[c].dropna().map(date_str).reindex(frame.index)
    return frame


def fetched_start(existing: pd.DataFrame, code: str, date_field: str, bootstrap: str, overlap_days: int) -> str:
    if existing.empty or date_field not in existing or 'ts_code' not in existing:
        return bootstrap
    dates = existing.loc[existing['ts_code'].eq(code),date_field].dropna()
    if dates.empty: return bootstrap
    latest = datetime.strptime(max(dates), '%Y%m%d')
    return max(datetime.strptime(bootstrap, '%Y%m%d'), latest-timedelta(days=overlap_days)).strftime('%Y%m%d')


def chunks(start: str, end: str, days: int = 270):
    current = datetime.strptime(start,'%Y%m%d')
    finish = datetime.strptime(end,'%Y%m%d')
    while current <= finish:
        stop = min(finish, current + timedelta(days=days-1))
        yield current.strftime('%Y%m%d'), stop.strftime('%Y%m%d')
        current = stop + timedelta(days=1)


def merge_save(path: Path, old: pd.DataFrame, incoming: list[pd.DataFrame], keys: tuple[str,...]) -> pd.DataFrame:
    good = [part.copy() for part in incoming if part is not None and not part.empty]
    if not good and old.empty:
        return old
    pieces = ([old] if not old.empty else []) + good
    frame = pd.concat(pieces, ignore_index=True, sort=False)
    for c in keys:
        if c not in frame.columns: raise ValueError(f'{path.name} 缺少关键字段 {c}')
    frame['ts_code'] = frame['ts_code'].astype(str).str.upper().str.strip()
    for key in keys:
        if key.endswith('_date'):
            frame[key] = frame[key].map(lambda v: date_str(v) if pd.notna(v) and str(v).strip() else '')
    frame = frame.drop_duplicates(list(keys),keep='last').sort_values(list(keys)).reset_index(drop=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temp,index=False,encoding='utf-8-sig',na_rep='')
    os.replace(temp,path)
    print(f'  已保存 {path.name}: {len(frame)} 行；本轮新增/覆盖 {sum(len(x) for x in good)} 行')
    return frame


def fetch_stream(pro, endpt: str, codes: list[str], *, old: pd.DataFrame, date_field: str,
                 start: str, end: str, overlap: int, extra: dict | None = None,
                 attempts: int = 3, pause: float = .7) -> tuple[list[pd.DataFrame],list[str]]:
    outputs, errors = [], []
    for code in codes:
        begin = fetched_start(old,code,date_field,start,overlap)
        if begin > end: continue
        print(f'  {endpt}: {code} {begin} ~ {end}')
        for lo,hi in chunks(begin,end):
            opts = {'ts_code':code,'start_date':lo,'end_date':hi}
            if extra: opts.update(extra)
            try:
                part = api_call(pro,endpt,attempts=attempts,pause=pause,**opts)
                if not part.empty:
                    required = {'ts_code', date_field}
                    if not required.issubset(part.columns):
                        raise ValueError(f'返回缺少字段 {required-set(part.columns)}')
                    outputs.append(part)
                else: print(f'    空窗口：{lo}-{hi}（不以零填充）')
            except Exception as exc:
                errors.append(f'{code} {lo}-{hi}: {exc}')
                print(f'    错误：{errors[-1]}')
            time.sleep(pause)
    return outputs, errors


def run(args, pro=None) -> int:
    root = Path(args.directory).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    w = watchlist(root)
    inferred = last_market_day(root)
    end = date_str(args.date) if args.date else (inferred or datetime.now().strftime('%Y%m%d'))
    start = date_str(args.start_date) if args.start_date else (datetime.strptime(end,'%Y%m%d') - timedelta(days=args.bootstrap_days)).strftime('%Y%m%d')
    if start > end: raise ValueError('开始日期不得晚于截止日期')
    print(f'ETF数据窗口：{start} 至 {end}；观察ETF {len(w)} 只；市场参考日 {inferred or "无"}')
    if pro is None: pro = client_from_env(root, getattr(args, 'config', 'config.yaml'))
    old_daily = read_old(root/'etf_daily.csv')
    daily, failed_daily = fetch_stream(pro,'fund_daily',list(w.ts_code),old=old_daily,date_field='trade_date',
                                      start=start,end=end,overlap=args.overlap_days,pause=args.pause)
    if failed_daily:
        print('ERROR: ETF日线抓取存在失败；为防止错误日报，不更新日线CSV。')
        return 2
    if not daily and old_daily.empty:
        print('ERROR: ETF日线无任何结果，不能生成报告。请检查ETF代码、权限和交易日期。')
        return 2
    # 只允许主数据齐全/无请求错误后提交；可选接口失败不能抹去已有历史。
    d = merge_save(root/'etf_daily.csv',old_daily,daily,SPECS['etf_daily.csv'])
    for field in ('close','amount'):
        if field not in d or d[field].isna().all():
            raise ValueError(f'ETF日线缺少 {field}，报告无法计算')
    # fund_basic 仅辅助校验名称，不获取需8000分的etf_basic，也不猜测跟踪指数。
    if args.refresh_names:
        try:
            basic = api_call(pro,'fund_basic',market='E',status='L',pause=args.pause)
            if {'ts_code','name'}.issubset(basic.columns):
                mapping = basic.set_index('ts_code')['name'].to_dict()
                mask = w['name'].eq('')
                w.loc[mask,'name'] = w.loc[mask,'ts_code'].map(mapping).fillna('')
                w.to_csv(root/'etf_watchlist.csv',index=False,encoding='utf-8-sig')
                print('  已用fund_basic填充空白基金名称；未知名称仍为空。')
        except Exception as exc:
            print(f'  注意：fund_basic名称校验跳过：{exc}')
    share_old = read_old(root/'etf_share.csv')
    share,error_share = fetch_stream(pro,'fund_share',list(w.ts_code),old=share_old,date_field='trade_date',
                                   start=start,end=end,overlap=args.overlap_days,pause=args.pause)
    if share or not share_old.empty:
        merge_save(root/'etf_share.csv',share_old,share,SPECS['etf_share.csv'])
    nav_old = read_old(root/'etf_nav.csv')
    nav,error_nav = fetch_stream(pro,'fund_nav',list(w.ts_code),old=nav_old,date_field='nav_date',
                               start=start,end=end,overlap=args.overlap_days,pause=args.pause)
    safe_nav = []
    for part in nav:
        if not {'ts_code','nav_date','unit_nav'}.issubset(part.columns):
            error_nav.append('fund_nav 返回字段缺少 ts_code/nav_date/unit_nav')
            continue
        if 'ann_date' in part:
            kept = part.loc[part['ann_date'].notna() & part['ann_date'].astype(str).str.strip().ne('')].copy()
            if len(kept)<len(part): print(f'  净值排除 {len(part)-len(kept)} 条无公告日记录，防止时间穿越')
            safe_nav.append(kept)
        else:
            # 缺少公告日就不添加公告日字段，防止把净值日期误称为披露日期。
            safe_nav.append(part)
            print('  注意：fund_nav未提供公告日；原样保存，历史时点是否已知无法核验。')
    if safe_nav or not nav_old.empty:
        fresh_announce = any('ann_date' in p.columns for p in safe_nav if not p.empty)
        old_announce = not nav_old.empty and 'ann_date' in nav_old.columns
        if old_announce and any('ann_date' not in p.columns for p in safe_nav if not p.empty):
            safe_nav = [p for p in safe_nav if 'ann_date' in p.columns]
            print('  警告：已有公告日版本，缺公告日的新记录不合并，避免混淆历史披露口径。')
        elif fresh_announce and not old_announce and not nav_old.empty:
            safe_nav = [p for p in safe_nav if 'ann_date' not in p.columns]
            print('  警告：原有净值无公告日，暂保留原口径；新公告日版本待人工迁移。')
        nav_keys = SPECS['etf_nav.csv'] if (old_announce or (nav_old.empty and fresh_announce)) else ('ts_code','nav_date')
        merge_save(root/'etf_nav.csv',nav_old,safe_nav,nav_keys)
    indices = sorted(set(x for x in w['index_code'] if x))
    index_old = read_old(root/'index_daily.csv')
    if indices:
        ix,error_ix = fetch_stream(pro,'index_daily',indices,old=index_old,date_field='trade_date',
                                  start=start,end=end,overlap=args.overlap_days,pause=args.pause)
        if ix or not index_old.empty:
            merge_save(root/'index_daily.csv',index_old,ix,SPECS['index_daily.csv'])
    else: error_ix=[]
    for label,errs in (('份额',error_share),('净值',error_nav),('指数',error_ix)):
        if errs: print(f'提示：{label}部分失败 {len(errs)} 窗口；该类数据按真实已有记录展示，缺口保持空白。')
    available = d.loc[d['trade_date'].astype(str).str.replace('-','',regex=False).eq(end)]
    if len(available)!=len(w):
        print(f'注意：{end} 仅返回 {len(available)}/{len(w)} 只ETF的日线；未返回品种不会冒充已更新。')
    if args.build_report:
        if available.empty:
            print(f'ERROR: {end} 没有任何ETF日线，拒绝生成过期报告。')
            return 3
        script = root/'report_03_etf_flow.py'
        if not script.exists(): raise FileNotFoundError(f'请把 report_03_etf_flow.py 放到 {root}')
        command=[sys.executable,str(script),'--input',str(root/'etf_daily.csv'),'--shares',str(root/'etf_share.csv'),
                 '--nav',str(root/'etf_nav.csv'),'--index',str(root/'index_daily.csv'),
                 '--map',str(root/'etf_watchlist.csv'),'--date',end,
                 '--output',str(root/'output/pdf/report_03_etf_flow.pdf'),
                 '--summary-json',str(root/'output/data/report_03_etf_flow.json')]
        print('生成PDF:', ' '.join(command))
        subprocess.run(command,check=True,cwd=root)
    return 0


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--directory',default=str(ROOT),help='所有CSV及脚本所在目录，默认本文件目录')
    parser.add_argument('--config',default='config.yaml',help='Token配置文件，默认读取--directory下的config.yaml')
    parser.add_argument('--date',help='预期收盘交易日YYYYMMDD；默认market_breadth.csv的最新日期，若不存在则本机日期')
    parser.add_argument('--start-date',help='可选首次追溯起点YYYYMMDD，如20250101')
    parser.add_argument('--bootstrap-days',type=int,default=420,help='首次下载回看自然日，默认420天')
    parser.add_argument('--overlap-days',type=int,default=14,help='增量回溯自然日，默认14天')
    parser.add_argument('--pause',type=float,default=0.4,help='每次API调用间隔秒数，默认0.4')
    parser.add_argument('--refresh-names',action='store_true',help='使用fund_basic填充观察表中留空的基金名称')
    parser.add_argument('--build-report',action='store_true',help='下载完直接运行report_03_etf_flow.py')
    return parser


def main():
    try:
        code=run(make_parser().parse_args())
    except (ValueError,FileNotFoundError,RuntimeError,subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        code=1
    sys.exit(code)

if __name__ == '__main__': main()
