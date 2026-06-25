"""每日运行脚本：增量更新数据→运行最优策略→生成HTML报告

回测最优策略: 动量(10日) - 年化+16.7%, 夏普0.659
用法:
    python daily_runner.py                    # 默认: 动量(10日)策略
    python daily_runner.py --strategy smooth  # 平滑动量(25日)
    python daily_runner.py --strategy risk    # 风险平价(低回撤)
"""
import sys
import os
import time
from datetime import datetime


def download_etf_data(codes, start_date='20200101'):
    """下载ETF数据(带重试机制)"""
    import akshare as ak
    import pandas as pd
    from config import Config

    close_dict = {}
    for code in codes:
        name = Config.ETF_NAMES.get(code, code)
        for attempt in range(3):
            try:
                df = ak.fund_etf_hist_em(symbol=code, period='daily',
                                          start_date=start_date, adjust='qfq')
                if df is not None and len(df) > 0:
                    df['日期'] = pd.to_datetime(df['日期'])
                    df = df.sort_values('日期').set_index('日期')
                    close_dict[code] = df['收盘'].astype(float)
                    break
            except Exception:
                if attempt < 2:
                    time.sleep(2)
        time.sleep(0.5)

    if not close_dict:
        return None
    return pd.DataFrame(close_dict).ffill().dropna()


def run_daily(strategy_name='momentum10', etf_pool='core'):
    """
    每日运行流程
    Args:
        strategy_name: 策略选择
            momentum10  - 动量(10日) [回测最优 年化+16.7%]
            momentum20  - 动量(20日)
            smooth      - 平滑动量(25日) [年化+8.7% 换手低]
            risk        - 风险平价 [夏普最高0.672 回撤最小-16.9%]
            ma          - 双均线(10/50)
        etf_pool: ETF池选择 (core=15只核心ETF)
    """
    from strategies import (
        MomentumStrategy, SmoothMomentumStrategy, MAStrategy, RiskParityStrategy,
    )
    from backtest import BacktestEngine
    from report_generator import ReportGenerator
    from config import Config
    import pandas as pd

    print(f"\n{'='*60}")
    print(f"  WiseETF每日轮动分析 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # 1. 获取ETF池
    codes = Config.DEFAULT_ETFS
    print(f"\n[1/4] ETF池: {len(codes)}只 - {[Config.ETF_NAMES.get(c,c) for c in codes]}")

    # 2. 下载数据
    print(f"\n[2/4] 下载ETF数据...")
    close_df = download_etf_data(codes)
    if close_df is None or len(close_df.columns) < 2:
        print("  数据不足，退出")
        return None

    print(f"  成功: {len(close_df.columns)}/{len(codes)}只ETF")
    print(f"  数据范围: {close_df.index[0].strftime('%Y-%m-%d')} ~ {close_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  交易日数: {len(close_df)}")

    # 3. 运行策略生成信号
    print(f"\n[3/4] 运行策略生成信号...")

    strategies = {
        'momentum10': ('动量(10日) [回测最优]', MomentumStrategy(lookback_period=10, top_n=1)),
        'momentum20': ('动量(20日)', MomentumStrategy(lookback_period=20, top_n=1)),
        'smooth': ('平滑动量(25日)', SmoothMomentumStrategy(lookback_period=25, top_n=1)),
        'risk': ('风险平价(20日) [低回撤]', RiskParityStrategy(vol_window=20, max_holdings=5)),
        'ma': ('双均线(10/50)', MAStrategy(short_ma=10, long_ma=50)),
    }

    if strategy_name not in strategies:
        strategy_name = 'momentum10'

    strategy_label, strategy = strategies[strategy_name]
    prices_dict = {col: close_df[col] for col in close_df.columns}
    today = close_df.index[-1]
    # 兼容不同策略的generate_signals签名
    try:
        signals = strategy.generate_signals(close_df, today, current_holdings=None)
    except TypeError:
        signals = strategy.generate_signals(close_df, today)

    print(f"  策略: {strategy_label}")
    print(f"  信号: {len(signals)}个")
    for s in signals:
        action = s['action']
        code = s['etf_code']
        name = Config.ETF_NAMES.get(code, code)
        price = s.get('price', 0)
        reason = s.get('reason', '')
        print(f"    {action} {code} {name} @{price:.3f} - {reason}")

    # 4. 生成HTML报告
    print(f"\n[4/4] 生成HTML报告...")
    current_prices = close_df.iloc[-1].to_dict()

    # 构建ETF列表(含分类)
    from etf_classifier import classify_etf
    etf_list = []
    for code in close_df.columns:
        name = Config.ETF_NAMES.get(code, code)
        etf_list.append({
            'code': code,
            'name': name,
            'category': classify_etf(name, code),
        })

    reporter = ReportGenerator(output_dir='reports')
    report_path = reporter.generate_daily_report(
        signals=signals,
        current_prices=current_prices,
        strategy_results=None,
        etf_list=etf_list,
        best_strategy_name=strategy_label,
    )

    print(f"\n{'='*60}")
    print(f"  报告已生成: {report_path}")
    print(f"  可在浏览器中打开查看")
    print(f"{'='*60}")

    return report_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='WiseETF每日轮动分析')
    parser.add_argument('--strategy', type=str, default='momentum10',
                        choices=['momentum10', 'momentum20', 'smooth', 'risk', 'ma'],
                        help='策略选择: momentum10(默认) momentum20 smooth risk ma')
    parser.add_argument('--etf-pool', type=str, default='core', help='ETF池')
    args = parser.parse_args()

    run_daily(strategy_name=args.strategy, etf_pool=args.etf_pool)
