"""
ETF轮动策略多策略对比回测模块

用法:
    python rotation_backtest.py                        # 使用默认ETF池和参数
    python rotation_backtest.py --start 2018-01-01     # 指定起始日期
    python rotation_backtest.py --etf-pool broad       # 使用宽基池
    python rotation_backtest.py --capital 200000       # 指定初始资金
"""

import argparse
import sys
import time
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime

from backtest import BacktestEngine
from strategies import (
    MomentumStrategy,
    TrendScoreStrategy,
    SmoothMomentumStrategy,
    ThreeFactorStrategy,
    FixedIncomePlusStrategy,
    DualPoolMomentumStrategy,
    DualMomentumStrategy,
    MAStrategy,
)
from config import Config

# ============================================================
# ETF候选池定义
# ============================================================

ETF_POOLS = {
    'full': {
        'name': '全品种池(15只)',
        'codes': Config.DEFAULT_ETFS,
    },
    'broad': {
        'name': '宽基风格池(5只)',
        'codes': ['510300', '510500', '159915', '510880', '518880'],
    },
    'style': {
        'name': '风格轮动池(4只)',
        'codes': ['510300', '510500', '510880', '159915'],
    },
    'global': {
        'name': '全球配置池(6只)',
        'codes': ['510300', '513100', '513500', '518880', '511010', '511880'],
    },
    'industry': {
        'name': '行业主题池(8只)',
        'codes': ['510300', '512880', '512010', '515030', '512480', '518880', '511010', '511880'],
    },
}


# ============================================================
# 样本数据生成（用于离线测试）
# ============================================================

def generate_sample_data(etf_codes, start_date, end_date):
    """
    生成模拟的ETF数据用于离线测试
    """
    print(f"\n{'='*60}")
    print(f"  [离线模式] 生成模拟ETF数据 ({start_date} ~ {end_date})")
    print(f"{'='*60}")
    
    # 生成日期序列
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
    
    # 为每个ETF生成不同的随机走势
    np.random.seed(42)
    close_dict = {}
    volume_dict = {}
    
    # 不同ETF的基础特征
    etf_characteristics = {
        '510300': {'start': 3.5, 'volatility': 0.015, 'drift': 0.0003},   # 沪深300
        '510500': {'start': 5.0, 'volatility': 0.018, 'drift': 0.0002},   # 中证500
        '159915': {'start': 2.0, 'volatility': 0.022, 'drift': 0.0004},   # 创业板
        '510880': {'start': 3.0, 'volatility': 0.012, 'drift': 0.0005},   # 红利
        '518880': {'start': 4.5, 'volatility': 0.010, 'drift': 0.0006},   # 黄金
        '513100': {'start': 1.5, 'volatility': 0.020, 'drift': 0.0007},   # 纳指
        '511010': {'start': 100, 'volatility': 0.002, 'drift': 0.0001},   # 国债
        '511880': {'start': 100, 'volatility': 0.001, 'drift': 0.00005},  # 银华日利
    }
    
    for code in etf_codes:
        name = Config.ETF_NAMES.get(code, code)
        chars = etf_characteristics.get(code, {'start': 1.0, 'volatility': 0.015, 'drift': 0.0003})
        
        # 生成随机价格序列
        returns = np.random.normal(chars['drift'], chars['volatility'], len(dates))
        prices = chars['start'] * np.exp(np.cumsum(returns))
        
        close_dict[code] = pd.Series(prices, index=dates)
        volume_dict[code] = pd.Series(
            np.random.randint(1000000, 10000000, len(dates)),
            index=dates
        )
        
        print(f"  [OK]   {name}({code}): {len(dates)}条数据")
    
    close_df = pd.DataFrame(close_dict)
    volume_df = pd.DataFrame(volume_dict)
    
    print(f"\n  成功生成 {len(close_df.columns)} 只ETF数据，共 {len(close_df)} 个交易日")
    print(f"  时间范围: {close_df.index[0].strftime('%Y-%m-%d')} ~ {close_df.index[-1].strftime('%Y-%m-%d')}")
    
    return close_df, volume_df


# ============================================================
# 数据获取
# ============================================================

def fetch_etf_data(etf_codes, start_date, end_date=None, cache_dir=None):
    """
    批量获取ETF历史数据，返回收盘价DataFrame和成交量DataFrame
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    # 格式化日期
    start_str = start_date.replace('-', '')
    end_str = end_date.replace('-', '')

    print(f"\n{'='*60}")
    print(f"  正在获取ETF数据 ({start_str} ~ {end_str})")
    print(f"{'='*60}")

    close_dict = {}
    volume_dict = {}
    failed = []

    for code in etf_codes:
        name = Config.ETF_NAMES.get(code, code)
        
        # 重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date=start_str,
                    end_date=end_str,
                    adjust="qfq"
                )

                if df is None or len(df) == 0:
                    print(f"  [WARN] {name}({code}) 无数据")
                    failed.append(code)
                    break

                df['日期'] = pd.to_datetime(df['日期'])
                df = df.sort_values('日期').set_index('日期')

                close_dict[code] = df['收盘'].astype(float)
                if '成交量' in df.columns:
                    volume_dict[code] = df['成交量'].astype(float)

                print(f"  [OK]   {name}({code}): {len(df)}条数据")
                time.sleep(0.5)  # 避免请求过快
                break  # 成功则跳出重试循环

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  [RETRY {attempt+1}/{max_retries}] {name}({code}): {e}")
                    time.sleep(2)  # 等待后重试
                else:
                    print(f"  [FAIL] {name}({code}): {e}")
                    failed.append(code)

    if failed:
        print(f"\n  共 {len(failed)} 只ETF获取失败: {failed}")

    # 构建对齐的DataFrame
    close_df = pd.DataFrame(close_dict).ffill().dropna()
    volume_df = pd.DataFrame(volume_dict).reindex(close_df.index).ffill().dropna() if volume_dict else None

    if close_df.empty:
        print(f"\n  [ERROR] 没有成功获取任何数据！")
        return close_df, volume_df

    print(f"\n  成功获取 {len(close_df.columns)} 只ETF数据，共 {len(close_df)} 个交易日")
    print(f"  时间范围: {close_df.index[0].strftime('%Y-%m-%d')} ~ {close_df.index[-1].strftime('%Y-%m-%d')}")

    return close_df, volume_df


# ============================================================
# 策略工厂
# ============================================================

def build_strategies():
    """构建所有28+策略"""
    from strategies import (
        MomentumStrategy, TrendScoreStrategy, MAStrategy,
        DualMomentumStrategy, SmoothMomentumStrategy, ThreeFactorStrategy,
        FixedIncomePlusStrategy, DualPoolMomentumStrategy,
        RSIStrategy, BollingerBandStrategy, DonchianChannelStrategy,
        TurtleStrategy, MACDStrategy, KDJStrategy, CCIStrategy,
        RiskParityStrategy, EqualWeightStrategy, ERPStrategy,
    )
    from strategies_ml import MLRotationStrategy
    
    return [
        # 动量类(4)
        MomentumStrategy(lookback_period=20, top_n=1),
        MomentumStrategy(lookback_period=60, top_n=1),
        SmoothMomentumStrategy(lookback_period=25, top_n=1),
        TrendScoreStrategy(trend_period=25, top_n=1),
        
        # 均线类(3)
        MAStrategy(short_ma=5, long_ma=20),
        MAStrategy(short_ma=10, long_ma=50),
        MAStrategy(short_ma=10, long_ma=100),
        
        # 趋势类(3)
        DonchianChannelStrategy(entry_period=20, exit_period=10),
        TurtleStrategy(entry_period=20, exit_period=10),
        MACDStrategy(fast=12, slow=26, signal=9),
        
        # 多因子类(3)
        DualMomentumStrategy(lookback_period=20, ma_short=10, ma_long=50),
        ThreeFactorStrategy(momentum_period=20, ma_period=20),
        DualPoolMomentumStrategy(lookback_period=20, ma_short=20, ma_long=60),
        
        # 均值回归类(3)
        RSIStrategy(rsi_period=14, oversold=30, overbought=70),
        BollingerBandStrategy(bb_period=20, std_mult=2.0),
        KDJStrategy(rsv_period=9),
        
        # 资产配置类(3)
        RiskParityStrategy(vol_window=20, max_holdings=5),
        FixedIncomePlusStrategy(momentum_period=13, ma_period=10, max_holdings=5),
        ERPStrategy(),
        
        # 技术指标类(2)
        CCIStrategy(cci_period=20),
        EqualWeightStrategy(pool_size=5),
        
        # 机器学习类(2)
        MLRotationStrategy(model_type='rf', lookback=20, prediction_period=5),
        MLRotationStrategy(model_type='xgb', lookback=20, prediction_period=5),
        
        # 不同参数变体(5)
        MomentumStrategy(lookback_period=10, top_n=1),
        MomentumStrategy(lookback_period=120, top_n=1),
        SmoothMomentumStrategy(lookback_period=60, top_n=1),
        ThreeFactorStrategy(momentum_period=60, ma_period=20),
        DualMomentumStrategy(lookback_period=60, ma_short=10, ma_long=60),
    ]


# ============================================================
# 回测执行
# ============================================================

def run_all_strategies(close_df, volume_df, initial_capital=100000,
                       commission_rate=0.0003, slippage=0.001):
    """运行所有策略并收集结果"""
    strategies = build_strategies()
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage=slippage
    )

    prices_dict = {col: close_df[col] for col in close_df.columns}

    print(f"\n{'='*60}")
    print(f"  开始回测 {len(strategies)} 个策略...")
    print(f"  初始资金: {initial_capital:,.0f} 元 | 佣金: {commission_rate*10000:.1f}‱ | 滑点: {slippage*100:.1f}%")
    print(f"{'='*60}\n")

    results = []
    for i, strategy in enumerate(strategies):
        try:
            result = engine.run_backtest(strategy, prices_dict)
            if result and result.get('total_return') is not None:
                result['strategy_name'] = strategy.name
                results.append(result)
                status = "OK"
            else:
                status = "EMPTY"
            print(f"  [{i+1:2d}/{len(strategies)}] {strategy.name:<40s} -> {status}")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(strategies)}] {strategy.name:<40s} -> ERROR: {e}")

    return results


# ============================================================
# 基准对比
# ============================================================

def calculate_benchmark_returns(close_df):
    """计算各ETF的买入持有收益作为基准"""
    benchmarks = {}
    for code in close_df.columns:
        if len(close_df[code]) >= 2:
            ret = (close_df[code].iloc[-1] / close_df[code].iloc[0] - 1) * 100
            name = Config.ETF_NAMES.get(code, code)
            benchmarks[name] = round(ret, 2)
    return benchmarks


# ============================================================
# 结果展示
# ============================================================

def print_comparison_table(results, benchmarks=None):
    """打印策略对比表格"""
    if not results:
        print("\n  没有可用的回测结果！")
        return

    # 按年化收益排序
    results.sort(key=lambda x: x.get('annual_return', 0), reverse=True)

    print(f"\n{'='*120}")
    print(f"  ETF轮动策略回测对比结果")
    print(f"  回测区间: {results[0].get('start_date', 'N/A')} ~ {results[0].get('end_date', 'N/A')}")
    print(f"{'='*120}")

    # 表头
    header = f"{'排名':>4s}  {'策略名称':<42s}  {'总收益':>8s}  {'年化收益':>8s}  {'最大回撤':>8s}  {'夏普比率':>8s}  {'Calmar':>7s}  {'胜率':>6s}  {'盈亏比':>6s}  {'交易次数':>6s}  {'最终净值':>10s}"
    print(header)
    print('-' * 120)

    for i, r in enumerate(results):
        total_ret = r.get('total_return', 0)
        annual_ret = r.get('annual_return', 0)
        max_dd = r.get('max_drawdown', 0)
        sharpe = r.get('sharpe_ratio', 0)
        calmar = r.get('calmar_ratio', 0)
        win_rate = r.get('win_rate', 0)
        pl_ratio = r.get('profit_loss_ratio', 0)
        trades = r.get('trade_count', 0)
        final_val = r.get('final_value', 0)
        name = r.get('strategy_name', 'Unknown')

        # 颜色标记
        ret_mark = '+' if total_ret >= 0 else ''
        dd_mark = ''

        print(f"  {i+1:>2d}.  {name:<42s}  {ret_mark}{total_ret:>7.1f}%  {ret_mark}{annual_ret:>7.1f}%  {max_dd:>7.1f}%  {sharpe:>8.3f}  {calmar:>7.2f}  {win_rate:>5.1f}%  {pl_ratio:>6.2f}  {trades:>5d}次  {final_val:>10,.0f}")

    print('-' * 120)

    # 最优策略摘要
    best = results[0]
    print(f"\n  最优策略(按年化收益): {best['strategy_name']}")
    print(f"    年化收益: {best['annual_return']:+.2f}%  |  最大回撤: {best['max_drawdown']:.2f}%  |  夏普比率: {best['sharpe_ratio']:.3f}")

    # 按夏普比率排序的最优
    by_sharpe = max(results, key=lambda x: x.get('sharpe_ratio', 0))
    print(f"\n  最优策略(按夏普比率): {by_sharpe['strategy_name']}")
    print(f"    年化收益: {by_sharpe['annual_return']:+.2f}%  |  最大回撤: {by_sharpe['max_drawdown']:.2f}%  |  夏普比率: {by_sharpe['sharpe_ratio']:.3f}")

    # 按Calmar排序的最优
    by_calmar = max(results, key=lambda x: x.get('calmar_ratio', 0))
    print(f"\n  最优策略(按Calmar比率): {by_calmar['strategy_name']}")
    print(f"    年化收益: {by_calmar['annual_return']:+.2f}%  |  最大回撤: {by_calmar['max_drawdown']:.2f}%  |  Calmar: {by_calmar['calmar_ratio']:.3f}")

    # 基准对比
    if benchmarks:
        print(f"\n  --- 基准对比(买入持有) ---")
        for name, ret in sorted(benchmarks.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name:<16s}: {ret:+.2f}%")

    return results


def print_yearly_breakdown(results, close_df):
    """打印最优策略的逐年收益"""
    if not results:
        return

    best = results[0]
    equity_curve = best.get('equity_curve', [])
    if not equity_curve:
        return

    df = pd.DataFrame(equity_curve)
    df['year'] = pd.to_datetime(df['date']).dt.year

    print(f"\n{'='*80}")
    print(f"  最优策略逐年收益: {best['strategy_name']}")
    print(f"{'='*80}")
    print(f"  {'年份':>6s}  {'年初净值':>10s}  {'年末净值':>10s}  {'年收益率':>8s}  {'最大回撤':>8s}")
    print(f"  {'-'*50}")

    years = sorted(df['year'].unique())
    for year in years:
        year_data = df[df['year'] == year]
        if len(year_data) < 2:
            continue
        start_val = year_data['portfolio_value'].iloc[0]
        end_val = year_data['portfolio_value'].iloc[-1]
        year_ret = (end_val / start_val - 1) * 100

        # 年内最大回撤
        peak = year_data['portfolio_value'].cummax()
        dd = (year_data['portfolio_value'] - peak) / peak * 100
        year_max_dd = dd.min()

        print(f"  {year:>6d}  {start_val:>10,.0f}  {end_val:>10,.0f}  {year_ret:>+7.1f}%  {year_max_dd:>7.1f}%")

    print(f"  {'-'*50}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='ETF轮动策略多策略对比回测')
    parser.add_argument('--start', type=str, default='2018-01-01',
                        help='回测起始日期 (默认: 2018-01-01)')
    parser.add_argument('--end', type=str, default=None,
                        help='回测结束日期 (默认: 今天)')
    parser.add_argument('--etf-pool', type=str, default='full',
                        choices=list(ETF_POOLS.keys()),
                        help='ETF候选池 (默认: full)')
    parser.add_argument('--capital', type=float, default=100000,
                        help='初始资金 (默认: 100000)')
    parser.add_argument('--commission', type=float, default=0.0003,
                        help='佣金费率 (默认: 0.0003)')
    parser.add_argument('--slippage', type=float, default=0.001,
                        help='滑点 (默认: 0.001)')
    parser.add_argument('--offline', action='store_true',
                        help='使用模拟数据进行离线测试')
    args = parser.parse_args()

    # 选择ETF池
    pool = ETF_POOLS[args.etf_pool]
    etf_codes = pool['codes']
    print(f"\n  使用ETF池: {pool['name']}")
    for code in etf_codes:
        name = Config.ETF_NAMES.get(code, code)
        print(f"    {code} - {name}")

    # 获取数据
    if args.offline:
        close_df, volume_df = generate_sample_data(etf_codes, args.start, args.end or datetime.now().strftime('%Y-%m-%d'))
    else:
        close_df, volume_df = fetch_etf_data(etf_codes, args.start, args.end)
        
        # 如果获取失败，提示使用离线模式
        if close_df.empty:
            print("\n  [提示] 网络数据获取失败，是否使用模拟数据进行离线测试？")
            print("  运行命令: python rotation_backtest.py --offline")
            sys.exit(1)

    if close_df.empty or len(close_df.columns) < 2:
        print("\n  数据不足，无法进行回测！")
        sys.exit(1)

    # 计算基准收益
    benchmarks = calculate_benchmark_returns(close_df)

    # 运行所有策略
    results = run_all_strategies(
        close_df, volume_df,
        initial_capital=args.capital,
        commission_rate=args.commission,
        slippage=args.slippage
    )

    # 打印对比结果
    print_comparison_table(results, benchmarks)

    # 打印逐年收益
    print_yearly_breakdown(results, close_df)

    print(f"\n{'='*60}")
    print(f"  回测完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
