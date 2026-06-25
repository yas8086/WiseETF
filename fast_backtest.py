"""
向量化ETF轮动策略回测（快速版）
避免逐日循环，用pandas向量化操作一次性计算所有信号和收益
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from config import Config

# ============================================================
# 向量化指标计算
# ============================================================

def calc_momentum(prices, period):
    """动量：过去N日收益率"""
    return prices.pct_change(period)

def calc_smooth_momentum_score(prices, period):
    """平滑动量得分：年化收益率 × R²（用numpy polyfit加速）"""
    scores = pd.Series(np.nan, index=prices.index)
    log_p = np.log(prices.values)
    for i in range(period, len(prices)):
        window = log_p[i-period:i]
        if np.isnan(window).any():
            continue
        x = np.arange(1, period+1, dtype=float)
        # numpy polyfit: 一次多项式拟合
        coeffs = np.polyfit(x, window, 1)
        slope = coeffs[0]
        # 计算R²
        predicted = np.polyval(coeffs, x)
        ss_res = np.sum((window - predicted)**2)
        ss_tot = np.sum((window - window.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        annual_ret = np.exp(slope * 252) - 1
        scores.iloc[i] = annual_ret * r2
    return scores

def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_ma(prices, period):
    return prices.rolling(period).mean()

def calc_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, (dif - dea) * 2

def calc_bollinger(prices, period=20, std_mult=2.0):
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    return ma, ma + std_mult * std, ma - std_mult * std

def calc_donchian(prices, entry=20, exit=10):
    high = prices.rolling(entry).max().shift(1)
    low = prices.rolling(exit).min().shift(1)
    return high, low

def calc_volatility(prices, period=20):
    return prices.pct_change().rolling(period).std() * np.sqrt(252)

# ============================================================
# 向量化策略信号生成
# ============================================================

def vectorized_momentum(close_df, period=20, top_n=1):
    """向量化动量轮动"""
    mom = close_df.pct_change(period)
    signals = pd.DataFrame(0, index=close_df.index, columns=close_df.columns)
    for i in range(len(close_df)):
        row = mom.iloc[i]
        if row.isna().all():
            continue
        top = row.nlargest(top_n)
        for code in top.index:
            val = top[code]
            if not np.isnan(val) and val > 0:
                signals.iat[i, signals.columns.get_loc(code)] = 1
    return signals

def vectorized_smooth_momentum(close_df, period=25, top_n=1):
    """向量化平滑动量轮动"""
    scores = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)
    for code in close_df.columns:
        scores[code] = calc_smooth_momentum_score(close_df[code], period)
    
    signals = pd.DataFrame(0, index=close_df.index, columns=close_df.columns)
    for i in range(len(close_df)):
        row = scores.iloc[i]
        if row.isna().all():
            continue
        top = row.nlargest(top_n)
        for code in top.index:
            val = top[code]
            if not np.isnan(val) and val > 0:
                signals.iat[i, signals.columns.get_loc(code)] = 1
    return signals

def vectorized_dual_ma(close_df, short=10, long=50):
    """向量化双均线轮动"""
    ma_short = close_df.rolling(short).mean()
    ma_long = close_df.rolling(long).mean()
    # 多头排列的ETF
    bullish = (ma_short > ma_long) & (close_df > ma_short)
    
    # 在多头ETF中选动量最强的
    mom = close_df.pct_change(20)
    signals = pd.DataFrame(0, index=close_df.index, columns=close_df.columns)
    for i in range(len(close_df)):
        bull = bullish.iloc[i]
        if bull.sum() == 0:
            continue
        bull_mom = mom.iloc[i][bull]
        if bull_mom.isna().all():
            continue
        top = bull_mom.idxmax()
        if not np.isnan(bull_mom[top]):
            signals.iat[i, signals.columns.get_loc(top)] = 1
    return signals

def vectorized_rsi(close_df, rsi_period=14, oversold=30, overbought=70):
    """向量化RSI策略"""
    rsi_df = close_df.apply(lambda x: calc_rsi(x, rsi_period))
    signals = pd.DataFrame(0, index=close_df.index, columns=close_df.columns)
    for i in range(len(close_df)):
        row = rsi_df.iloc[i]
        # 选RSI最低的（最超卖）
        valid = row[row < 40]
        if len(valid) == 0:
            continue
        top = valid.idxmin()
        signals.iat[i, signals.columns.get_loc(top)] = 1
    return signals

def vectorized_macd(close_df, fast=12, slow=26, signal=9):
    """向量化MACD策略（预计算优化版）"""
    # 预计算所有ETF的MACD柱状图
    hist_df = pd.DataFrame(0, index=close_df.index, columns=close_df.columns)
    golden_df = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)
    for code in close_df.columns:
        dif, dea, hist = calc_macd(close_df[code], fast, slow, signal)
        hist_df[code] = hist
        golden_df[code] = (dif > dea) & (hist > 0)
    
    # 每天选金叉且柱状图最大的ETF
    signals = pd.DataFrame(0, index=close_df.index, columns=close_df.columns)
    for i in range(len(close_df)):
        golden_row = golden_df.iloc[i]
        if golden_row.sum() == 0:
            continue
        # 在金叉ETF中选柱状图最大的
        hist_row = hist_df.iloc[i][golden_row]
        if hist_row.isna().all():
            continue
        top = hist_row.idxmax()
        if not np.isnan(hist_row[top]) and hist_row[top] > 0:
            signals.iat[i, signals.columns.get_loc(top)] = 1
    return signals

def vectorized_donchian(close_df, entry=20, exit_period=10):
    """向量化唐奇安通道突破"""
    signals = pd.DataFrame(0, index=close_df.index, columns=close_df.columns)
    for code in close_df.columns:
        high, low = calc_donchian(close_df[code], entry, exit_period)
        breakout = close_df[code] > high
        signals[code] = breakout.astype(int)
    
    # 每天选突破强度最大的
    result = pd.DataFrame(0, index=close_df.index, columns=close_df.columns)
    for i in range(len(close_df)):
        row = signals.iloc[i]
        if row.sum() == 0:
            continue
        # 选突破的ETF
        top_codes = row[row == 1].index.tolist()
        if len(top_codes) == 1:
            result.iat[i, result.columns.get_loc(top_codes[0])] = 1
        else:
            # 选突破强度最大的
            strengths = {}
            for code in top_codes:
                high, _ = calc_donchian(close_df[code].iloc[:i+1], entry, exit_period)
                strengths[code] = close_df[code].iloc[i] / high.iloc[-1] - 1
            top = max(strengths, key=strengths.get)
            result.iat[i, result.columns.get_loc(top)] = 1
    return result

def vectorized_risk_parity(close_df, vol_window=20, max_holdings=5):
    """向量化风险平价（选低波动率ETF）"""
    vol = close_df.pct_change().rolling(vol_window).std() * np.sqrt(252)
    signals = pd.DataFrame(0, index=close_df.index, columns=close_df.columns)
    for i in range(len(close_df)):
        row = vol.iloc[i]
        valid = row.dropna()
        if len(valid) < 2:
            continue
        # 选波动率最低的N只
        low_vol = valid.nsmallest(max_holdings)
        for code in low_vol.index:
            signals.iat[i, signals.columns.get_loc(code)] = 1
    return signals

def vectorized_trend_score(close_df, period=25, top_n=1):
    """向量化趋势得分（斜率×R²，用numpy polyfit加速）"""
    scores = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)
    for code in close_df.columns:
        prices = close_df[code].values
        for i in range(period, len(prices)):
            window = prices[i-period:i]
            if np.isnan(window).any():
                continue
            normalized = window / window[0]
            x = np.arange(1, period+1, dtype=float)
            coeffs = np.polyfit(x, normalized, 1)
            slope = coeffs[0]
            predicted = np.polyval(coeffs, x)
            ss_res = np.sum((normalized - predicted)**2)
            ss_tot = np.sum((normalized - normalized.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            scores.iloc[i][code] = 10000 * slope * r2
    
    signals = pd.DataFrame(0, index=close_df.index, columns=close_df.columns)
    for i in range(len(close_df)):
        row = scores.iloc[i]
        if row.isna().all():
            continue
        top = row.nlargest(top_n)
        for code in top.index:
            val = top[code]
            if not np.isnan(val) and val > 0:
                signals.iat[i, signals.columns.get_loc(code)] = 1
    return signals

# ============================================================
# 向量化回测引擎
# ============================================================

def vectorized_backtest(signals, close_df, initial_capital=100000, 
                         commission_rate=0.0003, slippage=0.001, rebalance_freq=1):
    """
    向量化回测
    signals: 每日持仓信号DataFrame (1=持有, 0=不持有)
    close_df: 收盘价DataFrame
    """
    # 信号后移1天避免未来函数
    positions = signals.shift(1).fillna(0)
    
    # 调仓频率控制
    if rebalance_freq > 1:
        new_positions = positions.copy()
        last_rebalance = -999
        for i in range(len(positions)):
            if i - last_rebalance >= rebalance_freq:
                last_rebalance = i
            else:
                new_positions.iloc[i] = new_positions.iloc[i-1] if i > 0 else 0
        positions = new_positions
    
    # 每日收益率
    daily_ret = close_df.pct_change()
    
    # 等权分配（持有的ETF等权）
    n_holdings = positions.sum(axis=1).replace(0, 1)
    weights = positions.div(n_holdings, axis=0)
    
    # 组合每日收益
    portfolio_ret = (weights * daily_ret).sum(axis=1)
    portfolio_ret = portfolio_ret.fillna(0)
    
    # 交易成本：换手率 × (佣金 + 滑点)
    turnover = (positions - positions.shift(1).fillna(0)).abs().sum(axis=1) / 2
    cost = turnover * (commission_rate + slippage)
    portfolio_ret -= cost
    
    # 净值曲线
    equity = (1 + portfolio_ret).cumprod() * initial_capital
    
    # 绩效指标
    final_value = equity.iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    days = (equity.index[-1] - equity.index[0]).days
    annual_return = ((final_value / initial_capital) ** (365/days) - 1) * 100 if days > 0 else 0
    
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    daily_returns = portfolio_ret.replace(0, np.nan).dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # 换手率统计
    total_turnover = turnover.sum()
    n_trades = int(total_turnover)
    
    return {
        'total_return': round(total_return, 2),
        'annual_return': round(annual_return, 2),
        'max_drawdown': round(max_drawdown, 2),
        'sharpe_ratio': round(sharpe, 3),
        'calmar_ratio': round(calmar, 3),
        'trade_count': n_trades,
        'final_value': round(final_value, 2),
        'equity_curve': equity,
    }

# ============================================================
# 主函数
# ============================================================

def run_fast_backtest():
    """运行快速向量化回测"""
    close_df = pd.read_pickle('real_close_data.pkl')
    
    print(f'真实数据: {len(close_df.columns)}只ETF, {len(close_df)}个交易日')
    print(f'时间范围: {close_df.index[0].strftime("%Y-%m-%d")} ~ {close_df.index[-1].strftime("%Y-%m-%d")}')
    print(f'ETF: {[Config.ETF_NAMES.get(c,c) for c in close_df.columns]}')
    print(f'\n开始向量化回测...\n')
    
    # 定义所有策略
    strategies = {
        '动量(10日)': lambda: vectorized_momentum(close_df, 10),
        '动量(20日)': lambda: vectorized_momentum(close_df, 20),
        '动量(60日)': lambda: vectorized_momentum(close_df, 60),
        '动量(120日)': lambda: vectorized_momentum(close_df, 120),
        '平滑动量(25日)': lambda: vectorized_smooth_momentum(close_df, 25),
        '平滑动量(60日)': lambda: vectorized_smooth_momentum(close_df, 60),
        '趋势得分(25日)': lambda: vectorized_trend_score(close_df, 25),
        '双均线(5/20)': lambda: vectorized_dual_ma(close_df, 5, 20),
        '双均线(10/50)': lambda: vectorized_dual_ma(close_df, 10, 50),
        '双均线(10/100)': lambda: vectorized_dual_ma(close_df, 10, 100),
        '唐奇安通道(20/10)': lambda: vectorized_donchian(close_df, 20, 10),
        'MACD(12/26/9)': lambda: vectorized_macd(close_df),
        'RSI超卖(14)': lambda: vectorized_rsi(close_df),
        '风险平价(20日)': lambda: vectorized_risk_parity(close_df, 20, 5),
    }
    
    results = []
    for name, func in strategies.items():
        print(f'  回测: {name}...', end=' ', flush=True)
        try:
            signals = func()
            r = vectorized_backtest(signals, close_df)
            r['strategy_name'] = name
            results.append(r)
            print(f'年化{r["annual_return"]:+.1f}%')
        except Exception as e:
            print(f'ERROR: {e}')
    
    # 排序
    results.sort(key=lambda x: x['annual_return'], reverse=True)
    
    # 打印结果
    print(f'\n{"="*120}')
    print(f'  ETF轮动策略真实数据回测结果（向量化快速版）')
    print(f'  回测区间: {close_df.index[0].strftime("%Y-%m-%d")} ~ {close_df.index[-1].strftime("%Y-%m-%d")}')
    print(f'  初始资金: 100,000元 | 佣金: 万3 | 滑点: 0.1%')
    print(f'{"="*120}')
    print(f'{"排名":>4s}  {"策略名称":<24s}  {"总收益":>8s}  {"年化":>8s}  {"最大回撤":>8s}  {"夏普":>7s}  {"Calmar":>7s}  {"交易数":>6s}  {"最终净值":>10s}')
    print('-'*90)
    
    for i, r in enumerate(results):
        tr = r['total_return']
        ar = r['annual_return']
        dd = r['max_drawdown']
        sh = r['sharpe_ratio']
        cl = r['calmar_ratio']
        tc = r['trade_count']
        fv = r['final_value']
        mark = '+' if tr >= 0 else ''
        print(f'  {i+1:>2d}.  {r["strategy_name"]:<24s}  {mark}{tr:>7.1f}%  {mark}{ar:>7.1f}%  {dd:>7.1f}%  {sh:>7.3f}  {cl:>7.2f}  {tc:>5d}次  {fv:>10,.0f}')
    
    print('-'*90)
    
    # 基准
    print(f'\n  --- 基准对比(买入持有) ---')
    for code in close_df.columns:
        ret = (close_df[code].iloc[-1] / close_df[code].iloc[0] - 1) * 100
        name = Config.ETF_NAMES.get(code, code)
        print(f'    {name:<16s}: {ret:+.2f}%')
    
    # 最优
    best = results[0]
    by_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
    by_calmar = max(results, key=lambda x: x['calmar_ratio'])
    print(f'\n  最优(年化): {best["strategy_name"]} -> 年化{best["annual_return"]:+.2f}% 回撤{best["max_drawdown"]:.2f}% 夏普{best["sharpe_ratio"]:.3f}')
    print(f'  最优(夏普): {by_sharpe["strategy_name"]} -> 年化{by_sharpe["annual_return"]:+.2f}% 回撤{by_sharpe["max_drawdown"]:.2f}% 夏普{by_sharpe["sharpe_ratio"]:.3f}')
    print(f'  最优(Calmar): {by_calmar["strategy_name"]} -> 年化{by_calmar["annual_return"]:+.2f}% 回撤{by_calmar["max_drawdown"]:.2f}% Calmar{by_calmar["calmar_ratio"]:.3f}')
    
    # 逐年收益
    print(f'\n{"="*80}')
    print(f'  最优策略逐年收益: {best["strategy_name"]}')
    print(f'{"="*80}')
    eq = best['equity_curve']
    eq_yearly = eq.resample('YE').last()
    eq_yearly_prev = eq.resample('YE').first()
    for i in range(len(eq_yearly)):
        year = eq_yearly.index[i].year
        start_val = eq_yearly_prev.iloc[i]
        end_val = eq_yearly.iloc[i]
        year_ret = (end_val / start_val - 1) * 100
        print(f'  {year}: {year_ret:+.2f}%')
    
    print(f'\n{"="*120}')
    return results


if __name__ == '__main__':
    run_fast_backtest()
