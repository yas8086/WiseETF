import pandas as pd
import numpy as np
from datetime import datetime
from strategies import MomentumStrategy, MAStrategy, DualMomentumStrategy

class BacktestEngine:
    def __init__(self, initial_capital=100000, commission_rate=0.0003,
                 slippage=0.001, rebalance_freq=1, signal_buffer=0.0):
        """
        Args:
            initial_capital: 初始资金
            commission_rate: 佣金费率(万3)
            slippage: 滑点(0.1%)
            rebalance_freq: 调仓频率(每N个交易日调仓一次)，默认1=每日
            signal_buffer: 信号缓冲区(排名差距<此值不换仓)，默认0=不缓冲
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.rebalance_freq = rebalance_freq
        self.signal_buffer = signal_buffer
    
    def run_backtest(self, strategy, prices_dict, start_date=None, end_date=None, benchmark_code=None):
        if not prices_dict:
            return None
        
        all_dates = sorted(set().union(*[set(df.index) for df in prices_dict.values()]))
        if start_date:
            all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
        if end_date:
            all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]
        
        prices_df = pd.DataFrame(prices_dict)
        prices_df = prices_df.ffill().dropna()
        
        capital = self.initial_capital
        holdings = {}
        signals = []
        equity_curve = []
        trades = []
        total_commission = 0
        total_slippage_cost = 0
        
        benchmark_prices = None
        if benchmark_code and benchmark_code in prices_df.columns:
            benchmark_prices = prices_df[benchmark_code]
        
        prev_date = None
        last_rebalance_date = None
        for date in prices_df.index:
            if prev_date is None or (date - prev_date).days >= 1:
                current_prices = prices_df.loc[date]
                
                # 调仓频率控制
                if last_rebalance_date is not None:
                    days_since = (date - last_rebalance_date).days
                    if days_since < self.rebalance_freq:
                        # 不调仓，只更新净值
                        portfolio_value = capital
                        for etf_code, pos in holdings.items():
                            if etf_code in current_prices.index and not np.isnan(current_prices[etf_code]):
                                portfolio_value += pos['shares'] * current_prices[etf_code]
                        equity_curve.append({
                            'date': date, 'portfolio_value': portfolio_value,
                            'cash': capital, 'holdings_value': portfolio_value - capital
                        })
                        prev_date = date
                        continue
                
                last_rebalance_date = date
                
                try:
                    import inspect
                    hist_prices = prices_df.loc[:date]
                    sig_kwargs = {
                        'current_holdings': list(holdings.keys()) if holdings else None,
                    }
                    # 检查策略是否支持 benchmark_prices 参数
                    gen_sig = inspect.signature(strategy.generate_signals)
                    if 'benchmark_prices' in gen_sig.parameters:
                        sig_kwargs['benchmark_prices'] = benchmark_prices.loc[:date] if benchmark_prices is not None else None
                    
                    day_signals = strategy.generate_signals(
                        hist_prices,
                        date,
                        **sig_kwargs
                    )
                    
                    for signal in day_signals:
                        etf_code = signal['etf_code']
                        action = signal['action']
                        price = signal['price']
                        
                        if action == 'BUY':
                            if capital > 0:
                                buy_count = len([s for s in day_signals if s['action'] == 'BUY'])
                                position_size = capital / buy_count if buy_count > 0 else capital
                                # 滑点：买入价格略高
                                exec_price = price * (1 + self.slippage)
                                shares = int(position_size / exec_price / 100) * 100
                                if shares > 0:
                                    cost = shares * exec_price
                                    commission = cost * self.commission_rate
                                    slippage_cost = shares * price * self.slippage
                                    total_commission += commission
                                    total_slippage_cost += slippage_cost
                                    holdings[etf_code] = {'shares': shares, 'cost': cost + commission}
                                    capital -= (cost + commission)
                                    trades.append({
                                        'date': date,
                                        'etf_code': etf_code,
                                        'action': 'BUY',
                                        'price': exec_price,
                                        'shares': shares,
                                        'commission': commission
                                    })
                                    signals.append(signal)
                        
                        elif action == 'SELL' and etf_code in holdings:
                            shares = holdings[etf_code]['shares']
                            # 滑点：卖出价格略低
                            exec_price = price * (1 - self.slippage)
                            revenue = shares * exec_price
                            commission = revenue * self.commission_rate
                            slippage_cost = shares * price * self.slippage
                            total_commission += commission
                            total_slippage_cost += slippage_cost
                            capital += (revenue - commission)
                                   
                            profit = (revenue - commission) - holdings[etf_code]['cost']
                            trades.append({
                                'date': date,
                                'etf_code': etf_code,
                                'action': 'SELL',
                                'price': exec_price,
                                'shares': shares,
                                'profit': profit,
                                'commission': commission
                            })
                            del holdings[etf_code]
                            signals.append(signal)
                
                except Exception as e:
                    pass
                
                portfolio_value = capital
                for etf_code, pos in holdings.items():
                    if etf_code in current_prices.index and not np.isnan(current_prices[etf_code]):
                        portfolio_value += pos['shares'] * current_prices[etf_code]
                
                equity_curve.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'cash': capital,
                    'holdings_value': portfolio_value - capital
                })
                
                prev_date = date
        
        results = self._calculate_performance(equity_curve, trades)
        results['strategy_name'] = strategy.name
        results['signals'] = signals
        results['equity_curve'] = equity_curve
        results['trades'] = trades
        results['total_commission'] = round(total_commission, 2)
        results['total_slippage_cost'] = round(total_slippage_cost, 2)
        
        return results
    
    def _calculate_performance(self, equity_curve, trades):
        if not equity_curve:
            return {}
        
        equity_df = pd.DataFrame(equity_curve)
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        annual_return = ((final_value / self.initial_capital) ** (365/days) - 1) * 100 if days > 0 else 0
        
        equity_df['peak'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        equity_df['daily_return'] = equity_df['portfolio_value'].pct_change()
        daily_returns = equity_df['daily_return'].dropna()
        
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calmar ratio: annual return / max drawdown
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        winning_trades = [t for t in trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit', 0) <= 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        # Profit/loss ratio
        avg_win = sum(t.get('profit', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = abs(sum(t.get('profit', 0) for t in losing_trades) / len(losing_trades)) if losing_trades else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        return {
            'total_return': round(total_return, 2),
            'annual_return': round(annual_return, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'calmar_ratio': round(calmar_ratio, 3),
            'win_rate': round(win_rate, 2),
            'profit_loss_ratio': round(profit_loss_ratio, 2),
            'trade_count': len(trades),
            'start_date': equity_df['date'].iloc[0],
            'end_date': equity_df['date'].iloc[-1],
            'final_value': round(final_value, 2)
        }

def optimize_strategy(etf_codes, prices_dict, start_date, end_date):
    results = []
    
    momentum_params = [(20, 1), (20, 2), (60, 1), (60, 2), (120, 1), (120, 2)]
    ma_params = [(5, 20), (10, 50), (10, 100), (20, 50), (20, 120)]
    
    engine = BacktestEngine()
    
    for lookback, top_n in momentum_params:
        strategy = MomentumStrategy(lookback_period=lookback, top_n=top_n)
        result = engine.run_backtest(strategy, prices_dict, start_date, end_date, benchmark_code=etf_codes[0])
        if result:
            result['params'] = {'type': 'momentum', 'lookback': lookback, 'top_n': top_n}
            results.append(result)
    
    for short_ma, long_ma in ma_params:
        strategy = MAStrategy(short_ma=short_ma, long_ma=long_ma)
        result = engine.run_backtest(strategy, prices_dict, start_date, end_date, benchmark_code=etf_codes[0])
        if result:
            result['params'] = {'type': 'ma', 'short_ma': short_ma, 'long_ma': long_ma}
            results.append(result)
    
    for lookback in [20, 60]:
        for short_ma, long_ma in [(10, 50), (10, 100)]:
            strategy = DualMomentumStrategy(
                lookback_period=lookback,
                ma_short=short_ma,
                ma_long=long_ma
            )
            result = engine.run_backtest(strategy, prices_dict, start_date, end_date, benchmark_code=etf_codes[0])
            if result:
                result['params'] = {
                    'type': 'dual_momentum',
                    'lookback': lookback,
                    'ma_short': short_ma,
                    'ma_long': long_ma
                }
                results.append(result)
    
    results.sort(key=lambda x: (x['annual_return'], -abs(x['max_drawdown'])), reverse=True)
    
    return results[:10]
