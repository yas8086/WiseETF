import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

class TrendScoreStrategy:
    """基于知乎文章的趋势得分策略（RSRS思想）
    使用收盘价序列的斜率和决定系数R²的乘积作为动量指标
    """
    def __init__(self, trend_period=25, top_n=1):
        self.trend_period = trend_period
        self.top_n = top_n
        self.name = f"趋势得分策略(N={trend_period},前{top_n}只)"
    
    def calculate_score(self, price_series):
        """计算趋势得分：斜率 × R² × 10000"""
        if len(price_series) < self.trend_period:
            return np.nan
        
        prices = price_series.values
        # 归一化价格序列
        normalized_prices = prices / prices[0]
        x = np.arange(1, self.trend_period + 1)
        
        # 线性回归拟合
        lr = LinearRegression()
        lr.fit(x.reshape(-1, 1), normalized_prices)
        
        # 斜率
        slope = lr.coef_[0]
        # 决定系数R²
        r_squared = lr.score(x.reshape(-1, 1), normalized_prices)
        # 得分 = 斜率 × R² × 10000
        score = 10000 * slope * r_squared
        
        return score
    
    def calculate_trend_scores(self, prices_df):
        """计算所有ETF的趋势得分"""
        scores = {}
        for etf_code in prices_df.columns:
            if len(prices_df[etf_code]) >= self.trend_period:
                price_series = prices_df[etf_code].tail(self.trend_period)
                score = self.calculate_score(price_series)
                if not np.isnan(score):
                    scores[etf_code] = score
        return scores
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        """生成交易信号"""
        scores = self.calculate_trend_scores(prices_df)
        
        if not scores:
            return []
        
        # 按得分从高到低排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [x[0] for x in sorted_scores[:self.top_n]]
        
        signals = []
        
        # 生成买入信号
        for etf_code in top_etfs:
            if current_holdings is None or etf_code not in current_holdings:
                signals.append({
                    'date': date,
                    'etf_code': etf_code,
                    'action': 'BUY',
                    'price': prices_df[etf_code].iloc[-1],
                    'reason': f"趋势得分 {scores[etf_code]:.4f}，排名前{self.top_n}"
                })
        
        # 生成卖出信号（不在前N名且当前持有的）
        for holding in (current_holdings or []):
            if holding not in top_etfs and holding in scores:
                signals.append({
                    'date': date,
                    'etf_code': holding,
                    'action': 'SELL',
                    'price': prices_df[holding].iloc[-1],
                    'reason': f"趋势得分 {scores[holding]:.4f}，已不在前{self.top_n}"
                })
        
        return signals

class MomentumStrategy:
    def __init__(self, lookback_period=20, top_n=1):
        self.lookback_period = lookback_period
        self.top_n = top_n
        self.name = f"动量策略({lookback_period}日,{top_n}只)"
    
    def calculate_momentum(self, prices_df):
        momentum = {}
        for etf_code in prices_df.columns:
            if len(prices_df[etf_code]) >= self.lookback_period:
                current_price = prices_df[etf_code].iloc[-1]
                past_price = prices_df[etf_code].iloc[-self.lookback_period]
                momentum[etf_code] = (current_price - past_price) / past_price * 100
        return momentum
    
    def generate_signals(self, prices_df, date):
        momentum = self.calculate_momentum(prices_df)
        sorted_momentum = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
        
        signals = []
        top_etfs = [x[0] for x in sorted_momentum[:self.top_n]]
        
        for etf_code in top_etfs:
            signals.append({
                'date': date,
                'etf_code': etf_code,
                'action': 'BUY',
                'price': prices_df[etf_code].iloc[-1],
                'reason': f"动量排名前{self.top_n}，{self.lookback_period}日收益率: {momentum[etf_code]:.2f}%"
            })
        
        return signals

class MAStrategy:
    def __init__(self, short_ma=10, long_ma=50):
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.name = f"均线策略(MA{short_ma}/MA{long_ma})"
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        signals = []
        
        for etf_code in prices_df.columns:
            if len(prices_df[etf_code]) < self.long_ma:
                continue
            
            prices = prices_df[etf_code]
            short_ma_val = prices.rolling(window=self.short_ma).mean().iloc[-1]
            long_ma_val = prices.rolling(window=self.long_ma).mean().iloc[-1]
            current_price = prices.iloc[-1]
            
            if current_holdings and etf_code in current_holdings:
                if short_ma_val < long_ma_val:
                    signals.append({
                        'date': date,
                        'etf_code': etf_code,
                        'action': 'SELL',
                        'price': current_price,
                        'reason': f"死叉信号：短期均线{short_ma_val:.3f} < 长期均线{long_ma_val:.3f}"
                    })
            else:
                if short_ma_val > long_ma_val:
                    signals.append({
                        'date': date,
                        'etf_code': etf_code,
                        'action': 'BUY',
                        'price': current_price,
                        'reason': f"金叉信号：短期均线{short_ma_val:.3f} > 长期均线{long_ma_val:.3f}"
                    })
        
        return signals

class DualMomentumStrategy:
    def __init__(self, lookback_period=20, ma_short=10, ma_long=50, protection_threshold=-5):
        self.momentum = MomentumStrategy(lookback_period, top_n=1)
        self.ma = MAStrategy(ma_short, ma_long)
        self.protection_threshold = protection_threshold
        self.name = f"双动量策略(动量{lookback_period}日+MA{ma_short}/{ma_long})"
    
    def generate_signals(self, prices_df, date, current_holdings=None, benchmark_prices=None):
        signals = []
        
        if benchmark_prices is not None and len(benchmark_prices) >= self.lookback_period:
            bench_return = (benchmark_prices.iloc[-1] - benchmark_prices.iloc[-self.lookback_period]) / benchmark_prices.iloc[-self.lookback_period] * 100
            
            if bench_return < self.protection_threshold:
                for etf in (current_holdings or []):
                    signals.append({
                        'date': date,
                        'etf_code': etf,
                        'action': 'SELL',
                        'price': prices_df[etf].iloc[-1] if etf in prices_df.columns else 0,
                        'reason': f"市场保护机制触发，基准收益率: {bench_return:.2f}%"
                    })
                return signals
        
        momentum_signals = self.momentum.generate_signals(prices_df, date)
        valid_signals = []
        
        for signal in momentum_signals:
            etf_code = signal['etf_code']
            if len(prices_df[etf_code]) >= self.ma.long_ma:
                prices = prices_df[etf_code]
                short_ma = prices.rolling(window=self.ma.short_ma).mean().iloc[-1]
                long_ma = prices.rolling(window=self.ma.long_ma).mean().iloc[-1]
                
                if short_ma > long_ma:
                    signal['reason'] += f" + 均线确认(MA{self.ma.short_ma}>{self.ma.long_ma})"
                    valid_signals.append(signal)
        
        if current_holdings:
            for holding in current_holdings:
                if holding in prices_df.columns and len(prices_df[holding]) >= self.ma.long_ma:
                    prices = prices_df[holding]
                    short_ma = prices.rolling(window=self.ma.short_ma).mean().iloc[-1]
                    long_ma = prices.rolling(window=self.ma.long_ma).mean().iloc[-1]
                    
                    if short_ma < long_ma and not any(s['etf_code'] == holding and s['action'] == 'SELL' for s in signals):
                        signals.append({
                            'date': date,
                            'etf_code': holding,
                            'action': 'SELL',
                            'price': prices.iloc[-1],
                            'reason': f"均线死叉卖出"
                        })
        
        signals.extend(valid_signals)
        return signals


class SmoothMomentumStrategy:
    """平滑动量策略（加权对数回归+R²）
    相比简单动量，能过滤暴涨暴跌的ETF，选出上涨最平稳、动量最强劲的品种
    """
    def __init__(self, lookback_period=25, top_n=1, min_score=-1.0):
        self.lookback_period = lookback_period
        self.top_n = top_n
        self.min_score = min_score
        self.name = f"平滑动量策略(N={lookback_period},前{top_n}只)"
    
    def calculate_smooth_momentum(self, price_series):
        """计算平滑动量得分：年化收益率 × R²"""
        if len(price_series) < self.lookback_period:
            return np.nan
        
        prices = price_series.tail(self.lookback_period).values
        
        # 对数收益
        log_prices = np.log(prices)
        
        # 时间序列（加权，近期权重更高）
        x = np.arange(1, self.lookback_period + 1).reshape(-1, 1)
        weights = np.linspace(1, 2, self.lookback_period)
        
        # 加权线性回归
        lr = LinearRegression()
        lr.fit(x, log_prices, sample_weight=weights)
        
        # 斜率（年化）
        slope = lr.coef_[0]
        annual_return = np.exp(slope * 252) - 1
        
        # R²（拟合优度）
        r_squared = lr.score(x, log_prices, sample_weight=weights)
        
        # 最终得分
        score = annual_return * r_squared
        
        return score
    
    def calculate_scores(self, prices_df):
        """计算所有ETF的平滑动量得分"""
        scores = {}
        for etf_code in prices_df.columns:
            if len(prices_df[etf_code]) >= self.lookback_period:
                score = self.calculate_smooth_momentum(prices_df[etf_code])
                if not np.isnan(score) and score > self.min_score:
                    scores[etf_code] = score
        return scores
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        """生成交易信号"""
        scores = self.calculate_scores(prices_df)
        
        if not scores:
            return []
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [x[0] for x in sorted_scores[:self.top_n]]
        
        signals = []
        
        for etf_code in top_etfs:
            if current_holdings is None or etf_code not in current_holdings:
                signals.append({
                    'date': date,
                    'etf_code': etf_code,
                    'action': 'BUY',
                    'price': prices_df[etf_code].iloc[-1],
                    'reason': f"平滑动量得分 {scores[etf_code]:.4f}，排名前{self.top_n}"
                })
        
        for holding in (current_holdings or []):
            if holding not in top_etfs and holding in scores:
                signals.append({
                    'date': date,
                    'etf_code': holding,
                    'action': 'SELL',
                    'price': prices_df[holding].iloc[-1],
                    'reason': f"平滑动量得分 {scores[holding]:.4f}，已不在前{self.top_n}"
                })
        
        return signals


class ThreeFactorStrategy:
    """三因子轮动策略（动量+趋势+成交量）
    动量因子：过去N日涨幅
    趋势因子：价格与均线的偏离度
    成交量因子：成交量相对均值的倍数
    """
    def __init__(self, momentum_period=20, ma_period=20, volume_period=20, 
                 momentum_weight=0.4, trend_weight=0.3, volume_weight=0.3, top_n=1):
        self.momentum_period = momentum_period
        self.ma_period = ma_period
        self.volume_period = volume_period
        self.momentum_weight = momentum_weight
        self.trend_weight = trend_weight
        self.volume_weight = volume_weight
        self.top_n = top_n
        self.name = f"三因子策略(动量{momentum_period}日+MA{ma_period}日+量{volume_period}日)"
    
    def calculate_factors(self, prices_df, volumes_df=None):
        """计算三因子得分"""
        scores = {}
        
        for etf_code in prices_df.columns:
            if len(prices_df[etf_code]) < max(self.momentum_period, self.ma_period):
                continue
            
            prices = prices_df[etf_code]
            
            # 动量因子：过去N日涨幅
            momentum = (prices.iloc[-1] / prices.iloc[-self.momentum_period] - 1) * 100
            
            # 趋势因子：价格与均线的偏离度
            ma = prices.rolling(window=self.ma_period).mean().iloc[-1]
            current_price = prices.iloc[-1]
            trend = (current_price / ma - 1) * 100 if ma > 0 else 0
            
            # 成交量因子（如果有成交量数据）
            volume_score = 0
            if volumes_df is not None and etf_code in volumes_df.columns:
                volumes = volumes_df[etf_code]
                if len(volumes) >= self.volume_period:
                    avg_volume = volumes.rolling(window=self.volume_period).mean().iloc[-1]
                    current_volume = volumes.iloc[-1]
                    if avg_volume > 0:
                        volume_ratio = current_volume / avg_volume
                        volume_score = min(volume_ratio, 3)  # 限制最大值
            
            # 综合得分
            total_score = (
                momentum * self.momentum_weight +
                trend * self.trend_weight +
                volume_score * self.volume_weight
            )
            
            scores[etf_code] = {
                'total': total_score,
                'momentum': momentum,
                'trend': trend,
                'volume': volume_score
            }
        
        return scores
    
    def generate_signals(self, prices_df, date, current_holdings=None, volumes_df=None):
        """生成交易信号"""
        scores = self.calculate_factors(prices_df, volumes_df)
        
        if not scores:
            return []
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]['total'], reverse=True)
        top_etfs = [x[0] for x in sorted_scores[:self.top_n]]
        
        signals = []
        
        for etf_code in top_etfs:
            if current_holdings is None or etf_code not in current_holdings:
                s = scores[etf_code]
                signals.append({
                    'date': date,
                    'etf_code': etf_code,
                    'action': 'BUY',
                    'price': prices_df[etf_code].iloc[-1],
                    'reason': f"综合得分{s['total']:.2f}(动量{s['momentum']:.1f}%+趋势{s['trend']:.1f}%+量{s['volume']:.1f})"
                })
        
        for holding in (current_holdings or []):
            if holding not in top_etfs and holding in scores:
                signals.append({
                    'date': date,
                    'etf_code': holding,
                    'action': 'SELL',
                    'price': prices_df[holding].iloc[-1],
                    'reason': f"综合得分下降，已不在前{self.top_n}"
                })
        
        return signals


class FixedIncomePlusStrategy:
    """固收+轮动策略
    核心思想：多资产类别轮动，始终保留国债ETF作为"压舱石"
    适合风险厌恶型投资者，追求稳定收益
    """
    def __init__(self, momentum_period=13, ma_period=10, max_holdings=5, 
                 bond_etf='511010', min_bond_ratio=0.3):
        self.momentum_period = momentum_period
        self.ma_period = ma_period
        self.max_holdings = max_holdings
        self.bond_etf = bond_etf
        self.min_bond_ratio = min_bond_ratio
        self.name = f"固收+策略(动量{momentum_period}日+MA{ma_period}日)"
    
    def calculate_momentum_and_trend(self, prices_df):
        """计算动量和趋势状态"""
        results = {}
        
        for etf_code in prices_df.columns:
            if len(prices_df[etf_code]) < max(self.momentum_period, self.ma_period):
                continue
            
            prices = prices_df[etf_code]
            
            # 动量：过去N日涨幅
            momentum = (prices.iloc[-1] / prices.iloc[-self.momentum_period] - 1) * 100
            
            # 趋势：价格是否站上均线
            ma = prices.rolling(window=self.ma_period).mean().iloc[-1]
            above_ma = prices.iloc[-1] > ma
            
            results[etf_code] = {
                'momentum': momentum,
                'above_ma': above_ma,
                'price': prices.iloc[-1]
            }
        
        return results
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        """生成交易信号"""
        results = self.calculate_momentum_and_trend(prices_df)
        
        if not results:
            return []
        
        # 筛选：涨幅>0 且 价格>均线
        valid_etfs = [
            (code, data) for code, data in results.items()
            if data['momentum'] > 0 and data['above_ma']
        ]
        
        # 按动量排序
        valid_etfs.sort(key=lambda x: x[1]['momentum'], reverse=True)
        
        # 选择前N只（确保包含债券ETF）
        selected = []
        has_bond = False
        
        for code, data in valid_etfs:
            if code == self.bond_etf:
                has_bond = True
            if len(selected) < self.max_holdings:
                selected.append(code)
        
        # 如果没有债券ETF且允许，强制加入
        if not has_bond and self.bond_etf in results and len(selected) < self.max_holdings:
            selected.append(self.bond_etf)
        
        signals = []
        
        # 生成买入信号
        for etf_code in selected:
            if current_holdings is None or etf_code not in current_holdings:
                data = results[etf_code]
                signals.append({
                    'date': date,
                    'etf_code': etf_code,
                    'action': 'BUY',
                    'price': data['price'],
                    'reason': f"动量{data['momentum']:.1f}%，趋势向上"
                })
        
        # 生成卖出信号
        for holding in (current_holdings or []):
            if holding not in selected and holding in results:
                signals.append({
                    'date': date,
                    'etf_code': holding,
                    'action': 'SELL',
                    'price': results[holding]['price'],
                    'reason': f"不在优选列表中"
                })
        
        return signals


class DualPoolMomentumStrategy:
    """双池动量策略
    静态池：核心ETF（宽基+行业+跨境）
    动态池：根据成交额动态选择流动性最好的ETF
    融合两个池子，选出动量最强的标的
    """
    def __init__(self, lookback_period=20, ma_short=20, ma_long=60, 
                 top_n=1, static_pool=None, dynamic_pool_size=10):
        self.lookback_period = lookback_period
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.top_n = top_n
        self.static_pool = static_pool or []
        self.dynamic_pool_size = dynamic_pool_size
        self.name = f"双池动量策略(动量{lookback_period}日+MA{ma_short}/{ma_long})"
    
    def calculate_momentum_with_trend_filter(self, prices_df):
        """计算动量并应用趋势过滤"""
        scores = {}
        
        for etf_code in prices_df.columns:
            if len(prices_df[etf_code]) < max(self.lookback_period, self.ma_long):
                continue
            
            prices = prices_df[etf_code]
            
            # 动量
            momentum = (prices.iloc[-1] / prices.iloc[-self.lookback_period] - 1) * 100
            
            # 双均线趋势过滤
            short_ma = prices.rolling(window=self.ma_short).mean().iloc[-1]
            long_ma = prices.rolling(window=self.ma_long).mean().iloc[-1]
            current_price = prices.iloc[-1]
            
            # 只有多头排列才允许参与评分
            if current_price > short_ma > long_ma:
                scores[etf_code] = momentum
        
        return scores
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        """生成交易信号"""
        scores = self.calculate_momentum_with_trend_filter(prices_df)
        
        if not scores:
            return []
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [x[0] for x in sorted_scores[:self.top_n]]
        
        signals = []
        
        for etf_code in top_etfs:
            if current_holdings is None or etf_code not in current_holdings:
                signals.append({
                    'date': date,
                    'etf_code': etf_code,
                    'action': 'BUY',
                    'price': prices_df[etf_code].iloc[-1],
                    'reason': f"双池动量{scores[etf_code]:.2f}%，趋势多头排列"
                })
        
        for holding in (current_holdings or []):
            if holding not in top_etfs and holding in scores:
                signals.append({
                    'date': date,
                    'etf_code': holding,
                    'action': 'SELL',
                    'price': prices_df[holding].iloc[-1],
                    'reason': f"动量排名下降，已不在前{self.top_n}"
                })
        
        return signals


class RSIStrategy:
    """RSI超卖反转策略
    RSI<30买入，RSI>70卖出
    """
    def __init__(self, rsi_period=14, oversold=30, overbought=70, top_n=1):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.top_n = top_n
        self.name = f"RSI策略(RSI{rsi_period},超卖{oversold}/超买{overbought})"
    
    def _calc_rsi(self, prices):
        """计算RSI指标"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        scores = {}
        for code in prices_df.columns:
            if len(prices_df[code]) < self.rsi_period + 1:
                continue
            rsi = self._calc_rsi(prices_df[code])
            current_rsi = rsi.iloc[-1]
            if not np.isnan(current_rsi):
                scores[code] = 100 - current_rsi if current_rsi < 40 else 0
        
        if not scores:
            return []
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [x[0] for x in sorted_scores[:self.top_n]]
        
        signals = []
        for code in top_etfs:
            rsi = self._calc_rsi(prices_df[code]).iloc[-1]
            if current_holdings is None or code not in current_holdings:
                signals.append({
                    'date': date, 'etf_code': code, 'action': 'BUY',
                    'price': prices_df[code].iloc[-1],
                    'reason': f"RSI={rsi:.1f}，超卖区域"
                })
        
        for holding in (current_holdings or []):
            if holding in prices_df.columns:
                rsi = self._calc_rsi(prices_df[holding]).iloc[-1]
                if rsi > self.overbought:
                    signals.append({
                        'date': date, 'etf_code': holding, 'action': 'SELL',
                        'price': prices_df[holding].iloc[-1],
                        'reason': f"RSI={rsi:.1f}，超买区域"
                    })
        return signals


class BollingerBandStrategy:
    """布林带均值回归策略
    价格触及下轨买入，触及上轨卖出
    """
    def __init__(self, bb_period=20, std_mult=2.0, top_n=1):
        self.bb_period = bb_period
        self.std_mult = std_mult
        self.top_n = top_n
        self.name = f"布林带策略(周期{bb_period},{std_mult}σ)"
    
    def _calc_bands(self, prices):
        """计算布林带"""
        ma = prices.rolling(window=self.bb_period).mean()
        std = prices.rolling(window=self.bb_period).std()
        upper = ma + self.std_mult * std
        lower = ma - self.std_mult * std
        return ma, upper, lower
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        scores = {}
        for code in prices_df.columns:
            if len(prices_df[code]) < self.bb_period:
                continue
            ma, upper, lower = self._calc_bands(prices_df[code])
            price = prices_df[code].iloc[-1]
            lower_band = lower.iloc[-1]
            std_val = prices_df[code].rolling(self.bb_period).std().iloc[-1]
            if not np.isnan(lower_band) and price <= lower_band * 1.02:
                scores[code] = (ma.iloc[-1] - price) / std_val if std_val and std_val > 0 else 0
        
        if not scores:
            return []
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [x[0] for x in sorted_scores[:self.top_n]]
        
        signals = []
        for code in top_etfs:
            if current_holdings is None or code not in current_holdings:
                signals.append({
                    'date': date, 'etf_code': code, 'action': 'BUY',
                    'price': prices_df[code].iloc[-1],
                    'reason': f"价格触及布林带下轨"
                })
        
        for holding in (current_holdings or []):
            if holding in prices_df.columns:
                ma, upper, lower = self._calc_bands(prices_df[holding])
                if prices_df[holding].iloc[-1] >= upper.iloc[-1]:
                    signals.append({
                        'date': date, 'etf_code': holding, 'action': 'SELL',
                        'price': prices_df[holding].iloc[-1],
                        'reason': f"价格触及布林带上轨"
                    })
        return signals


class DonchianChannelStrategy:
    """唐奇安通道突破策略
    价格突破N日最高点买入，跌破M日最低点卖出
    """
    def __init__(self, entry_period=20, exit_period=10, top_n=1):
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.top_n = top_n
        self.name = f"唐奇安通道策略(入场{entry_period}日/出场{exit_period}日)"
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        scores = {}
        for code in prices_df.columns:
            if len(prices_df[code]) < self.entry_period + 1:
                continue
            prices = prices_df[code]
            high_n = prices.iloc[-self.entry_period-1:-1].max()
            current = prices.iloc[-1]
            if current > high_n:
                scores[code] = (current / high_n - 1) * 100
        
        if not scores:
            return []
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [x[0] for x in sorted_scores[:self.top_n]]
        
        signals = []
        for code in top_etfs:
            if current_holdings is None or code not in current_holdings:
                signals.append({
                    'date': date, 'etf_code': code, 'action': 'BUY',
                    'price': prices_df[code].iloc[-1],
                    'reason': f"突破{self.entry_period}日高点"
                })
        
        for holding in (current_holdings or []):
            if holding in prices_df.columns and len(prices_df[holding]) > self.exit_period:
                prices = prices_df[holding]
                low_m = prices.iloc[-self.exit_period-1:-1].min()
                if prices.iloc[-1] < low_m:
                    signals.append({
                        'date': date, 'etf_code': holding, 'action': 'SELL',
                        'price': prices.iloc[-1],
                        'reason': f"跌破{self.exit_period}日低点"
                    })
        return signals


class TurtleStrategy(DonchianChannelStrategy):
    """海龟交易法则(唐奇安通道+ATR仓位管理)"""
    def __init__(self, entry_period=20, exit_period=10, atr_period=20):
        super().__init__(entry_period, exit_period)
        self.atr_period = atr_period
        self.name = f"海龟策略(入场{entry_period}/出场{exit_period}/ATR{atr_period})"


class MACDStrategy:
    """MACD金叉死叉策略"""
    def __init__(self, fast=12, slow=26, signal=9, top_n=1):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.top_n = top_n
        self.name = f"MACD策略({fast}/{slow}/{signal})"
    
    def _calc_macd(self, prices):
        ema_fast = prices.ewm(span=self.fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal, adjust=False).mean()
        macd_hist = (dif - dea) * 2
        return dif, dea, macd_hist
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        scores = {}
        for code in prices_df.columns:
            if len(prices_df[code]) < self.slow + self.signal:
                continue
            dif, dea, hist = self._calc_macd(prices_df[code])
            if dif.iloc[-1] > dea.iloc[-1] and hist.iloc[-1] > 0:
                scores[code] = hist.iloc[-1]
        
        if not scores:
            return []
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [x[0] for x in sorted_scores[:self.top_n]]
        
        signals = []
        for code in top_etfs:
            if current_holdings is None or code not in current_holdings:
                signals.append({
                    'date': date, 'etf_code': code, 'action': 'BUY',
                    'price': prices_df[code].iloc[-1],
                    'reason': f"MACD金叉，柱状图为正"
                })
        
        for holding in (current_holdings or []):
            if holding in prices_df.columns:
                dif, dea, hist = self._calc_macd(prices_df[holding])
                if dif.iloc[-1] < dea.iloc[-1]:
                    signals.append({
                        'date': date, 'etf_code': holding, 'action': 'SELL',
                        'price': prices_df[holding].iloc[-1],
                        'reason': f"MACD死叉"
                    })
        return signals


class KDJStrategy:
    """KDJ金叉策略"""
    def __init__(self, rsv_period=9, k_smooth=3, top_n=1):
        self.rsv_period = rsv_period
        self.k_smooth = k_smooth
        self.top_n = top_n
        self.name = f"KDJ策略(RSV{rsv_period})"
    
    def _calc_kdj(self, prices):
        low_n = prices.rolling(window=self.rsv_period).min()
        high_n = prices.rolling(window=self.rsv_period).max()
        rsv = (prices - low_n) / (high_n - low_n) * 100
        k = rsv.ewm(alpha=1/self.k_smooth, adjust=False).mean()
        d = k.ewm(alpha=1/self.k_smooth, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        scores = {}
        for code in prices_df.columns:
            if len(prices_df[code]) < self.rsv_period:
                continue
            k, d, j = self._calc_kdj(prices_df[code])
            if k.iloc[-1] > d.iloc[-1] and j.iloc[-1] < 30:
                scores[code] = 30 - j.iloc[-1]
        
        if not scores:
            return []
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [x[0] for x in sorted_scores[:self.top_n]]
        
        signals = []
        for code in top_etfs:
            if current_holdings is None or code not in current_holdings:
                signals.append({
                    'date': date, 'etf_code': code, 'action': 'BUY',
                    'price': prices_df[code].iloc[-1],
                    'reason': f"KDJ超卖金叉"
                })
        
        for holding in (current_holdings or []):
            if holding in prices_df.columns:
                k, d, j = self._calc_kdj(prices_df[holding])
                if k.iloc[-1] < d.iloc[-1] and j.iloc[-1] > 80:
                    signals.append({
                        'date': date, 'etf_code': holding, 'action': 'SELL',
                        'price': prices_df[holding].iloc[-1],
                        'reason': f"KDJ超买死叉"
                    })
        return signals


class CCIStrategy:
    """CCI顺势指标策略"""
    def __init__(self, cci_period=20, oversold=-100, overbought=100, top_n=1):
        self.cci_period = cci_period
        self.oversold = oversold
        self.overbought = overbought
        self.top_n = top_n
        self.name = f"CCI策略(周期{cci_period},±{abs(oversold)})"
    
    def _calc_cci(self, prices):
        tp = prices
        ma = tp.rolling(window=self.cci_period).mean()
        md = tp.rolling(window=self.cci_period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - ma) / (0.015 * md.replace(0, np.nan))
        return cci
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        scores = {}
        for code in prices_df.columns:
            if len(prices_df[code]) < self.cci_period:
                continue
            cci = self._calc_cci(prices_df[code])
            current_cci = cci.iloc[-1]
            if not np.isnan(current_cci) and current_cci > self.oversold and cci.iloc[-2] <= self.oversold:
                scores[code] = current_cci - self.oversold
        
        if not scores:
            return []
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [x[0] for x in sorted_scores[:self.top_n]]
        
        signals = []
        for code in top_etfs:
            if current_holdings is None or code not in current_holdings:
                signals.append({
                    'date': date, 'etf_code': code, 'action': 'BUY',
                    'price': prices_df[code].iloc[-1],
                    'reason': f"CCI从超卖区回升"
                })
        
        for holding in (current_holdings or []):
            if holding in prices_df.columns:
                cci = self._calc_cci(prices_df[holding])
                if cci.iloc[-1] > self.overbought:
                    signals.append({
                        'date': date, 'etf_code': holding, 'action': 'SELL',
                        'price': prices_df[holding].iloc[-1],
                        'reason': f"CCI进入超买区"
                    })
        return signals


class RiskParityStrategy:
    """风险平价策略
    各ETF风险贡献相等，波动大的配低权重
    """
    def __init__(self, vol_window=20, max_holdings=5, rebalance_days=20):
        self.vol_window = vol_window
        self.max_holdings = max_holdings
        self.rebalance_days = rebalance_days
        self.name = f"风险平价策略(波动{vol_window}日,持仓{max_holdings}只)"
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        vols = {}
        for code in prices_df.columns:
            if len(prices_df[code]) < self.vol_window:
                continue
            ret = prices_df[code].pct_change()
            vol = ret.tail(self.vol_window).std() * np.sqrt(252)
            if not np.isnan(vol) and vol > 0:
                vols[code] = vol
        
        if len(vols) < 2:
            return []
        
        sorted_vols = sorted(vols.items(), key=lambda x: x[1])
        selected = [x[0] for x in sorted_vols[:self.max_holdings]]
        
        signals = []
        for code in selected:
            if current_holdings is None or code not in current_holdings:
                signals.append({
                    'date': date, 'etf_code': code, 'action': 'BUY',
                    'price': prices_df[code].iloc[-1],
                    'reason': f"低波动率{vols[code]:.2%}，风险平价选择"
                })
        
        for holding in (current_holdings or []):
            if holding not in selected:
                signals.append({
                    'date': date, 'etf_code': holding, 'action': 'SELL',
                    'price': prices_df[holding].iloc[-1],
                    'reason': f"不在风险平价选择中"
                })
        return signals


class EqualWeightStrategy:
    """等权配置策略"""
    def __init__(self, pool_size=5, rebalance_days=20):
        self.pool_size = pool_size
        self.rebalance_days = rebalance_days
        self.name = f"等权配置策略(持仓{pool_size}只)"
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        available = [c for c in prices_df.columns if len(prices_df[c]) >= 60]
        selected = sorted(available)[:self.pool_size]
        
        signals = []
        for code in selected:
            if current_holdings is None or code not in current_holdings:
                signals.append({
                    'date': date, 'etf_code': code, 'action': 'BUY',
                    'price': prices_df[code].iloc[-1],
                    'reason': f"等权配置"
                })
        
        for holding in (current_holdings or []):
            if holding not in selected:
                signals.append({
                    'date': date, 'etf_code': holding, 'action': 'SELL',
                    'price': prices_df[holding].iloc[-1],
                    'reason': f"不在等权池中"
                })
        return signals


class ERPStrategy:
    """股债轮动策略(基于股权风险溢价ERP)
    ERP = 1/PE - 无风险利率(10年国债收益率)
    ERP高配股，低配债
    """
    def __init__(self, stock_etf='510300', bond_etf='511010', erp_threshold=0.03):
        self.stock_etf = stock_etf
        self.bond_etf = bond_etf
        self.erp_threshold = erp_threshold
        self.name = f"股债轮动ERP策略(阈值{erp_threshold})"
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        signals = []
        if self.stock_etf not in prices_df.columns or self.bond_etf not in prices_df.columns:
            return signals
        
        stock_prices = prices_df[self.stock_etf]
        bond_prices = prices_df[self.bond_etf]
        
        if len(stock_prices) < 60:
            return signals
        
        stock_mom = (stock_prices.iloc[-1] / stock_prices.iloc[-60] - 1)
        bond_mom = (bond_prices.iloc[-1] / bond_prices.iloc[-60] - 1)
        
        if stock_mom > bond_mom:
            if current_holdings is None or self.stock_etf not in current_holdings:
                signals.append({
                    'date': date, 'etf_code': self.stock_etf, 'action': 'BUY',
                    'price': stock_prices.iloc[-1],
                    'reason': f"股票动量{stock_mom:.2%}>债券{bond_mom:.2%}"
                })
            if current_holdings and self.bond_etf in current_holdings:
                signals.append({
                    'date': date, 'etf_code': self.bond_etf, 'action': 'SELL',
                    'price': bond_prices.iloc[-1],
                    'reason': f"转配股票"
                })
        else:
            if current_holdings is None or self.bond_etf not in current_holdings:
                signals.append({
                    'date': date, 'etf_code': self.bond_etf, 'action': 'BUY',
                    'price': bond_prices.iloc[-1],
                    'reason': f"债券动量{bond_mom:.2%}>股票{stock_mom:.2%}"
                })
            if current_holdings and self.stock_etf in current_holdings:
                signals.append({
                    'date': date, 'etf_code': self.stock_etf, 'action': 'SELL',
                    'price': stock_prices.iloc[-1],
                    'reason': f"转配债券"
                })
        return signals
