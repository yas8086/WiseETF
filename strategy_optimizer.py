"""策略优化模块：Optuna参数优化 + Walk-Forward分析"""
import numpy as np
import pandas as pd
from backtest import BacktestEngine
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class StrategyOptimizer:
    """策略参数优化器"""
    
    def __init__(self, prices_dict, initial_capital=100000):
        self.prices_dict = prices_dict
        self.engine = BacktestEngine(initial_capital=initial_capital)
    
    def optimize(self, strategy_class, param_space, n_trials=100, 
                metric='sharpe_ratio', train_end=None):
        """
        Optuna贝叶斯优化
        Args:
            strategy_class: 策略类
            param_space: 参数空间函数，接收trial返回参数dict
            n_trials: 试验次数
            metric: 优化指标(sharpe_ratio/annual_return/calmar_ratio)
            train_end: 训练集截止日期(用于样本外验证)
        Returns: dict {best_params, best_score, study}
        """
        def objective(trial):
            params = param_space(trial)
            strategy = strategy_class(**params)
            
            result = self.engine.run_backtest(strategy, self.prices_dict, end_date=train_end)
            if result and result.get(metric) is not None:
                return result[metric]
            return 0
        
        study = optuna.create_study(direction='maximize', 
                                     sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def walk_forward_test(self, strategy_class, best_params, prices_df,
                          train_years=3, test_months=6, step_months=6):
        """
        Walk-Forward滚动样本外测试
        Returns: dict {oos_sharpe, oos_returns, is_sharpe, degradation_ratio}
        """
        all_dates = prices_df.index
        if len(all_dates) < 252 * 4:
            return None
        
        oos_returns = []
        is_sharpes = []
        
        start = all_dates[0]
        while start < all_dates[-1]:
            train_end = start + pd.DateOffset(years=train_years)
            test_end = train_end + pd.DateOffset(months=test_months)
            
            if test_end > all_dates[-1]:
                test_end = all_dates[-1]
            
            train_data = {k: v.loc[start:train_end] for k, v in self.prices_dict.items() 
                         if isinstance(v, pd.Series)}
            train_data = {k: v for k, v in train_data.items() if len(v) > 60}
            
            if len(train_data) < 2:
                break
            
            test_data = {k: v.loc[train_end:test_end] for k, v in self.prices_dict.items()
                        if isinstance(v, pd.Series)}
            test_data = {k: v for k, v in test_data.items() if len(v) > 10}
            
            if len(test_data) < 2:
                break
            
            strategy = strategy_class(**best_params)
            is_result = self.engine.run_backtest(strategy, train_data)
            if is_result:
                is_sharpes.append(is_result.get('sharpe_ratio', 0))
            
            strategy = strategy_class(**best_params)
            oos_result = self.engine.run_backtest(strategy, test_data)
            if oos_result:
                oos_returns.append(oos_result.get('annual_return', 0))
            
            start += pd.DateOffset(months=step_months)
        
        is_sharpe = np.mean(is_sharpes) if is_sharpes else 0
        oos_sharpe = np.mean(oos_returns) if oos_returns else 0
        degradation = oos_sharpe / is_sharpe if is_sharpe != 0 else 0
        
        return {
            'is_sharpe': round(is_sharpe, 3),
            'oos_sharpe': round(oos_sharpe, 3),
            'degradation_ratio': round(degradation, 3),
            'oos_returns': oos_returns,
            'n_periods': len(oos_returns)
        }
    
    def monte_carlo_test(self, strategy, n_sim=200):
        """
        蒙特卡洛检验：打乱信号检验策略是否显著优于随机
        Returns: dict {p_value, real_sharpe, random_sharpes}
        """
        real_result = self.engine.run_backtest(strategy, self.prices_dict)
        if not real_result:
            return None
        
        real_sharpe = real_result.get('sharpe_ratio', 0)
        random_sharpes = []
        
        prices_df = pd.DataFrame(self.prices_dict)
        
        for _ in range(n_sim):
            np.random.seed(None)
            n_etfs = len(prices_df.columns)
            if n_etfs < 2:
                break
            random_code = np.random.choice(prices_df.columns)
            random_prices = {random_code: prices_df[random_code]}
            random_result = self.engine.run_backtest(
                type(strategy)(), random_prices
            )
            if random_result:
                random_sharpes.append(random_result.get('sharpe_ratio', 0))
        
        if not random_sharpes:
            return None
        
        from scipy.stats import percentileofscore
        p_value = 1 - percentileofscore(random_sharpes, real_sharpe) / 100
        
        return {
            'p_value': round(p_value, 4),
            'real_sharpe': round(real_sharpe, 3),
            'random_sharpe_mean': round(np.mean(random_sharpes), 3),
            'random_sharpe_std': round(np.std(random_sharpes), 3),
            'significant': p_value < 0.05
        }
