# WiseETF - A股ETF轮动策略系统

自动化ETF轮动策略回测与每日信号生成系统。

## 回测结果（2020-2026真实数据）

| 排名 | 策略 | 年化收益 | 最大回撤 | 夏普比率 |
|---|---|---|---|---|
| 1 | **动量(10日)** | **+16.7%** | -35.5% | 0.659 |
| 2 | 平滑动量(25日) | +8.7% | -38.3% | 0.444 |
| 3 | 双均线(10/50) | +6.3% | -51.7% | 0.367 |
| 4 | 风险平价(20日) | +4.2% | **-16.9%** | **0.672** |

> 初始资金10万，动量(10日)策略6年增长至26.6万

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 每日运行（生成HTML报告）

```bash
# 默认: 动量(10日)策略 - 回测最优
python daily_runner.py

# 其他策略
python daily_runner.py --strategy smooth   # 平滑动量(25日)
python daily_runner.py --strategy risk       # 风险平价(低回撤)
```

### 全量回测（14策略对比）

```bash
# 下载真实数据并回测
python fast_backtest.py
```

### 策略优化（Optuna参数搜索）

```python
from strategy_optimizer import StrategyOptimizer
from strategies import MomentumStrategy

opt = StrategyOptimizer(prices_dict)
result = opt.optimize(MomentumStrategy, lambda t: {
    'lookback_period': t.suggest_int('lookback_period', 5, 60),
    'top_n': t.suggest_int('top_n', 1, 3),
}, n_trials=100)
print(result['best_params'])
```

## 策略列表（28种）

| 类别 | 策略 |
|---|---|
| 动量类 | 简单动量(10/20/60/120日)、平滑动量(25/60日)、趋势得分 |
| 均线类 | 双均线(5/20, 10/50, 10/100) |
| 趋势类 | 唐奇安通道、海龟交易、MACD |
| 多因子类 | 双动量、三因子、双池动量 |
| 均值回归 | RSI超卖、布林带、KDJ |
| 资产配置 | 风险平价、等权、固收+、股债轮动ERP |
| 技术指标 | CCI、MACD、KDJ |
| 机器学习 | 随机森林、XGBoost |

## ETF池（15只）

| 分类 | ETF |
|---|---|
| 宽基 | 沪深300、中证500、创业板、科创50 |
| 行业 | 证券、医药、新能源、半导体 |
| 跨境 | 纳指、标普500、日经 |
| 商品 | 黄金 |
| 债券 | 国债 |
| 货币 | 银华日利 |

## GitHub Actions自动运行

每日北京时间15:30（收盘后）自动运行：
1. 下载最新ETF数据
2. 运行动量(10日)策略生成买卖信号
3. 生成HTML报告并推送到gh-pages分支

开启GitHub Pages后即可在浏览器查看每日报告。

## 目录结构

```
WiseETF/
├── daily_runner.py        # 每日运行脚本（最优策略）
├── fast_backtest.py       # 向量化快速回测（14策略对比）
├── strategies.py          # 26种传统策略
├── strategies_ml.py       # 机器学习策略(随机森林/XGBoost)
├── strategy_optimizer.py  # Optuna参数优化+Walk-Forward
├── backtest.py            # 回测引擎
├── rotation_backtest.py   # 多策略对比回测
├── report_generator.py    # HTML报告生成器
├── data_fetcher.py        # 数据获取(多线程+缓存)
├── data_cache.py          # 本地缓存管理
├── etf_classifier.py      # ETF分类器
├── config.py              # 配置
├── .github/workflows/     # GitHub Actions
└── reports/               # 生成的HTML报告
```

## 数据来源

- akshare（东方财富，免费无需注册）
- ETF数据从2005年开始（上证50ETF首只）

## 风险提示

本系统仅供学习研究，不构成投资建议。历史回测不代表未来收益。投资有风险，入市需谨慎。
