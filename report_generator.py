"""静态HTML报告生成器"""
import os
from datetime import datetime
import pandas as pd
import numpy as np


class ReportGenerator:
    """生成ETF轮动策略每日报告(静态HTML)"""
    
    def __init__(self, output_dir='reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_daily_report(self, signals, current_prices, strategy_results=None,
                               etf_list=None, best_strategy_name=None):
        """
        生成每日报告
        Args:
            signals: 今日交易信号列表
            current_prices: 当前价格 dict
            strategy_results: 策略回测结果列表
            etf_list: ETF列表
            best_strategy_name: 最优策略名称
        Returns: HTML文件路径
        """
        today = datetime.now().strftime('%Y-%m-%d')
        filename = f"daily_report_{today}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        sell_signals = [s for s in signals if s['action'] == 'SELL']
        
        html = self._build_html(
            today, buy_signals, sell_signals, current_prices,
            strategy_results, etf_list, best_strategy_name
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        index_path = os.path.join(self.output_dir, 'index.html')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return filepath
    
    def _build_html(self, date, buy_signals, sell_signals, prices, 
                    results, etf_list, best_strategy):
        buy_rows = self._signal_rows(buy_signals, prices)
        sell_rows = self._signal_rows(sell_signals, prices)
        strategy_table = self._strategy_table(results) if results else ''
        etf_table = self._etf_table(etf_list, prices) if etf_list else ''
        
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF轮动策略报告 - {date}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; 
               background: #f5f5f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }}
        .card .value {{ font-size: 28px; font-weight: bold; }}
        .card.buy .value {{ color: #e94560; }}
        .card.sell .value {{ color: #0f3460; }}
        .card.total .value {{ color: #16213e; }}
        .card.strategy .value {{ font-size: 16px; color: #e94560; }}
        table {{ width: 100%; border-collapse: collapse; background: white; 
                border-radius: 8px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        th {{ background: #16213e; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}
        .buy-tag {{ background: #e94560; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
        .sell-tag {{ background: #0f3460; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
        .positive {{ color: #e94560; font-weight: bold; }}
        .negative {{ color: #0f3460; font-weight: bold; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>ETF轮动策略每日报告</h1>
    <p>报告日期: {date}</p>
    
    <div class="summary">
        <div class="card buy">
            <div>买入信号</div>
            <div class="value">{len(buy_signals)}</div>
        </div>
        <div class="card sell">
            <div>卖出信号</div>
            <div class="value">{len(sell_signals)}</div>
        </div>
        <div class="card total">
            <div>监控ETF数</div>
            <div class="value">{len(prices) if prices else 0}</div>
        </div>
        <div class="card strategy">
            <div>当前策略</div>
            <div class="value">{best_strategy or 'N/A'}</div>
        </div>
    </div>
    
    <h2>买入建议</h2>
    <table>
        <thead><tr><th>ETF代码</th><th>ETF名称</th><th>当前价</th><th>信号</th><th>原因</th></tr></thead>
        <tbody>{buy_rows if buy_rows else '<tr><td colspan="5" style="text-align:center;color:#999;">暂无买入信号</td></tr>'}</tbody>
    </table>
    
    <h2>卖出建议</h2>
    <table>
        <thead><tr><th>ETF代码</th><th>ETF名称</th><th>当前价</th><th>信号</th><th>原因</th></tr></thead>
        <tbody>{sell_rows if sell_rows else '<tr><td colspan="5" style="text-align:center;color:#999;">暂无卖出信号</td></tr>'}</tbody>
    </table>
    
    {strategy_table}
    {etf_table}
    
    <div class="footer">
        <p>WiseETF轮动策略系统 | 数据来源: akshare(东方财富) | 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>本报告仅供学习研究，不构成投资建议。投资有风险，入市需谨慎。</p>
    </div>
</div>
</body>
</html>"""
    
    def _signal_rows(self, signals, prices):
        from config import Config
        rows = ''
        for s in signals:
            code = s.get('etf_code', '')
            name = Config.ETF_NAMES.get(code, code)
            price = s.get('price', 0)
            action = s.get('action', '')
            reason = s.get('reason', '')
            tag_class = 'buy-tag' if action == 'BUY' else 'sell-tag'
            rows += f"""<tr>
                <td>{code}</td><td>{name}</td><td>{price:.3f}</td>
                <td><span class="{tag_class}">{action}</span></td><td>{reason}</td>
            </tr>"""
        return rows
    
    def _strategy_table(self, results):
        if not results:
            return ''
        
        rows = ''
        for i, r in enumerate(results[:10]):
            name = r.get('strategy_name', '')
            annual = r.get('annual_return', 0)
            max_dd = r.get('max_drawdown', 0)
            sharpe = r.get('sharpe_ratio', 0)
            calmar = r.get('calmar_ratio', 0)
            ret_class = 'positive' if annual >= 0 else 'negative'
            rows += f"""<tr>
                <td>{i+1}</td><td>{name}</td>
                <td class="{ret_class}">{annual:+.2f}%</td>
                <td>{max_dd:.2f}%</td><td>{sharpe:.3f}</td><td>{calmar:.3f}</td>
            </tr>"""
        
        return f"""
    <h2>策略回测对比(Top 10)</h2>
    <table>
        <thead><tr><th>排名</th><th>策略名称</th><th>年化收益</th><th>最大回撤</th><th>夏普比率</th><th>Calmar</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>"""
    
    def _etf_table(self, etf_list, prices):
        if not etf_list:
            return ''
        
        from config import Config
        rows = ''
        for etf in etf_list[:50]:
            code = etf.get('code', '')
            name = etf.get('name', Config.ETF_NAMES.get(code, code))
            category = etf.get('category', '')
            price = prices.get(code, 0) if prices else 0
            rows += f"""<tr>
                <td>{code}</td><td>{name}</td><td>{category}</td><td>{price:.3f}</td>
            </tr>"""
        
        return f"""
    <h2>ETF池(前50只)</h2>
    <table>
        <thead><tr><th>代码</th><th>名称</th><th>分类</th><th>当前价</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>"""
