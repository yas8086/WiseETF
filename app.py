from flask import Flask, render_template, request, jsonify
from config import config
from models import db, ETFInfo, ETFPrice, StrategyResult, TradeSignal
from data_fetcher import ETFDataFetcher
from strategies import MomentumStrategy, MAStrategy, DualMomentumStrategy, TrendScoreStrategy
from backtest import BacktestEngine, optimize_strategy
from password_gen import verify_password, get_today_password
from report_service import ReportGenerator
import pandas as pd

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/normal')
    def normal():
        pw = request.args.get('pw', '')
        if not verify_password(pw):
            return "访问被拒绝（密码错误或已过期）", 403
        
        report_gen = ReportGenerator()
        report_data = report_gen.generate_report_data()
        
        return render_template('report.html', **report_data)
    
    @app.route('/api/password/today')
    def get_password():
        return jsonify({
            'success': True,
            'password': get_today_password(),
            'date': pd.Timestamp.now().strftime('%Y-%m-%d')
        })
    
    @app.route('/api/etf/list')
    def get_etf_list():
        fetcher = ETFDataFetcher()
        etf_list = fetcher.get_etf_list()
        return jsonify(etf_list)
    
    @app.route('/api/etf/update', methods=['POST'])
    def update_etf_data():
        data = request.get_json()
        etf_codes = data.get('etf_codes', None)
        
        fetcher = ETFDataFetcher()
        results = fetcher.update_all_etfs(etf_codes)
        
        return jsonify({
            'success': True,
            'message': f"成功更新 {len([k for k, v in results.items() if v > 0])} 个ETF数据",
            'details': results
        })
    
    @app.route('/api/etf/<code>/history')
    def get_etf_history(code):
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        fetcher = ETFDataFetcher()
        df = fetcher.get_etf_data_from_db(code, start_date, end_date)
        
        if df is not None:
            return jsonify({
                'success': True,
                'data': df.to_dict('records'),
                'count': len(df)
            })
        else:
            return jsonify({'success': False, 'message': '未找到数据'})
    
    @app.route('/api/backtest/run', methods=['POST'])
    def run_backtest():
        data = request.get_json()
        etf_codes = data.get('etf_codes', ['510300', '510500', '159915'])
        strategy_type = data.get('strategy_type', 'momentum')
        params = data.get('params', {})
        start_date = data.get('start_date', '2014-01-01')
        end_date = data.get('end_date', pd.Timestamp.now().strftime('%Y-%m-%d'))
        
        fetcher = ETFDataFetcher()
        prices_dict = {}
        for code in etf_codes:
            df = fetcher.get_etf_data_from_db(code, start_date, end_date)
            if df is not None:
                prices_dict[code] = df.set_index('date')['close']
        
        if not prices_dict:
            return jsonify({'success': False, 'message': '没有可用的ETF数据，请先更新数据'})
        
        if strategy_type == 'momentum':
            strategy = MomentumStrategy(
                lookback_period=params.get('lookback_period', 20),
                top_n=params.get('top_n', 1)
            )
        elif strategy_type == 'ma':
            strategy = MAStrategy(
                short_ma=params.get('short_ma', 10),
                long_ma=params.get('long_ma', 50)
            )
        elif strategy_type == 'dual_momentum':
            strategy = DualMomentumStrategy(
                lookback_period=params.get('lookback_period', 20),
                ma_short=params.get('ma_short', 10),
                ma_long=params.get('ma_long', 50)
            )
        else:
            return jsonify({'success': False, 'message': '未知的策略类型'})
        
        engine = BacktestEngine(initial_capital=data.get('initial_capital', 100000))
        result = engine.run_backtest(strategy, prices_dict, start_date, end_date, benchmark_code=etf_codes[0])
        
        if result:
            equity_df = pd.DataFrame(result['equity_curve'])
            
            result_obj = StrategyResult(
                strategy_name=result['strategy_name'],
                params=params,
                start_date=start_date,
                end_date=end_date,
                total_return=result['total_return'],
                annual_return=result['annual_return'],
                max_drawdown=result['max_drawdown'],
                sharpe_ratio=result['sharpe_ratio'],
                win_rate=result['win_rate'],
                trade_count=result['trade_count']
            )
            db.session.add(result_obj)
            db.session.flush()
            
            for signal in result['signals']:
                signal_obj = TradeSignal(
                    strategy_result_id=result_obj.id,
                    date=signal['date'] if isinstance(signal['date'], str) else signal['date'].strftime('%Y-%m-%d'),
                    etf_code=signal['etf_code'],
                    action=signal['action'],
                    price=signal['price'],
                    reason=signal['reason']
                )
                db.session.add(signal_obj)
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'result': {
                    'id': result_obj.id,
                    'strategy_name': result['strategy_name'],
                    'total_return': result['total_return'],
                    'annual_return': result['annual_return'],
                    'max_drawdown': result['max_drawdown'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'win_rate': result['win_rate'],
                    'trade_count': result['trade_count'],
                    'final_value': result['final_value'],
                    'equity_curve': equity_df.to_dict('records'),
                    'signals': result['signals'],
                    'trades': result['trades']
                }
            })
        else:
            return jsonify({'success': False, 'message': '回测失败'})
    
    @app.route('/api/optimize', methods=['POST'])
    def optimize_strategies():
        data = request.get_json()
        etf_codes = data.get('etf_codes', ['510300', '510500', '159915'])
        start_date = data.get('start_date', '2014-01-01')
        end_date = data.get('end_date', pd.Timestamp.now().strftime('%Y-%m-%d'))
        
        fetcher = ETFDataFetcher()
        prices_dict = {}
        for code in etf_codes:
            df = fetcher.get_etf_data_from_db(code, start_date, end_date)
            if df is not None:
                prices_dict[code] = df.set_index('date')['close']
        
        if not prices_dict:
            return jsonify({'success': False, 'message': '没有可用的ETF数据'})
        
        results = optimize_strategy(etf_codes, prices_dict, start_date, end_date)
        
        return jsonify({
            'success': True,
            'results': results[:10],
            'best_strategy': results[0] if results else None
        })
    
    @app.route('/api/strategies/history')
    def get_strategy_history():
        strategies = StrategyResult.query.order_by(StrategyResult.created_at.desc()).limit(20).all()
        
        result_list = []
        for s in strategies:
            result_list.append({
                'id': s.id,
                'strategy_name': s.strategy_name,
                'params': s.params,
                'total_return': s.total_return,
                'annual_return': s.annual_return,
                'max_drawdown': s.max_drawdown,
                'sharpe_ratio': s.sharpe_ratio,
                'created_at': s.created_at.strftime('%Y-%m-%d %H:%M:%S') if s.created_at else None
            })
        
        return jsonify({'success': True, 'strategies': result_list})
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8951, debug=True)
