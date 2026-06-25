from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class ETFInfo(db.Model):
    __tablename__ = 'etf_info'
    
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(10), unique=True, nullable=False)
    name = db.Column(db.String(50), nullable=False)
    category = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    price_data = db.relationship('ETFPrice', backref='etf', lazy=True)

class ETFPrice(db.Model):
    __tablename__ = 'etf_price'
    
    id = db.Column(db.Integer, primary_key=True)
    etf_code = db.Column(db.String(10), db.ForeignKey('etf_info.code'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.Float)
    
    __table_args__ = (db.UniqueConstraint('etf_code', 'date', name='unique_etf_date'),)

class StrategyResult(db.Model):
    __tablename__ = 'strategy_result'
    
    id = db.Column(db.Integer, primary_key=True)
    strategy_name = db.Column(db.String(50), nullable=False)
    params = db.Column(db.JSON)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    
    total_return = db.Column(db.Float)
    annual_return = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    win_rate = db.Column(db.Float)
    trade_count = db.Column(db.Integer)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    signals = db.relationship('TradeSignal', backref='strategy', lazy=True)

class TradeSignal(db.Model):
    __tablename__ = 'trade_signal'
    
    id = db.Column(db.Integer, primary_key=True)
    strategy_result_id = db.Column(db.Integer, db.ForeignKey('strategy_result.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    etf_code = db.Column(db.String(10), nullable=False)
    action = db.Column(db.String(10))  # 'BUY' or 'SELL'
    price = db.Column(db.Float)
    reason = db.Column(db.String(200))
