import os
from datetime import datetime

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'wise-etf-secret-key-2024'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///wise_etf.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    ETF_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # 扩充的ETF候选池（宽基+行业+跨境+商品+货币）
    DEFAULT_ETFS = [
        # 宽基指数
        '510300',  # 沪深300ETF
        '510500',  # 中证500ETF
        '159915',  # 创业板ETF
        '588000',  # 科创50ETF
        # 风格/红利
        '510880',  # 红利ETF
        # 行业
        '512880',  # 证券ETF
        '512010',  # 医药ETF
        '515030',  # 新能源ETF
        '512480',  # 半导体ETF
        # 跨境
        '513100',  # 纳指ETF
        '513500',  # 标普500ETF
        '513880',  # 日经ETF
        # 商品/避险
        '518880',  # 黄金ETF
        '511010',  # 国债ETF
        '511880',  # 银华日利（货币ETF）
    ]
    
    # ETF名称映射
    ETF_NAMES = {
        # 宽基指数
        '510300': '沪深300ETF',
        '510500': '中证500ETF',
        '159915': '创业板ETF',
        '588000': '科创50ETF',
        # 风格/红利
        '510880': '红利ETF',
        # 行业
        '512880': '证券ETF',
        '512010': '医药ETF',
        '515030': '新能源ETF',
        '512480': '半导体ETF',
        # 跨境
        '513100': '纳指ETF',
        '513500': '标普500ETF',
        '513880': '日经ETF',
        # 商品/避险
        '518880': '黄金ETF',
        '511010': '国债ETF',
        '511880': '银华日利',
    }
    
    BACKTEST_START_DATE = '2015-01-01'
    BACKTEST_END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # 趋势得分策略参数（基于RSRS思想）
    STRATEGY_PARAMS = {
        'momentum': {
            'lookback_period': [10, 20, 60],
            'top_n': [1, 2]
        },
        'trend_score': {
            'trend_period': [18, 25, 30],
            'top_n': [1]
        }
    }

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
