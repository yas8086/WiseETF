"""机器学习ETF轮动策略"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')


class MLRotationStrategy:
    """机器学习轮动策略(随机森林/XGBoost)
    用技术指标做特征，预测下一期涨跌方向，做多预测为涨的ETF
    """
    def __init__(self, model_type='rf', lookback=20, prediction_period=5, 
                 retrain_days=60, top_n=1, min_train_samples=500):
        self.model_type = model_type
        self.lookback = lookback
        self.prediction_period = prediction_period
        self.retrain_days = retrain_days
        self.top_n = top_n
        self.min_train_samples = min_train_samples
        self.models = {}
        self.last_train_date = {}
        self.name = f"ML策略({model_type.upper()},回看{lookback}日,预测{prediction_period}日)"
    
    def _build_features(self, prices):
        """构建技术指标特征"""
        df = pd.DataFrame(index=prices.index)
        df['price'] = prices
        
        # 动量特征
        for p in [5, 10, 20, 60]:
            df[f'mom_{p}'] = prices.pct_change(p)
        
        # 均线特征
        for p in [5, 10, 20, 60]:
            ma = prices.rolling(p).mean()
            df[f'ma_ratio_{p}'] = prices / ma - 1
        
        # 波动率特征
        for p in [10, 20]:
            df[f'vol_{p}'] = prices.pct_change().rolling(p).std()
        
        # RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        return df.dropna()
    
    def _create_model(self):
        """创建模型"""
        if self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=50,
                random_state=42, n_jobs=-1
            )
        else:  # xgboost
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.1,
                    random_state=42, n_jobs=-1, use_label_encoder=False,
                    eval_metric='logloss'
                )
            except ImportError:
                return GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                )
    
    def _train_one(self, code, prices):
        """训练单个ETF的模型"""
        features = self._build_features(prices)
        
        future_ret = prices.pct_change(self.prediction_period).shift(-self.prediction_period)
        labels = (future_ret > 0).astype(int)
        
        valid = features.index.intersection(labels.dropna().index)
        if len(valid) < self.min_train_samples:
            return False
        
        X = features.loc[valid]
        y = labels.loc[valid]
        
        model = self._create_model()
        model.fit(X, y)
        self.models[code] = model
        return True
    
    def generate_signals(self, prices_df, date, current_holdings=None):
        signals = []
        predictions = {}
        
        for code in prices_df.columns:
            if len(prices_df[code]) < self.min_train_samples:
                continue
            
            need_train = (code not in self.models or 
                         code not in self.last_train_date or
                         (date - self.last_train_date[code]).days >= self.retrain_days)
            
            if need_train:
                success = self._train_one(code, prices_df[code].loc[:date])
                if success:
                    self.last_train_date[code] = date
            
            if code in self.models:
                features = self._build_features(prices_df[code].loc[:date])
                if len(features) > 0:
                    latest = features.iloc[[-1]]
                    prob = self.models[code].predict_proba(latest)[0]
                    up_prob = prob[1] if len(prob) > 1 else prob[0]
                    if up_prob > 0.55:
                        predictions[code] = up_prob
        
        if not predictions:
            return []
        
        sorted_pred = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [x[0] for x in sorted_pred[:self.top_n]]
        
        for code in top_etfs:
            if current_holdings is None or code not in current_holdings:
                signals.append({
                    'date': date, 'etf_code': code, 'action': 'BUY',
                    'price': prices_df[code].iloc[-1],
                    'reason': f"ML预测上涨概率{predictions[code]:.1%}"
                })
        
        for holding in (current_holdings or []):
            if holding not in top_etfs and holding in predictions:
                signals.append({
                    'date': date, 'etf_code': holding, 'action': 'SELL',
                    'price': prices_df[holding].iloc[-1],
                    'reason': f"ML预测上涨概率下降"
                })
        return signals
