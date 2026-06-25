"""ETF数据本地缓存管理"""
import os
import pandas as pd
from datetime import datetime

class DataCache:
    def __init__(self, cache_dir='etf_data_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _cache_path(self, code):
        return os.path.join(self.cache_dir, f"{code}.parquet")
    
    def load(self, code):
        """加载缓存的ETF数据"""
        path = self._cache_path(code)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        return None
    
    def save(self, code, df):
        """保存ETF数据到缓存"""
        df.to_parquet(self._cache_path(code), index=False)
    
    def get_last_date(self, code):
        """获取缓存数据的最后日期"""
        df = self.load(code)
        if df is not None and len(df) > 0:
            return df['date'].iloc[-1]
        return None
    
    def incremental_update(self, code, new_df):
        """增量更新：合并新数据"""
        cached = self.load(code)
        if cached is not None:
            combined = pd.concat([cached, new_df]).drop_duplicates('date').sort_values('date')
        else:
            combined = new_df
        self.save(code, combined)
        return combined
