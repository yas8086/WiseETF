import akshare as ak
import pandas as pd
import os
from datetime import datetime, timedelta
from config import Config
from models import db, ETFInfo, ETFPrice

class ETFDataFetcher:
    def __init__(self):
        self.data_dir = Config.ETF_DATA_DIR
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def get_etf_list(self):
        try:
            df = ak.fund_etf_spot_em()
            etf_list = []
            for _, row in df.iterrows():
                etf_list.append({
                    'code': row['代码'],
                    'name': row['名称'],
                    'category': 'ETF'
                })
            return etf_list
        except Exception as e:
            print(f"获取ETF列表失败: {e}")
            return []
    
    def get_etf_history(self, etf_code, start_date=None, end_date=None):
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y%m%d')
            
            df = ak.fund_etf_hist_em(
                symbol=etf_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            
            if df is not None and len(df) > 0:
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume'
                })
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                return df.sort_values('date')
            return None
        except Exception as e:
            print(f"获取 {etf_code} 历史数据失败: {e}")
            return None

    def get_liquid_etfs(self, min_amount=5000, min_scale=2, min_list_days=365):
        """
        获取流动性好的ETF列表
        Args:
            min_amount: 最低日成交额(万元)，默认5000万
            min_scale: 最低份额(亿份)，默认2亿
            min_list_days: 最低上市天数，默认365天
        Returns: list[dict] 含code/name/category/amount/scale
        """
        import akshare as ak
        from etf_classifier import classify_etf
        from datetime import datetime, timedelta

        df = ak.fund_etf_spot_em()

        # 成交额过滤(元 -> 万元)
        df = df[df['成交额'] >= min_amount * 10000]
        # 规模过滤
        df = df[df['最新份额'] >= min_scale * 1e8]
        # 排除货币基金(单独处理)
        df = df[~df['名称'].str.contains('货币|日利|现金|添利|快线', na=False)]

        cutoff = (datetime.now() - timedelta(days=min_list_days)).strftime('%Y%m%d')

        result = []
        for _, row in df.iterrows():
            code = row['代码']
            name = row['名称']
            result.append({
                'code': code,
                'name': name,
                'category': classify_etf(name, code),
                'amount': row.get('成交额', 0),
                'scale': row.get('最新份额', 0),
                'total_value': row.get('总市值', 0)
            })
        return result

    def batch_fetch_history(self, codes, start_date='20050201', end_date=None,
                            max_workers=3, use_cache=True, progress_callback=None):
        """
        多线程批量获取ETF历史数据
        Args:
            codes: ETF代码列表
            start_date: 起始日期 YYYYMMDD
            end_date: 结束日期，默认今天
            max_workers: 最大并发数(建议3-5，防封IP)
            use_cache: 是否使用本地缓存
            progress_callback: 进度回调函数(done, total, code)
        Returns: (dict[str, pd.DataFrame], list) 成功的数据和失败的代码列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from data_cache import DataCache
        import time

        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        cache = DataCache() if use_cache else None
        results = {}
        failed = []

        def fetch_one(code):
            if cache:
                cached = cache.load(code)
                if cached is not None:
                    last_date = cached['date'].iloc[-1].strftime('%Y%m%d')
                    if last_date >= end_date:
                        return code, cached
                    try:
                        new_df = self.get_etf_history(code, last_date, end_date)
                        if new_df is not None and len(new_df) > 0:
                            combined = cache.incremental_update(code, new_df)
                            return code, combined
                        return code, cached
                    except:
                        return code, cached

            try:
                df = self.get_etf_history(code, start_date, end_date)
                if df is not None and len(df) > 0:
                    if cache:
                        cache.save(code, df)
                    return code, df
                return code, None
            except Exception as e:
                return code, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one, code): code for code in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    code, df = future.result()
                    if df is not None:
                        results[code] = df
                    else:
                        failed.append(code)
                except Exception as e:
                    failed.append(code)

                if progress_callback:
                    progress_callback(len(results) + len(failed), len(codes), code)

        return results, failed

    def save_etf_data_to_db(self, etf_code, df):
        try:
            for _, row in df.iterrows():
                existing = ETFPrice.query.filter_by(
                    etf_code=etf_code,
                    date=row['date'].date()
                ).first()
                
                if not existing:
                    price = ETFPrice(
                        etf_code=etf_code,
                        date=row['date'].date(),
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume']
                    )
                    db.session.add(price)
            
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            print(f"保存数据失败: {e}")
            return False
    
    def get_etf_data_from_db(self, etf_code, start_date=None, end_date=None):
        query = ETFPrice.query.filter_by(etf_code=etf_code)
        
        if start_date:
            query = query.filter(ETFPrice.date >= start_date)
        if end_date:
            query = query.filter(ETFPrice.date <= end_date)
        
        results = query.order_by(ETFPrice.date).all()
        
        if results:
            data = [{
                'date': r.date,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume
            } for r in results]
            return pd.DataFrame(data)
        return None
    
    def update_all_etfs(self, etf_codes=None):
        if etf_codes is None:
            etf_codes = Config.DEFAULT_ETFS
        
        results = {}
        for code in etf_codes:
            print(f"正在更新 {code} 数据...")
            df = self.get_etf_history(code)
            if df is not None:
                self.save_etf_data_to_db(code, df)
                results[code] = len(df)
            else:
                results[code] = 0
        
        return results
