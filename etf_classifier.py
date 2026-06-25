"""ETF分类器：根据名称和代码自动分类"""
import re

def classify_etf(name, code):
    """
    根据ETF名称和代码规则分类
    返回: 宽基/行业/跨境/商品/债券/货币/其他
    """
    # 货币ETF
    if re.search(r'货币|日利|现金|添利|快线|保证金', name):
        return '货币'
    
    # 债券ETF
    if code.startswith('511') or re.search(r'国债|信用|企债|转债|债券|利率', name):
        return '债券'
    
    # 商品ETF
    if re.search(r'黄金|白银|豆粕|原油|有色|商品', name):
        return '商品'
    
    # 跨境ETF
    if code.startswith('513') or re.search(r'纳指|标普|日经|德国|法国|恒生|中概|海外|美国|越南|印度', name):
        return '跨境'
    
    # 行业/主题ETF
    industry_keywords = r'证券|银行|保险|医药|医疗|生物|消费|食品|白酒|军工|国防|半导体|芯片|新能源|光伏|锂电|稀土|煤炭|钢铁|有色|地产|基建|建材|传媒|游戏|环保|电力|交运|汽车|机械|电子|通信|计算机|软件|人工智能|机器人|家电|农业|养殖|化工|石油|石化|券商|非银|金融|科技|创新'
    if re.search(industry_keywords, name):
        return '行业'
    
    # 宽基ETF
    if re.search(r'300|500|50|1000|创业板|科创|沪深|中证|上证|深证|国证|MSCI|A50', name):
        return '宽基'
    
    return '其他'


def get_category_pool(all_etfs, category):
    """筛选指定分类的ETF"""
    return [e for e in all_etfs if classify_etf(e.get('name',''), e.get('code','')) == category]
