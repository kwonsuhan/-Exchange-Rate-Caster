import pandas as pd

# 기준이 되는 날짜 데이터(임의로 생성)
def load_data():
    date = pd.read_csv('../data/numeric_raw_data/date.csv', header = 0)
    date.tail()
    
    won_mea = pd.read_csv('../data/numeric_raw_data/won_mea.csv', header = 0, encoding='CP949')
    won_mea.columns = ['DATE', 'USD_KRW', 'JPY_KRW', 'EUR_KRW', 'GBP_KRW']
    won_mea.tail()
    
    d = pd.read_csv('../data/numeric_raw_data/dollar.csv', header = 0, encoding='CP949')
    d.columns = ['DATE', 'temp', 'EUR_USD', 'GBP_USD', 'JPY_USD']
    d = d[['DATE', 'EUR_USD', 'GBP_USD', 'JPY_USD']]
    d.head()
    
    sp500 = pd.read_csv('../data/numeric_raw_data/SP500.csv', header = 0)
    sp500.columns = ['DATE', 'SP500']
    sp500.head()
    
    kospi = pd.read_csv('../data/numeric_raw_data/KOSPI.csv', header = 0)
    kospi.columns = ['DATE', 'KOSPI']
    kospi.head()
    
    krx_bond_index = pd.read_csv('../data/numeric_raw_data/KRX_bond_index.csv', header = 0, encoding='CP949')
    del krx_bond_index['순가격지수']
    del krx_bond_index['제로재투자지수']
    del krx_bond_index['콜재투자지수']
    del krx_bond_index['시장가격지수']
    krx_bond_index.columns = ['DATE', 'KRX_BOND']
    krx_bond_index.head()
    
    nikkei = pd.read_csv('../data/numeric_raw_data/nikkei.csv', header = 0)
    nikkei.head()
    
    dji = pd.read_csv('../data/numeric_raw_data/dji.csv', header = 0, encoding='CP949')
    dji.columns = ['DATE', 'DJI']
    dji.head()
    
    dgs1 = pd.read_csv('../data/numeric_raw_data/DGS1.csv', header = 0)
    dgs1.head()
    
    dgs3 = pd.read_csv('../data/numeric_raw_data/DGS3.csv', header = 0)
    dgs3.head()
    
    dgs5 = pd.read_csv('../data/numeric_raw_data/DGS5.csv', header = 0)
    dgs5.head()
    
    dgs10 = pd.read_csv('../data/numeric_raw_data/DGS10.csv', header = 0)
    dgs10.head()
    
    dgs6mo = pd.read_csv('../data/numeric_raw_data/DGS6MO.csv', header = 0)
    dgs6mo.head()
    
    wti = pd.read_csv('../data/numeric_raw_data/WTI Crude Oil.csv', header = 0)
    wti.columns = ['DATE', 'WTI']
    wti.head()
    
    cygas = pd.read_csv('../data/numeric_raw_data/ngas.csv', header = 0)
    cygas.columns = ['DATE', 'NTGAS']
    cygas.head()
    
    ktbi = pd.read_csv('../data/numeric_raw_data/KRX-Korea Treasury Bond Index.csv', header = 0, encoding='CP949')
    del ktbi['전일대비']
    del ktbi['순가격지수']
    del ktbi['전일대비.1']
    del ktbi['시장가격지수']
    del ktbi['전일대비.2']
    ktbi.columns = ['DATE', 'KTBI']
    ktbi.head()
    
    nasdaq = pd.read_csv('../data/numeric_raw_data/nasdaq.csv', header = 0)
    nasdaq.columns = ['DATE', 'NASDAQ']
    nasdaq.head()
    
    prime_index = pd.read_csv('../data/numeric_raw_data/prime_index.csv', header = 0, encoding='CP949')
    prime_index = prime_index[['적용일자', '총수익지수']]
    prime_index.columns = ['DATE', 'PRIME_INDEX']
    prime_index.head()
    
    gold = pd.read_csv('../data/numeric_raw_data/gold.csv', header = 0, encoding='CP949')
    gold.columns = ['DATE', 'GOLD']
    gold.tail()
    
    dolar_index = pd.read_csv('../data/numeric_raw_data/dollar_index.csv', header = 0, encoding='CP949')
    dolar_index.columns = ['DATE', 'DOLLAR_INDEX']
    dolar_index.tail()
    
    silver = pd.read_csv('../data/numeric_raw_data/silver.csv', header = 0, encoding='CP949')
    silver.columns = ['DATE', 'SILVER']
    silver.tail()
    
    DAX = pd.read_csv('../data/numeric_raw_data/DAX.csv', header = 0, encoding='CP949')
    DAX.columns = ['DATE', 'DAX']
    DAX.tail()
    
    FTSE = pd.read_csv('../data/numeric_raw_data/FTSE.csv', header = 0, encoding='CP949')
    FTSE.columns = ['DATE', 'FTSE']
    FTSE.head()
    
    CAC = pd.read_csv('../data/numeric_raw_data/CAC.csv', header = 0, encoding='CP949')
    CAC = CAC[['Date', 'Close']]
    CAC.columns = ['DATE', 'CAC']
    CAC.tail()
    
    # 수집한 수치 데이터 종합
    temp = pd.merge(date, won_mea, on = 'DATE', how = 'left')
    temp = pd.merge(temp, d, on = 'DATE', how = 'left')
    
    temp = pd.merge(temp, krx_bond_index, on = 'DATE', how = 'left')
    temp = pd.merge(temp, prime_index, on = 'DATE', how = 'left')
    temp = pd.merge(temp, ktbi, on = 'DATE', how = 'left')
    temp = pd.merge(temp, dolar_index, on = 'DATE', how = 'left')
    
    temp = pd.merge(temp, kospi, on = 'DATE', how = 'left')
    temp = pd.merge(temp, sp500, on = 'DATE', how = 'left')
    temp = pd.merge(temp, dji, on = 'DATE', how = 'left')
    temp = pd.merge(temp, nasdaq, on = 'DATE', how = 'left')
    temp = pd.merge(temp, nikkei, on = 'DATE', how = 'left')
    
    temp = pd.merge(temp, dgs1, on = 'DATE', how = 'left')
    temp = pd.merge(temp, dgs3, on = 'DATE', how = 'left')
    temp = pd.merge(temp, dgs5, on = 'DATE', how = 'left')
    temp = pd.merge(temp, dgs10, on = 'DATE', how = 'left')
    temp = pd.merge(temp, dgs6mo, on = 'DATE', how = 'left')
    
    temp = pd.merge(temp, gold, on = 'DATE', how = 'left')
    temp = pd.merge(temp, silver, on = 'DATE', how = 'left')
    temp = pd.merge(temp, wti, on = 'DATE', how = 'left')
    temp = pd.merge(temp, cygas, on = 'DATE', how = 'left')
    
    temp = pd.merge(temp, DAX, on = 'DATE', how = 'left')
    temp = pd.merge(temp, FTSE, on = 'DATE', how = 'left')
    temp = pd.merge(temp, CAC, on = 'DATE', how = 'left')
    
    v1 = temp['USD_KRW'][-1:].values[0]
    
    # 데이터가 비어있는 날짜 삭제
    temp = temp.dropna(axis=0)
    temp.reset_index(drop = True, inplace = True)
    temp = temp.loc[temp['DATE']!='2012-04-30']
    v2 = temp['USD_KRW'][-1:].values[0]
    
    return v1, v2, temp

def for_text(v1, v2):
    text = temp.copy()
    fill = v1 - v2
    text['USD_KRW'] = text['USD_KRW'].shift(-1) - text['USD_KRW']
    text['USD_KRW'] = text['USD_KRW'].fillna(fill)
    
    text = text[['DATE', 'JPY_KRW', 'EUR_KRW', 'GBP_KRW', 'EUR_USD', 'GBP_USD',
           'JPY_USD', 'KRX_BOND', 'PRIME_INDEX', 'KTBI', 'DOLLAR_INDEX', 'KOSPI',
           'SP500', 'DJI', 'NASDAQ', 'NIKKEI', 'DGS1', 'DGS3', 'DGS5', 'DGS10',
           'DGS6MO', 'GOLD', 'SILVER', 'WTI', 'NTGAS', 'DAX', 'FTSE', 'CAC', 'USD_KRW']]
    
    return text

def for_dual():
    dual = temp.copy()
    del dual['DATE']
    dual = dual[['JPY_KRW', 'EUR_KRW', 'GBP_KRW', 'EUR_USD', 'GBP_USD',
           'JPY_USD', 'KRX_BOND', 'PRIME_INDEX', 'KTBI', 'DOLLAR_INDEX', 'KOSPI',
           'SP500', 'DJI', 'NASDAQ', 'NIKKEI', 'DGS1', 'DGS3', 'DGS5', 'DGS10',
           'DGS6MO', 'GOLD', 'SILVER', 'WTI', 'NTGAS', 'DAX', 'FTSE', 'CAC', 'USD_KRW']]
    
    return dual

v1, v2, temp = load_data()

temp.to_csv('../data/numeric_pre_data/complx_numeric_data.csv',index = False)

dual = for_dual()
dual.to_csv('../data/numeric_pre_data/numeric_data_for_dual.csv',index = False)

text = for_text(v1,v2)
text.to_csv('../data/numeric_pre_data/numeric_data_for_text.csv',index = False)