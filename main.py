import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
#from datetime import datetime
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import pprint

import numpy as np
import ccxt
#Market
SEC1 = 'ETC/USDT'
SEC2 = 'ETH/USDT'
EXCHANGE = 'binance'
#Hyperparameters
TIMEFRAME = '1h'
HISTORY_BARS = 1000 #Период истории для вычисления стандартного отклонения, коинтеграции и коэффициента для рассчета спреда
TESTING_BARS = 24*7 #Период для теста стратегии или поиска сигнала (без вычисления новых коэффициентов)
MINIMUM_COINTEGRATION = .7
#Fine-tuning
#SPREAD_IND_PERIOD = 200
SPREAD_MA_PERIOD = 50 #sma period - which one is recommended? =< IND_PERIOD (is there reason to make it more?)
SPREAD_TOP_ZONE = .75
SPREAD_BOTTOM_ZONE = .25
#SPREAD_RANGE_PERIOD = 0 #0 = all processed history
ADF_LAST_CANDLES = 500
ADF_MA_PERIOD = 50
DISPLAY_ORIGINAL_PRICES = False #True - изначальные цены, False - умноженные на b
DISPLAY_ON_SECONDARY_AXIS = False #второй инструмент на отдельной шкале Y
    
class StationarityIndicator: #timeseries pseudo-adf indicator
    def __init__(self, length = 25):
        self.length = length
        self.buffer = pd.DataFrame(columns=['datetime','value'])
    def process(self, datetime, value):
        df = pd.DataFrame({"datetime":datetime,
                           "value":value},index=[0])
        self.buffer = self.buffer.append(df)
        if len(self.buffer.index) > self.length:
            self.buffer = self.buffer[1:]    
        return self.adf_test()
    def adf_test(self):
        if len(self.buffer.index) < self.length:
            return 1
        x = self.buffer["value"].values
        res = adfuller(x)
        return 1-res[1]
    
   
def load_history(sec, start_date, tf):
    if not exchange.has['fetchOHLCV']:
        raise Exception('%s does not have fetchOHLCV method')
    if exchange.has['fetchOHLCV'] == 'emulated':
        raise Exception('fetchOHLCV is emulated, experimental recommended for fallback use only')
    since = start_date# exchange.iso8601(start_date)
    all_candles = []
    symbol = sec['symbol']
    limit=None
    while True: #since < exchange.microseconds():
        candles = exchange.fetchOHLCV(symbol, tf, since, limit)
        if len(candles):
            if candles[len(candles) - 1][0] == since:
                break
            since = candles[len(candles) - 1][0]
            all_candles += candles
            print(exchange.iso8601(since))
        else:
            break
    for candle in all_candles:
        candle[0] = pd.to_datetime(candle[0], unit='ms') #exchange.iso8601(candle[0])
    return all_candles
def timeframe_to_microseconds(tf):
    conversion = {
        '1m':1*60*10**3,
        '3m':3*60*10**3,
        '5m':5*60*10**3,
        '15m':15*60*10**3,
        '30m':30*60*10**3,
        '1h':1*60*60*10**3,
        '2h':2*60*60*10**3,
        '4h':4*60*60*10**3,
        '6h':6*60*60*10**3,
        '8h':8*60*60*10**3,
        '12h':12*60*60*10**3,
        '1d':1*24*60*60*10**3,
        '3d':3*24*60*60*10**3,
        '1w':7*24*60*60*10**3,
        '1M':30*24*60*60*10**3
    }
    return conversion[tf]
def plot_strategy_chart(ts):
    if DISPLAY_ORIGINAL_PRICES:
        b_coeff = 1
    else:
        b_coeff = b
    fig = make_subplots(rows=2, cols=1, 
                        specs=[[{"secondary_y": True}],
                               [{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['price1'],name=SEC1, 
                             line = dict(color = 'blue', width = 2)),
                             row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['price2']/b_coeff,name=SEC2, 
                             line = dict(color = 'red', width = 2)),
                             row=1, col=1, secondary_y=DISPLAY_ON_SECONDARY_AXIS)
    
    fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['spread'],name="spread", 
                             line = dict(color = 'black', width = 1)),
                             row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['spread_ma'],name="spread_ma", 
                             line = dict(color = 'green', width = 1)),
                             row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['+std'],name="+std", 
                             line = dict(color = 'orange', width = 1)),
                             row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['-std'],name="-std", 
                             line = dict(color = 'orange', width = 1)),
                             row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['+std*2'],name="+std*2", 
                             line = dict(color = 'red', width = 1)),
                             row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['-std*2'],name="-std*2", 
                             line = dict(color = 'red', width = 1)),
                             row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['top_zone'],name="top_zone",
                             line = dict(color = 'gray', width = 1, dash='dash')),
                             row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['bottom_zone'],name="bottom_zone", 
                             line = dict(color = 'gray', width = 1, dash='dash')), 
                             row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['top'],name="top", 
                             line = dict(color = 'gray', width = 1, dash='dash'),
                             visible = 'legendonly'), row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['bottom'],name="bottom", 
                             line = dict(color = 'gray', width = 1, dash='dash'),
                             visible = 'legendonly'), row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Bar(x=ts['datetime'], y=ts['adf'],name="adf_test",
                             visible = 'legendonly'), row=2, col=1, secondary_y=True)
    
    fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['adf_ma'],name="adf_ma", 
                             line = dict(color = 'gray', width = 1),
                             visible = 'legendonly'), row=2, col=1, secondary_y=True)

    fig.show()
def plot_chart(ts):
    df = pd.DataFrame(ts)
    fig = go.Figure(data=[go.Candlestick(x=df[0],
                    open=df[1],
                    high=df[2],
                    low=df[3],
                    close=df[4])])
    fig.show()
def get_signal(ts):
    raise Exception("Not implemented")
#==========GET DATA==========
exchange = ccxt.binance()
exchange.enableRateLimit = True
print(exchange.timeframes)
exchange.load_markets(True)
print(exchange.symbols)
sec1 = exchange.markets[SEC1]
sec2 = exchange.markets[SEC2]
start_date = exchange.microseconds() // 1000 - HISTORY_BARS * timeframe_to_microseconds(TIMEFRAME)
candles1 = load_history(sec1, start_date, TIMEFRAME)
candles2 = load_history(sec2, start_date, TIMEFRAME)
#==========COMBINE PRICES==========
indexed_candles = pd.DataFrame(columns=['datetime','price1', 'price2'])
for candle1 in candles1:
    df = pd.DataFrame({"datetime":candle1[0],
                         "price1":candle1[4],},index=[candle1[0]])
    indexed_candles = indexed_candles.append(df, sort=False)
indexed_candles = indexed_candles.set_index(pd.DatetimeIndex(indexed_candles['datetime']))
for candle2 in candles2:  
    indexed_candles.loc[(candle2[0],'price2')] = candle2[4]
indexed_candles.drop_duplicates(subset='datetime', keep='last', inplace=True)
indexed_candles.sort_index(inplace=True)
if indexed_candles.isnull().values.any():
    raise Exception('Some values are null, implementation of missing data handling is probably needed')
#==========SPREAD==========
model = LinearRegression()
model.fit(indexed_candles['price1'].values.reshape((-1, 1)),indexed_candles['price2'].values)
b = model.coef_[0]
Y1 = indexed_candles['price1']
Y2 = indexed_candles['price2']
indexed_candles['spread'] = Y1*b - Y2
#==========STATIONARITY==========
stationarity_indicator = StationarityIndicator(ADF_LAST_CANDLES) 
indexed_candles['adf'] = indexed_candles.apply(lambda row : stationarity_indicator.process(row['datetime'], row['spread']), axis = 1)
indexed_candles['adf_ma'] = indexed_candles.loc[:,'adf'].rolling(window = ADF_MA_PERIOD).mean()
#==========SIGNAL ZONES==========
indexed_candles['spread_ma'] = indexed_candles.loc[:,'spread'].rolling(window = SPREAD_MA_PERIOD).mean()
sd = indexed_candles.loc[:,'spread'].std()
indexed_candles['+std'] = indexed_candles.loc[:,'spread_ma'] + sd
indexed_candles['-std'] = indexed_candles.loc[:,'spread_ma'] - sd
indexed_candles['+std*2'] = indexed_candles.loc[:,'spread_ma'] + sd * 2
indexed_candles['-std*2'] = indexed_candles.loc[:,'spread_ma'] - sd * 2
spread_min = indexed_candles.loc[:,'spread'].min()
spread_max = indexed_candles.loc[:,'spread'].max()
spread_channel = spread_max-spread_min
indexed_candles['bottom_zone'] = spread_min + spread_channel * SPREAD_BOTTOM_ZONE
indexed_candles['top_zone'] = spread_min + spread_channel * SPREAD_TOP_ZONE
#==========DEBUG==========
indexed_candles['top'] = spread_max
indexed_candles['bottom'] = spread_min
#==========DISPLAY==========
print(indexed_candles)
plot_strategy_chart(indexed_candles)
#==========SIGNAL==========
last_entry = indexed_candles[-1:]
spread = last_entry['spread'].values[0]
adf = last_entry['adf'].values[0]
std_plus = last_entry['+std'].values[0]
std_minus = last_entry['-std'].values[0]
std_plus_x2 = last_entry['+std*2'].values[0]
std_minus_x2 = last_entry['-std*2'].values[0]
signal_conditions = {
'Cointegration' : False,
'Std crossed:': False,
'Std2 crossed': False
}
print("%s-%s:"%(SEC1,SEC2))
signal_conditions['Cointegration'] = adf >= MINIMUM_COINTEGRATION
signal_conditions['Std crossed'] = spread > std_plus or spread < std_minus
signal_conditions['Std2 crossed'] = spread > std_plus_x2 or spread < std_minus_x2
print('%s/%s'%(sum(signal_conditions.values()), len(signal_conditions.values())))
pprint.pprint(signal_conditions)
if spread > std_plus and spread < std_plus_x2:
    print('Sell %s : %s' % (b,SEC1))
    print('Buy %s : %s' % (1, SEC2) )
if spread > std_plus_x2:
    print('Sell %s : %s' % (b*2,SEC1))
    print('Buy %s : %s' % (1*2, SEC2))
if spread < std_minus and spread > std_minus_x2:
    print('Buy %s : %s' % (b,SEC1))
    print('Sell %s : %s' % (1, SEC2))    
if spread < std_minus_x2:
    print('Buy %s : %s' % (b*2,SEC1))
    print('Sell %s : %s' % (1*2, SEC2))    