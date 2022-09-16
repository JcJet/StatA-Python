import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
#from datetime import datetime
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from random import randint 
from pprint import pprint
from scipy.optimize import nnls
import numpy as np
import ccxt
DARK_THEME = False
DOUBLE_CHARTS = False
USE_NNLS_REGRESSION = False #experiment with non-negative constraint on coefficient
COUNT_PAST_SIGNALS = False #поиск уже не актуальных сигналов в недавнем прошлом. для поиска новых пар в избранные
def test_arbitrage(
    SEC1 = 'ETC/USDT', 
    SEC2 = 'ETH/USDT', 
    EXCHANGE = None, 
    TIMEFRAME = '1h',

    HISTORY_BARS = 1000,
    TESTING_BARS = 24*7, 
    MINIMUM_COINTEGRATION = .95, 

    SPREAD_MA_PERIOD = 50, 
    SPREAD_TOP_ZONE = .75, 
    SPREAD_BOTTOM_ZONE = .25, 
    ADF_LAST_CANDLES = 1000, 
    ADF_MA_PERIOD = 50,
    DISPLAY_ORIGINAL_PRICES = True,
    DISPLAY_ON_SECONDARY_AXIS = True,
    DISPLAY_CHART = 'always',
    MIDDLE_ZONE = .05):  

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
                return 0
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
                #print(exchange.iso8601(since))
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

        fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['price1']*b_coeff,name=SEC1, 
                                 line = dict(color = 'blue', width = 2)),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['price2'],name=SEC2, 
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

        fig.add_trace(go.Scatter(x=ts['datetime'], y=ts['middle_zone'],name="middle_zone", 
                                 line = dict(color = 'gray', width = 1, dash='dash'),
                                 visible = 'legendonly'), row=2, col=1, secondary_y=False)
        
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
        fig.add_trace(go.Bar(x=ts['datetime'], y=ts['signal'],name="signal",
                             visible = 'legendonly'), row=2, col=1, secondary_y=True)        
        if DARK_THEME:
            fig.update_layout(template="plotly_dark")
        fig.show()
    def signal(last_entry):
        spread = last_entry['spread'].values[0]
        adf = last_entry['adf'].values[0]
        middle = last_entry['spread_ma'].values[0]
        std_plus = last_entry['+std'].values[0]
        std_minus = last_entry['-std'].values[0]
        std_plus_x2 = last_entry['+std*2'].values[0]
        std_minus_x2 = last_entry['-std*2'].values[0]
        signal_conditions = {
        'Cointegration' : False,
        'Std crossed': False,
        'Std2 crossed': False
        }
        signal_conditions['Cointegration'] = adf >= MINIMUM_COINTEGRATION
        signal_conditions['Std crossed'] = spread > std_plus or spread < std_minus
        signal_conditions['Std2 crossed'] = spread > std_plus_x2 or spread < std_minus_x2
        pos = (None,None)
        if all(signal_conditions.values()):
            sig = True
            if spread > std_plus and spread < std_plus_x2:
                print('Sell %s : %s' % (1,SEC1))
                print('Buy %s : %s' % (1, SEC2))
                pos = (-1,1)
            elif spread > std_plus_x2:
                print('Sell %s : %s' % (1*2,SEC1))
                print('Buy %s : %s' % (1*2, SEC2))
                pos = (-2,2)
            elif spread < std_minus and spread > std_minus_x2:
                print('Buy %s : %s' % (1,SEC1))
                print('Sell %s : %s' % (1, SEC2))
                pos = (1,-1)
            elif spread < std_minus_x2:
                print('Buy %s : %s' % (1*2,SEC1))
                print('Sell %s : %s' % (1*2, SEC2))
                pos = (2,-2)
            elif spread < middle+(middle*MIDDLE_ZONE) and spread > middle-(middle*MIDDLE_ZONE):
                pos = (0,0)
        else:
            sig = False
        res = {
        'Signal': sig,
        'Position': pos,
        'Cointegration': signal_conditions['Cointegration'],
        'Std crossed': signal_conditions['Std crossed'],
        'Std2 crossed': signal_conditions['Std2 crossed']
        }
        return res
#==========TESTING===========
    def backtest(candles):
        pos = 0
        good = 0
        bad = 0
        for candle in candles:
            sig = signal(candle)
            i
    #==========GET DATA==========
    exchange = EXCHANGE
    sec1 = exchange.markets[SEC1]
    sec2 = exchange.markets[SEC2]
    start_date = exchange.microseconds() // 1000 - HISTORY_BARS * timeframe_to_microseconds(TIMEFRAME)
    candles1 = load_history(sec1, start_date, TIMEFRAME)
    candles2 = load_history(sec2, start_date, TIMEFRAME)
    #==========COMBINE PRICES==========
    candles1.sort(key = lambda x : x[0])
    candles2.sort(key = lambda x : x[0])
    while candles1[-1][0] != candles2[-1][0]:
        candles2 = candles2[:-1]
    indexed_candles = pd.DataFrame(columns=['datetime','price1', 'price2'])
    for candle1 in candles1:
        df = pd.DataFrame({"datetime":candle1[0],
                           "price1":candle1[4],},index=[candle1[0]])
        indexed_candles = indexed_candles.append(df, sort=False)
    indexed_candles = indexed_candles.set_index(pd.DatetimeIndex(indexed_candles['datetime']))
    indexed_candles.drop_duplicates(subset='datetime', keep='last', inplace=True) #dupes from candles1
    for candle2 in candles2:  
        indexed_candles.loc[(candle2[0],'price2')] = candle2[4]
    indexed_candles.drop_duplicates(subset='datetime', keep='last', inplace=True) #dupes from candles2
    indexed_candles.sort_index(inplace=True)
    if indexed_candles.isnull().values.any():
        missing = len(indexed_candles.isna())
        print('indexed_candles had missing values, count:', missing)
        indexed_candles = indexed_candles.dropna(axis=0) # remove rows with missing date
        indexed_candles = indexed_candles.interpolate(method='time') #interpolate nan
    #==========SPREAD==========
    model = LinearRegression()
    model.fit(indexed_candles['price1'].values.reshape((-1, 1)),indexed_candles['price2'].values)
    b = model.coef_[0]
    Y1 = indexed_candles['price1']
    Y2 = indexed_candles['price2']
    indexed_candles['spread'] = Y1*b - Y2
    if USE_NNLS_REGRESSION:
        model = nnls(indexed_candles['price1'].values.reshape((-1, 1)),indexed_candles['price2'].values)
        b = model[0][0]
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
    indexed_candles['middle_zone'] = spread_min + spread_channel * .5
    indexed_candles['top_zone'] = spread_min + spread_channel * SPREAD_TOP_ZONE
    #==========DEBUG==========
    indexed_candles['top'] = spread_max
    indexed_candles['bottom'] = spread_min
    indexed_candles['signal'] = signal(indexed_candles[-1:])['Position'][0]
    #==========SIGNAL==========
    last = indexed_candles[-1:]
    res = signal(last)
    if res['Std crossed'] and COUNT_PAST_SIGNALS:
        for entry in indexed_candles[:-1]:
            if entry['Std crossed']:
                res = entry
                break
    #==========DISPLAY==========
    if DISPLAY_CHART == 'always':
        plot_strategy_chart(indexed_candles)
    if (DISPLAY_CHART == 'signal' and res['Std crossed']) or (DISPLAY_CHART=='cointegration' and res['Cointegration']):
        plot_strategy_chart(indexed_candles)
        if DOUBLE_CHARTS:          
            DISPLAY_ON_SECONDARY_AXIS = not DISPLAY_ON_SECONDARY_AXIS
            DISPLAY_ORIGINAL_PRICES = not DISPLAY_ORIGINAL_PRICES
            plot_strategy_chart(indexed_candles)
            DISPLAY_ON_SECONDARY_AXIS = not DISPLAY_ON_SECONDARY_AXIS
            DISPLAY_ORIGINAL_PRICES = not DISPLAY_ORIGINAL_PRICES
    return res
if __name__ == '__main__':
    exchange = ccxt.binance()
    exchange.enableRateLimit = True
    print(exchange.timeframes)
    exchange.load_markets(True)
    test_arbitrage(EXCHANGE=exchange)