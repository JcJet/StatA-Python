def reset_buffer(self):
    self.history_buffer = self.history_buffer[0:0]
    self.buffer = self.buffer[0:0]
def calc_spread_channel(self, price1, price2, datetime):
    self.b = self.calc_linear_regression_coeff()
    spread = self.calc_spread(price1, price2)   
    if len(self.buffer) < self.spread_ma_period:
        ma = spread
    else:
        ma = self.buffer.loc[:,'spread'].rolling(window=self.spread_ma_period, center=False).mean()[-1:].values[0]
    sd1 = ma + self.buffer.loc[:,'spread'].std()
    sd2 = ma - sd1
    sd3 = ma + sd1 * 2
    sd4 = ma - sd1 * 2
    zones = self.spread_range(spread, self.spread_top_zone, self.spread_bottom_zone)
    df = pd.DataFrame({"datetime":datetime,
                       "spread":spread,
                       "spread_ma":ma,
                       "+std":sd1,
                       "-std":sd2,
                       "+std*2":sd3,
                       "-std*2":sd4,
                       "top_zone":zones[0],
                       "bottom_zone":zones[1]}, index=[datetime])  
    self.buffer = self.buffer.append(df)
    if len(self.buffer.index) > self.length:
        self.buffer = self.buffer[1:]
    
    return df #[datetime, spread, ma, sd1, sd2, sd3, sd4]
def spread_range(self, cur_spread, top = .75, bottom = .25):
    #spread_min = self.buffer['spread'].min()
    #spread_max = self.buffer['spread'].max()
    if self.spread_range_period == 0:
        if cur_spread > self.current_max:
            self.current_max = cur_spread
        if cur_spread < self.current_min:
            self.current_min = cur_spread
        
    else:
        df = pd.DataFrame({"spread":cur_spread}, index = [0])
        self.range_buffer = self.range_buffer.append(df)
        if len(self.range_buffer) > self.spread_range_period:
            self.range_buffer = self.range_buffer[1:]
        self.current_max = self.range_buffer['spread'].max()
        self.current_min = self.range_buffer['spread'].min()
    channel = self.current_max - self.current_min
    return [self.current_min+channel*top, self.current_min+channel*bottom]    
def is_formed(self):
    return len(self.buffer.index) == self.length
#spread_indicator = SpreadIndicator(name='', length=SPREAD_IND_PERIOD)
#spread_df = pd.DataFrame()
#очень медленно, м.б. обрабатывать через .apply будет лучше, отдельным методом process_history
#for index, row in indexed_candles.iterrows():
#    spread_df = spread_df.append(spread_indicator.process(row['price1'], row['price2'], row['datetime']))
#spread_df = spread_df.set_index(pd.DatetimeIndex(spread_df['datetime']))
#spread_df = spread_df.drop(columns=['datetime'])
#indexed_candles = indexed_candles.join(spread_df,on=indexed_candles.index, sort=True)

class SpreadIndicator:
    def __init__(self, name='', length=25):
        self.name = name
        self.length = length
        self.history_buffer = pd.DataFrame(columns=['datetime','price1','price2'])
        self.b = 1
    def process(self, price1, price2, datetime):
        if price1 != None and price2 != None:
                df = pd.DataFrame({"datetime":datetime,
                                   "price1":price1,
                                   "price2":price2}, index=[datetime]) #стоит проверить сортировку значений в этих двух DataFrame
                self.history_buffer=self.history_buffer.append(df)
                if len(self.history_buffer.index) > self.length:
                    self.history_buffer = self.history_buffer[1:]
        else:
            raise Exception('no candle specified')
        self.b = self.calc_linear_regression_coeff()
        spread = self.calc_spread(price1, price2)
        df = pd.DataFrame({"datetime":datetime,
                           "spread":spread}, index=[datetime])          
        return df
    def calc_linear_regression_coeff(self):
        model = LinearRegression()
        model.fit(self.history_buffer['price1'].values.reshape((-1, 1)),self.history_buffer['price2'].values)
        return model.coef_[0]
    def calc_spread(self, price1, price2):
        b = self.b
        Y1 = price1
        Y2 = price2
        S = Y1*b - Y2                   #is 'b' coeff is really an efficient way to do it?
        return S
    
    def plot_chart(ts):
        df = pd.DataFrame(ts)
        fig = go.Figure(data=[go.Candlestick(x=df[0],
                        open=df[1],
                        high=df[2],
                        low=df[3],
                        close=df[4])])
        fig.show()