import datetime
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import product
from statsmodels.tsa.stattools import adfuller
from johansen_test import coint_johansen

#from strategy import Strategy
#from event import SignalEvent
#from backtest import Backtest
#from hft_data import HistoricCSVDataHandlerHFT
#from hft_portfolio import PortfolioHFT
#from execution import SimulatedExecutionHandler

from pyalgotrade import strategy
from pyalgotrade import dataseries
from pyalgotrade.dataseries import aligned
from pyalgotrade import plotter
from pyalgotrade.barfeed import ninjatraderfeed
from pyalgotrade.stratanalyzer import sharpe
from pyalgotrade.bar import Frequency

cointegration_tests = ['cadf','johansen_trace', 'johansen_eigen']


class StatArb(strategy.BacktestingStrategy):
    """
    Uses ordinary least squares (OLS) to perform a rolling linear regression to determine the hedge ratio 
    between a pair of equities. The z-score of the residuals time series is then calculated in a rolling fashion 
    and if it exceeds an interval of thresholds (defaulting to [0.5, 3.0]) then a long/short
    signal pair are generated (for the high threshold) or an exit signal pair are generated (for the low threshold). 
    """
    def __init__(self, events, feed, sec1, sec2, windowSize=100, zscore_exit=0.5, zscore_entry=3.0,
                 cadf_max_pval = 0.05, coint_method='johansen_trace', coint_test_k=1):
        """ 
        Initialises the stat arb strategy.
        Parameters: bars - The DataHandler object that provides bar information
        events - The Event Queue object. 
        """
        super(StatArb, self).__init__(feed)
        self.setUseAdjustedValues(True)
        self.events = events
        self.windowSize = windowSize
        self.zscore_exit = zscore_exit
        self.zscore_entry = zscore_entry
        self.cadf_max_pval = cadf_max_pval
        self.coint_method = coint_method
        self.coint_test_k = coint_test_k
        self.feed = feed
        self.x = feed[sec1].getAdjCloseDataSeries()
        self.y = feed[sec2].getAdjCloseDataSeries()
        self.sec1 = sec1
        self.sec2 = sec2
        self.datetime = datetime.datetime.utcnow()
        
        self.long_market = False
        self.short_market = False
        
        # These are used only for plotting purposes.
        self.__spread = dataseries.SequenceDataSeries()
        self.__hedgeRatio = dataseries.SequenceDataSeries()        
        #
        
        try:
            b=cointegration_tests.index(coint_method)
        except ValueError:
            raise ValueError('Unknown cointegration test.')
    def get_order_size(self):
        cash = self.getBroker().getCash(False)
        price1 = self.bars[self.sec1].getAdjClose()
        price2 = self.bars[self.sec2].getAdjClose()
        size1 = int(cash / (price1 + self.hedge_ratio * price2))
        size2 = int(size1 * self.hedge_ratio)
        return (size1, size2) 
    def buy_spread(self):
        amount1, amount2 = self.get_order_size(self.bars)
        self.marketOrder(self.sec1, amount1)
        self.marketOrder(self.sec2, amount2 * -1)

    def sell_spread(self):
        amount1, amount2 = self.get_order_size(self.bars)
        self.marketOrder(self.sec1, amount1 * -1)
        self.marketOrder(self.sec2, amount2)
    def close_position(self, instrument):
        currentPos = self.getBroker().getShares(instrument)
        if currentPos > 0:
            self.marketOrder(instrument, currentPos * -1)
        elif currentPos < 0:
            self.marketOrder(instrument, currentPos * -1)    
    def update_orders(self):
        """
        Calculates the actual x, y signal pairings to be sent to the signal generator.
        
        Parameters:
        zscore_last - The current zscore to test against 
        adf_pval - last p-value scre of CADF test on residuals
        """
        hr = []
        cadf_passed = True
        if self.coint_method == 'cadf':
            cadf_passed = adf_pval <= self.cadf_max_pval #CADF: меньше - значит сильнее сигнал
            hr.append(1.0)
            hr.append(abs(self.hedge_ratio))
        else:
            hr = self.hedge_ratio
            cadf_passed = adf_pval >= self.cadf_max_pval #Joh: больше - значит сильнее сигнал
        
        currentPos = abs(self.getBroker().getShares(self.sec1)) + abs(self.getBroker().getShares(self.sec2))
        if abs(self.zscore_last) <= self.zscore_exit or not cadf_passed:
            if currentPos != 0:
                self.close_position(self.sec1)
                self.close_position(self.sec2)
        elif self.zscore_last <= -self.zscore_entry and currentPos == 0:  # Buy spread when its value drops below target standard deviations.
            if cadf_passed:
                self.buy_spread()
        elif self.zscore_last >= self.zscore_entry and currentPos == 0:  # Short spread when its value rises above target standard deviations.
            if cadf_passed:
                self.sell_spread()
    
    def update_signal(self):
        """
        Generates a new set of signals based on the mean reversion strategy.
        Calculates the hedge ratio between the pair of tickers.
        We use OLS for this, althought we should ideal use CADF.
        """ 
        # Obtain the latest window of values for each component of the pair of tickers
        y = self.bars.get_latest_bars_values(self.pair[0], "close", N=self.windowSize)
        x = self.bars.get_latest_bars_values(self.pair[1], "close", N=self.windowSize)
        
        if y is not None and x is not None:
            # Check that all window periods are available
            if len(y) >= self.windowSize and len(x) >= self.windowSize:
                # Calculate the current hedge ratio using OLS
                self.hedge_ratio = sm.OLS(y,x).fit().params[0]
                
                # Calculate the current z-score of the residuals
                spread = y - self.hedge_ratio * x
                self.zscore_last = ((spread - spread.mean()) / spread.std())[-1] # в терминах "старой стратегии" - отклонение спреда в единицах СКО
                
                # CADF
                self.adf_pval = 1.0
                if self.coint_method == 'cadf':
                    self.adf_pval = adfuller(spread)[1]
                else:
                    # Тест на коинтеграцию Йохансена. При использовании метода нужно учитывать, что в, в отличии от CADF, результатом
                    # является не p-значение, а вероятность того, что активы коинтегрированы, т.е. пороговые значения
                    # будут совсем иные.
                    df = pd.DataFrame({'x':x, 'y':y})
                    test_stats = coint_johansen(df,0,int(self.coint_test_k),print_on_console=False)
                    cointegration_likelihood = 1.0
                    if self.coint_method == 'johansen_trace':
                        lr_stats = test_stats.lr1
                        cv_stats = test_stats.cvt
                    if self.coint_method == 'johansen_eigen':
                        lr_stats = test_stats.lr2
                        cv_stats = test_stats.cvm
                    for lr_index in range(len(lr_stats)):
                        lr = lr_stats[lr_index]
                        critical_value = 0.0
                        if lr >= cv_stats[lr_index][0]:
                            critical_value = 0.9
                        if lr >= cv_stats[lr_index][1]:
                            critical_value = 0.95
                        if lr >= cv_stats[lr_index][2]:
                            critical_value = 0.99
                        if cointegration_likelihood > critical_value:
                            cointegration_likelihood = critical_value
                    self.adf_pval = cointegration_likelihood
                    if self.adf_pval >= 0.9:
                        max_eigenvalue = -math.inf
                        max_ev_index = -1
                        for i in range(len(test_stats.eig)):
                            if test_stats.eig[i] > max_eigenvalue: #а не максимальное ли по модулю нужно?
                                max_eigenvalue = test_stats.eig[i]
                                max_ev_index = i
                        self.hedge_ratio = test_stats.evec[max_ev_index]
                    
    
    def onBars(self, bars):
        '''on new market data'''
        self.bars = bars
        self.update_signal()
        self.update_orders()
    
if __name__ == "__main__":
    csv_dir = '/home/jet/Downloads/'
    sec1 = 'SBER'
    sec2 = 'SBERP'
    feed = ninjatraderfeed.Feed(Frequency.MINUTE)
    windowSize = 100
    feed.addBarsFromCSV(sec1, csv_dir+sec1+".csv")
    feed.addBarsFromCSV(sec2, csv_dir+sec2+".csv")
    strat = StatArb(feed, sec1, sec2, windowSize)
    initial_capital = 100000.0
    heartbeat = 0.0
    start_date = datetime.datetime(2019,9,2)
    sharpeRatioAnalyzer = sharpe.SharpeRatio()
    strat.attachAnalyzer(sharpeRatioAnalyzer)
    
    strat.run()
    print("Sharpe ratio: %.2f" % sharpeRatioAnalyzer.getSharpeRatio(0.05))    
    
    
    ## Create the strategy parameter grid using the itertools cartesian product generator
    #strat_lookback = [200] #[50, 100, 200]
    #strat_z_entry = [2.0] #[2.0, 3.0, 4.0] # больше - немного повышает прибыль и просадку
    #strat_z_exit = [1.0] #[0.5, 1.0, 1.5]
    #strat_min_pval = [0.95] #johansen: [0.90, 0.95, 0.99] cadf: [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]
    #strat_coint_method = ['johansen_trace']
    #strat_coint_test_k = ['2'] #в книгах применялся лаг 1 и 2. Применяется только для Johansen, для ADF на данный момент используется значение по умолчанию - k=12*(nobs/100)^{1/4}
    #strat_params_list = list(product(strat_lookback, strat_z_entry, strat_z_exit, strat_min_pval, strat_coint_method, strat_coint_test_k))
    #output_file = "".join([str(s) for s in strat_coint_method])+"".join([str(s) for s in strat_lookback])+".csv"
    ## Create a list of dictionaries with the correct keyword/value pairs for the strategy parameters strat_params_dict_list
    
    #strat_params_dict_list = [
        #dict(windowSize=sp[0], zscore_entry=sp[1], zscore_exit=sp[2], cadf_max_pval=sp[3], coint_method=sp[4], coint_test_k = sp[5])
        #for sp in strat_params_list]
    
    
    #backtest = Backtest(csv_dir, initial_capital, heartbeat, 
                        #start_date, HistoricCSVDataHandlerHFT, SimulatedExecutionHandler, PortfolioHFT, StatArb,
                        #strat_params_list=strat_params_dict_list, out_file=output_file)
    
    #backtest.simulate_trading()