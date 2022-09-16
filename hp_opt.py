from arbitrage_strategy import *
from selected_pairs import construct_selected_pairs
from selected_pairs import selected_arb
import ccxt

exchange = ccxt.binance()
exchange.enableRateLimit = True
exchange.load_markets(True)
active_pairs = []
for pair in exchange.symbols:
    if exchange.markets[pair]['active'] == True:
        active_pairs.append(pair)
active_pairs = construct_selected_pairs(active_pairs)
pairs = []
for sec1 in active_pairs:
    for sec2 in active_pairs:
        if sec1 == sec2:
            continue
        pairs.append([sec1,sec2])
        
arb_pairs = selected_arb
adf_last_candles = [250] #? нужно соотнести с наличием или отсутствием коинтеграции
history_bars = [500] #[200,500,1000, 5000] #
spread_ma_period = [15,20,40,50,70,90] # чем больше МА, тем чаще пересечение зоны. примерный оптимум = history_bars / 20
for pair in arb_pairs:
    for adf in adf_last_candles:
#        for spread_ma in spread_ma_period:
            for bars in history_bars:
                print([pair,adf,bars])
                test_arbitrage(EXCHANGE=exchange, SEC1 = pair[0], SEC2 = pair[1], DISPLAY_CHART='always',
                               ADF_LAST_CANDLES = adf, SPREAD_MA_PERIOD = bars//20, HISTORY_BARS = bars)