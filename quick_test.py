from arbitrage_strategy import *
import pandas as pd
from random import randint 
from pprint import pprint
from selected_pairs import construct_selected_pairs
import ccxt

pairs = [
['ADA/BNB','ARDR/BNB']
]

exchange = ccxt.binance()
exchange.enableRateLimit = True
exchange.load_markets(True)
for pair in pairs:
    test_arbitrage(EXCHANGE=exchange, SEC1 = pair[0], SEC2 = pair[1], DISPLAY_CHART='always')