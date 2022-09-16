from arbitrage_strategy import *
import pandas as pd
from random import randint 
from pprint import pprint
from selected_pairs import construct_selected_pairs
import ccxt
from time import sleep
DARK_THEME = True
DOUBLE_CHARTS = False
USE_NNLS_REGRESSION = False #experiment with non-negative constraint on coefficient
SEARCH_TYPE = 'selected' #random pairs shuffle, when there's too much combinations to check them all...
COUNT_PAST_SIGNALS = False #поиск графиков с сигналами в недавнем прошлом. для поиска новых пар в избранные
exchange = ccxt.binance()
exchange.enableRateLimit = True
print(exchange.timeframes)
exchange.load_markets(True)
#print(exchange.symbols)
scoring_table = pd.DataFrame(columns=['Sec1', 'Sec2', 'Cointegration', 'Std crossed','Std2 crossed', 'Signal Now'])
active_pairs = []
for pair in exchange.symbols:
    if exchange.markets[pair]['active'] == True:
        active_pairs.append(pair)
print(active_pairs)
total_arbitrage_pairs = len(active_pairs) ** 2 - len(active_pairs)
cur_iter = 0
if SEARCH_TYPE == 'random':
    searched = []
    ind1 = 0
    ind2 = 0

    while cur_iter < len(active_pairs):
        while ind1 == ind2 or (ind1, ind2) in searched:
            ind1 = randint(0,len(active_pairs)-1)
            ind2 = randint(0,len(active_pairs)-1)
        sec1 = active_pairs[ind1]
        sec2 = active_pairs[ind2]
        searched.append((ind1,ind2))
        arbitrage_res = test_arbitrage(EXCHANGE=exchange, SEC1 = sec1, SEC2 = sec2, DISPLAY_CHART='signal')
        arbitrage_test = {}
        arbitrage_test['Sec1'] = sec1
        arbitrage_test['Sec2'] = sec2
        arbitrage_test['Cointegration'] = arbitrage_res['Cointegration']
        arbitrage_test['Std crossed'] = arbitrage_res['Std crossed']
        arbitrage_test['Std2 crossed'] = arbitrage_res['Std2 crossed']
        arbitrage_test['Signal Now'] = arbitrage_res['Signal']
        df = pd.DataFrame(arbitrage_test, index=[cur_iter])
        scoring_table.append(df, sort=False)
        cur_iter += 1
        pprint(arbitrage_test)
        print('%s/%s'%(cur_iter,total_arbitrage_pairs))          
if SEARCH_TYPE == 'all': 
    for sec1 in active_pairs:
        for sec2 in active_pairs:
            if sec1 == sec2:
                continue
            arbitrage_res = test_arbitrage(EXCHANGE=exchange, SEC1 = sec1, SEC2 = sec2, DISPLAY_CHART='signal')
            arbitrage_test = {}
            arbitrage_test['Sec1'] = sec1
            arbitrage_test['Sec2'] = sec2
            arbitrage_test['Cointegration'] = arbitrage_res['Cointegration']
            arbitrage_test['Std crossed'] = arbitrage_res['Std crossed']
            arbitrage_test['Std2 crossed'] = arbitrage_res['Std2 crossed']
            arbitrage_test['Signal Now'] = arbitrage_res['Signal']
            df = pd.DataFrame(arbitrage_test, index=[cur_iter])
            scoring_table.append(df, sort=False)
            cur_iter += 1
            pprint(arbitrage_test)
            print('%s/%s'%(cur_iter,total_arbitrage_pairs))
if SEARCH_TYPE == 'selected':
    active_pairs = construct_selected_pairs(active_pairs)
    total_arbitrage_pairs = len(active_pairs) ** 2 - len(active_pairs)
    for sec1 in active_pairs:
        for sec2 in active_pairs:
            if sec1 == sec2:
                continue
            try:
                arbitrage_res = test_arbitrage(EXCHANGE=exchange, SEC1 = sec1, SEC2 = sec2, DISPLAY_CHART='cointegration')
            except ccxt.base.errors.RequestTimeout:
                print('Connection time out, retry in 1 minute...')
                sleep(60)
                arbitrage_res = test_arbitrage(EXCHANGE=exchange, SEC1 = sec1, SEC2 = sec2, DISPLAY_CHART='cointegration')
            arbitrage_test = {}
            arbitrage_test['Sec1'] = sec1
            arbitrage_test['Sec2'] = sec2
            arbitrage_test['Cointegration'] = arbitrage_res['Cointegration']
            arbitrage_test['Std crossed'] = arbitrage_res['Std crossed']
            arbitrage_test['Std2 crossed'] = arbitrage_res['Std2 crossed']
            arbitrage_test['Signal Now'] = arbitrage_res['Signal']
            df = pd.DataFrame(arbitrage_test, index=[cur_iter])
            scoring_table.append(df, sort=False)
            cur_iter += 1
            pprint(arbitrage_test)
            print('%s/%s'%(cur_iter,total_arbitrage_pairs))        
scoring_table.to_csv('scoring_table.csv')