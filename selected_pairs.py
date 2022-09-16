marketcap_pairs = [
'BTC/USDT',
'ETH/USDT',
'ETC/USDT',
'ETH/BTC',
'LTC/ETH',
'LTC/BTC',
'LTC/USDT',
'EOS/USDT',
'EOS/BTC',
'BCH/USDT',
'BCH/BTC',
'XRP/USDT',
'XRP/ETH',
'XRP/BTC',
'TRX/USDT',
'ETC/TRX',
'QTUM/TRX',
'BSV/USDT',
'BSV/BTC',
'BSV/USDT',
'XLM/USDT',
'XLM/BTC',
'XLM/KRW',
'ADA/BNB',
'ARDR/BNB',
'BNB/USDT'
'XLM/USDT',
'ADA/USDT',
'ADA/BTC'
'XMR/USDT'
'XMR/BTC'
'LEO/USDT'
'LINK/USDT'
'NEO/USDT'
]
selected_coins = [
'BTC',
'USDT',
'ETH',
'ETC'
'LTC',
'BCH',
'EOS',
'XRP',
'TRX',
'QTUM',
'BSV',
'XLM',
'ADA',
'XMR',
'LEO',
'LINK',
'NEO'
]
selected_arb = [
['ADA/BNB','ARDR/BNB']
]
#TODO: ранжирование / фильтрация по спреду бида и аска - для данной стратегии это важно, т.к. небольшие, но верные движения цены
def construct_selected_pairs(exchange_pairs):
    all_pairs = []
    real_pairs = []
    for coin1 in selected_coins:
        for coin2 in selected_coins:
            if coin1 == coin2:
                continue
            all_pairs.append('%s/%s'%(coin1,coin2))
    for pair in all_pairs:
        if pair in exchange_pairs:
            real_pairs.append(pair)
    return real_pairs