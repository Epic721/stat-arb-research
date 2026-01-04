"""
data_loader.py - fetch and cache binance hourly data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import os

# try ccxt first, fall back to python-binance
try:
    import ccxt
    USE_CCXT = True
except ImportError:
    from binance.client import Client as BinanceClient
    USE_CCXT = False

#! CCXT ISN'T WORKING; NVM BINANCE IS RESTRICTED (rip) SO HAD TO USE VPN (DATA CACHED NOW)
#! ERROR:
#! Service unavailable from a restricted location according to 'b. Eligibility' in https://www.binance.com/en/terms. Please contact customer service if you believe you received this message in error.
# from binance.client import Client as BinanceClient
# USE_CCXT = False

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _ccxt_fetch_ohlcv(symbol, timeframe='1h', since='2022-01-01'):
    """fetch using ccxt"""
    exchange = ccxt.binance()
    since_ts = exchange.parse8601(f"{since}T00:00:00Z")
    
    all_data = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=1000)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since_ts = ohlcv[-1][0] + 1  # next ms after last candle
        if len(ohlcv) < 1000:
            break
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    return df


def _binance_fetch_ohlcv(symbol, interval='1h', start='2022-01-01'):
    """fetch using python-binance"""
    client = BinanceClient()
    
    klines = client.get_historical_klines(symbol, interval, start)
    
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
               'quote_volume', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore']
    
    df = pd.DataFrame(klines, columns=columns)
    df['open_time'] = df['open_time'].apply(lambda x: datetime.utcfromtimestamp(x/1000))
    df = df.set_index('open_time')
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


def fetch_single(symbol, start='2022-01-01', timeframe='1h'):
    """fetch ohlcv for a single symbol"""
    if USE_CCXT:
        return _ccxt_fetch_ohlcv(symbol, timeframe, start)
    else:
        return _binance_fetch_ohlcv(symbol, timeframe, start)


def load_universe(symbols, start='2022-01-01', refresh=False):
    """
    Load hourly close prices for all symbols.
    Caches to parquet so we don't re-fetch every time.
    
    Returns: DataFrame with datetime index and symbol columns
    """
    cache_file = DATA_DIR / f"universe_{'_'.join(symbols[:3])}_etc.parquet"
    
    if cache_file.exists() and not refresh:
        print(f"loading cached data from {cache_file}")
        px = pd.read_parquet(cache_file)
        return px
    
    print("fetching fresh data from binance...")
    close_data = {}
    
    for sym in symbols:
        print(f"  {sym}...", end=" ")
        try:
            df = fetch_single(sym, start=start)
            close_data[sym] = df['close']
            print("ok")
        except Exception as e:
            print(f"failed: {e}")
    
    px = pd.DataFrame(close_data)
    
    # reindex to regular hourly grid, ffill gaps
    full_idx = pd.date_range(px.index.min(), px.index.max(), freq='1h')
    px = px.reindex(full_idx)
    px = px.ffill()
    
    # cache it
    px.to_parquet(cache_file)
    print(f"cached to {cache_file}")
    
    return px


def get_returns(px):
    """simple pct returns"""
    return px.pct_change()


# default universe
UNIVERSE = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'TRXUSDT', 'DOTUSDT',
    'LINKUSDT', 'MATICUSDT', 'LTCUSDT', 'BCHUSDT', 'ATOMUSDT', 'UNIUSDT'
]


if __name__ == '__main__':
    # quick test
    px = load_universe(UNIVERSE, refresh=True)
    print(px.tail())
    print(f"shape: {px.shape}")

