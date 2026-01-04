"""
signals.py - alpha signal generation

Implements:
1. Time-Series Momentum (TSMOM)
2. Cross-Sectional Momentum (XSMOM)  
3. Mean Reversion (Z-score reversal)
"""

import pandas as pd
import numpy as np


def tsmom(px, lookback=24):
    """
    Time-Series Momentum
    Signal = sign(price_t - price_{t-N}) or continuous version
    
    Using continuous version: ret over lookback period
    """
    ret_lookback = px / px.shift(lookback) - 1
    # continuous signal (not just sign)
    signal = ret_lookback
    return signal


def xsmom(ret, lookback=24):
    """
    Cross-Sectional Momentum (relative strength)
    
    1. Compute N-period return for all assets
    2. Rank cross-sectionally
    3. Signal = Rank - Mean(Rank)
    4. Demean so sum(weights) == 0 (dollar neutral)
    """
    # rolling return over lookback
    cum_ret = (1 + ret).rolling(lookback, min_periods=1).apply(lambda x: x.prod() - 1, raw=True)
    
    # rank cross-sectionally (1 is worst, N is best)
    ranked = cum_ret.rank(axis=1)
    
    # demean: subtract mean rank --> dollar neutral
    demeaned = ranked.subtract(ranked.mean(axis=1), axis=0)
    
    return demeaned


def mean_reversion(px, lookback=24):
    """
    Mean Reversion (Z-score reversal)
    Signal = -1 * (Price - MA) / Sigma
    
    Negative because we bet on reversion TO the mean
    """
    ma = px.rolling(lookback, min_periods=1).mean()
    sigma = px.rolling(lookback, min_periods=1).std()
    
    # z-score of deviation, negated for reversion
    #! good performing entities are assumed to be overvalued and will revert to the mean and vice versa for bad performing entities
    zscore = (px - ma) / sigma
    signal = -1 * zscore
    
    return signal


def normalize_signal(signal):
    """
    Cross-sectionally demean (sum(weights) = 0) and scale to sum(|weights|) = 1
    """
    # demean 
    demeaned = signal.subtract(signal.mean(axis=1), axis=0)
    # scale to fully invested (abs sum = 1)
    normalized = demeaned.divide(demeaned.abs().sum(axis=1), axis=0)
    return normalized


def combine_signals(signals, weights=None):
    """
    Combine multiple signal DataFrames into one.
    Simple weighted average, default equal weight.
    
    signals: list of DataFrames
    weights: list of floats (must sum to 1)
    """
    if weights is None:
        weights = [1.0 / len(signals)] * len(signals)
    
    combined = signals[0] * 0  # init with zeros, same shape
    
    for sig, w in zip(signals, weights):
        combined = combined.add(sig * w, fill_value=0)
    
    return combined


def compute_all_signals(px, ret, params=None):
    """
    Convenience function to compute all signals at once.
    
    params: dict with lookbacks for each signal
    Returns: dict of signal DataFrames
    """
    if params is None:
        params = {
            'tsmom_lb': 168, # 7 days = 168 hours
            'xsmom_lb': 72, # 3 days
            'meanrev_lb': 24, # 1 day
        }
    
    signals = {}
    signals['tsmom'] = normalize_signal(tsmom(px, params['tsmom_lb']))
    signals['xsmom'] = normalize_signal(xsmom(ret, params['xsmom_lb']))
    signals['meanrev'] = normalize_signal(mean_reversion(px, params['meanrev_lb']))
    
    return signals


if __name__ == '__main__':

    # quick test with fake data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=500, freq='1h')
    symbols = ['BTC', 'ETH', 'SOL']
    
    px = pd.DataFrame(
        np.random.randn(500, 3).cumsum(axis=0) + 100,
        index=dates, columns=symbols
    )
    ret = px.pct_change()
    
    sigs = compute_all_signals(px, ret)
    for name, sig in sigs.items():
        print(f"\n{name}:")
        print(sig.tail())
        print(f"sum of weights:\n {sig.sum(axis=1).tail()}")

