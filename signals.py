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


def short_term_reversal(ret, lookback=24):
    """
    Short-term reversal (overnight effect).
    Bet against recent winners on very short timeframes.
    
    This is different from long-term mean reversion - it captures
    the well-documented short-term reversal effect.
    """
    # recent cumulative return
    cum_ret = ret.rolling(lookback, min_periods=1).sum()
    # bet against it
    signal = -1 * cum_ret
    return signal


def vol_adjusted_momentum(ret, lookback=168, vol_lookback=72):
    """
    Volatility-adjusted momentum (Sharpe-like signal).
    
    Ranks assets by their risk-adjusted returns rather than raw returns.
    Assets with high return AND low volatility get stronger signals.
    """
    # rolling return
    cum_ret = ret.rolling(lookback, min_periods=1).sum()
    # rolling volatility
    vol = ret.rolling(vol_lookback, min_periods=1).std()
    # sharpe-like ratio (avoid div by zero)
    vol = vol.replace(0, np.nan)
    risk_adj_ret = cum_ret / vol
    return risk_adj_ret.fillna(0)


def normalize_signal(signal):
    """
    Cross-sectionally demean (sum(weights) = 0) and scale to sum(|weights|) = 1
    """
    # demean 
    demeaned = signal.subtract(signal.mean(axis=1), axis=0)
    # scale to fully invested (abs sum = 1)
    normalized = demeaned.divide(demeaned.abs().sum(axis=1), axis=0)
    return normalized


def smooth_signal(signal, halflife=24):
    """
    Apply exponential smoothing to signal to reduce noise and turnover.
    halflife: number of periods for signal to decay by half (in hours)
    """
    return signal.ewm(halflife=halflife, adjust=False).mean()


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


def compute_all_signals(px, ret, params=None, signal_smooth_halflife=48):
    """
    Convenience function to compute all signals at once.
    
    params: dict with lookbacks for each signal
    signal_smooth_halflife: EMA halflife for signal smoothing (hours), 0 to disable
    Returns: dict of signal DataFrames
    """
    if params is None:
        # longer lookbacks = slower-moving signals = less turnover
        params = {
            'tsmom_lb': 336,   # 14 days (was 7)
            'xsmom_lb': 168,   # 7 days (was 3)
            'meanrev_lb': 48,  # 2 days (was 1)
        }
    
    signals = {}
    signals['tsmom'] = normalize_signal(tsmom(px, params['tsmom_lb']))
    signals['xsmom'] = normalize_signal(xsmom(ret, params['xsmom_lb']))
    signals['meanrev'] = normalize_signal(mean_reversion(px, params['meanrev_lb']))
    
    # apply smoothing to reduce signal noise
    if signal_smooth_halflife > 0:
        for name in signals:
            signals[name] = smooth_signal(signals[name], halflife=signal_smooth_halflife)
            # re-normalize after smoothing
            signals[name] = normalize_signal(signals[name])
    
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

