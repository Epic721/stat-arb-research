"""
optimization.py - portfolio construction and weighting

Implements general risk-adjusted weighting scheme with vol targeting
w = scalar * (Sigma^-1 * mu)
"""

import pandas as pd
import numpy as np
from numpy.linalg import inv


def rolling_cov(ret, window=720):
    """
    Compute rolling covariance matrix.
    window: lookback in hours (720h = 30 days)
    
    Returns: dict mapping date --> cov matrix
    """
    cov_dict = {}
    
    for i in range(window, len(ret)):
        dt = ret.index[i]
        window_ret = ret.iloc[i-window:i]
        cov_dict[dt] = window_ret.cov()
    
    return cov_dict


def shrink_cov(sigma, shrinkage=0.1):
    """
    Ledoit-Wolf style shrinkage toward diagonal.
    Helps with matrix inversion stability.
    """
    n = sigma.shape[0]
    diag = np.diag(np.diag(sigma))
    shrunk = (1 - shrinkage) * sigma + shrinkage * diag
    return shrunk


def general_weights(sigma, mu, shrinkage=0.1):
    """
    Markowitz optimal weights: w = Sigma^-1 * mu
    
    sigma: covariance matrix (numpy array or DataFrame)
    mu: expected returns / signal vector
    shrinkage: regularization amount
    
    Returns normalized weights (abs sum = 1)
    """
    #! working in strategy space, so historical data to estimate covariance matrix is roughly sufficient
    if isinstance(sigma, pd.DataFrame):
        sigma = sigma.values
    if isinstance(mu, pd.Series):
        mu = mu.values
    
    # shrink for stability
    sigma_shrunk = shrink_cov(sigma, shrinkage)
    
    try:
        sigma_inv = inv(sigma_shrunk)
    except np.linalg.LinAlgError:
        # fallback to pseudo-inverse if singular
        sigma_inv = np.linalg.pinv(sigma_shrunk)
    
    raw_weights = sigma_inv @ mu
    
    # normalize so |w|.sum() = 1
    normalized = raw_weights / np.abs(raw_weights).sum()
    
    return normalized


def vol_target_scalar(weights, sigma, target_vol=0.15, annualize_factor=np.sqrt(24*365)):
    """
    Compute scalar to hit a target annualized volatility.
    
    portfolio_vol = sqrt(w^T * Sigma * w)
    target_vol = scalar * portfolio_vol
    scalar = target_vol / portfolio_vol
    """
    if isinstance(sigma, pd.DataFrame):
        sigma = sigma.values
    if isinstance(weights, pd.Series):
        weights = weights.values
    
    port_var = weights @ sigma @ weights #! IMPLICIT TRANSPOSITION
    port_vol = np.sqrt(port_var) * annualize_factor
    
    if port_vol < 1e-8:  # avoid div by zero
        return 1.0
    
    scalar = target_vol / port_vol
    return scalar


def compute_optimal_weights(ret, signal, cov_window=720, shrinkage=0.1, target_vol=0.20):
    """
    Main function: compute rolling optimal weights.
    
    ret: returns DataFrame
    signal: combined signal DataFrame (mu)
    cov_window: hours for rolling cov (720 = 30 days)
    shrinkage: cov matrix regularization
    target_vol: annualized vol target
    
    Returns: DataFrame of position weights
    """
    # align
    common_idx = ret.index.intersection(signal.index)
    ret = ret.loc[common_idx]
    signal = signal.loc[common_idx]
    
    weights_list = []
    
    for i in range(cov_window, len(ret)):
        dt = ret.index[i]
        
        # rolling cov
        window_ret = ret.iloc[i-cov_window:i]
        sigma = window_ret.cov()
        
        # signal at this time
        mu = signal.iloc[i]
        
        # skip if signal has NaNs
        if mu.isna().any():
            weights_list.append(pd.Series(0, index=ret.columns, name=dt))
            continue
        
        # compute weights
        w = general_weights(sigma, mu, shrinkage)
        
        # apply vol target
        #! basically adjusting leverage (lever up) to hit target volatility
        scalar = vol_target_scalar(w, sigma, target_vol)
        w_scaled = w * scalar
        
        weights_list.append(pd.Series(w_scaled, index=ret.columns, name=dt))
    
    weights_df = pd.DataFrame(weights_list)
    return weights_df


def eqvol_weights(sigma):
    """
    Equal volatility weights (inverse vol).
    Simple alternative to full optimization.
    """
    if isinstance(sigma, pd.DataFrame):
        sigma = sigma.values
    
    vols = np.sqrt(np.diag(sigma)) #! diag is var of return streams
    w = 1.0 / vols
    w = w / np.abs(w).sum()
    return w


if __name__ == '__main__':
    # quick test
    np.random.seed(42)
    
    # fake returns
    dates = pd.date_range('2022-01-01', periods=1000, freq='1h')
    symbols = ['BTC', 'ETH', 'SOL']
    ret = pd.DataFrame(
        np.random.randn(1000, 3) * 0.01,
        index=dates, columns=symbols
    )
    
    # fake signal
    signal = pd.DataFrame(
        np.random.randn(1000, 3),
        index=dates, columns=symbols
    )
    signal = signal.subtract(signal.mean(axis=1), axis=0)  # demean
    
    weights = compute_optimal_weights(ret, signal, cov_window=168)
    print("weights shape:", weights.shape)
    print(weights.tail())
    print("\nabs sum of weights:")
    print(weights.abs().sum(axis=1).tail())

