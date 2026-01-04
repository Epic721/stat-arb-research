"""
backtest.py - vectorized backtesting engine

Handles:
- Return computation
- Transaction costs
- Turnover tracking
- Performance metrics
"""

import pandas as pd
import numpy as np


class Backtester:
    """
    Vectorized backtest engine.
    
    Usage:
        bt = Backtester(returns, weights, tcost_bps=5)
        results = bt.run()
    """
    
    def __init__(self, ret, weights, tcost_bps=5):
        """
        ret: DataFrame of asset returns
        weights: DataFrame of position weights (lagged by 1 for trading)
        tcost_bps: transaction cost in basis points per trade
        """
        # align indices
        common_idx = ret.index.intersection(weights.index)
        self.ret = ret.loc[common_idx]
        self.weights = weights.loc[common_idx]
        self.tcost_bps = tcost_bps
        
        self.results = None
    
    def compute_turnover(self):
        """turnover = sum of |weight changes|"""
        weight_diff = self.weights.fillna(0) - self.weights.shift().fillna(0)
        turnover = weight_diff.abs().sum(axis=1)
        return turnover
    
    def run(self):
        """
        Run the backtest.
        Returns dict with gross/net returns, turnover, etc.
        """
        # weights are applied at t, we get return at t+1
        # so we shift weights back by 1
        position = self.weights.shift(1)
        
        # gross return: sum(weight * asset_return)
        gross_ret = (position * self.ret).sum(axis=1)
        
        # turnover
        turnover = self.compute_turnover()
        
        # transaction cost
        tcost = turnover * self.tcost_bps * 1e-4
        
        # net return
        net_ret = gross_ret - tcost
        
        self.results = {
            'gross_ret': gross_ret,
            'net_ret': net_ret,
            'turnover': turnover,
            'tcost': tcost,
            'weights': self.weights,
        }
        
        return self.results
    
    def get_equity_curve(self, use_net=True):
        """cumulative returns"""
        if self.results is None:
            self.run()
        
        ret = self.results['net_ret'] if use_net else self.results['gross_ret']
        return (1 + ret).cumprod()
    
    def get_log_equity(self, use_net=True):
        """cumulative log returns (additive)"""
        if self.results is None:
            self.run()
        
        ret = self.results['net_ret'] if use_net else self.results['gross_ret']
        return ret.cumsum()


def calc_sharpe(ret, annualize=np.sqrt(24*365)):
    """annualized sharpe ratio"""
    return ret.mean() / ret.std() * annualize


def calc_sortino(ret, annualize=np.sqrt(24*365)):
    """annualized sortino ratio (downside deviation)"""
    downside = ret[ret < 0]
    downside_std = downside.std()
    if downside_std < 1e-8:
        return np.nan
    return ret.mean() / downside_std * annualize


def calc_max_drawdown(equity):
    """max drawdown from equity curve"""
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return dd.min()


def calc_drawdown_series(equity):
    """drawdown series for plotting"""
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return dd


def get_stats(ret, annualize=np.sqrt(24*365)):
    """
    Compute standard performance stats.
    
    ret: Series of returns
    Returns: dict of stats
    """
    equity = (1 + ret).cumprod()
    
    stats = {}
    stats['sharpe'] = calc_sharpe(ret, annualize)
    stats['sortino'] = calc_sortino(ret, annualize)
    stats['ann_ret'] = ret.mean() * (annualize ** 2)  # mean * hours_per_year
    stats['ann_vol'] = ret.std() * annualize
    stats['max_dd'] = calc_max_drawdown(equity)
    stats['total_ret'] = equity.iloc[-1] - 1
    #! can add tvalues for alpha and beta if needed
    
    return stats


def print_stats(ret, name='Strategy'):
    """pretty print stats"""
    stats = get_stats(ret)
    print(f"\n{name} Performance:")
    print(f"  Sharpe:     {stats['sharpe']:.2f}")
    print(f"  Sortino:    {stats['sortino']:.2f}")
    print(f"  Ann Return: {stats['ann_ret']*100:.1f}%")
    print(f"  Ann Vol:    {stats['ann_vol']*100:.1f}%")
    print(f"  Max DD:     {stats['max_dd']*100:.1f}%")
    print(f"  Total Ret:  {stats['total_ret']*100:.1f}%")


if __name__ == '__main__':
    # quick test with fake data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=1000, freq='1h')
    symbols = ['BTC', 'ETH', 'SOL']
    
    ret = pd.DataFrame(
        np.random.randn(1000, 3) * 0.01,
        index=dates, columns=symbols
    )
    
    # random weights, normalized
    weights = pd.DataFrame(
        np.random.randn(1000, 3),
        index=dates, columns=symbols
    )
    weights = weights.divide(weights.abs().sum(axis=1), axis=0)
    
    bt = Backtester(ret, weights, tcost_bps=5)
    results = bt.run()
    
    print_stats(results['net_ret'], 'Test Strategy')
    print(f"\nAvg Turnover: {results['turnover'].mean():.2%}")

