# Stat Arb Crypto

Statistical arbitrage strategies for cryptocurrency trading.

## Setup

```bash
pip install -r Req/requirements.txt
```

## Usage

Run `analysis.ipynb` - this is the main notebook that loads data, generates signals, runs optimization, and backtests everything.

## Files

- `data_loader.py` - fetches hourly OHLCV from Binance, caches to parquet
- `signals.py` - TSMOM, XSMOM, mean reversion signal generation
- `optimization.py` - general weighting (Sigma^-1 * mu) with vol targeting
- `backtest.py` - vectorized backtester with tcost handling
- `analysis.ipynb` - main entry point, runs everything and plots results

## Strategies

1. **Time-Series Momentum (TSMOM)** - trend following on individual assets
2. **Cross-Sectional Momentum (XSMOM)** - relative strength ranking
3. **Volatility-Adjusted Momentum** - risk-adjusted (Sharpe-like) momentum
4. **Short-Term Reversal** - overnight reversal effect
5. **Correlation-Based Pairs Trading** *(in progress)* - residual reversion against correlated baskets

Signals are combined (85% momentum, 15% reversal) and weighted using Markowitz optimization with volatility targeting.

