"""Synthetic price panels for logic tests. No network required."""
import numpy as np
import pandas as pd


def make_panel(n_days: int = 900, n_sym: int = 40, seed: int = 42):
    """Geometric random walks. Meaningless returns, valid structure."""
    r = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days, freq="D")
    syms = ["BTCUSDT"] + [f"SYM{i:02d}USDT" for i in range(n_sym - 1)]
    close, qvol = {}, {}
    for s in syms:
        drift = r.normal(0.0008, 0.0006)
        vol = r.uniform(0.02, 0.07)
        close[s] = 100 * np.exp(np.cumsum(r.normal(drift, vol, n_days)))
        qvol[s] = np.abs(r.normal(50e6, 15e6, n_days))
    return pd.DataFrame(close, index=dates), pd.DataFrame(qvol, index=dates)
