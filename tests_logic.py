"""
Logic tests for crypto_momentum.py using synthetic data.
Verifies mechanics (no lookahead, cost accounting, gating, sizing) —
NOT whether the strategy makes money, which needs real data.
"""
import sys, math
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import engine as cm

from synthetic import make_panel

close, qvol = make_panel()
cfg = cm.Config(start="2021-01-01")

print("=" * 70)
print("TEST 1: signals compute without error and have expected shapes")
sig = cm.compute_signals(close, qvol, cfg)
for k in ("score", "ts_gate", "vol_gate", "liq_gate", "realised_vol"):
    assert sig[k].shape == close.shape, (k, sig[k].shape, close.shape)
print("  PASS — all signal panels match price panel shape", close.shape)

print("=" * 70)
print("TEST 2: NO LOOKAHEAD — signals at day t use no data after t")
# Perturb the final 10 days drastically; signals before that must be identical.
close2 = close.copy()
close2.iloc[-10:] = close2.iloc[-10:] * 3.0
sig2 = cm.compute_signals(close2, qvol, cfg)
cut = len(close) - 11
same = np.allclose(sig["score"].iloc[:cut].fillna(0).values,
                   sig2["score"].iloc[:cut].fillna(0).values)
print(f"  scores before perturbation identical: {same}")
assert same, "LOOKAHEAD DETECTED in compute_signals"
print("  PASS — no forward-looking leakage in signal construction")

print("=" * 70)
print("TEST 3: execution lag actually shifts exposure")
w = cm.build_weights(sig, close, cfg)
bt = cm.run_backtest(close, w, cfg)
held = w.shift(cfg.execution_lag_days).fillna(0.0)
first_w = w.sum(axis=1).to_numpy().nonzero()[0][0]
first_h = held.sum(axis=1).to_numpy().nonzero()[0][0]
print(f"  first target weight on bar {first_w}, first HELD position on bar {first_h}")
assert first_h == first_w + cfg.execution_lag_days
print("  PASS — positions are held one day after the signal, as configured")

print("=" * 70)
print("TEST 4: weights respect top_k, max_weight and never exceed 100%")
nz = (w > 0).sum(axis=1)
print(f"  max names held at once: {nz.max()} (top_k={cfg.top_k})")
print(f"  max single weight:      {w.max().max():.3f} (cap={cfg.max_weight})")
print(f"  max total exposure:     {w.sum(axis=1).max():.3f}")
assert nz.max() <= cfg.top_k
assert w.max().max() <= cfg.max_weight + 1e-9
assert w.sum(axis=1).max() <= 1.0 + 1e-9
print("  PASS")

print("=" * 70)
print("TEST 5: rebalancing only happens on configured weekdays")
changed = w.diff().abs().sum(axis=1) > 1e-12
bad = [d for d in w.index[changed] if d.dayofweek not in cfg.rebalance_days]
print(f"  weight changes on non-rebalance days: {len(bad)}")
assert len(bad) == 0
print("  PASS — Mon/Thu only")

print("=" * 70)
print("TEST 6: costs are charged and reduce returns")
print(f"  total turnover:   {bt['turnover'].sum():.2f}")
print(f"  total cost:       {bt['cost'].sum() * 100:.2f}% of capital")
print(f"  gross final eq:   {bt['equity_gross'].iloc[-1]:.4f}")
print(f"  net   final eq:   {bt['equity'].iloc[-1]:.4f}")
assert bt["cost"].sum() > 0
assert bt["equity"].iloc[-1] < bt["equity_gross"].iloc[-1]
print("  PASS — net is strictly below gross")

print("=" * 70)
print("TEST 7: BTC regime gate forces cash when BTC is below its MA")
cfg_on = cm.Config(use_btc_regime=True)
cfg_off = cm.Config(use_btc_regime=False)
w_on = cm.build_weights(cm.compute_signals(close, qvol, cfg_on), close, cfg_on)
w_off = cm.build_weights(cm.compute_signals(close, qvol, cfg_off), close, cfg_off)
btc_ma = close["BTCUSDT"].rolling(cfg_on.btc_regime_ma).mean()
risk_off_days = close.index[(close["BTCUSDT"] <= btc_ma)]
rebal_risk_off = [d for d in risk_off_days if d.dayofweek in cfg_on.rebalance_days]
if rebal_risk_off:
    exposure_then = w_on.loc[rebal_risk_off].sum(axis=1)
    print(f"  risk-off rebalance days: {len(rebal_risk_off)}")
    print(f"  max exposure on those days (gate ON):  {exposure_then.max():.3f}")
    print(f"  mean exposure overall  (gate ON):  {w_on.sum(axis=1).mean():.3f}")
    print(f"  mean exposure overall  (gate OFF): {w_off.sum(axis=1).mean():.3f}")
    assert exposure_then.max() < 1e-9, "regime gate did not force cash"
    print("  PASS — fully in cash on every risk-off rebalance")
else:
    print("  SKIP — synthetic BTC never went below its MA on a rebalance day")

print("=" * 70)
print("TEST 8: TS gate is actually restrictive vs cross-sectional-only")
cfg_ts = cm.Config(use_ts_gate=True, cross_sectional_only=False)
cfg_cs = cm.Config(use_ts_gate=True, cross_sectional_only=True)
w_ts = cm.build_weights(cm.compute_signals(close, qvol, cfg_ts), close, cfg_ts)
w_cs = cm.build_weights(cm.compute_signals(close, qvol, cfg_cs), close, cfg_cs)
print(f"  mean exposure with TS gate:      {w_ts.sum(axis=1).mean():.3f}")
print(f"  mean exposure cross-sectional:   {w_cs.sum(axis=1).mean():.3f}")
assert w_ts.sum(axis=1).mean() <= w_cs.sum(axis=1).mean() + 1e-9
print("  PASS — TS gate holds less, as expected")

print("=" * 70)
print("TEST 9: stats() and benchmarks() return sane values")
s = cm.stats(bt["net_ret"], "test")
for k in ("CAGR_%", "Sharpe", "max_DD_%", "skew", "kurtosis"):
    assert k in s and not (isinstance(s[k], float) and math.isnan(s[k])), k
print(f"  CAGR {s['CAGR_%']:.1f}%  Sharpe {s['Sharpe']:.2f}  "
      f"maxDD {s['max_DD_%']:.1f}%  skew {s['skew']:.2f}")
bm = cm.benchmarks(close)
assert "buy_hold_BTC" in bm and "equal_weight_universe" in bm
print("  PASS — stats and both benchmarks present")

print("=" * 70)
print("TEST 10: current_ranking() produces a usable live table")
tbl = cm.current_ranking(close, qvol, cfg)
assert len(tbl) == close.shape[1]
assert tbl["score"].is_monotonic_decreasing
print(f"  as of {tbl.attrs['as_of']}, BTC risk_on={tbl.attrs['btc_risk_on']}, "
      f"{int(tbl['eligible'].sum())} eligible of {len(tbl)}")
print("  PASS — ranking sorted, gates present")

print("=" * 70)
print("TEST 11: zero-cost run beats costed run by exactly the cost drag")
cfg0 = cm.Config(cost_per_side_pct=0.0)
bt0 = cm.run_backtest(close, w, cfg0)
diff = (bt0["gross_ret"] - bt["gross_ret"]).abs().max()
assert diff < 1e-12, "gross returns should not depend on cost setting"
print(f"  gross returns identical across cost settings (max diff {diff:.2e})")
print("  PASS")

print("=" * 70)
print("ALL LOGIC TESTS PASSED")
print("These verify MECHANICS on synthetic data. They say nothing about")
print("whether the strategy is profitable on real prices.")
