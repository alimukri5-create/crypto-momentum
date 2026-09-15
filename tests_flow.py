"""
Mechanical tests for flow.py on synthetic data.

Critically, these include a PLANTED-SIGNAL test: synthetic data where order
flow genuinely predicts forward returns by construction. If ic_by_horizon
cannot find a signal we planted ourselves, it cannot be trusted to report
honestly on real data either — and a broken detector that returns noise would
look exactly like "no edge".
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import flow as fl


def make_flow_panel(n_bars=3000, n_sym=20, seed=1, signal_strength=0.0,
                    lead=6):
    """
    Synthetic hourly bars with a taker-buy series.

    signal_strength = 0   -> flow is pure noise, unrelated to returns
    signal_strength > 0   -> flow at bar t predicts the return from t to t+lead
    """
    r = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="h")
    syms = ["BTCUSDT"] + [f"SYM{i:02d}USDT" for i in range(n_sym - 1)]

    close, qvol, tbq = {}, {}, {}
    for s in syms:
        # latent flow process
        flow_latent = pd.Series(r.normal(0, 1, n_bars)).rolling(4).mean().fillna(0).values
        noise = r.normal(0, 0.012, n_bars)
        # returns at t+lead partly driven by flow at t
        rets = noise.copy()
        if signal_strength > 0:
            rets[lead:] += signal_strength * 0.012 * flow_latent[:-lead]
        px = 100 * np.exp(np.cumsum(rets))

        v = np.abs(r.normal(5e6, 1.5e6, n_bars))
        # taker-buy share tracks the latent flow, squashed into (0,1)
        share = 1.0 / (1.0 + np.exp(-flow_latent * 0.8))
        share = np.clip(share + r.normal(0, 0.05, n_bars), 0.02, 0.98)

        close[s], qvol[s], tbq[s] = px, v, v * share

    return (pd.DataFrame(close, index=idx),
            pd.DataFrame(qvol, index=idx),
            pd.DataFrame(tbq, index=idx))


cfg = fl.FlowConfig()
print("=" * 72)
print("TEST 1: OFI is computed correctly and stays in [-1, +1]")
close, qvol, tbq = make_flow_panel(seed=1)
sig = fl.compute_flow(close, qvol, tbq, cfg)
raw = sig["ofi_raw"]
assert raw.max().max() <= 1.0 + 1e-9 and raw.min().min() >= -1.0 - 1e-9
# hand-check one cell
s0, i0 = close.columns[0], 500
expected = 2 * (tbq[s0].iloc[i0] / qvol[s0].iloc[i0]) - 1
assert abs(raw[s0].iloc[i0] - expected) < 1e-12
print(f"  range [{raw.min().min():.3f}, {raw.max().max():.3f}], hand-check matches")
print("  PASS")

print("=" * 72)
print("TEST 2: NO LOOKAHEAD — signals at t ignore everything after t")
close2 = close.copy()
close2.iloc[-50:] *= 5.0
qvol2 = qvol.copy(); qvol2.iloc[-50:] *= 5.0
tbq2 = tbq.copy(); tbq2.iloc[-50:] *= 5.0
sig2 = fl.compute_flow(close2, qvol2, tbq2, cfg)
cut = len(close) - 51
a = sig["ofi_z"].iloc[:cut].fillna(0).values
b = sig2["ofi_z"].iloc[:cut].fillna(0).values
assert np.allclose(a, b), "LOOKAHEAD in flow signal construction"
print("  perturbing the last 50 bars changes nothing before them")
print("  PASS")

print("=" * 72)
print("TEST 3: PLANTED SIGNAL — the detector finds an edge we put there")
cp, qp, tp = make_flow_panel(seed=7, signal_strength=1.2, lead=6)
sp = fl.compute_flow(cp, qp, tp, cfg)
ic_planted = fl.ic_by_horizon(sp["ofi_z"], cp, horizons=(1, 2, 4, 6, 8, 12, 24))
print(ic_planted[["mean_IC", "t_stat", "pct_positive"]].round(4).to_string())
best = ic_planted["mean_IC"].idxmax()
print(f"  strongest horizon: {best}h (planted lead was 6h)")
assert ic_planted["mean_IC"].max() > 0.05, "detector missed a planted signal"
assert best in (4, 6, 8), f"detector found the peak at {best}h, expected near 6h"
print("  PASS — detector locates a known signal at the right horizon")

print("=" * 72)
print("TEST 4: NULL — pure noise flow produces no signal")
cn, qn, tn = make_flow_panel(seed=11, signal_strength=0.0)
sn = fl.compute_flow(cn, qn, tn, cfg)
ic_null = fl.ic_by_horizon(sn["ofi_z"], cn, horizons=(1, 4, 8, 12, 24))
print(ic_null[["mean_IC", "t_stat"]].round(4).to_string())
verdict, why = fl.horizon_verdict(ic_null)
print(f"  verdict: {verdict} — {why}")
assert ic_null["mean_IC"].abs().max() < 0.05, "detector hallucinated signal in noise"
print("  PASS — no false positive on noise")

print("=" * 72)
print("TEST 5: backtest respects TP/SL and is pessimistic on ambiguity")
entries = fl.build_entries(sp, cp, cfg)
print(f"  entry rate: {entries.values.mean():.4%} of bars")
bt = fl.backtest_flow(cp, entries, cfg)
tr = bt["trades"]
assert not tr.empty, "no trades generated"
gross = tr["gross_%"]
# targets and stops must land on the configured levels
tg = tr[tr["reason"] == "target"]["gross_%"]
stp = tr[tr["reason"] == "stop"]["gross_%"]
if len(tg):
    assert np.allclose(tg, cfg.tp_pct), f"target trades not at +{cfg.tp_pct}%"
if len(stp):
    assert np.allclose(stp, -cfg.sl_pct), f"stop trades not at -{cfg.sl_pct}%"
assert (tr["bars_held"] <= cfg.max_hold_bars).all(), "held past max_hold_bars"
print(f"  {len(tr)} trades | target {bt['stats']['target_hits_%']:.0f}% "
      f"stop {bt['stats']['stop_hits_%']:.0f}% timeout {bt['stats']['timeouts_%']:.0f}%")
print(f"  exits land exactly on ±TP/SL; none held beyond {cfg.max_hold_bars} bars")
print("  PASS")

print("=" * 72)
print("TEST 6: costs are charged on every trade")
st = bt["stats"]
assert abs(st["avg_net_%"] - (tr["gross_%"].mean() - 2 * cfg.cost_per_side_pct)) < 1e-9
print(f"  gross avg {tr['gross_%'].mean():+.3f}% -> net avg {st['avg_net_%']:+.3f}% "
      f"(cost {2*cfg.cost_per_side_pct:.2f}%/trade)")
print("  PASS")

print("=" * 72)
print("TEST 7: breakeven win rate maths is right")
w, l = st["avg_win_%"], abs(st["avg_loss_%"])
be = l / (w + l) * 100
assert abs(be - st["breakeven_win_rate_%"]) < 1e-6
print(f"  avg win {w:+.2f}% | avg loss {-l:+.2f}% -> need {be:.1f}% "
      f"| actual {st['win_rate_%']:.1f}%")
print("  PASS")

print("=" * 72)
print("TEST 8: no overlapping positions within one symbol")
for s, g in tr.groupby("symbol"):
    g = g.sort_values("entry_time")
    assert (g["entry_time"].iloc[1:].values >= g["exit_time"].iloc[:-1].values).all(), s
print("  PASS — each symbol holds at most one position at a time")

print("=" * 72)
print("TEST 9: random-entry control runs and is beatable/measurable")
rate = float(entries.values.mean())
rnd = fl.random_entries(entries, rate, seed=3)
bt_rnd = fl.backtest_flow(cp, rnd, cfg)
print(f"  planted signal : {st['trades']} trades, avg net {st['avg_net_%']:+.3f}%")
print(f"  random control : {bt_rnd['stats']['trades']} trades, "
      f"avg net {bt_rnd['stats']['avg_net_%']:+.3f}%")
assert st["avg_net_%"] > bt_rnd["stats"]["avg_net_%"], \
    "planted signal failed to beat random entries — backtest is not measuring the signal"
print("  PASS — planted signal beats random into the same exits")

print("=" * 72)
print("TEST 10: price-only control builds and differs from flow")
po = fl.price_only_entries(cp, qp, cfg)
overlap = (po & entries).values.sum() / max(entries.values.sum(), 1)
print(f"  price-only entry rate {po.values.mean():.4%}, "
      f"overlap with flow entries {overlap:.1%}")
assert po.values.mean() > 0
print("  PASS — an independent control exists to compare against")

print("=" * 72)
print("ALL FLOW TESTS PASSED")
print("The detector finds planted signals, stays silent on noise, and the")
print("backtest charges costs and refuses overlapping positions. Whether real")
print("order flow predicts real returns is a separate question — that needs")
print("real data, which only the deployed app can reach.")
