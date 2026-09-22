"""
Tests for engine.split_report on data where the answer is known by
construction. The point of the split test is to CATCH a strategy whose edge
lives in one period; a split test that cannot detect a planted one-period
effect is worse than useless, because it would bless exactly the failure it
exists to find.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import engine as eng


def make_case(edge_h1_daily, edge_h2_daily, n=1460, seed=0):
    """
    Build (bt, close) where the equal-weight control is a fixed noise path and
    the strategy is that same path plus a known daily edge, different in each
    half. Using the SAME noise means the edge is exactly what we planted, with
    no sampling difference between strategy and control to muddy it.
    """
    r = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    base = r.normal(0.0008, 0.03, n)

    half = n // 2
    add = np.concatenate([np.full(half, edge_h1_daily),
                          np.full(n - half, edge_h2_daily)])
    strat = base + add

    # close: two synthetic names whose mean return IS the control path, plus
    # BTC so benchmarks() has something to report.
    close = pd.DataFrame({
        "BTCUSDT": 100 * np.exp(np.cumsum(r.normal(0.0005, 0.025, n))),
        "AAAUSDT": 100 * np.cumprod(1 + base),
        "BBBUSDT": 100 * np.cumprod(1 + base),
    }, index=idx)
    bt = pd.DataFrame({"net_ret": strat}, index=idx)
    return bt, close


print("=" * 72)
print("TEST 1: equal-weight control is reconstructed exactly")
bt, close = make_case(0.0, 0.0, seed=1)
ctrl = eng.benchmarks(close)["equal_weight_universe"]
# AAA and BBB are identical paths, BTC is a third -> mean of 3 names.
manual = close.pct_change().fillna(0.0).mean(axis=1)
assert np.allclose(ctrl.values, manual.values), "control is not the universe mean"
print("  control == mean of per-name daily returns")
print("  PASS")

print("=" * 72)
print("TEST 2: PLANTED ONE-PERIOD EFFECT must be caught")
# strong first half, nothing in the second
bt, close = make_case(0.004, -0.0002, seed=2)
rep = eng.split_report(bt, close, n_splits=2)
print(rep["periods"][["period", "strategy_CAGR_%", "equal_weight_CAGR_%",
                      "edge_pp"]].round(2).to_string(index=False))
print(f"  verdict: {rep['verdict']}")
assert rep["verdict"] == "ONE-PERIOD EFFECT", rep["verdict"]
assert rep["edges"][0] > 0 and rep["edges"][1] <= 0
print("  PASS — detector finds an edge that lives in one half")

print("=" * 72)
print("TEST 3: GENUINELY STABLE edge must NOT be flagged")
bt, close = make_case(0.0020, 0.0018, seed=3)
rep = eng.split_report(bt, close, n_splits=2)
print(rep["periods"][["period", "strategy_CAGR_%", "equal_weight_CAGR_%",
                      "edge_pp"]].round(2).to_string(index=False))
print(f"  verdict: {rep['verdict']}")
assert rep["verdict"] == "HOLDS IN EVERY PERIOD", rep["verdict"]
print("  PASS — no false alarm on a stable edge")

print("=" * 72)
print("TEST 4: LOPSIDED but positive edge gets the middle verdict")
bt, close = make_case(0.0060, 0.0012, seed=4)
rep = eng.split_report(bt, close, n_splits=2)
e = rep["edges"]
print(f"  edges: {[round(x,1) for x in e]}  ratio {min(e)/max(e):.3f}")
print(f"  verdict: {rep['verdict']}")
assert rep["verdict"] == "HOLDS, BUT CONCENTRATED", rep["verdict"]
assert 0 < min(e) < 0.25 * max(e)
print("  PASS")

print("=" * 72)
print("TEST 5: edge_pp is exactly strategy CAGR minus control CAGR")
bt, close = make_case(0.001, 0.002, seed=5)
rep = eng.split_report(bt, close, n_splits=2)
for _, row in rep["periods"].iterrows():
    want = row["strategy_CAGR_%"] - row["equal_weight_CAGR_%"]
    assert abs(want - row["edge_pp"]) < 1e-9
print("  arithmetic checks out on every period")
print("  PASS")

print("=" * 72)
print("TEST 6: periods are contiguous, non-overlapping, and cover everything")
bt, close = make_case(0.001, 0.001, n=1461, seed=6)   # odd length on purpose
rep = eng.split_report(bt, close, n_splits=3)
p = rep["periods"]
assert len(p) == 3
assert p["days"].sum() == len(bt), f"{p['days'].sum()} != {len(bt)}"
for i in range(len(p) - 1):
    assert p["to"].iloc[i] < p["from"].iloc[i + 1], "periods overlap"
print(f"  3 periods, {p['days'].tolist()} days, no overlap, sums to {len(bt)}")
print("  PASS")

print("=" * 72)
print("TEST 7: yearly table lines up with the data")
bt, close = make_case(0.001, 0.001, n=1460, seed=7)
rep = eng.split_report(bt, close, n_splits=2)
y = rep["yearly"]
print(y.round(1).to_string(index=False))
assert y["days"].sum() == len(bt)
assert list(y["year"]) == sorted(y["year"]), "years out of order"
print("  PASS")

print("=" * 72)
print("TEST 8: NULL — no edge at all must not be called a win")
bt, close = make_case(0.0, 0.0, seed=8)
rep = eng.split_report(bt, close, n_splits=2)
print(f"  edges: {[round(x,2) for x in rep['edges']]}  verdict: {rep['verdict']}")
assert rep["verdict"] != "HOLDS IN EVERY PERIOD", "blessed a strategy with no edge"
print("  PASS")

print("=" * 72)
print("ALL SPLIT TESTS PASSED")
print("The split test catches a planted one-period effect, stays quiet on a")
print("stable edge, and refuses to bless no edge at all. Whether the REAL")
print("strategy passes is a separate question — only live data answers that.")
