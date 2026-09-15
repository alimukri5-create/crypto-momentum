"""
Cost-drag comparison across rebalance cadence and rank buffer.
Synthetic prices, so RETURNS are meaningless — but turnover and the cost
it implies are structural and carry over to real data.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
import engine as cm
from synthetic import make_panel

close, qvol = make_panel(n_days=1100, n_sym=40, seed=7)
years = len(close) / 365.0

rows = []
for cad_name, days in [("weekly (Mon)", (0,)),
                       ("twice weekly (Mon/Thu)", (0, 3)),
                       ("daily", tuple(range(7)))]:
    for buf in (1.0, 1.5, 2.0, 3.0):
        cfg = cm.Config(rebalance_days=days, hold_buffer=buf)
        sig = cm.compute_signals(close, qvol, cfg)
        w = cm.build_weights(sig, close, cfg)
        bt = cm.run_backtest(close, w, cfg)
        rows.append({
            "cadence": cad_name,
            "buffer": buf,
            "turnover_pa": bt["turnover"].sum() / years,
            "cost_drag_pa_%": bt["cost"].sum() / years * 100,
            "avg_hold_days": (bt["n_held"].sum() / max((w.diff().abs().sum(axis=1) > 0).sum(), 1)),
        })

df = pd.DataFrame(rows)
piv_cost = df.pivot(index="cadence", columns="buffer", values="cost_drag_pa_%").round(2)
piv_to = df.pivot(index="cadence", columns="buffer", values="turnover_pa").round(1)

order = ["weekly (Mon)", "twice weekly (Mon/Thu)", "daily"]
print("=" * 72)
print("ANNUAL COST DRAG (%) at 0.15% per side — lower is better")
print("=" * 72)
print(piv_cost.loc[order].to_string())
print()
print("=" * 72)
print("ANNUAL TURNOVER (x portfolio value)")
print("=" * 72)
print(piv_to.loc[order].to_string())
print()
print("Columns are the rank buffer (1.0 = no buffer, 2.0 = hold until a name")
print("drops out of the top 10 when holding 5).")
