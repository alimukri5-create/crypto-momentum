"""
Order-Flow Scalper — research page.

Separate from app.py on purpose. app.py is the thing Ali trades from and
should stay boring. This is the workbench where a signal gets proven or
killed BEFORE it earns a place there.

Run locally:  streamlit run scalper.py
Deploy:       a SECOND Streamlit app off this repo, main file scalper.py

The page is ordered the way the decision should be made:
  1. Does the signal predict anything, at what horizon?   (IC test)
  2. If yes, does a tradeable strategy on it make money after costs?
  3. Does it beat price-only and random controls doing the same thing?

Step 3 is the one that usually kills things, which is why it is not optional.
"""

import numpy as np
import pandas as pd
import streamlit as st

import data as dat
import flow as fl

st.set_page_config(page_title="Order-Flow Scalper", page_icon="🔬", layout="wide")

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Research settings")

with st.sidebar.expander("Data", expanded=True):
    interval = st.selectbox("Bar size", ["1h", "4h", "15m", "30m"], 0)
    months = st.slider("History (months)", 3, 36, 12,
                       help="Hourly bars over many months and many symbols "
                            "gets slow. Start small.")
    n_syms = st.slider("Symbols", 8, 40, 20)

with st.sidebar.expander("Signal", expanded=True):
    ofi_smooth = st.slider("OFI smoothing (bars)", 1, 24, 6)
    z_window = st.slider("Z-score baseline (bars)", 48, 720, 168, 24)
    min_z = st.slider("Entry threshold (z)", 0.0, 3.0, 1.0, 0.1)
    price_confirm = st.checkbox("Require price confirmation", True)
    min_rvol = st.slider("Min relative volume", 0.5, 3.0, 1.2, 0.1)

with st.sidebar.expander("Trade", expanded=True):
    tp = st.number_input("Take profit %", 0.5, 20.0, 3.0, 0.1)
    sl = st.number_input("Stop loss %", 0.2, 10.0, 2.0, 0.1)
    max_hold = st.slider("Max hold (bars)", 2, 72, 12)
    cost = st.number_input("Cost per side %", 0.0, 1.0, 0.15, 0.01)
    use_regime = st.checkbox("BTC regime gate", True)

cfg = fl.FlowConfig(
    ofi_smooth=int(ofi_smooth), z_window=int(z_window), min_ofi_z=float(min_z),
    require_price_confirm=price_confirm, min_rvol=float(min_rvol),
    use_btc_regime=use_regime, tp_pct=float(tp), sl_pct=float(sl),
    max_hold_bars=int(max_hold), cost_per_side_pct=float(cost),
)

BARS_PER_DAY = {"15m": 96, "30m": 48, "1h": 24, "4h": 6}

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

# See app.py: bump this whenever the return shape changes.
CACHE_VERSION = 2


@st.cache_data(show_spinner=False, ttl=3600)
def load(symbols, start, interval, cache_version=CACHE_VERSION):
    return dat.load_universe(symbols, start, interval=interval)

st.title("Order-Flow Scalper")
st.caption("Does aggressive-buy imbalance predict short-horizon returns? "
           "Proven or killed here before it reaches the trading app.")

start = (pd.Timestamp.utcnow().tz_localize(None)
         - pd.DateOffset(months=int(months))).strftime("%Y-%m-%d")
symbols = tuple(dat.UNIVERSE[:int(n_syms)])

try:
    with st.spinner(f"Fetching {len(symbols)} symbols of {interval} bars…"):
        _loaded = load(symbols, start, interval)
        if len(_loaded) < 4:
            st.error("Stale data cache from an older version. Reboot the app "
                     "(Manage app -> Reboot) to clear it.")
            st.stop()
        close, qvol, report, tbq = _loaded
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

if not report.get("has_flow"):
    st.error(
        "**No order-flow data available.** This venue is "
        f"`{report['venue']}`, and only Binance reports taker-buy volume. "
        "Coinbase does not break volume into buys and sells, so there is "
        "nothing to compute. Nothing on this page will work until the app "
        "can reach Binance."
    )
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Venue", report["venue"])
c2.metric("Symbols", len(report["loaded"]))
c3.metric("Bars", report["n_days"])
c4.metric("Through", str(report["end"])[:16])

sig = fl.compute_flow(close, qvol, tbq, cfg)

tab_ic, tab_bt, tab_ctrl, tab_sig = st.tabs(
    ["1 · Does it predict?", "2 · Does it make money?",
     "3 · Does it beat controls?", "Signal detail"])

# ==========================================================================
# 1 — THE TEST THAT DECIDES
# ==========================================================================

with tab_ic:
    st.subheader("Information coefficient by horizon")
    st.caption(
        "Rank correlation between the flow signal now and the return over the "
        "next N bars. No strategy, no parameters — nowhere for a curve fit to "
        "hide. If this is flat, nothing later can rescue it."
    )

    hz = (1, 2, 4, 6, 8, 12, 24, 48)
    with st.spinner("Measuring predictive power…"):
        ic = fl.ic_by_horizon(sig["ofi_z"], close, horizons=hz)

    if ic.empty:
        st.warning("Not enough overlapping data to measure.")
    else:
        verdict, why = fl.horizon_verdict(ic)
        colour = {"SIGNAL PRESENT": st.success,
                  "NOTHING HERE": st.error,
                  "TOO SMALL TO TRADE": st.warning,
                  "INVERTED": st.error,
                  "SUSPICIOUS": st.warning,
                  "NO DATA": st.info}.get(verdict, st.info)
        colour(f"**{verdict}** — {why}")

        bpd = BARS_PER_DAY.get(interval, 24)
        show = ic.copy()
        show["horizon_hours"] = [h * 24 / bpd for h in show.index]
        st.dataframe(
            show[["horizon_hours", "mean_IC", "t_stat", "t_stat_naive",
                  "pct_positive", "n_bars"]].round(4),
            use_container_width=True)
        st.bar_chart(ic["mean_IC"], use_container_width=True)

        st.caption(
            f"Bars to clear: |IC| ≥ {fl.MIN_USEFUL_IC} AND t-stat ≥ "
            f"{fl.MIN_T_STAT:g}. `t_stat` is widened by √h because "
            "overlapping forward windows make consecutive readings correlated; "
            "`t_stat_naive` is the uncorrected figure and is too generous. "
            f"An IC above {fl.TOO_GOOD_IC} on free public data almost "
            "certainly means a bug, not an edge."
        )

# ==========================================================================
# 2 — THE BACKTEST
# ==========================================================================

with tab_bt:
    entries = fl.build_entries(sig, close, cfg, dat.BENCHMARK)
    rate = float(entries.values.mean())
    st.caption(f"Signal fires on {rate:.3%} of symbol-bars.")

    with st.spinner("Running backtest…"):
        bt = fl.backtest_flow(close, entries, cfg)

    if bt["trades"].empty:
        st.warning("No trades. Loosen the entry threshold or relax the gates.")
    else:
        s = bt["stats"]
        m = st.columns(5)
        m[0].metric("Trades", s["trades"])
        m[1].metric("Win rate", f"{s['win_rate_%']:.1f}%",
                    f"need {s['breakeven_win_rate_%']:.1f}%")
        m[2].metric("Avg net/trade", f"{s['avg_net_%']:+.3f}%")
        m[3].metric("Profit factor", f"{s['profit_factor']:.2f}")
        m[4].metric("Avg hold", f"{s['avg_bars_held']:.1f} bars")

        if s["avg_net_%"] <= 0:
            st.error("**Loses money after costs.** The average trade is "
                     "negative — this is not tradeable as configured.")
        elif s["win_rate_%"] < s["breakeven_win_rate_%"]:
            st.error("**Win rate is below breakeven** for this target/stop "
                     "and cost combination.")
        else:
            st.success(f"Positive after costs: {s['avg_net_%']:+.3f}% per "
                       f"trade across {s['trades']} trades.")

        e = st.columns(4)
        e[0].metric("Target hits", f"{s['target_hits_%']:.0f}%")
        e[1].metric("Stops", f"{s['stop_hits_%']:.0f}%")
        e[2].metric("Timeouts", f"{s['timeouts_%']:.0f}%")
        e[3].metric("Total cost paid", f"{s['cost_drag_%']:.1f}%")

        if s["timeouts_%"] > 50:
            st.warning(
                f"{s['timeouts_%']:.0f}% of trades time out rather than "
                "reaching the target or stop. The target is too far away for "
                "this bar size and hold limit — the exits you designed are "
                "not the exits you are getting."
            )

        st.subheader("Equity curve (equal risk per trade)")
        st.line_chart(bt["equity"], use_container_width=True)

        st.subheader("Trades")
        st.dataframe(bt["trades"].tail(200), use_container_width=True, height=320)

# ==========================================================================
# 3 — CONTROLS
# ==========================================================================

with tab_ctrl:
    st.caption(
        "A strategy that makes money proves nothing until it beats these. "
        "**Price-only** runs the identical structure on price momentum instead "
        "of flow — if flow does not beat it, flow adds nothing. **Random** "
        "fires at the same rate on coin flips — if that also makes money, the "
        "exits are doing the work and the signal is decoration."
    )

    with st.spinner("Running controls…"):
        entries = fl.build_entries(sig, close, cfg, dat.BENCHMARK)
        rate = float(entries.values.mean())
        bt_flow = fl.backtest_flow(close, entries, cfg)
        bt_price = fl.backtest_flow(
            close, fl.price_only_entries(close, qvol, cfg, dat.BENCHMARK), cfg)
        rnd = [fl.backtest_flow(close, fl.random_entries(entries, rate, seed=k), cfg)
               for k in range(3)]

    rows = []
    for name, b in [("flow (OFI)", bt_flow), ("price momentum", bt_price)]:
        if b["stats"]:
            rows.append({"strategy": name, **{k: b["stats"][k] for k in
                        ("trades", "win_rate_%", "avg_net_%", "profit_factor",
                         "target_hits_%", "timeouts_%")}})
    rstats = [b["stats"] for b in rnd if b["stats"]]
    if rstats:
        rows.append({
            "strategy": "random (avg of 3)",
            "trades": int(np.mean([x["trades"] for x in rstats])),
            "win_rate_%": float(np.mean([x["win_rate_%"] for x in rstats])),
            "avg_net_%": float(np.mean([x["avg_net_%"] for x in rstats])),
            "profit_factor": float(np.nanmean([x["profit_factor"] for x in rstats])),
            "target_hits_%": float(np.mean([x["target_hits_%"] for x in rstats])),
            "timeouts_%": float(np.mean([x["timeouts_%"] for x in rstats])),
        })

    if rows:
        cmp = pd.DataFrame(rows).set_index("strategy").round(3)
        st.dataframe(cmp, use_container_width=True)

        f = cmp.loc["flow (OFI)", "avg_net_%"] if "flow (OFI)" in cmp.index else np.nan
        p = cmp.loc["price momentum", "avg_net_%"] if "price momentum" in cmp.index else np.nan
        r = cmp.loc["random (avg of 3)", "avg_net_%"] if "random (avg of 3)" in cmp.index else np.nan

        if f == f and r == r and f <= r:
            st.error("**Flow does not beat random entries.** The signal is "
                     "adding nothing; any profit is coming from the exit "
                     "structure. Stop here.")
        elif f == f and p == p and f <= p:
            st.warning("**Flow does not beat price momentum.** Order flow is "
                       "not contributing over what the chart already told you. "
                       "The simpler signal wins.")
        elif f == f:
            st.success("**Flow beats both controls.** That is the result worth "
                       "having — the edge is in the flow data, not in the "
                       "exits or in price.")

# ==========================================================================
# Signal detail
# ==========================================================================

with tab_sig:
    syms = list(close.columns)
    pick = st.selectbox("Symbol", syms, index=min(3, len(syms) - 1))
    n = st.slider("Bars to show", 100, 2000, 500, 50)

    d = pd.DataFrame({
        "price": close[pick].tail(n),
        "OFI (smoothed)": sig["ofi"][pick].tail(n),
        "OFI z-score": sig["ofi_z"][pick].tail(n),
    })
    st.line_chart(d[["price"]], use_container_width=True)
    st.line_chart(d[["OFI (smoothed)", "OFI z-score"]], use_container_width=True)

    st.caption(
        "OFI above zero means more volume lifted the ask than hit the bid over "
        "the smoothing window. The z-score asks whether that is unusual for "
        "this symbol, which is what the entry threshold tests."
    )

    st.subheader("Cumulative volume delta")
    st.line_chart(sig["cvd"][pick].tail(n), use_container_width=True)
    st.caption("Running total of signed volume. A rising CVD with a flat price "
               "means buyers are being absorbed — often read as distribution.")

st.divider()
st.caption(
    "Research tool. Signals are computed from data up to each bar and traded "
    "at the next bar's close; when a bar spans both target and stop, the stop "
    "is assumed to fill first. Not investment advice, and nothing here is "
    "proven until it clears all three tabs."
)
