"""
Crypto Momentum — Streamlit app.

Long-only spot. Time-series momentum gate, cross-sectional selection,
volatility-scaled sizing, BTC regime filter, twice-weekly rebalance.

Run locally:   streamlit run app.py
Deploy:        point Streamlit Community Cloud at this repo, main file app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import data as dat
import engine as eng

st.set_page_config(page_title="Crypto Momentum", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------
# Sidebar — parameters
# --------------------------------------------------------------------------

st.sidebar.title("Parameters")

with st.sidebar.expander("Data", expanded=True):
    start = st.date_input("History from", pd.Timestamp("2021-01-01")).strftime("%Y-%m-%d")
    universe_text = st.text_area(
        "Universe (one symbol per line)",
        value="\n".join(dat.UNIVERSE), height=120)
    universe = [s.strip().upper() for s in universe_text.split("\n") if s.strip()]

with st.sidebar.expander("Signal", expanded=True):
    h1 = st.number_input("Short horizon (d)", 5, 60, 14)
    h2 = st.number_input("Mid horizon (d)", 10, 120, 30)
    h3 = st.number_input("Long horizon (d)", 20, 250, 60)
    risk_adjust = st.checkbox("Risk-adjust momentum (return / vol)", True)
    use_ts_gate = st.checkbox("Time-series gate", True,
                              help="Asset must be above its own MA with all "
                                   "horizon returns positive. This is the "
                                   "better-evidenced half of the design.")
    ts_ma = st.number_input("TS gate MA (d)", 10, 200, 50)
    cross_only = st.checkbox("Cross-sectional only (disable TS gate)", False,
                             help="Run this to measure what the TS gate is worth.")

with st.sidebar.expander("Portfolio", expanded=True):
    top_k = st.slider("Names held", 1, 15, 5)
    buffer = st.slider("Rank buffer", 1.0, 4.0, 2.0, 0.5,
                       help="Hold a name until it leaves the top "
                            "(names x buffer). Cuts turnover.")
    sizing = st.selectbox("Sizing", ["inv_vol", "equal"], 0)
    max_w = st.slider("Max single weight", 0.1, 1.0, 0.35, 0.05)
    allow_cash = st.checkbox("Allow cash when few names qualify", True)

with st.sidebar.expander("Your portfolio", expanded=True):
    st.caption("Needed to turn target weights into actual orders.")
    cash_usd = st.number_input("Spare cash (USD)", 0.0, 1e9, 10000.0, 100.0)
    holdings_text = st.text_area(
        "What you hold now — one per line: SYMBOL UNITS",
        value="", height=110,
        placeholder="SOLUSDT 12.5\nETHUSDT 1.2\nBNBUSDT 4")
    min_trade_usd = st.number_input(
        "Ignore orders smaller than (USD)", 0.0, 10000.0, 50.0, 10.0,
        help="Stops you paying fees to move trivial amounts.")

with st.sidebar.expander("Regime & costs", expanded=True):
    use_btc = st.checkbox("BTC regime gate", True)
    btc_ma = st.number_input("BTC regime MA (d)", 10, 200, 50)
    cadence = st.selectbox("Rebalance", ["Weekly (Mon)",
                                         "Twice weekly (Mon/Thu)",
                                         "Daily"], 1)
    cost = st.number_input("Cost per side (%)", 0.0, 1.0, 0.15, 0.01)
    min_adv = st.number_input("Min 30d avg $ volume (M)", 0.0, 500.0, 5.0, 1.0)

CADENCE = {"Weekly (Mon)": (0,),
           "Twice weekly (Mon/Thu)": (0, 3),
           "Daily": tuple(range(7))}

cfg = eng.Config(
    horizons=(int(h1), int(h2), int(h3)),
    risk_adjust=risk_adjust,
    use_ts_gate=use_ts_gate,
    ts_ma_window=int(ts_ma),
    cross_sectional_only=cross_only,
    top_k=int(top_k),
    hold_buffer=float(buffer),
    sizing=sizing,
    max_weight=float(max_w),
    allow_cash=allow_cash,
    use_btc_regime=use_btc,
    btc_regime_ma=int(btc_ma),
    rebalance_days=CADENCE[cadence],
    cost_per_side_pct=float(cost),
    min_dollar_vol_musd=float(min_adv),
    start=start,
)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def load(symbols, start):
    return dat.load_universe(symbols, start)

st.title("Crypto Momentum")
st.caption("Long-only spot · time-series gate + cross-sectional selection · "
           "volatility-scaled sizing")

try:
    with st.spinner("Fetching market data…"):
        close, qvol, report = load(tuple(universe), start)
except Exception as e:
    st.error(f"Could not load market data: {e}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Venue", report["venue"])
c2.metric("Symbols", len(report["loaded"]))
c3.metric("Days", report["n_days"])
c4.metric("Through", report["end"])

if report["missing"] or report["too_short"]:
    with st.expander(f"⚠ {len(report['missing']) + len(report['too_short'])} symbols unavailable"):
        if report["missing"]:
            st.write("**Not on this venue:**", ", ".join(report["missing"]))
        if report["too_short"]:
            st.write("**Too little history:**", ", ".join(report["too_short"]))

res = eng.run_all(close, qvol, cfg)
bt, w = res["backtest"], res["weights"]

# --------------------------------------------------------------------------
# Live ranking — the tab you use on a rebalance day
# --------------------------------------------------------------------------

tab_live, tab_perf, tab_risk, tab_detail = st.tabs(
    ["Today", "Performance", "Risk", "Detail"])

def parse_holdings(text: str) -> pd.Series:
    """'SOLUSDT 12.5' per line -> Series of units. Tolerant of commas/tabs."""
    out = {}
    for raw in text.splitlines():
        line = raw.replace(",", " ").replace("\t", " ").strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        sym = parts[0].upper()
        if not sym.endswith("USDT") and not sym.endswith("USD"):
            sym = sym + "USDT"
        try:
            out[sym] = out.get(sym, 0.0) + float(parts[1])
        except ValueError:
            continue
    return pd.Series(out, dtype=float)


with tab_live:
    tbl = eng.current_ranking(close, qvol, cfg)
    risk_on = tbl.attrs["btc_risk_on"]
    last_px = close.iloc[-1]
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    is_rebal_today = today.dayofweek in cfg.rebalance_days

    # ---- target weights -------------------------------------------------
    picks = tbl[tbl["eligible"]].head(cfg.top_k)
    if cfg.use_btc_regime and risk_on is False:
        wts = pd.Series(dtype=float)          # regime overrides everything
    elif len(picks) == 0:
        wts = pd.Series(dtype=float)
    else:
        inv = 1.0 / picks["ann_vol_%"].replace(0, np.nan)
        wts = (inv / inv.sum()).clip(upper=cfg.max_weight) if cfg.sizing == "inv_vol" \
            else pd.Series(1.0 / len(picks), index=picks.index)
        if cfg.allow_cash:
            wts = wts * (len(picks) / cfg.top_k)

    # ---- current portfolio ----------------------------------------------
    held_units = parse_holdings(holdings_text)
    known = [s for s in held_units.index if s in last_px.index]
    unknown = [s for s in held_units.index if s not in last_px.index]
    held_usd = pd.Series(
        {s: held_units[s] * float(last_px[s]) for s in known}, dtype=float)
    equity = float(held_usd.sum()) + float(cash_usd)

    # ======================= WHAT TO DO TODAY ============================
    st.subheader("What to do today")

    if not is_rebal_today:
        nxt = min(((d - today.dayofweek) % 7) or 7 for d in cfg.rebalance_days)
        st.info(f"**Not a rebalance day.** Do nothing. "
                f"Next rebalance: {(today + pd.Timedelta(days=nxt)).strftime('%A %d %b')}.")
        st.caption("The orders below are what you would place on that day, "
                   "based on today's data. They will change by then.")

    if cfg.use_btc_regime and risk_on is False:
        st.error("**RISK OFF — BTC is below its trend.** The rule says sell "
                 "everything and sit in cash until it flips back.")
    elif cfg.use_btc_regime:
        st.success("**RISK ON — BTC is above its trend.** Positions permitted.")

    if equity <= 0:
        st.warning("Enter your holdings and spare cash in the sidebar "
                   "(**Your portfolio**) to get actual buy and sell orders.")
    else:
        target_usd = (wts * equity).reindex(
            sorted(set(wts.index) | set(held_usd.index))).fillna(0.0)
        current_usd = held_usd.reindex(target_usd.index).fillna(0.0)
        delta = target_usd - current_usd

        rows = []
        for sym in delta.index:
            d = float(delta[sym])
            cur, tgt = float(current_usd[sym]), float(target_usd[sym])
            px = float(last_px[sym]) if sym in last_px.index else np.nan
            if abs(d) < min_trade_usd:
                action = "HOLD" if tgt > 0 else "—"
                units = 0.0
            elif d > 0:
                action = "BUY"
                units = d / px if px and px > 0 else np.nan
            else:
                action = "SELL ALL" if tgt < min_trade_usd else "SELL"
                units = abs(d) / px if px and px > 0 else np.nan
            if action == "—":
                continue
            rows.append({"action": action, "symbol": sym,
                         "usd": round(abs(d) if action != "HOLD" else tgt, 2),
                         "approx_units": round(units, 6) if units == units else None,
                         "price": round(px, 4) if px == px else None,
                         "now_usd": round(cur, 2), "target_usd": round(tgt, 2)})

        orders = pd.DataFrame(rows)
        if orders.empty:
            st.success("**No trades needed.** Your book already matches the "
                       "target within the minimum trade size.")
        else:
            order_rank = {"SELL ALL": 0, "SELL": 1, "BUY": 2, "HOLD": 3}
            orders = orders.sort_values(
                by="action", key=lambda c: c.map(order_rank)).reset_index(drop=True)

            sells = orders[orders["action"].str.startswith("SELL")]
            buys = orders[orders["action"] == "BUY"]
            holds = orders[orders["action"] == "HOLD"]

            # NOTE: escape every "$" as "\$" — Streamlit's markdown treats a
            # pair of dollar signs as LaTeX math and mangles the amounts.
            def line(action, r):
                u = f"{r['approx_units']:g}" if r["approx_units"] else "?"
                return (f"- **{action} {r['symbol'].replace('USDT', '')}** — "
                        f"about **{u} units**  ·  \\${r['usd']:,.0f}  "
                        f"·  at ~\\${r['price']:,.4g}")

            if len(sells):
                st.markdown("#### 🔴 SELL")
                for _, r in sells.iterrows():
                    st.markdown(line(r["action"], r))
            if len(buys):
                st.markdown("#### 🟢 " + ("THEN BUY" if len(sells) else "BUY"))
                for _, r in buys.iterrows():
                    st.markdown(line("BUY", r))
            if len(holds):
                st.markdown("#### ⚪ LEAVE ALONE")
                st.markdown(", ".join(
                    f"**{r['symbol'].replace('USDT', '')}** (\\${r['target_usd']:,.0f})"
                    for _, r in holds.iterrows()))

            st.caption(
                "Sell before you buy so the cash is there. Market orders on "
                "spot. Units are approximate — prices move between now and "
                "when you place them, so size by the USD figure if they differ.")

            with st.expander("Order detail"):
                st.dataframe(orders, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Portfolio value", f"${equity:,.0f}")
        m2.metric("Invested after", f"{float(wts.sum()):.0%}")
        m3.metric("Cash after", f"${equity * (1 - float(wts.sum())):,.0f}")

    if unknown:
        st.warning(f"Not priced, ignored: {', '.join(unknown)}. "
                   "Check the symbol spelling.")

    # ---- reference: the target book -------------------------------------
    if len(wts):
        st.subheader(f"Target book as of {tbl.attrs['as_of']}")
        book = pd.DataFrame({
            "weight_%": (wts * 100).round(1),
            "score": picks["score"].reindex(wts.index).round(3),
            "30d_%": picks["ret_30d_%"].reindex(wts.index).round(1),
            "ann_vol_%": picks["ann_vol_%"].reindex(wts.index).round(0),
        })
        st.dataframe(book, use_container_width=True)
    elif not (cfg.use_btc_regime and risk_on is False):
        st.warning("No names pass the gates today — the rule says hold cash.")

    st.subheader("Full ranking")
    st.dataframe(
        tbl[["score", "ret_14d_%", "ret_30d_%", "ret_60d_%", "ann_vol_%",
             "adv_$M", "ts_gate", "vol_gate", "liq_gate", "eligible"]].round(2),
        use_container_width=True, height=420)

# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------

with tab_perf:
    stats_tbl = res["stats"].round(2)
    st.dataframe(stats_tbl, use_container_width=True)

    strat = bt["equity"]
    curves = pd.DataFrame({"Momentum (net)": strat})
    for name, r in eng.benchmarks(close).items():
        curves[name] = (1 + r).cumprod()
    st.subheader("Equity curves (log)")
    st.line_chart(np.log10(curves.clip(lower=1e-9)), use_container_width=True)

    beat_btc = (stats_tbl.loc["momentum (net)", "CAGR_%"]
                > stats_tbl.loc["buy_hold_BTC", "CAGR_%"]) \
        if "buy_hold_BTC" in stats_tbl.index else None

    if beat_btc is False:
        st.error("**Underperforms buy-and-hold BTC.** A long-only system that "
                 "loses to holding is not an edge. Do not deploy this.")
    elif beat_btc:
        margin = (stats_tbl.loc["momentum (net)", "CAGR_%"]
                  - stats_tbl.loc["buy_hold_BTC", "CAGR_%"])
        if margin < 10:
            st.warning(f"Beats BTC by {margin:.1f}pp CAGR. That margin is "
                       "within what survivorship bias alone could explain — "
                       "see the Risk tab.")
        else:
            st.success(f"Beats BTC by {margin:.1f}pp CAGR. Still check "
                       "survivorship on the Risk tab before trusting it.")

# --------------------------------------------------------------------------
# Risk — including the things that make the backtest a lie
# --------------------------------------------------------------------------

with tab_risk:
    st.subheader("Survivorship bias")
    late = report["late_listings"]
    st.write(
        f"The universe is **{len(report['loaded'])} names liquid today**. "
        f"Coins that were liquid in the past and have since died are absent, "
        f"which flatters every number on the Performance tab.")
    if late:
        st.write(f"**{len(late)} names listed after the start date** "
                 f"(these contribute no history early on): {', '.join(late)}")
    st.info("To bound the bias: rerun with the universe cut to names that "
            "existed at your start date. The gap between the two runs is a "
            "lower bound on the bias, not a correction for it.")

    st.subheader("Cost drag")
    years = len(bt) / 365.0
    d1, d2, d3 = st.columns(3)
    d1.metric("Annual turnover", f"{bt['turnover'].sum() / years:.1f}x")
    d2.metric("Annual cost drag", f"{bt['cost'].sum() / years * 100:.2f}%")
    d3.metric("Total cost paid", f"{bt['cost'].sum() * 100:.1f}%")
    st.caption("Raise the rank buffer or drop to weekly rebalancing to cut "
               "this. Twice-weekly typically costs ~1.4pp/yr more than weekly.")

    st.subheader("Exposure and drawdown")
    e1, e2, e3 = st.columns(3)
    e1.metric("Avg exposure", f"{bt['exposure'].mean():.0%}")
    e2.metric("Days fully in cash", f"{(bt['exposure'] == 0).mean():.0%}")
    e3.metric("Avg names held", f"{bt['n_held'].mean():.1f}")

    dd = bt["equity"] / bt["equity"].cummax() - 1
    st.area_chart(dd, use_container_width=True)

    st.subheader("Fat tails")
    st.write(
        "Han, Kang & Ryu warn that crypto returns are skewed and fat-tailed "
        "enough that mean return alone misleads. Check skew and kurtosis on "
        "the Performance tab, not just CAGR.")

# --------------------------------------------------------------------------
# Detail
# --------------------------------------------------------------------------

with tab_detail:
    st.subheader("Holdings over time")
    held = (w > 0).astype(int)
    st.caption("Number of days each name was held")
    counts = held.sum().sort_values(ascending=False)
    st.bar_chart(counts[counts > 0], use_container_width=True)

    st.subheader("Weights on the last 30 rebalances")
    recent = w[(w.diff().abs().sum(axis=1) > 0)].tail(30)
    st.dataframe((recent.loc[:, (recent > 0).any()] * 100).round(1),
                 use_container_width=True)

    st.subheader("Configuration in force")
    st.json({k: (list(v) if isinstance(v, tuple) else v)
             for k, v in cfg.__dict__.items()})

st.divider()
st.caption(
    "Logic tested on synthetic data; results depend entirely on real prices "
    "and the universe you choose. Not investment advice — this is a research "
    "tool, and the survivorship and cost caveats above are load-bearing.")
