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
        close, qvol, report, _tbq = load(tuple(universe), start)
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

with tab_live:
    tbl = eng.current_ranking(close, qvol, cfg)
    risk_on = tbl.attrs["btc_risk_on"]
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()

    w_all = res["weights"]
    rebal_rows = [i for i, d in enumerate(w_all.index)
                  if d.dayofweek in cfg.rebalance_days]

    def book_at(i):
        row = w_all.iloc[i]
        return row[row > 0].sort_values(ascending=False)

    now_book = book_at(rebal_rows[-1]) if rebal_rows else pd.Series(dtype=float)
    prev_book = book_at(rebal_rows[-2]) if len(rebal_rows) > 1 else pd.Series(dtype=float)

    added = [s for s in now_book.index if s not in prev_book.index]
    dropped = [s for s in prev_book.index if s not in now_book.index]
    kept = [s for s in now_book.index if s in prev_book.index]

    def nm(s):
        return s.replace("USDT", "").replace("USD", "")

    # Prices and exit levels are derived HERE from `close`, not read off the
    # ranking table. The ranking table's columns depend on the engine version
    # that happens to be deployed; `close` is always present. One less thing
    # that can silently render "$?".
    last_px = close.iloc[-1]
    exit_lvl = close.rolling(cfg.ts_ma_window).mean().iloc[-1]
    btc_exit = (float(close[dat.BENCHMARK].rolling(cfg.btc_regime_ma).mean().iloc[-1])
                if dat.BENCHMARK in close.columns else None)

    def fmt(v):
        if v is None or v != v:
            return "n/a"
        v = float(v)
        if v >= 1000:  return f"{v:,.0f}"
        if v >= 1:     return f"{v:,.2f}"
        return f"{v:,.4f}"

    def px(s):
        return fmt(last_px.get(s))

    def ex(s):
        return fmt(exit_lvl.get(s))

    is_rebal_today = today.dayofweek in cfg.rebalance_days
    day_label = today.strftime("%A %d %b")

    # ---------------------------------------------------------------- RISK OFF
    if cfg.use_btc_regime and risk_on is False:
        lvl = btc_exit
        st.markdown(f"# 🔴 Sell everything")
        st.markdown("Bitcoin is below its trend line. Close every position "
                    "at market and stay in cash.")
        if lvl:
            st.caption(f"Back in when BTC closes above \\${lvl:,.0f}.")

    else:
        # ------------------------------------------------------------- SELL
        if dropped:
            st.markdown("# Sell")
            for s in dropped:
                st.markdown(f"### {nm(s)} &nbsp; — &nbsp; sell now at market "
                            f"(~\\${px(s)})")

        # -------------------------------------------------------------- BUY
        if added:
            st.markdown("# Buy")
            for s in added:
                st.markdown(
                    f"### {nm(s)} &nbsp; — &nbsp; buy at ~\\${px(s)}\n"
                    f"Sell it if it closes below **\\${ex(s)}**")

        # ------------------------------------------------------------- HOLD
        if kept:
            st.markdown("# Hold")
            for s in kept:
                st.markdown(
                    f"**{nm(s)}** — sell if it closes below "
                    f"**\\${ex(s)}**  ·  now \\${px(s)}")

        if not dropped and not added and not kept:
            st.markdown("# Nothing to hold")
            st.markdown("No coin passes the rules right now. Stay in cash.")
        elif not dropped and not added:
            st.markdown("### ✅ No trades — nothing changed since last time.")

        lvl = btc_exit
        if lvl:
            st.markdown("---")
            st.markdown(f"**Sell everything if Bitcoin closes below "
                        f"\\${lvl:,.0f}.**")

    # ------------------------------------------------------------- timing
    st.markdown("---")
    if is_rebal_today:
        st.markdown(f"*{day_label} — trading day. Act on the above.*")
    else:
        nxt = min(((d - today.dayofweek) % 7) or 7 for d in cfg.rebalance_days)
        st.markdown(f"*{day_label} — not a trading day, nothing to do. "
                    f"Next check: {(today + pd.Timedelta(days=nxt)).strftime('%A %d %b')}.*")

    st.caption("Exit levels move a little each day — re-read them each time "
               "you open this. Prices are yesterday's close; buy at whatever "
               "the market shows when you place the order.")

    with st.expander("How much of each (the split the backtest used)"):
        if len(now_book):
            st.dataframe(pd.DataFrame({
                "share of your crypto money": (now_book * 100).round(0).astype(int).astype(str) + "%",
                "price": [px(s) for s in now_book.index],
                "sell below": [ex(s) for s in now_book.index],
            }), use_container_width=True)
            st.caption("Splitting evenly instead is simpler and close, but the "
                       "backtested numbers used this volatility-weighted split.")

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
