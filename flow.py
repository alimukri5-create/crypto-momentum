"""
Order-flow imbalance signals and the honest test of whether they work.

THE IDEA
--------
Binance klines report `taker_buy_quote` — the share of each bar's quote volume
that lifted the ask rather than hit the bid. That is order-flow imbalance:
aggressive buying versus aggressive selling. It is free, it backfills years,
and it is not in the candle. Most flow signals need a recorded websocket feed
that nobody has historically; this one does not.

    OFI = 2 * (taker_buy_quote / quote_volume) - 1

    +1  every trade lifted the ask (pure aggressive buying)
     0  balanced
    -1  every trade hit the bid (pure aggressive selling)

WHAT THE LITERATURE SAYS
------------------------
"The Quarter-Hour Effect: Periodic Algorithmic Trading and Return
Predictability in Cryptocurrency Futures" (arXiv 2607.09426) finds that
quarter-hour order imbalance predicts returns over a 4-12 HOUR horizon —
slow enough to be tradeable by hand, fast enough that the portfolio system
cannot capture it. The paper does NOT test transaction costs, which is the
question that decides whether any of this is real for us.

Supporting: Vafin, "Order-Flow Imbalance and Short-Horizon Return
Predictability in Cryptocurrency Markets" (SSRN 6938742); Anastasopoulos &
Gradojevic, "Order flow and cryptocurrency returns".

THE TEST THAT COMES FIRST
-------------------------
Before any backtest, `ic_by_horizon` measures the rank correlation between
the signal today and the return over the next N hours, for a range of N.
That single table answers "is there anything here at all" and "at which
horizon", without any strategy, parameters, or curve-fitting to hide behind.

If IC is ~0 everywhere, nothing that follows can rescue it, and we stop.
If IC peaks in the 4-12h band, the paper replicates on spot data and the
strategy is worth building.

An IC of 0.02-0.05 is normal and useful for this kind of signal. An IC above
0.15 on a free, public, widely-known data field almost certainly means a bug
— most likely lookahead. Be suspicious of good news here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================================
# SIGNALS
# ============================================================================

@dataclass
class FlowConfig:
    # --- signal construction ---
    ofi_smooth: int = 6           # bars to smooth raw OFI over
    z_window: int = 168           # bars for the z-score baseline (168h = 1wk)
    cvd_window: int = 24          # bars for cumulative volume delta slope

    # --- entry gates ---
    min_ofi_z: float = 1.0        # how unusual the buying pressure must be
    require_price_confirm: bool = True   # price must not be falling
    price_confirm_bars: int = 4
    min_rvol: float = 1.2         # volume vs its own recent baseline
    use_btc_regime: bool = True
    btc_regime_bars: int = 168    # BTC above its own N-bar mean

    # --- liquidity ---
    min_quote_vol_musd: float = 1.0      # per-bar average, millions

    # --- exits (Ali's numbers) ---
    tp_pct: float = 3.0
    sl_pct: float = 2.0
    max_hold_bars: int = 12       # the documented 4-12h window

    # --- costs ---
    cost_per_side_pct: float = 0.15

    # --- position limits ---
    max_concurrent: int = 3


def compute_flow(close: pd.DataFrame, qvol: pd.DataFrame, tbq: pd.DataFrame,
                 cfg: FlowConfig) -> dict:
    """
    All quantities at bar t use only data up to and including bar t.
    Execution lag is applied separately in the backtest.
    """
    # --- raw order-flow imbalance, in [-1, +1] ---
    ofi_raw = (2.0 * (tbq / qvol.replace(0, np.nan)) - 1.0).clip(-1, 1)
    ofi = ofi_raw.rolling(cfg.ofi_smooth, min_periods=2).mean()

    # --- z-score it against its own recent history, per symbol ---
    mu = ofi.rolling(cfg.z_window, min_periods=cfg.z_window // 3).mean()
    sd = ofi.rolling(cfg.z_window, min_periods=cfg.z_window // 3).std()
    ofi_z = (ofi - mu) / sd.replace(0, np.nan)

    # --- cumulative volume delta and its slope ---
    # signed quote volume: +buy, -sell
    signed = qvol * ofi_raw
    cvd = signed.fillna(0).cumsum()
    cvd_slope = cvd.diff(cfg.cvd_window) / qvol.rolling(
        cfg.cvd_window, min_periods=2).sum().replace(0, np.nan)

    # --- supporting series ---
    ret = close.pct_change()
    rvol = qvol / qvol.rolling(cfg.z_window, min_periods=24).mean().replace(0, np.nan)
    price_ok = close > close.shift(cfg.price_confirm_bars)
    adv_musd = qvol.rolling(cfg.z_window, min_periods=24).mean() / 1e6

    return {
        "ofi_raw": ofi_raw,
        "ofi": ofi,
        "ofi_z": ofi_z,
        "cvd": cvd,
        "cvd_slope": cvd_slope,
        "rvol": rvol,
        "price_ok": price_ok,
        "adv_musd": adv_musd,
        "ret": ret,
    }


# ============================================================================
# THE HONEST TEST — does the signal predict anything, and at what horizon?
# ============================================================================

def ic_by_horizon(signal: pd.DataFrame, close: pd.DataFrame,
                  horizons=(1, 2, 4, 6, 8, 12, 24, 48),
                  method: str = "spearman") -> pd.DataFrame:
    """
    Information coefficient: cross-sectional rank correlation between the
    signal at bar t and the forward return from t to t+h, averaged over bars.

    This is the test that decides whether to continue. No strategy, no
    parameters, nowhere for a curve fit to hide.

    Returns one row per horizon: mean IC, standard error, t-statistic, and the
    share of bars where IC was positive.
    """
    out = []
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        # align, drop bars with too few names to rank
        ics = []
        sig_v = signal.replace([np.inf, -np.inf], np.nan)
        for i in range(len(signal)):
            s = sig_v.iloc[i]
            f = fwd.iloc[i]
            both = pd.concat([s, f], axis=1).dropna()
            if len(both) >= 8:
                c = both.iloc[:, 0].corr(both.iloc[:, 1], method=method)
                if c == c:
                    ics.append(c)
        if not ics:
            continue
        arr = np.array(ics)
        se = arr.std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else np.nan
        # Consecutive bars share overlapping forward windows, so their ICs are
        # serially correlated and the naive standard error is far too small.
        # Widening it by sqrt(h) is the standard rough correction for h-period
        # overlap. Without it EVERY horizon looks significant.
        se_adj = se * math.sqrt(h) if se == se else np.nan
        out.append({
            "horizon_bars": h,
            "mean_IC": arr.mean(),
            "std_err": se_adj,
            "t_stat": arr.mean() / se_adj if se_adj and se_adj > 0 else np.nan,
            "t_stat_naive": arr.mean() / se if se and se > 0 else np.nan,
            "pct_positive": (arr > 0).mean() * 100,
            "n_bars": len(arr),
        })
    return pd.DataFrame(out).set_index("horizon_bars")


# A signal must clear BOTH bars to count: statistically distinguishable from
# noise AND economically large enough to survive costs. Statistical
# significance alone is meaningless here — with thousands of bars, an IC of
# 0.005 can be "significant" and still worth nothing.
MIN_USEFUL_IC = 0.02
MIN_T_STAT = 3.0
TOO_GOOD_IC = 0.15


def horizon_verdict(ic: pd.DataFrame) -> tuple[str, str]:
    """Plain-language read on an IC table. (verdict, explanation)."""
    if ic.empty:
        return "NO DATA", "Not enough overlapping data to measure anything."

    # Direction matters: this is a LONG-ONLY buy signal, so only positive IC
    # is usable. A strong negative IC means the signal predicts the opposite.
    pos = ic[ic["mean_IC"] > 0]
    best = pos["mean_IC"].idxmax() if not pos.empty else ic["mean_IC"].idxmax()
    best_ic = float(ic.loc[best, "mean_IC"])
    best_t = float(ic.loc[best, "t_stat"])

    worst = float(ic["mean_IC"].min())
    if pos.empty or (abs(worst) > best_ic * 2 and abs(worst) > MIN_USEFUL_IC):
        return ("INVERTED",
                f"The strongest relationship is NEGATIVE (IC {worst:.3f}). "
                "High order-flow imbalance precedes LOWER returns, not higher "
                "— the opposite of the thesis. Buying on it would be backwards.")

    if best_ic > TOO_GOOD_IC:
        return ("SUSPICIOUS",
                f"IC of {best_ic:.3f} at {best}h is too good for a free, "
                "public data field. Suspect lookahead or a bug before "
                "believing it.")

    if abs(best_t) < MIN_T_STAT:
        return ("NOTHING HERE",
                f"Best positive horizon {best}h: IC {best_ic:.3f}, t-stat "
                f"{best_t:.1f}. Below {MIN_T_STAT:g} is not distinguishable "
                "from noise. No strategy fixes this.")

    if best_ic < MIN_USEFUL_IC:
        return ("TOO SMALL TO TRADE",
                f"Best positive horizon {best}h: IC {best_ic:.3f} with t-stat "
                f"{best_t:.1f}. Statistically real but economically tiny — an "
                f"IC under {MIN_USEFUL_IC} will not clear a 0.3% round trip.")

    in_band = 4 <= best <= 12
    return ("SIGNAL PRESENT",
            f"Best at {best}h: IC {best_ic:.3f}, t-stat {best_t:.1f}. "
            + ("Inside the 4-12h band the literature predicts, which is the "
               "result we were looking for."
               if in_band else
               "OUTSIDE the 4-12h band the literature predicts — the effect "
               "may be something else, so treat the paper as unreplicated."))


# ============================================================================
# BACKTEST — event-driven, one position per symbol, fixed TP/SL
# ============================================================================

def backtest_flow(close: pd.DataFrame, entries: pd.DataFrame,
                  cfg: FlowConfig, execution_lag: int = 1) -> dict:
    """
    Every entry signal opens a position at the NEXT bar's close and exits on
    whichever comes first: take profit, stop loss, or max hold. Mirrors a
    resting OCO with a time-out, which is what Ali can actually place.

    Conservative on ambiguity: if a bar's range spans both TP and SL, the
    STOP is assumed to fill first.
    """
    trades = []
    open_count = pd.Series(0, index=close.index)

    for sym in close.columns:
        px = close[sym]
        sig = entries[sym] if sym in entries.columns else None
        if sig is None:
            continue
        idx = close.index
        i = 0
        n = len(idx)
        while i < n - execution_lag - 1:
            if not bool(sig.iloc[i]):
                i += 1
                continue
            entry_i = i + execution_lag
            entry_px = px.iloc[entry_i]
            if not np.isfinite(entry_px) or entry_px <= 0:
                i += 1
                continue

            tp = entry_px * (1 + cfg.tp_pct / 100.0)
            sl = entry_px * (1 - cfg.sl_pct / 100.0)
            exit_i, exit_px, reason = None, None, None

            for j in range(entry_i + 1, min(entry_i + 1 + cfg.max_hold_bars, n)):
                p = px.iloc[j]
                if not np.isfinite(p):
                    continue
                # stop checked first — the pessimistic assumption
                if p <= sl:
                    exit_i, exit_px, reason = j, sl, "stop"
                    break
                if p >= tp:
                    exit_i, exit_px, reason = j, tp, "target"
                    break
            if exit_i is None:
                exit_i = min(entry_i + cfg.max_hold_bars, n - 1)
                exit_px, reason = px.iloc[exit_i], "timeout"

            gross = exit_px / entry_px - 1.0
            net = gross - 2 * cfg.cost_per_side_pct / 100.0
            trades.append({
                "symbol": sym,
                "entry_time": idx[entry_i], "exit_time": idx[exit_i],
                "bars_held": exit_i - entry_i,
                "entry_px": entry_px, "exit_px": exit_px,
                "gross_%": gross * 100, "net_%": net * 100,
                "reason": reason,
            })
            open_count.iloc[entry_i:exit_i + 1] += 1
            i = exit_i + 1          # no overlapping positions in one symbol

    tr = pd.DataFrame(trades)
    if tr.empty:
        return {"trades": tr, "stats": {}, "equity": pd.Series(dtype=float)}

    tr = tr.sort_values("entry_time").reset_index(drop=True)

    wins = tr["net_%"] > 0
    gross_win = tr.loc[tr["net_%"] > 0, "net_%"].sum()
    gross_loss = abs(tr.loc[tr["net_%"] < 0, "net_%"].sum())
    net_win = tr.loc[wins, "net_%"].mean() if wins.any() else np.nan
    net_loss = tr.loc[~wins, "net_%"].mean() if (~wins).any() else np.nan
    be_wr = (abs(net_loss) / (net_win + abs(net_loss)) * 100
             if (net_win == net_win and net_loss == net_loss
                 and (net_win + abs(net_loss)) > 0) else np.nan)

    # Equal-risk equity curve: each trade risks the same fraction.
    eq = (1 + tr["net_%"] / 100.0 / max(cfg.max_concurrent, 1)).cumprod()

    stats = {
        "trades": len(tr),
        "win_rate_%": wins.mean() * 100,
        "breakeven_win_rate_%": be_wr,
        "avg_net_%": tr["net_%"].mean(),
        "median_net_%": tr["net_%"].median(),
        "avg_win_%": net_win,
        "avg_loss_%": net_loss,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else np.nan,
        "total_net_%": tr["net_%"].sum(),
        "avg_bars_held": tr["bars_held"].mean(),
        "target_hits_%": (tr["reason"] == "target").mean() * 100,
        "stop_hits_%": (tr["reason"] == "stop").mean() * 100,
        "timeouts_%": (tr["reason"] == "timeout").mean() * 100,
        "cost_drag_%": len(tr) * 2 * cfg.cost_per_side_pct,
    }
    return {"trades": tr, "stats": stats, "equity": eq}


def build_entries(sig: dict, close: pd.DataFrame, cfg: FlowConfig,
                  benchmark: str = "BTCUSDT") -> pd.DataFrame:
    """Boolean entry grid from the flow signals and gates."""
    ok = sig["ofi_z"] >= cfg.min_ofi_z
    ok &= sig["rvol"] >= cfg.min_rvol
    ok &= sig["adv_musd"] >= cfg.min_quote_vol_musd
    if cfg.require_price_confirm:
        ok &= sig["price_ok"]
    if cfg.use_btc_regime and benchmark in close.columns:
        b = close[benchmark]
        risk_on = b > b.rolling(cfg.btc_regime_bars, min_periods=24).mean()
        ok = ok.where(risk_on, False)
    return ok.fillna(False)


# ============================================================================
# CONTROLS — the comparison that makes the result mean something
# ============================================================================

def price_only_entries(close: pd.DataFrame, qvol: pd.DataFrame,
                       cfg: FlowConfig, benchmark: str = "BTCUSDT") -> pd.DataFrame:
    """
    Same structure, same gates, but driven by PRICE momentum instead of flow.
    If flow does not beat this, flow adds nothing and we have learned that
    cheaply rather than after weeks of tuning.
    """
    mom = close.pct_change(cfg.ofi_smooth)
    mz = ((mom - mom.rolling(cfg.z_window, min_periods=cfg.z_window // 3).mean())
          / mom.rolling(cfg.z_window, min_periods=cfg.z_window // 3).std().replace(0, np.nan))
    rvol = qvol / qvol.rolling(cfg.z_window, min_periods=24).mean().replace(0, np.nan)
    adv = qvol.rolling(cfg.z_window, min_periods=24).mean() / 1e6

    ok = (mz >= cfg.min_ofi_z) & (rvol >= cfg.min_rvol) & (adv >= cfg.min_quote_vol_musd)
    if cfg.require_price_confirm:
        ok &= close > close.shift(cfg.price_confirm_bars)
    if cfg.use_btc_regime and benchmark in close.columns:
        b = close[benchmark]
        ok = ok.where(b > b.rolling(cfg.btc_regime_bars, min_periods=24).mean(), False)
    return ok.fillna(False)


def random_entries(template: pd.DataFrame, rate: float, seed: int = 0) -> pd.DataFrame:
    """
    Random entries at the same frequency — the null hypothesis. If the real
    signal does not beat coin flips into the same TP/SL structure, the exits
    are doing the work and the signal is decoration.
    """
    rng = np.random.default_rng(seed)
    arr = rng.random(template.shape) < rate
    return pd.DataFrame(arr, index=template.index, columns=template.columns)
