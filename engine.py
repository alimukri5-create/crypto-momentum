"""
Momentum engine: signals, portfolio construction, backtest, statistics.

Design rationale and the literature behind it are in README.md. The short
version: TIME-SERIES momentum is the gate (each asset must trend against its
own history), cross-sectional ranking only selects among survivors. That
ordering follows Han, Kang & Ryu (2024), who find TS momentum in crypto
strong and cross-sectional "almost non-existent".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import BENCHMARK


@dataclass
class Config:
    # --- Momentum signal ---
    # Multiple horizons, per the JFQA trend-factor finding that single-lookback
    # signals leave information on the table.
    horizons: tuple = (14, 30, 60)
    horizon_weights: tuple = (0.25, 0.50, 0.25)   # 30d carries most weight
    vol_window: int = 30          # realised vol lookback, for risk-adjusting
    risk_adjust: bool = True      # score = return / vol, not raw return

    # --- Time-series gate (the stronger evidence) ---
    use_ts_gate: bool = True
    ts_ma_window: int = 50        # asset must close above its own MA
    ts_require_positive: bool = True   # and every horizon return must be > 0

    # --- Volume confirmation ---
    use_volume_filter: bool = True
    vol_ratio_window: int = 30
    min_volume_ratio: float = 0.8   # recent dollar volume vs its own baseline

    # --- Liquidity floor ---
    min_dollar_vol_musd: float = 5.0   # 30d average daily $ volume, millions

    # --- Portfolio ---
    top_k: int = 5                # how many names to hold
    # Rank buffer: keep holding a name until it falls out of the top
    # (top_k * hold_buffer). Standard turnover control for momentum books —
    # without it you churn every time two names swap places at the boundary.
    hold_buffer: float = 2.0      # 1.0 = no buffer
    sizing: str = "inv_vol"       # "inv_vol" | "equal"
    max_weight: float = 0.35      # cap on any single name
    allow_cash: bool = True       # hold cash when fewer than top_k qualify

    # --- Regime gate ---
    use_btc_regime: bool = True
    btc_regime_ma: int = 50       # BTC above its own MA => risk on

    # --- Execution ---
    rebalance_days: tuple = (0, 3)    # 0=Mon, 3=Thu  (twice weekly)
    cost_per_side_pct: float = 0.15   # 0.1% spot taker + 0.05% slippage
    execution_lag_days: int = 1       # signal on day t, trade on day t+1

    # --- Backtest window ---
    start: str = "2021-01-01"
    end: str | None = None

    # --- Experiment switches ---
    cross_sectional_only: bool = False   # True = disable TS gate, rank only


CFG = Config()


# ============================================================================
# SIGNALS
# ============================================================================

def compute_signals(close: pd.DataFrame, quote_vol: pd.DataFrame, cfg: Config = CFG) -> dict:
    """
    All signals are computed from data available up to and including day t.
    Execution is lagged separately in run_backtest, so there is no lookahead
    as long as you do not reach forward here.
    """
    daily_ret = close.pct_change()
    realised_vol = daily_ret.rolling(cfg.vol_window).std() * math.sqrt(365)

    # --- multi-horizon momentum, optionally risk-adjusted ---
    score = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    total_w = sum(cfg.horizon_weights)
    horizon_rets = {}
    for h, w in zip(cfg.horizons, cfg.horizon_weights):
        r = close.pct_change(h)
        horizon_rets[h] = r
        contrib = r / realised_vol.replace(0, np.nan) if cfg.risk_adjust else r
        score = score.add(contrib * (w / total_w), fill_value=0.0)

    # --- time-series gate: is this asset trending against its OWN history? ---
    ma = close.rolling(cfg.ts_ma_window).mean()
    ts_gate = close > ma
    if cfg.ts_require_positive:
        for h in cfg.horizons:
            ts_gate &= horizon_rets[h] > 0

    # --- volume confirmation ---
    vol_base = quote_vol.rolling(cfg.vol_ratio_window).mean()
    vol_recent = quote_vol.rolling(7).mean()
    vol_ratio = vol_recent / vol_base.replace(0, np.nan)
    vol_gate = vol_ratio >= cfg.min_volume_ratio

    # --- liquidity floor ---
    adv_musd = quote_vol.rolling(30).mean() / 1e6
    liq_gate = adv_musd >= cfg.min_dollar_vol_musd

    return {
        "score": score,
        "realised_vol": realised_vol,
        "ts_gate": ts_gate,
        "vol_gate": vol_gate,
        "liq_gate": liq_gate,
        "vol_ratio": vol_ratio,
        "adv_musd": adv_musd,
        "horizon_rets": horizon_rets,
        "daily_ret": daily_ret,
    }


def build_weights(sig: dict, close: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """Target weights per day. Rebalance days only; held flat in between."""
    score = sig["score"]
    eligible = sig["liq_gate"].copy()
    if cfg.use_volume_filter:
        eligible &= sig["vol_gate"]
    if cfg.use_ts_gate and not cfg.cross_sectional_only:
        eligible &= sig["ts_gate"]

    # BTC regime: a single risk-on/off switch over the whole book
    if cfg.use_btc_regime and BENCHMARK in close.columns:
        btc = close[BENCHMARK]
        risk_on = btc > btc.rolling(cfg.btc_regime_ma).mean()
    else:
        risk_on = pd.Series(True, index=close.index)

    inv_vol = 1.0 / sig["realised_vol"].replace(0, np.nan)

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    is_rebal = close.index.dayofweek.isin(cfg.rebalance_days)

    current = pd.Series(0.0, index=close.columns)
    for i, date in enumerate(close.index):
        if is_rebal[i]:
            regime_ok = bool(risk_on.iloc[i]) if not pd.isna(risk_on.iloc[i]) else False
            if not regime_ok:
                current = pd.Series(0.0, index=close.columns)
            else:
                row = score.iloc[i].where(eligible.iloc[i])
                row = row.dropna()
                row = row[row > 0]                       # never hold a negative score

                if cfg.hold_buffer > 1.0:
                    # Keep existing holdings while they remain inside the
                    # widened band; fill any free slots from the top.
                    band = row.nlargest(int(round(cfg.top_k * cfg.hold_buffer))).index
                    incumbents = [s for s in current[current > 0].index if s in band]
                    picks = list(incumbents[:cfg.top_k])
                    for s in row.nlargest(len(row)).index:
                        if len(picks) >= cfg.top_k:
                            break
                        if s not in picks:
                            picks.append(s)
                    picks = pd.Index(picks)
                else:
                    picks = row.nlargest(cfg.top_k).index

                if len(picks) == 0:
                    current = pd.Series(0.0, index=close.columns)
                else:
                    if cfg.sizing == "inv_vol":
                        raw = inv_vol.iloc[i].reindex(picks).fillna(0.0)
                        if raw.sum() <= 0:
                            raw = pd.Series(1.0, index=picks)
                    else:
                        raw = pd.Series(1.0, index=picks)

                    w = raw / raw.sum()
                    w = w.clip(upper=cfg.max_weight)
                    # If capping freed capital and we are not allowed cash,
                    # redistribute proportionally among uncapped names.
                    if not cfg.allow_cash and w.sum() < 1.0:
                        w = w / w.sum()
                    # Scale down if fewer names than top_k and cash allowed:
                    # invested fraction = len(picks)/top_k
                    if cfg.allow_cash:
                        w = w * (len(picks) / cfg.top_k)

                    current = pd.Series(0.0, index=close.columns)
                    current.loc[w.index] = w.values

        weights.iloc[i] = current.values

    return weights


# ============================================================================
# BACKTEST
# ============================================================================

def run_backtest(close: pd.DataFrame, weights: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """
    Lags weights by execution_lag_days so the signal from day t is traded on
    day t+lag. Costs charged on turnover.
    """
    ret = close.pct_change().fillna(0.0)
    held = weights.shift(cfg.execution_lag_days).fillna(0.0)

    gross = (held * ret).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cfg.cost_per_side_pct / 100.0)
    net = gross - cost

    out = pd.DataFrame({
        "gross_ret": gross,
        "cost": cost,
        "net_ret": net,
        "turnover": turnover,
        "exposure": held.sum(axis=1),
        "n_held": (held > 0).sum(axis=1),
    })
    out["equity"] = (1 + out["net_ret"]).cumprod()
    out["equity_gross"] = (1 + out["gross_ret"]).cumprod()
    return out


def stats(returns: pd.Series, name: str = "strategy") -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / 365.0
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * math.sqrt(365)
    sharpe = (r.mean() * 365) / vol if vol > 0 else np.nan
    downside = r[r < 0].std() * math.sqrt(365)
    sortino = (r.mean() * 365) / downside if downside > 0 else np.nan
    dd = (eq / eq.cummax() - 1)
    # Fat tails matter here — Han et al. warn mean return alone misleads.
    return {
        "name": name,
        "total_return_%": (eq.iloc[-1] - 1) * 100,
        "CAGR_%": cagr * 100,
        "vol_%": vol * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "max_DD_%": dd.min() * 100,
        "calmar": (cagr / abs(dd.min())) if dd.min() < 0 else np.nan,
        "skew": r.skew(),
        "kurtosis": r.kurtosis(),
        "best_day_%": r.max() * 100,
        "worst_day_%": r.min() * 100,
        "pct_days_up": (r > 0).mean() * 100,
    }




# ============================================================================
# LIVE RANKING — what to actually buy on a rebalance day
# ============================================================================

def current_ranking(close: pd.DataFrame, quote_vol: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    sig = compute_signals(close, quote_vol, cfg)
    i = -1
    tbl = pd.DataFrame({
        "score": sig["score"].iloc[i],
        "ret_14d_%": sig["horizon_rets"][cfg.horizons[0]].iloc[i] * 100,
        "ret_30d_%": sig["horizon_rets"][cfg.horizons[1]].iloc[i] * 100,
        "ret_60d_%": sig["horizon_rets"][cfg.horizons[2]].iloc[i] * 100,
        "ann_vol_%": sig["realised_vol"].iloc[i] * 100,
        "adv_$M": sig["adv_musd"].iloc[i],
        "ts_gate": sig["ts_gate"].iloc[i],
        "vol_gate": sig["vol_gate"].iloc[i],
        "liq_gate": sig["liq_gate"].iloc[i],
    })
    tbl["eligible"] = tbl["liq_gate"] & (tbl["vol_gate"] if cfg.use_volume_filter else True)
    if cfg.use_ts_gate and not cfg.cross_sectional_only:
        tbl["eligible"] &= tbl["ts_gate"]
    tbl = tbl.sort_values("score", ascending=False)

    btc_on = None
    if cfg.use_btc_regime and BENCHMARK in close.columns:
        b = close[BENCHMARK]
        btc_on = bool(b.iloc[-1] > b.rolling(cfg.btc_regime_ma).mean().iloc[-1])
    tbl.attrs["btc_risk_on"] = btc_on
    tbl.attrs["as_of"] = str(close.index[-1].date())
    return tbl




# ============================================================================
# BENCHMARKS
# ============================================================================

def benchmarks(close: pd.DataFrame) -> dict:
    """Buy-and-hold BTC, and an equal-weight basket of the whole universe."""
    out = {}
    if BENCHMARK in close.columns:
        out["buy_hold_BTC"] = close[BENCHMARK].pct_change().fillna(0.0)
    out["equal_weight_universe"] = close.pct_change().fillna(0.0).mean(axis=1)
    return out


def run_all(close: pd.DataFrame, qvol: pd.DataFrame, cfg: Config):
    """One call: signals -> weights -> backtest -> stats table."""
    sig = compute_signals(close, qvol, cfg)
    w = build_weights(sig, close, cfg)
    bt = run_backtest(close, w, cfg)

    rows = [stats(bt["net_ret"], "momentum (net)"),
            stats(bt["gross_ret"], "momentum (gross)")]
    for k, v in benchmarks(close).items():
        rows.append(stats(v, k))
    table = pd.DataFrame(rows).set_index("name")

    return {"signals": sig, "weights": w, "backtest": bt, "stats": table}
