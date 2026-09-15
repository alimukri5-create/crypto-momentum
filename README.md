# Crypto Momentum

Long-only spot crypto momentum system. Time-series momentum gate,
cross-sectional selection, volatility-scaled sizing, BTC regime filter,
configurable rebalance cadence. Streamlit front end.

**This is a research tool. The backtest it produces is biased upward for
reasons documented below, and none of its numbers have been validated against
real prices by the author of this code.**

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy on Streamlit Community Cloud: point it at this repo, main file `app.py`.

---

## Why it is built this way

The design follows the evidence, which does not say what most momentum
tooling assumes.

**Time-series momentum, not cross-sectional, is the better-supported effect
in crypto.** Han, Kang & Ryu (2024), *Time-Series and Cross-Sectional Momentum
in the Cryptocurrency Market: A Comprehensive Analysis under Realistic
Assumptions*, find evidence for time-series momentum is strong while
cross-sectional momentum is "almost non-existent." They also warn that crypto
returns are skewed and fat-tailed enough that mean return alone is an
inadequate test of profitability.
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4675565>

**But cross-sectional is not dead.** Drogen & Hoffstein (2023) find a
long-only cross-sectional effect at 30-day formation / 7-day holding that
beats a Bitcoin benchmark. It is a short practitioner paper by authors with a
commercial interest in the finding, so it is weighted below the above.
<https://papers.ssrn.com/abstract=4322637>

**Signals should combine horizons and use volume.** *A Trend Factor for the
Cross Section of Cryptocurrency Returns* (Journal of Financial and
Quantitative Analysis) builds a signal from price **and** volume across
**multiple** time horizons over 3,000+ coins, finds it survives transaction
costs and works in large liquid coins.
<https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/trend-factor-for-the-cross-section-of-cryptocurrency-returns/4C1509ACBA33D5DCAF0AC24379148178>

So the pipeline is:

1. **Time-series gate** (the strong evidence) — an asset is eligible only if
   it trades above its own moving average with every horizon return positive.
2. **Cross-sectional ranking** (the weaker evidence) — rank only the
   survivors, take the top K.
3. **Multi-horizon, volume-aware scoring** — 14/30/60 day returns, risk
   adjusted by realised volatility, with a volume confirmation filter.
4. **Volatility-scaled sizing** — weight inversely to realised vol, capped.
5. **BTC regime gate** — a single risk-on/off switch over the whole book.

Toggle `cross_sectional_only` in the sidebar to disable the TS gate and
measure what it is actually worth on your data rather than trusting the above.

---

## What this does NOT handle

### Survivorship bias — the big one

The default universe is names that are liquid **today**. Every coin that was
liquid in 2021 and has since collapsed or delisted is absent. A momentum
strategy backtested on survivors looks better than it was, because the
strategy's real failure mode — riding a name up until it doesn't come back —
is precisely what the missing names represent.

To bound it: run once as-is, then run again with the universe cut to coins
already listed at your start date. The gap is a **lower bound** on the bias,
not a correction.

If the strategy beats buy-and-hold BTC only narrowly, assume survivorship
explains the difference and treat the result as unproven. The app says this
to you on the Performance tab when the margin is thin.

### Costs are real and cadence is expensive

Measured turnover (synthetic data, but turnover is structural), annual cost
drag at 0.15% per side:

| Cadence | No buffer | Buffer 2.0 |
|---|---|---|
| Weekly | 3.2% | 2.7% |
| Twice weekly | 5.3% | 4.1% |
| Daily | 10.8% | 7.0% |

The rank buffer holds a name until it drops out of the top (K x buffer)
rather than the top K, which stops churn from names swapping places at the
boundary. It is on by default at 2.0.

### Other limits

- **Liquidity is approximate on Coinbase.** That venue reports base-unit
  volume; quote volume is estimated as volume x typical price.
- **No slippage model beyond a flat per-side cost.** Thin names will be worse.
- **Daily bars only.** Intraday execution quality is not modelled.
- **No shorting, no leverage, no derivatives.** Spot long only by design.

---

## Data

`data.py` tries venues in order and uses the first that answers:

1. **Binance** — best coverage, native quote volume. **Blocks US IPs (451).**
2. **Coinbase** — works from US. Fewer listings, approximate quote volume.

This matters because Streamlit Community Cloud runs in the US, so a
Binance-only app works locally and fails on deploy. The fallback exists for
that reason. Five names in the default universe are not on Coinbase (BNB,
RUNE, THETA, EGLD, FTM) and are dropped automatically when running from a US
IP; the app tells you which.

---

## Testing

```bash
python tests_logic.py       # 11 mechanical tests on synthetic data
python tests_turnover.py    # cost drag across cadence and buffer
```

`tests_logic.py` verifies: no lookahead in signal construction, execution
lagged behind signal, weights respecting top-K and position caps,
rebalancing only on configured days, costs charged and reducing returns, the
regime gate genuinely forcing cash, and the TS gate being restrictive.

**These test mechanics, not profitability.** Passing tests mean the backtest
is computing what it claims to compute. They say nothing about whether the
strategy makes money.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — Today / Performance / Risk / Detail tabs |
| `engine.py` | Signals, portfolio construction, backtest, statistics |
| `data.py` | Venue adapters and universe loading |
| `synthetic.py` | Synthetic price panels for tests |
| `tests_logic.py` | Mechanical correctness suite |
| `tests_turnover.py` | Cost drag comparison |

---

## Using it

The **Today** tab is the operational one. On a rebalance day it shows the BTC
regime state, the target book with weights, and the full ranking with every
gate's pass/fail. If the regime is risk-off it says hold cash, and means it.

Everything else is for deciding whether to trust the Today tab.
