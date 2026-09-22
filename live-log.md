# Crypto Momentum — live observation log

Started 15 Sep 2026. Paper only; no real money committed.
Each rebalance day (Mon/Thu) the scheduled check appends a row.

Purpose: answer one question honestly after a few weeks — did the calls it
made actually work? The app recomputes from history and keeps no record of
what it said, so this file IS the record.

---

## Baseline — Tue 15 Sep 2026 (not a rebalance day)

Data through 2026-09-15 · venue binance · 40 symbols

| | coin | price | exit level | room |
|---|---|---|---|---|
| SELL | DOT | $0.9890 | — | dropped from ranking |
| BUY | INJ | $5.86 | $5.04 | 16.3% |
| HOLD | ICP | $2.56 | $2.38 | **7.6%** ← closest to exit |
| HOLD | NEAR | $2.39 | $1.89 | 26.5% |
| HOLD | UNI | $6.56 | $4.74 | 38.4% |
| HOLD | ARB | $0.1397 | $0.1024 | 36.4% |

Regime: RISK ON. Sell everything if BTC closes below **$71,792**.

Notes:
- ICP has the least room to its exit — first candidate to be dropped.
- Book is INJ, ICP, NEAR, UNI, ARB once the DOT→INJ swap is made.

---

## Log

<!-- Scheduled checks append below this line. Newest at the bottom. -->


## Thu 17 Sep 2026 — rebalance day

Data through 2026-09-17 · venue binance · 37 symbols (⚠ 3 unavailable, was 40 on 15 Sep)

| | coin | price | exit level | room |
|---|---|---|---|---|
| SELL | ICP | ~$2.56 | $2.38 | closed below / dropped from ranking |
| BUY | DOT | ~$1.01 | $0.8821 | 12.7% |
| HOLD | NEAR | $2.69 | $1.93 | 28.3% |
| HOLD | UNI | $6.73 | $4.85 | 27.9% |
| HOLD | INJ | $5.49 | $5.06 | **7.8%** ← closest to exit |
| HOLD | ARB | $0.1660 | $0.1062 | 36.0% |

Regime: RISK ON. Sell everything if BTC closes below **$72,183** (was $71,792).

Changed since last entry:
- ICP out (flagged on 15 Sep as the closest to its exit — that call was right).
- DOT in at ~$1.01. Note: DOT was the SELL on 15 Sep at $0.9890, so the system
  is buying back the same name 2 days later 2.1% higher. Round-trip cost paid
  for nothing.
- Previous entry's BUY, INJ at $5.86, is now $5.49 — **down 6.3%**.
- NEAR, UNI, ARB all up and unchanged in the book.

Within ~5% of exit: none. INJ is nearest at 7.8% and is the likely next drop.

Churn: 2 of 2 observations have changed the book (1 of 1 actual rebalance days).
Backtest expected roughly half. Too few points to conclude anything yet, but the
DOT sell-then-rebuy inside 48h is exactly the pattern that makes realised turnover
exceed the 4.1%/yr cost model. Watch it.

---

## Mon 21 Sep 2026 — rebalance day

Data through 2026-09-21 · venue binance · 37 symbols (⚠ 3 unavailable, same as 17 Sep)

| | coin | price | exit level | room |
|---|---|---|---|---|
| SELL | DOT | ~$1.17 | $0.8821 (prev) | dropped from ranking |
| BUY | AVAX | ~$11.32 | $7.36 | 35.0% |
| HOLD | UNI | $8.99 | $5.24 | 41.7% |
| HOLD | INJ | $7.76 | $5.27 | **32.1%** ← closest to exit |
| HOLD | NEAR | $4.26 | $2.12 | 50.2% |
| HOLD | ARB | $0.2239 | $0.1174 | 47.6% |

Regime: RISK ON. Sell everything if BTC closes below **$73,658** (was $72,183).

Changed since last entry:
- DOT out after 4 days, sold ~$1.17 against the ~$1.01 buy on 17 Sep — **up ~15.8%**.
  So the 15 Sep sell-then-rebuy round trip ended up profitable this time; the
  cost objection stands but the call itself worked.
- AVAX in at ~$11.32, a new name (not previously in the book).
- INJ, flagged on 17 Sep as nearest its exit at 7.8% room, did not get dropped:
  it went $5.49 → $7.76, **up 41%**, and now has 32.1% room. That flag was wrong
  in direction — proximity to the MA was noise, not a signal of a coming drop.
- UNI $6.73 → $8.99 (+33.6%), NEAR $2.69 → $4.26 (+58.4%), ARB $0.1660 → $0.2239
  (+34.9%). Broad move up across the whole book.

Within ~5% of exit: none. The entire book is 32%+ above its exit level — the
widest margin recorded so far.

Churn: 3 of 3 observations have changed the book (2 of 2 actual rebalance days).
Backtest expected roughly half. It is changing every single time. Turnover is
running hotter than the 4.1%/yr cost model assumed. Still only 2 rebalances,
so not yet statistically meaningful — but the pattern is 100%, not 50%.

---
