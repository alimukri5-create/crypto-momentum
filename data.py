"""
Market data layer with venue fallback.

WHY THIS EXISTS
---------------
Binance geo-blocks US IP addresses (HTTP 451). Streamlit Community Cloud runs
in the US. So an app that only speaks Binance works in Colab and dies on
deploy. This module tries venues in order and uses the first that answers.

  1. Binance      — best coverage, native quote volume. Fails from US IPs.
  2. Coinbase     — works from US. Fewer listings, base-unit volume only.

Volume note: Coinbase reports volume in BASE units (e.g. BTC), Binance in
QUOTE units (USDT). We convert Coinbase to approximate quote volume as
volume * typical_price. It is an approximation and the liquidity filter is
correspondingly approximate on that venue.
"""

from __future__ import annotations

import time
import datetime as dt

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------
# Universe. Canonical form is the Binance symbol; venues map from it.
# --------------------------------------------------------------------------

UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
    "MATICUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT",
    "ETCUSDT", "XLMUSDT", "NEARUSDT", "ALGOUSDT", "FILUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT",
    "AAVEUSDT", "MKRUSDT", "RUNEUSDT", "GRTUSDT", "SANDUSDT",
    "AXSUSDT", "THETAUSDT", "EGLDUSDT", "FTMUSDT", "HBARUSDT",
    "VETUSDT", "ICPUSDT", "TIAUSDT", "SEIUSDT", "LDOUSDT",
]

BENCHMARK = "BTCUSDT"

COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume"]


# --------------------------------------------------------------------------
# Venue adapters
# --------------------------------------------------------------------------

class Venue:
    name = "base"

    def __init__(self, session: requests.Session | None = None):
        self.s = session or requests.Session()
        self.s.headers.update({"User-Agent": "crypto-momentum/1.0"})

    def available(self) -> bool:
        raise NotImplementedError

    def fetch(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError


class Binance(Venue):
    name = "binance"
    HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]

    def __init__(self, session=None):
        super().__init__(session)
        self.host = None

    def available(self) -> bool:
        for h in self.HOSTS:
            try:
                r = self.s.get(f"{h}/api/v3/ping", timeout=8)
                if r.status_code == 200:
                    self.host = h
                    return True
            except Exception:
                continue
        return False

    def fetch(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
        rows, cursor = [], int(start.timestamp() * 1000)
        while True:
            try:
                r = self.s.get(
                    f"{self.host}/api/v3/klines",
                    params={"symbol": symbol, "interval": "1d",
                            "limit": 1000, "startTime": cursor},
                    timeout=25)
            except Exception:
                break
            if r.status_code in (400, 451):
                return pd.DataFrame(columns=COLUMNS)
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 1000:
                break
            cursor = batch[-1][0] + 1
            time.sleep(0.1)

        if not rows:
            return pd.DataFrame(columns=COLUMNS)

        df = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "tb", "tq", "ig"])
        df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
        for c in COLUMNS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.set_index("date")[COLUMNS].sort_index()


class Coinbase(Venue):
    """Works from US IPs. 300 candles per request, so we page backwards."""
    name = "coinbase"
    HOST = "https://api.exchange.coinbase.com"

    # Binance symbol -> Coinbase product. Names absent here are unavailable.
    MAP = {
        "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD",
        "XRPUSDT": "XRP-USD", "ADAUSDT": "ADA-USD", "AVAXUSDT": "AVAX-USD",
        "DOGEUSDT": "DOGE-USD", "DOTUSDT": "DOT-USD", "LINKUSDT": "LINK-USD",
        "MATICUSDT": "MATIC-USD", "LTCUSDT": "LTC-USD", "BCHUSDT": "BCH-USD",
        "ATOMUSDT": "ATOM-USD", "UNIUSDT": "UNI-USD", "ETCUSDT": "ETC-USD",
        "XLMUSDT": "XLM-USD", "NEARUSDT": "NEAR-USD", "ALGOUSDT": "ALGO-USD",
        "FILUSDT": "FIL-USD", "APTUSDT": "APT-USD", "ARBUSDT": "ARB-USD",
        "OPUSDT": "OP-USD", "INJUSDT": "INJ-USD", "SUIUSDT": "SUI-USD",
        "AAVEUSDT": "AAVE-USD", "MKRUSDT": "MKR-USD", "GRTUSDT": "GRT-USD",
        "SANDUSDT": "SAND-USD", "AXSUSDT": "AXS-USD", "HBARUSDT": "HBAR-USD",
        "VETUSDT": "VET-USD", "ICPUSDT": "ICP-USD", "TIAUSDT": "TIA-USD",
        "SEIUSDT": "SEI-USD", "LDOUSDT": "LDO-USD",
        # Not listed on Coinbase: BNB, RUNE, THETA, EGLD, FTM
    }

    def available(self) -> bool:
        try:
            r = self.s.get(f"{self.HOST}/products/BTC-USD", timeout=8)
            return r.status_code == 200
        except Exception:
            return False

    def fetch(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
        product = self.MAP.get(symbol)
        if product is None:
            return pd.DataFrame(columns=COLUMNS)

        frames, cursor = [], pd.Timestamp.utcnow().tz_localize(None).normalize()
        while cursor > start:
            window_start = max(start, cursor - pd.Timedelta(days=299))
            try:
                r = self.s.get(
                    f"{self.HOST}/products/{product}/candles",
                    params={"granularity": 86400,
                            "start": window_start.isoformat(),
                            "end": cursor.isoformat()},
                    timeout=25)
            except Exception:
                break
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            frames.append(pd.DataFrame(
                batch, columns=["time", "low", "high", "open", "close", "volume"]))
            cursor = window_start - pd.Timedelta(days=1)
            time.sleep(0.25)   # Coinbase rate limit is strict

        if not frames:
            return pd.DataFrame(columns=COLUMNS)

        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
        df = df.drop_duplicates("date").set_index("date").sort_index()
        # Coinbase volume is in BASE units — approximate quote volume.
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        df["quote_volume"] = df["volume"] * typical
        for c in COLUMNS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[COLUMNS]


VENUES = [Binance, Coinbase]


def pick_venue(session: requests.Session | None = None) -> Venue:
    for cls in VENUES:
        v = cls(session)
        if v.available():
            return v
    raise RuntimeError(
        "No market data venue reachable. Binance blocks US IPs and Coinbase "
        "did not answer either — check network or run this locally.")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_universe(symbols=None, start="2021-01-01", progress=None):
    """
    Returns (panel_close, panel_quote_volume, report).
    `progress` is an optional callable(i, n, symbol, ok) for UI feedback.
    """
    symbols = list(symbols or UNIVERSE)
    start_ts = pd.Timestamp(start)
    sess = requests.Session()
    venue = pick_venue(sess)

    data, missing, short = {}, [], []
    for i, sym in enumerate(symbols, 1):
        try:
            df = venue.fetch(sym, start_ts)
        except Exception:
            df = pd.DataFrame(columns=COLUMNS)
        ok = (not df.empty) and len(df) >= 120
        if ok:
            data[sym] = df
        elif df.empty:
            missing.append(sym)
        else:
            short.append(f"{sym}({len(df)}d)")
        if progress:
            progress(i, len(symbols), sym, ok)

    if not data:
        raise RuntimeError(f"No usable data from {venue.name}.")

    close = pd.DataFrame({s: d["close"] for s, d in data.items()}).sort_index()
    qvol = pd.DataFrame({s: d["quote_volume"] for s, d in data.items()}).sort_index()

    report = {
        "venue": venue.name,
        "loaded": sorted(data.keys()),
        "missing": missing,
        "too_short": short,
        "start": str(close.index[0].date()),
        "end": str(close.index[-1].date()),
        "n_days": len(close),
        # Survivorship flag: names that only start later than the requested
        # start are late listings, which is the visible half of the bias.
        "late_listings": sorted(
            s for s in data
            if data[s].index[0] > start_ts + pd.Timedelta(days=30)),
    }
    return close, qvol, report
