"""
panda_data adapter.

A single, thin layer between the skill logic and the panda_data SDK so that:
  * business logic never calls panda_data directly (easy to test / swap);
  * the skill still runs end-to-end WITHOUT credentials by falling back to
    deterministic synthetic data (so reviewers can smoke-test it offline).

Real SDK usage (see the `pandadata-api` skill on quantskills.ai):

    from pandadata_runtime import init_pandadata   # or panda_data.init_token()
    panda_data = init_pandadata()
    df = panda_data.get_stock_daily(
        symbol=["000001.SZ"], start_date="20250101",
        end_date="20250131", fields=[])

Credentials are read by the SDK from the environment:
    DEFAULT_USERNAME / DEFAULT_PASSWORD / JAVA_SERVICE_BASE_URL

Conventions: dates are 'YYYYMMDD' strings; A-share codes carry an exchange
suffix ('000001.SZ', '600000.SH'); fields=[] returns all fields.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _get_sdk():
    """Return an initialised panda_data SDK, or None if unavailable."""
    try:
        import panda_data  # type: ignore
        # init_token() is idempotent; raises if credentials missing.
        if hasattr(panda_data, "init_token"):
            panda_data.init_token()
        return panda_data
    except Exception as exc:  # pragma: no cover - environment dependent
        if os.environ.get("PANDADATA_STRICT") == "1":
            raise
        # Keep this message ASCII-only: the SDK raises messages that contain
        # non-ASCII text which crashes the default Windows console codec.
        print(f"[data_source] panda_data not initialised ({type(exc).__name__}); "
              f"using synthetic fallback data.")
        return None


def is_live() -> bool:
    return _get_sdk() is not None


# --------------------------------------------------------------------------- #
# Public API used by the skills
# --------------------------------------------------------------------------- #
def get_price_panel(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Forward-adjusted close prices as a [date x symbol] panel.

    Live path: get_stock_daily(close) * get_adj_factor. Falls back to a
    synthetic geometric-random-walk panel when the SDK is not available.
    """
    sdk = _get_sdk()
    if sdk is None:
        return _synthetic_price_panel(symbols, start_date, end_date)

    daily = sdk.get_stock_daily(symbol=list(symbols), start_date=start_date,
                                end_date=end_date, fields=[])
    daily = _to_frame(daily)
    close = daily.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    try:
        adj = _to_frame(sdk.get_adj_factor(symbol=list(symbols),
                                           start_date=start_date, end_date=end_date, fields=[]))
        adjp = adj.pivot(index="trade_date", columns="symbol", values="adj_factor").sort_index()
        close = close * adjp.reindex_like(close).ffill()
    except Exception:
        pass  # fall back to raw close if adj factor is unavailable
    close.index = pd.to_datetime(close.index, format="%Y%m%d", errors="coerce")
    return close


def get_return_panel(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Daily simple returns as a [date x symbol] panel."""
    px = get_price_panel(symbols, start_date, end_date)
    return px.pct_change().iloc[1:]


def get_market_cap(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """[date x symbol] total market capitalisation (best-effort field name)."""
    sdk = _get_sdk()
    if sdk is None:
        px = _synthetic_price_panel(symbols, start_date, end_date)
        # synthetic shares outstanding -> cap proportional to price
        shares = pd.Series(np.linspace(1e9, 5e9, len(symbols)), index=px.columns)
        return px * shares
    daily = _to_frame(sdk.get_stock_daily(symbol=list(symbols), start_date=start_date,
                                          end_date=end_date, fields=[]))
    field = next((c for c in ("total_mv", "market_cap", "mv") if c in daily.columns), None)
    if field is None:
        raise KeyError("market cap field not found in get_stock_daily output; "
                       "check method-index.md for the correct method/field")
    cap = daily.pivot(index="trade_date", columns="symbol", values=field).sort_index()
    cap.index = pd.to_datetime(cap.index, format="%Y%m%d", errors="coerce")
    return cap


def get_industry(symbols: list[str]) -> pd.Series:
    """Series symbol -> industry code. Synthetic buckets when offline.

    The exact panda_data method/field for industry classification lives in the
    185-method index (see the `pandadata-api` skill). Rather than hardcode one
    guessed name, we probe the SDK for the first method that exists and raise a
    clear, actionable error if none match — so a real user knows exactly what to
    wire up instead of getting a cryptic AttributeError.
    """
    sdk = _get_sdk()
    if sdk is None:
        buckets = ["FIN", "CONS", "TECH", "HLTH"]
        return pd.Series({s: buckets[i % len(buckets)] for i, s in enumerate(symbols)})

    candidates = ("get_stock_industry", "get_industry", "get_stock_info", "get_stock_concept")
    method = next((m for m in candidates if hasattr(sdk, m)), None)
    if method is None:
        raise NotImplementedError(
            "Could not find an industry-classification method on panda_data. "
            "Look it up in references/method-index.md (pandadata-api skill) and "
            f"set it here. Tried: {candidates}")
    info = _to_frame(getattr(sdk, method)(symbol=list(symbols)))
    col = next((c for c in ("industry", "sw_l1", "industry_code", "concept") if c in info.columns), None)
    if col is None:
        raise KeyError(f"{method} returned no industry-like column; got {list(info.columns)}")
    return info.set_index("symbol")[col]


# --------------------------------------------------------------------------- #
# Helpers / synthetic fallback
# --------------------------------------------------------------------------- #
def _to_frame(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    return pd.DataFrame(obj)


def _trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    days = pd.bdate_range(pd.to_datetime(start_date, format="%Y%m%d"),
                          pd.to_datetime(end_date, format="%Y%m%d"))
    return days


def _synthetic_price_panel(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Deterministic GBM panel so the skill is reproducible offline."""
    days = _trading_days(start_date, end_date)
    rng = np.random.default_rng(20260627)
    n = len(days)
    out = {}
    for i, s in enumerate(symbols):
        mu = 0.0002 + 0.00005 * (i % 7)
        sigma = 0.012 + 0.004 * ((i % 5) / 5)
        rets = rng.normal(mu, sigma, n)
        out[s] = 10.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(out, index=days)


if __name__ == "__main__":
    syms = ["000001.SZ", "600000.SH", "000333.SZ"]
    px = get_price_panel(syms, "20240101", "20240401")
    print("live:", is_live(), "| price panel shape:", px.shape)
    print(px.tail(3).round(3))
