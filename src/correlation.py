"""
Sentiment-Price Correlation Engine
  log_event(ticker, headline, geo_score, price)  -> None  (rolling 100-sample log)
  get_pearson(ticker)                            -> dict  {r, p_value, n, data, interpretation}
  get_scatter_data(ticker)                       -> list[dict]  ready for Plotly scatter

Each time the user analyses a headline in the dashboard, app.py calls log_event().
Pearson r is computed between the logged geo_scores and the corresponding
1-bar price return (price relative to the first logged price baseline).
"""

import math
import time
from pathlib import Path

import diskcache

_CACHE_DIR = str(Path(__file__).parent.parent / "data" / ".corr_cache")
_cache     = diskcache.Cache(_CACHE_DIR)
MAX_LOG    = 100        # keep last 100 events per ticker


# ── Event logging ──────────────────────────────────────────────────────────────
def log_event(ticker: str, headline: str, geo_score: float, price: float) -> None:
    """
    Append one (timestamp, headline, geo_score, price) record to the rolling log.
    No TTL — entries persist until evicted by the MAX_LOG rolling window.
    """
    key = f"corr_log:{ticker.upper()}"
    log: list = _cache.get(key, [])
    log.append({
        "ts":        round(time.time(), 1),
        "headline":  headline[:80],
        "geo_score": round(float(geo_score), 4),
        "price":     round(float(price), 4),
    })
    if len(log) > MAX_LOG:
        log = log[-MAX_LOG:]
    _cache.set(key, log)        # no expiry — persistent rolling window


def get_log(ticker: str) -> list[dict]:
    """Return the full rolling log for a ticker (newest last)."""
    return _cache.get(f"corr_log:{ticker.upper()}", [])


def clear_log(ticker: str) -> None:
    _cache.delete(f"corr_log:{ticker.upper()}")


# ── Math helpers ───────────────────────────────────────────────────────────────
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pearson_r(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """
    Returns (r, two-tailed p-value).
    p-value uses t-distribution approximation (valid for n > 5).
    """
    n = len(xs)
    if n < 3:
        return 0.0, 1.0

    mx = sum(xs) / n
    my = sum(ys) / n

    num   = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    ss_x  = sum((x - mx) ** 2 for x in xs)
    ss_y  = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(ss_x * ss_y) if (ss_x > 0 and ss_y > 0) else 0.0

    if denom == 0.0:
        return 0.0, 1.0

    r = max(-1.0, min(1.0, num / denom))
    if abs(r) >= 1.0:
        return round(r, 4), 0.0

    t_stat = r * math.sqrt(n - 2) / math.sqrt(1.0 - r ** 2)
    p_val  = 2.0 * (1.0 - _norm_cdf(abs(t_stat)))
    return round(r, 4), round(p_val, 4)


# ── Public API ─────────────────────────────────────────────────────────────────
def get_pearson(ticker: str) -> dict:
    """
    Compute Pearson correlation between geo_score and price return.

    Price return is expressed as percent change from the first logged price
    (baseline = log[0]['price']).  This turns absolute prices into returns
    so they are comparable across different tickers / price scales.

    Returns:
        {r, p_value, n, interpretation, significant, data}
    """
    log = get_log(ticker)
    n   = len(log)

    if n < 3:
        return {
            "r": None, "p_value": None, "n": n,
            "interpretation": f"Need >= 3 samples (have {n})",
            "significant": False, "data": [],
        }

    base_price = log[0]["price"]
    xs: list[float] = []
    ys: list[float] = []
    data_rows       = []

    for entry in log:
        geo   = entry["geo_score"]
        ret   = ((entry["price"] - base_price) / base_price * 100.0
                 if base_price else 0.0)
        xs.append(geo)
        ys.append(round(ret, 4))
        data_rows.append({
            "geo":        geo,
            "price_ret":  round(ret, 4),
            "price":      entry["price"],
            "headline":   entry["headline"],
            "ts":         entry["ts"],
        })

    r, p = _pearson_r(xs, ys)

    abs_r = abs(r) if r is not None else 0.0
    strength = (
        "Strong"     if abs_r >= 0.70 else
        "Moderate"   if abs_r >= 0.40 else
        "Weak"       if abs_r >= 0.20 else
        "Negligible"
    )
    direction = "positive" if r >= 0 else "negative"
    sig       = (p is not None) and (p < 0.05) and (n >= 10)

    return {
        "r":              r,
        "p_value":        p,
        "n":              n,
        "interpretation": f"{strength} {direction} correlation (r={r:+.3f})",
        "significant":    sig,
        "data":           data_rows,
    }


def get_scatter_data(ticker: str) -> list[dict]:
    """Convenience wrapper — returns just the data list."""
    return get_pearson(ticker).get("data", [])


# ── Bootstrap mode ─────────────────────────────────────────────────────────────
def bootstrap_from_history(ticker: str, df, headline_score: float) -> None:
    """
    Seed the rolling log from the last 30 daily closes when no real events exist.
    Uses the supplied geo_score scaled by the daily return direction as a proxy.
    Only runs if the log has < 3 entries (first-launch bootstrap).
    """
    log = get_log(ticker)
    if len(log) >= 3:
        return

    closes = df["Close"].astype(float)
    sample = closes.tail(30).tolist()
    now    = time.time()

    for i, price in enumerate(sample):
        # Synthetic geo proxy: daily return direction ± small noise around headline_score
        if i > 0:
            ret   = (sample[i] - sample[i - 1]) / sample[i - 1]
            proxy = round(max(-1.0, min(1.0, headline_score * 0.5 + ret * 20)), 3)
        else:
            proxy = round(headline_score * 0.5, 3)
        log.append({
            "ts":        now - (30 - i) * 86400,
            "headline":  f"[bootstrap day -{30 - i}]",
            "geo_score": proxy,
            "price":     round(price, 4),
        })

    _cache.set(f"corr_log:{ticker.upper()}", log[-MAX_LOG:])
