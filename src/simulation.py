"""
Monte Carlo Simulation Engine
  run_monte_carlo(df, geo_score, tech_score, n_paths, horizon) -> dict
    {paths, percentiles, last_price, horizon, stats, cone_data}

Methodology:
  1. Fit GBM parameters (mu, sigma) from historical log-returns.
  2. Geo-score biases the drift: bearish geo -> lower mu, bullish -> higher mu.
  3. ATR-14% biases sigma: high ATR -> wider dispersion.
  4. Run n_paths simulations of horizon daily steps using vectorised numpy.
  5. Return percentile bands (5/25/50/75/95) for the Probability Cone chart.

No external dependencies beyond numpy — keeps startup fast.
"""

import math
import random
from pathlib import Path

import numpy as np
import diskcache

_CACHE_DIR = str(Path(__file__).parent.parent / "data" / ".sim_cache")
_cache     = diskcache.Cache(_CACHE_DIR)
_TTL       = 3_600   # 1 h — simulation results rotate with market conditions

# ── Core simulator ─────────────────────────────────────────────────────────────
def run_monte_carlo(
    df,
    geo_score:   float = 0.0,
    tech_score:  float = 0.0,
    atr_pct:     float | None = None,
    n_paths:     int = 1_000,
    horizon:     int = 30,
    seed:        int | None = 42,
) -> dict:
    """
    Run Monte Carlo simulation for the given OHLCV DataFrame.

    Args:
        df          : pandas DataFrame with at least a 'Close' column
        geo_score   : Gemini geopolitical sentiment (-1 to +1) — biases drift
        tech_score  : technical composite (-1 to +1) — secondary drift bias
        atr_pct     : ATR-14 as % of price — biases volatility (None = auto)
        n_paths     : number of simulation paths (default 1,000)
        horizon     : forecast horizon in trading days (default 30)
        seed        : random seed for reproducibility (None = random)

    Returns:
        {
          last_price, horizon, n_paths,
          paths:      np.ndarray (n_paths x horizon+1),
          percentiles: {5, 25, 50, 75, 95} -> list[float],
          dates:      list of offsets [0..horizon],
          stats:      {mu, sigma, geo_adj, vol_adj, prob_up, exp_return},
          cone_data:  dict ready for Plotly,
          warning:    str | None,
        }
    """
    closes     = df["Close"].astype(float).dropna()
    last_price = float(closes.iloc[-1])
    n          = len(closes)

    if n < 30:
        return {"error": f"Need >= 30 history bars, have {n}"}

    # ── 1. Historical GBM parameters (252-day annualised) ────────────────────
    log_rets  = np.log(closes / closes.shift(1)).dropna().values
    hist_mu   = float(np.mean(log_rets))      # daily
    hist_sig  = float(np.std(log_rets, ddof=1))

    # ── 2. Geo + Tech drift adjustment ───────────────────────────────────────
    # geo_score in [-1,1] shifts daily drift by up to ±0.0005 (≈ ±12.5% annual)
    # tech_score adds smaller secondary bias ±0.0002
    geo_adj  = geo_score  * 0.0005
    tech_adj = tech_score * 0.0002
    mu       = hist_mu + geo_adj + tech_adj

    # ── 3. ATR-based volatility scaling ──────────────────────────────────────
    # If ATR% is supplied and differs significantly from implied sigma, blend.
    if atr_pct is not None and atr_pct > 0:
        atr_daily_sig = atr_pct / 100.0 / math.sqrt(252)  # rough daily vol from ATR%
        sigma = 0.6 * hist_sig + 0.4 * atr_daily_sig
    else:
        sigma = hist_sig
    sigma = max(sigma, 0.0001)   # floor

    # ── 4. Simulate paths (vectorised) ────────────────────────────────────────
    rng   = np.random.default_rng(seed)
    dt    = 1.0                          # daily steps
    drift = (mu - 0.5 * sigma ** 2) * dt
    # shape: (n_paths, horizon)
    Z      = rng.standard_normal((n_paths, horizon))
    shocks = drift + sigma * math.sqrt(dt) * Z
    # log-price paths, starting from 0
    log_paths = np.cumsum(shocks, axis=1)
    # prepend 0 (starting point) -> shape (n_paths, horizon+1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
    paths     = last_price * np.exp(log_paths)

    # ── 5. Percentile bands ────────────────────────────────────────────────────
    pctiles = {}
    for p in [5, 10, 25, 50, 75, 90, 95]:
        pctiles[p] = np.percentile(paths, p, axis=0).tolist()

    # ── 6. Statistics ─────────────────────────────────────────────────────────
    final_prices    = paths[:, -1]
    prob_up         = float(np.mean(final_prices > last_price))
    exp_return      = float(np.mean(final_prices / last_price - 1) * 100)
    exp_return_med  = float((np.median(final_prices) / last_price - 1) * 100)
    loss_5pct       = float(np.percentile(final_prices, 5))
    gain_95pct      = float(np.percentile(final_prices, 95))

    # Volatility squeeze warning in simulation context
    warning = None
    if sigma > hist_sig * 1.5:
        warning = f"High-volatility regime detected (sigma {sigma:.4f} vs hist {hist_sig:.4f})"

    # ── 7. Cone data (ready for Plotly) ───────────────────────────────────────
    dates = list(range(horizon + 1))   # 0..horizon (offsets from today)

    cone_data = {
        "dates":   dates,
        "p05":     pctiles[5],
        "p10":     pctiles[10],
        "p25":     pctiles[25],
        "p50":     pctiles[50],
        "p75":     pctiles[75],
        "p90":     pctiles[90],
        "p95":     pctiles[95],
        "last":    last_price,
        "n_paths": n_paths,
    }

    return {
        "last_price":   last_price,
        "horizon":      horizon,
        "n_paths":      n_paths,
        "paths":        paths,            # full matrix if caller needs it
        "percentiles":  pctiles,
        "dates":        dates,
        "cone_data":    cone_data,
        "warning":      warning,
        "stats": {
            "hist_mu":       round(hist_mu, 6),
            "hist_sigma":    round(hist_sig, 6),
            "mu":            round(mu, 6),
            "sigma":         round(sigma, 6),
            "geo_adj":       round(geo_adj, 6),
            "tech_adj":      round(tech_adj, 6),
            "prob_up":       round(prob_up, 4),
            "exp_return":    round(exp_return, 3),
            "exp_return_med":round(exp_return_med, 3),
            "loss_5pct":     round(loss_5pct, 2),
            "gain_95pct":    round(gain_95pct, 2),
            "annual_vol":    round(sigma * math.sqrt(252) * 100, 2),
        },
        "error": None,
    }


def run_monte_carlo_cached(
    ticker:     str,
    df,
    geo_score:  float,
    tech_score: float,
    atr_pct:    float | None,
    n_paths:    int = 1_000,
    horizon:    int = 30,
    force:      bool = False,
) -> dict:
    """
    Cached wrapper — reruns only when inputs change materially.
    """
    geo_r   = round(geo_score, 1)
    tech_r  = round(tech_score, 1)
    atr_r   = round(atr_pct or 0, 1)
    from datetime import datetime
    today   = datetime.now().strftime("%Y-%m-%d")
    key     = f"mc:{ticker}:{today}:{geo_r}:{tech_r}:{atr_r}:{n_paths}:{horizon}"

    if not force and key in _cache:
        cached = _cache[key]
        cached["cached"] = True
        return cached

    result = run_monte_carlo(df, geo_score, tech_score, atr_pct, n_paths, horizon)
    result["cached"] = False
    if result.get("error") is None:
        # Don't store the full paths matrix in cache (large); store cone only
        result_to_cache = {k: v for k, v in result.items() if k != "paths"}
        _cache.set(key, result_to_cache, expire=_TTL)
    return result


def build_cone_figure(cone_data: dict, ticker: str, cur: str = "$") -> object:
    """
    Build a Plotly figure for the Probability Cone.
    Returns a go.Figure.
    """
    import plotly.graph_objects as go

    d    = cone_data
    x    = d["dates"]
    last = d["last"]

    fig = go.Figure()

    # 90% band (lightest)
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=d["p95"] + d["p05"][::-1],
        fill="toself",
        fillcolor="rgba(99,102,241,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% band",
        hoverinfo="skip",
        showlegend=True,
    ))
    # 80% band
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=d["p90"] + d["p10"][::-1],
        fill="toself",
        fillcolor="rgba(99,102,241,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% band",
        hoverinfo="skip",
        showlegend=True,
    ))
    # 50% band
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=d["p75"] + d["p25"][::-1],
        fill="toself",
        fillcolor="rgba(99,102,241,0.22)",
        line=dict(color="rgba(0,0,0,0)"),
        name="50% band",
        hoverinfo="skip",
        showlegend=True,
    ))
    # Median path
    fig.add_trace(go.Scatter(
        x=x, y=d["p50"],
        mode="lines",
        line=dict(color="#a5b4fc", width=2, dash="dot"),
        name="Median (p50)",
        hovertemplate=f"Day %{{x}}<br>Median: {cur}%{{y:,.2f}}<extra></extra>",
    ))
    # 5th percentile (worst case)
    fig.add_trace(go.Scatter(
        x=x, y=d["p05"],
        mode="lines",
        line=dict(color="#ff4757", width=1.2, dash="dot"),
        name="5th pctile (bear)",
        hovertemplate=f"Day %{{x}}<br>Bear: {cur}%{{y:,.2f}}<extra></extra>",
    ))
    # 95th percentile (best case)
    fig.add_trace(go.Scatter(
        x=x, y=d["p95"],
        mode="lines",
        line=dict(color="#00e676", width=1.2, dash="dot"),
        name="95th pctile (bull)",
        hovertemplate=f"Day %{{x}}<br>Bull: {cur}%{{y:,.2f}}<extra></extra>",
    ))
    # Current price marker
    fig.add_hline(
        y=last, line_dash="dot", line_color="rgba(255,255,255,0.25)",
        annotation_text=f"Current {cur}{last:,.2f}",
        annotation_font_color="#94a3b8",
        annotation_font_size=10,
    )

    fig.update_layout(
        title=dict(
            text=f"{ticker} — Monte Carlo Probability Cone ({d['n_paths']:,} paths, 30-day horizon)",
            font=dict(size=13, color="#94a3b8"),
        ),
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.015)",
        xaxis=dict(
            title="Trading Days from Today",
            gridcolor="rgba(255,255,255,0.04)",
        ),
        yaxis=dict(
            title=f"Price ({cur})",
            side="right",
            gridcolor="rgba(255,255,255,0.04)",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=1.08,
            font=dict(size=10),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest import load_or_fetch

    df  = load_or_fetch("SPY")
    res = run_monte_carlo(df, geo_score=-0.3, tech_score=0.2, atr_pct=1.6)
    s   = res["stats"]
    print(f"Last price: ${res['last_price']:.2f}")
    print(f"30-day forecast:  mu={s['mu']:.6f}  sigma={s['sigma']:.6f}")
    print(f"Prob up: {s['prob_up']:.1%}  |  E[return]: {s['exp_return']:+.2f}%")
    print(f"5th pctile: ${s['loss_5pct']:.2f}  |  95th: ${s['gain_95pct']:.2f}")
    print(f"Annual vol: {s['annual_vol']:.1f}%")
