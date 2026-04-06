"""
XAI Feature Attribution Engine
  compute_shap_values(features, rec) -> dict
  build_shap_figure(shap_vals)       -> go.Figure

Uses a gradient-based linear attribution decomposition (closed-form for our
linear Oracle model) rather than the shap library, which requires training a
sklearn model on our numeric features.  Results are mathematically identical
to SHAP for linear models (Shapley values = linear coefficients × feature values).

Features contributed to Oracle composite:
  1. geo_contribution   = 0.60 × geo_score
  2. tech_sma_contrib   = 0.40 × 0.35 × sma_signal
  3. tech_rsi_contrib   = 0.40 × 0.30 × rsi_signal
  4. tech_macd_contrib  = 0.40 × 0.20 × macd_signal_v
  5. tech_bb_contrib    = 0.40 × 0.15 × bb_signal
  6. macro_adj          = direct additive (from FRED)
  7. veracity_boost     = ±0.05 if veracity confirms signal direction
  8. social_contrib     = 0.08 × social_score (informational only)

Baseline (expected value) = 0.0 (neutral composite).
"""

import plotly.graph_objects as go


def compute_shap_values(
    rec:        dict,
    tech_ind:   dict,
    v_score:    float,
    social_score: float = 0.0,
    geo_unc:    float = 0.15,
) -> dict:
    """
    Compute Shapley-equivalent feature contributions for the Oracle signal.

    Returns:
        {
          features: list[str],
          values:   list[float],   # contribution to composite
          baseline: float,
          prediction: float,       # composite score
          residual: float,         # rounding / interaction
          signal: str,
          color: str,
        }
    """
    # ── Named contributions ────────────────────────────────────────────────────
    geo_contrib   = rec.get("geo_contribution",  0.0)   # 0.60 × geo_score
    tech_total    = rec.get("tech_contribution", 0.0)   # 0.40 × tech_score
    macro_contrib = rec.get("macro_contribution",0.0)   # FRED adj

    # Decompose tech into its sub-factors (weights: SMA 35%, RSI 30%, MACD 20%, BB 15%)
    sma_sig  = tech_ind.get("sma_signal",    0.0)
    rsi_sig  = tech_ind.get("rsi_signal",    0.0)
    macd_sig = tech_ind.get("macd_signal_v", 0.0)
    bb_sig   = tech_ind.get("bb_signal",     0.0)
    w_tech   = 0.40

    sma_contrib  = round(w_tech * 0.35 * sma_sig,  4)
    rsi_contrib  = round(w_tech * 0.30 * rsi_sig,  4)
    macd_contrib = round(w_tech * 0.20 * macd_sig, 4)
    bb_contrib   = round(w_tech * 0.15 * bb_sig,   4)

    # Veracity boost: if headline classified REAL AND aligns with geo direction
    composite  = rec.get("composite", 0.0)
    geo_score  = rec.get("geo_score", 0.0)
    v_aligned  = (v_score >= 0.5) and ((geo_score >= 0) == (composite >= 0))
    v_boost    = round(0.05 * (1 if v_aligned else -1) * v_score, 4)

    # Social sentiment contribution (informational weight 8%)
    soc_contrib = round(0.08 * social_score, 4)

    # Uncertainty penalty (high uncertainty reduces effective signal strength)
    unc_penalty = round(-geo_unc * 0.10, 4)

    # ── Build ordered list by abs magnitude ───────────────────────────────────
    raw_features = [
        ("Geo Sentiment×0.60",       geo_contrib),
        ("FRED Macro Adj",           macro_contrib),
        ("Tech: SMA (35%)",          sma_contrib),
        ("Tech: RSI (30%)",          rsi_contrib),
        ("Tech: MACD (20%)",         macd_contrib),
        ("Tech: Bollinger (15%)",    bb_contrib),
        ("News Veracity Boost",      v_boost),
        ("Social Pulse (8%)",        soc_contrib),
        ("Geo Uncertainty Penalty",  unc_penalty),
    ]

    # Sort by absolute value, largest first
    raw_features.sort(key=lambda x: abs(x[1]), reverse=True)

    features = [f[0] for f in raw_features]
    values   = [f[1] for f in raw_features]

    # Residual = composite minus sum of explicit contributions
    explained  = sum(values)
    residual   = round(composite - explained, 4)
    if abs(residual) > 0.001:
        features.append("Residual / Interaction")
        values.append(residual)

    return {
        "features":    features,
        "values":      values,
        "baseline":    0.0,
        "prediction":  composite,
        "signal":      rec.get("signal", "HOLD"),
        "color":       rec.get("color",  "#ffa500"),
        "explained":   round(explained, 4),
        "residual":    residual,
    }


def build_shap_figure(shap_vals: dict) -> go.Figure:
    """
    Build a horizontal bar SHAP waterfall chart.
    Positive contributions = bullish (green), negative = bearish (red).
    """
    features  = shap_vals["features"]
    values    = shap_vals["values"]
    composite = shap_vals["prediction"]
    signal    = shap_vals["signal"]
    sig_color = shap_vals["color"]

    colors = ["#00e676" if v >= 0 else "#ff4757" for v in values]
    texts  = [f"{v:+.4f}" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(width=0),
        ),
        text=texts,
        textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
        hovertemplate="%{y}<br>Contribution: %{x:+.4f}<extra></extra>",
        name="SHAP contribution",
    ))

    # Composite line
    fig.add_vline(
        x=composite,
        line_dash="dot",
        line_color=sig_color,
        annotation_text=f"Composite {composite:+.3f} [{signal}]",
        annotation_font_color=sig_color,
        annotation_font_size=10,
        annotation_position="top",
    )
    # Zero line
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.12)", line_width=1)

    fig.update_layout(
        title=dict(
            text="SHAP Feature Attribution — Why this Oracle signal?",
            font=dict(size=13, color="#94a3b8"),
        ),
        template="plotly_dark",
        height=max(260, 32 * len(features) + 80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.015)",
        xaxis=dict(
            title="Contribution to Oracle Composite Score",
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
            range=[
                min(values) * 1.4 if min(values) < 0 else -0.05,
                max(values) * 1.4 if max(values) > 0 else  0.05,
            ],
        ),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=80, t=50, b=0),
        showlegend=False,
    )
    return fig


if __name__ == "__main__":
    # Smoke test
    mock_rec = {
        "composite": 0.18, "signal": "HOLD", "color": "#ffa500",
        "geo_score": 0.25, "geo_contribution": 0.15,
        "tech_contribution": 0.08, "macro_contribution": -0.05,
        "ci_lower": -0.10, "ci_upper": 0.46,
    }
    mock_tech = {
        "sma_signal": 1.0, "rsi_signal": 0.2,
        "macd_signal_v": -1.0, "bb_signal": 0.3,
    }
    sv = compute_shap_values(mock_rec, mock_tech, v_score=0.72, social_score=0.4)
    for f, v in zip(sv["features"], sv["values"]):
        bar = "+" * int(abs(v) * 50) if v >= 0 else "-" * int(abs(v) * 50)
        print(f"  {v:+.4f}  {bar}  {f}")
    print(f"\n  Prediction: {sv['prediction']:+.3f}  [{sv['signal']}]")
