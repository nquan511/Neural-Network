import numpy as np
from typing import Dict, Any, Optional


def _to_returns(series: np.ndarray, kind: str = "diff") -> np.ndarray:
    """Convert price series to returns.

    kind: "diff" for simple differences, "log" for log returns.
    """
    series = np.asarray(series).astype(float)
    if series.ndim != 1:
        raise ValueError("series must be 1D array of prices")
    if kind == "diff":
        return np.diff(series)
    elif kind == "log":
        eps = 1e-12
        return np.diff(np.log(np.clip(series, eps, None)))
    else:
        raise ValueError(f"Unknown return kind: {kind}")


def _trim_by_lag(a: np.ndarray, b: np.ndarray, lag: int) -> (np.ndarray, np.ndarray):
    """Align arrays for given lag.

    lag > 0 means prediction leads actual by 'lag': compare pred[t] with true[t+lag].
    lag < 0 means prediction lags actual by 'abs(lag)'.
    """
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    if lag >= 0:
        if lag >= n:
            return a[:0], b[:0]
        return a[: n - lag], b[lag:]
    else:
        k = -lag
        if k >= n:
            return a[:0], b[:0]
        return a[k:], b[: n - k]


def _hit_ratio(a: np.ndarray, b: np.ndarray, min_abs_return: Optional[float] = None) -> (float, int):
    """Directional hit ratio: mean(sign(a) == sign(b)).
    Optionally ignore small-magnitude moves below min_abs_return.
    Returns (hit_ratio, n_used).
    """
    if min_abs_return is not None:
        mask = (np.abs(a) >= min_abs_return) | (np.abs(b) >= min_abs_return)
        a = a[mask]
        b = b[mask]
    if len(a) == 0:
        return np.nan, 0
    sa = np.sign(a)
    sb = np.sign(b)
    return float(np.mean(sa == sb)), int(len(a))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    va = np.var(a)
    vb = np.var(b)
    if va <= 0 or vb <= 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def lead_lag_grid(
    y_true_prices: np.ndarray,
    y_pred_prices: np.ndarray,
    max_lag: int = 24,
    return_kind: str = "diff",
    min_abs_return: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute lead–lag metrics across lags in [-max_lag, ..., +max_lag].

    Returns a dict with arrays for lags, hit_ratio, hit_ratio_ci_low/hi (normal approx),
    corr_of_returns, sign_corr (corr of signs), and summary best lags by metrics.

    Conventions:
    - lag > 0: predictions lead actuals by 'lag' steps.
    - lag < 0: predictions lag actuals by 'abs(lag)' steps.
    """
    r_true = _to_returns(np.asarray(y_true_prices).ravel(), kind=return_kind)
    r_pred = _to_returns(np.asarray(y_pred_prices).ravel(), kind=return_kind)

    lags = np.arange(-max_lag, max_lag + 1)
    hit = np.zeros_like(lags, dtype=float)
    n_used = np.zeros_like(lags, dtype=int)
    corr = np.zeros_like(lags, dtype=float)
    sign_corr = np.zeros_like(lags, dtype=float)

    for i, lag in enumerate(lags):
        a, b = _trim_by_lag(r_pred, r_true, lag)
        h, n = _hit_ratio(a, b, min_abs_return=min_abs_return)
        hit[i] = h
        n_used[i] = n
        corr[i] = _corr(a, b)
        # correlation of signs
        if len(a) >= 2:
            sa = np.sign(a)
            sb = np.sign(b)
            sign_corr[i] = _corr(sa, sb)
        else:
            sign_corr[i] = np.nan

    # normal-approx CI for hit ratio
    ci_low = np.full_like(hit, np.nan, dtype=float)
    ci_high = np.full_like(hit, np.nan, dtype=float)
    for i, (p, n) in enumerate(zip(hit, n_used)):
        if n > 0 and np.isfinite(p):
            se = np.sqrt(max(p * (1 - p), 1e-9) / n)
            ci_low[i] = p - 1.96 * se
            ci_high[i] = p + 1.96 * se

    # summaries
    def _best_idx(x: np.ndarray) -> int:
        if not np.any(np.isfinite(x)):
            return -1
        return int(np.nanargmax(x))

    idx_best_hit = _best_idx(hit)
    idx_best_corr = _best_idx(corr)
    best = {
        "best_lag_by_hit": int(lags[idx_best_hit]) if idx_best_hit >= 0 else None,
        "best_hit_ratio": float(hit[idx_best_hit]) if idx_best_hit >= 0 else np.nan,
        "best_hit_n": int(n_used[idx_best_hit]) if idx_best_hit >= 0 else 0,
        "best_lag_by_corr": int(lags[idx_best_corr]) if idx_best_corr >= 0 else None,
        "best_corr": float(corr[idx_best_corr]) if idx_best_corr >= 0 else np.nan,
        "zero_lag_hit": float(hit[lags == 0][0]) if np.any(lags == 0) else np.nan,
        "zero_lag_corr": float(corr[lags == 0][0]) if np.any(lags == 0) else np.nan,
    }

    return {
        "lags": lags,
        "hit_ratio": hit,
        "hit_ratio_ci_low": ci_low,
        "hit_ratio_ci_high": ci_high,
        "n_used": n_used,
        "corr_of_returns": corr,
        "sign_corr": sign_corr,
        "summary": best,
    }


def lead_lag_report(metrics: Dict[str, Any]) -> str:
    s = metrics.get("summary", {})
    lag_hit = s.get("best_lag_by_hit")
    lag_corr = s.get("best_lag_by_corr")
    txt = []
    txt.append("Lead–Lag Summary")
    if lag_hit is not None:
        txt.append(
            f"- Best lag by directional hit: {lag_hit} (pred leads if >0). Hit={s.get('best_hit_ratio'):.3f}, N={s.get('best_hit_n')}"
        )
    if lag_corr is not None:
        txt.append(
            f"- Best lag by return correlation: {lag_corr} (pred leads if >0). Corr={s.get('best_corr'):.3f}"
        )
    if "zero_lag_hit" in s:
        txt.append(f"- Zero-lag hit ratio: {s.get('zero_lag_hit'):.3f}")
    if "zero_lag_corr" in s:
        txt.append(f"- Zero-lag return corr: {s.get('zero_lag_corr'):.3f}")
    return "\n".join(txt)


def plot_lead_lag(metrics: Dict[str, Any], use_plotly: bool = True):
    lags = metrics["lags"]
    hit = metrics["hit_ratio"]
    corr = metrics["corr_of_returns"]
    sign_corr = metrics["sign_corr"]
    ci_low = metrics["hit_ratio_ci_low"]
    ci_high = metrics["hit_ratio_ci_high"]

    if use_plotly:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lags, y=hit, mode="lines+markers", name="Hit Ratio"))
            fig.add_trace(go.Scatter(x=lags, y=corr, mode="lines+markers", name="Return Corr", yaxis="y2"))
            fig.add_trace(go.Scatter(x=lags, y=sign_corr, mode="lines+markers", name="Sign Corr", yaxis="y2"))
            # CI band for hit ratio
            fig.add_trace(go.Scatter(x=np.concatenate([lags, lags[::-1]]),
                                     y=np.concatenate([ci_high, ci_low[::-1]]),
                                     fill='toself', fillcolor='rgba(0,0,255,0.1)',
                                     line=dict(color='rgba(0,0,255,0)'),
                                     name='Hit CI', showlegend=True))

            fig.update_layout(
                title="Lead–Lag Direction Metrics",
                xaxis_title="Lag (pred leads if > 0)",
                yaxis=dict(title="Hit Ratio", range=[0, 1]),
                yaxis2=dict(title="Correlation", overlaying='y', side='right'),
                legend=dict(orientation='h')
            )
            return fig
        except Exception:
            use_plotly = False

    # Fallback to matplotlib if Plotly not available
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(lags, hit, label="Hit Ratio", color="C0")
    ax1.fill_between(lags, ci_low, ci_high, color="C0", alpha=0.15, label="Hit CI")
    ax2.plot(lags, corr, label="Return Corr", color="C1")
    ax2.plot(lags, sign_corr, label="Sign Corr", color="C2")
    ax1.set_xlabel("Lag (pred leads if > 0)")
    ax1.set_ylabel("Hit Ratio")
    ax2.set_ylabel("Correlation")
    ax1.set_ylim(0.0, 1.0)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")
    fig.tight_layout()
    return fig

