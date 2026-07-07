"""Performance metrics for evaluating a market making strategy's backtest run."""

from __future__ import annotations

import numpy as np

# Documented assumption used throughout this project: one simulated tick == one
# second of trading time. A trading year has ~252 days of 6.5 trading hours.
TICKS_PER_YEAR = 252 * 6.5 * 3600


def sharpe_ratio(pnl_series: np.ndarray) -> float:
    """Annualized Sharpe ratio of tick-over-tick P&L changes."""
    pnl_series = np.asarray(pnl_series, dtype=float)
    if len(pnl_series) < 2:
        return 0.0
    returns = np.diff(pnl_series)
    std = np.std(returns)
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(TICKS_PER_YEAR))


def sortino_ratio(pnl_series: np.ndarray) -> float:
    """Annualized Sortino ratio: like Sharpe, but only penalizes downside P&L moves."""
    pnl_series = np.asarray(pnl_series, dtype=float)
    if len(pnl_series) < 2:
        return 0.0
    returns = np.diff(pnl_series)
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 0.0
    if downside_std == 0:
        return 0.0
    return float(np.mean(returns) / downside_std * np.sqrt(TICKS_PER_YEAR))


def max_drawdown(pnl_series: np.ndarray) -> float:
    """Largest peak-to-trough decline in cumulative P&L."""
    pnl_series = np.asarray(pnl_series, dtype=float)
    if len(pnl_series) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(pnl_series)
    drawdowns = running_peak - pnl_series
    return float(np.max(drawdowns))


def fill_rate(n_fills: int, n_quotes: int) -> float:
    """Fraction of posted quotes that were filled (fully or partially)."""
    if n_quotes == 0:
        return 0.0
    return n_fills / n_quotes


def inventory_risk(inventory_series: np.ndarray) -> dict:
    """Time-weighted average and peak absolute inventory held during the run."""
    inventory_series = np.asarray(inventory_series, dtype=float)
    if len(inventory_series) == 0:
        return {"avg_abs_inventory": 0.0, "max_abs_inventory": 0.0}
    return {
        "avg_abs_inventory": float(np.mean(np.abs(inventory_series))),
        "max_abs_inventory": float(np.max(np.abs(inventory_series))),
    }


def adverse_selection(fills: list, mid_series: np.ndarray, horizon: int = 20) -> float:
    """Average post-fill price move against the market maker.

    For each fill, compares the mid price at fill time to the mid price `horizon`
    ticks later. A positive value means prices moved against the maker on average
    (bought right before a drop, or sold right before a rally) — the classic
    picking-off / adverse-selection risk of resting limit orders.
    """
    mid_series = np.asarray(mid_series, dtype=float)
    n = len(mid_series)
    costs = []
    for fill in fills:
        t = fill["tick_index"]
        if t + horizon >= n:
            continue
        move = mid_series[t + horizon] - mid_series[t]
        cost = -move if fill["side"] == "buy" else move
        costs.append(cost)
    return float(np.mean(costs)) if costs else 0.0


def summarize_run(
    pnl_series: np.ndarray,
    inventory_series: np.ndarray,
    fills: list,
    mid_series: np.ndarray,
    n_quotes: int,
    spread_series: np.ndarray = None,
) -> dict:
    """Bundle every metric above into a single results dict for one backtest run."""
    pnl_series = np.asarray(pnl_series, dtype=float)
    summary = {
        "total_pnl": float(pnl_series[-1]) if len(pnl_series) else 0.0,
        "sharpe": sharpe_ratio(pnl_series),
        "sortino": sortino_ratio(pnl_series),
        "max_drawdown": max_drawdown(pnl_series),
        "pnl_volatility": float(np.std(np.diff(pnl_series))) if len(pnl_series) > 1 else 0.0,
        "fill_rate": fill_rate(len(fills), n_quotes),
        "adverse_selection": adverse_selection(fills, mid_series),
        **inventory_risk(inventory_series),
    }
    if spread_series is not None and len(spread_series) > 0:
        summary["avg_spread"] = float(np.mean(spread_series))
    return summary
