"""Market making strategies: a correctly-derived Avellaneda-Stoikov quoter, and a
naive fixed-spread baseline used to quantify how much the optimal strategy adds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from hft_mm.metrics import TICKS_PER_YEAR

DEFAULT_VOLATILITY = 0.0005  # fallback per-tick sigma before any realized-vol feature is ready


@dataclass
class Quote:
    bid_price: Optional[float]
    bid_qty: int
    ask_price: Optional[float]
    ask_qty: int


def _snap(price: float, tick_size: float) -> float:
    return round(price / tick_size) * tick_size


class AvellanedaStoikovMarketMaker:
    """Optimal market maker from Avellaneda & Stoikov (2008), 'High-frequency
    trading in a limit order book'.

        reservation price:  r(t) = s - q * gamma * sigma^2 * (T - t)
        optimal spread:      delta = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)

    `q` is current inventory, `sigma` the (online-estimated) per-tick volatility of
    the mid price, `gamma` risk aversion, `k` the exponential decay of order-arrival
    intensity with distance from mid (a calibration input in the original paper,
    exposed here as a constructor parameter), and `T - t` the number of ticks left
    in the session — both the inventory skew and the risk-aversion component of the
    spread shrink to zero as the close approaches, mechanically flattening the
    strategy into a symmetric mid-quoter right at T.

    `sigma` must be on the same time scale as `T - t` (both "per tick" here). The
    `realized_vol_*` features are annualized for readability elsewhere, so this
    class de-annualizes them internally before using them in the formula above —
    a unit mismatch here is a classic way to silently break this model.
    """

    def __init__(
        self,
        risk_aversion: float = 0.1,
        order_arrival_decay: float = 5.0,
        quote_size: int = 50,
        max_position: int = 500,
        fallback_volatility: float = DEFAULT_VOLATILITY,
        min_spread_ticks: int = 1,
    ):
        self.gamma = risk_aversion
        self.k = order_arrival_decay
        self.quote_size = quote_size
        self.max_position = max_position
        self.fallback_volatility = fallback_volatility
        self.min_spread_ticks = min_spread_ticks

    def quote(
        self,
        mid_price: float,
        features: Dict[str, float],
        inventory: int,
        t: float,
        T: float,
        tick_size: float = 0.01,
    ) -> Quote:
        """Compute this tick's bid/ask given the current book, inventory, and
        position in the trading session (`t` ticks elapsed out of `T`)."""
        time_left = max(T - t, 0.0)
        sigma = self._per_tick_volatility(features)

        reservation_price = mid_price - inventory * self.gamma * sigma**2 * time_left
        spread = self.gamma * sigma**2 * time_left + (2 / self.gamma) * math.log(
            1 + self.gamma / self.k
        )
        spread = max(spread, self.min_spread_ticks * tick_size)

        bid_price = _snap(reservation_price - spread / 2, tick_size)
        ask_price = _snap(reservation_price + spread / 2, tick_size)
        if bid_price >= ask_price:
            ask_price = bid_price + tick_size

        bid_qty = self.quote_size if inventory < self.max_position and bid_price > 0 else 0
        ask_qty = self.quote_size if inventory > -self.max_position and ask_price > 0 else 0

        return Quote(
            bid_price if bid_qty else None,
            bid_qty,
            ask_price if ask_qty else None,
            ask_qty,
        )

    def _per_tick_volatility(self, features: Dict[str, float]) -> float:
        """De-annualize the realized-vol feature back to per-tick units for the
        pricing formula; falls back to a fixed constant before any window is ready.
        """
        for window in (100, 500, 20, 1000):
            key = f"realized_vol_{window}"
            if key in features and features[key] > 0:
                return features[key] / math.sqrt(TICKS_PER_YEAR)
        return self.fallback_volatility


class NaiveMarketMaker:
    """Baseline strategy: a fixed symmetric spread around the mid price, no
    inventory skew and no volatility adjustment. This is the control group used
    to quantify how much the Avellaneda-Stoikov strategy actually improves on —
    same book, same order flow, same seed, only the quoting logic differs.
    """

    def __init__(self, spread: float = 0.10, quote_size: int = 50, max_position: int = 500):
        self.spread = spread
        self.quote_size = quote_size
        self.max_position = max_position

    def quote(
        self,
        mid_price: float,
        features: Dict[str, float],
        inventory: int,
        t: float,
        T: float,
        tick_size: float = 0.01,
    ) -> Quote:
        bid_price = _snap(mid_price - self.spread / 2, tick_size)
        ask_price = _snap(mid_price + self.spread / 2, tick_size)
        bid_qty = self.quote_size if inventory < self.max_position and bid_price > 0 else 0
        ask_qty = self.quote_size if inventory > -self.max_position and ask_price > 0 else 0
        return Quote(
            bid_price if bid_qty else None,
            bid_qty,
            ask_price if ask_qty else None,
            ask_qty,
        )
