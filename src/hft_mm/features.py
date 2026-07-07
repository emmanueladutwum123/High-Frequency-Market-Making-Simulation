"""Real-time order book and price feature engineering for the market maker.

Replaces two placeholders from the original notebook: `bid_ask_imbalance` used to
return `np.random.uniform(-1, 1)` (random noise presented as a feature), and
`microprice` used to just average price history. Both are now computed from real
resting order book depth.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from hft_mm.metrics import TICKS_PER_YEAR
from hft_mm.order_book import LimitOrderBook, Side


class MarketDataProcessor:
    """Rolling price/volume/spread history plus live order-book features.

    Each window-size feature becomes available as soon as that window has enough
    history, rather than being gated behind the single largest configured window —
    so a moderate-length demo run actually exercises every feature instead of
    leaving the largest windows permanently empty.
    """

    def __init__(self, window_sizes: Optional[List[int]] = None):
        self.window_sizes = sorted(window_sizes or [20, 100, 500])
        max_window = max(self.window_sizes)
        self.price_history: deque = deque(maxlen=max_window)
        self.volume_history: deque = deque(maxlen=max_window)
        self.spread_history: deque = deque(maxlen=max_window)

    def update(self, lob: LimitOrderBook, trade_volume: int = 0) -> None:
        """Record the current book state for one tick."""
        mid = lob.get_mid_price()
        if mid is None:
            return
        bid, ask = lob.get_best_bid_ask()
        spread = (ask - bid) if (bid is not None and ask is not None) else 0.0
        self.price_history.append(mid)
        self.volume_history.append(trade_volume)
        self.spread_history.append(spread)

    def calculate_features(self, lob: LimitOrderBook) -> Dict[str, float]:
        """Compute the current feature vector from rolling history plus live book depth."""
        if len(self.price_history) < min(self.window_sizes):
            return {}

        features: Dict[str, float] = {}
        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history)
        spreads = np.array(self.spread_history)

        microprice = self.calculate_microprice(lob)
        imbalance = self.calculate_imbalance(lob)
        if microprice is not None:
            features["microprice"] = microprice
        if imbalance is not None:
            features["bid_ask_imbalance"] = imbalance

        for window in self.window_sizes:
            if len(prices) < window:
                continue
            window_prices = prices[-window:]
            returns = np.diff(np.log(window_prices))
            if len(returns) > 0 and np.std(returns) > 0:
                features[f"realized_vol_{window}"] = float(
                    np.std(returns) * np.sqrt(TICKS_PER_YEAR)
                )
            features[f"price_range_{window}"] = float(
                (np.max(window_prices) - np.min(window_prices)) / np.mean(window_prices)
            )

        short_w = self.window_sizes[0]
        long_w = self.window_sizes[1] if len(self.window_sizes) > 1 else short_w

        if len(volumes) >= 2 * short_w:
            recent = np.mean(volumes[-short_w:])
            prior = np.mean(volumes[-2 * short_w : -short_w])
            features["volume_accel"] = float(recent / (prior + 1e-6))

        if len(spreads) >= short_w:
            features["spread_volatility"] = float(np.std(spreads[-short_w:]))

        if len(prices) >= long_w:
            features["momentum"] = float(
                np.mean(prices[-short_w:]) / np.mean(prices[-long_w:]) - 1
            )
            features["z_score"] = float(
                (prices[-1] - np.mean(prices[-long_w:])) / (np.std(prices[-long_w:]) + 1e-6)
            )

        return features

    @staticmethod
    def calculate_microprice(lob: LimitOrderBook) -> Optional[float]:
        """Size-weighted microprice: skews toward the side with less resting size,
        since the thinner side is statistically more likely to be consumed next.
        """
        bid = lob.best_bid()
        ask = lob.best_ask()
        if bid is None or ask is None:
            return None
        bid_size, ask_size = bid.remaining, ask.remaining
        total = bid_size + ask_size
        if total == 0:
            return (bid.price + ask.price) / 2
        return (bid.price * ask_size + ask.price * bid_size) / total

    @staticmethod
    def calculate_imbalance(lob: LimitOrderBook, levels: int = 3) -> Optional[float]:
        """Order-book imbalance in [-1, 1] from real resting depth across the top
        `levels` price levels: positive means more resting bid depth than ask
        depth (buy-side pressure).
        """
        bid_depth = sum(qty for _, qty in lob.get_depth(Side.BID, levels))
        ask_depth = sum(qty for _, qty in lob.get_depth(Side.ASK, levels))
        total = bid_depth + ask_depth
        if total == 0:
            return None
        return (bid_depth - ask_depth) / total
