"""Synthetic market environment: a Brownian-motion fundamental price plus
Poisson-arrival noise-trader order flow that actually drives the limit order book.

This replaces the original notebook's `generate_market_data()` DataFrame, which was
built with realistic-looking volatility clustering but never actually consumed by
the backtest loop. Here, the environment's order flow is what the book — and any
resting market-maker quotes — actually trade against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from hft_mm.order_book import LimitOrderBook, Side, Trade


@dataclass
class TickEvent:
    tick_index: int
    timestamp: float
    fundamental_price: float


class MarketEnvironment:
    """Drives a LimitOrderBook with synthetic noise-trader order flow around a
    Brownian-motion fundamental price — the same kind of setup used to validate
    the Avellaneda-Stoikov model in the original paper and in most academic HFT
    literature: no external data dependency, fully reproducible from a seed.

    Each tick:
      1. The fundamental price takes one Brownian step (`dS = sigma * dW`).
      2. With probability `limit_order_prob`, a noise trader posts a resting limit
         order at a random offset from the fundamental — this is what gives the
         book real depth, which the microprice/imbalance features depend on.
      3. With probability `market_order_prob`, a noise trader submits a market
         order that crosses the book. This is what actually generates fills,
         including fills against a market maker's resting quotes, through the
         order book's real price-time-priority matching engine — not a fixed
         coin-flip fill probability disconnected from the state of the book.
    """

    def __init__(
        self,
        lob: LimitOrderBook,
        initial_price: float = 100.0,
        sigma: float = 0.01,
        limit_order_prob: float = 0.5,
        market_order_prob: float = 0.15,
        limit_offset_scale: float = 0.15,
        order_size_mean: int = 100,
        seed: Optional[int] = None,
    ):
        self.lob = lob
        self.price = initial_price
        self.sigma = sigma
        self.limit_order_prob = limit_order_prob
        self.market_order_prob = market_order_prob
        self.limit_offset_scale = limit_offset_scale
        self.order_size_mean = order_size_mean
        self.rng = np.random.default_rng(seed)

    def seed_book(self, levels: int = 5, spread: float = 0.10) -> None:
        """Prime the book with resting orders on each side so it isn't empty at tick 0."""
        for i in range(levels):
            offset = spread / 2 + i * self.lob.tick_size
            self.lob.add_limit_order(Side.BID, self.price - offset, self.order_size_mean, 0.0)
            self.lob.add_limit_order(Side.ASK, self.price + offset, self.order_size_mean, 0.0)

    def step(self, tick_index: int) -> Tuple[TickEvent, List[Trade]]:
        """Advance the fundamental price and inject one tick's worth of noise-trader
        order flow into the book. Returns the tick event and every trade generated
        this tick (including any that filled a resting market-maker quote)."""
        self.price += self.rng.normal(0, self.sigma)
        self.price = max(self.price, self.lob.tick_size)

        trades: List[Trade] = []

        if self.rng.random() < self.limit_order_prob:
            trades += self._submit_noise_limit_order(tick_index)

        if self.rng.random() < self.market_order_prob:
            trades += self._submit_noise_market_order(tick_index)

        event = TickEvent(
            tick_index=tick_index, timestamp=float(tick_index), fundamental_price=self.price
        )
        return event, trades

    def _submit_noise_limit_order(self, tick_index: int) -> List[Trade]:
        side = Side.BID if self.rng.random() < 0.5 else Side.ASK
        offset = abs(self.rng.exponential(self.limit_offset_scale)) + self.lob.tick_size
        raw_price = self.price - offset if side is Side.BID else self.price + offset
        price = round(raw_price / self.lob.tick_size) * self.lob.tick_size
        qty = max(1, int(self.rng.poisson(self.order_size_mean)))
        _, trades = self.lob.add_limit_order(side, price, qty, float(tick_index))
        return trades

    def _submit_noise_market_order(self, tick_index: int) -> List[Trade]:
        side = Side.BID if self.rng.random() < 0.5 else Side.ASK
        qty = max(1, int(self.rng.poisson(self.order_size_mean)))
        _, trades = self.lob.add_market_order(side, qty, float(tick_index))
        return trades
