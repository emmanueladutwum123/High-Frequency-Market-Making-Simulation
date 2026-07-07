"""Ties the market environment, order book, feature engineering, and a market
making strategy together into a repeatable backtest with correct P&L accounting.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from hft_mm.features import MarketDataProcessor
from hft_mm.order_book import LimitOrderBook, Side
from hft_mm.simulator import MarketEnvironment

FEE_RATE = 0.0002  # 2 bps per filled share, charged to the market maker


class Backtester:
    """Runs a market-making strategy against a MarketEnvironment for `n_ticks`,
    recording P&L, inventory, spread, and fills tick-by-tick.

    P&L accounting: `cash` moves on every fill (buy: cash -= price*qty*(1+fee);
    sell: cash += price*qty*(1-fee)); total P&L at any tick is
    `cash + inventory * mid_price` (mark-to-market). Every fill comes from a real
    `Trade` the matching engine returned, with an unambiguous side determined from
    whether the strategy's own order was the resting maker or the crossing taker —
    this replaces the original notebook's inverted fill-direction bug, which
    guessed the fill side from `not filled_quote.is_bid` instead of a real trade.

    Each tick: the environment advances first (so this tick's noise-trader flow can
    fill quotes resting from the *previous* tick), then the strategy re-quotes via
    cancel-replace — resting orders don't just vanish between ticks, and a
    strategy can only get credit for fills that happened while its quote was live.
    """

    def __init__(
        self,
        strategy,
        n_ticks: int = 20_000,
        tick_size: float = 0.01,
        initial_price: float = 100.0,
        sigma: float = 0.01,
        seed: Optional[int] = None,
        fee_rate: float = FEE_RATE,
    ):
        # sigma default calibrated so per-tick price noise stays smaller than a
        # typical quoted spread: at sigma=0.05 the fundamental moves through
        # quoted price levels almost every tick, so *every* strategy (including
        # the naive baseline) gets chronically picked off and loses money
        # regardless of quoting logic — market making only has an edge to earn
        # when quoting is fast relative to how quickly the "true" price moves.
        self.strategy = strategy
        self.n_ticks = n_ticks
        self.tick_size = tick_size
        self.fee_rate = fee_rate

        self.lob = LimitOrderBook(tick_size=tick_size)
        self.env = MarketEnvironment(self.lob, initial_price=initial_price, sigma=sigma, seed=seed)
        self.features = MarketDataProcessor()

        self.cash = 0.0
        self.inventory = 0
        self.bid_order_id: Optional[int] = None
        self.ask_order_id: Optional[int] = None

        self.pnl_series: List[float] = []
        self.inventory_series: List[float] = []
        self.spread_series: List[float] = []
        self.mid_series: List[float] = []
        self.fills: List[dict] = []
        self.n_quotes_posted = 0

    def run(self) -> Dict[str, object]:
        """Run the full backtest and return the recorded series/fills."""
        self.env.seed_book()

        for t in range(self.n_ticks):
            _, trades = self.env.step(t)
            self._process_fills(t, trades)

            self.features.update(self.lob)
            feature_vec = self.features.calculate_features(self.lob)

            self._cancel_resting_quotes()

            mid = self.lob.get_mid_price()
            if mid is not None:
                quote = self.strategy.quote(
                    mid_price=mid,
                    features=feature_vec,
                    inventory=self.inventory,
                    t=t,
                    T=self.n_ticks,
                    tick_size=self.tick_size,
                )
                self._post_quotes(quote, t)

            self._record_tick()

        return {
            "pnl_series": np.array(self.pnl_series),
            "inventory_series": np.array(self.inventory_series),
            "spread_series": np.array(self.spread_series),
            "mid_series": np.array(self.mid_series),
            "fills": self.fills,
            "n_quotes": self.n_quotes_posted,
        }

    def _process_fills(self, tick_index: int, trades: list) -> None:
        for trade in trades:
            if trade.maker_order_id == self.bid_order_id or trade.taker_order_id == self.bid_order_id:
                side = "buy"
            elif trade.maker_order_id == self.ask_order_id or trade.taker_order_id == self.ask_order_id:
                side = "sell"
            else:
                continue

            if side == "buy":
                self.cash -= trade.price * trade.quantity * (1 + self.fee_rate)
                self.inventory += trade.quantity
            else:
                self.cash += trade.price * trade.quantity * (1 - self.fee_rate)
                self.inventory -= trade.quantity

            self.fills.append(
                {
                    "tick_index": tick_index,
                    "side": side,
                    "price": trade.price,
                    "quantity": trade.quantity,
                }
            )

    def _cancel_resting_quotes(self) -> None:
        if self.bid_order_id is not None:
            self.lob.cancel_order(self.bid_order_id)
            self.bid_order_id = None
        if self.ask_order_id is not None:
            self.lob.cancel_order(self.ask_order_id)
            self.ask_order_id = None

    def _post_quotes(self, quote, tick_index: int) -> None:
        if quote.bid_price is not None and quote.bid_qty > 0:
            order, trades = self.lob.add_limit_order(
                Side.BID, quote.bid_price, quote.bid_qty, float(tick_index)
            )
            self.bid_order_id = order.order_id
            self._process_fills(tick_index, trades)
            if order.remaining <= 0:
                self.bid_order_id = None
            self.n_quotes_posted += 1

        if quote.ask_price is not None and quote.ask_qty > 0:
            order, trades = self.lob.add_limit_order(
                Side.ASK, quote.ask_price, quote.ask_qty, float(tick_index)
            )
            self.ask_order_id = order.order_id
            self._process_fills(tick_index, trades)
            if order.remaining <= 0:
                self.ask_order_id = None
            self.n_quotes_posted += 1

    def _record_tick(self) -> None:
        mid = self.lob.get_mid_price()
        bid, ask = self.lob.get_best_bid_ask()
        spread = (ask - bid) if (bid is not None and ask is not None) else 0.0
        mark_price = mid if mid is not None else (self.mid_series[-1] if self.mid_series else 0.0)

        self.pnl_series.append(self.cash + self.inventory * mark_price)
        self.inventory_series.append(self.inventory)
        self.spread_series.append(spread)
        self.mid_series.append(mark_price)
