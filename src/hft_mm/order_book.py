"""Limit order book with real price-time-priority matching.

This is the core fix over a naive "heap of resting orders" implementation: incoming
orders are actually matched against the opposite side of the book when price allows,
producing `Trade` objects, instead of just being pushed onto a heap that never trades.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Optional


class Side(Enum):
    BID = "bid"
    ASK = "ask"

    @property
    def opposite(self) -> "Side":
        return Side.ASK if self is Side.BID else Side.BID


@dataclass
class Order:
    order_id: int
    side: Side
    price: float
    quantity: int
    timestamp: float
    remaining: int = field(init=False)

    def __post_init__(self):
        self.remaining = self.quantity


@dataclass
class Trade:
    trade_id: int
    price: float
    quantity: int
    timestamp: float
    maker_order_id: int
    taker_order_id: int
    aggressor_side: Side


class LimitOrderBook:
    """Price-time-priority limit order book with a real matching engine.

    Bids sit in a max-heap (price negated), asks in a min-heap; equal prices break
    ties on timestamp so earlier orders fill first. Cancelled/filled orders are
    removed from `orders` immediately but left in the heap (lazy deletion) — every
    heap read skips and permanently pops any entry whose order is no longer live,
    so a cancelled order can never be reported as the best bid/ask or matched against.
    """

    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self._bid_heap: list[tuple[float, float, int]] = []
        self._ask_heap: list[tuple[float, float, int]] = []
        self.orders: dict[int, Order] = {}
        self.depth: dict[Side, dict[float, int]] = {Side.BID: {}, Side.ASK: {}}
        self.trades: list[Trade] = []
        self._order_id_counter = count(1)
        self._trade_id_counter = count(1)

    def next_order_id(self) -> int:
        """Return the next unique order id."""
        return next(self._order_id_counter)

    def _heap_for(self, side: Side) -> list:
        return self._bid_heap if side is Side.BID else self._ask_heap

    def _push(self, order: Order) -> None:
        heap = self._heap_for(order.side)
        key = -order.price if order.side is Side.BID else order.price
        heapq.heappush(heap, (key, order.timestamp, order.order_id))

    def _adjust_depth(self, side: Side, price: float, delta_qty: int) -> None:
        book = self.depth[side]
        book[price] = book.get(price, 0) + delta_qty
        if book[price] <= 0:
            book.pop(price, None)

    def _clean_top(self, side: Side) -> Optional[Order]:
        """Pop stale (cancelled/filled) heap entries and return the live Order
        now resting at the top of `side`, or None if that side is empty."""
        heap = self._heap_for(side)
        while heap:
            _, _, order_id = heap[0]
            order = self.orders.get(order_id)
            if order is None or order.remaining <= 0:
                heapq.heappop(heap)
                continue
            return order
        return None

    def best_bid(self) -> Optional[Order]:
        """Return the highest-priority resting bid Order, or None."""
        return self._clean_top(Side.BID)

    def best_ask(self) -> Optional[Order]:
        """Return the highest-priority resting ask Order, or None."""
        return self._clean_top(Side.ASK)

    def get_best_bid_ask(self) -> tuple[Optional[float], Optional[float]]:
        """Return (best_bid_price, best_ask_price), either None if that side is empty."""
        bid = self.best_bid()
        ask = self.best_ask()
        return (bid.price if bid else None, ask.price if ask else None)

    def get_mid_price(self) -> Optional[float]:
        """Return the mid price, or None if either side of the book is empty."""
        bid, ask = self.get_best_bid_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def get_depth(self, side: Side, levels: int = 5) -> list[tuple[float, int]]:
        """Return up to `levels` (price, quantity) pairs for one side, best price first."""
        book = self.depth[side]
        prices = sorted(book.keys(), reverse=(side is Side.BID))
        return [(p, book[p]) for p in prices[:levels]]

    def cancel_order(self, order_id: int) -> bool:
        """Remove a resting order's liquidity. Returns False if it was already gone."""
        order = self.orders.pop(order_id, None)
        if order is None:
            return False
        self._adjust_depth(order.side, order.price, -order.remaining)
        order.remaining = 0
        return True

    def _match(
        self,
        side: Side,
        quantity: int,
        timestamp: float,
        taker_order_id: int,
        price_limit: Optional[float],
    ) -> tuple[int, list[Trade]]:
        """Match `quantity` on `side` against the opposite side of the book.

        `price_limit=None` matches unconditionally (market-order semantics); otherwise
        matching stops once the opposite side's best price no longer crosses the limit.
        Returns (unfilled_remaining_quantity, trades_generated).
        """
        opposite = side.opposite
        trades: list[Trade] = []
        remaining = quantity

        while remaining > 0:
            resting = self._clean_top(opposite)
            if resting is None:
                break
            if price_limit is not None:
                crosses = (
                    price_limit >= resting.price
                    if side is Side.BID
                    else price_limit <= resting.price
                )
                if not crosses:
                    break

            fill_qty = min(remaining, resting.remaining)
            trade = Trade(
                trade_id=next(self._trade_id_counter),
                price=resting.price,
                quantity=fill_qty,
                timestamp=timestamp,
                maker_order_id=resting.order_id,
                taker_order_id=taker_order_id,
                aggressor_side=side,
            )
            trades.append(trade)
            self.trades.append(trade)

            remaining -= fill_qty
            resting.remaining -= fill_qty
            self._adjust_depth(opposite, resting.price, -fill_qty)

            if resting.remaining <= 0:
                self.orders.pop(resting.order_id, None)

        return remaining, trades

    def add_limit_order(
        self, side: Side, price: float, quantity: int, timestamp: float
    ) -> tuple[Order, list[Trade]]:
        """Submit a limit order: matches against the opposite side while price
        crosses (trading at the resting/maker's price), then rests any remainder
        on the book. Returns the resulting Order (remaining may be 0) and the trades.
        """
        order_id = self.next_order_id()
        remaining, trades = self._match(side, quantity, timestamp, order_id, price)

        order = Order(order_id, side, price, quantity, timestamp)
        order.remaining = remaining
        if remaining > 0:
            self.orders[order_id] = order
            self._adjust_depth(side, price, remaining)
            self._push(order)

        return order, trades

    def add_market_order(
        self, side: Side, quantity: int, timestamp: float
    ) -> tuple[int, list[Trade]]:
        """Submit a market order: takes available liquidity from the opposite side
        regardless of price. Any unfilled quantity is dropped, not rested.
        Returns (filled_quantity, trades).
        """
        order_id = self.next_order_id()
        remaining, trades = self._match(side, quantity, timestamp, order_id, None)
        return quantity - remaining, trades
