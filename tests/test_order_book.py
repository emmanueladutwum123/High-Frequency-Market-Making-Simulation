"""Tests for the LimitOrderBook matching engine — the part of the original
notebook that had no crossing logic and never actually removed cancelled orders
from the heap.
"""

from hft_mm.order_book import LimitOrderBook, Side


def test_non_crossing_orders_rest_on_book():
    lob = LimitOrderBook()
    lob.add_limit_order(Side.BID, 99.00, 100, 0.0)
    lob.add_limit_order(Side.ASK, 101.00, 100, 0.0)

    bid, ask = lob.get_best_bid_ask()
    assert bid == 99.00
    assert ask == 101.00
    assert lob.trades == []


def test_crossing_order_fully_matches_at_makers_price():
    lob = LimitOrderBook()
    lob.add_limit_order(Side.ASK, 100.50, 50, 0.0)

    taker, trades = lob.add_limit_order(Side.BID, 101.00, 50, 1.0)

    assert taker.remaining == 0
    assert len(trades) == 1
    trade = trades[0]
    assert trade.price == 100.50  # trades execute at the resting (maker's) price
    assert trade.quantity == 50
    assert lob.get_best_bid_ask() == (None, None)


def test_partial_fill_leaves_correct_remainder_resting():
    lob = LimitOrderBook()
    lob.add_limit_order(Side.ASK, 100.00, 30, 0.0)

    taker, trades = lob.add_limit_order(Side.BID, 100.00, 100, 1.0)

    assert len(trades) == 1
    assert trades[0].quantity == 30
    assert taker.remaining == 70
    bid, ask = lob.get_best_bid_ask()
    assert bid == 100.00  # the unfilled 70 now rests as the best bid
    assert ask is None


def test_price_time_priority_same_price_fills_earlier_order_first():
    lob = LimitOrderBook()
    order1, _ = lob.add_limit_order(Side.BID, 100.00, 50, 0.0)
    order2, _ = lob.add_limit_order(Side.BID, 100.00, 50, 1.0)

    _, trades = lob.add_limit_order(Side.ASK, 100.00, 50, 2.0)

    assert len(trades) == 1
    assert trades[0].maker_order_id == order1.order_id  # earlier timestamp fills first
    assert lob.orders[order2.order_id].remaining == 50  # later order untouched


def test_cancel_removes_liquidity_from_best_bid_ask():
    lob = LimitOrderBook()
    order, _ = lob.add_limit_order(Side.BID, 100.00, 50, 0.0)
    lob.add_limit_order(Side.BID, 99.50, 50, 1.0)

    assert lob.get_best_bid_ask()[0] == 100.00

    lob.cancel_order(order.order_id)

    # the cancelled order must not be reported as the best bid, nor matched against
    assert lob.get_best_bid_ask()[0] == 99.50
    _, trades = lob.add_limit_order(Side.ASK, 100.00, 50, 2.0)
    assert trades == []  # nothing crosses 100.00 anymore; the cancelled bid is gone


def test_cancel_unknown_order_id_returns_false():
    lob = LimitOrderBook()
    assert lob.cancel_order(9999) is False


def test_market_order_takes_available_liquidity_and_drops_remainder():
    lob = LimitOrderBook()
    lob.add_limit_order(Side.ASK, 100.00, 20, 0.0)

    filled_qty, trades = lob.add_market_order(Side.BID, 50, 1.0)

    assert filled_qty == 20  # only 20 were available; the other 30 are dropped, not rested
    assert len(trades) == 1
    assert lob.get_best_bid_ask() == (None, None)
    assert len(lob.orders) == 0


def test_depth_tracks_resting_quantity_and_updates_on_fill():
    lob = LimitOrderBook()
    lob.add_limit_order(Side.BID, 100.00, 40, 0.0)
    lob.add_limit_order(Side.BID, 100.00, 10, 1.0)

    assert lob.get_depth(Side.BID)[0] == (100.00, 50)

    lob.add_limit_order(Side.ASK, 100.00, 20, 2.0)

    assert lob.get_depth(Side.BID)[0] == (100.00, 30)


def test_mid_price_is_none_when_one_side_empty():
    lob = LimitOrderBook()
    lob.add_limit_order(Side.BID, 100.00, 10, 0.0)
    assert lob.get_mid_price() is None
