"""Tests for MarketDataProcessor — specifically the two features that were literal
placeholders in the original notebook (`bid_ask_imbalance` returned random noise,
`microprice` just averaged price history) and are now computed from real book state.
"""

import numpy as np
import pytest

from hft_mm.features import MarketDataProcessor
from hft_mm.metrics import TICKS_PER_YEAR
from hft_mm.order_book import LimitOrderBook, Side


def test_microprice_skews_toward_the_thinner_side():
    lob = LimitOrderBook()
    lob.add_limit_order(Side.BID, 99.0, 100, 0.0)  # thick bid
    lob.add_limit_order(Side.ASK, 101.0, 50, 0.0)  # thin ask

    microprice = MarketDataProcessor.calculate_microprice(lob)

    expected = (99.0 * 50 + 101.0 * 100) / 150
    assert microprice == pytest.approx(expected)
    assert microprice > 100.0  # skews above the plain mid, toward the thinner (ask) side


def test_microprice_is_none_when_one_side_empty():
    lob = LimitOrderBook()
    lob.add_limit_order(Side.BID, 99.0, 100, 0.0)
    assert MarketDataProcessor.calculate_microprice(lob) is None


def test_imbalance_is_positive_when_bid_depth_exceeds_ask_depth():
    lob = LimitOrderBook()
    lob.add_limit_order(Side.BID, 99.0, 100, 0.0)
    lob.add_limit_order(Side.ASK, 101.0, 20, 0.0)

    imbalance = MarketDataProcessor.calculate_imbalance(lob)

    assert imbalance == pytest.approx((100 - 20) / 120)
    assert -1.0 <= imbalance <= 1.0


def test_imbalance_is_none_when_book_is_empty():
    lob = LimitOrderBook()
    assert MarketDataProcessor.calculate_imbalance(lob) is None


def test_realized_vol_matches_manual_log_return_std():
    proc = MarketDataProcessor(window_sizes=[10])
    prices = [100.0, 101.0] * 5
    for p in prices:
        proc.price_history.append(p)
        proc.volume_history.append(0)
        proc.spread_history.append(0.1)

    lob = LimitOrderBook()
    lob.add_limit_order(Side.BID, 99.0, 10, 0.0)
    lob.add_limit_order(Side.ASK, 101.0, 10, 0.0)

    features = proc.calculate_features(lob)

    expected_returns = np.diff(np.log(np.array(prices)))
    expected_vol = np.std(expected_returns) * np.sqrt(TICKS_PER_YEAR)
    assert features["realized_vol_10"] == pytest.approx(expected_vol)


def test_smaller_window_available_before_larger_window_is_full():
    proc = MarketDataProcessor(window_sizes=[5, 10])
    for p in [100, 101, 99, 102, 98]:
        proc.price_history.append(p)
        proc.volume_history.append(0)
        proc.spread_history.append(0.1)

    lob = LimitOrderBook()
    lob.add_limit_order(Side.BID, 97.0, 10, 0.0)
    lob.add_limit_order(Side.ASK, 103.0, 10, 0.0)

    features = proc.calculate_features(lob)

    assert "realized_vol_5" in features
    assert "realized_vol_10" not in features


def test_calculate_features_empty_before_smallest_window_ready():
    proc = MarketDataProcessor(window_sizes=[20, 100])
    lob = LimitOrderBook()
    assert proc.calculate_features(lob) == {}
