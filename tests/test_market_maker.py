"""Tests for the market making strategies: the corrected Avellaneda-Stoikov
formula (the original notebook's version diverged for low volatility and ignored
the time-horizon term entirely) and the naive fixed-spread baseline.
"""

import math

import pytest

from hft_mm.market_maker import AvellanedaStoikovMarketMaker, NaiveMarketMaker
from hft_mm.metrics import TICKS_PER_YEAR


def _annualized(sigma_per_tick: float) -> float:
    return sigma_per_tick * math.sqrt(TICKS_PER_YEAR)


def test_reservation_price_skews_down_when_long():
    mm = AvellanedaStoikovMarketMaker(risk_aversion=0.5, fallback_volatility=0.001)
    flat = mm.quote(100.0, {}, inventory=0, t=0, T=1000)
    long = mm.quote(100.0, {}, inventory=200, t=0, T=1000)

    flat_mid = (flat.bid_price + flat.ask_price) / 2
    long_mid = (long.bid_price + long.ask_price) / 2

    assert long_mid < flat_mid  # long inventory skews quotes down to encourage selling


def test_reservation_price_skews_up_when_short():
    mm = AvellanedaStoikovMarketMaker(risk_aversion=0.5, fallback_volatility=0.001)
    flat = mm.quote(100.0, {}, inventory=0, t=0, T=1000)
    short = mm.quote(100.0, {}, inventory=-200, t=0, T=1000)

    flat_mid = (flat.bid_price + flat.ask_price) / 2
    short_mid = (short.bid_price + short.ask_price) / 2

    assert short_mid > flat_mid  # short inventory skews quotes up to encourage covering


def test_spread_widens_with_higher_volatility():
    mm = AvellanedaStoikovMarketMaker(risk_aversion=0.5)
    low = mm.quote(100.0, {"realized_vol_100": _annualized(0.0001)}, inventory=0, t=0, T=1000)
    high = mm.quote(100.0, {"realized_vol_100": _annualized(0.01)}, inventory=0, t=0, T=1000)

    low_spread = low.ask_price - low.bid_price
    high_spread = high.ask_price - high.bid_price
    assert high_spread > low_spread


def test_spread_shrinks_as_session_close_approaches():
    mm = AvellanedaStoikovMarketMaker(risk_aversion=0.5)
    features = {"realized_vol_100": _annualized(0.01)}
    early = mm.quote(100.0, features, inventory=0, t=0, T=1000)
    late = mm.quote(100.0, features, inventory=0, t=999, T=1000)

    assert (late.ask_price - late.bid_price) <= (early.ask_price - early.bid_price)


def test_bid_never_crosses_ask_across_a_parameter_grid():
    for gamma in (0.01, 0.1, 1.0, 5.0):
        for sigma in (0.0001, 0.001, 0.01):
            for inventory in (-400, 0, 400):
                mm = AvellanedaStoikovMarketMaker(
                    risk_aversion=gamma, fallback_volatility=sigma, max_position=500
                )
                q = mm.quote(100.0, {}, inventory=inventory, t=0, T=2000)
                if q.bid_price is not None and q.ask_price is not None:
                    assert q.bid_price < q.ask_price


def test_extreme_skew_disables_a_side_instead_of_quoting_a_negative_price():
    mm = AvellanedaStoikovMarketMaker(risk_aversion=5.0, fallback_volatility=0.05, max_position=1000)
    q = mm.quote(100.0, {}, inventory=999, t=0, T=100_000)
    assert q.bid_price is None or q.bid_price > 0
    assert q.ask_price is None or q.ask_price > 0


def test_quote_stops_bid_side_at_max_long_position():
    mm = AvellanedaStoikovMarketMaker(max_position=500)
    q = mm.quote(100.0, {}, inventory=500, t=0, T=1000)
    assert q.bid_qty == 0
    assert q.bid_price is None
    assert q.ask_qty > 0


def test_quote_stops_ask_side_at_max_short_position():
    mm = AvellanedaStoikovMarketMaker(max_position=500)
    q = mm.quote(100.0, {}, inventory=-500, t=0, T=1000)
    assert q.ask_qty == 0
    assert q.ask_price is None
    assert q.bid_qty > 0


def test_naive_maker_is_always_symmetric_around_mid():
    naive = NaiveMarketMaker(spread=0.10, max_position=500)
    for inventory in (-400, 0, 300):
        q = naive.quote(100.0, {}, inventory=inventory, t=0, T=1000)
        mid = (q.bid_price + q.ask_price) / 2
        assert mid == pytest.approx(100.0, abs=1e-9)
        assert (q.ask_price - q.bid_price) == pytest.approx(0.10, abs=1e-9)


def test_naive_maker_ignores_inventory_when_deciding_price():
    naive = NaiveMarketMaker(spread=0.10, max_position=500)
    flat = naive.quote(100.0, {}, inventory=0, t=0, T=1000)
    skewed = naive.quote(100.0, {}, inventory=300, t=0, T=1000)
    assert flat.bid_price == skewed.bid_price
    assert flat.ask_price == skewed.ask_price
