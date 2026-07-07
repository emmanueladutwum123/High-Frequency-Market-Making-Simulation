"""Tests for the Backtester's P&L accounting and run determinism.

The original notebook guessed fill direction from `not filled_quote.is_bid`,
inverting P&L. Here every fill comes from a real matching-engine Trade, so the
strongest test is an independent replay of the recorded `fills` log against the
Backtester's own running cash/inventory state.
"""

import numpy as np
import pytest

from hft_mm.backtester import Backtester
from hft_mm.market_maker import AvellanedaStoikovMarketMaker


def test_pnl_accounting_matches_independent_fill_replay():
    mm = AvellanedaStoikovMarketMaker()
    bt = Backtester(mm, n_ticks=3000, seed=1)
    result = bt.run()

    replay_cash = 0.0
    replay_inventory = 0
    for fill in result["fills"]:
        if fill["side"] == "buy":
            replay_cash -= fill["price"] * fill["quantity"] * (1 + bt.fee_rate)
            replay_inventory += fill["quantity"]
        else:
            replay_cash += fill["price"] * fill["quantity"] * (1 - bt.fee_rate)
            replay_inventory -= fill["quantity"]

    assert replay_cash == pytest.approx(bt.cash)
    assert replay_inventory == bt.inventory


def test_final_pnl_equals_cash_plus_inventory_times_final_mid():
    mm = AvellanedaStoikovMarketMaker()
    bt = Backtester(mm, n_ticks=2000, seed=5)
    result = bt.run()

    expected_final_pnl = bt.cash + bt.inventory * result["mid_series"][-1]
    assert result["pnl_series"][-1] == pytest.approx(expected_final_pnl)


def test_run_produces_finite_results_of_expected_length():
    mm = AvellanedaStoikovMarketMaker()
    bt = Backtester(mm, n_ticks=2000, seed=5)
    result = bt.run()

    assert len(result["pnl_series"]) == 2000
    assert len(result["inventory_series"]) == 2000
    assert np.isfinite(result["pnl_series"]).all()
    assert np.isfinite(result["mid_series"]).all()


def test_run_is_deterministic_given_the_same_seed():
    bt1 = Backtester(AvellanedaStoikovMarketMaker(), n_ticks=1000, seed=123)
    r1 = bt1.run()

    bt2 = Backtester(AvellanedaStoikovMarketMaker(), n_ticks=1000, seed=123)
    r2 = bt2.run()

    np.testing.assert_array_equal(r1["pnl_series"], r2["pnl_series"])
    np.testing.assert_array_equal(r1["inventory_series"], r2["inventory_series"])


def test_fills_are_well_formed():
    mm = AvellanedaStoikovMarketMaker()
    bt = Backtester(mm, n_ticks=10_000, seed=1)
    result = bt.run()

    assert len(result["fills"]) > 0  # this parameterization should actually trade
    for fill in result["fills"]:
        assert fill["price"] > 0
        assert fill["quantity"] > 0
        assert fill["side"] in ("buy", "sell")
