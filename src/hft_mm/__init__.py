"""High-frequency market making simulation: a real matching-engine limit order book,
an Avellaneda-Stoikov market maker, and a backtester to evaluate it against a naive baseline.
"""

from hft_mm.order_book import LimitOrderBook, Order, Side, Trade
from hft_mm.features import MarketDataProcessor
from hft_mm.market_maker import AvellanedaStoikovMarketMaker, NaiveMarketMaker
from hft_mm.simulator import MarketEnvironment
from hft_mm.backtester import Backtester
from hft_mm import metrics

__all__ = [
    "LimitOrderBook",
    "Order",
    "Side",
    "Trade",
    "MarketDataProcessor",
    "AvellanedaStoikovMarketMaker",
    "NaiveMarketMaker",
    "MarketEnvironment",
    "Backtester",
    "metrics",
]
