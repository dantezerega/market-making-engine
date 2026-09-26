from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

from lob_sim.orderbook import OrderBook


# snapshot
@dataclass
class BookSnapshot:
    tick: int
    timestamp: float
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid_price: Optional[float]
    spread: Optional[float]
    relative_spread: Optional[float]
    imbalance: Optional[float]
    weighted_mid: Optional[float]
    bid_levels: List[Tuple[float, float]]
    ask_levels: List[Tuple[float, float]]


def compute_snapshot(book: OrderBook, tick: int, timestamp: float, depth: int = 5) -> BookSnapshot:
    """Compute all microstructure metrics for a given order book state.

    Imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty), range [-1, 1].
    Weighted mid: (ask * bid_vol + bid * ask_vol) / (bid_vol + ask_vol).
    Ref: Gould et al. (2013), "Limit Order Books", Quantitative Finance.
    """
    # fetch
    bid = book.best_bid
    ask = book.best_ask
    bid_levels = book.get_bid_levels(depth)
    ask_levels = book.get_ask_levels(depth)

    # mid
    mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
    # spread
    spread = (ask - bid) if bid is not None and ask is not None else None
    rel_spread = spread / mid if mid is not None and mid > 0 and spread is not None else None

    # imbalance
    bid_vol = sum(q for _, q in bid_levels)
    ask_vol = sum(q for _, q in ask_levels)
    total_vol = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else None

    # weighted
    weighted_mid: Optional[float]
    if bid_levels and ask_levels and bid is not None and ask is not None:
        best_bid_vol = bid_levels[0][1]
        best_ask_vol = ask_levels[0][1]
        denom = best_bid_vol + best_ask_vol
        weighted_mid = (ask * best_bid_vol + bid * best_ask_vol) / denom if denom > 0 else mid
    else:
        weighted_mid = mid

    return BookSnapshot(
        tick=tick,
        timestamp=timestamp,
        best_bid=bid,
        best_ask=ask,
        mid_price=mid,
        spread=spread,
        relative_spread=rel_spread,
        imbalance=imbalance,
        weighted_mid=weighted_mid,
        bid_levels=bid_levels,
        ask_levels=ask_levels,
    )
