from __future__ import annotations
from collections import deque
from typing import Dict, Deque, List, Optional, Tuple
import itertools

from lob_sim.orders import Order, OrderSide, OrderStatus, OrderType, Trade


class OrderBook:
    def __init__(self) -> None:
        # bids
        self._bids: Dict[float, Deque[Order]] = {}
        # asks
        self._asks: Dict[float, Deque[Order]] = {}
        # registry
        self._orders: Dict[int, Order] = {}
        self._trades: List[Trade] = []
        self._order_id_gen = itertools.count(1)
        self._trade_id_gen = itertools.count(1)

    @property
    def best_bid(self) -> Optional[float]:
        # top
        return max(self._bids) if self._bids else None

    @property
    def best_ask(self) -> Optional[float]:
        # top
        return min(self._asks) if self._asks else None

    @property
    def trades(self) -> List[Trade]:
        # snapshot
        return list(self._trades)

    def get_bid_levels(self, depth: int = 10) -> List[Tuple[float, float]]:
        # descending
        sorted_prices = sorted(self._bids.keys(), reverse=True)[:depth]
        return [(p, sum(o.remaining_quantity for o in self._bids[p])) for p in sorted_prices]

    def get_ask_levels(self, depth: int = 10) -> List[Tuple[float, float]]:
        # ascending
        sorted_prices = sorted(self._asks.keys())[:depth]
        return [(p, sum(o.remaining_quantity for o in self._asks[p])) for p in sorted_prices]

    def prune_stale_levels(self, min_bid_price: float, max_ask_price: float) -> None:
        # bids
        for price in list(self._bids.keys()):
            if price < min_bid_price:
                for order in list(self._bids.get(price, [])):
                    self.cancel_order(order.order_id)
        # asks
        for price in list(self._asks.keys()):
            if price > max_ask_price:
                for order in list(self._asks.get(price, [])):
                    self.cancel_order(order.order_id)

    def add_limit_order(
        self, side: OrderSide, price: float, quantity: float, timestamp: float
    ) -> Tuple[Order, List[Trade]]:
        order_id = next(self._order_id_gen)
        # construct
        order = Order(
            order_id=order_id,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            timestamp=timestamp,
            price=price,
        )
        # register
        self._orders[order_id] = order
        trades = self._match_limit(order)
        # rest
        if order.remaining_quantity > 0 and order.status != OrderStatus.FILLED:
            book = self._bids if side == OrderSide.BID else self._asks
            if price not in book:
                book[price] = deque()
            book[price].append(order)
        return order, trades

    def add_market_order(
        self, side: OrderSide, quantity: float, timestamp: float
    ) -> Tuple[Order, List[Trade]]:
        order_id = next(self._order_id_gen)
        # construct
        order = Order(
            order_id=order_id,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            timestamp=timestamp,
        )
        # register
        self._orders[order_id] = order
        trades = self._match_market(order)
        return order, trades

    def cancel_order(self, order_id: int) -> bool:
        order = self._orders.get(order_id)
        # guard
        if order is None or order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return False
        # mark
        order.status = OrderStatus.CANCELLED
        book = self._bids if order.side == OrderSide.BID else self._asks
        price = order.price
        if price is not None and price in book:
            # remove
            try:
                book[price].remove(order)
            except ValueError:
                pass
            # cleanup
            if not book[price]:
                del book[price]
        return True

    def _match_limit(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        # crossing
        if order.side == OrderSide.BID:
            contra_book = self._asks
            get_sorted = lambda: sorted(contra_book.keys())
            price_ok = lambda p: p <= order.price  # type: ignore[operator]
        else:
            contra_book = self._bids
            get_sorted = lambda: sorted(contra_book.keys(), reverse=True)
            price_ok = lambda p: p >= order.price  # type: ignore[operator]

        # walk
        while order.remaining_quantity > 0:
            available = [p for p in get_sorted() if price_ok(p)]
            if not available:
                break
            trades.extend(self._fill_against_level(order, contra_book, available[0]))

        return trades

    def _match_market(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        # sweep
        if order.side == OrderSide.BID:
            contra_book = self._asks
            get_sorted = lambda: sorted(contra_book.keys())
        else:
            contra_book = self._bids
            get_sorted = lambda: sorted(contra_book.keys(), reverse=True)

        # walk
        while order.remaining_quantity > 0 and contra_book:
            prices = get_sorted()
            if not prices:
                break
            trades.extend(self._fill_against_level(order, contra_book, prices[0]))

        return trades

    def _fill_against_level(
        self,
        aggressor: Order,
        contra_book: Dict[float, Deque[Order]],
        price: float,
    ) -> List[Trade]:
        trades: List[Trade] = []
        level = contra_book.get(price)
        if not level:
            return trades

        # FIFO
        while level and aggressor.remaining_quantity > 0:
            passive = level[0]
            fill_qty = min(aggressor.remaining_quantity, passive.remaining_quantity)
            trade_id = next(self._trade_id_gen)

            # record
            if aggressor.side == OrderSide.BID:
                trade = Trade(
                    trade_id=trade_id,
                    buy_order_id=aggressor.order_id,
                    sell_order_id=passive.order_id,
                    price=price,
                    quantity=fill_qty,
                    timestamp=aggressor.timestamp,
                )
            else:
                trade = Trade(
                    trade_id=trade_id,
                    buy_order_id=passive.order_id,
                    sell_order_id=aggressor.order_id,
                    price=price,
                    quantity=fill_qty,
                    timestamp=aggressor.timestamp,
                )

            # update
            aggressor.filled_quantity += fill_qty
            passive.filled_quantity += fill_qty

            aggressor.status = (
                OrderStatus.FILLED if aggressor.remaining_quantity == 0 else OrderStatus.PARTIAL
            )
            passive.status = (
                OrderStatus.FILLED if passive.remaining_quantity == 0 else OrderStatus.PARTIAL
            )

            # evict
            if passive.status == OrderStatus.FILLED:
                level.popleft()

            trades.append(trade)
            self._trades.append(trade)

        # purge
        if not level:
            del contra_book[price]

        return trades
