"""Avellaneda-Stoikov / naive market-making engine, driving the vendored
LOB simulator's own OrderBook and background order-flow process
(unchanged matching/arrival logic — see lob_sim/). This module owns the
tick loop, the MM agent's own quote-cancel-requote logic, fill
bookkeeping, and the P&L / inventory / markout metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np

from lob_sim.orderbook import OrderBook
from lob_sim.orders import Order, OrderSide, OrderStatus, Trade
from lob_sim.simulation import ou_step, sample_order_size

from as_model.model import optimal_quotes

StrategyName = Literal["avellaneda_stoikov", "naive"]


@dataclass
class EngineParams:
    # session / background flow (mirrors lob_sim.simulation.SimulationParams
    # so the background order-generation process is identical across runs)
    n_ticks: int = 2000
    dt: float = 1.0
    ou_mu: float = 100.0
    ou_theta: float = 0.05
    ou_sigma: float = 0.05
    lambda_market: float = 1.5
    lambda_limit: float = 4.0
    order_size_mu: float = 1.5
    order_size_sigma: float = 0.5
    tick_size: float = 0.01
    initial_spread: float = 0.10
    n_initial_levels: int = 10
    seed: Optional[int] = 42

    # market maker
    strategy: StrategyName = "avellaneda_stoikov"
    gamma: float = 0.1          # risk aversion
    sigma: float = 0.05         # calibrated separately, overridable here
    k: float = 1.5              # order arrival intensity decay
    A: float = 140.0            # arrival rate scale (diagnostic only)
    quote_size: float = 10.0    # size posted on each side, per requote
    naive_half_spread: float = 0.05  # baseline: fixed, no inventory skew
    max_abs_inventory: Optional[float] = None  # optional hard inventory cap


@dataclass
class Fill:
    tick: int
    timestamp: float
    side: str            # "buy" or "sell" from the MM's perspective
    price: float
    quantity: float
    mid_at_fill: float


@dataclass
class EngineResult:
    ticks: np.ndarray
    mid_price: np.ndarray
    reservation_price: np.ndarray
    bid_quote: np.ndarray
    ask_quote: np.ndarray
    inventory: np.ndarray
    cash: np.ndarray
    mtm_pnl: np.ndarray
    spread_pnl_cum: np.ndarray
    inventory_pnl_cum: np.ndarray
    fills: List[Fill]
    params: EngineParams


def _cancel_if_open(book: OrderBook, order: Optional[Order]) -> None:
    if order is not None and order.status not in (OrderStatus.FILLED, OrderStatus.CANCELLED):
        book.cancel_order(order.order_id)


def run_mm_session(params: EngineParams) -> EngineResult:
    rng = np.random.default_rng(params.seed)
    book = OrderBook()

    price = params.ou_mu
    # seed the book the same way the base simulator does, so background
    # flow has resting liquidity to trade against from tick 0.
    half_spread0 = params.initial_spread / 2.0
    for i in range(params.n_initial_levels):
        bid_price = round(price - half_spread0 - i * params.tick_size, 6)
        ask_price = round(price + half_spread0 + i * params.tick_size, 6)
        bid_qty = float(rng.integers(50, 200))
        ask_qty = float(rng.integers(50, 200))
        if bid_price > 0:
            book.add_limit_order(OrderSide.BID, bid_price, bid_qty, 0.0)
        if ask_price > 0:
            book.add_limit_order(OrderSide.ASK, ask_price, ask_qty, 0.0)

    T = params.n_ticks * params.dt

    ticks = [0]
    mid_hist = [price]
    resv_hist = [price]
    bid_q_hist = [price]
    ask_q_hist = [price]
    inv_hist = [0.0]
    cash_hist = [0.0]

    inventory = 0.0
    cash = 0.0
    my_bid_order: Optional[Order] = None
    my_ask_order: Optional[Order] = None
    fills: List[Fill] = []
    seen_trade_ids: set[int] = set()

    timestamp = 0.0

    for tick in range(1, params.n_ticks + 1):
        timestamp += params.dt
        time_left = max(T - timestamp, params.dt)  # avoid t==T singularity

        # --- 1. diffuse the fundamental / background flow (identical to
        # lob_sim.simulation.run_simulation's own loop; unchanged logic) ---
        epsilon = rng.standard_normal()
        price = ou_step(price, params.ou_mu, params.ou_theta, params.ou_sigma, params.dt, epsilon)

        n_market = int(rng.poisson(params.lambda_market * params.dt))
        n_limit = int(rng.poisson(params.lambda_limit * params.dt))

        # --- 2. cancel MM's stale quotes, compute new ones, repost ---
        _cancel_if_open(book, my_bid_order)
        _cancel_if_open(book, my_ask_order)

        if params.strategy == "avellaneda_stoikov":
            bid_px, ask_px, resv, _spread = optimal_quotes(
                price, inventory, params.gamma, params.sigma, time_left, params.k
            )
        elif params.strategy == "naive":
            resv = price
            bid_px = price - params.naive_half_spread
            ask_px = price + params.naive_half_spread
        else:
            raise ValueError(f"Unknown strategy {params.strategy}")

        bid_px = round(max(bid_px, params.tick_size), 4)
        ask_px = round(max(ask_px, bid_px + params.tick_size), 4)

        skip_bid = params.max_abs_inventory is not None and inventory >= params.max_abs_inventory
        skip_ask = params.max_abs_inventory is not None and inventory <= -params.max_abs_inventory

        my_bid_order, bid_trades = (None, [])
        my_ask_order, ask_trades = (None, [])
        if not skip_bid:
            my_bid_order, bid_trades = book.add_limit_order(OrderSide.BID, bid_px, params.quote_size, timestamp)
        if not skip_ask:
            my_ask_order, ask_trades = book.add_limit_order(OrderSide.ASK, ask_px, params.quote_size, timestamp)

        # --- 3. background market/limit order flow trades against the book,
        # potentially filling our resting quotes from step above too, i.e.
        # arrivals here can also cross OUR own posted orders (that's the
        # "posts limit orders that interact with simulated flow" requirement) --
        for _ in range(n_market):
            side = OrderSide.BID if rng.random() < 0.5 else OrderSide.ASK
            qty = sample_order_size(params.order_size_mu, params.order_size_sigma, rng)
            book.add_market_order(side, qty, timestamp)

        for _ in range(n_limit):
            side = OrderSide.BID if rng.random() < 0.5 else OrderSide.ASK
            qty = sample_order_size(params.order_size_mu, params.order_size_sigma, rng)
            offset = float(rng.exponential(0.05))
            limit_price = round(price - offset, 4) if side == OrderSide.BID else round(price + offset, 4)
            if limit_price >= params.tick_size:
                book.add_limit_order(side, limit_price, qty, timestamp)

        # --- 4. prune stale resting levels (same policy as base sim) ---
        stale_range = 5.0 * params.ou_sigma * np.sqrt(params.n_ticks * params.dt)
        book.prune_stale_levels(price - stale_range, price + stale_range)

        # --- 5. attribute fills to the MM by scanning all NEW trades this
        # tick for our two order ids (fills can happen either at our own
        # add_limit_order call above, if we crossed the book, or later in
        # the tick when incoming background flow lifts/hits our resting
        # quotes) ---
        my_ids = set()
        if my_bid_order is not None:
            my_ids.add(my_bid_order.order_id)
        if my_ask_order is not None:
            my_ids.add(my_ask_order.order_id)

        all_new_trades = list(bid_trades) + list(ask_trades)
        for t in book.trades[-(n_market + n_limit + 4):]:
            if t.trade_id not in seen_trade_ids:
                all_new_trades.append(t)

        for trade in all_new_trades:
            if trade.trade_id in seen_trade_ids:
                continue
            seen_trade_ids.add(trade.trade_id)
            if trade.buy_order_id in my_ids:
                # we bought: cash down, inventory up
                cash -= trade.price * trade.quantity
                inventory += trade.quantity
                fills.append(Fill(tick, timestamp, "buy", trade.price, trade.quantity, price))
            elif trade.sell_order_id in my_ids:
                cash += trade.price * trade.quantity
                inventory -= trade.quantity
                fills.append(Fill(tick, timestamp, "sell", trade.price, trade.quantity, price))

        ticks.append(tick)
        mid_hist.append(price)
        resv_hist.append(resv)
        bid_q_hist.append(bid_px)
        ask_q_hist.append(ask_px)
        inv_hist.append(inventory)
        cash_hist.append(cash)

    mid_arr = np.array(mid_hist)
    inv_arr = np.array(inv_hist)
    cash_arr = np.array(cash_hist)
    mtm_pnl = cash_arr + inv_arr * mid_arr

    # --- spread vs inventory P&L decomposition ---
    # spread capture per fill: edge realized vs the mid prevailing AT the
    # moment of the fill (captured half-spread x fill size).
    spread_pnl_cum = np.zeros(len(ticks))
    spread_running = 0.0
    fill_by_tick: Dict[int, float] = {}
    for f in fills:
        edge = (f.price - f.mid_at_fill) * f.quantity if f.side == "sell" else (f.mid_at_fill - f.price) * f.quantity
        fill_by_tick[f.tick] = fill_by_tick.get(f.tick, 0.0) + edge
    for i, tk in enumerate(ticks):
        spread_running += fill_by_tick.get(tk, 0.0)
        spread_pnl_cum[i] = spread_running
    inventory_pnl_cum = mtm_pnl - spread_pnl_cum

    return EngineResult(
        ticks=np.array(ticks),
        mid_price=mid_arr,
        reservation_price=np.array(resv_hist),
        bid_quote=np.array(bid_q_hist),
        ask_quote=np.array(ask_q_hist),
        inventory=inv_arr,
        cash=cash_arr,
        mtm_pnl=mtm_pnl,
        spread_pnl_cum=spread_pnl_cum,
        inventory_pnl_cum=inventory_pnl_cum,
        fills=fills,
        params=params,
    )
