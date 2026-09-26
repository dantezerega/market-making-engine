from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from lob_sim.orderbook import OrderBook
from lob_sim.orders import OrderSide
from lob_sim.metrics import BookSnapshot, compute_snapshot


# config
@dataclass
class SimulationParams:
    n_ticks: int = 1000
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
    depth: int = 5
    seed: Optional[int] = 42


# output
@dataclass
class SimulationResult:
    snapshots: List[BookSnapshot]
    price_path: List[float]
    ticks: List[int]


def ou_step(price: float, mu: float, theta: float, sigma: float, dt: float, epsilon: float) -> float:
    """Ornstein-Uhlenbeck mean-reverting diffusion, Euler-Maruyama discretization.

    dS = theta * (mu - S) * dt + sigma * sqrt(dt) * epsilon
    theta: mean-reversion speed
    mu: long-run mean
    Uhlenbeck & Ornstein (1930), "On the Theory of the Brownian Motion".
    """
    # revert
    return price + theta * (mu - price) * dt + sigma * np.sqrt(dt) * epsilon


def sample_order_size(mu: float, sigma: float, rng: np.random.Generator) -> float:
    """Log-normal order size distribution: X ~ exp(N(mu, sigma^2))."""
    # lognormal
    return float(np.exp(rng.normal(mu, sigma)))


def _initialize_book(
    book: OrderBook,
    mid_price: float,
    spread: float,
    n_levels: int,
    tick_size: float,
    rng: np.random.Generator,
) -> None:
    # seed
    half_spread = spread / 2.0
    for i in range(n_levels):
        bid_price = round(mid_price - half_spread - i * tick_size, 6)
        ask_price = round(mid_price + half_spread + i * tick_size, 6)
        # random
        bid_qty = float(rng.integers(50, 200))
        ask_qty = float(rng.integers(50, 200))
        if bid_price > 0:
            book.add_limit_order(OrderSide.BID, bid_price, bid_qty, 0.0)
        if ask_price > 0:
            book.add_limit_order(OrderSide.ASK, ask_price, ask_qty, 0.0)


def run_simulation(params: SimulationParams) -> SimulationResult:
    # rng
    rng = np.random.default_rng(params.seed)
    book = OrderBook()
    price = params.ou_mu

    _initialize_book(
        book, price, params.initial_spread, params.n_initial_levels, params.tick_size, rng
    )

    snapshots: List[BookSnapshot] = []
    price_path: List[float] = [price]
    ticks: List[int] = [0]
    timestamp = 0.0

    for tick in range(1, params.n_ticks + 1):
        timestamp += params.dt
        # diffuse
        epsilon = rng.standard_normal()
        price = ou_step(price, params.ou_mu, params.ou_theta, params.ou_sigma, params.dt, epsilon)

        # arrivals
        n_market = int(rng.poisson(params.lambda_market * params.dt))
        n_limit = int(rng.poisson(params.lambda_limit * params.dt))

        # market
        for _ in range(n_market):
            side = OrderSide.BID if rng.random() < 0.5 else OrderSide.ASK
            qty = sample_order_size(params.order_size_mu, params.order_size_sigma, rng)
            book.add_market_order(side, qty, timestamp)

        # limit
        for _ in range(n_limit):
            side = OrderSide.BID if rng.random() < 0.5 else OrderSide.ASK
            qty = sample_order_size(params.order_size_mu, params.order_size_sigma, rng)
            offset = float(rng.exponential(0.05))
            if side == OrderSide.BID:
                limit_price = round(price - offset, 4)
            else:
                limit_price = round(price + offset, 4)
            if limit_price >= params.tick_size:
                book.add_limit_order(side, limit_price, qty, timestamp)

        # prune
        stale_range = 5.0 * params.ou_sigma * np.sqrt(params.n_ticks * params.dt)
        book.prune_stale_levels(price - stale_range, price + stale_range)

        # snapshot
        snapshot = compute_snapshot(book, tick, timestamp, depth=params.depth)
        snapshots.append(snapshot)
        price_path.append(price)
        ticks.append(tick)

    return SimulationResult(snapshots=snapshots, price_path=price_path, ticks=ticks)
