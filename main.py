"""End-to-end runner:
  1. Runs a background-only (no MM) session on the vendored LOB sim to
     estimate sigma and calibrate k/A from its own price/trade process.
  2. Runs the Avellaneda-Stoikov MM session and the naive baseline MM
     session, both against fresh instances of the same simulator/flow.
  3. Computes markout, inventory-risk, and P&L-decomposition metrics
     for both.
  4. Runs a gamma sweep.
  5. Writes all plots to ./output/ and prints a text summary report.

Run with:  python main.py
"""
from __future__ import annotations

import os

import numpy as np

from lob_sim.simulation import SimulationParams, run_simulation
from lob_sim.orders import OrderSide

from as_model.engine import EngineParams, run_mm_session
from as_model.calibration import estimate_sigma, calibrate_k_A
from as_model.analytics import compute_markouts, compute_inventory_risk
from as_model.sweep import run_gamma_sweep
from as_model.plots import (
    plot_quotes_and_mid,
    plot_inventory,
    plot_pnl_decomposition,
    plot_naive_vs_as_comparison,
    plot_gamma_sweep,
    plot_markout_curves,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def calibrate_from_background_run_direct(n_ticks: int = 3000, seed: int = 7):
    """Direct-access version: re-implements the background-only loop using
    lob_sim internals so we retain the OrderBook's full trade tape (the
    public run_simulation() function only returns snapshots+price path)."""
    from lob_sim.orderbook import OrderBook
    from lob_sim.simulation import ou_step, sample_order_size, _initialize_book
    from lob_sim.metrics import compute_snapshot

    params = SimulationParams(n_ticks=n_ticks, seed=seed)
    rng = np.random.default_rng(params.seed)
    book = OrderBook()
    price = params.ou_mu
    _initialize_book(book, price, params.initial_spread, params.n_initial_levels, params.tick_size, rng)

    price_path = [price]
    trade_records = []  # (timestamp, price)
    timestamp = 0.0

    for tick in range(1, params.n_ticks + 1):
        timestamp += params.dt
        epsilon = rng.standard_normal()
        price = ou_step(price, params.ou_mu, params.ou_theta, params.ou_sigma, params.dt, epsilon)

        n_market = int(rng.poisson(params.lambda_market * params.dt))
        n_limit = int(rng.poisson(params.lambda_limit * params.dt))

        n_trades_before = len(book.trades)
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

        stale_range = 5.0 * params.ou_sigma * np.sqrt(params.n_ticks * params.dt)
        book.prune_stale_levels(price - stale_range, price + stale_range)

        for trade in book.trades[n_trades_before:]:
            trade_records.append((timestamp, trade.price))

        price_path.append(price)

    sigma = estimate_sigma(price_path, params.dt)

    # distance of each trade from the fundamental price AT that timestamp:
    # approximate fundamental via linear interpolation of the recorded
    # price_path (price_path[i] corresponds to timestamp i*dt).
    distances = []
    for ts, trade_price in trade_records:
        idx = min(int(round(ts / params.dt)), len(price_path) - 1)
        fundamental = price_path[idx]
        distances.append(abs(trade_price - fundamental))

    trade_window_seconds = params.n_ticks * params.dt
    if len(distances) < 10:
        return sigma, 1.5, 140.0, 0.0, len(distances)

    k, A, r2, _, _ = calibrate_k_A(distances, trade_window_seconds)
    return sigma, k, A, r2, len(distances)


def print_header(text: str) -> None:
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print_header("STEP 1 — Calibrating sigma, k, A from the LOB simulator's own flow")
    sigma, k, A, r2, n_trades = calibrate_from_background_run_direct(n_ticks=3000, seed=7)
    print(f"  Estimated sigma (per-tick vol of mid price) : {sigma:.6f}")
    print(f"  Calibrated k (intensity decay)               : {k:.4f}")
    print(f"  Calibrated A (arrival rate scale)             : {A:.4f}")
    print(f"  log-linear fit R^2                            : {r2:.4f}  (n_trades={n_trades})")

    n_ticks = 2000
    base = EngineParams(
        n_ticks=n_ticks,
        gamma=0.005,
        sigma=sigma,
        k=k,
        A=A,
        quote_size=10.0,
        naive_half_spread=0.099,  # matched to AS's session-average half-spread
        seed=123,                  # so the comparison isolates the inventory-skew
    )                              # effect rather than differing spread width

    print_header("STEP 2 — Running Avellaneda-Stoikov MM session")
    as_params = EngineParams(**{**base.__dict__, "strategy": "avellaneda_stoikov"})
    as_result = run_mm_session(as_params)
    print(f"  Fills: {len(as_result.fills)}  Final inventory: {as_result.inventory[-1]:.1f}  Final P&L: {as_result.mtm_pnl[-1]:.4f}")

    print_header("STEP 3 — Running naive symmetric baseline MM session")
    naive_params = EngineParams(**{**base.__dict__, "strategy": "naive"})
    naive_result = run_mm_session(naive_params)
    print(f"  Fills: {len(naive_result.fills)}  Final inventory: {naive_result.inventory[-1]:.1f}  Final P&L: {naive_result.mtm_pnl[-1]:.4f}")

    print_header("STEP 4 — Adverse selection: markout metrics")
    as_markouts = compute_markouts(as_result, horizons=(1, 10, 100))
    naive_markouts = compute_markouts(naive_result, horizons=(1, 10, 100))
    for h in (1, 10, 100):
        a = as_markouts[h]
        n = naive_markouts[h]
        print(f"  Horizon {h:>4} ticks | AS mean markout: {a.mean_bps:+7.3f} bps (n={a.n_fills:4d}) | "
              f"Naive mean markout: {n.mean_bps:+7.3f} bps (n={n.n_fills:4d})")

    print_header("STEP 5 — Inventory risk & P&L decomposition")
    as_risk = compute_inventory_risk(as_result)
    naive_risk = compute_inventory_risk(naive_result)
    print(f"  {'Metric':<28}{'Avellaneda-Stoikov':>22}{'Naive':>18}")
    print(f"  {'Inventory variance':<28}{as_risk.inventory_variance:>22.4f}{naive_risk.inventory_variance:>18.4f}")
    print(f"  {'Max |inventory|':<28}{as_risk.max_abs_inventory:>22.1f}{naive_risk.max_abs_inventory:>18.1f}")
    print(f"  {'Total P&L':<28}{as_risk.pnl_total:>22.4f}{naive_risk.pnl_total:>18.4f}")
    print(f"  {'P&L volatility (per tick)':<28}{as_risk.pnl_vol:>22.6f}{naive_risk.pnl_vol:>18.6f}")
    print(f"  {'Spread P&L (total)':<28}{as_risk.spread_pnl_total:>22.4f}{naive_risk.spread_pnl_total:>18.4f}")
    print(f"  {'Inventory P&L (total)':<28}{as_risk.inventory_pnl_total:>22.4f}{naive_risk.inventory_pnl_total:>18.4f}")
    print(f"  {'Risk-adjusted P&L proxy':<28}{as_risk.sharpe_like:>22.4f}{naive_risk.sharpe_like:>18.4f}")

    print_header("STEP 6 — Gamma sensitivity sweep")
    gammas = np.geomspace(0.0008, 0.05, 12)
    sweep_points = run_gamma_sweep(base, gammas)
    for p in sweep_points:
        print(f"  gamma={p.gamma:8.4f}  inv_var={p.inventory_variance:10.4f}  pnl={p.pnl_total:10.4f}  "
              f"risk_adj={p.sharpe_like:8.4f}")

    print_header("STEP 7 — Writing plots to ./output/")
    plot_quotes_and_mid(as_result, os.path.join(OUTPUT_DIR, "01_quotes_vs_mid.png"),
                         "Avellaneda-Stoikov: reservation price & quotes vs mid")
    plot_inventory(as_result, os.path.join(OUTPUT_DIR, "02_inventory_path.png"),
                   "Avellaneda-Stoikov: inventory path")
    plot_pnl_decomposition(as_result, os.path.join(OUTPUT_DIR, "03_pnl_decomposition.png"),
                           "Avellaneda-Stoikov: P&L decomposition (spread vs inventory)")
    plot_naive_vs_as_comparison(as_result, naive_result, os.path.join(OUTPUT_DIR, "04_naive_vs_as.png"))
    plot_gamma_sweep(sweep_points, os.path.join(OUTPUT_DIR, "05_gamma_sweep.png"))
    plot_markout_curves(as_markouts, naive_markouts, os.path.join(OUTPUT_DIR, "06_markout_curves.png"))
    print("  Done. Files: 01_quotes_vs_mid.png, 02_inventory_path.png, 03_pnl_decomposition.png,")
    print("               04_naive_vs_as.png, 05_gamma_sweep.png, 06_markout_curves.png")

    print_header("SUMMARY")
    variance_reduction = 1.0 - (as_risk.inventory_variance / naive_risk.inventory_variance) if naive_risk.inventory_variance > 0 else float("nan")
    print(f"  AS reduces inventory variance vs naive by: {variance_reduction*100:+.1f}%")
    print(f"  AS risk-adjusted P&L proxy vs naive       : {as_risk.sharpe_like:.4f} vs {naive_risk.sharpe_like:.4f}")
    print()


if __name__ == "__main__":
    main()
