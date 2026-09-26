"""Matplotlib dashboard: all required charts, saved as PNG files (no
plotly dependency needed for the static report; matplotlib is enough
and keeps this runnable headless in one shot)."""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from as_model.engine import EngineResult
from as_model.analytics import MarkoutSummary
from as_model.sweep import SweepPoint


def plot_quotes_and_mid(result: EngineResult, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(result.ticks, result.mid_price, label="Mid price", color="black", linewidth=1)
    ax.plot(result.ticks, result.reservation_price, label="Reservation price r(s,q,t)", color="tab:purple", linewidth=1, alpha=0.8)
    ax.plot(result.ticks, result.bid_quote, label="Bid quote", color="tab:green", linewidth=0.6, alpha=0.7)
    ax.plot(result.ticks, result.ask_quote, label="Ask quote", color="tab:red", linewidth=0.6, alpha=0.7)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_inventory(result: EngineResult, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result.ticks, result.inventory, color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Inventory (shares)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pnl_decomposition(result: EngineResult, out_path: str, title: str) -> None:
    """Line-based decomposition (NOT matplotlib stackplot): stackplot silently
    misrepresents the total whenever a component goes negative (it stacks by
    naive cumulative sum, so the visual top stops matching the true total
    P&L line once inventory P&L turns negative — exactly the common case
    here). Plotting the three cumulative series directly avoids that trap."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(result.ticks, result.spread_pnl_cum, label="Spread P&L (captured half-spread x fills)", color="tab:green", linewidth=1.4)
    ax.plot(result.ticks, result.inventory_pnl_cum, label="Inventory P&L (mark-to-market swings)", color="tab:orange", linewidth=1.4)
    ax.plot(result.ticks, result.mtm_pnl, color="black", linewidth=1.6, label="Total P&L (= spread + inventory)")
    ax.fill_between(result.ticks, 0, result.spread_pnl_cum, color="tab:green", alpha=0.15)
    ax.fill_between(result.ticks, 0, result.inventory_pnl_cum, color="tab:orange", alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Cumulative P&L")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_naive_vs_as_comparison(
    as_result: EngineResult,
    naive_result: EngineResult,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(as_result.ticks, as_result.inventory, label="Avellaneda-Stoikov", color="tab:blue")
    axes[0].plot(naive_result.ticks, naive_result.inventory, label="Naive symmetric", color="tab:gray", alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("Inventory")
    axes[0].set_title("Inventory path: Avellaneda-Stoikov vs naive baseline")
    axes[0].legend(fontsize=8)

    axes[1].plot(as_result.ticks, as_result.mtm_pnl, label="Avellaneda-Stoikov", color="tab:blue")
    axes[1].plot(naive_result.ticks, naive_result.mtm_pnl, label="Naive symmetric", color="tab:gray", alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_xlabel("Tick")
    axes[1].set_ylabel("Cumulative P&L")
    axes[1].set_title("P&L: Avellaneda-Stoikov vs naive baseline")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_gamma_sweep(points: List[SweepPoint], out_path: str) -> None:
    gammas = [p.gamma for p in points]
    variances = [p.inventory_variance for p in points]
    pnl = [p.pnl_total for p in points]
    sharpe = [p.sharpe_like for p in points]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(gammas, variances, marker="o", color="tab:red")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Gamma (risk aversion, log scale)")
    axes[0].set_ylabel("Inventory variance")
    axes[0].set_title("Risk-aversion vs inventory variance")

    ax2 = axes[1]
    ax2.plot(gammas, pnl, marker="o", color="tab:green", label="Total P&L")
    ax2.set_xscale("log")
    ax2.set_xlabel("Gamma (risk aversion, log scale)")
    ax2.set_ylabel("Total P&L", color="tab:green")
    ax2.set_title("Risk-aversion vs P&L / risk-adjusted P&L")
    ax2b = ax2.twinx()
    ax2b.plot(gammas, sharpe, marker="s", color="tab:purple", label="P&L / P&L-vol (risk-adjusted)")
    ax2b.set_ylabel("Risk-adjusted P&L proxy", color="tab:purple")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_markout_curves(
    as_markouts: Dict[int, MarkoutSummary],
    naive_markouts: Dict[int, MarkoutSummary],
    out_path: str,
) -> None:
    horizons = sorted(as_markouts.keys())
    as_means = [as_markouts[h].mean_bps for h in horizons]
    naive_means = [naive_markouts[h].mean_bps for h in horizons]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(horizons, as_means, marker="o", label="Avellaneda-Stoikov", color="tab:blue")
    ax.plot(horizons, naive_means, marker="s", label="Naive symmetric", color="tab:gray")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Horizon after fill (ticks, log scale)")
    ax.set_ylabel("Mean markout (bps, positive = adverse)")
    ax.set_title("Adverse-selection markout curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
