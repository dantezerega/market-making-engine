"""Adverse-selection markout metrics and inventory-risk statistics.

Markout convention: for each fill, we measure how the fundamental price
moved over the N ticks AFTER the fill, expressed so that POSITIVE = bad
for the market maker (the price moved in the direction that makes the
fill look like it was against informed/adverse flow), NEGATIVE = good
(you were paid the spread and the price didn't move against you, or
even helped you).

  buy fill (we bought at f.price): adverse move = price falling afterwards
      markout_bps = (f.price - mid_{t+N}) / f.price * 1e4
  sell fill (we sold at f.price): adverse move = price rising afterwards
      markout_bps = (mid_{t+N} - f.price) / f.price * 1e4
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from as_model.engine import EngineResult, Fill


@dataclass
class MarkoutSummary:
    horizon_ticks: int
    mean_bps: float
    median_bps: float
    std_bps: float
    n_fills: int
    per_fill_bps: np.ndarray


def compute_markouts(result: EngineResult, horizons: Sequence[int] = (1, 10, 100)) -> Dict[int, MarkoutSummary]:
    mid = result.mid_price
    ticks = result.ticks
    tick_to_idx = {int(t): i for i, t in enumerate(ticks)}
    n = len(ticks)

    out: Dict[int, MarkoutSummary] = {}
    for h in horizons:
        vals: List[float] = []
        for f in result.fills:
            idx = tick_to_idx.get(f.tick)
            if idx is None:
                continue
            future_idx = idx + h
            if future_idx >= n:
                continue  # can't measure this horizon near the end of the session
            future_mid = mid[future_idx]
            if f.side == "buy":
                bps = (f.price - future_mid) / f.price * 1e4
            else:
                bps = (future_mid - f.price) / f.price * 1e4
            vals.append(bps)
        arr = np.array(vals) if vals else np.array([np.nan])
        out[h] = MarkoutSummary(
            horizon_ticks=h,
            mean_bps=float(np.nanmean(arr)),
            median_bps=float(np.nanmedian(arr)),
            std_bps=float(np.nanstd(arr)),
            n_fills=len(vals),
            per_fill_bps=arr,
        )
    return out


@dataclass
class InventoryRiskSummary:
    inventory_variance: float
    max_abs_inventory: float
    pnl_total: float
    pnl_vol: float
    spread_pnl_total: float
    inventory_pnl_total: float
    spread_pnl_vol: float
    inventory_pnl_vol: float
    sharpe_like: float  # mean pnl increment / std pnl increment, risk-adjusted proxy


def compute_inventory_risk(result: EngineResult) -> InventoryRiskSummary:
    inv = result.inventory
    pnl = result.mtm_pnl
    spread_pnl = result.spread_pnl_cum
    inventory_pnl = result.inventory_pnl_cum

    pnl_incr = np.diff(pnl)
    spread_incr = np.diff(spread_pnl)
    inv_incr = np.diff(inventory_pnl)

    pnl_vol = float(np.std(pnl_incr, ddof=1)) if len(pnl_incr) > 1 else 0.0
    sharpe_like = float(np.mean(pnl_incr) / pnl_vol) if pnl_vol > 0 else float("nan")

    return InventoryRiskSummary(
        inventory_variance=float(np.var(inv, ddof=1)),
        max_abs_inventory=float(np.max(np.abs(inv))),
        pnl_total=float(pnl[-1]),
        pnl_vol=pnl_vol,
        spread_pnl_total=float(spread_pnl[-1]),
        inventory_pnl_total=float(inventory_pnl[-1]),
        spread_pnl_vol=float(np.std(spread_incr, ddof=1)) if len(spread_incr) > 1 else 0.0,
        inventory_pnl_vol=float(np.std(inv_incr, ddof=1)) if len(inv_incr) > 1 else 0.0,
        sharpe_like=sharpe_like,
    )
