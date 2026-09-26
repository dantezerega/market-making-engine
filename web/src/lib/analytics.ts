/**
 * Adverse-selection markout and inventory-risk/P&L-decomposition metrics.
 * Ported from as_model/analytics.py.
 */
import { EngineResult } from "./engine";

export interface MarkoutSummary {
  horizonTicks: number;
  meanBps: number;
  medianBps: number;
  stdBps: number;
  nFills: number;
}

export function computeMarkouts(result: EngineResult, horizons: number[]): Record<number, MarkoutSummary> {
  const mid = result.midPrice;
  const tickToIdx = new Map<number, number>();
  result.ticks.forEach((t, i) => tickToIdx.set(t, i));
  const n = result.ticks.length;

  const out: Record<number, MarkoutSummary> = {};
  for (const h of horizons) {
    const vals: number[] = [];
    for (const f of result.fills) {
      const idx = tickToIdx.get(f.tick);
      if (idx === undefined) continue;
      const futureIdx = idx + h;
      if (futureIdx >= n) continue;
      const futureMid = mid[futureIdx];
      const bps = f.side === "buy" ? ((f.price - futureMid) / f.price) * 1e4 : ((futureMid - f.price) / f.price) * 1e4;
      vals.push(bps);
    }
    if (vals.length === 0) {
      out[h] = { horizonTicks: h, meanBps: NaN, medianBps: NaN, stdBps: NaN, nFills: 0 };
      continue;
    }
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const sorted = [...vals].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const variance = vals.length > 1 ? vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (vals.length - 1) : 0;
    out[h] = { horizonTicks: h, meanBps: mean, medianBps: median, stdBps: Math.sqrt(variance), nFills: vals.length };
  }
  return out;
}

export interface InventoryRiskSummary {
  inventoryVariance: number;
  maxAbsInventory: number;
  pnlTotal: number;
  pnlVol: number;
  spreadPnlTotal: number;
  inventoryPnlTotal: number;
  spreadPnlVol: number;
  inventoryPnlVol: number;
  sharpeLike: number;
}

function variance(arr: number[]): number {
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return arr.length > 1 ? arr.reduce((a, b) => a + (b - mean) ** 2, 0) / (arr.length - 1) : 0;
}

function diff(arr: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < arr.length; i++) out.push(arr[i] - arr[i - 1]);
  return out;
}

export function computeInventoryRisk(result: EngineResult): InventoryRiskSummary {
  const inv = result.inventory;
  const pnl = result.mtmPnl;
  const spreadPnl = result.spreadPnlCum;
  const inventoryPnl = result.inventoryPnlCum;

  const pnlIncr = diff(pnl);
  const spreadIncr = diff(spreadPnl);
  const invIncr = diff(inventoryPnl);

  const pnlVol = pnlIncr.length > 1 ? Math.sqrt(variance(pnlIncr)) : 0;
  const meanPnlIncr = pnlIncr.reduce((a, b) => a + b, 0) / pnlIncr.length;
  const sharpeLike = pnlVol > 0 ? meanPnlIncr / pnlVol : NaN;

  return {
    inventoryVariance: variance(inv),
    maxAbsInventory: Math.max(...inv.map(Math.abs)),
    pnlTotal: pnl[pnl.length - 1],
    pnlVol,
    spreadPnlTotal: spreadPnl[spreadPnl.length - 1],
    inventoryPnlTotal: inventoryPnl[inventoryPnl.length - 1],
    spreadPnlVol: spreadIncr.length > 1 ? Math.sqrt(variance(spreadIncr)) : 0,
    inventoryPnlVol: invIncr.length > 1 ? Math.sqrt(variance(invIncr)) : 0,
    sharpeLike,
  };
}
