/**
 * Gamma sensitivity sweep — ported from as_model/sweep.py.
 */
import { EngineParams, runMmSession } from "./engine";
import { computeInventoryRisk } from "./analytics";

export interface SweepPoint {
  gamma: number;
  inventoryVariance: number;
  maxAbsInventory: number;
  pnlTotal: number;
  pnlVol: number;
  sharpeLike: number;
}

export function geomspace(start: number, stop: number, num: number): number[] {
  const logStart = Math.log(start);
  const logStop = Math.log(stop);
  const step = (logStop - logStart) / (num - 1);
  return Array.from({ length: num }, (_, i) => Math.exp(logStart + i * step));
}

export function runGammaSweep(baseParams: EngineParams, gammas: number[]): SweepPoint[] {
  return gammas.map((gamma) => {
    const p: EngineParams = { ...baseParams, gamma, strategy: "avellaneda_stoikov" };
    const result = runMmSession(p);
    const risk = computeInventoryRisk(result);
    return {
      gamma,
      inventoryVariance: risk.inventoryVariance,
      maxAbsInventory: risk.maxAbsInventory,
      pnlTotal: risk.pnlTotal,
      pnlVol: risk.pnlVol,
      sharpeLike: risk.sharpeLike,
    };
  });
}
