/**
 * Parameter calibration — ported from as_model/calibration.py.
 * sigma from realized price increments; k, A from a log-linear fit of
 * empirical fill-intensity-vs-distance on a background (no-MM) run.
 */
import { OrderBook } from "./orderbook";
import { Rng } from "./rng";
import { initializeBook, stepBackgroundFlow, SimBackgroundParams } from "./simulation";

export function estimateSigma(pricePath: number[], dt: number): number {
  const increments: number[] = [];
  for (let i = 1; i < pricePath.length; i++) {
    increments.push(pricePath[i] - pricePath[i - 1]);
  }
  const mean = increments.reduce((a, b) => a + b, 0) / increments.length;
  const variance = increments.reduce((a, b) => a + (b - mean) ** 2, 0) / (increments.length - 1);
  return Math.sqrt(variance) / Math.sqrt(dt);
}

export interface CalibrationResult {
  sigma: number;
  k: number;
  A: number;
  rSquared: number;
  nTrades: number;
}

function linregress(x: number[], y: number[]): { slope: number; intercept: number; r: number } {
  const n = x.length;
  const meanX = x.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let denX = 0;
  let denY = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  const slope = num / denX;
  const intercept = meanY - slope * meanX;
  const r = num / Math.sqrt(denX * denY);
  return { slope, intercept, r };
}

export function calibrateFromBackgroundRun(nTicks: number, seed: number): CalibrationResult {
  const params: SimBackgroundParams = {
    ouMu: 100.0,
    ouTheta: 0.05,
    ouSigma: 0.05,
    dt: 1.0,
    lambdaMarket: 1.5,
    lambdaLimit: 4.0,
    orderSizeMu: 1.5,
    orderSizeSigma: 0.5,
    tickSize: 0.01,
    initialSpread: 0.1,
    nInitialLevels: 10,
  };

  const rng = new Rng(seed);
  const book = new OrderBook();
  let price = params.ouMu;
  initializeBook(book, price, params, rng);

  const pricePath: number[] = [price];
  const tradeRecords: Array<{ ts: number; price: number }> = [];
  let timestamp = 0;

  for (let tick = 1; tick <= nTicks; tick++) {
    timestamp += params.dt;
    const nTradesBefore = book.trades.length;
    price = stepBackgroundFlow(book, price, timestamp, params, rng);

    const staleRange = 5.0 * params.ouSigma * Math.sqrt(nTicks * params.dt);
    book.pruneStaleLevels(price - staleRange, price + staleRange);

    for (let i = nTradesBefore; i < book.trades.length; i++) {
      tradeRecords.push({ ts: timestamp, price: book.trades[i].price });
    }
    pricePath.push(price);
  }

  const sigma = estimateSigma(pricePath, params.dt);

  const distances: number[] = [];
  for (const { ts, price: tradePrice } of tradeRecords) {
    const idx = Math.min(Math.round(ts / params.dt), pricePath.length - 1);
    const fundamental = pricePath[idx];
    distances.push(Math.abs(tradePrice - fundamental));
  }

  if (distances.length < 10) {
    return { sigma, k: 1.5, A: 140.0, rSquared: 0, nTrades: distances.length };
  }

  const tradeWindowSeconds = nTicks * params.dt;
  const sorted = [...distances].sort((a, b) => a - b);
  const p95 = sorted[Math.floor(0.95 * sorted.length)];
  const maxDist = Math.max(p95, 1e-6);

  const nBins = 20;
  const edges: number[] = [];
  for (let i = 0; i <= nBins; i++) edges.push((i / nBins) * maxDist);
  const counts = new Array(nBins).fill(0);
  for (const d of distances) {
    if (d >= maxDist) continue;
    const binIdx = Math.min(nBins - 1, Math.floor((d / maxDist) * nBins));
    counts[binIdx]++;
  }
  const binCenters = edges.slice(0, nBins).map((e, i) => 0.5 * (e + edges[i + 1]));

  const minCountPerBin = 3;
  let mask = counts.map((c) => c >= minCountPerBin);
  if (mask.filter(Boolean).length < 3) {
    mask = counts.map((c) => c > 0);
  }

  const centers: number[] = [];
  const logIntensity: number[] = [];
  for (let i = 0; i < nBins; i++) {
    if (mask[i]) {
      const intensity = counts[i] / tradeWindowSeconds;
      centers.push(binCenters[i]);
      logIntensity.push(Math.log(intensity));
    }
  }

  const { slope, intercept, r } = linregress(centers, logIntensity);
  const k = -slope;
  const A = Math.exp(intercept);

  return {
    sigma,
    k: k > 0 && isFinite(k) ? k : 1.5,
    A: k > 0 && isFinite(k) ? A : 140.0,
    rSquared: r * r,
    nTrades: distances.length,
  };
}
