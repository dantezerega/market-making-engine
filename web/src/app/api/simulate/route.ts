import { NextRequest, NextResponse } from "next/server";
import { EngineParams, runMmSession } from "@/lib/engine";
import { computeMarkouts, computeInventoryRisk } from "@/lib/analytics";
import { calibrateFromBackgroundRun } from "@/lib/calibration";

export const runtime = "nodejs";
export const maxDuration = 30;

export interface SimulateRequestBody {
  nTicks?: number;
  gamma?: number;
  seed?: number;
  quoteSize?: number;
  autoCalibrate?: boolean;
  sigma?: number;
  k?: number;
  A?: number;
  naiveHalfSpread?: number;
}

export async function POST(req: NextRequest) {
  const body = (await req.json()) as SimulateRequestBody;

  const nTicks = body.nTicks ?? 2000;
  const seed = body.seed ?? 123;
  const gamma = body.gamma ?? 0.005;
  const quoteSize = body.quoteSize ?? 10.0;

  let sigma = body.sigma ?? 0.0503;
  let k = body.k ?? 10.79;
  let A = body.A ?? 0.75;
  let calibration = null;

  if (body.autoCalibrate) {
    const calib = calibrateFromBackgroundRun(3000, 7);
    sigma = calib.sigma;
    k = calib.k;
    A = calib.A;
    calibration = calib;
  }

  const naiveHalfSpread = body.naiveHalfSpread ?? 0.099;

  const base: EngineParams = {
    nTicks,
    dt: 1.0,
    ouMu: 100.0,
    ouTheta: 0.05,
    ouSigma: 0.05,
    lambdaMarket: 1.5,
    lambdaLimit: 4.0,
    orderSizeMu: 1.5,
    orderSizeSigma: 0.5,
    tickSize: 0.01,
    initialSpread: 0.1,
    nInitialLevels: 10,
    seed,
    strategy: "avellaneda_stoikov",
    gamma,
    sigma,
    k,
    A,
    quoteSize,
    naiveHalfSpread,
  };

  const asResult = runMmSession({ ...base, strategy: "avellaneda_stoikov" });
  const naiveResult = runMmSession({ ...base, strategy: "naive" });

  const asMarkouts = computeMarkouts(asResult, [1, 10, 100]);
  const naiveMarkouts = computeMarkouts(naiveResult, [1, 10, 100]);
  const asRisk = computeInventoryRisk(asResult);
  const naiveRisk = computeInventoryRisk(naiveResult);

  // Downsample series for payload size (keep every Nth tick beyond a cap).
  const maxPoints = 2000;
  const stride = Math.max(1, Math.floor(asResult.ticks.length / maxPoints));
  const downsample = <T,>(arr: T[]): T[] => arr.filter((_, i) => i % stride === 0);

  return NextResponse.json({
    calibration,
    params: { sigma, k, A, gamma, nTicks, naiveHalfSpread },
    as: {
      ticks: downsample(asResult.ticks),
      midPrice: downsample(asResult.midPrice),
      reservationPrice: downsample(asResult.reservationPrice),
      bidQuote: downsample(asResult.bidQuote),
      askQuote: downsample(asResult.askQuote),
      inventory: downsample(asResult.inventory),
      spreadPnlCum: downsample(asResult.spreadPnlCum),
      inventoryPnlCum: downsample(asResult.inventoryPnlCum),
      mtmPnl: downsample(asResult.mtmPnl),
      nFills: asResult.fills.length,
      risk: asRisk,
      markouts: asMarkouts,
    },
    naive: {
      ticks: downsample(naiveResult.ticks),
      midPrice: downsample(naiveResult.midPrice),
      inventory: downsample(naiveResult.inventory),
      mtmPnl: downsample(naiveResult.mtmPnl),
      nFills: naiveResult.fills.length,
      risk: naiveRisk,
      markouts: naiveMarkouts,
    },
  });
}
