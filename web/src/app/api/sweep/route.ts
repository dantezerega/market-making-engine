import { NextRequest, NextResponse } from "next/server";
import { EngineParams } from "@/lib/engine";
import { runGammaSweep, geomspace } from "@/lib/sweep";

export const runtime = "nodejs";
export const maxDuration = 60;

export interface SweepRequestBody {
  nTicks?: number;
  seed?: number;
  sigma?: number;
  k?: number;
  A?: number;
  quoteSize?: number;
  naiveHalfSpread?: number;
  gammaMin?: number;
  gammaMax?: number;
  nPoints?: number;
}

export async function POST(req: NextRequest) {
  const body = (await req.json()) as SweepRequestBody;

  const nTicks = body.nTicks ?? 1500; // smaller than main sim so 12-point sweep stays fast
  const seed = body.seed ?? 123;
  const sigma = body.sigma ?? 0.0503;
  const k = body.k ?? 10.79;
  const A = body.A ?? 0.75;
  const quoteSize = body.quoteSize ?? 10.0;
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
    gamma: 0.005,
    sigma,
    k,
    A,
    quoteSize,
    naiveHalfSpread,
  };

  const gammas = geomspace(body.gammaMin ?? 0.0008, body.gammaMax ?? 0.05, body.nPoints ?? 12);
  const points = runGammaSweep(base, gammas);

  return NextResponse.json({ points });
}
