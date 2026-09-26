/**
 * Background price process and order-flow generation — ported from
 * lob_sim/simulation.py. Ornstein-Uhlenbeck mean-reverting mid-price plus
 * Poisson market/limit order arrivals, log-normal order sizes.
 */
import { Rng } from "./rng";
import { OrderBook } from "./orderbook";

export function ouStep(price: number, mu: number, theta: number, sigma: number, dt: number, epsilon: number): number {
  return price + theta * (mu - price) * dt + sigma * Math.sqrt(dt) * epsilon;
}

export function sampleOrderSize(mu: number, sigma: number, rng: Rng): number {
  return Math.exp(mu + sigma * rng.standardNormal());
}

export interface SimBackgroundParams {
  ouMu: number;
  ouTheta: number;
  ouSigma: number;
  dt: number;
  lambdaMarket: number;
  lambdaLimit: number;
  orderSizeMu: number;
  orderSizeSigma: number;
  tickSize: number;
  initialSpread: number;
  nInitialLevels: number;
}

export function initializeBook(book: OrderBook, midPrice: number, params: SimBackgroundParams, rng: Rng): void {
  const halfSpread = params.initialSpread / 2.0;
  for (let i = 0; i < params.nInitialLevels; i++) {
    const bidPrice = round(midPrice - halfSpread - i * params.tickSize, 6);
    const askPrice = round(midPrice + halfSpread + i * params.tickSize, 6);
    const bidQty = rng.integer(50, 200);
    const askQty = rng.integer(50, 200);
    if (bidPrice > 0) book.addLimitOrder("bid", bidPrice, bidQty, 0.0);
    if (askPrice > 0) book.addLimitOrder("ask", askPrice, askQty, 0.0);
  }
}

export function round(x: number, decimals: number): number {
  const f = Math.pow(10, decimals);
  return Math.round(x * f) / f;
}

/** One tick of background flow: diffuse price, generate market+limit orders. */
export function stepBackgroundFlow(
  book: OrderBook,
  price: number,
  timestamp: number,
  params: SimBackgroundParams,
  rng: Rng
): number {
  const epsilon = rng.standardNormal();
  const newPrice = ouStep(price, params.ouMu, params.ouTheta, params.ouSigma, params.dt, epsilon);

  const nMarket = rng.poisson(params.lambdaMarket * params.dt);
  const nLimit = rng.poisson(params.lambdaLimit * params.dt);

  for (let i = 0; i < nMarket; i++) {
    const side = rng.next() < 0.5 ? "bid" : "ask";
    const qty = sampleOrderSize(params.orderSizeMu, params.orderSizeSigma, rng);
    book.addMarketOrder(side, qty, timestamp);
  }

  for (let i = 0; i < nLimit; i++) {
    const side = rng.next() < 0.5 ? "bid" : "ask";
    const qty = sampleOrderSize(params.orderSizeMu, params.orderSizeSigma, rng);
    const offset = rng.exponential(0.05);
    const limitPrice = side === "bid" ? round(newPrice - offset, 4) : round(newPrice + offset, 4);
    if (limitPrice >= params.tickSize) {
      book.addLimitOrder(side, limitPrice, qty, timestamp);
    }
  }

  return newPrice;
}
