/**
 * Avellaneda-Stoikov / naive market-making engine — ported from
 * as_model/engine.py. Drives the TS OrderBook + background flow, runs
 * either strategy, tracks fills/inventory/cash, decomposes P&L into
 * spread capture vs inventory mark-to-market.
 */
import { OrderBook, Order } from "./orderbook";
import { Rng } from "./rng";
import { initializeBook, stepBackgroundFlow, SimBackgroundParams } from "./simulation";
import { optimalQuotes } from "./model";

export type StrategyName = "avellaneda_stoikov" | "naive";

export interface EngineParams {
  nTicks: number;
  dt: number;
  ouMu: number;
  ouTheta: number;
  ouSigma: number;
  lambdaMarket: number;
  lambdaLimit: number;
  orderSizeMu: number;
  orderSizeSigma: number;
  tickSize: number;
  initialSpread: number;
  nInitialLevels: number;
  seed: number;

  strategy: StrategyName;
  gamma: number;
  sigma: number;
  k: number;
  A: number;
  quoteSize: number;
  naiveHalfSpread: number;
}

export interface Fill {
  tick: number;
  timestamp: number;
  side: "buy" | "sell";
  price: number;
  quantity: number;
  midAtFill: number;
}

export interface EngineResult {
  ticks: number[];
  midPrice: number[];
  reservationPrice: number[];
  bidQuote: number[];
  askQuote: number[];
  inventory: number[];
  cash: number[];
  mtmPnl: number[];
  spreadPnlCum: number[];
  inventoryPnlCum: number[];
  fills: Fill[];
}

function cancelIfOpen(book: OrderBook, order: Order | null): void {
  if (order && order.status !== "filled" && order.status !== "cancelled") {
    book.cancelOrder(order.orderId);
  }
}

function round(x: number, decimals: number): number {
  const f = Math.pow(10, decimals);
  return Math.round(x * f) / f;
}

export function runMmSession(params: EngineParams): EngineResult {
  const rng = new Rng(params.seed);
  const book = new OrderBook();

  const bgParams: SimBackgroundParams = {
    ouMu: params.ouMu,
    ouTheta: params.ouTheta,
    ouSigma: params.ouSigma,
    dt: params.dt,
    lambdaMarket: params.lambdaMarket,
    lambdaLimit: params.lambdaLimit,
    orderSizeMu: params.orderSizeMu,
    orderSizeSigma: params.orderSizeSigma,
    tickSize: params.tickSize,
    initialSpread: params.initialSpread,
    nInitialLevels: params.nInitialLevels,
  };

  let price = params.ouMu;
  initializeBook(book, price, bgParams, rng);

  const T = params.nTicks * params.dt;

  const ticks: number[] = [0];
  const midHist: number[] = [price];
  const resvHist: number[] = [price];
  const bidQHist: number[] = [price];
  const askQHist: number[] = [price];
  const invHist: number[] = [0];
  const cashHist: number[] = [0];

  let inventory = 0;
  let cash = 0;
  let myBidOrder: Order | null = null;
  let myAskOrder: Order | null = null;
  const fills: Fill[] = [];
  const seenTradeIds = new Set<number>();

  let timestamp = 0;

  for (let tick = 1; tick <= params.nTicks; tick++) {
    timestamp += params.dt;
    const timeLeft = Math.max(T - timestamp, params.dt);

    // 1. cancel stale MM quotes
    cancelIfOpen(book, myBidOrder);
    cancelIfOpen(book, myAskOrder);

    // 2. compute new quotes
    let bidPx: number, askPx: number, resv: number;
    if (params.strategy === "avellaneda_stoikov") {
      const q = optimalQuotes(price, inventory, params.gamma, params.sigma, timeLeft, params.k);
      bidPx = q.bid;
      askPx = q.ask;
      resv = q.reservationPrice;
    } else {
      resv = price;
      bidPx = price - params.naiveHalfSpread;
      askPx = price + params.naiveHalfSpread;
    }
    bidPx = round(Math.max(bidPx, params.tickSize), 4);
    askPx = round(Math.max(askPx, bidPx + params.tickSize), 4);

    const [bidOrder, bidTrades] = book.addLimitOrder("bid", bidPx, params.quoteSize, timestamp);
    const [askOrder, askTrades] = book.addLimitOrder("ask", askPx, params.quoteSize, timestamp);
    myBidOrder = bidOrder;
    myAskOrder = askOrder;

    // 3. background flow steps (diffuses price, generates market/limit orders)
    price = stepBackgroundFlow(book, price, timestamp, bgParams, rng);

    // 4. prune stale levels
    const staleRange = 5.0 * params.ouSigma * Math.sqrt(params.nTicks * params.dt);
    book.pruneStaleLevels(price - staleRange, price + staleRange);

    // 5. attribute fills
    const myIds = new Set<number>();
    if (myBidOrder) myIds.add(myBidOrder.orderId);
    if (myAskOrder) myIds.add(myAskOrder.orderId);

    const candidateTrades = [...bidTrades, ...askTrades, ...book.trades.slice(-50)];
    for (const trade of candidateTrades) {
      if (seenTradeIds.has(trade.tradeId)) continue;
      seenTradeIds.add(trade.tradeId);
      if (myIds.has(trade.buyOrderId)) {
        cash -= trade.price * trade.quantity;
        inventory += trade.quantity;
        fills.push({ tick, timestamp, side: "buy", price: trade.price, quantity: trade.quantity, midAtFill: price });
      } else if (myIds.has(trade.sellOrderId)) {
        cash += trade.price * trade.quantity;
        inventory -= trade.quantity;
        fills.push({ tick, timestamp, side: "sell", price: trade.price, quantity: trade.quantity, midAtFill: price });
      }
    }

    ticks.push(tick);
    midHist.push(price);
    resvHist.push(resv);
    bidQHist.push(bidPx);
    askQHist.push(askPx);
    invHist.push(inventory);
    cashHist.push(cash);
  }

  const mtmPnl = midHist.map((mid, i) => cashHist[i] + invHist[i] * mid);

  const fillByTick = new Map<number, number>();
  for (const f of fills) {
    const edge = f.side === "sell" ? (f.price - f.midAtFill) * f.quantity : (f.midAtFill - f.price) * f.quantity;
    fillByTick.set(f.tick, (fillByTick.get(f.tick) ?? 0) + edge);
  }
  const spreadPnlCum: number[] = [];
  let running = 0;
  for (const tk of ticks) {
    running += fillByTick.get(tk) ?? 0;
    spreadPnlCum.push(running);
  }
  const inventoryPnlCum = mtmPnl.map((v, i) => v - spreadPnlCum[i]);

  return {
    ticks,
    midPrice: midHist,
    reservationPrice: resvHist,
    bidQuote: bidQHist,
    askQuote: askQHist,
    inventory: invHist,
    cash: cashHist,
    mtmPnl,
    spreadPnlCum,
    inventoryPnlCum,
    fills,
  };
}
