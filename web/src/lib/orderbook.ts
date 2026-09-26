/**
 * Limit order book matching engine — ported from lob_sim/orderbook.py and
 * lob_sim/orders.py (Python reference, vendored from dantezerega/orderbook).
 * Price-time priority (FIFO at each price level), same semantics as the
 * Python version: add_limit_order returns [order, trades], cancel_order,
 * best_bid/best_ask, prune_stale_levels.
 */

export type OrderSide = "bid" | "ask";
export type OrderStatus = "open" | "partial" | "filled" | "cancelled";

export interface Order {
  orderId: number;
  side: OrderSide;
  quantity: number;
  filledQuantity: number;
  timestamp: number;
  price: number;
  status: OrderStatus;
}

export interface Trade {
  tradeId: number;
  buyOrderId: number;
  sellOrderId: number;
  price: number;
  quantity: number;
  timestamp: number;
}

function remaining(o: Order): number {
  return o.quantity - o.filledQuantity;
}

export class OrderBook {
  private bids: Map<number, Order[]> = new Map();
  private asks: Map<number, Order[]> = new Map();
  private orders: Map<number, Order> = new Map();
  private tradesList: Trade[] = [];
  private nextOrderId = 1;
  private nextTradeId = 1;

  get bestBid(): number | null {
    if (this.bids.size === 0) return null;
    return Math.max(...this.bids.keys());
  }

  get bestAsk(): number | null {
    if (this.asks.size === 0) return null;
    return Math.min(...this.asks.keys());
  }

  get trades(): Trade[] {
    return this.tradesList;
  }

  addLimitOrder(side: OrderSide, price: number, quantity: number, timestamp: number): [Order, Trade[]] {
    const order: Order = {
      orderId: this.nextOrderId++,
      side,
      quantity,
      filledQuantity: 0,
      timestamp,
      price,
      status: "open",
    };
    this.orders.set(order.orderId, order);
    const trades = this.matchLimit(order);
    if (remaining(order) > 0 && order.status !== "filled") {
      const book = side === "bid" ? this.bids : this.asks;
      if (!book.has(price)) book.set(price, []);
      book.get(price)!.push(order);
    }
    return [order, trades];
  }

  addMarketOrder(side: OrderSide, quantity: number, timestamp: number): [Order, Trade[]] {
    const order: Order = {
      orderId: this.nextOrderId++,
      side,
      quantity,
      filledQuantity: 0,
      timestamp,
      price: NaN,
      status: "open",
    };
    this.orders.set(order.orderId, order);
    const trades = this.matchMarket(order);
    return [order, trades];
  }

  cancelOrder(orderId: number): boolean {
    const order = this.orders.get(orderId);
    if (!order || order.status === "filled" || order.status === "cancelled") return false;
    order.status = "cancelled";
    const book = order.side === "bid" ? this.bids : this.asks;
    const level = book.get(order.price);
    if (level) {
      const idx = level.findIndex((o) => o.orderId === orderId);
      if (idx >= 0) level.splice(idx, 1);
      if (level.length === 0) book.delete(order.price);
    }
    return true;
  }

  pruneStaleLevels(minBidPrice: number, maxAskPrice: number): void {
    for (const price of Array.from(this.bids.keys())) {
      if (price < minBidPrice) {
        for (const o of [...(this.bids.get(price) ?? [])]) this.cancelOrder(o.orderId);
      }
    }
    for (const price of Array.from(this.asks.keys())) {
      if (price > maxAskPrice) {
        for (const o of [...(this.asks.get(price) ?? [])]) this.cancelOrder(o.orderId);
      }
    }
  }

  private matchLimit(order: Order): Trade[] {
    const trades: Trade[] = [];
    const contraBook = order.side === "bid" ? this.asks : this.bids;
    const priceOk = (p: number) => (order.side === "bid" ? p <= order.price : p >= order.price);
    while (remaining(order) > 0) {
      const available = Array.from(contraBook.keys())
        .filter(priceOk)
        .sort((a, b) => (order.side === "bid" ? a - b : b - a));
      if (available.length === 0) break;
      trades.push(...this.fillAgainstLevel(order, contraBook, available[0]));
    }
    return trades;
  }

  private matchMarket(order: Order): Trade[] {
    const trades: Trade[] = [];
    const contraBook = order.side === "bid" ? this.asks : this.bids;
    while (remaining(order) > 0 && contraBook.size > 0) {
      const prices = Array.from(contraBook.keys()).sort((a, b) => (order.side === "bid" ? a - b : b - a));
      if (prices.length === 0) break;
      trades.push(...this.fillAgainstLevel(order, contraBook, prices[0]));
    }
    return trades;
  }

  private fillAgainstLevel(aggressor: Order, contraBook: Map<number, Order[]>, price: number): Trade[] {
    const trades: Trade[] = [];
    const level = contraBook.get(price);
    if (!level || level.length === 0) return trades;

    while (level.length > 0 && remaining(aggressor) > 0) {
      const passive = level[0];
      const fillQty = Math.min(remaining(aggressor), remaining(passive));
      const tradeId = this.nextTradeId++;

      const trade: Trade =
        aggressor.side === "bid"
          ? {
              tradeId,
              buyOrderId: aggressor.orderId,
              sellOrderId: passive.orderId,
              price,
              quantity: fillQty,
              timestamp: aggressor.timestamp,
            }
          : {
              tradeId,
              buyOrderId: passive.orderId,
              sellOrderId: aggressor.orderId,
              price,
              quantity: fillQty,
              timestamp: aggressor.timestamp,
            };

      aggressor.filledQuantity += fillQty;
      passive.filledQuantity += fillQty;
      aggressor.status = remaining(aggressor) === 0 ? "filled" : "partial";
      passive.status = remaining(passive) === 0 ? "filled" : "partial";

      if (passive.status === "filled") level.shift();

      trades.push(trade);
      this.tradesList.push(trade);
    }

    if (level.length === 0) contraBook.delete(price);
    return trades;
  }
}
