/**
 * Avellaneda-Stoikov (2008) closed-form market making model.
 * Ported 1:1 from as_model/model.py (Python reference implementation).
 * See the project README for the full HJB derivation. This module only
 * carries the closed-form results:
 *
 *   Reservation price:  r(s,q,t) = s - q * gamma * sigma^2 * (T - t)
 *   Optimal spread:     delta = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)
 *   Quotes:             bid = r - delta/2 ; ask = r + delta/2
 */

export function reservationPrice(
  mid: number,
  inventory: number,
  gamma: number,
  sigma: number,
  timeLeft: number
): number {
  return mid - inventory * gamma * sigma * sigma * timeLeft;
}

export function optimalSpread(
  gamma: number,
  sigma: number,
  timeLeft: number,
  k: number
): number {
  const inventoryTerm = gamma * sigma * sigma * timeLeft;
  const microstructureTerm = (2.0 / gamma) * Math.log(1.0 + gamma / k);
  return inventoryTerm + microstructureTerm;
}

export interface Quotes {
  bid: number;
  ask: number;
  reservationPrice: number;
  spread: number;
}

export function optimalQuotes(
  mid: number,
  inventory: number,
  gamma: number,
  sigma: number,
  timeLeft: number,
  k: number
): Quotes {
  const r = reservationPrice(mid, inventory, gamma, sigma, timeLeft);
  const spread = optimalSpread(gamma, sigma, timeLeft, k);
  return { bid: r - spread / 2, ask: r + spread / 2, reservationPrice: r, spread };
}

export function fillIntensity(delta: number, A: number, k: number): number {
  return A * Math.exp(-k * delta);
}
