/**
 * Deterministic, seedable PRNG (mulberry32) plus the sampling helpers the
 * engine needs (standard normal via Box-Muller, Poisson via Knuth's
 * algorithm, exponential via inverse-CDF). This does not need to bit-match
 * the Python/numpy reference RNG — it only needs to be a valid,
 * reproducible-per-seed stochastic process for the web demo.
 */

export class Rng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  // uniform in [0, 1)
  next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  standardNormal(): number {
    // Box-Muller
    let u = 0;
    let v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  poisson(lambda: number): number {
    if (lambda <= 0) return 0;
    if (lambda < 30) {
      // Knuth's algorithm
      const L = Math.exp(-lambda);
      let k = 0;
      let p = 1;
      do {
        k++;
        p *= this.next();
      } while (p > L);
      return k - 1;
    }
    // normal approximation for large lambda
    const n = Math.round(lambda + Math.sqrt(lambda) * this.standardNormal());
    return Math.max(0, n);
  }

  exponential(scale: number): number {
    const u = this.next();
    return -scale * Math.log(1 - u);
  }

  integer(lo: number, hi: number): number {
    return Math.floor(this.next() * (hi - lo)) + lo;
  }
}
