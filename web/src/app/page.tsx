"use client";

import { useState, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from "recharts";

interface RiskSummary {
  inventoryVariance: number;
  maxAbsInventory: number;
  pnlTotal: number;
  pnlVol: number;
  spreadPnlTotal: number;
  inventoryPnlTotal: number;
  sharpeLike: number;
}

interface MarkoutSummary {
  horizonTicks: number;
  meanBps: number;
  nFills: number;
}

interface SimResponse {
  calibration: { sigma: number; k: number; A: number; rSquared: number; nTrades: number } | null;
  params: { sigma: number; k: number; A: number; gamma: number; nTicks: number; naiveHalfSpread: number };
  as: {
    ticks: number[];
    midPrice: number[];
    reservationPrice: number[];
    bidQuote: number[];
    askQuote: number[];
    inventory: number[];
    spreadPnlCum: number[];
    inventoryPnlCum: number[];
    mtmPnl: number[];
    nFills: number;
    risk: RiskSummary;
    markouts: Record<string, MarkoutSummary>;
  };
  naive: {
    ticks: number[];
    midPrice: number[];
    inventory: number[];
    mtmPnl: number[];
    nFills: number;
    risk: RiskSummary;
    markouts: Record<string, MarkoutSummary>;
  };
}

interface SweepPoint {
  gamma: number;
  inventoryVariance: number;
  maxAbsInventory: number;
  pnlTotal: number;
  pnlVol: number;
  sharpeLike: number;
}

function fmt(n: number, decimals = 2): string {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  return n.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

export default function Home() {
  const [nTicks, setNTicks] = useState(2000);
  const [gamma, setGamma] = useState(0.005);
  const [seed, setSeed] = useState(123);
  const [quoteSize, setQuoteSize] = useState(10);
  const [autoCalibrate, setAutoCalibrate] = useState(true);

  const [loading, setLoading] = useState(false);
  const [sweepLoading, setSweepLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SimResponse | null>(null);
  const [sweepData, setSweepData] = useState<SweepPoint[] | null>(null);

  const runSimulation = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nTicks, gamma, seed, quoteSize, autoCalibrate }),
      });
      if (!res.ok) throw new Error(`Simulation failed: ${res.status}`);
      const json = (await res.json()) as SimResponse;
      setData(json);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [nTicks, gamma, seed, quoteSize, autoCalibrate]);

  const runSweep = useCallback(async () => {
    setSweepLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/sweep", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed,
          quoteSize,
          sigma: data?.params.sigma,
          k: data?.params.k,
          A: data?.params.A,
          naiveHalfSpread: data?.params.naiveHalfSpread,
        }),
      });
      if (!res.ok) throw new Error(`Sweep failed: ${res.status}`);
      const json = (await res.json()) as { points: SweepPoint[] };
      setSweepData(json.points);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSweepLoading(false);
    }
  }, [seed, quoteSize, data]);

  const quotesChartData =
    data?.as.ticks.map((t, i) => ({
      tick: t,
      mid: data.as.midPrice[i],
      reservation: data.as.reservationPrice[i],
      bid: data.as.bidQuote[i],
      ask: data.as.askQuote[i],
    })) ?? [];

  const inventoryChartData =
    data?.as.ticks.map((t, i) => ({
      tick: t,
      as: data.as.inventory[i],
      naive: data.naive.inventory[i],
    })) ?? [];

  const pnlChartData =
    data?.as.ticks.map((t, i) => ({
      tick: t,
      as: data.as.mtmPnl[i],
      naive: data.naive.mtmPnl[i],
    })) ?? [];

  const pnlDecompChartData =
    data?.as.ticks.map((t, i) => ({
      tick: t,
      spread: data.as.spreadPnlCum[i],
      inventory: data.as.inventoryPnlCum[i],
      total: data.as.mtmPnl[i],
    })) ?? [];

  const markoutChartData = data
    ? [1, 10, 100].map((h) => ({
        horizon: h,
        as: data.as.markouts[h]?.meanBps,
        naive: data.naive.markouts[h]?.meanBps,
      }))
    : [];

  return (
    <div className="container">
      <h1>Avellaneda-Stoikov Market Making Engine</h1>
      <p className="subtitle">
        Closed-form optimal market-making model (Avellaneda &amp; Stoikov, 2008), run live against a simulated
        limit order book. Tune the parameters below, run a session, and compare against a naive fixed-spread
        baseline on inventory risk, P&amp;L decomposition, and adverse-selection markout.{" "}
        <a href="https://github.com/dantezerega/market-making-engine" target="_blank" rel="noreferrer">
          Full derivation &amp; source →
        </a>
      </p>

      <div className="panel">
        <h2>Parameters</h2>
        <div className="controls-grid">
          <div className="control">
            <label>
              <span>Ticks (session length)</span>
              <span>{nTicks}</span>
            </label>
            <input
              type="range"
              min={200}
              max={4000}
              step={100}
              value={nTicks}
              onChange={(e) => setNTicks(Number(e.target.value))}
            />
          </div>
          <div className="control">
            <label>
              <span>Gamma (risk aversion)</span>
              <span>{gamma.toFixed(4)}</span>
            </label>
            <input
              type="range"
              min={0.0005}
              max={0.05}
              step={0.0005}
              value={gamma}
              onChange={(e) => setGamma(Number(e.target.value))}
            />
          </div>
          <div className="control">
            <label>
              <span>Quote size</span>
              <span>{quoteSize}</span>
            </label>
            <input
              type="range"
              min={1}
              max={50}
              step={1}
              value={quoteSize}
              onChange={(e) => setQuoteSize(Number(e.target.value))}
            />
          </div>
          <div className="control">
            <label>
              <span>Seed</span>
            </label>
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </div>
        </div>
        <div className="btn-row">
          <button onClick={runSimulation} disabled={loading}>
            {loading && <span className="spinner" />}
            {loading ? "Running…" : "Run simulation"}
          </button>
          <button className="secondary" onClick={runSweep} disabled={sweepLoading || !data}>
            {sweepLoading && <span className="spinner" />}
            {sweepLoading ? "Sweeping…" : "Run gamma sweep"}
          </button>
          <label className="checkbox">
            <input type="checkbox" checked={autoCalibrate} onChange={(e) => setAutoCalibrate(e.target.checked)} />
            Calibrate sigma/k/A from a fresh background LOB run (unchecked = use last calibrated / default values)
          </label>
        </div>
        {error && <p style={{ color: "var(--danger)", marginTop: 10 }}>{error}</p>}
        {data?.calibration && (
          <p className="footer-note" style={{ marginTop: 12 }}>
            Calibrated from a 3,000-tick background (no-MM) run: sigma≈{fmt(data.calibration.sigma, 5)}, k≈
            {fmt(data.calibration.k, 3)}, A≈{fmt(data.calibration.A, 3)} (log-linear fit R²≈
            {fmt(data.calibration.rSquared, 3)}, n_trades={data.calibration.nTrades}).
          </p>
        )}
      </div>

      {data && (
        <>
          <div className="panel">
            <h2>Head-to-head: Avellaneda-Stoikov vs naive symmetric baseline</h2>
            <div className="summary-banner">
              <div className="summary-stat">
                <div className="label">Inventory variance reduction</div>
                <div className="value" style={{ color: "var(--accent2)" }}>
                  {fmt(
                    (1 - data.as.risk.inventoryVariance / data.naive.risk.inventoryVariance) * 100,
                    1
                  )}
                  %
                </div>
              </div>
              <div className="summary-stat">
                <div className="label">AS total P&amp;L</div>
                <div className="value" style={{ color: data.as.risk.pnlTotal >= 0 ? "var(--accent2)" : "var(--danger)" }}>
                  {fmt(data.as.risk.pnlTotal)}
                </div>
              </div>
              <div className="summary-stat">
                <div className="label">Naive total P&amp;L</div>
                <div
                  className="value"
                  style={{ color: data.naive.risk.pnlTotal >= 0 ? "var(--accent2)" : "var(--danger)" }}
                >
                  {fmt(data.naive.risk.pnlTotal)}
                </div>
              </div>
              <div className="summary-stat">
                <div className="label">Risk-adjusted P&amp;L (AS vs naive)</div>
                <div className="value">
                  {fmt(data.as.risk.sharpeLike, 4)} vs {fmt(data.naive.risk.sharpeLike, 4)}
                </div>
              </div>
            </div>

            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th className="as-col">Avellaneda-Stoikov</th>
                  <th className="naive-col">Naive symmetric</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Inventory variance</td>
                  <td className="as-col">{fmt(data.as.risk.inventoryVariance)}</td>
                  <td className="naive-col">{fmt(data.naive.risk.inventoryVariance)}</td>
                </tr>
                <tr>
                  <td>Max |inventory|</td>
                  <td className="as-col">{fmt(data.as.risk.maxAbsInventory, 1)}</td>
                  <td className="naive-col">{fmt(data.naive.risk.maxAbsInventory, 1)}</td>
                </tr>
                <tr>
                  <td>Fills</td>
                  <td className="as-col">{data.as.nFills}</td>
                  <td className="naive-col">{data.naive.nFills}</td>
                </tr>
                <tr>
                  <td>Spread P&amp;L</td>
                  <td className="as-col">{fmt(data.as.risk.spreadPnlTotal)}</td>
                  <td className="naive-col">{fmt(data.naive.risk.spreadPnlTotal)}</td>
                </tr>
                <tr>
                  <td>Inventory P&amp;L</td>
                  <td className="as-col">{fmt(data.as.risk.inventoryPnlTotal)}</td>
                  <td className="naive-col">{fmt(data.naive.risk.inventoryPnlTotal)}</td>
                </tr>
                <tr>
                  <td>P&amp;L volatility (per tick)</td>
                  <td className="as-col">{fmt(data.as.risk.pnlVol, 4)}</td>
                  <td className="naive-col">{fmt(data.naive.risk.pnlVol, 4)}</td>
                </tr>
                <tr>
                  <td>Markout @1 tick (bps)</td>
                  <td className="as-col">{fmt(data.as.markouts[1]?.meanBps, 3)}</td>
                  <td className="naive-col">{fmt(data.naive.markouts[1]?.meanBps, 3)}</td>
                </tr>
                <tr>
                  <td>Markout @10 ticks (bps)</td>
                  <td className="as-col">{fmt(data.as.markouts[10]?.meanBps, 3)}</td>
                  <td className="naive-col">{fmt(data.naive.markouts[10]?.meanBps, 3)}</td>
                </tr>
                <tr>
                  <td>Markout @100 ticks (bps)</td>
                  <td className="as-col">{fmt(data.as.markouts[100]?.meanBps, 3)}</td>
                  <td className="naive-col">{fmt(data.naive.markouts[100]?.meanBps, 3)}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="panel">
            <h2>Reservation price &amp; quotes vs mid</h2>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={quotesChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#232838" />
                <XAxis dataKey="tick" stroke="#8b93a7" fontSize={11} />
                <YAxis stroke="#8b93a7" fontSize={11} domain={["auto", "auto"]} />
                <Tooltip contentStyle={{ background: "#131722", border: "1px solid #232838" }} />
                <Legend />
                <Line type="monotone" dataKey="mid" stroke="#e6e9f0" dot={false} name="Mid price" strokeWidth={1.4} />
                <Line
                  type="monotone"
                  dataKey="reservation"
                  stroke="#c792ea"
                  dot={false}
                  name="Reservation price"
                  strokeWidth={1.2}
                />
                <Line type="monotone" dataKey="bid" stroke="#33d69f" dot={false} name="Bid" strokeWidth={0.8} strokeOpacity={0.7} />
                <Line type="monotone" dataKey="ask" stroke="#ff6b6b" dot={false} name="Ask" strokeWidth={0.8} strokeOpacity={0.7} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-grid two-col">
            <div className="panel">
              <h2>Inventory path: AS vs naive</h2>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={inventoryChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#232838" />
                  <XAxis dataKey="tick" stroke="#8b93a7" fontSize={11} />
                  <YAxis stroke="#8b93a7" fontSize={11} />
                  <Tooltip contentStyle={{ background: "#131722", border: "1px solid #232838" }} />
                  <Legend />
                  <Line type="monotone" dataKey="as" stroke="#5b8cff" dot={false} name="Avellaneda-Stoikov" strokeWidth={1.4} />
                  <Line type="monotone" dataKey="naive" stroke="#8b93a7" dot={false} name="Naive symmetric" strokeWidth={1.2} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="panel">
              <h2>Cumulative P&amp;L: AS vs naive</h2>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={pnlChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#232838" />
                  <XAxis dataKey="tick" stroke="#8b93a7" fontSize={11} />
                  <YAxis stroke="#8b93a7" fontSize={11} />
                  <Tooltip contentStyle={{ background: "#131722", border: "1px solid #232838" }} />
                  <Legend />
                  <Line type="monotone" dataKey="as" stroke="#5b8cff" dot={false} name="Avellaneda-Stoikov" strokeWidth={1.4} />
                  <Line type="monotone" dataKey="naive" stroke="#8b93a7" dot={false} name="Naive symmetric" strokeWidth={1.2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="panel">
            <h2>P&amp;L decomposition (Avellaneda-Stoikov): spread capture vs inventory swings</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={pnlDecompChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#232838" />
                <XAxis dataKey="tick" stroke="#8b93a7" fontSize={11} />
                <YAxis stroke="#8b93a7" fontSize={11} />
                <Tooltip contentStyle={{ background: "#131722", border: "1px solid #232838" }} />
                <Legend />
                <Line type="monotone" dataKey="spread" stroke="#33d69f" dot={false} name="Spread P&L" strokeWidth={1.4} />
                <Line type="monotone" dataKey="inventory" stroke="#ffb84d" dot={false} name="Inventory P&L" strokeWidth={1.4} />
                <Line type="monotone" dataKey="total" stroke="#e6e9f0" dot={false} name="Total P&L" strokeWidth={1.6} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="panel">
            <h2>Adverse-selection markout curves</h2>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={markoutChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#232838" />
                <XAxis dataKey="horizon" stroke="#8b93a7" fontSize={11} scale="log" domain={["auto", "auto"]} />
                <YAxis stroke="#8b93a7" fontSize={11} label={{ value: "bps (+ = adverse)", angle: -90, position: "insideLeft", fill: "#8b93a7", fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#131722", border: "1px solid #232838" }} />
                <Legend />
                <Line type="monotone" dataKey="as" stroke="#5b8cff" name="Avellaneda-Stoikov" strokeWidth={2} />
                <Line type="monotone" dataKey="naive" stroke="#8b93a7" name="Naive symmetric" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {sweepData && (
        <div className="panel">
          <h2>Gamma sensitivity sweep — the risk/return dial</h2>
          <div className="chart-grid two-col">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={sweepData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#232838" />
                <XAxis dataKey="gamma" stroke="#8b93a7" fontSize={11} scale="log" domain={["auto", "auto"]} />
                <YAxis stroke="#8b93a7" fontSize={11} />
                <Tooltip contentStyle={{ background: "#131722", border: "1px solid #232838" }} />
                <Legend />
                <Line type="monotone" dataKey="inventoryVariance" stroke="#ff6b6b" name="Inventory variance" strokeWidth={2} dot />
              </LineChart>
            </ResponsiveContainer>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={sweepData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#232838" />
                <XAxis dataKey="gamma" stroke="#8b93a7" fontSize={11} scale="log" domain={["auto", "auto"]} />
                <YAxis stroke="#8b93a7" fontSize={11} />
                <Tooltip contentStyle={{ background: "#131722", border: "1px solid #232838" }} />
                <Legend />
                <Line type="monotone" dataKey="pnlTotal" stroke="#33d69f" name="Total P&L" strokeWidth={2} dot />
                <Line type="monotone" dataKey="sharpeLike" stroke="#c792ea" name="Risk-adjusted P&L proxy" strokeWidth={2} dot />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <p className="footer-note">
        Model: r(s,q,t) = s − qγσ²(T−t); δ = γσ²(T−t) + (2/γ)ln(1+γ/k). Naive baseline uses a fixed half-spread
        matched to the AS session&apos;s time-average half-spread, isolating the inventory-skew effect. All
        computation runs server-side in a Vercel serverless function per request — nothing is persisted.
      </p>
    </div>
  );
}
