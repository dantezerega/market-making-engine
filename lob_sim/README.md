# LOB Simulator (vendored)

Core matching-engine modules vendored **unmodified** from
[dantezerega/orderbook](https://github.com/dantezerega/orderbook)
(commit `52d3424`): `orders.py`, `orderbook.py`, `metrics.py`,
`simulation.py`. Only the internal `from X import Y` statements were
changed to `from lob_sim.X import Y` so the modules work as a
sub-package here; no matching/order-generation logic was touched, per
the project's "no rewrite of the sim itself" requirement.

Not vendored (not needed for this project): `app.py` (Streamlit UI),
`main.py` (that repo's own CLI), `market_impact.py` (Almgren-Chriss
slippage model), `visualize.py` (Plotly charts for that repo's own UI).
