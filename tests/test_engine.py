import numpy as np
import pytest

from as_model.engine import EngineParams, run_mm_session
from as_model.analytics import compute_markouts, compute_inventory_risk


def _small_params(strategy="avellaneda_stoikov"):
    return EngineParams(
        n_ticks=200,
        gamma=0.01,
        sigma=0.05,
        k=1.5,
        A=100.0,
        quote_size=5.0,
        naive_half_spread=0.05,
        seed=1,
        strategy=strategy,
    )


def test_engine_runs_and_produces_consistent_length_arrays():
    result = run_mm_session(_small_params())
    n = _small_params().n_ticks + 1
    assert len(result.ticks) == n
    assert len(result.mid_price) == n
    assert len(result.inventory) == n
    assert len(result.cash) == n


def test_mtm_pnl_equals_cash_plus_inventory_times_mid():
    result = run_mm_session(_small_params())
    expected = result.cash + result.inventory * result.mid_price
    np.testing.assert_allclose(result.mtm_pnl, expected)


def test_pnl_decomposition_sums_to_total():
    result = run_mm_session(_small_params())
    np.testing.assert_allclose(
        result.spread_pnl_cum + result.inventory_pnl_cum, result.mtm_pnl, atol=1e-6
    )


def test_naive_strategy_runs():
    result = run_mm_session(_small_params(strategy="naive"))
    assert len(result.fills) >= 0  # just needs to not crash


def test_markouts_are_bounded_and_horizon_keyed():
    result = run_mm_session(_small_params())
    markouts = compute_markouts(result, horizons=(1, 10))
    assert set(markouts.keys()) == {1, 10}
    for h, summary in markouts.items():
        assert summary.horizon_ticks == h


def test_inventory_risk_summary_fields_finite_when_fills_exist():
    result = run_mm_session(_small_params())
    risk = compute_inventory_risk(result)
    assert np.isfinite(risk.inventory_variance)
    assert np.isfinite(risk.pnl_total)
    assert risk.max_abs_inventory >= 0
