import math
import numpy as np
import pytest

from as_model.model import reservation_price, optimal_spread, optimal_quotes, fill_intensity


def test_reservation_price_at_zero_inventory_equals_mid():
    assert reservation_price(mid=100.0, inventory=0.0, gamma=0.1, sigma=0.05, time_left=10.0) == pytest.approx(100.0)


def test_reservation_price_long_inventory_shifts_down():
    r_long = reservation_price(mid=100.0, inventory=5.0, gamma=0.1, sigma=0.05, time_left=10.0)
    r_flat = reservation_price(mid=100.0, inventory=0.0, gamma=0.1, sigma=0.05, time_left=10.0)
    assert r_long < r_flat


def test_reservation_price_short_inventory_shifts_up():
    r_short = reservation_price(mid=100.0, inventory=-5.0, gamma=0.1, sigma=0.05, time_left=10.0)
    r_flat = reservation_price(mid=100.0, inventory=0.0, gamma=0.1, sigma=0.05, time_left=10.0)
    assert r_short > r_flat


def test_reservation_price_shift_vanishes_as_time_runs_out():
    r_now = reservation_price(mid=100.0, inventory=5.0, gamma=0.1, sigma=0.05, time_left=1e-9)
    assert r_now == pytest.approx(100.0, abs=1e-6)


def test_optimal_spread_positive():
    s = optimal_spread(gamma=0.1, sigma=0.05, time_left=10.0, k=1.5)
    assert s > 0


def test_optimal_spread_widens_with_time_left():
    s_far = optimal_spread(gamma=0.1, sigma=0.05, time_left=100.0, k=1.5)
    s_near = optimal_spread(gamma=0.1, sigma=0.05, time_left=1.0, k=1.5)
    assert s_far > s_near


def test_optimal_spread_gamma_to_zero_limit_matches_2_over_k():
    # As gamma -> 0, (2/gamma) ln(1+gamma/k) -> 2/k (L'Hopital), and the
    # inventory term -> 0, so spread -> 2/k.
    k = 2.0
    small_gamma = 1e-6
    s = optimal_spread(gamma=small_gamma, sigma=0.05, time_left=0.0, k=k)
    assert s == pytest.approx(2.0 / k, rel=1e-3)


def test_optimal_quotes_bid_below_ask():
    bid, ask, r, spread = optimal_quotes(mid=100.0, inventory=0.0, gamma=0.1, sigma=0.05, time_left=10.0, k=1.5)
    assert bid < ask
    assert ask - bid == pytest.approx(spread)
    assert bid < r < ask


def test_fill_intensity_decays_with_distance():
    near = fill_intensity(delta=0.01, A=100.0, k=1.5)
    far = fill_intensity(delta=1.0, A=100.0, k=1.5)
    assert near > far
