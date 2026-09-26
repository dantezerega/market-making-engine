"""
Avellaneda-Stoikov (2008) closed-form market making model.

=========================================================================
DERIVATION (HJB -> closed form)
=========================================================================

Setup
-----
- Mid price follows arithmetic Brownian motion:  dS_t = sigma dW_t
  (sigma constant — this is the model's key simplifying assumption,
  see README limitations section).
- A single market maker controls two point processes: the arrival of
  a "buy fill" (someone hits her bid) and a "sell fill" (someone lifts
  her ask). She chooses how far to place her bid/ask from the mid,
  delta_b and delta_a. Placing quotes further from the mid lowers the
  probability/intensity of getting filled but earns more spread per fill.
- Fill intensity is modelled as a Poisson process whose intensity decays
  exponentially in distance from the mid:
        lambda(delta) = A * exp(-k * delta)
  This is an empirical regularity from LOB data (Avellaneda-Stoikov cite
  it as consistent with observed order flow) — A is the arrival-rate
  scale (intensity of flow at zero distance) and k is the decay rate
  (how fast intensity dies off as you quote further away — a "thin"
  vs "deep" book parameter).
- The maker has CARA (exponential) utility over terminal wealth:
        u(x) = -exp(-gamma * x)
  gamma is the risk-aversion coefficient.
- State variables: cash x_t, inventory q_t (shares held), mid price s_t.
  Wealth at terminal time T (liquidating inventory at the prevailing
  mid) is  X_T = x_T + q_T * s_T.

Value function and HJB equation
--------------------------------
Define the maker's value function
    V(s, x, q, t) = max_{delta_b, delta_a}  E[ -exp(-gamma (x_T + q_T s_T)) | s_t=s, x_t=x, q_t=q ]

The dynamic programming principle over an infinitesimal interval, together
with Ito's lemma for the diffusion term and the jump terms for each fill
process, gives the Hamilton-Jacobi-Bellman equation:

    V_t + (1/2) sigma^2 V_ss
        + max_{delta_b} { lambda(delta_b) [ V(s, x - s + delta_b, q+1, t) - V(s,x,q,t) ] }
        + max_{delta_a} { lambda(delta_a) [ V(s, x + s + delta_a, q-1, t) - V(s,x,q,t) ] }
        = 0,          V(s,x,q,T) = -exp(-gamma(x + q s))

Reading the jump terms: if the bid gets hit, cash falls by (s - delta_b)
(you buy at s - delta_b) and inventory rises by 1; symmetric story for the ask.

Ansatz (exponential CARA structure is preserved by the dynamics)
------------------------------------------------------------------
Guess a separable form:
    V(s, x, q, t) = -exp(-gamma x) exp(-gamma q s) exp(-gamma theta(q, t))

Substituting into the HJB equation and dividing through by the common
exponential factor turns the PDE for V into a system of coupled ODEs for
theta(q, t), one per inventory level q. Avellaneda-Stoikov show that a
further second-order Taylor expansion of theta(q,t) around q=0 in the
inventory variable — valid because in practice |q| stays in a modest
range around zero for a well-behaved maker — decouples the ODEs and
yields, after collecting terms:

    theta(q, t) = -0.5 * (T - t) * sigma^2 * q^2    (leading order term)

which is exactly the piece that reduces the value function to a function
of a single quadratic penalty on inventory risk. Plugging this back into
the definition of the reservation (indifference) price — the price s_r
at which the maker is indifferent between holding q and q+/-1 shares
given time remaining T-t — gives:

    RESERVATION PRICE:
        r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

Interpretation: if you're long (q>0) your indifference price is BELOW
the mid (you'd rather sell than buy more — you already have inventory
risk), and vice versa if short. The size of the shift scales with how
much inventory you're carrying (q), how averse you are to risk (gamma),
how volatile the underlying is (sigma^2), and how much time is left for
that inventory risk to hurt you (T-t) — as t -> T the shift vanishes
because there's no time left for adverse price moves to matter (you're
about to mark to market and be done).

Optimal spread
---------------
Taking first-order conditions on the two maximizations in the HJB
equation (optimizing delta_b and delta_a given lambda(delta)=A e^{-k delta}
and the exponential value function), and using the same q~0 expansion,
Avellaneda-Stoikov derive that the TOTAL optimal spread (delta_a + delta_b,
i.e., ask distance from reservation price plus bid distance from
reservation price) is:

    OPTIMAL SPREAD:
        delta = gamma * sigma^2 * (T - t)  +  (2 / gamma) * ln(1 + gamma / k)

The two pieces have distinct economic meaning:
  - gamma * sigma^2 * (T-t): the inventory-risk premium — same magnitude
    as the reservation-price shift, it widens the total quoted spread
    to compensate for the risk of being adversely selected while holding
    a position for the remaining horizon.
  - (2/gamma) ln(1 + gamma/k): the pure "monopolistic" microstructure
    spread you'd charge even with zero risk aversion pressure from
    inventory — it depends only on gamma and k (how liquid/thin the
    book is). As gamma -> 0 this term -> 2/k (independent of gamma, using
    L'Hopital / limit), matching the risk-neutral optimal spread. As k
    grows (a "thick", highly liquid book where quoting slightly further
    out kills your fill probability fast) this term shrinks.

Quotes are placed symmetrically around the reservation price:
    bid = r(s,q,t) - delta/2
    ask = r(s,q,t) + delta/2

Equivalently (and how this repo implements it):
    bid = s - q*gamma*sigma^2*(T-t) - delta/2
    ask = s - q*gamma*sigma^2*(T-t) + delta/2

Note the inventory skew shows up TWICE in the raw bid/ask relative to
the mid: once through the reservation-price shift and, because the
total spread `delta` itself grows with the same gamma*sigma^2*(T-t)
term, again through spread widening. Both effects push a long maker to
quote a more aggressive (tighter to mid, or crossing) ask and a more
passive (further from mid) bid — exactly the mean-reversion pressure
requested in the project spec.

=========================================================================
References
=========================================================================
Avellaneda, M. and Stoikov, S. (2008), "High-frequency trading in a
limit order book", Quantitative Finance, 8(3), 217-224.
Gueant, O., Lehalle, C-A., Fernandez-Tapia, J. (2013), "Dealing with
the inventory risk: a solution to the market making problem", Math.
Financial Econ. — for the exact (non-asymptotic) solution.
"""
from __future__ import annotations

import numpy as np


def reservation_price(mid: float, inventory: float, gamma: float, sigma: float, time_left: float) -> float:
    """r(s,q,t) = s - q * gamma * sigma^2 * (T - t)

    mid: current mid/fundamental price s
    inventory: current inventory q (positive = long)
    gamma: risk aversion coefficient
    sigma: volatility of the mid-price process (same units/timescale as time_left)
    time_left: T - t, remaining time in the session
    """
    return mid - inventory * gamma * (sigma ** 2) * time_left


def optimal_spread(gamma: float, sigma: float, time_left: float, k: float) -> float:
    """delta = gamma*sigma^2*(T-t) + (2/gamma) * ln(1 + gamma/k)

    Total (bid+ask) optimal spread around the reservation price.
    """
    inventory_term = gamma * (sigma ** 2) * time_left
    microstructure_term = (2.0 / gamma) * np.log(1.0 + gamma / k)
    return inventory_term + microstructure_term


def optimal_quotes(
    mid: float,
    inventory: float,
    gamma: float,
    sigma: float,
    time_left: float,
    k: float,
) -> tuple[float, float, float, float]:
    """Return (bid, ask, reservation_price, spread).

    bid = r - spread/2 ; ask = r + spread/2
    """
    r = reservation_price(mid, inventory, gamma, sigma, time_left)
    spread = optimal_spread(gamma, sigma, time_left, k)
    return r - spread / 2.0, r + spread / 2.0, r, spread


def fill_intensity(delta: float, A: float, k: float) -> float:
    """lambda(delta) = A * exp(-k * delta) — the assumed order-arrival
    intensity as a function of quoted distance from the mid. Used for
    calibration diagnostics and sensitivity analysis, not directly by
    the closed-form spread formula (which depends on k only, in the
    standard asymptotic/symmetric approximation used above)."""
    return A * np.exp(-k * delta)
