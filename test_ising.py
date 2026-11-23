"""Tests for ising.py. Run with:  pytest"""

import itertools
import numpy as np
from ising import random_lattice, sweep, energy, simulate, TC_EXACT


def test_exact_critical_temperature():
    assert abs(TC_EXACT - 2.2692) < 1e-4


def test_ground_state_energy():
    # All spins aligned: each of the 2N bonds contributes -J
    L = 8
    spins = np.ones((L, L), dtype=int)
    assert energy(spins) == -2 * L * L


def test_low_temperature_stays_ordered():
    np.random.seed(1)
    spins = np.ones((16, 16), dtype=int)
    for _ in range(200):
        sweep(spins, 0.5)
    assert abs(spins.mean()) > 0.95


def test_high_temperature_is_disordered():
    np.random.seed(2)
    spins = np.ones((32, 32), dtype=int)
    for _ in range(300):
        sweep(spins, 6.0)
    assert abs(spins.mean()) < 0.2


def test_matches_exact_solution_4x4():
    # For a 4x4 lattice there are only 2^16 states, so we can compute the exact
    # average energy by brute force and compare it with the Monte Carlo result.
    L, T = 4, 2.5
    Z = 0.0
    E_sum = 0.0
    for state in itertools.product([-1, 1], repeat=L * L):
        spins = np.array(state).reshape(L, L)
        E = energy(spins)
        weight = np.exp(-E / T)
        Z += weight
        E_sum += E * weight
    exact = E_sum / Z / (L * L)

    np.random.seed(123)
    mc = simulate(L, T, n_sweeps=20000, n_equil=1000)["energy"]
    assert abs(mc - exact) < 0.02
