"""2D Ising model simulated with the Metropolis Monte Carlo algorithm.

Units: J = 1 and k_B = 1. Square lattice with periodic boundaries.
Run `python ising.py` to reproduce the figures and the numbers in the README.
"""

import numpy as np
import matplotlib.pyplot as plt

J = 1.0
TC_EXACT = 2 / np.log(1 + np.sqrt(2))  # Onsager's exact critical temperature, ~2.2692


def random_lattice(L):
    """L x L lattice of random spins (+1 or -1)."""
    return np.random.choice([-1, 1], size=(L, L))


def neighbour_sum(spins):
    """For every site, the sum of its 4 nearest neighbours (periodic boundaries)."""
    return (np.roll(spins, 1, axis=0) + np.roll(spins, -1, axis=0)
            + np.roll(spins, 1, axis=1) + np.roll(spins, -1, axis=1))


def sweep(spins, T):
    """One Metropolis sweep: we try to flip every spin once.

    To do it with NumPy (fast) instead of Python loops, we split the lattice
    like a chessboard. Spins of the same colour are never neighbours, so all
    of them can be updated at the same time without affecting each other.
    """
    L = spins.shape[0]
    i, j = np.indices((L, L))
    colour = (i + j) % 2

    for c in (0, 1):
        dE = 2 * J * spins * neighbour_sum(spins)   # energy change if we flip the spin
        # Metropolis rule: always accept if dE <= 0, otherwise accept with prob exp(-dE/T)
        accept = (dE <= 0) | (np.random.rand(L, L) < np.exp(-dE / T))
        flip = accept & (colour == c)
        spins[flip] *= -1


def energy(spins):
    """Total energy. Each bond is counted once (right and down neighbours)."""
    right = np.roll(spins, -1, axis=1)
    down = np.roll(spins, -1, axis=0)
    return -J * np.sum(spins * (right + down))


def simulate(L, T, n_sweeps=10000, n_equil=2000):
    """Simulate at temperature T and return the average observables (per spin)."""
    N = L * L
    spins = random_lattice(L)

    for _ in range(n_equil):          # let the system reach equilibrium first
        sweep(spins, T)

    E = np.zeros(n_sweeps)
    M = np.zeros(n_sweeps)
    for k in range(n_sweeps):         # then measure
        sweep(spins, T)
        E[k] = energy(spins)
        M[k] = np.sum(spins)

    return {
        "energy": E.mean() / N,
        "magnetisation": np.abs(M).mean() / N,
        "specific_heat": E.var() / (T**2 * N),
        "susceptibility": (np.mean(M**2) - np.abs(M).mean()**2) / (T * N),
        # Binder cumulant: curves for different L cross at Tc
        "binder": 1 - np.mean(M**4) / (3 * np.mean(M**2)**2),
    }


def crossing(T, curve_a, curve_b):
    """Temperature where two curves cross (linear interpolation between points)."""
    diff = curve_a - curve_b
    for k in range(len(T) - 1):
        if diff[k] * diff[k + 1] < 0:
            return T[k] - diff[k] * (T[k + 1] - T[k]) / (diff[k + 1] - diff[k])
    return None


def main():
    np.random.seed(0)
    sizes = [8, 16, 32, 48]
    temps = np.linspace(1.8, 2.8, 41)

    # results[L][quantity] is a list with one value per temperature
    results = {}
    for L in sizes:
        runs = [simulate(L, T) for T in temps]
        results[L] = {q: np.array([r[q] for r in runs]) for q in runs[0]}
        print(f"L = {L} done")

    # --- Estimate Tc from the crossings of the Binder cumulant ---
    crossings = []
    for La, Lb in zip(sizes[:-1], sizes[1:]):
        Tc = crossing(temps, results[La]["binder"], results[Lb]["binder"])
        crossings.append(Tc)
        print(f"Binder crossing L={La}/{Lb}: Tc = {Tc:.3f}")
    print(f"Mean estimate: {np.mean(crossings):.3f}   exact (Onsager): {TC_EXACT:.4f}")
    for L in sizes:
        print(f"Susceptibility peak, L={L}: T = {temps[np.argmax(results[L]['susceptibility'])]:.3f}")

    # --- Figures ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [("energy", "E / N"), ("magnetisation", "|M| / N"),
              ("specific_heat", "specific heat"), ("susceptibility", "susceptibility")]
    for ax, (q, label) in zip(axes.flat, panels):
        for L in sizes:
            ax.plot(temps, results[L][q], "o-", ms=3, label=f"L = {L}")
        ax.axvline(TC_EXACT, color="k", ls="--", lw=0.8)
        ax.set_xlabel("T")
        ax.set_ylabel(label)
    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig("imgs/magnitudes_ising.png", dpi=150)

    plt.figure(figsize=(6, 4.5))
    for L in sizes:
        plt.plot(temps, results[L]["binder"], "o-", ms=3, label=f"L = {L}")
    plt.axvline(TC_EXACT, color="k", ls="--", lw=0.8, label="exact Tc")
    plt.xlabel("T")
    plt.ylabel("Binder cumulant")
    plt.legend()
    plt.tight_layout()
    plt.savefig("imgs/binder_cumulant.png", dpi=150)

    # Snapshots of a 64 x 64 lattice below, at and above Tc
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, T in zip(axes, [1.5, TC_EXACT, 3.5]):
        spins = random_lattice(64)
        for _ in range(1500):
            sweep(spins, T)
        ax.imshow(spins, cmap="coolwarm")
        ax.set_title(f"T = {T:.2f}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("imgs/espines_ising.png", dpi=150)


if __name__ == "__main__":
    main()
