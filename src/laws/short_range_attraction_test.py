#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np


IN_PATH = Path("results/final_states/seed_inputs/phase_separation_final.npy")
OUT_DIR = Path("results/laws/short_range_attraction")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STEPS = 30
DT = 0.06

A_SHORT = 1.0   # attraction
A_LONG = 0.6    # repulsion


def laplacian(x):
    out = np.zeros_like(x)
    for ax in range(x.ndim):
        out += np.roll(x, 1, axis=ax) + np.roll(x, -1, axis=ax) - 2 * x
    return out


def smooth(x, n=2):
    y = x.copy()
    for _ in range(n):
        y = y + 0.25 * laplacian(y)
    return y


def evolve(phi):
    # local attraction
    short = smooth(phi, n=1)

    # widespread aversion
    long = smooth(phi, n=4)

    interaction = A_SHORT * short - A_LONG * long

    return phi + DT * interaction


def main():
    arr = np.load(IN_PATH).astype(np.float64)

    print("\n=== SHORT-RANGE ATTRACTION TEST ===")

    for step in range(N_STEPS + 1):
        print(f"step={step} mean={arr.mean():.4f} std={arr.std():.4f}")

        if step < N_STEPS:
            arr = evolve(arr)
            arr = np.clip(arr, -3, 3)

    np.save(OUT_DIR / "short_range_final.npy", arr)

    print("\n[OK] saved short_range_final.npy")
    print("[DONE] test complete")


if __name__ == "__main__":
    main()