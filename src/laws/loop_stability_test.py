#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np


IN_PATH = Path("results/final_states/laws/short_range_final.npy")
OUT_DIR = Path("results/laws/loop_stability")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STEPS = 30
DT = 0.04


def laplacian(x):
    out = np.zeros_like(x)
    for ax in range(x.ndim):
        out += np.roll(x, 1, axis=ax) + np.roll(x, -1, axis=ax) - 2 * x
    return out


def evolve(phi):
    return phi + DT * laplacian(phi)


def main():
    arr = np.load(IN_PATH).astype(np.float64)

    print("\n=== LOOP STABILITY TEST ===")

    for step in range(N_STEPS + 1):
        print(f"step={step} std={arr.std():.4f}")

        if step < N_STEPS:
            arr = evolve(arr)

    np.save(OUT_DIR / "loop_stability_final.npy", arr)

    print("\n[OK] saved loop_stability_final.npy")
    print("[DONE] loop stability test complete")


if __name__ == "__main__":
    main()