#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results/magnetism/proto_atom_magnetic_sector_scan")
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 180
N_STEPS = 260
DT = 0.08

WAVE_GAIN = 0.24
DAMP = 0.996

# phase-shifted oscillating sources
SRC1 = (SIZE // 2 - 25, SIZE // 2)
SRC2 = (SIZE // 2 + 25, SIZE // 2)

AMP = 1.0
OMEGA = 0.12

# ------------------------------------------------------------

def laplacian(phi):
    return (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0)
        + np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1)
        - 4.0 * phi
    )

def gradient(phi):
    gy, gx = np.gradient(phi)
    return gx, gy

def divergence(gx, gy):
    return np.gradient(gx, axis=1) + np.gradient(gy, axis=0)

def curl_2d(gx, gy):
    # curl z en 2D
    return np.gradient(gy, axis=1) - np.gradient(gx, axis=0)

# ------------------------------------------------------------

def main():
    print("\n=== MAGNETIC SECTOR SCAN ===")

    phi = np.zeros((SIZE, SIZE))
    psi = np.zeros_like(phi)

    curl_energy = []
    div_energy = []

    snapshots = {}

    for step in range(N_STEPS):

        source = np.zeros_like(phi)

        # oscillations en quadrature → clé du transverse
        source[SRC1] += AMP * np.sin(OMEGA * step)
        source[SRC2] += AMP * np.cos(OMEGA * step)

        lap = laplacian(phi)

        psi = DAMP * psi + DT * (WAVE_GAIN * lap + source)
        phi = phi + DT * psi

        # champ effectif
        Ex, Ey = gradient(phi)

        divE = divergence(Ex, Ey)
        curlE = curl_2d(Ex, Ey)

        curl_energy.append(float(np.mean(np.abs(curlE))))
        div_energy.append(float(np.mean(np.abs(divE))))

        if step in [0, 40, 80, 140, 200, N_STEPS - 1]:
            snapshots[step] = {
                "phi": phi.copy(),
                "curl": curlE.copy(),
                "div": divE.copy()
            }

        if step % 40 == 0:
            print(
                f"step={step} "
                f"curl_mean={np.mean(np.abs(curlE)):.6e} "
                f"div_mean={np.mean(np.abs(divE)):.6e}"
            )

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    ratio = np.array(curl_energy) / (np.array(div_energy) + 1e-12)

    print("\n=== TRANSVERSE RATIO ===")
    print(f"mean curl/div ≈ {np.mean(ratio):.6f}")

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    plt.figure(figsize=(10, 4))
    plt.plot(curl_energy, label="curl energy")
    plt.plot(div_energy, label="div energy")
    plt.legend()
    plt.title("Curl vs Divergence energy")
    plt.tight_layout()
    plt.savefig(OUT / "curl_vs_div_energy.png", dpi=260)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ratio)
    plt.title("Transverse ratio (curl/div)")
    plt.tight_layout()
    plt.savefig(OUT / "transverse_ratio.png", dpi=260)
    plt.close()

    # --------------------------------------------------------
    # SNAPSHOTS
    # --------------------------------------------------------

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for ax, step in zip(axes, snapshots.keys()):
        ax.imshow(snapshots[step]["curl"], cmap="coolwarm")
        ax.set_title(f"curl step={step}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUT / "curl_snapshots.png", dpi=260)
    plt.close()

    print("[DONE] magnetic sector scan complete")


if __name__ == "__main__":
    main()