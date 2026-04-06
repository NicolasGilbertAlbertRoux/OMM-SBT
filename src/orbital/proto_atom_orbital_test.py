#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SIZE = 32
N_STEPS = 120

BETA = 8.75
SEED = 3

# background oscillation
OMEGA = 0.35
BACKGROUND_GAIN = 0.06

# node feedback
NODE_GAIN = 0.10

# material accumulation
MATTER_GAIN = 0.05

# low orbital anisotropy
ORBITAL_GAIN = 0.018
ORBITAL_OMEGA = 0.11

OUT = Path("results/orbital/proto_atom_orbital")
OUT.mkdir(parents=True, exist_ok=True)


def laplacian(phi):
    return (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1)
        - 4 * phi
    )


def compute_flux(phi):
    gx = np.gradient(phi, axis=0)
    gy = np.gradient(phi, axis=1)
    flux_mag = np.sqrt(gx**2 + gy**2)
    return gx, gy, flux_mag


def compute_node_field(phi, flux_mag):
    return flux_mag * np.abs(phi)


def compute_divergence(gx, gy):
    return np.gradient(gx, axis=0) + np.gradient(gy, axis=1)


def normalize(x):
    s = np.std(x)
    if s < 1e-12:
        return x * 0.0
    return (x - np.mean(x)) / s


def update(phi, phi0, step):
    lap = laplacian(phi)
    gx, gy, flux_mag = compute_flux(phi)
    node_field = compute_node_field(phi, flux_mag)
    div_flux = compute_divergence(gx, gy)

    # 1) overall ground beat
    background = BACKGROUND_GAIN * np.sin(OMEGA * step) * phi0

    # 2) node feedback
    node_feedback = NODE_GAIN * normalize(node_field)

    # 3) material accumulation
    matter = MATTER_GAIN * normalize(-div_flux)

    # 4) very soft radial seal
    y, x = np.indices(phi.shape)
    cx, cy = phi.shape[1] / 2, phi.shape[0] / 2
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-8
    confinement = -0.00025 * r * phi

    # 5) weak, rotating orbital anisotropy
    theta = np.arctan2(dy, dx)
    orbital_pattern = np.cos(2.0 * theta - ORBITAL_OMEGA * step)
    orbital = ORBITAL_GAIN * orbital_pattern * normalize(node_field)

    phi_next = (
        phi
        + 0.10 * lap
        - 0.010 * BETA * phi
        + background
        + node_feedback
        + matter
        + confinement
        + orbital
    )

    return phi_next, flux_mag, node_field, div_flux


def compute_occupancy(node_field):
    return node_field / (node_field.max() + 1e-8)


def compute_centroid(occ):
    y, x = np.indices(occ.shape)
    total = occ.sum() + 1e-8
    cx = (x * occ).sum() / total
    cy = (y * occ).sum() / total
    return cx, cy


def main():
    np.random.seed(SEED)
    phi0 = np.random.uniform(-3, 3, (SIZE, SIZE))
    phi = phi0.copy()

    std_hist = []
    node_hist = []
    flux_hist = []

    print("\n=== PROTO ATOM ORBITAL TEST ===")

    for step in range(N_STEPS):
        phi, flux_mag, node_field, div_flux = update(phi, phi0, step)

        std_hist.append(float(np.std(phi)))
        node_hist.append(float(np.max(node_field)))
        flux_hist.append(float(np.max(flux_mag)))

        if step % 10 == 0:
            print(
                f"step={step} "
                f"std={np.std(phi):.4f} "
                f"flux={np.max(flux_mag):.4f} "
                f"node={np.max(node_field):.4f}"
            )

    occ = compute_occupancy(node_field)
    cx, cy = compute_centroid(occ)

    # main figure
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].imshow(phi, cmap="viridis")
    axes[0].set_title("phi")

    axes[1].imshow(node_field, cmap="inferno")
    axes[1].set_title("node field")

    axes[2].imshow(occ, cmap="hot")
    axes[2].scatter(cx, cy, c="cyan", s=60)
    axes[2].set_title("occupancy + center")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_img = OUT / "proto_atom_orbital.png"
    plt.savefig(out_img, dpi=220)
    plt.close()

    # diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    axes[0].plot(std_hist)
    axes[0].set_title("phi std")

    axes[1].plot(flux_hist)
    axes[1].set_title("flux max")

    axes[2].plot(node_hist)
    axes[2].set_title("node max")

    plt.tight_layout()
    out_diag = OUT / "proto_atom_orbital_diagnostics.png"
    plt.savefig(out_diag, dpi=220)
    plt.close()

    np.save(OUT / "phi_final.npy", phi)
    np.save(OUT / "node_field_final.npy", node_field)
    np.save(OUT / "std_history.npy", np.array(std_hist))
    np.save(OUT / "flux_max_history.npy", np.array(flux_hist))
    np.save(OUT / "node_max_history.npy", np.array(node_hist))

    print(f"[OK] saved {out_img}")
    print(f"[OK] saved {out_diag}")
    print("[DONE] proto atom orbital test complete")


if __name__ == "__main__":
    main()