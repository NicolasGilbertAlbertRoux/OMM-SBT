#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SIZE = 32
N_STEPS = 120
BETA = 8.75
SEED = 3

OMEGA_BG = 0.22
BACKGROUND_GAIN = 0.035

OMEGA_LOCAL = 0.47
LOCAL_BEAT_GAIN = 0.085

NODE_GAIN = 0.085
MATTER_GAIN = 0.095
FLUX_GAIN = 0.045

CENTER_GAIN = 0.012
EDGE_PENALTY = 0.10

ROTATION_VALUES = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025]

OUT = Path("results/orbital/proto_atom_rotation_scan")
OUT.mkdir(parents=True, exist_ok=True)


def laplacian(phi):
    return (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1)
        - 4 * phi
    )


def compute_flux(phi):
    dphi_dy = np.gradient(phi, axis=0)
    dphi_dx = np.gradient(phi, axis=1)
    flux_mag = np.sqrt(dphi_dx**2 + dphi_dy**2)
    return dphi_dx, dphi_dy, flux_mag


def compute_node_field(phi, flux_mag):
    return flux_mag * np.abs(phi)


def normalize(x):
    s = np.std(x)
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s


def gaussian_center_mask(shape, sigma=6.0):
    y, x = np.indices(shape)
    cy = (shape[0] - 1) / 2
    cx = (shape[1] - 1) / 2
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return np.exp(-r2 / (2.0 * sigma * sigma))


def edge_mask(shape, width=4.0):
    y, x = np.indices(shape)
    h, w = shape
    d_edge = np.minimum.reduce([x, y, w - 1 - x, h - 1 - y])
    return np.exp(-d_edge / width)


def make_local_phase_field(shape, rng):
    return rng.uniform(0.0, 2.0 * np.pi, size=shape)


def run_simulation(rotation_gain):

    rng = np.random.default_rng(SEED)
    phi0 = rng.uniform(-3, 3, (SIZE, SIZE))
    phi = phi0.copy()
    phase0 = make_local_phase_field(phi.shape, rng)

    for step in range(N_STEPS):

        lap = laplacian(phi)
        dphi_dx, dphi_dy, flux_mag = compute_flux(phi)
        node_field = compute_node_field(phi, flux_mag)

        div_flux = np.gradient(dphi_dx, axis=1) + np.gradient(dphi_dy, axis=0)

        background = BACKGROUND_GAIN * np.sin(OMEGA_BG * step) * phi0
        local_beating = LOCAL_BEAT_GAIN * np.sin(OMEGA_LOCAL * step + phase0) * phi

        matter_feedback = MATTER_GAIN * normalize(-div_flux)
        node_feedback = NODE_GAIN * normalize(node_field)
        flux_feedback = FLUX_GAIN * normalize(flux_mag)

        center = gaussian_center_mask(phi.shape)
        edges = edge_mask(phi.shape)

        center_capture = CENTER_GAIN * center * normalize(node_field + np.maximum(-div_flux, 0.0))
        anti_edge = -EDGE_PENALTY * edges * normalize(node_field)

        # rotation réelle (advection)
        y, x = np.indices(phi.shape)
        cy = (phi.shape[0] - 1) / 2
        cx = (phi.shape[1] - 1) / 2

        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx**2 + dy**2) + 1e-8

        vx = -dy / r
        vy = dx / r

        rot_mask = center
        advection = -(vx * dphi_dx + vy * dphi_dy)

        rotation_feedback = rotation_gain * rot_mask * np.tanh(advection)

        phi = (
            phi
            + 0.085 * lap
            - 0.008 * BETA * phi
            + background
            + local_beating
            + matter_feedback
            + node_feedback
            + flux_feedback
            + center_capture
            + anti_edge
            + rotation_feedback
        )

    return phi, node_field


def compute_metrics(phi, node_field):
    occ = node_field / (node_field.max() + 1e-8)

    y, x = np.indices(occ.shape)
    total = occ.sum() + 1e-8

    cx = (x * occ).sum() / total
    cy = (y * occ).sum() / total

    dx = x - cx
    dy = y - cy

    r = np.sqrt(dx**2 + dy**2)

    radial_mean = (r * occ).sum() / total
    radial_std = np.sqrt(((r - radial_mean) ** 2 * occ).sum() / total)

    return cx, cy, radial_std


def main():

    print("\n=== ROTATION SCAN ===")

    fig, axes = plt.subplots(2, len(ROTATION_VALUES), figsize=(14, 5))

    for i, rot in enumerate(ROTATION_VALUES):

        phi, node_field = run_simulation(rot)
        cx, cy, spread = compute_metrics(phi, node_field)

        print(f"rot={rot:.3f} | spread={spread:.4f}")

        axes[0, i].imshow(phi, cmap="viridis")
        axes[0, i].set_title(f"rot={rot:.3f}")
        axes[0, i].axis("off")

        occ = node_field / (node_field.max() + 1e-8)

        axes[1, i].imshow(occ, cmap="hot")
        axes[1, i].scatter(cx, cy, c="cyan", s=40)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "rotation_scan.png", dpi=220)
    plt.close()

    print("[DONE] rotation scan complete")


if __name__ == "__main__":
    main()