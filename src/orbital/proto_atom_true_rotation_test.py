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

# true rotation due to tangential advection
ROTATION_GAIN = 0.015

OUT = Path("results/orbital/proto_atom_true_rotation")
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


def compute_divergence(vx, vy):
    return np.gradient(vx, axis=1) + np.gradient(vy, axis=0)


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
    d_left = x
    d_right = (w - 1) - x
    d_top = y
    d_bottom = (h - 1) - y
    d_edge = np.minimum(np.minimum(d_left, d_right), np.minimum(d_top, d_bottom))
    return np.exp(-d_edge / width)


def make_local_phase_field(shape, rng):
    return rng.uniform(0.0, 2.0 * np.pi, size=shape)


def update(phi, phi0, phase0, step):
    lap = laplacian(phi)

    dphi_dx, dphi_dy, flux_mag = compute_flux(phi)
    node_field = compute_node_field(phi, flux_mag)

    # divergence of the reconstructed scalar “flow”
    div_flux = np.gradient(dphi_dx, axis=1) + np.gradient(dphi_dy, axis=0)

    # background oscillation
    background = BACKGROUND_GAIN * np.sin(OMEGA_BG * step) * phi0

    # local internal oscillation
    local_beating = LOCAL_BEAT_GAIN * np.sin(OMEGA_LOCAL * step + phase0) * phi

    matter_feedback = MATTER_GAIN * normalize(-div_flux)
    node_feedback = NODE_GAIN * normalize(node_field)
    flux_feedback = FLUX_GAIN * normalize(flux_mag)

    center = gaussian_center_mask(phi.shape, sigma=6.0)
    edges = edge_mask(phi.shape, width=4.0)

    center_capture = CENTER_GAIN * center * normalize(node_field + np.maximum(-div_flux, 0.0))
    anti_edge = -EDGE_PENALTY * edges * normalize(node_field)

    # =========================
    # TRUE ROTATION
    # =========================
    y, x = np.indices(phi.shape)
    cy = (phi.shape[0] - 1) / 2
    cx = (phi.shape[1] - 1) / 2

    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2) + 1e-8

    # tangential velocity around the center
    vx = -dy / r
    vy = dx / r

    # structure location to force
    rot_mask = np.clip(center, 0.0, 1.0)

    # rotational advection term
    advection = -(vx * dphi_dx + vy * dphi_dy)
    rotation_feedback = ROTATION_GAIN * rot_mask * np.tanh(advection)

    phi_next = (
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
    rng = np.random.default_rng(SEED)
    phi0 = rng.uniform(-3, 3, (SIZE, SIZE))
    phi = phi0.copy()
    phase0 = make_local_phase_field(phi.shape, rng)

    std_hist = []
    flux_hist = []
    node_hist = []

    print("\n=== PROTO ATOM TRUE ROTATION TEST ===")
    print("[DEBUG] TRUE ROTATION ACTIVE =", ROTATION_GAIN)

    for step in range(N_STEPS):
        phi, flux_mag, node_field, div_flux = update(phi, phi0, phase0, step)

        std_hist.append(float(np.std(phi)))
        flux_hist.append(float(np.max(flux_mag)))
        node_hist.append(float(np.max(node_field)))

        if step % 10 == 0:
            print(
                f"step={step} "
                f"std={np.std(phi):.4f} "
                f"flux={np.max(flux_mag):.4f} "
                f"node={np.max(node_field):.4f}"
            )

    occ = compute_occupancy(node_field)
    cx, cy = compute_centroid(occ)

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
    plt.savefig(OUT / "proto_atom_true_rotation.png", dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    axes[0].plot(std_hist)
    axes[0].set_title("phi std")

    axes[1].plot(flux_hist)
    axes[1].set_title("flux max")

    axes[2].plot(node_hist)
    axes[2].set_title("node max")

    plt.tight_layout()
    plt.savefig(OUT / "proto_atom_true_rotation_diagnostics.png", dpi=220)
    plt.close()

    print("[DONE] proto atom true rotation test complete")


if __name__ == "__main__":
    main()