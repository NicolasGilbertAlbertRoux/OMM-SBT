#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================
# FINAL BEST PARAMETERS
# ===============================

SIZE = 128
N_STEPS = 180
SEED = 3
BETA = 8.75

CENTER_GAIN = 0.012
NODE_GAIN = 0.100
MATTER_GAIN = 0.098

OMEGA_BG = 0.22
BACKGROUND_GAIN = 0.035

OMEGA_LOCAL = 0.47
LOCAL_BEAT_GAIN = 0.085

FLUX_GAIN = 0.045
EDGE_PENALTY = 0.10

PHI_CLIP = 6.0
FLUX_CLIP = 1e6
NODE_CLIP = 1e8

OUT = Path("results/proto_atoms/proto_atom_final_best")
OUT.mkdir(parents=True, exist_ok=True)


# ===============================
# CORE FUNCTIONS
# ===============================

def laplacian(phi):
    return (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1)
        - 4 * phi
    )


def compute_flux(phi):
    gx = np.gradient(phi, axis=0)
    gy = np.gradient(phi, axis=1)

    gx = np.clip(gx, -FLUX_CLIP, FLUX_CLIP)
    gy = np.clip(gy, -FLUX_CLIP, FLUX_CLIP)

    flux_mag = np.sqrt(gx * gx + gy * gy)
    flux_mag = np.clip(flux_mag, 0.0, FLUX_CLIP)

    return gx, gy, flux_mag


def compute_divergence(gx, gy):
    div = np.gradient(gx, axis=0) + np.gradient(gy, axis=1)
    div = np.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)
    return div


def compute_node_field(phi, flux_mag):
    node = flux_mag * np.abs(phi)
    node = np.clip(node, 0.0, NODE_CLIP)
    node = np.nan_to_num(node, nan=0.0, posinf=NODE_CLIP, neginf=0.0)
    return node


def normalize(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    m = np.mean(x)
    s = np.std(x)
    if not np.isfinite(s) or s < 1e-12:
        return np.zeros_like(x)
    z = (x - m) / s
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z


def gaussian_center_mask(shape, sigma=None):
    if sigma is None:
        sigma = shape[0] / 6
    y, x = np.indices(shape)
    cy = (shape[0] - 1) / 2
    cx = (shape[1] - 1) / 2
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return np.exp(-r2 / (2.0 * sigma * sigma))


def edge_mask(shape, width=None):
    if width is None:
        width = shape[0] / 8
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


def compute_occupancy(node_field):
    m = float(np.max(node_field))
    if not np.isfinite(m) or m <= 1e-12:
        return np.zeros_like(node_field)
    occ = node_field / m
    occ = np.nan_to_num(occ, nan=0.0, posinf=0.0, neginf=0.0)
    return occ


def weighted_centroid(w):
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 1e-12:
        h, ww = w.shape
        return (ww - 1) / 2, (h - 1) / 2

    y, x = np.indices(w.shape)
    cx = float(np.sum(x * w) / total)
    cy = float(np.sum(y * w) / total)
    return cx, cy


# ===============================
# UPDATE
# ===============================

def update(phi, phi0, phase0, step):
    lap = laplacian(phi)
    gx, gy, flux_mag = compute_flux(phi)
    div_flux = compute_divergence(gx, gy)
    node_field = compute_node_field(phi, flux_mag)

    background = BACKGROUND_GAIN * np.sin(OMEGA_BG * step) * phi0
    local_beating = LOCAL_BEAT_GAIN * np.sin(OMEGA_LOCAL * step + phase0) * phi

    matter_feedback = MATTER_GAIN * normalize(-div_flux)
    node_feedback = NODE_GAIN * normalize(node_field)
    flux_feedback = FLUX_GAIN * normalize(flux_mag)

    center = gaussian_center_mask(phi.shape)
    edges = edge_mask(phi.shape)

    center_capture = CENTER_GAIN * center * normalize(node_field + np.maximum(-div_flux, 0.0))
    anti_edge = -EDGE_PENALTY * edges * normalize(node_field)

    stabilizer = -0.015 * phi * np.abs(phi)
    stabilizer = np.nan_to_num(stabilizer, nan=0.0, posinf=0.0, neginf=0.0)

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
        + stabilizer
    )

    phi_next = np.nan_to_num(phi_next, nan=0.0, posinf=0.0, neginf=0.0)
    phi_next = np.clip(phi_next, -PHI_CLIP, PHI_CLIP)

    return phi_next, flux_mag, node_field


# ===============================
# MAIN
# ===============================

def main():
    rng = np.random.default_rng(SEED)
    phi0 = rng.uniform(-3, 3, (SIZE, SIZE))
    phi = phi0.copy()
    phase0 = make_local_phase_field(phi.shape, rng)

    std_hist = []
    flux_hist = []
    node_hist = []

    print("\n=== PROTO ATOM FINAL BEST RENDER ===")
    print(f"[DEBUG] SIZE={SIZE}, STEPS={N_STEPS}")
    print(
        f"[DEBUG] CENTER_GAIN={CENTER_GAIN}, "
        f"NODE_GAIN={NODE_GAIN}, MATTER_GAIN={MATTER_GAIN}"
    )

    for step in range(N_STEPS):
        phi, flux_mag, node_field = update(phi, phi0, phase0, step)

        std_hist.append(float(np.std(phi)))
        flux_hist.append(float(np.max(flux_mag)))
        node_hist.append(float(np.max(node_field)))

        if step % 20 == 0:
            print(
                f"step={step} "
                f"std={np.std(phi):.4f} "
                f"flux={np.max(flux_mag):.4f} "
                f"node={np.max(node_field):.4f}"
            )

    occ = compute_occupancy(node_field)
    cx, cy = weighted_centroid(occ)

    # ===============================
    # FINAL FIGURE
    # ===============================

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    axes[0].imshow(phi, cmap="viridis")
    axes[0].set_title("Field φ")

    axes[1].imshow(node_field, cmap="inferno")
    axes[1].set_title("Node field")

    axes[2].imshow(occ, cmap="hot")
    axes[2].scatter(cx, cy, c="cyan", s=50)
    axes[2].set_title("Occupancy + centroid")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    final_img = OUT / "proto_atom_final_best.png"
    plt.savefig(final_img, dpi=600)
    plt.close()

    # diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(std_hist)
    axes[0].set_title("phi std")

    axes[1].plot(flux_hist)
    axes[1].set_title("flux max")

    axes[2].plot(node_hist)
    axes[2].set_title("node max")

    plt.tight_layout()
    diag_img = OUT / "proto_atom_final_best_diagnostics.png"
    plt.savefig(diag_img, dpi=400)
    plt.close()

    np.save(OUT / "phi_final.npy", phi)
    np.save(OUT / "node_field_final.npy", node_field)
    np.save(OUT / "occupancy_final.npy", occ)

    print(f"[OK] wrote {final_img}")
    print(f"[OK] wrote {diag_img}")
    print("[DONE] proto atom final best render complete")


if __name__ == "__main__":
    main()