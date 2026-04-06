#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("results/proto_atoms/proto_atom_pair_interaction")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

SIZE = 160
N_STEPS = 220

BETA = 9.50
CENTER_GAIN = 0.014
NODE_GAIN = 0.080
MATTER_GAIN = 0.107
FLUX_GAIN = 0.040
EDGE_PENALTY = 0.12

OMEGA_BG = 0.22
BACKGROUND_GAIN = 0.035
OMEGA_LOCAL = 0.47
LOCAL_BEAT_GAIN = 0.085

PHI_CLIP = 6.0
FLUX_CLIP = 1e6
NODE_CLIP = 1e8

SEED_A = 8
SEED_B = 11

SHIFT_A = (-18, -14)
SHIFT_B = (18, 14)

MIX_COEFF = 0.90


# ============================================================
# CORE
# ============================================================

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
    return np.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)


def compute_node_field(phi, flux_mag):
    node = flux_mag * np.abs(phi)
    node = np.clip(node, 0.0, NODE_CLIP)
    return np.nan_to_num(node, nan=0.0, posinf=NODE_CLIP, neginf=0.0)


def normalize(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    m = float(np.mean(x))
    s = float(np.std(x))
    if (not np.isfinite(s)) or s < 1e-12:
        return np.zeros_like(x)
    z = (x - m) / s
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


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
    if (not np.isfinite(m)) or m <= 1e-12:
        return np.zeros_like(node_field)
    occ = node_field / m
    return np.nan_to_num(occ, nan=0.0, posinf=0.0, neginf=0.0)


def weighted_centroid(w):
    total = float(np.sum(w))
    if (not np.isfinite(total)) or total <= 1e-12:
        h, ww = w.shape
        return (ww - 1) / 2, (h - 1) / 2
    y, x = np.indices(w.shape)
    cx = float(np.sum(x * w) / total)
    cy = float(np.sum(y * w) / total)
    return cx, cy


# ============================================================
# SINGLE ATOM BUILDER
# ============================================================

def run_single_atom(seed, n_steps=180, size=SIZE):
    rng = np.random.default_rng(seed)

    phi0 = rng.uniform(-3, 3, (size, size))
    phi = phi0.copy()
    phase0 = make_local_phase_field(phi.shape, rng)

    for step in range(n_steps):
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
            + stabilizer
        )

        phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        phi = np.clip(phi, -PHI_CLIP, PHI_CLIP)

    return phi


def shifted(arr, dy, dx):
    return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)


# ============================================================
# PAIR DYNAMICS
# ============================================================

def update_pair(phi, phi0, phase0, step):
    lap = laplacian(phi)
    gx, gy, flux_mag = compute_flux(phi)
    div_flux = compute_divergence(gx, gy)
    node_field = compute_node_field(phi, flux_mag)

    background = BACKGROUND_GAIN * np.sin(OMEGA_BG * step) * phi0
    local_beating = LOCAL_BEAT_GAIN * np.sin(OMEGA_LOCAL * step + phase0) * phi

    matter_feedback = MATTER_GAIN * normalize(-div_flux)
    node_feedback = NODE_GAIN * normalize(node_field)
    flux_feedback = FLUX_GAIN * normalize(flux_mag)

    center = gaussian_center_mask(phi.shape, sigma=phi.shape[0] / 4)
    edges = edge_mask(phi.shape)

    center_capture = 0.5 * CENTER_GAIN * center * normalize(node_field + np.maximum(-div_flux, 0.0))
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


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n=== PROTO ATOM PAIR INTERACTION ===")
    print(
        f"[DEBUG] β={BETA:.2f}, C={CENTER_GAIN:.3f}, N={NODE_GAIN:.3f}, "
        f"M={MATTER_GAIN:.3f}, F={FLUX_GAIN:.3f}, E={EDGE_PENALTY:.2f}"
    )
    print(f"[DEBUG] seeds = ({SEED_A}, {SEED_B})")

    phi_a = run_single_atom(SEED_A)
    phi_b = run_single_atom(SEED_B)

    phi_a = shifted(phi_a, SHIFT_A[0], SHIFT_A[1])
    phi_b = shifted(phi_b, SHIFT_B[0], SHIFT_B[1])

    phi0 = MIX_COEFF * phi_a + MIX_COEFF * phi_b
    phi = phi0.copy()

    rng = np.random.default_rng(999)
    phase0 = make_local_phase_field(phi.shape, rng)

    std_hist = []
    flux_hist = []
    node_hist = []
    dist_hist = []

    for step in range(N_STEPS):
        phi, flux_mag, node_field = update_pair(phi, phi0, phase0, step)

        occ = compute_occupancy(node_field)
        cx, cy = weighted_centroid(occ)

        std_hist.append(float(np.std(phi)))
        flux_hist.append(float(np.max(flux_mag)))
        node_hist.append(float(np.max(node_field)))
        dist_hist.append(float(np.sqrt((cx - (SIZE - 1) / 2) ** 2 + (cy - (SIZE - 1) / 2) ** 2)))

        if step % 20 == 0:
            print(
                f"step={step} "
                f"std={np.std(phi):.4f} "
                f"flux={np.max(flux_mag):.4f} "
                f"node={np.max(node_field):.4f} "
                f"center_drift={dist_hist[-1]:.4f}"
            )

    occ = compute_occupancy(node_field)
    cx, cy = weighted_centroid(occ)

    # figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(phi, cmap="viridis")
    axes[0].set_title("Field φ (pair final)")
    axes[1].imshow(node_field, cmap="inferno")
    axes[1].set_title("Node field")
    axes[2].imshow(occ, cmap="hot")
    axes[2].scatter(cx, cy, c="cyan", s=50)
    axes[2].set_title("Occupancy + centroid")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out1 = OUT / "pair_interaction_final.png"
    plt.savefig(out1, dpi=320)
    plt.close(fig)

    # figure 2
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(std_hist)
    axes[0].set_title("phi std")
    axes[1].plot(flux_hist)
    axes[1].set_title("flux max")
    axes[2].plot(node_hist)
    axes[2].set_title("node max")
    axes[3].plot(dist_hist)
    axes[3].set_title("center drift")
    plt.tight_layout()
    out2 = OUT / "pair_interaction_diagnostics.png"
    plt.savefig(out2, dpi=320)
    plt.close(fig)

    np.save(OUT / "phi_pair_final.npy", phi)
    np.save(OUT / "node_pair_final.npy", node_field)
    np.save(OUT / "occ_pair_final.npy", occ)

    print(f"[OK] wrote {out1}")
    print(f"[OK] wrote {out2}")
    print("[DONE] proto atom pair interaction complete")


if __name__ == "__main__":
    main()