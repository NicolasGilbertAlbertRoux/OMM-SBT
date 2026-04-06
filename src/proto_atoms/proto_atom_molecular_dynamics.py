#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

OUT = Path("results/proto_atoms/proto_atom_molecular_dynamics")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

SIZE = 192
N_STEPS = 320

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

# meilleur proto-molécule trouvé
SEEDS = [3, 5, 7]
SHIFT_LIST = [
    (-22, -18),
    (20, -16),
    (0, 22),
]
MIX_COEFF = 0.85


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


def shifted(arr, dy, dx):
    return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)


# ============================================================
# SINGLE ATOM
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


# ============================================================
# COMPOSITE DYNAMICS
# ============================================================

def update_composite(phi, phi0, phase0, step):
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
# METRICS
# ============================================================

def detect_components(occ, thr=0.35):
    mask = occ > thr
    lab, n = label(mask)

    centers = []
    masses = []

    for k in range(1, n + 1):
        m = lab == k
        mass = float(occ[m].sum())
        if mass <= 1e-12:
            continue
        y, x = np.indices(occ.shape)
        cx = float((x[m] * occ[m]).sum() / mass)
        cy = float((y[m] * occ[m]).sum() / mass)
        centers.append((cx, cy))
        masses.append(mass)

    order = np.argsort(masses)[::-1]
    centers = [centers[i] for i in order]
    masses = [masses[i] for i in order]
    return centers, masses


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n=== PROTO ATOM MOLECULAR DYNAMICS ===")
    print(f"[DEBUG] seeds = {SEEDS}")

    fields = [run_single_atom(s) for s in SEEDS]
    shifted_fields = [shifted(fields[i], SHIFT_LIST[i][0], SHIFT_LIST[i][1]) for i in range(3)]

    phi0 = sum(MIX_COEFF * f for f in shifted_fields)
    phi = phi0.copy()

    rng = np.random.default_rng(777 + sum(SEEDS))
    phase0 = make_local_phase_field(phi.shape, rng)

    std_hist = []
    flux_hist = []
    node_hist = []
    drift_hist = []
    comp_hist = []

    snapshots = {}
    snapshot_steps = [0, 80, 160, 240, 319]

    for step in range(N_STEPS):
        phi, flux_mag, node_field = update_composite(phi, phi0, phase0, step)

        occ = compute_occupancy(node_field)
        cx, cy = weighted_centroid(occ)
        drift = float(np.sqrt((cx - (SIZE - 1) / 2) ** 2 + (cy - (SIZE - 1) / 2) ** 2))
        centers, masses = detect_components(occ, thr=0.35)

        std_hist.append(float(np.std(phi)))
        flux_hist.append(float(np.max(flux_mag)))
        node_hist.append(float(np.max(node_field)))
        drift_hist.append(drift)
        comp_hist.append(len(centers))

        if step in snapshot_steps:
            snapshots[step] = occ.copy()

        if step % 40 == 0:
            print(
                f"step={step} "
                f"std={np.std(phi):.4f} "
                f"flux={np.max(flux_mag):.4f} "
                f"node={np.max(node_field):.4f} "
                f"n_comp={len(centers)}"
            )

    occ = compute_occupancy(node_field)
    centers, masses = detect_components(occ, thr=0.35)

    # figure 1: final triptych
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(phi, cmap="viridis")
    axes[0].set_title("Field φ")
    axes[1].imshow(node_field, cmap="inferno")
    axes[1].set_title("Node field")
    axes[2].imshow(occ, cmap="hot")
    axes[2].set_title("Occupancy")

    for c in centers[:5]:
        axes[2].scatter(c[0], c[1], c="cyan", s=30)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig1 = OUT / "molecular_dynamics_final.png"
    plt.savefig(fig1, dpi=300)
    plt.close()

    # figure 2: diagnostics
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(std_hist)
    axes[0].set_title("phi std")
    axes[1].plot(flux_hist)
    axes[1].set_title("flux max")
    axes[2].plot(node_hist)
    axes[2].set_title("node max")
    axes[3].plot(comp_hist)
    axes[3].set_title("n strong components")
    plt.tight_layout()
    fig2 = OUT / "molecular_dynamics_diagnostics.png"
    plt.savefig(fig2, dpi=300)
    plt.close()

    # figure 3: snapshots
    fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(4 * len(snapshot_steps), 4))
    if len(snapshot_steps) == 1:
        axes = [axes]

    for ax, step in zip(axes, snapshot_steps):
        ax.imshow(snapshots[step], cmap="hot")
        ax.set_title(f"step={step}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig3 = OUT / "molecular_dynamics_snapshots.png"
    plt.savefig(fig3, dpi=300)
    plt.close()

    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print(f"[OK] wrote {fig3}")
    print("[DONE] proto atom molecular dynamics complete")


if __name__ == "__main__":
    main()