#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results/proto_atoms/proto_atom_effective_dipole")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

SIZE = 224
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

# best lead compound / good geometric candidate
SEEDS = [12, 2, 7]

SHIFT_MAP = {
    12: (0, 0),
    2: (-26, 0),
    7: (0, 26),
}

MIX_COEFF = 0.82


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
    d = np.minimum.reduce([x, w - 1 - x, y, h - 1 - y])
    return np.exp(-d / width)


def make_phase(shape, rng):
    return rng.uniform(0.0, 2.0 * np.pi, size=shape)


def shifted(arr, dy, dx):
    return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)


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
# SIMULATION
# ============================================================

def run_single(seed):
    rng = np.random.default_rng(seed)
    phi0 = rng.uniform(-3, 3, (SIZE, SIZE))
    phi = phi0.copy()
    phase0 = make_phase(phi.shape, rng)

    for step in range(180):
        lap = laplacian(phi)
        gx, gy, flux = compute_flux(phi)
        div = compute_divergence(gx, gy)
        node = compute_node_field(phi, flux)

        phi = (
            phi
            + 0.085 * lap
            - 0.008 * BETA * phi
            + BACKGROUND_GAIN * np.sin(OMEGA_BG * step) * phi0
            + LOCAL_BEAT_GAIN * np.sin(OMEGA_LOCAL * step + phase0) * phi
            + MATTER_GAIN * normalize(-div)
            + NODE_GAIN * normalize(node)
            + FLUX_GAIN * normalize(flux)
            + CENTER_GAIN * gaussian_center_mask(phi.shape) * normalize(node)
            - EDGE_PENALTY * edge_mask(phi.shape) * normalize(node)
            - 0.015 * phi * np.abs(phi)
        )

        phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        phi = np.clip(phi, -PHI_CLIP, PHI_CLIP)

    return phi


def run_composite():
    fields = {s: run_single(s) for s in SEEDS}

    phi0 = sum(MIX_COEFF * shifted(fields[s], *SHIFT_MAP[s]) for s in SEEDS)
    phi = phi0.copy()

    rng = np.random.default_rng(42)
    phase = make_phase(phi.shape, rng)

    for step in range(N_STEPS):
        lap = laplacian(phi)
        gx, gy, flux = compute_flux(phi)
        div = compute_divergence(gx, gy)
        node = compute_node_field(phi, flux)

        phi = (
            phi
            + 0.085 * lap
            - 0.008 * BETA * phi
            + BACKGROUND_GAIN * np.sin(OMEGA_BG * step) * phi0
            + LOCAL_BEAT_GAIN * np.sin(OMEGA_LOCAL * step + phase) * phi
            + MATTER_GAIN * normalize(-div)
            + NODE_GAIN * normalize(node)
            + FLUX_GAIN * normalize(flux)
            + 0.5 * CENTER_GAIN * gaussian_center_mask(phi.shape, SIZE / 4) * normalize(node)
            - EDGE_PENALTY * edge_mask(phi.shape) * normalize(node)
            - 0.015 * phi * np.abs(phi)
        )

        phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        phi = np.clip(phi, -PHI_CLIP, PHI_CLIP)

    gx, gy, flux = compute_flux(phi)
    node = compute_node_field(phi, flux)
    occ = compute_occupancy(node)

    return phi, node, occ, gx, gy


# ============================================================
# ORIENTATION / DIPOLE
# ============================================================

def weighted_covariance(w, cx, cy):
    y, x = np.indices(w.shape)
    total = float(np.sum(w)) + 1e-12

    dx = x - cx
    dy = y - cy

    cxx = float(np.sum(w * dx * dx) / total)
    cyy = float(np.sum(w * dy * dy) / total)
    cxy = float(np.sum(w * dx * dy) / total)

    return np.array([[cxx, cxy], [cxy, cyy]], dtype=float)


def compute_dipole_like_vector(w, cx, cy):
    y, x = np.indices(w.shape)
    dx = x - cx
    dy = y - cy

    # moment dipolaire-like pondéré
    px = float(np.sum(w * dx))
    py = float(np.sum(w * dy))

    amp = float(np.sqrt(px * px + py * py))
    ang = float(np.degrees(np.arctan2(py, px))) if amp > 1e-12 else np.nan
    return px, py, amp, ang


def half_plane_asymmetry(w, cx, cy, axis_angle_rad):
    y, x = np.indices(w.shape)
    dx = x - cx
    dy = y - cy

    # normale à l'axe
    nx = -np.sin(axis_angle_rad)
    ny =  np.cos(axis_angle_rad)

    side = dx * nx + dy * ny

    plus_mass = float(np.sum(w[side >= 0]))
    minus_mass = float(np.sum(w[side < 0]))
    total = plus_mass + minus_mass + 1e-12

    asym = (plus_mass - minus_mass) / total
    return plus_mass, minus_mass, float(asym)


def flux_circulation_indicator(gx, gy, cx, cy, w):
    y, x = np.indices(w.shape)
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-12

    # composante tangentielle locale: e_theta = (-dy/r, dx/r)
    tx = -dy / r
    ty =  dx / r

    tangential_flux = gx * tx + gy * ty

    # moyenne pondérée par occupancy
    circ = float(np.sum(w * tangential_flux) / (np.sum(w) + 1e-12))
    return circ


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n=== PROTO ATOM EFFECTIVE DIPOLE ===")

    phi, node, occ, gx, gy = run_composite()

    cx, cy = weighted_centroid(occ)
    cov = weighted_covariance(occ, cx, cy)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = eigvecs[:, 0]
    major_angle_rad = float(np.arctan2(major[1], major[0]))
    major_angle_deg = float(np.degrees(major_angle_rad))
    anisotropy = float(eigvals[0] / (eigvals[1] + 1e-12))

    px, py, pamp, pang = compute_dipole_like_vector(occ, cx, cy)
    plus_mass, minus_mass, asym = half_plane_asymmetry(occ, cx, cy, major_angle_rad)
    circulation = flux_circulation_indicator(gx, gy, cx, cy, occ)

    summary = pd.DataFrame([{
        "centroid_x": cx,
        "centroid_y": cy,
        "eigval_major": eigvals[0],
        "eigval_minor": eigvals[1],
        "anisotropy_ratio": anisotropy,
        "major_axis_angle_deg": major_angle_deg,
        "dipole_px": px,
        "dipole_py": py,
        "dipole_amplitude": pamp,
        "dipole_angle_deg": pang,
        "half_plane_plus_mass": plus_mass,
        "half_plane_minus_mass": minus_mass,
        "half_plane_asymmetry": asym,
        "circulation_indicator": circulation,
    }])

    summary_csv = OUT / "effective_dipole_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(summary.to_string(index=False))

    # --------------------------------------------------------
    # Figure 1: occupancy + major axis + dipole arrow
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(occ, cmap="hot")
    ax.scatter(cx, cy, c="cyan", s=70, label="centroid")

    scale_axis = 28.0
    x0 = cx - scale_axis * major[0]
    y0 = cy - scale_axis * major[1]
    x1 = cx + scale_axis * major[0]
    y1 = cy + scale_axis * major[1]
    ax.plot([x0, x1], [y0, y1], color="cyan", linewidth=2, label="major axis")

    if np.isfinite(pang):
        norm = pamp + 1e-12
        ux = px / norm
        uy = py / norm
        scale_dip = 22.0
        ax.arrow(
            cx, cy,
            scale_dip * ux,
            scale_dip * uy,
            color="lime",
            width=0.4,
            head_width=3.0,
            length_includes_head=True
        )

    ax.set_title("Effective dipole / orientation")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "effective_dipole_orientation.png", dpi=260)
    plt.close(fig)

    # --------------------------------------------------------
    # Figure 2: phi / node / occ
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(phi, cmap="viridis")
    axes[0].set_title("Field φ")
    axes[1].imshow(node, cmap="inferno")
    axes[1].set_title("Node field")
    axes[2].imshow(occ, cmap="hot")
    axes[2].set_title("Occupancy")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(OUT / "effective_dipole_triptych.png", dpi=260)
    plt.close(fig)

    print(f"[OK] wrote {summary_csv}")
    print(f"[OK] wrote {OUT / 'effective_dipole_orientation.png'}")
    print(f"[OK] wrote {OUT / 'effective_dipole_triptych.png'}")
    print("[DONE] effective dipole complete")


if __name__ == "__main__":
    main()