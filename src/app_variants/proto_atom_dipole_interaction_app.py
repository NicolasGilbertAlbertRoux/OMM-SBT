#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHI_CLIP = 6.0
FLUX_CLIP = 1e6
NODE_CLIP = 1e8

OUT = ROOT / "results" / "app_variants" / "proto_atom_dipole_interaction"
OUT.mkdir(parents=True, exist_ok=True)

FIG_OUT = ROOT / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# Defaults from parent
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

SEEDS = [12, 2, 7]
LOCAL_SHIFT_MAP = {
    12: (0, 0),
    2: (-26, 0),
    7: (4, 34),
}
LOCAL_WEIGHT_MAP = {
    12: 0.82,
    2: 0.82,
    7: 0.94,
}


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


def weighted_covariance(w, cx, cy):
    y, x = np.indices(w.shape)
    total = float(np.sum(w)) + 1e-12
    dx = x - cx
    dy = y - cy

    cxx = float(np.sum(w * dx * dx) / total)
    cyy = float(np.sum(w * dy * dy) / total)
    cxy = float(np.sum(w * dx * dy) / total)

    return np.array([[cxx, cxy], [cxy, cyy]], dtype=float)


def principal_axis_data(w):
    cx, cy = weighted_centroid(w)
    cov = weighted_covariance(w, cx, cy)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = eigvecs[:, 0]
    angle = float(np.degrees(np.arctan2(major[1], major[0])))
    anisotropy = float(eigvals[0] / (eigvals[1] + 1e-12))

    return cx, cy, major, angle, anisotropy, eigvals


def signed_phi_dipole(phi):
    pos = np.clip(phi, 0.0, None)
    neg = np.clip(-phi, 0.0, None)

    y, x = np.indices(phi.shape)

    pos_mass = float(pos.sum())
    neg_mass = float(neg.sum())

    if pos_mass <= 1e-12 or neg_mass <= 1e-12:
        return np.nan, np.nan, np.nan, np.nan, 0.0, np.nan

    pos_cx = float((x * pos).sum() / pos_mass)
    pos_cy = float((y * pos).sum() / pos_mass)
    neg_cx = float((x * neg).sum() / neg_mass)
    neg_cy = float((y * neg).sum() / neg_mass)

    dx = pos_cx - neg_cx
    dy = pos_cy - neg_cy
    amp = float(np.sqrt(dx * dx + dy * dy))
    ang = float(np.degrees(np.arctan2(dy, dx)))

    return pos_cx, pos_cy, neg_cx, neg_cy, amp, ang


def detect_components(occ, thr=0.35):
    mask = occ > thr
    lab, n = label(mask)

    centers = []
    masses = []

    y, x = np.indices(occ.shape)

    for k in range(1, n + 1):
        m = lab == k
        mass = float(occ[m].sum())
        if mass <= 1e-12:
            continue
        cx = float((x[m] * occ[m]).sum() / mass)
        cy = float((y[m] * occ[m]).sum() / mass)
        centers.append((cx, cy))
        masses.append(mass)

    order = np.argsort(masses)[::-1]
    centers = [centers[i] for i in order]
    masses = [masses[i] for i in order]
    return centers, masses


def interaction_score(final_distance, dipole_amp, anisotropy):
    dist_term = max(0.0, 1.0 - final_distance / 90.0)
    dip_term = min(dipole_amp / 6.0, 1.0)
    anis_term = min((anisotropy - 1.0) / 0.5, 1.0)
    score = 0.45 * dist_term + 0.35 * dip_term + 0.20 * max(0.0, anis_term)
    return float(score)


def run_single(seed, size):
    rng = np.random.default_rng(seed)
    phi0 = rng.uniform(-3, 3, (size, size))
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


def build_local_dipole(size):
    fields = {s: run_single(s, size) for s in SEEDS}
    phi = np.zeros((size, size), dtype=float)

    for s in SEEDS:
        dy, dx = LOCAL_SHIFT_MAP[s]
        phi += LOCAL_WEIGHT_MAP[s] * shifted(fields[s], dy, dx)

    return phi


def rotate180(arr):
    return np.flipud(np.fliplr(arr))


def update_field(phi, phi0, phase0, step, size):
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
        + 0.5 * CENTER_GAIN * gaussian_center_mask(phi.shape, size / 4) * normalize(node)
        - EDGE_PENALTY * edge_mask(phi.shape) * normalize(node)
        - 0.015 * phi * np.abs(phi)
    )

    phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
    phi = np.clip(phi, -PHI_CLIP, PHI_CLIP)
    return phi, gx, gy, node


def run_case(size, n_steps, distance, angle_deg, flip_b):
    theta = np.deg2rad(angle_deg)
    dx = int(round((distance * np.cos(theta)) / 2.0))
    dy = int(round((distance * np.sin(theta)) / 2.0))

    offset_a = (-dy, -dx)
    offset_b = (dy, dx)

    dip_a = build_local_dipole(size)
    dip_b = build_local_dipole(size)
    if flip_b:
        dip_b = rotate180(dip_b)

    phi0 = np.zeros((size, size), dtype=float)
    phi0 += shifted(dip_a, offset_a[0], offset_a[1])
    phi0 += shifted(dip_b, offset_b[0], offset_b[1])

    phi = phi0.copy()
    rng = np.random.default_rng(9000 + abs(offset_a[0]) + abs(offset_b[0]) + int(flip_b))
    phase = make_phase(phi.shape, rng)

    snapshots = {}

    for step in range(n_steps):
        phi, gx, gy, node = update_field(phi, phi0, phase, step, size)
        occ = compute_occupancy(node)

        if step in [0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]:
            snapshots[step] = occ.copy()

    occ = compute_occupancy(node)
    cx, cy, major, axis_angle_deg, anisotropy, eigvals = principal_axis_data(occ)
    pos_cx, pos_cy, neg_cx, neg_cy, dip_amp, dip_ang = signed_phi_dipole(phi)
    centers, masses = detect_components(occ, thr=0.35)

    if len(centers) >= 2:
        final_distance = float(np.sqrt((centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2))
    else:
        final_distance = np.nan

    score = interaction_score(final_distance if np.isfinite(final_distance) else 120.0, dip_amp, anisotropy)

    return {
        "occ": occ,
        "snapshots": snapshots,
        "final_distance": final_distance,
        "anisotropy_ratio": anisotropy,
        "major_axis_angle_deg": axis_angle_deg,
        "dipole_amplitude": dip_amp,
        "dipole_angle_deg": dip_ang,
        "interaction_score": score,
    }


def main():
    parser = argparse.ArgumentParser(description="Interactive dipole interaction based on the real OMM code.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=360)
    parser.add_argument("--distance", type=float, default=60.0)
    parser.add_argument("--angle_deg", type=float, default=0.0)
    parser.add_argument("--flip_b", type=int, default=0)

    args = parser.parse_args()

    result = run_case(
        size=args.size,
        n_steps=args.n_steps,
        distance=args.distance,
        angle_deg=args.angle_deg,
        flip_b=bool(args.flip_b),
    )

    # Main figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(result["occ"], cmap="hot")
    axes[0].set_title(
        f"Final occupancy\n"
        f"d={result['final_distance']:.1f} | "
        f"dip={result['dipole_amplitude']:.2f}"
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].bar(
        ["interaction"],
        [result["interaction_score"]],
    )
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Interaction score")

    plt.tight_layout()
    fig_main = FIG_OUT / "app_proto_dipole.png"
    plt.savefig(fig_main, dpi=260)
    plt.close(fig)

    # Snapshot figure
    snaps = result["snapshots"]
    keys = sorted(snaps.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]

    for ax, k in zip(axes, keys):
        ax.imshow(snaps[k], cmap="hot")
        ax.set_title(f"step={k}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig_snap = FIG_OUT / "app_proto_dipole_snapshots.png"
    plt.savefig(fig_snap, dpi=220)
    plt.close(fig)

    print(f"[OK] wrote {fig_main}")
    print(f"[OK] wrote {fig_snap}")
    print("\n=== DIPOLE REPORT ===")
    print(f"interaction_score={result['interaction_score']:.6f}")
    print(f"final_distance={result['final_distance']:.6f}")
    print(f"dipole_amplitude={result['dipole_amplitude']:.6f}")
    print(f"anisotropy_ratio={result['anisotropy_ratio']:.6f}")
    print(f"major_axis_angle_deg={result['major_axis_angle_deg']:.6f}")
    print(f"dipole_angle_deg={result['dipole_angle_deg']:.6f}")


if __name__ == "__main__":
    main()
atom1 = gaussian_blob(cx - dx/2, cy - dy/2)
atom2 = gaussian_blob(cx + dx/2, cy + dy/2)

field = atom1 - atom2  # dipole structure

# ===============================
# METRICS
# ===============================

interaction_strength = np.sum(np.abs(atom1 * atom2))

if interaction_strength < 0.01:
    regime = "independent"
elif interaction_strength < 0.1:
    regime = "weak coupling"
else:
    regime = "dipole formed"

# ===============================
# SAVE
# ===============================

plt.imshow(field, cmap="coolwarm")
plt.axis("off")
plt.savefig(OUT / "app_proto_dipole.png", dpi=300)
plt.close()

# ===============================
# PRINT
# ===============================

print(f"interaction_strength={interaction_strength}")
print(f"regime={regime}")
