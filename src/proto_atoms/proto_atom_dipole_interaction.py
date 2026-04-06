#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label

OUT = Path("results/proto_atoms/proto_atom_dipole_interaction")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

SIZE = 256
N_STEPS = 360

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

# elementary dipole: broken-symmetry version
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

SCENARIOS = [
    {
        "name": "parallel_close",
        "offset_a": (-28, -18),
        "offset_b": (28, 18),
        "flip_b": False,
    },
    {
        "name": "antiparallel_close",
        "offset_a": (-28, -18),
        "offset_b": (28, 18),
        "flip_b": True,
    },
    {
        "name": "parallel_far",
        "offset_a": (-44, -28),
        "offset_b": (44, 28),
        "flip_b": False,
    },
]


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
# SINGLE ATOM / SINGLE DIPOLE
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


def build_local_dipole():
    fields = {s: run_single(s) for s in SEEDS}
    phi = np.zeros((SIZE, SIZE), dtype=float)

    for s in SEEDS:
        phi += LOCAL_WEIGHT_MAP[s] * shifted(fields[s], *LOCAL_SHIFT_MAP[s])

    return phi


def rotate180(arr):
    return np.flipud(np.fliplr(arr))


# ============================================================
# COMPOSITE DYNAMICS
# ============================================================

def update_field(phi, phi0, phase0, step):
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
        + 0.5 * CENTER_GAIN * gaussian_center_mask(phi.shape, SIZE / 4) * normalize(node)
        - EDGE_PENALTY * edge_mask(phi.shape) * normalize(node)
        - 0.015 * phi * np.abs(phi)
    )

    phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
    phi = np.clip(phi, -PHI_CLIP, PHI_CLIP)
    return phi, gx, gy, node


# ============================================================
# OBSERVABLES
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


# ============================================================
# MAIN
# ============================================================

def run_scenario(cfg):
    dip_a = build_local_dipole()
    dip_b = build_local_dipole()
    if cfg["flip_b"]:
        dip_b = rotate180(dip_b)

    phi0 = np.zeros((SIZE, SIZE), dtype=float)
    phi0 += shifted(dip_a, cfg["offset_a"][0], cfg["offset_a"][1])
    phi0 += shifted(dip_b, cfg["offset_b"][0], cfg["offset_b"][1])

    phi = phi0.copy()

    rng = np.random.default_rng(9000 + abs(cfg["offset_a"][0]) + abs(cfg["offset_b"][0]) + int(cfg["flip_b"]))
    phase = make_phase(phi.shape, rng)

    std_hist = []
    flux_hist = []
    node_hist = []
    distance_hist = []

    snapshots = {}

    for step in range(N_STEPS):
        phi, gx, gy, node = update_field(phi, phi0, phase, step)
        occ = compute_occupancy(node)
        centers, masses = detect_components(occ, thr=0.35)

        if len(centers) >= 2:
            d = float(np.sqrt((centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2))
        else:
            d = np.nan

        std_hist.append(float(np.std(phi)))
        flux_hist.append(float(np.max(np.sqrt(gx * gx + gy * gy))))
        node_hist.append(float(np.max(node)))
        distance_hist.append(d)

        if step in [0, 120, 240, N_STEPS - 1]:
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
        "name": cfg["name"],
        "flip_b": cfg["flip_b"],
        "offset_a_y": cfg["offset_a"][0],
        "offset_a_x": cfg["offset_a"][1],
        "offset_b_y": cfg["offset_b"][0],
        "offset_b_x": cfg["offset_b"][1],
        "final_std": std_hist[-1],
        "final_flux_max": flux_hist[-1],
        "final_node_max": node_hist[-1],
        "final_distance": final_distance,
        "anisotropy_ratio": anisotropy,
        "major_axis_angle_deg": axis_angle_deg,
        "dipole_amplitude": dip_amp,
        "dipole_angle_deg": dip_ang,
        "interaction_score": score,
        "occ": occ,
        "snapshots": snapshots,
    }


def main():
    print("\n=== PROTO ATOM DIPOLE INTERACTION ===")

    rows = []
    cases = []

    for cfg in SCENARIOS:
        result = run_scenario(cfg)
        cases.append(result)

        rows.append({
            "scenario": result["name"],
            "flip_b": result["flip_b"],
            "final_std": result["final_std"],
            "final_flux_max": result["final_flux_max"],
            "final_node_max": result["final_node_max"],
            "final_distance": result["final_distance"],
            "anisotropy_ratio": result["anisotropy_ratio"],
            "major_axis_angle_deg": result["major_axis_angle_deg"],
            "dipole_amplitude": result["dipole_amplitude"],
            "dipole_angle_deg": result["dipole_angle_deg"],
            "interaction_score": result["interaction_score"],
        })

        print(
            f"{result['name']}: "
            f"d={result['final_distance']:.3f} "
            f"dip={result['dipole_amplitude']:.3f} "
            f"anis={result['anisotropy_ratio']:.3f} "
            f"score={result['interaction_score']:.3f}"
        )

    df = pd.DataFrame(rows).sort_values("interaction_score", ascending=False)
    summary_csv = OUT / "dipole_interaction_summary.csv"
    df.to_csv(summary_csv, index=False)

    print("\n=== DIPOLE INTERACTION SUMMARY ===")
    print(df.to_string(index=False))

    # --------------------------------------------------------
    # figure 1: gallery
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 5))
    if len(cases) == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        ax.imshow(case["occ"], cmap="hot")
        ax.set_title(
            f"{case['name']}\n"
            f"d={case['final_distance']:.1f}\n"
            f"dip={case['dipole_amplitude']:.2f}\n"
            f"score={case['interaction_score']:.2f}"
        )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig1 = OUT / "dipole_interaction_gallery.png"
    plt.savefig(fig1, dpi=260)
    plt.close(fig)

    # --------------------------------------------------------
    # figure 2: score bars
    # --------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.bar(df["scenario"], df["interaction_score"])
    plt.ylabel("interaction score")
    plt.title("Dipole-dipole interaction score")
    plt.tight_layout()
    fig2 = OUT / "dipole_interaction_scores.png"
    plt.savefig(fig2, dpi=260)
    plt.close()

    print(f"[OK] wrote {summary_csv}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print("[DONE] dipole interaction complete")


if __name__ == "__main__":
    main()