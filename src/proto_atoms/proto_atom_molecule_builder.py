#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label

OUT = Path("results/proto_atoms/proto_atom_molecule_builder")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

SIZE = 192
N_STEPS = 260

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

# pool prometteur issu de la proto-chimie
SEED_POOL = [2, 3, 5, 7, 8, 11, 12]

TRIPLE_LIST = [
    (2, 12, 7),
    (2, 12, 3),
    (7, 12, 11),
    (3, 7, 12),
    (5, 7, 8),
    (2, 8, 11),
    (3, 5, 7),
    (2, 3, 12),
]

MIX_COEFF = 0.85

# triangle initial
SHIFT_LIST = [
    (-22, -18),
    (20, -16),
    (0, 22),
]

TOP_K = 8


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
# MULTI-BODY DYNAMICS
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


def classify_molecule(occ):
    centers, masses = detect_components(occ, thr=0.35)
    n_comp = len(centers)

    if n_comp == 0:
        return "diffuse_failure", n_comp, np.nan

    if n_comp == 1:
        return "compact_molecule", n_comp, 0.0

    # distances among top 3 centers if present
    pts = centers[:3]
    if len(pts) >= 2:
        ds = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                ds.append(float(np.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)))
        mean_d = float(np.mean(ds)) if ds else np.nan
    else:
        mean_d = np.nan

    if n_comp == 2 and mean_d < 14.0:
        return "dimer_plus", n_comp, mean_d

    if n_comp >= 3 and mean_d < 18.0:
        return "triadic_molecule", n_comp, mean_d

    if n_comp >= 3 and mean_d < 32.0:
        return "loose_cluster", n_comp, mean_d

    return "fragmented", n_comp, mean_d


def molecule_score(mol_class, mean_d, final_center_drift, final_node_max):
    base = {
        "compact_molecule": 1.00,
        "triadic_molecule": 0.85,
        "dimer_plus": 0.70,
        "loose_cluster": 0.45,
        "fragmented": 0.15,
        "diffuse_failure": 0.00,
    }.get(mol_class, 0.0)

    d_term = 0.0 if np.isnan(mean_d) else max(0.0, 1.0 - mean_d / 40.0)
    drift_term = max(0.0, 1.0 - final_center_drift / 4.0)
    node_term = min(final_node_max / 25.0, 1.0)

    score = 0.55 * base + 0.20 * d_term + 0.15 * drift_term + 0.10 * node_term
    return float(score)


# ============================================================
# MAIN
# ============================================================

def run_triple(seeds):
    fields = [run_single_atom(s) for s in seeds]
    shifted_fields = [shifted(fields[i], SHIFT_LIST[i][0], SHIFT_LIST[i][1]) for i in range(3)]

    phi0 = sum(MIX_COEFF * f for f in shifted_fields)
    phi = phi0.copy()

    rng = np.random.default_rng(777 + sum(seeds))
    phase0 = make_local_phase_field(phi.shape, rng)

    std_hist = []
    flux_hist = []
    node_hist = []
    drift_hist = []

    for step in range(N_STEPS):
        phi, flux_mag, node_field = update_composite(phi, phi0, phase0, step)

        occ = compute_occupancy(node_field)
        cx, cy = weighted_centroid(occ)
        drift = float(np.sqrt((cx - (SIZE - 1) / 2) ** 2 + (cy - (SIZE - 1) / 2) ** 2))

        std_hist.append(float(np.std(phi)))
        flux_hist.append(float(np.max(flux_mag)))
        node_hist.append(float(np.max(node_field)))
        drift_hist.append(drift)

    occ = compute_occupancy(node_field)
    mol_class, n_comp, mean_d = classify_molecule(occ)

    return {
        "seeds": seeds,
        "molecule_class": mol_class,
        "n_components": n_comp,
        "mean_component_distance": mean_d,
        "final_std": std_hist[-1],
        "final_flux_max": flux_hist[-1],
        "final_node_max": node_hist[-1],
        "final_center_drift": drift_hist[-1],
        "mean_center_drift": float(np.mean(drift_hist)),
        "occupancy": occ.copy(),
        "phi": phi.copy(),
    }


def save_gallery(df, cases, out_path, n_show=8):
    top = df.head(n_show)

    ncols = 4
    nrows = math.ceil(len(top) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for k, (_, row) in enumerate(top.iterrows()):
        tid = int(row["triple_id"])
        occ = cases[tid]["occupancy"]

        ax = axes[k]
        ax.imshow(occ, cmap="hot")
        ax.set_title(
            f"{int(row['seed_a'])}+{int(row['seed_b'])}+{int(row['seed_c'])}\n"
            f"{row['molecule_class']}\n"
            f"n={int(row['n_components'])} score={row['molecule_score']:.2f}"
        )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=260)
    plt.close(fig)


def main():
    rows = []
    cases = {}

    print("\n=== PROTO ATOM MOLECULE BUILDER ===")
    print(
        f"[DEBUG] β={BETA:.2f}, C={CENTER_GAIN:.3f}, N={NODE_GAIN:.3f}, "
        f"M={MATTER_GAIN:.3f}, F={FLUX_GAIN:.3f}, E={EDGE_PENALTY:.2f}"
    )

    for triple_id, triple in enumerate(TRIPLE_LIST):
        result = run_triple(triple)
        mscore = molecule_score(
            result["molecule_class"],
            result["mean_component_distance"],
            result["final_center_drift"],
            result["final_node_max"],
        )

        cases[triple_id] = result

        rows.append({
            "triple_id": triple_id,
            "seed_a": triple[0],
            "seed_b": triple[1],
            "seed_c": triple[2],
            "molecule_class": result["molecule_class"],
            "n_components": result["n_components"],
            "mean_component_distance": result["mean_component_distance"],
            "final_std": result["final_std"],
            "final_flux_max": result["final_flux_max"],
            "final_node_max": result["final_node_max"],
            "final_center_drift": result["final_center_drift"],
            "mean_center_drift": result["mean_center_drift"],
            "molecule_score": mscore,
        })

        print(
            f"triple={triple[0]}+{triple[1]}+{triple[2]} | "
            f"class={result['molecule_class']:<16} "
            f"n={result['n_components']} "
            f"d={result['mean_component_distance']:.3f} "
            f"score={mscore:.3f}"
        )

    df = pd.DataFrame(rows).sort_values(
        ["molecule_score", "mean_component_distance"],
        ascending=[False, True]
    )

    summary_csv = OUT / "molecule_builder_summary.csv"
    df.to_csv(summary_csv, index=False)

    counts = (
        df.groupby("molecule_class")
        .agg(
            n_cases=("triple_id", "count"),
            mean_components=("n_components", "mean"),
            mean_distance=("mean_component_distance", "mean"),
            mean_score=("molecule_score", "mean"),
            mean_final_drift=("final_center_drift", "mean"),
        )
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )

    counts_csv = OUT / "molecule_builder_counts.csv"
    counts.to_csv(counts_csv, index=False)

    print("\n=== MOLECULE CLASS COUNTS ===")
    print(counts.to_string(index=False))

    save_gallery(df, cases, OUT / "molecule_builder_gallery.png", n_show=TOP_K)

    print(f"\n[OK] wrote {summary_csv}")
    print(f"[OK] wrote {counts_csv}")
    print(f"[OK] wrote {OUT / 'molecule_builder_gallery.png'}")
    print("[DONE] proto atom molecule builder complete")


if __name__ == "__main__":
    main()