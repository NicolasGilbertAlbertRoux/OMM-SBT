#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label

OUT = Path("results/proto_atoms/proto_atom_valence_map")
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

SEED_POOL = [2, 3, 5, 7, 8, 11, 12]

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


def classify_pair_behavior(occ):
    centers, masses = detect_components(occ, thr=0.35)
    n_comp = len(centers)

    if n_comp == 0:
        return "diffuse_failure", np.nan

    if n_comp == 1:
        return "fusion", 0.0

    c1 = centers[0]
    c2 = centers[1]
    d = float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))

    if d < 10.0:
        return "tight_bond", d
    if d < 28.0:
        return "loose_bond", d
    return "separated", d


def bond_score(pair_class, pair_distance, final_center_drift, final_node_max):
    base = {
        "tight_bond": 1.00,
        "loose_bond": 0.65,
        "fusion": 0.55,
        "separated": 0.10,
        "diffuse_failure": 0.00,
    }.get(pair_class, 0.0)

    d_term = 0.0 if np.isnan(pair_distance) else max(0.0, 1.0 - pair_distance / 40.0)
    drift_term = max(0.0, 1.0 - final_center_drift / 4.0)
    node_term = min(final_node_max / 25.0, 1.0)

    score = 0.55 * base + 0.20 * d_term + 0.15 * drift_term + 0.10 * node_term
    return float(score)


# ============================================================
# MAIN
# ============================================================

def run_pair(seed_a, seed_b):
    phi_a = run_single_atom(seed_a)
    phi_b = run_single_atom(seed_b)

    phi_a = shifted(phi_a, SHIFT_A[0], SHIFT_A[1])
    phi_b = shifted(phi_b, SHIFT_B[0], SHIFT_B[1])

    phi0 = MIX_COEFF * phi_a + MIX_COEFF * phi_b
    phi = phi0.copy()

    rng = np.random.default_rng(999 + seed_a * 100 + seed_b)
    phase0 = make_local_phase_field(phi.shape, rng)

    std_hist = []
    flux_hist = []
    node_hist = []
    drift_hist = []

    for step in range(N_STEPS):
        phi, flux_mag, node_field = update_pair(phi, phi0, phase0, step)

        occ = compute_occupancy(node_field)
        cx, cy = weighted_centroid(occ)
        drift = float(np.sqrt((cx - (SIZE - 1) / 2) ** 2 + (cy - (SIZE - 1) / 2) ** 2))

        std_hist.append(float(np.std(phi)))
        flux_hist.append(float(np.max(flux_mag)))
        node_hist.append(float(np.max(node_field)))
        drift_hist.append(drift)

    occ = compute_occupancy(node_field)
    pair_class, d12 = classify_pair_behavior(occ)

    return {
        "seed_a": seed_a,
        "seed_b": seed_b,
        "pair_class": pair_class,
        "pair_distance": d12,
        "final_std": std_hist[-1],
        "final_flux_max": flux_hist[-1],
        "final_node_max": node_hist[-1],
        "final_center_drift": drift_hist[-1],
        "mean_center_drift": float(np.mean(drift_hist)),
    }


def main():
    rows = []

    print("\n=== PROTO ATOM VALENCE MAP ===")
    print(
        f"[DEBUG] β={BETA:.2f}, C={CENTER_GAIN:.3f}, N={NODE_GAIN:.3f}, "
        f"M={MATTER_GAIN:.3f}, F={FLUX_GAIN:.3f}, E={EDGE_PENALTY:.2f}"
    )

    pair_list = list(itertools.combinations(SEED_POOL, 2))

    for seed_a, seed_b in pair_list:
        result = run_pair(seed_a, seed_b)
        bscore = bond_score(
            result["pair_class"],
            result["pair_distance"],
            result["final_center_drift"],
            result["final_node_max"],
        )

        rows.append({
            "seed_a": result["seed_a"],
            "seed_b": result["seed_b"],
            "pair_class": result["pair_class"],
            "pair_distance": result["pair_distance"],
            "final_std": result["final_std"],
            "final_flux_max": result["final_flux_max"],
            "final_node_max": result["final_node_max"],
            "final_center_drift": result["final_center_drift"],
            "mean_center_drift": result["mean_center_drift"],
            "bond_score": bscore,
        })

        print(
            f"pair={seed_a}+{seed_b} | "
            f"class={result['pair_class']:<12} "
            f"score={bscore:.3f}"
        )

    df = pd.DataFrame(rows)

    # valence = number of sufficiently strong bonds
    valence_rows = []
    for seed in SEED_POOL:
        g = df[(df["seed_a"] == seed) | (df["seed_b"] == seed)].copy()
        g["partner"] = np.where(g["seed_a"] == seed, g["seed_b"], g["seed_a"])

        n_tight = int((g["pair_class"] == "tight_bond").sum())
        n_loose = int((g["pair_class"] == "loose_bond").sum())
        n_good = int((g["bond_score"] >= 0.60).sum())
        mean_score = float(g["bond_score"].mean())
        max_score = float(g["bond_score"].max())
        best_partner = int(g.loc[g["bond_score"].idxmax(), "partner"])

        valence_rows.append({
            "seed": seed,
            "n_tight_bonds": n_tight,
            "n_loose_bonds": n_loose,
            "n_good_bonds": n_good,
            "mean_bond_score": mean_score,
            "max_bond_score": max_score,
            "best_partner": best_partner,
        })

    valence_df = pd.DataFrame(valence_rows).sort_values(
        ["n_good_bonds", "max_bond_score", "mean_bond_score"],
        ascending=[False, False, False]
    )

    summary_csv = OUT / "valence_map_summary.csv"
    valence_df.to_csv(summary_csv, index=False)

    pair_csv = OUT / "valence_map_pairs.csv"
    df.to_csv(pair_csv, index=False)

    print("\n=== VALENCE MAP ===")
    print(valence_df.to_string(index=False))

    # --------------------------------------------------------
    # Figure 1: valence bars
    # --------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(valence_df))
    plt.bar(x - 0.2, valence_df["n_tight_bonds"], width=0.2, label="tight")
    plt.bar(x,       valence_df["n_loose_bonds"], width=0.2, label="loose")
    plt.bar(x + 0.2, valence_df["n_good_bonds"], width=0.2, label="good (score>=0.60)")
    plt.xticks(x, valence_df["seed"])
    plt.xlabel("seed / proto-atom")
    plt.ylabel("bond count")
    plt.title("Emergent proto-valence map")
    plt.legend()
    plt.tight_layout()
    fig1 = OUT / "valence_map_bars.png"
    plt.savefig(fig1, dpi=240)
    plt.close()

    # --------------------------------------------------------
    # Figure 2: mean vs max bond score
    # --------------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(valence_df["mean_bond_score"], valence_df["max_bond_score"])
    for _, row in valence_df.iterrows():
        plt.text(row["mean_bond_score"], row["max_bond_score"], str(int(row["seed"])), fontsize=9)
    plt.xlabel("mean bond score")
    plt.ylabel("max bond score")
    plt.title("Proto-atomic valence landscape")
    plt.tight_layout()
    fig2 = OUT / "valence_map_scatter.png"
    plt.savefig(fig2, dpi=240)
    plt.close()

    print(f"\n[OK] wrote {summary_csv}")
    print(f"[OK] wrote {pair_csv}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print("[DONE] proto atom valence map complete")


if __name__ == "__main__":
    main()