#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FAMILY TO CLASSIFY
# ============================================================

OUT = Path("results/proto_atoms/proto_atom_classifier")
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 128
N_STEPS = 180

BETA = 8.50
CENTER_GAIN = 0.014
NODE_GAIN = 0.085
MATTER_GAIN = 0.104
FLUX_GAIN = 0.040
EDGE_PENALTY = 0.12

SEED_LIST = list(range(1, 13))

OMEGA_BG = 0.22
BACKGROUND_GAIN = 0.035
OMEGA_LOCAL = 0.47
LOCAL_BEAT_GAIN = 0.085

PHI_CLIP = 6.0
FLUX_CLIP = 1e6
NODE_CLIP = 1e8


# ============================================================
# NUMERICAL CORE
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


# ============================================================
# METRICS
# ============================================================

def weighted_centroid(w):
    total = float(np.sum(w))
    if (not np.isfinite(total)) or total <= 1e-12:
        h, ww = w.shape
        return (ww - 1) / 2, (h - 1) / 2

    y, x = np.indices(w.shape)
    cx = float(np.sum(x * w) / total)
    cy = float(np.sum(y * w) / total)
    return cx, cy


def centroid_offset(occ):
    cx, cy = weighted_centroid(occ)
    cx0 = (occ.shape[1] - 1) / 2
    cy0 = (occ.shape[0] - 1) / 2
    return float(np.sqrt((cx - cx0) ** 2 + (cy - cy0) ** 2))


def weighted_covariance(w, cx, cy):
    total = float(np.sum(w))
    if total <= 1e-12:
        return np.eye(2)

    y, x = np.indices(w.shape)
    dx = x - cx
    dy = y - cy

    cxx = float(np.sum(w * dx * dx) / total)
    cyy = float(np.sum(w * dy * dy) / total)
    cxy = float(np.sum(w * dx * dy) / total)

    return np.array([[cxx, cxy], [cxy, cyy]], dtype=float)


def anisotropy_ratio(occ):
    cx, cy = weighted_centroid(occ)
    cov = weighted_covariance(occ, cx, cy)
    eigvals = np.linalg.eigvalsh(cov)
    return float(eigvals.max() / (eigvals.min() + 1e-8))


def core_mass_fraction(occ, r_core=10.0):
    total = float(np.sum(occ))
    if total <= 1e-12:
        return 0.0
    h, w = occ.shape
    y, x = np.indices(occ.shape)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    core = occ[r < r_core]
    return float(core.sum() / total)


def radial_stats(occ):
    total = float(np.sum(occ))
    if total <= 1e-12:
        return 0.0, 0.0
    h, w = occ.shape
    y, x = np.indices(occ.shape)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_mean = float(np.sum(r * occ) / total)
    r_std = float(np.sqrt(np.sum(((r - r_mean) ** 2) * occ) / total))
    return r_mean, r_std


def angular_profile(w, cx, cy, r_min=4.0, r_max=18.0, n_bins=48):
    y, x = np.indices(w.shape)
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    mask = (r >= r_min) & (r <= r_max)
    theta_sel = theta[mask]
    w_sel = w[mask]

    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    values = np.zeros(n_bins)

    for i in range(n_bins):
        m = (theta_sel >= bins[i]) & (theta_sel < bins[i + 1])
        if np.any(m):
            values[i] = np.sum(w_sel[m])

    centers = 0.5 * (bins[:-1] + bins[1:])
    vmax = float(np.max(values))
    if vmax > 0:
        values = values / vmax
    return centers, values


def count_lobes(profile, threshold=0.55):
    n = len(profile)
    count = 0
    for i in range(n):
        left = profile[(i - 1) % n]
        center = profile[i]
        right = profile[(i + 1) % n]
        if center > left and center > right and center >= threshold:
            count += 1
    return count


def split_score(occ):
    total = float(np.sum(occ))
    if total <= 1e-12:
        return 0.0

    h, w = occ.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    y, x = np.indices(occ.shape)

    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring = (r > 3.0) & (r < 18.0)

    top = float(occ[ring & (y < cy)].sum())
    bottom = float(occ[ring & (y > cy)].sum())
    left = float(occ[ring & (x < cx)].sum())
    right = float(occ[ring & (x > cx)].sum())

    tb = abs(top - bottom) / (top + bottom + 1e-8)
    lr = abs(left - right) / (left + right + 1e-8)
    return float(max(tb, lr))


def classify_state(core, split, lobes, anis, offset):
    if offset > 2.5:
        return "decentered"

    if core >= 0.18 and split <= 0.08 and 6 <= lobes <= 12:
        return "compact_symmetric"

    if core >= 0.14 and split <= 0.15 and lobes >= 8:
        return "compact_multilobed"

    if split >= 0.18:
        return "split_multilobed"

    if lobes <= 4:
        return "few_lobed"

    if lobes >= 10:
        return "high_lobed"

    if anis >= 1.25:
        return "anisotropic_shell"

    return "generic_multilobed"


# ============================================================
# DYNAMICS
# ============================================================

def run_case(seed):
    rng = np.random.default_rng(seed)

    phi0 = rng.uniform(-3, 3, (SIZE, SIZE))
    phi = phi0.copy()
    phase0 = make_local_phase_field(phi.shape, rng)

    std_hist = []
    flux_hist = []
    node_hist = []

    for step in range(N_STEPS):
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

        std_hist.append(float(np.std(phi)))
        flux_hist.append(float(np.max(flux_mag)))
        node_hist.append(float(np.max(node_field)))

    gx, gy, flux_mag = compute_flux(phi)
    node_field = compute_node_field(phi, flux_mag)
    occ = compute_occupancy(node_field)

    cx, cy = weighted_centroid(occ)
    off = centroid_offset(occ)
    anis = anisotropy_ratio(occ)
    core = core_mass_fraction(occ)
    split = split_score(occ)
    r_mean, r_std = radial_stats(occ)

    ang_centers, ang_values = angular_profile(occ, cx, cy)
    n_lobes = count_lobes(ang_values, threshold=0.55)

    state_class = classify_state(core, split, n_lobes, anis, off)

    return {
        "seed": seed,
        "final_std": std_hist[-1],
        "final_flux_max": flux_hist[-1],
        "final_node_max": node_hist[-1],
        "centroid_x": cx,
        "centroid_y": cy,
        "centroid_offset": off,
        "anisotropy": anis,
        "core_mass_fraction": core,
        "split_score": split,
        "radial_mean": r_mean,
        "radial_std": r_std,
        "n_lobes": n_lobes,
        "state_class": state_class,
        "phi": phi.copy(),
        "node_field": node_field.copy(),
        "occupancy": occ.copy(),
    }


# ============================================================
# PLOTTING
# ============================================================

def save_gallery(results_df, cases, out_path):
    n = len(results_df)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for k, (_, row) in enumerate(results_df.iterrows()):
        seed = int(row["seed"])
        occ = cases[seed]["occupancy"]

        ax = axes[k]
        ax.imshow(occ, cmap="hot")
        ax.set_title(
            f"seed={seed} | {row['state_class']}\n"
            f"off={row['centroid_offset']:.2f} core={row['core_mass_fraction']:.2f}\n"
            f"split={row['split_score']:.2f} lobes={int(row['n_lobes'])}"
        )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=260)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    rows = []
    cases = {}

    print("\n=== PROTO ATOM CLASSIFIER ===")
    print(
        f"[DEBUG] β={BETA:.2f}, C={CENTER_GAIN:.3f}, N={NODE_GAIN:.3f}, "
        f"M={MATTER_GAIN:.3f}, F={FLUX_GAIN:.3f}, E={EDGE_PENALTY:.2f}"
    )

    for seed in SEED_LIST:
        result = run_case(seed)
        cases[seed] = result

        rows.append({
            "seed": result["seed"],
            "final_std": result["final_std"],
            "final_flux_max": result["final_flux_max"],
            "final_node_max": result["final_node_max"],
            "centroid_x": result["centroid_x"],
            "centroid_y": result["centroid_y"],
            "centroid_offset": result["centroid_offset"],
            "anisotropy": result["anisotropy"],
            "core_mass_fraction": result["core_mass_fraction"],
            "split_score": result["split_score"],
            "radial_mean": result["radial_mean"],
            "radial_std": result["radial_std"],
            "n_lobes": result["n_lobes"],
            "state_class": result["state_class"],
        })

        print(
            f"seed={seed:02d} | "
            f"class={result['state_class']:<18} "
            f"off={result['centroid_offset']:.3f} "
            f"core={result['core_mass_fraction']:.3f} "
            f"split={result['split_score']:.3f} "
            f"lobes={result['n_lobes']:02d}"
        )

    df = pd.DataFrame(rows).sort_values(
        ["state_class", "centroid_offset", "split_score"],
        ascending=[True, True, True]
    )

    summary_csv = OUT / "proto_atom_classification_summary.csv"
    df.to_csv(summary_csv, index=False)

    save_gallery(df, cases, OUT / "proto_atom_classification_gallery.png")

    counts = (
        df.groupby("state_class")
        .agg(
            n_cases=("seed", "count"),
            mean_offset=("centroid_offset", "mean"),
            mean_core=("core_mass_fraction", "mean"),
            mean_split=("split_score", "mean"),
            mean_lobes=("n_lobes", "mean"),
            mean_anisotropy=("anisotropy", "mean"),
        )
        .reset_index()
        .sort_values("n_cases", ascending=False)
    )

    counts_csv = OUT / "proto_atom_classification_counts.csv"
    counts.to_csv(counts_csv, index=False)

    print("\n=== CLASS COUNTS ===")
    print(counts.to_string(index=False))

    print(f"\n[OK] wrote {summary_csv}")
    print(f"[OK] wrote {counts_csv}")
    print(f"[OK] wrote {OUT / 'proto_atom_classification_gallery.png'}")
    print("[DONE] proto atom classification complete")


if __name__ == "__main__":
    main()