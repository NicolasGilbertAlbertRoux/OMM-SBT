#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

OUT = Path("results/cosmology/cosmic_mantle_expansion_scan")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# GLOBAL SETTINGS
# ============================================================

NX, NY = 320, 320
DT = 0.05
N_STEPS = 900

CENTER = np.array([NX / 2, NY / 2], dtype=float)

# underlying field
C_WAVE = 0.75
FIELD_MASS = 0.0
FIELD_DAMP = 0.0012

# effective energy density
ALPHA_GRAD2 = 0.5
BETA_PI2 = 0.5
GAMMA_PHI2 = 0.0

# local geometry
LOCAL_MODE = "helmholtz"   # "poisson" or "helmholtz"
LOCAL_ITERS = 80
LOCAL_MASS = 0.08
LOCAL_WEIGHT = 1.0

# default best two-scale regime
DEFAULT_EPSILON_GLOBAL = 0.02
DEFAULT_GLOBAL_SIGMA = 4.0
DEFAULT_GLOBAL_ETA = 0.005
DEFAULT_GLOBAL_MASS = 0.10

# source scale scan
SOURCE_SIGMAS = [3.5, 5.0, 7.0]

# global mantle scan
EPSILON_GLOBAL_VALUES = [0.0, 0.02, 0.05]
GLOBAL_SIGMA_VALUES = [4.0, 8.0, 16.0]
GLOBAL_ETA_VALUES = [0.005, 0.02]
GLOBAL_MASS_VALUES = [0.0, 0.10]

# expansion estimator radii
RADII = np.array([8, 12, 18, 26, 36, 52, 72], dtype=float)

TOP_N = 16
SNAP_STEPS = [0, 180, 360, 540, 720, N_STEPS - 1]

# ============================================================
# NUMERICAL TOOLS
# ============================================================

def laplacian(phi: np.ndarray) -> np.ndarray:
    return (
        np.roll(phi, 1, axis=0)
        + np.roll(phi, -1, axis=0)
        + np.roll(phi, 1, axis=1)
        + np.roll(phi, -1, axis=1)
        - 4.0 * phi
    )

def gradient(arr: np.ndarray):
    gy, gx = np.gradient(arr)
    return gx, gy

def gaussian(pos: np.ndarray, sigma: float) -> np.ndarray:
    y, x = np.indices((NY, NX))
    r2 = (x - pos[0])**2 + (y - pos[1])**2
    return np.exp(-r2 / (2.0 * sigma**2))

def solve_poisson_like(source: np.ndarray, n_iters: int, mass: float = 0.0):
    pot = np.zeros_like(source)
    for _ in range(n_iters):
        denom = 4.0 + mass**2
        pot = (
            np.roll(pot, 1, axis=0)
            + np.roll(pot, -1, axis=0)
            + np.roll(pot, 1, axis=1)
            + np.roll(pot, -1, axis=1)
            - source
        ) / denom
        pot[0, :] = 0.0
        pot[-1, :] = 0.0
        pot[:, 0] = 0.0
        pot[:, -1] = 0.0
    return pot

def ring_mean_abs(arr: np.ndarray, radius: float, tol: float = 1.5):
    y, x = np.indices(arr.shape)
    rr = np.sqrt((x - CENTER[0])**2 + (y - CENTER[1])**2)
    mask = np.abs(rr - radius) <= tol
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(arr[mask])))

def fit_exp_tail(x, y):
    # Fit y = A exp(-b x) on positive y
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = y > 1e-15
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan
    xx = x[mask]
    yy = y[mask]
    logy = np.log(yy)
    coeffs = np.polyfit(xx, logy, 1)
    slope, intercept = coeffs[0], coeffs[1]
    b = -slope
    A = np.exp(intercept)
    yfit = A * np.exp(-b * xx)
    mse = float(np.mean((yy - yfit)**2))
    lam = float(1.0 / b) if b > 1e-12 else np.nan
    return A, b, mse, lam

# ============================================================
# FIELD + GEOMETRY
# ============================================================

def evolve_field(phi: np.ndarray, pi: np.ndarray, src: np.ndarray):
    pi = pi + DT * (
        C_WAVE**2 * laplacian(phi)
        - FIELD_MASS**2 * phi
        + src
        - FIELD_DAMP * pi
    )
    phi = phi + DT * pi
    return phi, pi

def effective_energy_density(phi: np.ndarray, pi: np.ndarray) -> np.ndarray:
    gx, gy = gradient(phi)
    grad2 = gx**2 + gy**2
    rho = ALPHA_GRAD2 * grad2 + BETA_PI2 * (pi**2) + GAMMA_PHI2 * (phi**2)
    return rho

def build_local_geometry(rho_eff: np.ndarray):
    if LOCAL_MODE == "poisson":
        return solve_poisson_like(rho_eff, LOCAL_ITERS, mass=0.0)
    if LOCAL_MODE == "helmholtz":
        return solve_poisson_like(rho_eff, LOCAL_ITERS, mass=LOCAL_MASS)
    raise ValueError(f"Unknown LOCAL_MODE: {LOCAL_MODE}")

def build_two_scale_geometry(rho_eff, psi_global_mem, epsilon_global, global_sigma, global_eta, global_mass):
    psi_local = build_local_geometry(rho_eff)
    rho_global = gaussian_filter(rho_eff, sigma=global_sigma, mode="nearest")
    psi_global_inst = solve_poisson_like(rho_global, LOCAL_ITERS, mass=global_mass)
    psi_global_mem = (1.0 - global_eta) * psi_global_mem + global_eta * psi_global_inst
    psi_total = LOCAL_WEIGHT * psi_local + epsilon_global * psi_global_mem
    return psi_local, psi_global_mem, psi_total

# ============================================================
# COSMIC PROXIES
# ============================================================

def estimate_scale_factor(profile_values: np.ndarray, radii: np.ndarray):
    """
    Heuristic cosmic scale proxy:
    weighted mean radius of the geometry profile.
    """
    vals = np.asarray(profile_values, dtype=float)
    w = np.maximum(vals, 0.0)
    s = np.sum(w)
    if s <= 1e-18:
        return np.nan
    return float(np.sum(radii * w) / s)

def estimate_width(profile_values: np.ndarray, radii: np.ndarray):
    vals = np.asarray(profile_values, dtype=float)
    w = np.maximum(vals, 0.0)
    s = np.sum(w)
    if s <= 1e-18:
        return np.nan
    mean_r = np.sum(radii * w) / s
    var_r = np.sum(((radii - mean_r)**2) * w) / s
    return float(np.sqrt(max(var_r, 0.0)))

# ============================================================
# SINGLE RUN
# ============================================================

def run_case(source_sigma, epsilon_global, global_sigma, global_eta, global_mass):
    src = gaussian(CENTER, sigma=source_sigma)

    phi = np.zeros((NY, NX), dtype=float)
    pi = np.zeros_like(phi)
    psi_global_mem = np.zeros_like(phi)

    rows = []
    snapshots = {}

    for step in range(N_STEPS):
        phi, pi = evolve_field(phi, pi, src)
        rho_eff = effective_energy_density(phi, pi)
        psi_local, psi_global_mem, psi_total = build_two_scale_geometry(
            rho_eff, psi_global_mem, epsilon_global, global_sigma, global_eta, global_mass
        )

        local_profile = np.array([ring_mean_abs(psi_local, r) for r in RADII], dtype=float)
        global_profile = np.array([ring_mean_abs(psi_global_mem, r) for r in RADII], dtype=float)
        total_profile = np.array([ring_mean_abs(psi_total, r) for r in RADII], dtype=float)

        a_local = estimate_scale_factor(local_profile, RADII)
        a_global = estimate_scale_factor(global_profile, RADII)
        a_total = estimate_scale_factor(total_profile, RADII)

        w_local = estimate_width(local_profile, RADII)
        w_global = estimate_width(global_profile, RADII)
        w_total = estimate_width(total_profile, RADII)

        # effective screening on total profile
        A, b, mse_exp, lambda_eff = fit_exp_tail(RADII, total_profile)

        rows.append({
            "step": step,
            "time": step * DT,
            "mean_rho": float(np.mean(rho_eff)),
            "max_rho": float(np.max(rho_eff)),
            "mean_abs_psi_local": float(np.mean(np.abs(psi_local))),
            "mean_abs_psi_global": float(np.mean(np.abs(psi_global_mem))),
            "mean_abs_psi_total": float(np.mean(np.abs(psi_total))),
            "a_local": a_local,
            "a_global": a_global,
            "a_total": a_total,
            "width_local": w_local,
            "width_global": w_global,
            "width_total": w_total,
            "lambda_eff": lambda_eff,
            "exp_tail_mse": mse_exp,
        })

        if step in SNAP_STEPS:
            snapshots[step] = {
                "rho": rho_eff.copy(),
                "psi_total": psi_total.copy(),
                "psi_global": psi_global_mem.copy(),
            }

        if step % 150 == 0:
            print(
                f"step={step} "
                f"mean_rho={np.mean(rho_eff):.6e} "
                f"a_total={a_total:.6f} "
                f"lambda_eff={lambda_eff:.6f}"
            )

    hist = pd.DataFrame(rows)
    hist["H_total"] = np.gradient(hist["a_total"].values, DT) / np.maximum(hist["a_total"].values, 1e-12)
    hist["H_global"] = np.gradient(hist["a_global"].values, DT) / np.maximum(hist["a_global"].values, 1e-12)
    hist["m_eff"] = 1.0 / np.maximum(hist["lambda_eff"].values, 1e-12)

    final = hist.iloc[-1]

    return hist, snapshots, {
        "source_sigma": source_sigma,
        "epsilon_global": epsilon_global,
        "global_sigma": global_sigma,
        "global_eta": global_eta,
        "global_mass": global_mass,
        "a_total_final": float(final["a_total"]),
        "H_total_final": float(final["H_total"]),
        "lambda_eff_final": float(final["lambda_eff"]),
        "m_eff_final": float(final["m_eff"]),
        "a_total_mean": float(np.nanmean(hist["a_total"])),
        "H_total_mean": float(np.nanmean(hist["H_total"])),
        "lambda_eff_mean": float(np.nanmean(hist["lambda_eff"])),
        "m_eff_mean": float(np.nanmean(hist["m_eff"])),
        "exp_tail_mse_mean": float(np.nanmean(hist["exp_tail_mse"])),
        "mean_abs_psi_total_final": float(final["mean_abs_psi_total"]),
        "max_rho_final": float(final["max_rho"]),
    }

# ============================================================
# MAIN SCAN
# ============================================================

def main():
    print("\n=== COSMIC MANTLE EXPANSION SCAN ===")

    cases = list(itertools.product(
        SOURCE_SIGMAS,
        EPSILON_GLOBAL_VALUES,
        GLOBAL_SIGMA_VALUES,
        GLOBAL_ETA_VALUES,
        GLOBAL_MASS_VALUES,
    ))

    summary_rows = []
    best_artifacts = None
    best_score = None

    total = len(cases)

    for i, (source_sigma, eps_g, sig_g, eta_g, m_g) in enumerate(cases, start=1):
        print(
            f"\n[{i}/{total}] "
            f"src_sigma={source_sigma:.1f} eps={eps_g:.2f} "
            f"g_sigma={sig_g:.1f} eta={eta_g:.3f} g_mass={m_g:.2f}"
        )

        hist, snapshots, summary = run_case(source_sigma, eps_g, sig_g, eta_g, m_g)
        summary_rows.append(summary)

        # Score heuristic:
        # prefer smooth expanding mantle + lower exp fit mse + larger lambda + finite H
        score = (
            summary["exp_tail_mse_mean"]
            - 0.02 * summary["lambda_eff_mean"]
            + 0.005 * abs(summary["H_total_mean"])
        )

        if best_score is None or score < best_score:
            best_score = score
            best_artifacts = (hist.copy(), snapshots.copy(), summary.copy())

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["exp_tail_mse_mean", "lambda_eff_mean"],
        ascending=[True, False]
    )

    summary_csv = OUT / "cosmic_mantle_expansion_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    top_df = summary_df.head(TOP_N).copy()
    top_csv = OUT / "cosmic_mantle_expansion_top.csv"
    top_df.to_csv(top_csv, index=False)

    print("\n=== TOP CONFIGURATIONS ===")
    print(top_df.to_string(index=False))

    # top comparison figure
    x = np.arange(len(top_df))
    labels = [
        f"src={r.source_sigma}|e={r.epsilon_global}|s={r.global_sigma}|η={r.global_eta}|m={r.global_mass}"
        for _, r in top_df.iterrows()
    ]

    plt.figure(figsize=(14, 5))
    plt.bar(x - 0.25, top_df["exp_tail_mse_mean"], width=0.25, label="mean exp-tail mse")
    plt.bar(x, top_df["lambda_eff_mean"], width=0.25, label="mean lambda_eff")
    plt.bar(x + 0.25, np.abs(top_df["H_total_mean"]), width=0.25, label="|mean H_total|")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.title("Cosmic mantle expansion top configurations")
    plt.legend()
    plt.tight_layout()
    top_fig = OUT / "cosmic_mantle_expansion_top.png"
    plt.savefig(top_fig, dpi=220)
    plt.close()

    # Best-case detailed outputs
    hist, snapshots, summary = best_artifacts
    best_hist_csv = OUT / "cosmic_mantle_expansion_best_history.csv"
    hist.to_csv(best_hist_csv, index=False)

    print("\n=== BEST CASE ===")
    print(pd.Series(summary).to_string())

    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["time"], hist["a_local"], label="a_local")
    plt.plot(hist["time"], hist["a_global"], label="a_global")
    plt.plot(hist["time"], hist["a_total"], label="a_total")
    plt.xlabel("time")
    plt.ylabel("pseudo scale factor")
    plt.title("Cosmic mantle pseudo-scale evolution")
    plt.legend()
    plt.tight_layout()
    fig1 = OUT / "cosmic_scale_factor.png"
    plt.savefig(fig1, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["time"], hist["H_global"], label="H_global")
    plt.plot(hist["time"], hist["H_total"], label="H_total")
    plt.xlabel("time")
    plt.ylabel("pseudo Hubble rate")
    plt.title("Cosmic mantle pseudo-expansion rate")
    plt.legend()
    plt.tight_layout()
    fig2 = OUT / "cosmic_hubble_rate.png"
    plt.savefig(fig2, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["time"], hist["lambda_eff"], label="lambda_eff")
    plt.plot(hist["time"], hist["m_eff"], label="m_eff")
    plt.xlabel("time")
    plt.ylabel("effective range / screening")
    plt.title("Effective screening evolution")
    plt.legend()
    plt.tight_layout()
    fig3 = OUT / "cosmic_screening_evolution.png"
    plt.savefig(fig3, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["time"], hist["mean_rho"], label="mean_rho")
    plt.plot(hist["time"], hist["max_rho"], label="max_rho")
    plt.xlabel("time")
    plt.ylabel("energy density")
    plt.title("Energy density evolution")
    plt.legend()
    plt.tight_layout()
    fig4 = OUT / "cosmic_energy_density_evolution.png"
    plt.savefig(fig4, dpi=220)
    plt.close()

    final_snap = snapshots[SNAP_STEPS[-1]]

    plt.figure(figsize=(6, 6))
    plt.imshow(final_snap["rho"], cmap="inferno")
    plt.scatter([CENTER[0]], [CENTER[1]], s=40)
    plt.title("Final effective energy density")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig5 = OUT / "cosmic_final_energy_density.png"
    plt.savefig(fig5, dpi=220)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(final_snap["psi_global"], cmap="coolwarm")
    plt.scatter([CENTER[0]], [CENTER[1]], s=40)
    plt.title("Final global mantle geometry")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig6 = OUT / "cosmic_final_global_geometry.png"
    plt.savefig(fig6, dpi=220)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(final_snap["psi_total"], cmap="coolwarm")
    plt.scatter([CENTER[0]], [CENTER[1]], s=40)
    plt.title("Final total geometry")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig7 = OUT / "cosmic_final_total_geometry.png"
    plt.savefig(fig7, dpi=220)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for ax, step in zip(axes, SNAP_STEPS):
        ax.imshow(snapshots[step]["psi_total"], cmap="coolwarm")
        ax.set_title(f"step={step}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig8 = OUT / "cosmic_geometry_snapshots.png"
    plt.savefig(fig8, dpi=220)
    plt.close(fig)

    print(f"[OK] wrote {summary_csv}")
    print(f"[OK] wrote {top_csv}")
    print(f"[OK] wrote {top_fig}")
    print(f"[OK] wrote {best_hist_csv}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print(f"[OK] wrote {fig3}")
    print(f"[OK] wrote {fig4}")
    print(f"[OK] wrote {fig5}")
    print(f"[OK] wrote {fig6}")
    print(f"[OK] wrote {fig7}")
    print(f"[OK] wrote {fig8}")
    print("[DONE] cosmic mantle expansion scan complete")

if __name__ == "__main__":
    main()