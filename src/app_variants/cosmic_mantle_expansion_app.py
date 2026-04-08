#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

OUT = Path("results/app_variants/cosmic_mantle_expansion")
OUT.mkdir(parents=True, exist_ok=True)

FIG_OUT = Path("figures")
FIG_OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# DEFAULTS FROM PARENT SCRIPT
# ============================================================

NX_DEFAULT, NY_DEFAULT = 320, 320
DT_DEFAULT = 0.05
N_STEPS_DEFAULT = 900

C_WAVE_DEFAULT = 0.75
FIELD_MASS_DEFAULT = 0.0
FIELD_DAMP_DEFAULT = 0.0012

ALPHA_GRAD2_DEFAULT = 0.5
BETA_PI2_DEFAULT = 0.5
GAMMA_PHI2_DEFAULT = 0.0

LOCAL_ITERS_DEFAULT = 80
LOCAL_MASS_DEFAULT = 0.08
LOCAL_WEIGHT_DEFAULT = 1.0

RADII = np.array([8, 12, 18, 26, 36, 52, 72], dtype=float)
SNAP_STEPS_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


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


def gaussian(shape_x: int, shape_y: int, pos: np.ndarray, sigma: float) -> np.ndarray:
    y, x = np.indices((shape_y, shape_x))
    r2 = (x - pos[0]) ** 2 + (y - pos[1]) ** 2
    return np.exp(-r2 / (2.0 * sigma ** 2))


def solve_poisson_like(source: np.ndarray, n_iters: int, mass: float = 0.0):
    pot = np.zeros_like(source)
    for _ in range(n_iters):
        denom = 4.0 + mass ** 2
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


def ring_mean_abs(arr: np.ndarray, center: np.ndarray, radius: float, tol: float = 1.5):
    y, x = np.indices(arr.shape)
    rr = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = np.abs(rr - radius) <= tol
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(arr[mask])))


def fit_exp_tail(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = y > 1e-15
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan, np.nan
    xx = x[mask]
    yy = y[mask]
    logy = np.log(yy)
    coeffs = np.polyfit(xx, logy, 1)
    slope, intercept = coeffs[0], coeffs[1]
    b = -slope
    A = np.exp(intercept)
    yfit = A * np.exp(-b * xx)
    mse = float(np.mean((yy - yfit) ** 2))
    lam = float(1.0 / b) if b > 1e-12 else np.nan
    return A, b, mse, lam


# ============================================================
# FIELD + GEOMETRY
# ============================================================

def evolve_field(phi, pi, src, dt, c_wave, field_mass, field_damp):
    pi = pi + dt * (
        c_wave ** 2 * laplacian(phi)
        - field_mass ** 2 * phi
        + src
        - field_damp * pi
    )
    phi = phi + dt * pi
    return phi, pi


def effective_energy_density(phi, pi, alpha_grad2, beta_pi2, gamma_phi2):
    gx, gy = gradient(phi)
    grad2 = gx ** 2 + gy ** 2
    rho = alpha_grad2 * grad2 + beta_pi2 * (pi ** 2) + gamma_phi2 * (phi ** 2)
    return rho


def build_local_geometry(rho_eff, local_iters, local_mass):
    return solve_poisson_like(rho_eff, local_iters, mass=local_mass)


def build_two_scale_geometry(
    rho_eff,
    psi_global_mem,
    epsilon_global,
    global_sigma,
    global_eta,
    global_mass,
    local_iters,
    local_mass,
    local_weight,
):
    psi_local = build_local_geometry(rho_eff, local_iters, local_mass)
    rho_global = gaussian_filter(rho_eff, sigma=global_sigma, mode="nearest")
    psi_global_inst = solve_poisson_like(rho_global, local_iters, mass=global_mass)
    psi_global_mem = (1.0 - global_eta) * psi_global_mem + global_eta * psi_global_inst
    psi_total = local_weight * psi_local + epsilon_global * psi_global_mem
    return psi_local, psi_global_mem, psi_total


# ============================================================
# COSMIC PROXIES
# ============================================================

def estimate_scale_factor(profile_values: np.ndarray, radii: np.ndarray):
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
    var_r = np.sum(((radii - mean_r) ** 2) * w) / s
    return float(np.sqrt(max(var_r, 0.0)))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Interactive cosmic mantle expansion run.")
    parser.add_argument("--nx", type=int, default=NX_DEFAULT)
    parser.add_argument("--ny", type=int, default=NY_DEFAULT)
    parser.add_argument("--dt", type=float, default=DT_DEFAULT)
    parser.add_argument("--n_steps", type=int, default=360)

    parser.add_argument("--source_sigma", type=float, default=3.5)
    parser.add_argument("--epsilon_global", type=float, default=0.05)
    parser.add_argument("--global_sigma", type=float, default=16.0)
    parser.add_argument("--global_eta", type=float, default=0.02)
    parser.add_argument("--global_mass", type=float, default=0.0)

    parser.add_argument("--c_wave", type=float, default=C_WAVE_DEFAULT)
    parser.add_argument("--field_mass", type=float, default=FIELD_MASS_DEFAULT)
    parser.add_argument("--field_damp", type=float, default=FIELD_DAMP_DEFAULT)

    parser.add_argument("--local_iters", type=int, default=LOCAL_ITERS_DEFAULT)
    parser.add_argument("--local_mass", type=float, default=LOCAL_MASS_DEFAULT)
    parser.add_argument("--local_weight", type=float, default=LOCAL_WEIGHT_DEFAULT)

    args = parser.parse_args()

    nx, ny = args.nx, args.ny
    center = np.array([nx / 2, ny / 2], dtype=float)

    snap_steps = sorted(set(int((args.n_steps - 1) * f) for f in SNAP_STEPS_FRACTIONS))

    print("\n=== COSMIC MANTLE EXPANSION APP RUN ===")

    src = gaussian(nx, ny, center, sigma=args.source_sigma)

    phi = np.zeros((ny, nx), dtype=float)
    pi = np.zeros_like(phi)
    psi_global_mem = np.zeros_like(phi)

    rows = []
    snapshots = {}

    for step in range(args.n_steps):
        phi, pi = evolve_field(
            phi, pi, src,
            dt=args.dt,
            c_wave=args.c_wave,
            field_mass=args.field_mass,
            field_damp=args.field_damp,
        )

        rho_eff = effective_energy_density(
            phi, pi,
            alpha_grad2=ALPHA_GRAD2_DEFAULT,
            beta_pi2=BETA_PI2_DEFAULT,
            gamma_phi2=GAMMA_PHI2_DEFAULT,
        )

        psi_local, psi_global_mem, psi_total = build_two_scale_geometry(
            rho_eff,
            psi_global_mem,
            epsilon_global=args.epsilon_global,
            global_sigma=args.global_sigma,
            global_eta=args.global_eta,
            global_mass=args.global_mass,
            local_iters=args.local_iters,
            local_mass=args.local_mass,
            local_weight=args.local_weight,
        )

        local_profile = np.array([ring_mean_abs(psi_local, center, r) for r in RADII], dtype=float)
        global_profile = np.array([ring_mean_abs(psi_global_mem, center, r) for r in RADII], dtype=float)
        total_profile = np.array([ring_mean_abs(psi_total, center, r) for r in RADII], dtype=float)

        a_local = estimate_scale_factor(local_profile, RADII)
        a_global = estimate_scale_factor(global_profile, RADII)
        a_total = estimate_scale_factor(total_profile, RADII)

        w_local = estimate_width(local_profile, RADII)
        w_global = estimate_width(global_profile, RADII)
        w_total = estimate_width(total_profile, RADII)

        _, _, mse_exp, lambda_eff = fit_exp_tail(RADII, total_profile)

        rows.append({
            "step": step,
            "time": step * args.dt,
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

        if step in snap_steps:
            snapshots[step] = {
                "rho": rho_eff.copy(),
                "psi_total": psi_total.copy(),
                "psi_global": psi_global_mem.copy(),
            }

        if step % 60 == 0:
            print(
                f"step={step} "
                f"mean_rho={np.mean(rho_eff):.6e} "
                f"a_total={a_total:.6f} "
                f"lambda_eff={lambda_eff:.6f}"
            )

    hist = pd.DataFrame(rows)
    hist["H_total"] = np.gradient(hist["a_total"].values, args.dt) / np.maximum(hist["a_total"].values, 1e-12)
    hist["H_global"] = np.gradient(hist["a_global"].values, args.dt) / np.maximum(hist["a_global"].values, 1e-12)
    hist["m_eff"] = 1.0 / np.maximum(hist["lambda_eff"].values, 1e-12)

    hist.to_csv(OUT / "cosmic_mantle_expansion_history.csv", index=False)

    final = hist.iloc[-1]

    # Figure 1
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["time"], hist["a_local"], label="a_local")
    plt.plot(hist["time"], hist["a_global"], label="a_global")
    plt.plot(hist["time"], hist["a_total"], label="a_total")
    plt.xlabel("time")
    plt.ylabel("pseudo scale factor")
    plt.title("Cosmic mantle pseudo-scale evolution")
    plt.legend()
    plt.tight_layout()
    fig1 = FIG_OUT / "app_cosmic_scale_factor.png"
    plt.savefig(fig1, dpi=220)
    plt.close()

    # Figure 2
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["time"], hist["H_global"], label="H_global")
    plt.plot(hist["time"], hist["H_total"], label="H_total")
    plt.xlabel("time")
    plt.ylabel("pseudo Hubble rate")
    plt.title("Cosmic mantle pseudo-expansion rate")
    plt.legend()
    plt.tight_layout()
    fig2 = FIG_OUT / "app_cosmic_hubble_rate.png"
    plt.savefig(fig2, dpi=220)
    plt.close()

    # Figure 3
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["time"], hist["lambda_eff"], label="lambda_eff")
    plt.plot(hist["time"], hist["m_eff"], label="m_eff")
    plt.xlabel("time")
    plt.ylabel("effective range / screening")
    plt.title("Effective screening evolution")
    plt.legend()
    plt.tight_layout()
    fig3 = FIG_OUT / "app_cosmic_screening_evolution.png"
    plt.savefig(fig3, dpi=220)
    plt.close()

    # Figure 4
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for ax, step in zip(axes, snap_steps):
        ax.imshow(snapshots[step]["psi_total"], cmap="coolwarm")
        ax.set_title(f"step={step}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig4 = FIG_OUT / "app_cosmic_geometry_snapshots.png"
    plt.savefig(fig4, dpi=220)
    plt.close(fig)

    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print(f"[OK] wrote {fig3}")
    print(f"[OK] wrote {fig4}")
    print("\n=== COSMIC MANTLE REPORT ===")
    print(f"a_total_final={float(final['a_total']):.6f}")
    print(f"H_total_final={float(final['H_total']):.6f}")
    print(f"lambda_eff_final={float(final['lambda_eff']):.6f}")
    print(f"m_eff_final={float(final['m_eff']):.6f}")
    print(f"a_total_mean={float(np.nanmean(hist['a_total'])):.6f}")
    print(f"H_total_mean={float(np.nanmean(hist['H_total'])):.6f}")
    print(f"lambda_eff_mean={float(np.nanmean(hist['lambda_eff'])):.6f}")
    print(f"m_eff_mean={float(np.nanmean(hist['m_eff'])):.6f}")
    print(f"exp_tail_mse_mean={float(np.nanmean(hist['exp_tail_mse'])):.6f}")
    print(f"mean_abs_psi_total_final={float(final['mean_abs_psi_total']):.6f}")
    print(f"max_rho_final={float(final['max_rho']):.6f}")
    print("[DONE] cosmic mantle expansion app complete")


if __name__ == "__main__":
    main()