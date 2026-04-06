#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, label

OUT = Path("results/research_1_6/two_scale_mantle_instability_test")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# GLOBAL SETTINGS
# ============================================================

NX, NY = 260, 260
DT = 0.06
N_STEPS = 420

CENTER = np.array([130.0, 130.0], dtype=float)
SRC = CENTER.copy()

# underlying field
C_WAVE = 0.75
FIELD_MASS = 0.0
FIELD_DAMP = 0.001

# effective energy density
ALPHA_GRAD2 = 0.5
BETA_PI2 = 0.5
GAMMA_PHI2 = 0.0

# local geometry
LOCAL_MODE = "helmholtz"
LOCAL_ITERS = 80
LOCAL_MASS = 0.08
LOCAL_WEIGHT = 1.0

# best two-scale zone found
EPSILON_GLOBAL = 0.02
GLOBAL_SIGMA = 4.0
GLOBAL_ETA = 0.005
GLOBAL_MASS = 0.10

# test particles
X0 = 78.0
VX0 = 0.050
IMPACT_Y = np.array(
    [94.0, 100.0, 106.0, 112.0, 118.0, 124.0, 130.0,
     136.0, 142.0, 148.0, 154.0, 160.0, 166.0],
    dtype=float
)

PARTICLE_DAMP = 0.9995
GEOM_COUPLING = -1.8

TAIL_MIN_IMPACT = 12.0
SNAP_STEPS = [0, 120, 240, 360, N_STEPS - 1]

# ============================================================
# TEST CASE
# ============================================================

CASES = {
    "stable_radial": {
        "instability_mode": "none",
        "anisotropy_amp": 0.0,
        "split_shift": 0.0,
        "ring_mix": 0.0,
    },
    "quadrupolar_soft": {
        "instability_mode": "quadrupole",
        "anisotropy_amp": 0.20,
        "split_shift": 0.0,
        "ring_mix": 0.0,
    },
    "quadrupolar_strong": {
        "instability_mode": "quadrupole",
        "anisotropy_amp": 0.45,
        "split_shift": 0.0,
        "ring_mix": 0.0,
    },
    "split_doublet_soft": {
        "instability_mode": "doublet",
        "anisotropy_amp": 0.0,
        "split_shift": 6.0,
        "ring_mix": 0.0,
    },
    "split_doublet_strong": {
        "instability_mode": "doublet",
        "anisotropy_amp": 0.0,
        "split_shift": 12.0,
        "ring_mix": 0.0,
    },
    "ring_fragmented": {
        "instability_mode": "ring",
        "anisotropy_amp": 0.0,
        "split_shift": 0.0,
        "ring_mix": 0.40,
    },
    "quadrupole_plus_ring": {
        "instability_mode": "quadrupole_ring",
        "anisotropy_amp": 0.30,
        "split_shift": 0.0,
        "ring_mix": 0.30,
    },
}

# ============================================================
# TOOLS
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

def gaussian(pos: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    y, x = np.indices((NY, NX))
    r2 = (x - pos[0])**2 + (y - pos[1])**2
    return np.exp(-r2 / (2.0 * sigma**2))

def bilinear_sample(arr: np.ndarray, pos: np.ndarray) -> float:
    x = float(np.clip(pos[0], 0, NX - 1.001))
    y = float(np.clip(pos[1], 0, NY - 1.001))

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, NX - 1)
    y1 = min(y0 + 1, NY - 1)

    tx = x - x0
    ty = y - y0

    v00 = arr[y0, x0]
    v10 = arr[y0, x1]
    v01 = arr[y1, x0]
    v11 = arr[y1, x1]

    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )

def sample_vector(gx: np.ndarray, gy: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return np.array([bilinear_sample(gx, pos), bilinear_sample(gy, pos)], dtype=float)

# ============================================================
# UNDERLYING FIELD
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

# ============================================================
# GEOMETRY
# ============================================================

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

def build_local_geometry(rho_eff: np.ndarray):
    if LOCAL_MODE == "poisson":
        return solve_poisson_like(rho_eff, LOCAL_ITERS, mass=0.0)
    if LOCAL_MODE == "helmholtz":
        return solve_poisson_like(rho_eff, LOCAL_ITERS, mass=LOCAL_MASS)
    raise ValueError(f"Unknown LOCAL_MODE: {LOCAL_MODE}")

def make_ring_mode(shape, center, sigma):
    y, x = np.indices(shape)
    dx = x - center[0]
    dy = y - center[1]
    r = np.sqrt(dx**2 + dy**2)
    r0 = sigma * 1.8
    return np.exp(-((r - r0)**2) / (2.0 * (0.8 * sigma)**2))

def build_global_instability(rho_eff, cfg, psi_global_mem):
    # global base mantel
    rho_global = gaussian_filter(rho_eff, sigma=GLOBAL_SIGMA, mode="nearest")
    psi_global_base = solve_poisson_like(rho_global, LOCAL_ITERS, mass=GLOBAL_MASS)

    mode = cfg["instability_mode"]

    if mode == "none":
        psi_global_inst = psi_global_base

    elif mode == "quadrupole":
        y, x = np.indices((NY, NX))
        dx = x - CENTER[0]
        dy = y - CENTER[1]
        theta = np.arctan2(dy, dx)
        mod = 1.0 + cfg["anisotropy_amp"] * np.cos(2.0 * theta)
        psi_global_inst = psi_global_base * mod

    elif mode == "doublet":
        rho1 = gaussian_filter(rho_eff, sigma=GLOBAL_SIGMA, mode="nearest")
        rho2 = gaussian_filter(rho_eff, sigma=GLOBAL_SIGMA, mode="nearest")
        shift = cfg["split_shift"]

        psi1 = solve_poisson_like(np.roll(rho1, int(-shift), axis=1), LOCAL_ITERS, mass=GLOBAL_MASS)
        psi2 = solve_poisson_like(np.roll(rho2, int(+shift), axis=1), LOCAL_ITERS, mass=GLOBAL_MASS)
        psi_global_inst = 0.5 * (psi1 + psi2)

    elif mode == "ring":
        ring = make_ring_mode((NY, NX), CENTER, GLOBAL_SIGMA)
        ring = ring / max(np.max(ring), 1e-12)
        psi_global_inst = psi_global_base + cfg["ring_mix"] * np.mean(np.abs(psi_global_base)) * ring

    elif mode == "quadrupole_ring":
        y, x = np.indices((NY, NX))
        dx = x - CENTER[0]
        dy = y - CENTER[1]
        theta = np.arctan2(dy, dx)
        mod = 1.0 + cfg["anisotropy_amp"] * np.cos(2.0 * theta)

        ring = make_ring_mode((NY, NX), CENTER, GLOBAL_SIGMA)
        ring = ring / max(np.max(ring), 1e-12)

        psi_global_inst = psi_global_base * mod + cfg["ring_mix"] * np.mean(np.abs(psi_global_base)) * ring

    else:
        raise ValueError(f"Unknown instability mode: {mode}")

    psi_global_mem = (
        (1.0 - GLOBAL_ETA) * psi_global_mem
        + GLOBAL_ETA * psi_global_inst
    )

    return psi_global_inst, psi_global_mem

# ============================================================
# FITS
# ============================================================

def model_inv(x, a, b):
    return a / (x + b)

def model_inv2(x, a, b):
    return a / (x**2 + b)

def model_exp(x, a, b):
    return a * np.exp(-b * x)

def model_yukawa(x, a, b, c):
    return a * np.exp(-b * x) / (x + c)

def fit_models(x, y):
    models = [
        ("1_over_r", model_inv, [1.0, 1.0]),
        ("1_over_r2", model_inv2, [10.0, 1.0]),
        ("exp", model_exp, [1.0, 0.1]),
        ("yukawa", model_yukawa, [1.0, 0.05, 1.0]),
    ]

    fits = {}
    for name, model, p0 in models:
        try:
            popt, _ = curve_fit(model, x, y, p0=p0, maxfev=40000)
            yfit = model(x, *popt)
            mse = float(np.mean((y - yfit)**2))
            fits[name] = {"params": popt, "yfit": yfit, "mse": mse}
        except Exception:
            pass
    return fits

# ============================================================
# METRICS
# ============================================================

def mantle_fragmentation_score(psi_global):
    abspsi = np.abs(psi_global)
    vmax = float(np.max(abspsi))
    if vmax <= 1e-15:
        return 0, 0.0
    mask = abspsi > (0.35 * vmax)
    lbl, ncomp = label(mask)
    filled_fraction = float(np.mean(mask))
    return int(ncomp), filled_fraction

# ============================================================
# RUN CASE
# ============================================================

def run_case(name, cfg):
    print(f"\n=== CASE: {name} ===")

    src = gaussian(SRC)

    phi = np.zeros((NY, NX))
    pi = np.zeros_like(phi)
    psi_global_mem = np.zeros((NY, NX))

    particles = []
    for i, y0 in enumerate(IMPACT_Y, start=1):
        particles.append({
            "name": f"P{i}",
            "y0": float(y0),
            "pos": np.array([X0, y0], dtype=float),
            "vel": np.array([VX0, 0.0], dtype=float),
            "min_dist_to_center": np.inf,
            "history": [],
        })

    snapshots = {}

    for step in range(N_STEPS):
        phi, pi = evolve_field(phi, pi, src)
        rho_eff = effective_energy_density(phi, pi)

        psi_local = build_local_geometry(rho_eff)
        psi_global_inst, psi_global_mem = build_global_instability(rho_eff, cfg, psi_global_mem)

        psi_total = LOCAL_WEIGHT * psi_local + EPSILON_GLOBAL * psi_global_mem
        gx_geom, gy_geom = gradient(psi_total)

        for part in particles:
            pos = part["pos"]
            vel = part["vel"]

            F_geom = sample_vector(gx_geom, gy_geom, pos)
            vel = PARTICLE_DAMP * vel + DT * GEOM_COUPLING * F_geom
            pos = pos + DT * vel

            pos[0] = np.clip(pos[0], 2, NX - 3)
            pos[1] = np.clip(pos[1], 2, NY - 3)

            part["pos"] = pos
            part["vel"] = vel

            d = float(np.linalg.norm(pos - CENTER))
            part["min_dist_to_center"] = min(part["min_dist_to_center"], d)

            part["history"].append({
                "x": pos[0],
                "y": pos[1],
                "vx": vel[0],
                "vy": vel[1],
                "psi_local": float(bilinear_sample(psi_local, pos)),
                "psi_global": float(bilinear_sample(psi_global_mem, pos)),
                "psi_total": float(bilinear_sample(psi_total, pos)),
            })

        if step in SNAP_STEPS:
            snapshots[step] = {
                "psi_total": psi_total.copy(),
                "particles": [(p["name"], p["pos"].copy()) for p in particles],
            }

        if step % 100 == 0:
            ncomp, frac = mantle_fragmentation_score(psi_global_mem)
            print(
                f"step={step} "
                f"components={ncomp} "
                f"filled={frac:.4f} "
                f"mean|psi_global|={np.mean(np.abs(psi_global_mem)):.6e}"
            )

    rows = []
    for part in particles:
        dfp = pd.DataFrame(part["history"])

        v0 = np.array([VX0, 0.0], dtype=float)
        vf = dfp[["vx", "vy"]].iloc[-1].values.astype(float)

        angle0 = np.degrees(np.arctan2(v0[1], v0[0]))
        anglef = np.degrees(np.arctan2(vf[1], vf[0]))
        dtheta = float(anglef - angle0)

        rows.append({
            "impact_param": abs(part["y0"] - CENTER[1]),
            "abs_deflection_deg": abs(dtheta),
            "min_dist_to_center": float(part["min_dist_to_center"]),
            "mean_abs_psi_local": float(np.mean(np.abs(dfp["psi_local"]))),
            "mean_abs_psi_global": float(np.mean(np.abs(dfp["psi_global"]))),
            "mean_abs_psi_total": float(np.mean(np.abs(dfp["psi_total"]))),
        })

    df = pd.DataFrame(rows).sort_values("impact_param")
    tail_df = df[df["impact_param"] >= TAIL_MIN_IMPACT].copy()

    x = tail_df["impact_param"].values.astype(float)
    y = tail_df["abs_deflection_deg"].values.astype(float) + 1e-12

    fits = fit_models(x, y)
    best_name = min(fits.keys(), key=lambda k: fits[k]["mse"])
    best = fits[best_name]

    ncomp, frac = mantle_fragmentation_score(psi_global_mem)

    plt.figure(figsize=(6, 6))
    plt.imshow(psi_global_mem, cmap="coolwarm")
    plt.scatter([CENTER[0]], [CENTER[1]], s=45)
    plt.title(f"{name}: global mantle")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(OUT / f"{name}_global_mantle.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(psi_total, cmap="coolwarm")
    plt.scatter([CENTER[0]], [CENTER[1]], s=45)
    plt.title(f"{name}: total geometry")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(OUT / f"{name}_total_geometry.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.imshow(psi_total, cmap="coolwarm")
    for part in particles:
        dfp = pd.DataFrame(part["history"])
        plt.plot(dfp["x"], dfp["y"], linewidth=1.0)
    plt.scatter([CENTER[0]], [CENTER[1]], s=45)
    plt.title(f"{name}: trajectories")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(OUT / f"{name}_trajectories.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, label="tail data")
    for fit_name, fit_data in fits.items():
        plt.plot(x, fit_data["yfit"], label=f"{fit_name} mse={fit_data['mse']:.2e}")
    plt.xlabel("impact parameter")
    plt.ylabel("deflection")
    plt.title(f"{name}: tail fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"{name}_tail_fit.png", dpi=220)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for ax, step in zip(axes, SNAP_STEPS):
        ax.imshow(snapshots[step]["psi_total"], cmap="coolwarm")
        for _, pos in snapshots[step]["particles"]:
            ax.scatter(pos[0], pos[1], s=10)
        ax.set_title(f"step={step}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUT / f"{name}_snapshots.png", dpi=220)
    plt.close(fig)

    return {
        "case": name,
        "best_model": best_name,
        "best_mse": float(best["mse"]),
        "tail_mean_deflection": float(np.mean(tail_df["abs_deflection_deg"])),
        "tail_max_deflection": float(np.max(tail_df["abs_deflection_deg"])),
        "tail_mean_min_distance": float(np.mean(tail_df["min_dist_to_center"])),
        "tail_mean_local_sampled": float(np.mean(tail_df["mean_abs_psi_local"])),
        "tail_mean_global_sampled": float(np.mean(tail_df["mean_abs_psi_global"])),
        "tail_mean_total_sampled": float(np.mean(tail_df["mean_abs_psi_total"])),
        "global_components": ncomp,
        "global_filled_fraction": frac,
    }

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n=== TWO-SCALE MANTLE INSTABILITY TEST ===")

    rows = []
    for name, cfg in CASES.items():
        row = run_case(name, cfg)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = OUT / "two_scale_mantle_instability_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))

    plt.figure(figsize=(10, 5))
    x = np.arange(len(df))
    plt.bar(x - 0.25, df["best_mse"], width=0.25, label="best mse")
    plt.bar(x, df["tail_mean_deflection"], width=0.25, label="tail mean deflection")
    plt.bar(x + 0.25, df["global_components"], width=0.25, label="global components")
    plt.xticks(x, df["case"], rotation=25, ha="right")
    plt.title("Mantle instability comparison")
    plt.legend()
    plt.tight_layout()
    out_fig = OUT / "two_scale_mantle_instability_comparison.png"
    plt.savefig(out_fig, dpi=220)
    plt.close()

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_fig}")
    print("[DONE] two-scale mantle instability test complete")

if __name__ == "__main__":
    main()