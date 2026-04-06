#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

OUT = Path("results/geometry/geodesic_two_scale_geometry")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
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

# best current energy source
ALPHA_GRAD2 = 0.5
BETA_PI2 = 0.5
GAMMA_PHI2 = 0.0

# local component
LOCAL_MODE = "helmholtz"     # "poisson" or "helmholtz"
LOCAL_ITERS = 80
LOCAL_MASS = 0.08            # m small local workforce

# global component (universal mantle)
GLOBAL_SMOOTH_SIGMA = 10.0   # very smooth
GLOBAL_ITERS = 80
GLOBAL_MASS = 0.0            # free by default

# relative weight
EPSILON_GLOBAL = 0.18        # global mantle coupling
LOCAL_WEIGHT = 1.0

# slow global memory
GLOBAL_MEMORY_ETA = 0.03

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
    return np.array(
        [bilinear_sample(gx, pos), bilinear_sample(gy, pos)],
        dtype=float,
    )

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

def build_two_scale_geometry(rho_eff: np.ndarray, psi_global_mem: np.ndarray):
    # local
    if LOCAL_MODE == "poisson":
        psi_local = solve_poisson_like(rho_eff, LOCAL_ITERS, mass=0.0)
    elif LOCAL_MODE == "helmholtz":
        psi_local = solve_poisson_like(rho_eff, LOCAL_ITERS, mass=LOCAL_MASS)
    else:
        raise ValueError(f"Unknown LOCAL_MODE: {LOCAL_MODE}")

    # global / manteau cosmique
    rho_global = gaussian_filter(rho_eff, sigma=GLOBAL_SMOOTH_SIGMA, mode="nearest")
    psi_global_inst = solve_poisson_like(rho_global, GLOBAL_ITERS, mass=GLOBAL_MASS)

    psi_global_mem = (
        (1.0 - GLOBAL_MEMORY_ETA) * psi_global_mem
        + GLOBAL_MEMORY_ETA * psi_global_inst
    )

    psi_total = LOCAL_WEIGHT * psi_local + EPSILON_GLOBAL * psi_global_mem
    return psi_local, psi_global_inst, psi_global_mem, psi_total

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
# MAIN
# ============================================================

def main():
    print("\n=== GEODESIC TWO-SCALE GEOMETRY ===")

    src = gaussian(SRC)

    phi = np.zeros((NY, NX))
    pi = np.zeros_like(phi)

    psi_global_mem = np.zeros((NY, NX))

    particles = []
    for i, y0 in enumerate(IMPACT_Y, start=1):
        particles.append({
            "name": f"TS{i}",
            "y0": float(y0),
            "pos": np.array([X0, y0], dtype=float),
            "vel": np.array([VX0, 0.0], dtype=float),
            "history": [],
            "min_dist_to_center": np.inf,
            "closest_step": None,
        })

    snapshots = {}

    for step in range(N_STEPS):
        phi, pi = evolve_field(phi, pi, src)
        rho_eff = effective_energy_density(phi, pi)

        psi_local, psi_global_inst, psi_global_mem, psi_total = build_two_scale_geometry(
            rho_eff, psi_global_mem
        )

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
            if d < part["min_dist_to_center"]:
                part["min_dist_to_center"] = d
                part["closest_step"] = step

            part["history"].append({
                "step": step,
                "x": pos[0],
                "y": pos[1],
                "vx": vel[0],
                "vy": vel[1],
                "speed": float(np.linalg.norm(vel)),
                "psi_local_sampled": float(bilinear_sample(psi_local, pos)),
                "psi_global_sampled": float(bilinear_sample(psi_global_mem, pos)),
                "psi_total_sampled": float(bilinear_sample(psi_total, pos)),
                "dist_to_center": d,
            })

        if step in SNAP_STEPS:
            snapshots[step] = {
                "psi_local": psi_local.copy(),
                "psi_global": psi_global_mem.copy(),
                "psi_total": psi_total.copy(),
                "particles": [(p["name"], p["pos"].copy()) for p in particles],
            }

        if step % 100 == 0:
            print(
                f"step={step} "
                f"mean_rho={np.mean(rho_eff):.6e} "
                f"mean_|psi_local|={np.mean(np.abs(psi_local)):.6e} "
                f"mean_|psi_global|={np.mean(np.abs(psi_global_mem)):.6e} "
                f"mean_|psi_total|={np.mean(np.abs(psi_total)):.6e}"
            )

    rows = []
    summary_rows = []

    for part in particles:
        dfp = pd.DataFrame(part["history"])
        for row in part["history"]:
            row["particle"] = part["name"]
            rows.append(row)

        v0 = np.array([VX0, 0.0], dtype=float)
        vf = dfp[["vx", "vy"]].iloc[-1].values.astype(float)

        angle0 = np.degrees(np.arctan2(v0[1], v0[0]))
        anglef = np.degrees(np.arctan2(vf[1], vf[0]))
        dtheta = float(anglef - angle0)

        impact = abs(part["y0"] - CENTER[1])

        summary_rows.append({
            "particle": part["name"],
            "impact_param": float(impact),
            "x_final": float(dfp["x"].iloc[-1]),
            "y_final": float(dfp["y"].iloc[-1]),
            "vx_final": float(dfp["vx"].iloc[-1]),
            "vy_final": float(dfp["vy"].iloc[-1]),
            "speed_final": float(dfp["speed"].iloc[-1]),
            "deflection_deg": dtheta,
            "abs_deflection_deg": float(abs(dtheta)),
            "min_dist_to_center": float(part["min_dist_to_center"]),
            "closest_step": int(part["closest_step"]),
            "mean_abs_psi_local": float(np.mean(np.abs(dfp["psi_local_sampled"]))),
            "mean_abs_psi_global": float(np.mean(np.abs(dfp["psi_global_sampled"]))),
            "mean_abs_psi_total": float(np.mean(np.abs(dfp["psi_total_sampled"]))),
        })

    hist_df = pd.DataFrame(rows)
    hist_csv = OUT / "two_scale_geometry_history.csv"
    hist_df.to_csv(hist_csv, index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values("impact_param")
    summary_csv = OUT / "two_scale_geometry_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== TWO-SCALE GEOMETRY SUMMARY ===")
    print(summary_df.to_string(index=False))

    tail_df = summary_df[summary_df["impact_param"] >= TAIL_MIN_IMPACT].copy()
    x = tail_df["impact_param"].values.astype(float)
    y = tail_df["abs_deflection_deg"].values.astype(float) + 1e-12

    fits = fit_models(x, y)

    fit_rows = []
    print("\n=== FIT RESULTS ===")
    for name, data in fits.items():
        print(name, "params=", data["params"], "mse=", data["mse"])
        fit_rows.append({
            "model": name,
            "params": np.array2string(data["params"], precision=8),
            "mse": data["mse"],
        })

    fit_df = pd.DataFrame(fit_rows).sort_values("mse")
    fit_csv = OUT / "two_scale_geometry_fit_models.csv"
    fit_df.to_csv(fit_csv, index=False)

    # figures
    plt.figure(figsize=(6, 6))
    plt.imshow(psi_local, cmap="coolwarm")
    plt.scatter([CENTER[0]], [CENTER[1]], s=45)
    plt.title("Two-scale local geometry")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig1 = OUT / "two_scale_local_geometry.png"
    plt.savefig(fig1, dpi=220)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(psi_global_mem, cmap="coolwarm")
    plt.scatter([CENTER[0]], [CENTER[1]], s=45)
    plt.title("Two-scale global mantle")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig2 = OUT / "two_scale_global_geometry.png"
    plt.savefig(fig2, dpi=220)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(psi_total, cmap="coolwarm")
    plt.scatter([CENTER[0]], [CENTER[1]], s=45)
    plt.title("Two-scale total geometry")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig3 = OUT / "two_scale_total_geometry.png"
    plt.savefig(fig3, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.imshow(psi_total, cmap="coolwarm")
    for part in particles:
        dfp = pd.DataFrame(part["history"])
        plt.plot(dfp["x"], dfp["y"], linewidth=1.0)
    plt.scatter([CENTER[0]], [CENTER[1]], s=45)
    plt.title("Two-scale trajectories")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig4 = OUT / "two_scale_trajectories.png"
    plt.savefig(fig4, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, label="tail data")
    for name, data in fits.items():
        plt.plot(x, data["yfit"], label=f"{name} mse={data['mse']:.2e}")
    plt.xlabel("impact parameter")
    plt.ylabel("deflection")
    plt.title("Two-scale tail fit")
    plt.legend()
    plt.tight_layout()
    fig5 = OUT / "two_scale_tail_fit.png"
    plt.savefig(fig5, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(summary_df["impact_param"], summary_df["mean_abs_psi_local"], marker="o", label="local")
    plt.plot(summary_df["impact_param"], summary_df["mean_abs_psi_global"], marker="o", label="global")
    plt.plot(summary_df["impact_param"], summary_df["mean_abs_psi_total"], marker="o", label="total")
    plt.xlabel("impact parameter")
    plt.ylabel("mean sampled geometry")
    plt.title("Sampled local/global/total geometry")
    plt.legend()
    plt.tight_layout()
    fig6 = OUT / "two_scale_sampled_geometry.png"
    plt.savefig(fig6, dpi=220)
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
    fig7 = OUT / "two_scale_snapshots.png"
    plt.savefig(fig7, dpi=220)
    plt.close(fig)

    print(f"[OK] wrote {hist_csv}")
    print(f"[OK] wrote {summary_csv}")
    print(f"[OK] wrote {fit_csv}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print(f"[OK] wrote {fig3}")
    print(f"[OK] wrote {fig4}")
    print(f"[OK] wrote {fig5}")
    print(f"[OK] wrote {fig6}")
    print(f"[OK] wrote {fig7}")
    print("[DONE] geodesic two-scale geometry complete")

if __name__ == "__main__":
    main()