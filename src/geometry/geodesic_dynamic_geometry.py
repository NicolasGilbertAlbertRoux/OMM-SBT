#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

OUT = Path("results/geometry/geodesic_dynamic_geometry")
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# PARAMS
# ------------------------------------------------------------
NX, NY = 260, 260
DT = 0.06
N_STEPS = 520

# underlying field
C_WAVE = 0.75
FIELD_MASS = 0.0
FIELD_DAMP = 0.001

# dynamic geometry
C_GEOM = 1.00
GEOM_DAMP = 0.0005
GEOM_MASS = 0.0

CENTER = np.array([130.0, 130.0])
SRC = CENTER.copy()

# particle-geometry coupling
PARTICLE_DAMP = 0.9995
GEOM_COUPLING = -1.8

# effective energy density
ALPHA_GRAD2 = 0.5
BETA_PI2 = 0.5
GAMMA_PHI2 = 0.0

# test particles
IMPACT_Y = np.array([
    94.0, 100.0, 106.0, 112.0, 118.0, 124.0,
    130.0,
    136.0, 142.0, 148.0, 154.0, 160.0, 166.0
], dtype=float)

X0 = 78.0
VX0 = 0.050

TAIL_MIN_IMPACT = 12.0
SNAP_STEPS = [0, 120, 240, 360, N_STEPS - 1]

# ------------------------------------------------------------
def laplacian(phi):
    return (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0)
        + np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1)
        - 4.0 * phi
    )

def gradient(arr):
    gy, gx = np.gradient(arr)
    return gx, gy

def gaussian(pos, sigma=4.0):
    y, x = np.indices((NY, NX))
    r2 = (x - pos[0])**2 + (y - pos[1])**2
    return np.exp(-r2 / (2.0 * sigma**2))

def evolve_field(phi, pi, src):
    pi = pi + DT * (
        C_WAVE**2 * laplacian(phi)
        - FIELD_MASS**2 * phi
        + src
        - FIELD_DAMP * pi
    )
    phi = phi + DT * pi
    return phi, pi

def evolve_geom(psi, chi, rho):
    # dynamic geometry: source wave
    chi = chi + DT * (
        C_GEOM**2 * laplacian(psi)
        - GEOM_MASS**2 * psi
        + rho
        - GEOM_DAMP * chi
    )
    psi = psi + DT * chi
    return psi, chi

def bilinear_sample(arr, pos):
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

def sample_vector(gx, gy, pos):
    return np.array([
        bilinear_sample(gx, pos),
        bilinear_sample(gy, pos),
    ], dtype=float)

def effective_energy_density(phi, pi):
    gx, gy = gradient(phi)
    grad2 = gx**2 + gy**2
    rho = ALPHA_GRAD2 * grad2 + BETA_PI2 * (pi**2) + GAMMA_PHI2 * (phi**2)
    return rho, grad2

# ------------------------------------------------------------
# FITS
# ------------------------------------------------------------
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
            mse = float(np.mean((y - yfit) ** 2))
            fits[name] = {"params": popt, "yfit": yfit, "mse": mse}
        except Exception:
            pass
    return fits

# ------------------------------------------------------------
def main():
    print("\n=== GEODESIC DYNAMIC GEOMETRY ===")

    src = gaussian(SRC)

    # underlying field
    phi = np.zeros((NY, NX))
    pi = np.zeros_like(phi)

    # dynamic geometry
    psi = np.zeros((NY, NX))
    chi = np.zeros_like(psi)

    particles = []
    for i, y0 in enumerate(IMPACT_Y, start=1):
        particles.append({
            "name": f"DG{i}",
            "y0": float(y0),
            "pos": np.array([X0, y0], dtype=float),
            "vel": np.array([VX0, 0.0], dtype=float),
            "history": [],
            "min_dist_to_center": np.inf,
            "closest_step": None,
        })

    rho_hist = []
    psi_hist = []
    snapshots = {}

    for step in range(N_STEPS):
        phi, pi = evolve_field(phi, pi, src)
        rho_eff, grad2 = effective_energy_density(phi, pi)

        psi, chi = evolve_geom(psi, chi, rho_eff)

        gx_geom, gy_geom = gradient(psi)

        rho_hist.append(float(np.mean(rho_eff)))
        psi_hist.append(float(np.mean(np.abs(psi))))

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

            d_center = float(np.linalg.norm(pos - CENTER))
            if d_center < part["min_dist_to_center"]:
                part["min_dist_to_center"] = d_center
                part["closest_step"] = step

            part["history"].append({
                "step": step,
                "x": pos[0],
                "y": pos[1],
                "vx": vel[0],
                "vy": vel[1],
                "speed": float(np.linalg.norm(vel)),
                "rho_local": float(bilinear_sample(rho_eff, pos)),
                "psi_local": float(bilinear_sample(psi, pos)),
                "dist_to_center": d_center,
                "force_geom_x": float(F_geom[0]),
                "force_geom_y": float(F_geom[1]),
            })

        if step in SNAP_STEPS:
            snapshots[step] = {
                "psi": psi.copy(),
                "rho": rho_eff.copy(),
                "particles": [(p["name"], p["pos"].copy()) for p in particles],
            }

        if step % 100 == 0:
            print(
                f"step={step} "
                f"mean_rho={rho_hist[-1]:.6e} "
                f"mean_|psi|={psi_hist[-1]:.6e}"
            )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------
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
            "mean_abs_psi_sampled": float(np.mean(np.abs(dfp["psi_local"]))),
            "max_abs_psi_sampled": float(np.max(np.abs(dfp["psi_local"]))),
        })

    hist_df = pd.DataFrame(rows)
    hist_csv = OUT / "dynamic_geometry_history.csv"
    hist_df.to_csv(hist_csv, index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values("impact_param")
    summary_csv = OUT / "dynamic_geometry_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== DYNAMIC GEOMETRY SUMMARY ===")
    print(summary_df.to_string(index=False))

    # --------------------------------------------------------
    # FITS
    # --------------------------------------------------------
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
    fit_csv = OUT / "dynamic_geometry_fit_models.csv"
    fit_df.to_csv(fit_csv, index=False)

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(rho_eff, cmap="inferno")
    plt.scatter([CENTER[0]], [CENTER[1]], s=50)
    plt.title("Effective energy density")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig1 = OUT / "dynamic_geometry_energy_density.png"
    plt.savefig(fig1, dpi=220)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(psi, cmap="coolwarm")
    plt.scatter([CENTER[0]], [CENTER[1]], s=50)
    plt.title("Dynamic geometric field")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig2 = OUT / "dynamic_geometry_field.png"
    plt.savefig(fig2, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.imshow(psi, cmap="coolwarm")
    for part in particles:
        dfp = pd.DataFrame(part["history"])
        plt.plot(dfp["x"], dfp["y"], label=part["name"])
        plt.scatter(dfp["x"].iloc[0], dfp["y"].iloc[0], s=10)
        plt.scatter(dfp["x"].iloc[-1], dfp["y"].iloc[-1], s=10)
    plt.scatter([CENTER[0]], [CENTER[1]], s=50)
    plt.title("Dynamic-geometry trajectories")
    plt.legend(ncol=2, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig3 = OUT / "dynamic_geometry_trajectories.png"
    plt.savefig(fig3, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, label="tail data")
    for name, data in fits.items():
        plt.plot(x, data["yfit"], label=f"{name} mse={data['mse']:.2e}")
    plt.xlabel("impact parameter")
    plt.ylabel("deflection")
    plt.title("Dynamic-geometry tail fit")
    plt.legend()
    plt.tight_layout()
    fig4 = OUT / "dynamic_geometry_tail_fit.png"
    plt.savefig(fig4, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(summary_df["impact_param"], summary_df["mean_abs_psi_sampled"], marker="o")
    plt.xlabel("impact parameter")
    plt.ylabel("mean |psi sampled|")
    plt.title("Sampled dynamic geometry vs impact")
    plt.tight_layout()
    fig5 = OUT / "dynamic_geometry_sampled_field.png"
    plt.savefig(fig5, dpi=220)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for ax, step in zip(axes, SNAP_STEPS):
        ax.imshow(snapshots[step]["psi"], cmap="coolwarm")
        for _, pos in snapshots[step]["particles"]:
            ax.scatter(pos[0], pos[1], s=10)
        ax.set_title(f"step={step}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig6 = OUT / "dynamic_geometry_snapshots.png"
    plt.savefig(fig6, dpi=220)
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
    print("[DONE] geodesic dynamic geometry complete")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()