#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results/geometry/geodesic_energy_based_geometry")
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# PARAMS
# ------------------------------------------------------------
NX, NY = 260, 260
DT = 0.06
N_STEPS = 420

C_WAVE = 0.75
FIELD_MASS = 0.0
FIELD_DAMP = 0.001

SAT_SCALE = 0.08

CENTER = np.array([130.0, 130.0])
SRC = CENTER.copy()

# test particles
TEST_PARTICLES = [
    {"name": "E1", "pos": np.array([78.0,  94.0]), "vel": np.array([0.050, 0.000])},
    {"name": "E2", "pos": np.array([78.0, 106.0]), "vel": np.array([0.050, 0.000])},
    {"name": "E3", "pos": np.array([78.0, 118.0]), "vel": np.array([0.050, 0.000])},
    {"name": "E4", "pos": np.array([78.0, 130.0]), "vel": np.array([0.050, 0.000])},
    {"name": "E5", "pos": np.array([78.0, 142.0]), "vel": np.array([0.050, 0.000])},
    {"name": "E6", "pos": np.array([78.0, 154.0]), "vel": np.array([0.050, 0.000])},
    {"name": "E7", "pos": np.array([78.0, 166.0]), "vel": np.array([0.050, 0.000])},
]

PARTICLE_DAMP = 0.9995
GEOM_COUPLING = -1.8

# effective energy density
ALPHA_GRAD2 = 1.0
BETA_PI2 = 1.0
GAMMA_PHI2 = 0.2

POISSON_RELAX_ITERS = 140

SNAP_STEPS = [0, 100, 200, 300, N_STEPS - 1]

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

def evolve(phi, pi, src):
    pi = pi + DT * (
        C_WAVE**2 * laplacian(phi)
        - FIELD_MASS**2 * phi
        + src
        - FIELD_DAMP * pi
    )
    phi = phi + DT * pi
    return phi, pi

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

# ------------------------------------------------------------
# EFFECTIVE ENERGY DENSITY
# ------------------------------------------------------------
def effective_energy_density(phi, pi):
    gx, gy = gradient(phi)
    grad2 = gx**2 + gy**2
    rho = (
        ALPHA_GRAD2 * grad2
        + BETA_PI2 * (pi**2)
        + GAMMA_PHI2 * (phi**2)
    )
    return rho, grad2

# ------------------------------------------------------------
# POISSON RESOLUTION : ∇² Φ = rho
# ------------------------------------------------------------
def solve_poisson_potential(rho, n_iters=POISSON_RELAX_ITERS):
    """
    Résolution simple par relaxation de Jacobi :
    ∇² Φ = rho  ->  Φ = moyenne_voisins - rho/4
    """
    pot = np.zeros_like(rho)

    for _ in range(n_iters):
        pot_new = 0.25 * (
            np.roll(pot, 1, axis=0)
            + np.roll(pot, -1, axis=0)
            + np.roll(pot, 1, axis=1)
            + np.roll(pot, -1, axis=1)
            - rho
        )

        # simple boundary conditions: zero potential
        pot_new[0, :] = 0.0
        pot_new[-1, :] = 0.0
        pot_new[:, 0] = 0.0
        pot_new[:, -1] = 0.0

        pot = pot_new

    return pot

# ------------------------------------------------------------
def main():
    print("\n=== GEODESIC ENERGY-BASED GEOMETRY ===")

    src = gaussian(SRC)

    phi = np.zeros((NY, NX))
    pi = np.zeros_like(phi)

    particles = []
    for p in TEST_PARTICLES:
        particles.append({
            "name": p["name"],
            "pos": p["pos"].copy(),
            "vel": p["vel"].copy(),
            "history": [],
            "min_dist_to_center": np.inf,
            "closest_step": None,
        })

    rho_hist = []
    pot_hist = []
    snapshots = {}

    for step in range(N_STEPS):
        phi, pi = evolve(phi, pi, src)

        rho_eff, grad2 = effective_energy_density(phi, pi)
        phi_geom = solve_poisson_potential(rho_eff)

        gx_geom, gy_geom = gradient(phi_geom)

        rho_hist.append(float(np.mean(rho_eff)))
        pot_hist.append(float(np.mean(np.abs(phi_geom))))

        for part in particles:
            pos = part["pos"]
            vel = part["vel"]

            # geometric force derived from the potential
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
                "geom_potential_local": float(bilinear_sample(phi_geom, pos)),
                "grad2_local": float(bilinear_sample(grad2, pos)),
                "dist_to_center": d_center,
                "force_geom_x": F_geom[0],
                "force_geom_y": F_geom[1],
            })

        if step in SNAP_STEPS:
            snapshots[step] = {
                "rho_eff": rho_eff.copy(),
                "phi_geom": phi_geom.copy(),
                "particles": [(p["name"], p["pos"].copy()) for p in particles],
            }

        if step % 100 == 0:
            print(
                f"step={step} "
                f"mean_rho={rho_hist[-1]:.6e} "
                f"mean_|Phi_geom|={pot_hist[-1]:.6e}"
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

        v0 = np.array(TEST_PARTICLES[[pp["name"] for pp in TEST_PARTICLES].index(part["name"])]["vel"], dtype=float)
        vf = dfp[["vx", "vy"]].iloc[-1].values.astype(float)

        angle0 = np.degrees(np.arctan2(v0[1], v0[0]))
        anglef = np.degrees(np.arctan2(vf[1], vf[0]))
        deflection_deg = anglef - angle0

        impact_param = abs(TEST_PARTICLES[[pp["name"] for pp in TEST_PARTICLES].index(part["name"])]["pos"][1] - CENTER[1])

        summary_rows.append({
            "particle": part["name"],
            "impact_param": float(impact_param),
            "x_final": float(dfp["x"].iloc[-1]),
            "y_final": float(dfp["y"].iloc[-1]),
            "vx_final": float(dfp["vx"].iloc[-1]),
            "vy_final": float(dfp["vy"].iloc[-1]),
            "speed_final": float(dfp["speed"].iloc[-1]),
            "deflection_deg": float(deflection_deg),
            "abs_deflection_deg": float(abs(deflection_deg)),
            "min_dist_to_center": float(part["min_dist_to_center"]),
            "closest_step": int(part["closest_step"]),
            "mean_abs_rho_sampled": float(np.mean(np.abs(dfp["rho_local"]))),
            "mean_abs_phi_geom_sampled": float(np.mean(np.abs(dfp["geom_potential_local"]))),
            "max_abs_phi_geom_sampled": float(np.max(np.abs(dfp["geom_potential_local"]))),
        })

    hist_df = pd.DataFrame(rows)
    hist_csv = OUT / "energy_based_geometry_history.csv"
    hist_df.to_csv(hist_csv, index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values("impact_param")
    summary_csv = OUT / "energy_based_geometry_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== ENERGY-BASED GEOMETRY SUMMARY ===")
    print(summary_df.to_string(index=False))

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    # final energy density
    plt.figure(figsize=(6, 6))
    plt.imshow(rho_eff, cmap="inferno")
    plt.scatter([CENTER[0]], [CENTER[1]], s=60)
    plt.title("Effective energy density")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig1 = OUT / "effective_energy_density.png"
    plt.savefig(fig1, dpi=220)
    plt.close()

    # final geometric potential
    plt.figure(figsize=(6, 6))
    plt.imshow(phi_geom, cmap="coolwarm")
    plt.scatter([CENTER[0]], [CENTER[1]], s=60)
    plt.title("Geometric potential from energy density")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig2 = OUT / "geometric_potential.png"
    plt.savefig(fig2, dpi=220)
    plt.close()

    # trajectories on a potential
    plt.figure(figsize=(7, 7))
    plt.imshow(phi_geom, cmap="coolwarm")
    for part in particles:
        dfp = pd.DataFrame(part["history"])
        plt.plot(dfp["x"], dfp["y"], label=part["name"])
        plt.scatter(dfp["x"].iloc[0], dfp["y"].iloc[0], s=14)
        plt.scatter(dfp["x"].iloc[-1], dfp["y"].iloc[-1], s=14)
    plt.scatter([CENTER[0]], [CENTER[1]], s=60)
    plt.title("Trajectories on energy-based geometric potential")
    plt.legend(ncol=2, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig3 = OUT / "energy_based_trajectories.png"
    plt.savefig(fig3, dpi=220)
    plt.close()

    # deflection vs. impact
    plt.figure(figsize=(7, 4.5))
    plt.plot(summary_df["impact_param"], summary_df["abs_deflection_deg"], marker="o")
    plt.xlabel("impact parameter")
    plt.ylabel("|deflection| (deg)")
    plt.title("Energy-based deflection vs impact")
    plt.tight_layout()
    fig4 = OUT / "energy_based_deflection_vs_impact.png"
    plt.savefig(fig4, dpi=220)
    plt.close()

    # perceived potential vs. impact
    plt.figure(figsize=(7, 4.5))
    plt.plot(summary_df["impact_param"], summary_df["mean_abs_phi_geom_sampled"], marker="o")
    plt.xlabel("impact parameter")
    plt.ylabel("mean |Phi_geom sampled|")
    plt.title("Sampled geometric potential vs impact")
    plt.tight_layout()
    fig5 = OUT / "energy_based_sampled_potential_vs_impact.png"
    plt.savefig(fig5, dpi=220)
    plt.close()

    # snapshots
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for ax, step in zip(axes, SNAP_STEPS):
        ax.imshow(snapshots[step]["phi_geom"], cmap="coolwarm")
        for _, pos in snapshots[step]["particles"]:
            ax.scatter(pos[0], pos[1], s=10)
        ax.set_title(f"step={step}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig6 = OUT / "energy_based_snapshots.png"
    plt.savefig(fig6, dpi=220)
    plt.close(fig)

    print(f"[OK] wrote {hist_csv}")
    print(f"[OK] wrote {summary_csv}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print(f"[OK] wrote {fig3}")
    print(f"[OK] wrote {fig4}")
    print(f"[OK] wrote {fig5}")
    print(f"[OK] wrote {fig6}")
    print("[DONE] geodesic energy-based geometry complete")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()