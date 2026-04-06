#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results/geometry/geodesic_lensing_scan")
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# PARAMS
# ------------------------------------------------------------
NX, NY = 220, 220
DT = 0.06
N_STEPS = 320

C_WAVE = 0.75
FIELD_MASS = 0.01
FIELD_DAMP = 0.004

SAT_SCALE = 0.08

# sources of the geometric field
SRC1 = np.array([90.0, 110.0])
SRC2 = np.array([130.0, 110.0])

# test particles: various impact parameters
TEST_PARTICLES = [
    {"name": "L1", "pos": np.array([72.0, 94.0]),  "vel": np.array([0.055, 0.000])},
    {"name": "L2", "pos": np.array([72.0, 102.0]), "vel": np.array([0.055, 0.000])},
    {"name": "L3", "pos": np.array([72.0, 110.0]), "vel": np.array([0.055, 0.000])},
    {"name": "L4", "pos": np.array([72.0, 118.0]), "vel": np.array([0.055, 0.000])},
    {"name": "L5", "pos": np.array([72.0, 126.0]), "vel": np.array([0.055, 0.000])},
]

# geometric solution
GEOM_COUPLING = -2.4
PARTICLE_DAMP = 0.9995

# geometric region of interest
GEOM_CENTER = np.array([110.0, 110.0])
GEOM_RADIUS = 26.0

SNAP_STEPS = [0, 60, 120, 180, 240, N_STEPS - 1]

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

def saturate(phi):
    return np.tanh(SAT_SCALE * phi)

def gaussian(pos, sigma=3.0):
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

def effective_geometry(phi):
    phi_sat = saturate(phi)
    gx, gy = gradient(phi_sat)
    grad2 = gx**2 + gy**2
    lap = laplacian(phi_sat)

    # effective curvature proxy
    curvature = lap + 0.5 * grad2
    return curvature, grad2

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
def main():
    print("\n=== GEODESIC LENSING SCAN ===")

    # fixed dipole field
    src = gaussian(SRC1) - gaussian(SRC2)

    phi = np.zeros((NY, NX))
    pi = np.zeros_like(phi)

    # init particles
    particles = []
    for p in TEST_PARTICLES:
        particles.append({
            "name": p["name"],
            "pos": p["pos"].copy(),
            "vel": p["vel"].copy(),
            "history": [],
            "min_dist_to_center": np.inf,
            "time_in_zone": 0,
        })

    curvature_hist = []
    snapshots = {}

    for step in range(N_STEPS):
        phi, pi = evolve(phi, pi, src)

        curvature, grad2 = effective_geometry(phi)
        gx_curv, gy_curv = gradient(curvature)

        curvature_hist.append(float(np.mean(np.abs(curvature))))

        for part in particles:
            pos = part["pos"]
            vel = part["vel"]

            # geometric force
            F_geom = sample_vector(gx_curv, gy_curv, pos)

            vel = PARTICLE_DAMP * vel + DT * GEOM_COUPLING * F_geom
            pos = pos + DT * vel

            pos[0] = np.clip(pos[0], 2, NX - 3)
            pos[1] = np.clip(pos[1], 2, NY - 3)

            part["pos"] = pos
            part["vel"] = vel

            d_center = float(np.linalg.norm(pos - GEOM_CENTER))
            part["min_dist_to_center"] = min(part["min_dist_to_center"], d_center)
            if d_center <= GEOM_RADIUS:
                part["time_in_zone"] += 1

            part["history"].append({
                "step": step,
                "x": pos[0],
                "y": pos[1],
                "vx": vel[0],
                "vy": vel[1],
                "speed": float(np.linalg.norm(vel)),
                "curvature_local": float(bilinear_sample(curvature, pos)),
                "grad2_local": float(bilinear_sample(grad2, pos)),
                "force_geom_x": F_geom[0],
                "force_geom_y": F_geom[1],
                "dist_to_center": d_center,
            })

        if step in SNAP_STEPS:
            snapshots[step] = {
                "curvature": curvature.copy(),
                "particles": [(p["name"], p["pos"].copy()) for p in particles],
            }

        if step % 60 == 0:
            print(f"step={step} mean_curvature={curvature_hist[-1]:.6e}")

    # --------------------------------------------------------
    # detailed export history
    # --------------------------------------------------------
    rows = []
    summary_rows = []

    for p in particles:
        dfp = pd.DataFrame(p["history"])
        for row in p["history"]:
            row["particle"] = p["name"]
            rows.append(row)

        v0 = np.array(TEST_PARTICLES[[pp["name"] for pp in TEST_PARTICLES].index(p["name"])]["vel"], dtype=float)
        vf = dfp[["vx", "vy"]].iloc[-1].values.astype(float)

        angle0 = np.degrees(np.arctan2(v0[1], v0[0]))
        anglef = np.degrees(np.arctan2(vf[1], vf[0]))
        deflection_deg = anglef - angle0

        summary_rows.append({
            "particle": p["name"],
            "x_initial": float(TEST_PARTICLES[[pp["name"] for pp in TEST_PARTICLES].index(p["name"])]["pos"][0]),
            "y_initial": float(TEST_PARTICLES[[pp["name"] for pp in TEST_PARTICLES].index(p["name"])]["pos"][1]),
            "x_final": float(dfp["x"].iloc[-1]),
            "y_final": float(dfp["y"].iloc[-1]),
            "vx_final": float(dfp["vx"].iloc[-1]),
            "vy_final": float(dfp["vy"].iloc[-1]),
            "speed_final": float(dfp["speed"].iloc[-1]),
            "deflection_deg": float(deflection_deg),
            "min_dist_to_center": float(p["min_dist_to_center"]),
            "time_in_zone_steps": int(p["time_in_zone"]),
            "mean_abs_curvature_sampled": float(np.mean(np.abs(dfp["curvature_local"]))),
            "max_abs_curvature_sampled": float(np.max(np.abs(dfp["curvature_local"]))),
        })

    hist_df = pd.DataFrame(rows)
    hist_csv = OUT / "geodesic_lensing_particle_history.csv"
    hist_df.to_csv(hist_csv, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT / "geodesic_lensing_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    # trajectories on a curvature map
    plt.figure(figsize=(7, 7))
    plt.imshow(curvature, cmap="coolwarm")
    for p in particles:
        dfp = pd.DataFrame(p["history"])
        plt.plot(dfp["x"], dfp["y"], label=p["name"])
        plt.scatter(dfp["x"].iloc[0], dfp["y"].iloc[0], s=35)
        plt.scatter(dfp["x"].iloc[-1], dfp["y"].iloc[-1], s=35)
    circle = plt.Circle((GEOM_CENTER[0], GEOM_CENTER[1]), GEOM_RADIUS, fill=False, linestyle="--")
    plt.gca().add_patch(circle)
    plt.title("Geodesic-like lensing trajectories")
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig1 = OUT / "geodesic_lensing_trajectories.png"
    plt.savefig(fig1, dpi=220)
    plt.close()

    # angular deflection
    plt.figure(figsize=(7, 4))
    plt.bar(summary_df["particle"], summary_df["deflection_deg"])
    plt.title("Angular deflection by particle")
    plt.tight_layout()
    fig2 = OUT / "geodesic_lensing_deflection.png"
    plt.savefig(fig2, dpi=220)
    plt.close()

    # minimum distance to center
    plt.figure(figsize=(7, 4))
    plt.bar(summary_df["particle"], summary_df["min_dist_to_center"])
    plt.title("Minimum distance to geometric center")
    plt.tight_layout()
    fig3 = OUT / "geodesic_lensing_min_distance.png"
    plt.savefig(fig3, dpi=220)
    plt.close()

    # time in zone
    plt.figure(figsize=(7, 4))
    plt.bar(summary_df["particle"], summary_df["time_in_zone_steps"])
    plt.title("Time spent in curvature zone")
    plt.tight_layout()
    fig4 = OUT / "geodesic_lensing_time_in_zone.png"
    plt.savefig(fig4, dpi=220)
    plt.close()

    # snapshots
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for ax, step in zip(axes, snapshots.keys()):
        ax.imshow(snapshots[step]["curvature"], cmap="coolwarm")
        for name, pos in snapshots[step]["particles"]:
            ax.scatter(pos[0], pos[1], s=20)
        ax.set_title(f"step={step}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig5 = OUT / "geodesic_lensing_snapshots.png"
    plt.savefig(fig5, dpi=220)
    plt.close(fig)

    print("\n=== GEODESIC LENSING SUMMARY ===")
    print(summary_df.to_string(index=False))

    print(f"[OK] wrote {hist_csv}")
    print(f"[OK] wrote {summary_csv}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print(f"[OK] wrote {fig3}")
    print(f"[OK] wrote {fig4}")
    print(f"[OK] wrote {fig5}")
    print("[DONE] geodesic lensing scan complete")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()