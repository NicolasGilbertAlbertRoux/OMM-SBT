#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results/magnetism/proto_atom_magnetic_domain_interaction")
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# PARAMS
# ------------------------------------------------------------
NX, NY = 220, 220
DT = 0.06
N_STEPS = 260

C_WAVE = 0.75
FIELD_MASS = 0.003
FIELD_DAMP = 0.002

SAT_SCALE = 0.08

# domain grid
GRID_SIZE = 4
SPACING = 8

# interaction strength
K_COUPLING = -0.05

# ------------------------------------------------------------
def laplacian(phi):
    return (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0)
        + np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1)
        - 4 * phi
    )

def gradient(arr):
    gy, gx = np.gradient(arr)
    return gx, gy

def saturate(phi):
    return np.tanh(SAT_SCALE * phi)

def gaussian(pos, sigma=5.0):
    y, x = np.indices((NY, NX))
    r2 = (x - pos[0])**2 + (y - pos[1])**2
    return np.exp(-r2 / (2 * sigma**2))

def make_domain(center, angle):
    """
    Builds a domain centered around a moving point.
    angle in radians.
    """
    sources = []
    offset = (GRID_SIZE - 1) / 2.0

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            dx = (i - offset) * SPACING
            dy = (j - offset) * SPACING

            x = center[0] + dx
            y = center[1] + dy

            phase = np.cos(angle) * dx + np.sin(angle) * dy
            amp = np.sign(np.sin(phase + 1e-6))

            sources.append((np.array([x, y], dtype=float), amp))

    return sources

def evolve(phi, pi, src):
    pi = pi + DT * (
        C_WAVE**2 * laplacian(phi)
        - FIELD_MASS**2 * phi
        + src
        - FIELD_DAMP * pi
    )
    phi = phi + DT * pi
    return phi, pi

def build_source(sources):
    field = np.zeros((NY, NX))
    for pos, amp in sources:
        field += amp * gaussian(pos)
    return field

def domain_force(phi_other, sources):
    """
    The average force exerted on a domain, obtained by summing the gradients
    of the opposing field over all the dipoles in the domain.
    """
    gx, gy = gradient(saturate(phi_other))
    F = np.zeros(2)

    for pos, amp in sources:
        x = int(np.clip(pos[0], 1, NX - 2))
        y = int(np.clip(pos[1], 1, NY - 2))
        F += amp * np.array([gx[y, x], gy[y, x]])

    return F / len(sources)

# ------------------------------------------------------------
def run_case(angleA, angleB, label):
    posA = np.array([90.0, 110.0], dtype=float)
    posB = np.array([130.0, 110.0], dtype=float)

    phiA = np.zeros((NY, NX))
    piA = np.zeros_like(phiA)

    phiB = np.zeros((NY, NX))
    piB = np.zeros_like(phiB)

    velA = np.zeros(2)
    velB = np.zeros(2)

    history = []

    for step in range(N_STEPS):
        # Rebuilds domains based on current positions
        sourcesA = make_domain(posA, angleA)
        sourcesB = make_domain(posB, angleB)

        srcA = build_source(sourcesA)
        srcB = build_source(sourcesB)

        phiA, piA = evolve(phiA, piA, srcA)
        phiB, piB = evolve(phiB, piB, srcB)

        # Mutual forces
        F_A = domain_force(phiB, sourcesA)
        F_B = domain_force(phiA, sourcesB)

        velA += DT * K_COUPLING * F_A
        velB += DT * K_COUPLING * F_B

        posA += DT * velA
        posB += DT * velB

        # keep within the field
        posA[0] = np.clip(posA[0], 20, NX - 20)
        posA[1] = np.clip(posA[1], 20, NY - 20)
        posB[0] = np.clip(posB[0], 20, NX - 20)
        posB[1] = np.clip(posB[1], 20, NY - 20)

        dvec = posB - posA
        d = np.linalg.norm(dvec)

        ex, ey = dvec / (d + 1e-12)

        # projection onto the cross-domain axis
        F_A_proj = F_A[0] * ex + F_A[1] * ey
        F_B_proj = -(F_B[0] * ex + F_B[1] * ey)

        history.append({
            "step": step,
            "distance": d,
            "force_A_norm": np.linalg.norm(F_A),
            "force_B_norm": np.linalg.norm(F_B),
            "force_A_proj": F_A_proj,
            "force_B_proj": F_B_proj,
            "ax": posA[0],
            "ay": posA[1],
            "bx": posB[0],
            "by": posB[1],
        })

    return pd.DataFrame(history)

# ------------------------------------------------------------
def main():
    print("\n=== MAGNETIC DOMAIN INTERACTION ===")

    cases = {
        "aligned": (0, 0),
        "opposed": (0, np.pi),
        "crossed": (0, np.pi / 2),
    }

    results = {}

    for name, (a, b) in cases.items():
        print(f"→ {name}")
        df = run_case(a, b, name)
        results[name] = df

    # --------------------------------------------------------
    # distance
    # --------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    for name, df in results.items():
        plt.plot(df["step"], df["distance"], label=name)
    plt.legend()
    plt.title("Domain interaction distance")
    plt.xlabel("step")
    plt.ylabel("distance")
    plt.tight_layout()
    plt.savefig(OUT / "distance_comparison.png", dpi=220)
    plt.close()

    # --------------------------------------------------------
    # force projections
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for name, df in results.items():
        plt.plot(df["step"], df["force_A_proj"], label=f"{name} A_proj")
        plt.plot(df["step"], df["force_B_proj"], linestyle="--", label=f"{name} B_proj")
    plt.legend(ncol=2)
    plt.title("Projected domain forces")
    plt.xlabel("step")
    plt.ylabel("projected force")
    plt.tight_layout()
    plt.savefig(OUT / "force_projection_comparison.png", dpi=220)
    plt.close()

    # --------------------------------------------------------
    # trajectories
    # --------------------------------------------------------
    plt.figure(figsize=(6, 6))
    for name, df in results.items():
        plt.plot(df["ax"], df["ay"], label=f"{name} A")
        plt.plot(df["bx"], df["by"], label=f"{name} B")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.legend(fontsize=8)
    plt.title("Domain trajectories")
    plt.tight_layout()
    plt.savefig(OUT / "trajectory_comparison.png", dpi=220)
    plt.close()

    # --------------------------------------------------------
    # summary CSV
    # --------------------------------------------------------
    rows = []
    for name, df in results.items():
        rows.append({
            "case": name,
            "distance_initial": float(df["distance"].iloc[0]),
            "distance_final": float(df["distance"].iloc[-1]),
            "distance_min": float(df["distance"].min()),
            "distance_max": float(df["distance"].max()),
            "mean_force_A_norm": float(df["force_A_norm"].mean()),
            "mean_force_B_norm": float(df["force_B_norm"].mean()),
            "mean_force_A_proj": float(df["force_A_proj"].mean()),
            "mean_force_B_proj": float(df["force_B_proj"].mean()),
            "max_abs_force_A_proj": float(np.max(np.abs(df["force_A_proj"]))),
            "max_abs_force_B_proj": float(np.max(np.abs(df["force_B_proj"]))),
        })

    summary_df = pd.DataFrame(rows)
    summary_csv = OUT / "domain_interaction_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== DOMAIN INTERACTION SUMMARY ===")
    print(summary_df.to_string(index=False))

    print(f"[OK] wrote {OUT / 'distance_comparison.png'}")
    print(f"[OK] wrote {OUT / 'force_projection_comparison.png'}")
    print(f"[OK] wrote {OUT / 'trajectory_comparison.png'}")
    print(f"[OK] wrote {summary_csv}")
    print("[DONE] magnetic domain interaction")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()