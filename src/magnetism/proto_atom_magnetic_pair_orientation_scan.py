#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results/magnetism/proto_atom_magnetic_pair_orientation_scan")
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 180
N_STEPS = 260
DT = 0.08
WAVE_GAIN = 0.24
DAMP = 0.996

# fixed positions
CENTER = np.array([SIZE//2, SIZE//2])
SHIFT = 30

# tested orientations (in degrees)
ANGLE_CASES = [
    (0, 0),
    (0, 180),
    (0, 90),
    (45, -45),
]

# --------------------------------------------------

def laplacian(phi):
    return (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0)
        + np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1)
        - 4.0 * phi
    )

def build_oriented_source(pos, angle_deg):
    angle = np.radians(angle_deg)
    dx = np.cos(angle)
    dy = np.sin(angle)

    src = np.zeros((SIZE, SIZE))
    y, x = np.indices((SIZE, SIZE))

    # oriented dipole
    src += np.exp(-((x-(pos[0]+dx*3))**2 + (y-(pos[1]+dy*3))**2)/6)
    src -= np.exp(-((x-(pos[0]-dx*3))**2 + (y-(pos[1]-dy*3))**2)/6)

    return src

def compute_E(phi):
    gy, gx = np.gradient(phi)
    return -gx, -gy

def compute_curl(Fx, Fy):
    dFy_dx = np.gradient(Fy, axis=1)
    dFx_dy = np.gradient(Fx, axis=0)
    return dFy_dx - dFx_dy

def compute_alignment(Fx, Fy):
    norm = np.sqrt(Fx**2 + Fy**2) + 1e-12
    return np.mean(np.abs(Fx/norm))  # proxy simple

# --------------------------------------------------

def run_case(angleA, angleB):

    phi = np.zeros((SIZE, SIZE))
    psi = np.zeros_like(phi)

    posA = CENTER + np.array([-SHIFT, 0])
    posB = CENTER + np.array([SHIFT, 0])

    align_hist = []
    curl_hist = []

    for step in range(N_STEPS):

        src = np.zeros_like(phi)

        src += build_oriented_source(posA, angleA) * np.sin(0.1*step)
        src += build_oriented_source(posB, angleB) * np.sin(0.1*step + np.pi/2)

        lap = laplacian(phi)

        psi = DAMP * psi + DT * (WAVE_GAIN * lap + src)
        phi = phi + DT * psi

        Fx, Fy = compute_E(phi)
        curl = compute_curl(Fx, Fy)

        align_hist.append(compute_alignment(Fx, Fy))
        curl_hist.append(np.mean(np.abs(curl)))

    return {
        "angleA": angleA,
        "angleB": angleB,
        "align_mean": float(np.mean(align_hist)),
        "curl_mean": float(np.mean(curl_hist)),
        "align_final": float(align_hist[-1]),
        "curl_final": float(curl_hist[-1]),
    }

# --------------------------------------------------

def main():
    print("\n=== MAGNETIC PAIR ORIENTATION SCAN ===")

    results = []

    for a, b in ANGLE_CASES:
        print(f"→ case ({a}, {b})")
        res = run_case(a, b)
        results.append(res)

    df = pd.DataFrame(results)
    out_csv = OUT / "orientation_scan_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== RESULTS ===")
    print(df.to_string(index=False))

    # plot
    plt.figure(figsize=(8,5))
    for _, row in df.iterrows():
        label = f"{int(row['angleA'])}/{int(row['angleB'])}"
        plt.scatter(row["align_final"], row["curl_final"], label=label, s=80)

    plt.xlabel("alignment")
    plt.ylabel("curl")
    plt.title("Magnetic orientation scan")
    plt.legend()
    plt.tight_layout()

    fig = OUT / "orientation_scan_plot.png"
    plt.savefig(fig, dpi=260)
    plt.close()

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {fig}")
    print("[DONE] orientation scan complete")

# --------------------------------------------------

if __name__ == "__main__":
    main()