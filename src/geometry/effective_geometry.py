#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("results/geometry/effective_geometry")
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# PARAMS
# ------------------------------------------------------------
NX, NY = 220, 220
DT = 0.06
N_STEPS = 220

C_WAVE = 0.75
FIELD_MASS = 0.01
FIELD_DAMP = 0.004

SAT_SCALE = 0.08

# ------------------------------------------------------------
def laplacian(phi):
    return (
        np.roll(phi,1,0)+np.roll(phi,-1,0)+
        np.roll(phi,1,1)+np.roll(phi,-1,1)-4*phi
    )

def gradient(arr):
    gy, gx = np.gradient(arr)
    return gx, gy

def saturate(phi):
    return np.tanh(SAT_SCALE * phi)

def gaussian(pos, sigma=2.0):
    y,x = np.indices((NY,NX))
    r2 = (x-pos[0])**2 + (y-pos[1])**2
    return np.exp(-r2/(2*sigma**2))

# ------------------------------------------------------------
def evolve(phi, pi, src):
    pi = pi + DT * (
        C_WAVE**2 * laplacian(phi)
        - FIELD_MASS**2 * phi
        + src
        - FIELD_DAMP * pi
    )
    phi = phi + DT * pi
    return phi, pi

# ------------------------------------------------------------
def effective_geometry(phi):

    phi_sat = saturate(phi)

    gx, gy = gradient(phi_sat)
    grad2 = gx**2 + gy**2

    lap = laplacian(phi_sat)

    # tentative "courbure"
    curvature = lap + 0.5 * grad2

    return curvature, grad2

# ------------------------------------------------------------
def main():

    print("\n=== EFFECTIVE GEOMETRY SCAN ===")

    # two sources
    src = (
        gaussian([90,110], sigma=3) -
        gaussian([130,110], sigma=3)
    )

    phi = np.zeros((NY,NX))
    pi = np.zeros_like(phi)

    history_curv = []

    for step in range(N_STEPS):

        phi, pi = evolve(phi, pi, src)

        curv, grad2 = effective_geometry(phi)

        mean_curv = np.mean(np.abs(curv))
        history_curv.append(mean_curv)

        if step % 40 == 0:
            print(f"step={step} curvature={mean_curv:.6e}")

    # --------------------------------------------------------
    # plots
    # --------------------------------------------------------

    plt.figure(figsize=(5,4))
    plt.imshow(curv, cmap="coolwarm")
    plt.title("Effective curvature field")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUT/"curvature_field.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5,4))
    plt.plot(history_curv)
    plt.title("Mean curvature over time")
    plt.tight_layout()
    plt.savefig(OUT/"curvature_timeseries.png", dpi=200)
    plt.close()

    print("[DONE] effective geometry")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()