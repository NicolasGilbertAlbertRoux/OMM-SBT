#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("results/magnetism/proto_atom_magnetic_alignment")
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 180
N_STEPS = 260
DT = 0.08

# flow parameters
EDGE_DAMP = 0.992
EDGE_DRIVE = 0.20
EDGE_DIFF = 0.10
LOOP_GAIN = 0.30

# proto-atom
CENTER = (SIZE // 2, SIZE // 2)

# mandatory guidance (test)
ORIENTATION_ANGLE = np.pi / 4


def laplacian(a):
    return (
        np.roll(a,1,0)+np.roll(a,-1,0)+
        np.roll(a,1,1)+np.roll(a,-1,1)-4*a
    )


def gradient(a):
    gy, gx = np.gradient(a)
    return gx, gy


def curl_2d(fx, fy):
    dFy_dx = np.gradient(fy, axis=1)
    dFx_dy = np.gradient(fx, axis=0)
    return dFy_dx - dFx_dy


def divergence(fx, fy):
    dFx_dx = np.gradient(fx, axis=1)
    dFy_dy = np.gradient(fy, axis=0)
    return dFx_dx + dFy_dy


def local_loop(fx, fy):
    return fx - np.roll(fy,-1,1) - np.roll(fx,-1,0) + fy


def oriented_seed_field():
    """
    Directional proto-atom:
    controlled directional anisotropy
    """
    y, x = np.indices((SIZE, SIZE))
    cx, cy = CENTER

    dx = x - cx
    dy = y - cy

    r2 = dx**2 + dy**2

    # oriented axis
    ux = np.cos(ORIENTATION_ANGLE)
    uy = np.sin(ORIENTATION_ANGLE)

    projection = dx * ux + dy * uy

    # anisotropic structure
    return np.exp(-r2 / 120) * (1 + 0.6 * projection / (np.sqrt(r2)+1e-6))


def main():
    print("\n=== MAGNETIC ALIGNMENT TEST ===")

    phi = oriented_seed_field()
    psi = np.zeros_like(phi)

    Fx = np.zeros_like(phi)
    Fy = np.zeros_like(phi)

    alignment_hist = []
    curl_hist = []

    for step in range(N_STEPS):

        # simple adjustment of the medium
        psi = 0.996 * psi + DT * (0.25 * laplacian(phi))
        phi = phi + DT * psi

        gx, gy = gradient(phi)

        loopF = local_loop(Fx, Fy)
        glx, gly = gradient(loopF)

        Lx = -gly
        Ly = glx

        Fx = (
            EDGE_DAMP * Fx +
            DT * (EDGE_DRIVE * gx + EDGE_DIFF * laplacian(Fx) + LOOP_GAIN * Lx)
        )

        Fy = (
            EDGE_DAMP * Fy +
            DT * (EDGE_DRIVE * gy + EDGE_DIFF * laplacian(Fy) + LOOP_GAIN * Ly)
        )

        curlF = curl_2d(Fx, Fy)

        # measure alignment of flow ↔ structure
        normF = np.sqrt(Fx**2 + Fy**2) + 1e-12
        normG = np.sqrt(gx**2 + gy**2) + 1e-12

        mask = (normF > 1e-10) & (normG > 1e-10)
        dot = np.zeros_like(Fx)
        dot[mask] = (Fx[mask] * gx[mask] + Fy[mask] * gy[mask]) / (normF[mask] * normG[mask])
        alignment = float(np.mean(dot[mask])) if np.any(mask) else 0.0

        alignment_hist.append(alignment)
        curl_hist.append(np.mean(np.abs(curlF)))

        if step % 40 == 0:
            print(f"step={step} align={alignment:.4f} curl={curl_hist[-1]:.6e}")

    # figure alignment
    plt.figure()
    plt.plot(alignment_hist)
    plt.title("Flux / structure alignment")
    plt.tight_layout()
    plt.savefig(OUT / "alignment.png", dpi=260)
    plt.close()

    # figure curl
    plt.figure()
    plt.plot(curl_hist)
    plt.title("Curl intensity")
    plt.tight_layout()
    plt.savefig(OUT / "curl.png", dpi=260)
    plt.close()

    # final field
    step_q = 6
    yy, xx = np.mgrid[0:SIZE:step_q, 0:SIZE:step_q]

    plt.figure(figsize=(6,6))
    plt.imshow(phi, cmap="coolwarm", alpha=0.6)
    plt.quiver(xx, yy,
               Fx[::step_q,::step_q],
               Fy[::step_q,::step_q],
               color="black")
    plt.title("Magnetic alignment field")
    plt.tight_layout()
    plt.savefig(OUT / "quiver.png", dpi=260)
    plt.close()

    print("[DONE] magnetic alignment test complete")


if __name__ == "__main__":
    main()