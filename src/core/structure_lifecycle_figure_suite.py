#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# GLOBAL PARAMETERS
# ============================================================

NX, NY = 260, 260
DT = 0.06
CENTER = np.array([130.0, 130.0], dtype=float)

# underlying field
C_WAVE = 0.75
FIELD_MASS = 0.0
FIELD_DAMP = 0.001

# local geometry
LOCAL_ITERS = 80
LOCAL_MASS = 0.08
LOCAL_WEIGHT = 1.0

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

def gaussian(pos: np.ndarray, sigma: float = 4.0, amp: float = 1.0) -> np.ndarray:
    y, x = np.indices((NY, NX))
    r2 = (x - pos[0]) ** 2 + (y - pos[1]) ** 2
    return amp * np.exp(-r2 / (2.0 * sigma ** 2))

def solve_poisson_like(source: np.ndarray, n_iters: int, mass: float = 0.0) -> np.ndarray:
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

# ============================================================
# PHYSICAL PROXIES
# ============================================================

def effective_energy_density(phi: np.ndarray, pi: np.ndarray) -> np.ndarray:
    gx, gy = gradient(phi)
    grad2 = gx**2 + gy**2
    return 0.5 * grad2 + 0.5 * (pi**2)

def build_local_geometry(rho_eff: np.ndarray) -> np.ndarray:
    return solve_poisson_like(rho_eff, LOCAL_ITERS, mass=LOCAL_MASS)

def build_global_geometry(rho_eff: np.ndarray, global_sigma: float, global_mass: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    rho_global = gaussian_filter(rho_eff, sigma=global_sigma, mode="nearest")
    return solve_poisson_like(rho_global, LOCAL_ITERS, mass=global_mass)

# ============================================================
# SOURCE BUILDERS
# ============================================================

def make_radial_source(source_sigma: float = 4.0, amp: float = 1.0) -> np.ndarray:
    return gaussian(CENTER, sigma=source_sigma, amp=amp)

def make_split_doublet_source(source_sigma: float = 4.0, sep: float = 14.0, amp: float = 1.0) -> np.ndarray:
    left = CENTER + np.array([-sep / 2.0, 0.0])
    right = CENTER + np.array([sep / 2.0, 0.0])
    return gaussian(left, sigma=source_sigma, amp=amp) + gaussian(right, sigma=source_sigma, amp=amp)

def make_asymmetric_fragment_source(source_sigma: float = 4.0) -> np.ndarray:
    a = gaussian(CENTER + np.array([-9.0, -4.0]), sigma=source_sigma, amp=1.0)
    b = gaussian(CENTER + np.array([10.0, 6.0]), sigma=source_sigma * 0.9, amp=0.72)
    c = gaussian(CENTER + np.array([0.0, 11.0]), sigma=source_sigma * 0.7, amp=0.35)
    return a + b + c

# ============================================================
# EVOLUTION
# ============================================================

def evolve_field(phi: np.ndarray, pi: np.ndarray, src: np.ndarray, extra_damp: float = 0.0):
    total_damp = FIELD_DAMP + extra_damp
    pi = pi + DT * (
        C_WAVE**2 * laplacian(phi)
        - FIELD_MASS**2 * phi
        + src
        - total_damp * pi
    )
    phi = phi + DT * pi
    return phi, pi

# ============================================================
# FIGURE HELPER
# ============================================================

def save_field_figure(arr: np.ndarray, path: Path, title: str):
    plt.figure(figsize=(6.5, 6.0))
    plt.imshow(arr, cmap="coolwarm", origin="upper")
    plt.scatter([CENTER[0]], [CENTER[1]], s=30)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=260)
    plt.close()

# ============================================================
# SCENARIO 1 — FRAGMENTATION
# ============================================================

def run_fragmentation_example():
    print("\n=== FRAGMENTATION EXAMPLE ===")

    phi = np.zeros((NY, NX), dtype=float)
    pi = np.zeros_like(phi)

    src = make_asymmetric_fragment_source(source_sigma=4.0)

    n_steps = 420
    capture_step = 320
    captured = None

    for step in range(n_steps):
        phi, pi = evolve_field(phi, pi, src, extra_damp=0.0005)
        rho_eff = effective_energy_density(phi, pi)
        psi_local = build_local_geometry(rho_eff)
        psi_global = build_global_geometry(rho_eff, global_sigma=6.0, global_mass=0.05)
        psi_total = LOCAL_WEIGHT * psi_local + 0.06 * psi_global

        if step == capture_step:
            captured = psi_total.copy()

        if step % 100 == 0:
            print(f"step={step} mean|psi|={np.mean(np.abs(psi_total)):.6e}")

    if captured is None:
        captured = psi_total

    out = OUT / "fragmentation_example.png"
    save_field_figure(captured, out, "Fragmentation example")
    print(f"[OK] wrote {out}")

# ============================================================
# SCENARIO 2 — DESTRUCTION / COLLAPSE
# ============================================================

def run_destruction_collapse():
    print("\n=== DESTRUCTION COLLAPSE ===")

    phi = np.zeros((NY, NX), dtype=float)
    pi = np.zeros_like(phi)

    src = make_radial_source(source_sigma=3.5, amp=1.0)

    n_steps = 520
    capture_step = 500
    captured = None

    for step in range(n_steps):
        # brutal over-coupling + wide mantle
        phi, pi = evolve_field(phi, pi, src, extra_damp=0.012)
        rho_eff = effective_energy_density(phi, pi)
        psi_local = build_local_geometry(rho_eff)
        psi_global = build_global_geometry(rho_eff, global_sigma=28.0, global_mass=0.0)
        psi_total = LOCAL_WEIGHT * psi_local + 0.22 * psi_global

        # nonlinear saturation-like collapse
        psi_total = np.tanh(1.4 * psi_total)

        if step == capture_step:
            captured = psi_total.copy()

        if step % 100 == 0:
            print(f"step={step} mean|psi|={np.mean(np.abs(psi_total)):.6e}")

    if captured is None:
        captured = psi_total

    out = OUT / "destruction_collapse.png"
    save_field_figure(captured, out, "Collapse and dissolution")
    print(f"[OK] wrote {out}")

# ============================================================
# SCENARIO 3 — DIFFUSION
# ============================================================

def run_destruction_diffusion():
    print("\n=== DESTRUCTION DIFFUSION ===")

    phi = np.zeros((NY, NX), dtype=float)
    pi = np.zeros_like(phi)

    src = make_radial_source(source_sigma=3.5, amp=0.85)

    n_steps = 520
    capture_step = 500
    captured = None

    for step in range(n_steps):
        # high damping + weak geometry -> homogeneous diffusion
        phi, pi = evolve_field(phi, pi, src, extra_damp=0.03)
        rho_eff = effective_energy_density(phi, pi)
        psi_local = solve_poisson_like(rho_eff, LOCAL_ITERS, mass=0.35)
        psi_total = 0.85 * psi_local

        if step == capture_step:
            captured = psi_total.copy()

        if step % 100 == 0:
            print(f"step={step} mean|psi|={np.mean(np.abs(psi_total)):.6e}")

    if captured is None:
        captured = psi_total

    out = OUT / "destruction_diffusion.png"
    save_field_figure(captured, out, "Diffusion after structure loss")
    print(f"[OK] wrote {out}")

# ============================================================
# SCENARIO 4 — RECONSTRUCTION
# ============================================================

def run_reconstruction_cycle():
    print("\n=== RECONSTRUCTION CYCLE ===")

    phi = np.zeros((NY, NX), dtype=float)
    pi = np.zeros_like(phi)

    src = make_split_doublet_source(source_sigma=3.8, sep=16.0, amp=0.95)

    # Phase 1: partial destruction
    for step in range(220):
        phi, pi = evolve_field(phi, pi, src, extra_damp=0.018)

    # Phase 2: return to stable regime
    n_steps = 320
    capture_step = 300
    captured = None

    for step in range(n_steps):
        phi, pi = evolve_field(phi, pi, src, extra_damp=0.001)
        rho_eff = effective_energy_density(phi, pi)
        psi_local = build_local_geometry(rho_eff)
        psi_global = build_global_geometry(rho_eff, global_sigma=8.0, global_mass=0.10)
        psi_total = LOCAL_WEIGHT * psi_local + 0.03 * psi_global

        if step == capture_step:
            captured = psi_total.copy()

        if step % 100 == 0:
            print(f"rebuild_step={step} mean|psi|={np.mean(np.abs(psi_total)):.6e}")

    if captured is None:
        captured = psi_total

    out = OUT / "reconstruction_cycle.png"
    save_field_figure(captured, out, "Reconstruction after dissolution")
    print(f"[OK] wrote {out}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n=== STRUCTURE LIFECYCLE FIGURE SUITE ===")
    print("This run generates:")
    print(" - fragmentation_example.png")
    print(" - destruction_collapse.png")
    print(" - destruction_diffusion.png")
    print(" - reconstruction_cycle.png")

    run_fragmentation_example()
    run_destruction_collapse()
    run_destruction_diffusion()
    run_reconstruction_cycle()

    print("\n[DONE] all lifecycle figures generated")

if __name__ == "__main__":
    main()