#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np


IN_PATH = Path("results/final_states/laws/loop_on_background_final.npy")
OUT_DIR = Path("results/laws/flux_triptych")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STEPS = 20
DT = 0.08
LOCAL_PASSES = 1
MESO_PASSES = 4
FLUX_GAIN = 1.0


def smooth_periodic(x: np.ndarray, passes: int = 1) -> np.ndarray:
    out = x.copy()
    ndim = x.ndim
    for _ in range(passes):
        acc = out.copy()
        for ax in range(ndim):
            acc += np.roll(out, 1, axis=ax)
            acc += np.roll(out, -1, axis=ax)
        out = acc / (1 + 2 * ndim)
    return out


def grad_periodic(x: np.ndarray) -> np.ndarray:
    grads = []
    for ax in range(x.ndim):
        g = 0.5 * (np.roll(x, -1, axis=ax) - np.roll(x, 1, axis=ax))
        grads.append(g)
    return np.stack(grads, axis=0)  # (ndim, ...)


def div_periodic(v: np.ndarray) -> np.ndarray:
    # v shape = (ndim, ...)
    out = np.zeros_like(v[0])
    for ax in range(v.shape[0]):
        dv = 0.5 * (np.roll(v[ax], -1, axis=ax) - np.roll(v[ax], 1, axis=ax))
        out += dv
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - m) / s


def build_triptych(phi: np.ndarray):
    """
    Triptych:
    - scalar field phi
    - emergent flux J from multi-scale gradient mismatch
    - matter-node field N from flux convergence + scalar contrast
    """
    phi_local = smooth_periodic(phi, LOCAL_PASSES)
    phi_meso = smooth_periodic(phi, MESO_PASSES)

    g_local = grad_periodic(phi_local)
    g_meso = grad_periodic(phi_meso)

    # emergent flux = local desire - mesoscopic background
    flux = -FLUX_GAIN * (g_local - g_meso)

    div_flux = div_periodic(flux)
    flux_mag = np.sqrt(np.sum(flux**2, axis=0))

    # node field = strong convergence + structured scalar contrast
    convergence = np.maximum(zscore(-div_flux), 0.0)
    contrast = np.maximum(zscore(np.abs(phi - phi_meso)), 0.0)
    node_field = convergence * contrast

    return flux, flux_mag, node_field, div_flux


def evolve_phi(phi: np.ndarray, div_flux: np.ndarray) -> np.ndarray:
    # conservative continuity-like update
    out = phi - DT * div_flux

    # keep bounded but do not force any shape
    out = out - np.mean(out)
    scale = np.std(out)
    if scale > 1e-12:
        out = out / scale

    out = np.clip(out, -3.0, 3.0)
    return out


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {IN_PATH}")

    phi = np.load(IN_PATH).astype(np.float64)

    print("\n=== FLUX TRIPTYCH TEST ===")

    for step in range(N_STEPS + 1):
        flux, flux_mag, node_field, div_flux = build_triptych(phi)

        print(
            f"step={step} "
            f"phi_std={phi.std():.4f} "
            f"flux_mean={flux_mag.mean():.4f} "
            f"flux_max={flux_mag.max():.4f} "
            f"node_mean={node_field.mean():.4f} "
            f"node_max={node_field.max():.4f}"
        )

        if step < N_STEPS:
            phi = evolve_phi(phi, div_flux)

    np.save(OUT_DIR / "triptych_phi_final.npy", phi)
    np.save(OUT_DIR / "triptych_flux_mag_final.npy", flux_mag)
    np.save(OUT_DIR / "triptych_node_field_final.npy", node_field)
    np.save(OUT_DIR / "triptych_div_flux_final.npy", div_flux)

    print("\n[OK] saved triptych_phi_final.npy")
    print("[OK] saved triptych_flux_mag_final.npy")
    print("[OK] saved triptych_node_field_final.npy")
    print("[OK] saved triptych_div_flux_final.npy")
    print("[DONE] flux triptych test complete.")


if __name__ == "__main__":
    main()