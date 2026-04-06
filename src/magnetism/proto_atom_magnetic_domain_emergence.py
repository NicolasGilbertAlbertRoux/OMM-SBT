#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results/magnetism/proto_atom_magnetic_domain_emergence")
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 220
N_STEPS = 320
DT = 0.08

# support field
PHI_GAIN = 0.24
PHI_DAMP = 0.996

# directional flow
EDGE_DAMP = 0.992
EDGE_DRIVE = 0.22
EDGE_DIFF = 0.10
LOOP_GAIN = 0.32

# structures
N_STRUCTURES = 16
SEED_SPACING = 28
SOURCE_AMPLITUDE = 1.0
OMEGA = 0.12

SNAP_STEPS = [0, 40, 80, 140, 220, N_STEPS - 1]

# two regimes to compare
CASES = {
    "aligned_domain": "aligned",
    "random_domain": "random",
}


def laplacian(arr):
    return (
        np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1)
        - 4.0 * arr
    )


def gradient(arr):
    gy, gx = np.gradient(arr)
    return gx, gy


def divergence(fx, fy):
    return np.gradient(fx, axis=1) + np.gradient(fy, axis=0)


def curl_2d(fx, fy):
    return np.gradient(fy, axis=1) - np.gradient(fx, axis=0)


def local_loop_intensity(fx, fy):
    top = fx
    bottom = np.roll(fx, -1, axis=0)
    left = fy
    right = np.roll(fy, -1, axis=1)
    return top - right - bottom + left


def build_oriented_source(shape, pos, angle_deg, amplitude=1.0):
    angle = np.radians(angle_deg)
    ux = np.cos(angle)
    uy = np.sin(angle)

    y, x = np.indices(shape)
    x0, y0 = pos

    # dipôle orienté minimal
    s_plus = np.exp(-((x - (x0 + 3.0 * ux))**2 + (y - (y0 + 3.0 * uy))**2) / 8.0)
    s_minus = np.exp(-((x - (x0 - 3.0 * ux))**2 + (y - (y0 - 3.0 * uy))**2) / 8.0)

    return amplitude * (s_plus - s_minus)


def structure_positions():
    cx = SIZE // 2
    cy = SIZE // 2

    offsets = [-1.5, -0.5, 0.5, 1.5]
    positions = []

    for iy in offsets:
        for ix in offsets:
            x = cx + int(ix * SEED_SPACING)
            y = cy + int(iy * SEED_SPACING)
            positions.append((x, y))

    return positions[:N_STRUCTURES]


def structure_angles(mode):
    if mode == "aligned":
        # a bit of noise surrounding a dominant trend
        base = 35.0
        rng = np.random.default_rng(1234)
        return [base + rng.normal(0, 6) for _ in range(N_STRUCTURES)]

    if mode == "random":
        rng = np.random.default_rng(5678)
        return list(rng.uniform(0, 180, size=N_STRUCTURES))

    raise ValueError(f"Unknown mode: {mode}")


def global_flux_alignment(Fx, Fy, angles_deg):
    """
    Simple measurement:
    the average projection of the flow onto the average direction of the structures.
    """
    mean_angle = np.radians(np.mean(angles_deg))
    ux = np.cos(mean_angle)
    uy = np.sin(mean_angle)

    normF = np.sqrt(Fx**2 + Fy**2) + 1e-12
    proj = (Fx * ux + Fy * uy) / normF

    mask = normF > 1e-8
    return float(np.mean(np.abs(proj[mask]))) if np.any(mask) else 0.0


def run_case(case_name, mode):
    print(f"\n=== CASE: {case_name} ===")

    positions = structure_positions()
    angles = structure_angles(mode)

    phi = np.zeros((SIZE, SIZE), dtype=float)
    psi = np.zeros_like(phi)

    Fx = np.zeros_like(phi)
    Fy = np.zeros_like(phi)

    div_hist = []
    curl_hist = []
    loop_hist = []
    align_hist = []

    snapshots = {}

    for step in range(N_STEPS):
        source = np.zeros_like(phi)

        # all structures oscillate with a small common modulation
        for pos, ang in zip(positions, angles):
            source += build_oriented_source(
                phi.shape,
                pos,
                ang,
                amplitude=SOURCE_AMPLITUDE * np.sin(OMEGA * step)
            )

        # support field
        psi = PHI_DAMP * psi + DT * (PHI_GAIN * laplacian(phi) + source)
        phi = phi + DT * psi

        gx, gy = gradient(phi)

        loopF = local_loop_intensity(Fx, Fy)
        glx, gly = gradient(loopF)

        Lx = -gly
        Ly = glx

        Fx = (
            EDGE_DAMP * Fx
            + DT * (
                EDGE_DRIVE * gx
                + EDGE_DIFF * laplacian(Fx)
                + LOOP_GAIN * Lx
            )
        )

        Fy = (
            EDGE_DAMP * Fy
            + DT * (
                EDGE_DRIVE * gy
                + EDGE_DIFF * laplacian(Fy)
                + LOOP_GAIN * Ly
            )
        )

        divF = divergence(Fx, Fy)
        curlF = curl_2d(Fx, Fy)
        loopF = local_loop_intensity(Fx, Fy)

        div_mean = float(np.mean(np.abs(divF)))
        curl_mean = float(np.mean(np.abs(curlF)))
        loop_mean = float(np.mean(np.abs(loopF)))
        align = global_flux_alignment(Fx, Fy, angles)

        div_hist.append(div_mean)
        curl_hist.append(curl_mean)
        loop_hist.append(loop_mean)
        align_hist.append(align)

        if step in SNAP_STEPS:
            snapshots[step] = {
                "phi": phi.copy(),
                "divF": divF.copy(),
                "curlF": curlF.copy(),
                "loopF": loopF.copy(),
                "Fx": Fx.copy(),
                "Fy": Fy.copy(),
            }

        if step % 40 == 0:
            print(
                f"step={step} "
                f"div={div_mean:.6e} "
                f"curl={curl_mean:.6e} "
                f"loop={loop_mean:.6e} "
                f"align={align:.4f}"
            )

    summary = {
        "case": case_name,
        "mode": mode,
        "mean_div": float(np.mean(div_hist)),
        "max_div": float(np.max(div_hist)),
        "mean_curl": float(np.mean(curl_hist)),
        "max_curl": float(np.max(curl_hist)),
        "mean_loop": float(np.mean(loop_hist)),
        "max_loop": float(np.max(loop_hist)),
        "mean_alignment": float(np.mean(align_hist)),
        "final_alignment": float(align_hist[-1]),
        "final_div": float(div_hist[-1]),
        "final_curl": float(curl_hist[-1]),
        "final_loop": float(loop_hist[-1]),
    }

    # temporal figures
    plt.figure(figsize=(10, 5))
    plt.plot(div_hist, label="div")
    plt.plot(curl_hist, label="curl")
    plt.plot(loop_hist, label="loop")
    plt.plot(align_hist, label="alignment")
    plt.legend()
    plt.title(f"{case_name}: divergence / curl / loop / alignment")
    plt.tight_layout()
    plt.savefig(OUT / f"{case_name}_timeseries.png", dpi=260)
    plt.close()

    # snapshots curl
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for ax, step in zip(axes, SNAP_STEPS):
        ax.imshow(snapshots[step]["curlF"], cmap="coolwarm")
        ax.set_title(f"curl step={step}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUT / f"{case_name}_curl_snapshots.png", dpi=260)
    plt.close(fig)

    # snapshots loop
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for ax, step in zip(axes, SNAP_STEPS):
        ax.imshow(snapshots[step]["loopF"], cmap="coolwarm")
        ax.set_title(f"loop step={step}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUT / f"{case_name}_loop_snapshots.png", dpi=260)
    plt.close(fig)

    # quiver final
    Fx_final = snapshots[SNAP_STEPS[-1]]["Fx"]
    Fy_final = snapshots[SNAP_STEPS[-1]]["Fy"]
    phi_final = snapshots[SNAP_STEPS[-1]]["phi"]

    step_q = 6
    yy, xx = np.mgrid[0:SIZE:step_q, 0:SIZE:step_q]

    plt.figure(figsize=(7, 7))
    plt.imshow(phi_final, cmap="coolwarm", alpha=0.65)
    plt.quiver(
        xx, yy,
        Fx_final[::step_q, ::step_q],
        Fy_final[::step_q, ::step_q],
        color="black",
        pivot="mid",
        scale=30
    )
    # marking of centers
    for (x, y), ang in zip(positions, angles):
        plt.scatter([x], [y], s=25)
    plt.title(f"{case_name}: final domain flux")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(OUT / f"{case_name}_quiver_final.png", dpi=260)
    plt.close()

    return summary


def main():
    print("\n=== PROTO ATOM MAGNETIC DOMAIN EMERGENCE ===")

    summaries = []
    for case_name, mode in CASES.items():
        summaries.append(run_case(case_name, mode))

    df = pd.DataFrame(summaries)
    out_csv = OUT / "magnetic_domain_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== DOMAIN SUMMARY ===")
    print(df.to_string(index=False))

    # final comparison
    plt.figure(figsize=(8, 5))
    x = np.arange(len(df))
    plt.bar(x - 0.2, df["final_curl"], width=0.2, label="final_curl")
    plt.bar(x, df["final_loop"], width=0.2, label="final_loop")
    plt.bar(x + 0.2, df["final_alignment"], width=0.2, label="final_alignment")
    plt.xticks(x, df["case"])
    plt.title("Magnetic domain comparison")
    plt.legend()
    plt.tight_layout()
    out_fig = OUT / "magnetic_domain_comparison.png"
    plt.savefig(out_fig, dpi=260)
    plt.close()

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_fig}")
    print("[DONE] proto atom magnetic domain emergence complete")


if __name__ == "__main__":
    main()