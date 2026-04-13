#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path("results/reviewer_checks/clean_runs/omm_no_clip")
OUT.mkdir(parents=True, exist_ok=True)


GRID_SIZE = 160
DT = 0.03
N_STEPS = 220

LAPLACIAN_GAIN = 0.22
DAMPING = 0.998

SEED = 42
SIGMA = 6.0
SOURCE_AMPLITUDE = 0.035
SOURCE_DURATION = 24

CENTER = np.array([GRID_SIZE // 2, GRID_SIZE // 2], dtype=float)


def laplacian(field: np.ndarray) -> np.ndarray:
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )


def gaussian_source(
    center_xy: np.ndarray,
    amplitude: float,
    sigma: float,
    size: int,
) -> np.ndarray:
    y, x = np.indices((size, size))
    dx = x - center_xy[0]
    dy = y - center_xy[1]
    r2 = dx * dx + dy * dy
    return amplitude * np.exp(-r2 / (2.0 * sigma * sigma))


def compute_energy_proxy(phi: np.ndarray, psi: np.ndarray) -> float:
    gy, gx = np.gradient(phi)
    return float(np.mean(phi**2) + np.mean(psi**2) + 0.25 * np.mean(gx**2 + gy**2))


def run_no_clip() -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(SEED)

    phi = 1e-4 * rng.normal(size=(GRID_SIZE, GRID_SIZE))
    psi = np.zeros_like(phi)

    source_profile = gaussian_source(
        center_xy=CENTER,
        amplitude=1.0,
        sigma=SIGMA,
        size=GRID_SIZE,
    )

    energy_rows = []

    for step in range(N_STEPS):
        source = np.zeros_like(phi)

        if step < SOURCE_DURATION:
            envelope = np.sin(np.pi * (step + 1) / (SOURCE_DURATION + 1))
            source = SOURCE_AMPLITUDE * envelope * source_profile

        lap = laplacian(phi)
        psi = DAMPING * psi + DT * (LAPLACIAN_GAIN * lap + source)
        phi = phi + DT * psi

        energy_rows.append(
            {
                "step": step,
                "energy_proxy": compute_energy_proxy(phi, psi),
                "max_abs_phi": float(np.max(np.abs(phi))),
                "mean_abs_phi": float(np.mean(np.abs(phi))),
            }
        )

    return phi, pd.DataFrame(energy_rows)


def save_field_figure(field: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(field, cmap="coolwarm")
    plt.title("OMM-like field without clipping")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_energy_figure(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(df["step"], df["energy_proxy"], label="energy proxy")
    plt.xlabel("step")
    plt.ylabel("energy proxy")
    plt.title("No-clipping run: energy trace")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    print("=== CLEAN OMM RUN (NO CLIP) ===")

    field, energy_df = run_no_clip()

    np.save(OUT / "omm_no_clip_final.npy", field)
    energy_df.to_csv(OUT / "omm_no_clip_energy.csv", index=False)

    save_field_figure(field, OUT / "omm_no_clip_final_field.png")
    save_energy_figure(energy_df, OUT / "omm_no_clip_energy_trace.png")

    print(f"[OK] wrote {OUT / 'omm_no_clip_final.npy'}")
    print(f"[OK] wrote {OUT / 'omm_no_clip_energy.csv'}")
    print(f"[OK] wrote {OUT / 'omm_no_clip_final_field.png'}")
    print(f"[OK] wrote {OUT / 'omm_no_clip_energy_trace.png'}")


if __name__ == "__main__":
    main()