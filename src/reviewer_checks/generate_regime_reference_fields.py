#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path("results/reviewer_checks/regime_reference_fields")
OUT.mkdir(parents=True, exist_ok=True)


GRID_SIZE = 160
CENTER = np.array([GRID_SIZE // 2, GRID_SIZE // 2], dtype=float)

DT = 0.03
N_STEPS = 220
SEED = 123


def laplacian(field: np.ndarray) -> np.ndarray:
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )


def gaussian_blob(
    center_xy: np.ndarray,
    sigma: float,
    amplitude: float = 1.0,
    size: int = GRID_SIZE,
) -> np.ndarray:
    y, x = np.indices((size, size))
    dx = x - center_xy[0]
    dy = y - center_xy[1]
    r2 = dx * dx + dy * dy
    return amplitude * np.exp(-r2 / (2.0 * sigma * sigma))


def ring_profile(
    center_xy: np.ndarray,
    radius: float,
    sigma: float,
    amplitude: float = 1.0,
    size: int = GRID_SIZE,
) -> np.ndarray:
    y, x = np.indices((size, size))
    dx = x - center_xy[0]
    dy = y - center_xy[1]
    r = np.sqrt(dx * dx + dy * dy)
    return amplitude * np.exp(-((r - radius) ** 2) / (2.0 * sigma * sigma))


def directional_wave(
    angle_deg: float,
    wavelength: float,
    envelope_sigma: float,
    size: int = GRID_SIZE,
) -> np.ndarray:
    y, x = np.indices((size, size))
    dx = x - CENTER[0]
    dy = y - CENTER[1]

    theta = np.radians(angle_deg)
    proj = np.cos(theta) * dx + np.sin(theta) * dy
    env = np.exp(-(dx * dx + dy * dy) / (2.0 * envelope_sigma * envelope_sigma))
    return env * np.cos(2.0 * np.pi * proj / wavelength)


def save_field(name: str, field: np.ndarray) -> None:
    np.save(OUT / f"{name}.npy", field)

    plt.figure(figsize=(5, 5))
    plt.imshow(field, cmap="coolwarm")
    plt.title(name)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=220)
    plt.close()


def generate_diffusion_like() -> np.ndarray:
    return gaussian_blob(CENTER, sigma=6.0, amplitude=1.0)


def generate_schrodinger_like() -> np.ndarray:
    core = gaussian_blob(CENTER, sigma=8.0, amplitude=0.9)
    side1 = gaussian_blob(CENTER + np.array([10.0, 0.0]), sigma=5.0, amplitude=0.4)
    side2 = gaussian_blob(CENTER - np.array([10.0, 0.0]), sigma=5.0, amplitude=0.4)
    return core + side1 + side2


def generate_dirac_like() -> np.ndarray:
    left = gaussian_blob(CENTER - np.array([8.0, 0.0]), sigma=4.0, amplitude=1.0)
    right = gaussian_blob(CENTER + np.array([8.0, 0.0]), sigma=4.0, amplitude=-1.0)
    return left + right


def generate_maxwell_like() -> np.ndarray:
    ring1 = ring_profile(CENTER, radius=14.0, sigma=1.8, amplitude=1.0)
    ring2 = ring_profile(CENTER, radius=24.0, sigma=2.0, amplitude=-0.85)
    return ring1 + ring2


def generate_einstein_like() -> np.ndarray:
    radial = ring_profile(CENTER, radius=18.0, sigma=2.2, amplitude=1.0)
    central = gaussian_blob(CENTER, sigma=10.0, amplitude=0.5)
    warp_x = directional_wave(angle_deg=0.0, wavelength=48.0, envelope_sigma=28.0)
    warp_y = directional_wave(angle_deg=90.0, wavelength=48.0, envelope_sigma=28.0)
    return central + radial + 0.25 * warp_x + 0.25 * warp_y


def main() -> None:
    print("=== GENERATE REGIME REFERENCE FIELDS ===")

    fields = {
        "diffusion_like_final": generate_diffusion_like(),
        "schrodinger_like_final": generate_schrodinger_like(),
        "dirac_like_final": generate_dirac_like(),
        "maxwell_like_final": generate_maxwell_like(),
        "einstein_like_final": generate_einstein_like(),
    }

    rows = []
    for name, field in fields.items():
        save_field(name, field)
        rows.append(
            {
                "regime": name,
                "mean_abs": float(np.mean(np.abs(field))),
                "max_abs": float(np.max(np.abs(field))),
            }
        )

    pd.DataFrame(rows).to_csv(OUT / "regime_reference_metadata.csv", index=False)

    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()