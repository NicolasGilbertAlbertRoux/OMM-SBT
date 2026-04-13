#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path("results/reviewer_checks/regime_5way")
OUT.mkdir(parents=True, exist_ok=True)

REFERENCE_DIR = Path("results/reviewer_checks/regime_reference_fields")
NO_CLIP_DIR = Path("results/reviewer_checks/clean_runs/omm_no_clip")


def load_field(path: Path) -> np.ndarray:
    return np.load(path)


def kurtosis(field: np.ndarray) -> float:
    x = field.ravel().astype(float)
    x = x - np.mean(x)
    m2 = np.mean(x**2) + 1e-12
    m4 = np.mean(x**4)
    return float(m4 / (m2**2))


def participation_ratio(field: np.ndarray) -> float:
    w = np.abs(field.ravel()) ** 2
    s1 = np.sum(w)
    s2 = np.sum(w**2) + 1e-12
    return float((s1**2) / s2)


def active_fraction(field: np.ndarray, sigma_factor: float = 1.5) -> float:
    x = field.ravel().astype(float)
    thr = sigma_factor * np.std(x)
    return float(np.mean(np.abs(x) > thr))


def radial_concentration(field: np.ndarray) -> float:
    h, w = field.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices(field.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    weights = np.abs(field)
    total = np.sum(weights) + 1e-12
    mean_r = np.sum(r * weights) / total
    max_r = np.max(r) + 1e-12
    return float(mean_r / max_r)


def angular_anisotropy(field: np.ndarray) -> float:
    h, w = field.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices(field.shape)
    theta = np.arctan2(y - cy, x - cx)

    weights = np.abs(field)
    cos2 = np.sum(weights * np.cos(2.0 * theta))
    sin2 = np.sum(weights * np.sin(2.0 * theta))
    norm = np.sum(weights) + 1e-12
    return float(np.sqrt(cos2**2 + sin2**2) / norm)


def ring_contrast(field: np.ndarray) -> float:
    h, w = field.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices(field.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    inner = np.abs(field[(r >= 8.0) & (r < 16.0)])
    outer = np.abs(field[(r >= 18.0) & (r < 28.0)])

    if len(inner) == 0 or len(outer) == 0:
        return 0.0

    return float(np.mean(outer) - np.mean(inner))


def summarize_field(name: str, field: np.ndarray) -> dict:
    return {
        "regime": name,
        "kurtosis": kurtosis(field),
        "participation_ratio": participation_ratio(field),
        "active_fraction": active_fraction(field),
        "radial_concentration": radial_concentration(field),
        "angular_anisotropy": angular_anisotropy(field),
        "ring_contrast": ring_contrast(field),
        "mean_abs": float(np.mean(np.abs(field))),
        "max_abs": float(np.max(np.abs(field))),
    }


def field_paths() -> dict[str, Path]:
    return {
        "diffusion_like": REFERENCE_DIR / "diffusion_like_final.npy",
        "schrodinger_like": REFERENCE_DIR / "schrodinger_like_final.npy",
        "dirac_like": REFERENCE_DIR / "dirac_like_final.npy",
        "maxwell_like": REFERENCE_DIR / "maxwell_like_final.npy",
        "einstein_like": REFERENCE_DIR / "einstein_like_final.npy",
        "omm_no_clip": NO_CLIP_DIR / "omm_no_clip_final.npy",
    }


def bar_plot(df: pd.DataFrame, col: str) -> None:
    plt.figure(figsize=(9, 4))
    plt.bar(df["regime"], df[col])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(col)
    plt.title(f"5-way regime comparison: {col}")
    plt.tight_layout()
    plt.savefig(OUT / f"{col}_comparison.png", dpi=220)
    plt.close()


def scatter_plot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    for _, row in df.iterrows():
        plt.scatter(
            row["radial_concentration"],
            row["angular_anisotropy"],
            s=120,
            label=row["regime"],
        )
        plt.text(
            row["radial_concentration"] + 0.003,
            row["angular_anisotropy"] + 0.0002,
            row["regime"],
            fontsize=9,
        )

    plt.xlabel("radial concentration")
    plt.ylabel("angular anisotropy")
    plt.title("Regime separation in descriptor space")
    plt.tight_layout()
    plt.savefig(OUT / "regime_5way_scatter.png", dpi=220)
    plt.close()


def field_grid(fields: dict[str, np.ndarray]) -> None:
    names = list(fields.keys())
    n = len(names)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.ravel()

    for ax, name in zip(axes, names):
        ax.imshow(fields[name], cmap="coolwarm")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(names):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "regime_5way_field_grid.png", dpi=220)
    plt.close(fig)


def main() -> None:
    print("=== 5-WAY REGIME COMPARISON ===")

    rows = []
    loaded_fields: dict[str, np.ndarray] = {}

    for regime, path in field_paths().items():
        if not path.exists():
            print(f"[WARN] Missing {path}, skipping {regime}")
            continue

        field = load_field(path)
        loaded_fields[regime] = field
        rows.append(summarize_field(regime, field))

    if not rows:
        print("[ERROR] No fields were loaded.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "regime_5way_metrics.csv", index=False)

    for col in [
        "kurtosis",
        "participation_ratio",
        "active_fraction",
        "radial_concentration",
        "angular_anisotropy",
        "ring_contrast",
        "mean_abs",
        "max_abs",
    ]:
        bar_plot(df, col)

    scatter_plot(df)
    field_grid(loaded_fields)

    print(df.to_string(index=False))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()