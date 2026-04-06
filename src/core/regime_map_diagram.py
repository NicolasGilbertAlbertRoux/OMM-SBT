#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

def add_box(ax, x, y, w, h, text, fontsize=14):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=2.0,
        facecolor="white",
        edgecolor="black"
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize
    )

def main():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "Emergent Regime Map", ha="center", va="center", fontsize=24)

    # central core
    add_box(ax, 0.39, 0.43, 0.22, 0.12, "Unified Field Model", fontsize=16)

    # surrounding regimes
    add_box(ax, 0.10, 0.70, 0.22, 0.12, "Schrödinger-like\nwave regime", fontsize=14)
    add_box(ax, 0.68, 0.70, 0.22, 0.12, "Dirac-like\nstructured regime", fontsize=14)
    add_box(ax, 0.10, 0.20, 0.22, 0.12, "Maxwell-like\nflux regime", fontsize=14)
    add_box(ax, 0.68, 0.20, 0.22, 0.12, "Einstein / Helmholtz-like\ngeometry regime", fontsize=14)
    add_box(ax, 0.39, 0.08, 0.22, 0.12, "Cosmic mantle\nexpansion regime", fontsize=14)

    # labels
    ax.text(0.50, 0.62, "topological organization", ha="center", va="center", fontsize=12)
    ax.text(0.50, 0.35, "energy-to-geometry coupling", ha="center", va="center", fontsize=12)

    # simple connecting lines
    line_kw = dict(color="black", linewidth=1.8)

    # top
    ax.plot([0.50, 0.50], [0.55, 0.70], **line_kw)
    # bottom
    ax.plot([0.50, 0.50], [0.43, 0.20], **line_kw)
    ax.plot([0.50, 0.50], [0.43, 0.20], **line_kw)
    ax.plot([0.50, 0.50], [0.43, 0.20], **line_kw)
    ax.plot([0.50, 0.50], [0.43, 0.20], alpha=0)
    ax.plot([0.50, 0.50], [0.20, 0.20], alpha=0)

    # left upper
    ax.plot([0.39, 0.32], [0.50, 0.76], **line_kw)
    # right upper
    ax.plot([0.61, 0.68], [0.50, 0.76], **line_kw)
    # left lower
    ax.plot([0.39, 0.32], [0.46, 0.26], **line_kw)
    # right lower
    ax.plot([0.61, 0.68], [0.46, 0.26], **line_kw)
    # bottom center
    ax.plot([0.50, 0.50], [0.43, 0.20], **line_kw)
    ax.plot([0.50, 0.50], [0.20, 0.20], alpha=0)
    ax.plot([0.50, 0.50], [0.20, 0.20], alpha=0)
    ax.plot([0.50, 0.50], [0.20, 0.20], alpha=0)
    ax.plot([0.50, 0.50], [0.20, 0.20], alpha=0)
    ax.plot([0.50, 0.50], [0.20, 0.20], alpha=0)
    ax.plot([0.50, 0.50], [0.20, 0.20], alpha=0)
    ax.plot([0.50, 0.50], [0.20, 0.20], alpha=0)
    ax.plot([0.50, 0.50], [0.20, 0.20], alpha=0)
    ax.plot([0.50, 0.50], [0.20, 0.20], alpha=0)
    ax.plot([0.50, 0.50], [0.20, 0.20], alpha=0)
    ax.plot([0.50, 0.50], [0.20, 0.14], **line_kw)

    ax.text(
        0.5, 0.01,
        "Distinct physical-like sectors emerge as effective regimes of a common energetic substrate.",
        ha="center", va="bottom", fontsize=11
    )

    plt.tight_layout()
    out = OUT / "regime_map_diagram.png"
    plt.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] wrote {out}")

if __name__ == "__main__":
    main()