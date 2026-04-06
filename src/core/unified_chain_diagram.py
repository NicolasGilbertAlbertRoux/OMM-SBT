#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

def add_box(ax, x, y, w, h, text, fontsize=16):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=2.2,
        facecolor="white",
        edgecolor="black"
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize
    )

def add_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.2,
        color="black"
    )
    ax.add_patch(arrow)

def main():
    fig, ax = plt.subplots(figsize=(16, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.34
    w = 0.12
    h = 0.30

    xs = [0.03, 0.19, 0.35, 0.51, 0.67, 0.83]
    labels = [
        "Field",
        "Topology",
        "Flux",
        "Energy",
        "Geometry",
        "Trajectories",
    ]

    for x, label in zip(xs, labels):
        add_box(ax, x, y, w, h, label, fontsize=16)

    for i in range(len(xs) - 1):
        add_arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(
        0.5, 0.90,
        "Unified Emergence Chain",
        ha="center", va="center",
        fontsize=22
    )

    ax.text(
        0.5, 0.12,
        "A single discrete energetic substrate generates structures, laws and effective geometry across scales.",
        ha="center", va="center",
        fontsize=13
    )

    plt.tight_layout()
    out = OUT / "unified_chain_diagram.png"
    plt.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"[OK] wrote {out}")

if __name__ == "__main__":
    main()