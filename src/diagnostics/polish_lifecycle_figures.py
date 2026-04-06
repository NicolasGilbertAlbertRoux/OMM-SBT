#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

IN_DIR = Path("figures")
OUT_DIR = Path("figures/polished")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    ("fragmentation_example.png", "fragmentation_example_polished.png"),
    ("destruction_collapse.png", "destruction_collapse_polished.png"),
    ("destruction_diffusion.png", "destruction_diffusion_polished.png"),
    ("reconstruction_cycle.png", "reconstruction_cycle_polished.png"),
]

def crop_center(img, size_ratio=0.6):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    dx = int(w * size_ratio / 2)
    dy = int(h * size_ratio / 2)
    return img[cy - dy:cy + dy, cx - dx:cx + dx]

def enhance_contrast(img):
    img = img.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def polish_image(src_name: str, out_name: str):
    src = IN_DIR / src_name
    img = mpimg.imread(src)

    # crop (zoom visuel)
    img = crop_center(img, 0.55)

    # contraste
    img = enhance_contrast(img)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(img, interpolation="bicubic")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0.02)
    out = OUT_DIR / out_name
    plt.savefig(out, dpi=500, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"[OK] wrote {out}")

def main():
    print("\n=== POLISH LIFECYCLE FIGURES (ENHANCED) ===")
    for src_name, out_name in FILES:
        polish_image(src_name, out_name)
    print("[DONE] polished lifecycle figures")

if __name__ == "__main__":
    main()