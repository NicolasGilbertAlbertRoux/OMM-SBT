#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN = Path("results/final_states/proto_atoms/lobe_physics_enriched.csv")
OUT = Path("results/proto_atoms/proto_atom_periodic_map")
OUT.mkdir(parents=True, exist_ok=True)


def label_family(row):
    lobes = int(row["n_lobes"])
    instability = float(row["instability_score"])
    core = float(row["core_mass_fraction"])
    split = float(row["split_score"])
    offset = float(row["centroid_offset"])

    if offset > 2.2:
        return "decentered"

    if instability > 0.85:
        return "unstable"

    if lobes <= 4:
        return "few-lobed"

    if 5 <= lobes <= 7:
        if core >= 0.14:
            return "compact-mid"
        return "mid-lobed"

    if 8 <= lobes <= 10:
        if split >= 0.17:
            return "split-high"
        return "high-lobed"

    if lobes >= 11:
        return "very-high-lobed"

    return "generic"


def proto_period(row):
    lobes = int(row["n_lobes"])
    if lobes <= 4:
        return 1
    if lobes <= 7:
        return 2
    if lobes <= 10:
        return 3
    return 4


def proto_group(row):
    core = float(row["core_mass_fraction"])
    split = float(row["split_score"])
    anis = float(row["anisotropy"])

    if core >= 0.16:
        return 1
    if split >= 0.17:
        return 2
    if anis >= 1.17:
        return 3
    return 4


def stability_band(instability):
    if instability < 0.50:
        return "stable"
    if instability < 0.80:
        return "meta-stable"
    return "unstable"


def main():
    if not IN.exists():
        raise FileNotFoundError(f"Missing file: {IN}")

    df = pd.read_csv(IN)

    df["proto_family"] = df.apply(label_family, axis=1)
    df["proto_period"] = df.apply(proto_period, axis=1)
    df["proto_group"] = df.apply(proto_group, axis=1)
    df["stability_band"] = df["instability_score"].apply(stability_band)

    # useful sorting
    df = df.sort_values(
        ["proto_period", "proto_group", "instability_score", "centroid_offset"],
        ascending=[True, True, True, True]
    )

    summary_csv = OUT / "proto_periodic_map_summary.csv"
    df.to_csv(summary_csv, index=False)

    # aggregate by family
    fam = (
        df.groupby("proto_family")
        .agg(
            n_cases=("seed", "count"),
            mean_lobes=("n_lobes", "mean"),
            mean_core=("core_mass_fraction", "mean"),
            mean_split=("split_score", "mean"),
            mean_offset=("centroid_offset", "mean"),
            mean_instability=("instability_score", "mean"),
        )
        .reset_index()
        .sort_values(["mean_instability", "mean_offset"], ascending=[True, True])
    )

    fam_csv = OUT / "proto_family_aggregate.csv"
    fam.to_csv(fam_csv, index=False)

    # “table” aggregate
    table = (
        df.groupby(["proto_period", "proto_group"])
        .agg(
            n_cases=("seed", "count"),
            mean_lobes=("n_lobes", "mean"),
            mean_core=("core_mass_fraction", "mean"),
            mean_split=("split_score", "mean"),
            mean_instability=("instability_score", "mean"),
        )
        .reset_index()
    )

    table_csv = OUT / "proto_periodic_table_cells.csv"
    table.to_csv(table_csv, index=False)

    print("\n=== PROTO PERIODIC MAP ===")
    print(df[[
        "seed", "n_lobes", "core_mass_fraction", "split_score",
        "centroid_offset", "instability_score",
        "proto_period", "proto_group", "proto_family", "stability_band"
    ]].to_string(index=False))

    print("\n=== FAMILY AGGREGATE ===")
    print(fam.to_string(index=False))

    print("\n=== TABLE CELLS ===")
    print(table.to_string(index=False))

    # ---------------------------------------------------------
    # Figure 1: scatter lobes vs instability, colored by family
    # ---------------------------------------------------------
    families = sorted(df["proto_family"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    color_map = {fam: colors[i] for i, fam in enumerate(families)}

    plt.figure(figsize=(7, 5))
    for fam_name in families:
        g = df[df["proto_family"] == fam_name]
        plt.scatter(
            g["n_lobes"],
            g["instability_score"],
            label=fam_name,
            s=70,
            color=color_map[fam_name]
        )
    plt.xlabel("n_lobes")
    plt.ylabel("instability_score")
    plt.title("Proto-atomic families in lobe/instability space")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig1 = OUT / "proto_families_scatter.png"
    plt.savefig(fig1, dpi=240)
    plt.close()

    # ---------------------------------------------------------
    # Figure 2: pseudo periodic map
    # ---------------------------------------------------------
    max_period = int(df["proto_period"].max())
    max_group = int(df["proto_group"].max())

    grid_value = np.full((max_period, max_group), np.nan)
    grid_text = {}

    for _, row in table.iterrows():
        p = int(row["proto_period"]) - 1
        g = int(row["proto_group"]) - 1
        grid_value[p, g] = row["mean_instability"]
        grid_text[(p, g)] = (
            f"n={int(row['n_cases'])}\n"
            f"L={row['mean_lobes']:.1f}\n"
            f"I={row['mean_instability']:.2f}"
        )

    plt.figure(figsize=(8, 5))
    im = plt.imshow(grid_value, cmap="viridis_r", aspect="auto")
    plt.colorbar(im, label="mean instability")

    for p in range(max_period):
        for g in range(max_group):
            if (p, g) in grid_text:
                plt.text(g, p, grid_text[(p, g)], ha="center", va="center", color="white", fontsize=9)

    plt.xticks(range(max_group), [f"G{g+1}" for g in range(max_group)])
    plt.yticks(range(max_period), [f"P{p+1}" for p in range(max_period)])
    plt.title("Emergent proto-periodic table")
    plt.tight_layout()
    fig2 = OUT / "proto_periodic_table.png"
    plt.savefig(fig2, dpi=240)
    plt.close()

    # ---------------------------------------------------------
    # Figure 3: family counts
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 4.5))
    plt.bar(fam["proto_family"], fam["n_cases"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("count")
    plt.title("Proto-atomic family counts")
    plt.tight_layout()
    fig3 = OUT / "proto_family_counts.png"
    plt.savefig(fig3, dpi=240)
    plt.close()

    print(f"\n[OK] wrote {summary_csv}")
    print(f"[OK] wrote {fam_csv}")
    print(f"[OK] wrote {table_csv}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print(f"[OK] wrote {fig3}")
    print("[DONE] proto periodic map complete")


if __name__ == "__main__":
    main()