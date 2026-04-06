#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

OUT = Path("results/proto_atoms/proto_atom_three_body_dynamics")
OUT.mkdir(parents=True, exist_ok=True)

INPUT = Path("results/final_states/proto_atoms/dipole_field_map_summary.csv")

N_STEPS = 360
DT = 0.07
TRANS_DAMP = 0.992
FORCE_GAIN = 120.0

POT_CONST = -0.083422
POT_A = -40.110664
POT_B = 1080.226908
POT_C = -8866.345575


def angular_wrap_deg(x):
    return ((x + 180.0) % 360.0) - 180.0


def build_score_interpolator(df):
    rows = []
    for _, row in df.iterrows():
        d = float(row["distance"])
        ang = float(row["angle_deg"])
        ori = float(row["orientation_deg"])
        s = float(row["interaction_score"])

        for dang in (-360.0, 0.0, 360.0):
            for dori in (-360.0, 0.0, 360.0):
                rows.append([d, ang + dang, ori + dori, s])

    ext = np.array(rows, dtype=float)
    pts = ext[:, :3]
    vals = ext[:, 3]
    return LinearNDInterpolator(pts, vals, fill_value=np.nan)


def radial_potential(r):
    r = max(float(r), 1e-6)
    return POT_CONST + POT_A / r + POT_B / (r ** 2) + POT_C / (r ** 3)


def radial_score_fallback(r):
    return -radial_potential(r)


def continuous_score(interp, r, angle_deg, orientation_deg=0.0):
    val = interp(float(r), float(angle_deg), float(orientation_deg))
    if np.isnan(val):
        return radial_score_fallback(r)
    return float(val)


def mantle_envelope(r, r0=18.0, r1=95.0):
    inner = 1.0 - np.exp(-(r / r0) ** 2)
    outer = np.exp(-r / r1)
    return inner * outer


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def pair_force(interp, pa, pb):
    dvec = pb - pa
    dist = float(np.linalg.norm(dvec)) + 1e-12
    ang = float(np.degrees(np.arctan2(dvec[1], dvec[0])))
    s_p = continuous_score(interp, dist + 1.5, ang, 0.0)
    s_m = continuous_score(interp, max(1.0, dist - 1.5), ang, 0.0)
    dS_dr = (s_p - s_m) / 3.0
    env = mantle_envelope(dist)
    fmag = FORCE_GAIN * env * dS_dr
    return fmag * unit(dvec), dist


def main():
    print("\n=== PROTO ATOM THREE BODY DYNAMICS ===")

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT}\n"
            "Run proto_atom_dipole_field_map.py first."
        )

    df = pd.read_csv(INPUT)
    interp = build_score_interpolator(df)

    pos = np.array([
        [-22.0, -10.0],
        [ 22.0, -10.0],
        [  0.0,  22.0],
    ], dtype=float)

    vel = np.array([
        [ 0.08,  0.02],
        [-0.08,  0.02],
        [ 0.00, -0.05],
    ], dtype=float)

    history = []

    for step in range(N_STEPS):
        forces = np.zeros_like(pos)

        for i in range(3):
            for j in range(i + 1, 3):
                fij, dij = pair_force(interp, pos[i], pos[j])
                forces[i] += fij
                forces[j] -= fij

        vel = TRANS_DAMP * vel + DT * forces
        pos = pos + DT * vel

        d01 = np.linalg.norm(pos[1] - pos[0])
        d02 = np.linalg.norm(pos[2] - pos[0])
        d12 = np.linalg.norm(pos[2] - pos[1])

        history.append({
            "step": step,
            "x0": pos[0, 0], "y0": pos[0, 1],
            "x1": pos[1, 0], "y1": pos[1, 1],
            "x2": pos[2, 0], "y2": pos[2, 1],
            "d01": d01, "d02": d02, "d12": d12,
        })

        if step % 40 == 0:
            print(f"step={step} d01={d01:.2f} d02={d02:.2f} d12={d12:.2f}")

    hist_df = pd.DataFrame(history)
    hist_csv = OUT / "three_body_history.csv"
    hist_df.to_csv(hist_csv, index=False)

    plt.figure(figsize=(7, 7))
    plt.plot(hist_df["x0"], hist_df["y0"], label="body 0")
    plt.plot(hist_df["x1"], hist_df["y1"], label="body 1")
    plt.plot(hist_df["x2"], hist_df["y2"], label="body 2")
    plt.scatter(hist_df["x0"].iloc[0], hist_df["y0"].iloc[0], s=70)
    plt.scatter(hist_df["x1"].iloc[0], hist_df["y1"].iloc[0], s=70)
    plt.scatter(hist_df["x2"].iloc[0], hist_df["y2"].iloc[0], s=70)
    plt.title("Three-body proto-dynamics")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    fig1 = OUT / "three_body_trajectories.png"
    plt.savefig(fig1, dpi=260)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(hist_df["step"], hist_df["d01"])
    axes[0].set_title("d01")
    axes[1].plot(hist_df["step"], hist_df["d02"])
    axes[1].set_title("d02")
    axes[2].plot(hist_df["step"], hist_df["d12"])
    axes[2].set_title("d12")
    plt.tight_layout()
    fig2 = OUT / "three_body_distances.png"
    plt.savefig(fig2, dpi=260)
    plt.close(fig)

    print(f"[OK] wrote {hist_csv}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print("[DONE] proto atom three body dynamics complete")


if __name__ == "__main__":
    main()