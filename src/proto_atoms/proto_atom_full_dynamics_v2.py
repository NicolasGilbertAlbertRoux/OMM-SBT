#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

OUT = Path("results/proto_atoms/proto_atom_full_dynamics_v2")
OUT.mkdir(parents=True, exist_ok=True)

INPUT = Path("results/final_states/proto_atoms/dipole_field_map_summary.csv")

# ------------------------------------------------------------
# simulation params
# ------------------------------------------------------------
N_STEPS = 420
DT = 0.08

TRANS_DAMP = 0.992
ROT_DAMP = 0.992

FORCE_GAIN = 140.0
TORQUE_GAIN = 18.0

MASS_A = 1.0
MASS_B = 1.0
INERTIA_A = 1.0
INERTIA_B = 1.0

# Initial conditions
POS_A0 = np.array([-22.0, -8.0], dtype=float)
POS_B0 = np.array([ 22.0,  8.0], dtype=float)

VEL_A0 = np.array([ 0.16,  0.03], dtype=float)
VEL_B0 = np.array([-0.16, -0.03], dtype=float)

THETA_A0 = np.radians(10.0)
THETA_B0 = np.radians(170.0)

OMEGA_A0 = 0.02
OMEGA_B0 = -0.02

# estimated effective radial potential
POT_CONST = -0.083422
POT_A = -40.110664
POT_B = 1080.226908
POT_C = -8866.345575


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def angular_wrap_deg(x):
    return ((x + 180.0) % 360.0) - 180.0


def angular_wrap_rad(x):
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


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
    # score ~ -V
    return -radial_potential(r)


def continuous_score(interp, r, angle_deg, orientation_deg):
    val = interp(float(r), float(angle_deg), float(orientation_deg))
    if np.isnan(val):
        return radial_score_fallback(r)
    return float(val)


def mantle_envelope(r, r0=18.0, r1=95.0):
    r = float(r)
    inner = 1.0 - np.exp(-(r / r0) ** 2)
    outer = np.exp(-r / r1)
    return inner * outer


def score_gradients(interp, r, angle_deg, orientation_deg):
    # dérivées numériques locales
    srp = continuous_score(interp, r + 1.5, angle_deg, orientation_deg)
    srm = continuous_score(interp, max(1.0, r - 1.5), angle_deg, orientation_deg)
    dS_dr = (srp - srm) / 3.0

    sap = continuous_score(interp, r, angle_deg + 4.0, orientation_deg)
    sam = continuous_score(interp, r, angle_deg - 4.0, orientation_deg)
    dS_dangle = (sap - sam) / 8.0

    sop = continuous_score(interp, r, angle_deg, orientation_deg + 4.0)
    som = continuous_score(interp, r, angle_deg, orientation_deg - 4.0)
    dS_dori = (sop - som) / 8.0

    return dS_dr, dS_dangle, dS_dori


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    print("\n=== PROTO ATOM FULL DYNAMICS V2 ===")

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT}\n"
            "Run proto_atom_dipole_field_map.py first."
        )

    df = pd.read_csv(INPUT)
    interp = build_score_interpolator(df)

    pos_a = POS_A0.copy()
    pos_b = POS_B0.copy()

    vel_a = VEL_A0.copy()
    vel_b = VEL_B0.copy()

    theta_a = THETA_A0
    theta_b = THETA_B0

    omega_a = OMEGA_A0
    omega_b = OMEGA_B0

    hist = []

    for step in range(N_STEPS):
        dvec = pos_b - pos_a
        dist = float(np.linalg.norm(dvec)) + 1e-12
        rhat = unit(dvec)

        placement_angle_deg = float(np.degrees(np.arctan2(dvec[1], dvec[0])))

        # relative orientation of B with respect to A
        orientation_rel_deg = float(np.degrees(angular_wrap_rad(theta_b - theta_a)))

        score = continuous_score(interp, dist, placement_angle_deg, orientation_rel_deg)
        env = mantle_envelope(dist)

        dS_dr, dS_dangle, dS_dori = score_gradients(interp, dist, placement_angle_deg, orientation_rel_deg)

        # ----------------------------------------------------
        # principal radial force
        # ----------------------------------------------------
        force_mag = FORCE_GAIN * env * dS_dr
        force_ab = force_mag * rhat
        force_ba = -force_ab

        # ----------------------------------------------------
        # a small tangential coupling derived from the geometry
        # to allow for curvature/orbit without inventing a law
        # ----------------------------------------------------
        that = np.array([-rhat[1], rhat[0]])
        tangential_mag = 0.25 * FORCE_GAIN * env * dS_dangle
        tangential_force = tangential_mag * that

        force_ab = force_ab + tangential_force
        force_ba = -force_ab

        # ----------------------------------------------------
        # torques
        # dS_dori affects the relative orientation
        # ----------------------------------------------------
        torque_a = -TORQUE_GAIN * env * dS_dori
        torque_b = +TORQUE_GAIN * env * dS_dori

        # update rotation
        omega_a = ROT_DAMP * omega_a + DT * torque_a / INERTIA_A
        omega_b = ROT_DAMP * omega_b + DT * torque_b / INERTIA_B

        theta_a = theta_a + DT * omega_a
        theta_b = theta_b + DT * omega_b

        # update translation
        vel_a = TRANS_DAMP * vel_a + DT * force_ab / MASS_A
        vel_b = TRANS_DAMP * vel_b + DT * force_ba / MASS_B

        pos_a = pos_a + DT * vel_a
        pos_b = pos_b + DT * vel_b

        hist.append({
            "step": step,
            "ax": pos_a[0],
            "ay": pos_a[1],
            "bx": pos_b[0],
            "by": pos_b[1],
            "vx_a": vel_a[0],
            "vy_a": vel_a[1],
            "vx_b": vel_b[0],
            "vy_b": vel_b[1],
            "theta_a_deg": np.degrees(theta_a),
            "theta_b_deg": np.degrees(theta_b),
            "omega_a": omega_a,
            "omega_b": omega_b,
            "distance": dist,
            "placement_angle_deg": placement_angle_deg,
            "orientation_rel_deg": orientation_rel_deg,
            "interaction_score": score,
            "mantle_envelope": env,
            "dS_dr": dS_dr,
            "dS_dangle": dS_dangle,
            "dS_dori": dS_dori,
            "force_x": force_ab[0],
            "force_y": force_ab[1],
            "torque_a": torque_a,
            "torque_b": torque_b,
        })

        if step % 40 == 0:
            print(
                f"step={step} "
                f"d={dist:.3f} "
                f"score={score:.3f} "
                f"env={env:.3f} "
                f"dS_dr={dS_dr:.5f} "
                f"F=({force_ab[0]:.4f},{force_ab[1]:.4f}) "
                f"ω=({omega_a:.4f},{omega_b:.4f})"
            )

    hist_df = pd.DataFrame(hist)
    hist_csv = OUT / "full_dynamics_v2_history.csv"
    hist_df.to_csv(hist_csv, index=False)

    print("\n=== FINAL STATE ===")
    print(hist_df.tail(1).to_string(index=False))

    # --------------------------------------------------------
    # figure 1 trajectories
    # --------------------------------------------------------
    plt.figure(figsize=(7, 7))
    plt.plot(hist_df["ax"], hist_df["ay"], label="body A")
    plt.plot(hist_df["bx"], hist_df["by"], label="body B")
    plt.scatter(hist_df["ax"].iloc[0], hist_df["ay"].iloc[0], s=80, label="A start")
    plt.scatter(hist_df["bx"].iloc[0], hist_df["by"].iloc[0], s=80, label="B start")
    plt.scatter(hist_df["ax"].iloc[-1], hist_df["ay"].iloc[-1], s=80, label="A end")
    plt.scatter(hist_df["bx"].iloc[-1], hist_df["by"].iloc[-1], s=80, label="B end")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Full proto-dynamics v2 trajectories")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    fig1 = OUT / "full_dynamics_v2_trajectories.png"
    plt.savefig(fig1, dpi=260)
    plt.close()

    # --------------------------------------------------------
    # figure 2 diagnostics
    # --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(hist_df["step"], hist_df["distance"])
    axes[0, 0].set_title("Distance over time")

    axes[0, 1].plot(hist_df["step"], hist_df["interaction_score"])
    axes[0, 1].set_title("Interaction score over time")

    axes[1, 0].plot(hist_df["step"], hist_df["theta_a_deg"], label="theta A")
    axes[1, 0].plot(hist_df["step"], hist_df["theta_b_deg"], label="theta B")
    axes[1, 0].set_title("Dipole orientations")
    axes[1, 0].legend()

    axes[1, 1].plot(hist_df["step"], hist_df["omega_a"], label="omega A")
    axes[1, 1].plot(hist_df["step"], hist_df["omega_b"], label="omega B")
    axes[1, 1].set_title("Angular velocities")
    axes[1, 1].legend()

    plt.tight_layout()
    fig2 = OUT / "full_dynamics_v2_diagnostics.png"
    plt.savefig(fig2, dpi=260)
    plt.close(fig)

    print(f"[OK] wrote {hist_csv}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print("[DONE] proto atom full dynamics v2 complete")


if __name__ == "__main__":
    main()