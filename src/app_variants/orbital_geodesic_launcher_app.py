#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS (CLI)
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--size", type=int, default=256)
parser.add_argument("--n_steps", type=int, default=500)
parser.add_argument("--dt", type=float, default=0.05)

# central body
parser.add_argument("--central_mass", type=float, default=5000.0)
parser.add_argument("--softening", type=float, default=5.0)

# initial condition
parser.add_argument("--init_radius", type=float, default=80.0)
parser.add_argument("--init_speed", type=float, default=2.5)
parser.add_argument("--init_angle_deg", type=float, default=90.0)

# OMM field influence
parser.add_argument("--field_strength", type=float, default=0.5)
parser.add_argument("--field_sigma", type=float, default=30.0)

# magnetic effect
parser.add_argument("--magnetic_strength", type=float, default=0.0)
parser.add_argument("--magnetic_angle_deg", type=float, default=0.0)

args = parser.parse_args()

# ============================================================
# GRID
# ============================================================

NX = NY = args.size
CENTER = np.array([NX / 2, NY / 2])

y, x = np.indices((NY, NX))
dx = x - CENTER[0]
dy = y - CENTER[1]
r = np.sqrt(dx**2 + dy**2)

# ============================================================
# CENTRAL MASS (REAL ATTRACTOR)
# ============================================================

def central_acceleration(pos):
    d = pos - CENTER
    r2 = np.sum(d**2) + args.softening**2
    return -args.central_mass * d / (r2**1.5)

# ============================================================
# OMM FIELD (SMOOTH GEOMETRY)
# ============================================================

psi_field = np.exp(-(r**2) / (2 * args.field_sigma**2))

def field_acceleration(pos):
    px, py = int(pos[0]), int(pos[1])
    if px < 1 or px >= NX-1 or py < 1 or py >= NY-1:
        return np.zeros(2)

    gx = psi_field[py, px+1] - psi_field[py, px-1]
    gy = psi_field[py+1, px] - psi_field[py-1, px]

    return -args.field_strength * np.array([gx, gy])

# ============================================================
# MAGNETIC ANISOTROPY
# ============================================================

theta = np.deg2rad(args.magnetic_angle_deg)
mag_dir = np.array([np.cos(theta), np.sin(theta)])

def magnetic_acceleration(pos, vel):
    if args.magnetic_strength == 0:
        return np.zeros(2)

    # projection anisotrope
    proj = np.dot(vel, mag_dir)
    return args.magnetic_strength * proj * mag_dir

# ============================================================
# INITIAL CONDITIONS
# ============================================================

angle = np.deg2rad(args.init_angle_deg)

pos = CENTER + np.array([
    args.init_radius * np.cos(0),
    args.init_radius * np.sin(0)
])

vel = np.array([
    args.init_speed * np.cos(angle),
    args.init_speed * np.sin(angle)
])

trajectory = [pos.copy()]
speeds = []
radii = []

# ============================================================
# INTEGRATION
# ============================================================

for _ in range(args.n_steps):

    acc = (
        central_acceleration(pos)
        + field_acceleration(pos)
        + magnetic_acceleration(pos, vel)
    )

    vel += args.dt * acc
    pos += args.dt * vel

    trajectory.append(pos.copy())
    speeds.append(np.linalg.norm(vel))
    radii.append(np.linalg.norm(pos - CENTER))

trajectory = np.array(trajectory)

# ============================================================
# SCALE PROXY (OMM-style)
# ============================================================

scale_proxy = np.cumsum(radii) / (np.arange(len(radii)) + 1)

# ============================================================
# PLOTS
# ============================================================

plt.figure(figsize=(6, 6))
plt.imshow(psi_field, cmap="coolwarm")
plt.plot(trajectory[:, 0], trajectory[:, 1], color="black")
plt.scatter([CENTER[0]], [CENTER[1]], color="white", s=40)
plt.title("Emergent geometry + trajectory")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("figures/app_orbital_launcher.png", dpi=200)
plt.close()

# diagnostics

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].plot(radii)
axs[0].set_title("Radius")

axs[1].plot(speeds)
axs[1].set_title("Speed")

axs[2].plot(scale_proxy)
axs[2].set_title("Emergent scale proxy")

plt.tight_layout()
plt.savefig("figures/app_orbital_launcher_diagnostics.png", dpi=200)
plt.close()

# ============================================================
# METRICS
# ============================================================

print(f"final_radius={radii[-1]}")
print(f"mean_radius={np.mean(radii)}")
print(f"max_speed={np.max(speeds)}")
print(f"mean_speed={np.mean(speeds)}")# ============================================================
# GRID
# ============================================================

N = args.size
CENTER = np.array([N / 2, N / 2])

def laplacian(phi):
    return (
        np.roll(phi, 1, 0)
        + np.roll(phi, -1, 0)
        + np.roll(phi, 1, 1)
        + np.roll(phi, -1, 1)
        - 4 * phi
    )

def gradient(arr):
    gy, gx = np.gradient(arr)
    return gx, gy

def gaussian(pos, sigma):
    y, x = np.indices((N, N))
    r2 = (x - pos[0])**2 + (y - pos[1])**2
    return np.exp(-r2 / (2 * sigma**2))

def solve_poisson(source, iters=60, mass=0.0):
    pot = np.zeros_like(source)
    for _ in range(iters):
        pot = (
            np.roll(pot, 1, 0)
            + np.roll(pot, -1, 0)
            + np.roll(pot, 1, 1)
            + np.roll(pot, -1, 1)
            - source
        ) / (4 + mass**2)
    return pot

# ============================================================
# FIELD
# ============================================================

phi = np.zeros((N, N))
pi = np.zeros_like(phi)

src = gaussian(CENTER, args.source_sigma)

psi_global_mem = np.zeros_like(phi)

# ============================================================
# TRAJECTORY
# ============================================================

pos = np.array([args.x0, args.y0], dtype=float)
vel = np.array([args.vx0, args.vy0], dtype=float)

traj = []

# ============================================================
# LOOP
# ============================================================

for step in range(args.n_steps):

    # field evolve
    pi += args.dt * (0.75**2 * laplacian(phi) + src - 0.001 * pi)
    phi += args.dt * pi

    gx, gy = gradient(phi)
    rho = 0.5 * (gx**2 + gy**2) + 0.5 * (pi**2)

    # geometry
    psi_local = solve_poisson(rho, mass=0.08)

    rho_g = gaussian_filter(rho, sigma=args.global_sigma)
    psi_g = solve_poisson(rho_g, mass=args.global_mass)

    psi_global_mem = (
        (1 - args.global_eta) * psi_global_mem
        + args.global_eta * psi_g
    )

    psi_total = psi_local + args.epsilon_global * psi_global_mem

    # force
    fx, fy = gradient(psi_total)

    ix = int(np.clip(pos[0], 0, N - 1))
    iy = int(np.clip(pos[1], 0, N - 1))

    acc = -np.array([fx[iy, ix], fy[iy, ix]])

    # integrate
    vel += args.dt * acc
    pos += args.dt * vel

    traj.append(pos.copy())

traj = np.array(traj)

# ============================================================
# OUTPUT
# ============================================================

plt.figure(figsize=(6,6))
plt.imshow(psi_total, cmap="coolwarm")
plt.plot(traj[:,0], traj[:,1], 'k-')
plt.scatter([CENTER[0]], [CENTER[1]], c='white')
plt.title("Emergent geometry + trajectory")
plt.xticks([])
plt.yticks([])

path = OUT / "app_orbital_launcher.png"
plt.savefig(path, dpi=200)
plt.close()

print(f"final_x={pos[0]:.4f}")
print(f"final_y={pos[1]:.4f}")
print(f"final_speed={np.linalg.norm(vel):.4f}")
