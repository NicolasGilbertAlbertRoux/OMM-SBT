import numpy as np
import matplotlib.pyplot as plt

# ==============================
# GLOBAL SETTINGS
# ==============================

DT = 0.05
STEPS = 400

# masses
M = 1.0

# amortissement
DAMPING = 0.98

# ==============================
# SCORE (enhanced model)
# ==============================

def interaction_score(d, theta, phi):
    return (
        0.06888
        + 0.08448 * np.cos(theta)
        + 0.00363 * np.cos(phi)
        - 0.00453 * np.cos(theta - phi)
        - 0.00219 * np.cos(theta)**2
        + 0.00587 * np.cos(phi)**2
        + 0.02614 * np.cos(theta - phi)**2
        + 40.11 / d
        - 1080.22 / d**2
        + 8866.34 / d**3
    )

# ==============================
# OSCILLATORY MANTLE
# ==============================

def mantle_envelope(d):
    return 1.0 / (1.0 + (d / 60.0)**2)

# ==============================
# NUMERICAL GRADIENT
# ==============================

def radial_gradient(d, theta, phi):
    eps = 1e-3
    s1 = interaction_score(d + eps, theta, phi)
    s2 = interaction_score(d - eps, theta, phi)
    return (s1 - s2) / (2 * eps)

def angular_gradient(theta, d, phi):
    eps = 1e-3
    s1 = interaction_score(d, theta + eps, phi)
    s2 = interaction_score(d, theta - eps, phi)
    return (s1 - s2) / (2 * eps)

# ==============================
# INITIALIZATION
# ==============================

A = np.array([-20.0, -8.0])
B = np.array([20.0, 8.0])

vA = np.array([0.0, 0.0])
vB = np.array([0.0, 0.0])

theta_A = 0.0
theta_B = np.pi

traj_A = []
traj_B = []
distances = []
scores = []

# ==============================
# SIMULATION
# ==============================

for step in range(STEPS):

    r_vec = B - A
    d = np.linalg.norm(r_vec)
    r_hat = r_vec / (d + 1e-9)

    # angle between dipoles
    theta = theta_A
    phi = theta_B

    # score
    score = interaction_score(d, theta, phi)
    env = mantle_envelope(d)

    # gradients
    dS_dr = radial_gradient(d, theta, phi)
    dS_dtheta = angular_gradient(theta, d, phi)

    # ==========================
    # FORCE
    # ==========================
    F_mag = -dS_dr * env
    F = F_mag * r_hat

    # ==========================
    # TORQUE (dipole rotation)
    # ==========================
    torque_A = -dS_dtheta * env
    torque_B = +dS_dtheta * env

    # update orientation
    theta_A += torque_A * DT
    theta_B += torque_B * DT

    # ==========================
    # DYNAMIC TRANSLATION
    # ==========================
    vA += F / M * DT
    vB -= F / M * DT

    # light damping
    vA *= DAMPING
    vB *= DAMPING

    A += vA * DT
    B += vB * DT

    # ==========================
    # LOG
    # ==========================
    traj_A.append(A.copy())
    traj_B.append(B.copy())
    distances.append(d)
    scores.append(score)

    if step % 50 == 0:
        print(f"step={step} d={d:.2f} score={score:.3f} F={F_mag:.4f}")

# ==============================
# VISUALIZATION
# ==============================

traj_A = np.array(traj_A)
traj_B = np.array(traj_B)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(traj_A[:,0], traj_A[:,1], label="A")
plt.plot(traj_B[:,0], traj_B[:,1], label="B")
plt.scatter(traj_A[0,0], traj_A[0,1], c='green')
plt.scatter(traj_B[0,0], traj_B[0,1], c='red')
plt.title("Trajectoires")
plt.legend()

plt.subplot(1,2,2)
plt.plot(distances, label="distance")
plt.plot(scores, label="score")
plt.title("Dynamique")
plt.legend()

plt.tight_layout()
plt.show()