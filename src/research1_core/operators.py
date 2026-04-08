from __future__ import annotations

import numpy as np


def graph_laplacians(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degree = adjacency.sum(axis=1)
    degree_matrix = np.diag(degree)
    laplacian = degree_matrix - adjacency

    with np.errstate(divide="ignore"):
        inv_sqrt_degree = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)

    d_inv_sqrt = np.diag(inv_sqrt_degree)
    laplacian_norm = np.eye(adjacency.shape[0]) - d_inv_sqrt @ adjacency @ d_inv_sqrt
    return laplacian, laplacian_norm, degree


def degree_potential(degree: np.ndarray) -> np.ndarray:
    max_degree = np.max(degree) if len(degree) else 0.0
    if max_degree <= 0:
        return np.zeros_like(degree, dtype=float)
    return degree / max_degree


def build_effective_hamiltonian(
    laplacian_norm: np.ndarray,
    degree: np.ndarray,
    alpha: float,
    beta_pot: float,
) -> np.ndarray:
    potential = np.diag(degree_potential(degree))
    return alpha * laplacian_norm + beta_pot * potential


def build_dirac_like(incidence: np.ndarray) -> np.ndarray:
    n, m = incidence.shape
    z_nn = np.zeros((n, n), dtype=float)
    z_mm = np.zeros((m, m), dtype=float)
    top = np.hstack([z_nn, incidence])
    bottom = np.hstack([incidence.T, z_mm])
    return np.vstack([top, bottom])


def build_edge_laplacian(incidence: np.ndarray) -> np.ndarray:
    return incidence.T @ incidence