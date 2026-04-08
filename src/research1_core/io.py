from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_nodes_edges(
    detail_dir: str | Path,
    beta: float,
    seed: int,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load node and edge CSV files for a given beta and seed."""
    detail_dir = Path(detail_dir)
    nodes_path = detail_dir / f"filament_nodes_beta{beta:.2f}_seed{seed}.csv"
    edges_path = detail_dir / f"filament_edges_beta{beta:.2f}_seed{seed}.csv"

    if not nodes_path.exists() or not edges_path.exists():
        return None, None

    return pd.read_csv(nodes_path), pd.read_csv(edges_path)


def load_density(
    density_dir: str | Path,
    beta: float,
    seed: int,
) -> Optional[np.ndarray]:
    """Load mantle density NPY for a given beta and seed."""
    density_dir = Path(density_dir)
    density_path = density_dir / f"collective_density_beta{beta:.2f}_seed{seed}.npy"

    if not density_path.exists():
        return None

    return np.load(density_path)
    return np.load(density_path)
