from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def infer_node_id_column(nodes: pd.DataFrame) -> str:
    for col in ["node_id", "id", "node", "index", "u_id", "v_id"]:
        if col in nodes.columns:
            return col
    for col in nodes.columns:
        if col.endswith("_id"):
            return col
    raise ValueError(f"Could not infer node id column from {list(nodes.columns)}")


def infer_edge_columns(edges: pd.DataFrame) -> Tuple[str, str]:
    candidates = [
        ("source", "target"),
        ("src", "dst"),
        ("u", "v"),
        ("u_id", "v_id"),
        ("node_u", "node_v"),
        ("from", "to"),
    ]
    for a, b in candidates:
        if a in edges.columns and b in edges.columns:
            return a, b
    raise ValueError(f"Could not infer edge columns from {list(edges.columns)}")


def extract_coords(nodes: pd.DataFrame) -> np.ndarray | None:
    coord_sets = [
        ("x", "y", "z", "t"),
        ("u_x", "u_y", "u_z", "u_t"),
    ]
    for cols in coord_sets:
        if all(c in nodes.columns for c in cols):
            return nodes[list(cols)].to_numpy(dtype=float)
    return None


def build_node_index(nodes: pd.DataFrame) -> Tuple[List[int], Dict[int, int]]:
    node_id_col = infer_node_id_column(nodes)
    node_ids = list(nodes[node_id_col].astype(int).values)
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    return node_ids, node_to_idx


def build_adjacency(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    use_edge_weight: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    node_ids, node_to_idx = build_node_index(nodes)
    src_col, dst_col = infer_edge_columns(edges)

    n = len(node_ids)
    adjacency = np.zeros((n, n), dtype=float)

    for _, row in edges.iterrows():
        u = int(row[src_col])
        v = int(row[dst_col])

        if u not in node_to_idx or v not in node_to_idx or u == v:
            continue

        i = node_to_idx[u]
        j = node_to_idx[v]

        weight = 1.0
        if use_edge_weight:
            for wcol in ["weight", "w", "edge_weight"]:
                if wcol in edges.columns:
                    try:
                        weight = float(row[wcol])
                    except Exception:
                        weight = 1.0
                    break

        adjacency[i, j] += weight
        adjacency[j, i] += weight

    return adjacency, node_ids


def orient_edge(u_coord: Tuple[float, ...], v_coord: Tuple[float, ...]) -> bool:
    return tuple(u_coord) < tuple(v_coord)


def build_incidence(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    orient_with_coordinates: bool = True,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[int]]:
    node_ids, node_to_idx = build_node_index(nodes)
    src_col, dst_col = infer_edge_columns(edges)
    coords = extract_coords(nodes)

    node_coord: Dict[int, Tuple[float, ...]] = {}
    if orient_with_coordinates and coords is not None:
        for nid, coord in zip(node_ids, coords):
            node_coord[int(nid)] = tuple(coord.tolist())

    edge_list: List[Tuple[int, int]] = []
    for _, row in edges.iterrows():
        u = int(row[src_col])
        v = int(row[dst_col])

        if u not in node_to_idx or v not in node_to_idx or u == v:
            continue

        if orient_with_coordinates and u in node_coord and v in node_coord:
            forward = orient_edge(node_coord[u], node_coord[v])
            a, b = (u, v) if forward else (v, u)
        else:
            a, b = (u, v) if u < v else (v, u)

        edge_list.append((a, b))

    edge_list = list(dict.fromkeys(edge_list))

    n = len(node_ids)
    m = len(edge_list)
    incidence = np.zeros((n, m), dtype=float)

    for j, (a, b) in enumerate(edge_list):
        incidence[node_to_idx[a], j] = -1.0
        incidence[node_to_idx[b], j] = 1.0

    return incidence, edge_list, node_ids