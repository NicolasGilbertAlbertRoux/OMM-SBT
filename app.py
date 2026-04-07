#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
from pathlib import Path

import streamlit as st

PYTHON = sys.executable

DOMAIN_COMMANDS = {
    "Proto-atom (stable structure)": [PYTHON, "src/proto_atoms/proto_atom_final_render_best.py"],
    "Proto-periodic classification": [PYTHON, "src/proto_atoms/proto_atom_periodic_map.py"],
    "Dipole / binding regime": [PYTHON, "src/proto_atoms/proto_atom_dipole_interaction.py"],
    "Magnetic alignment": [PYTHON, "src/magnetism/proto_atom_magnetic_alignment_test.py"],
    "Orbital regime": [PYTHON, "src/orbital/proto_atom_orbital_test.py"],
    "Two-scale geometry": [PYTHON, "src/geometry/geodesic_two_scale_geometry.py"],
    "Cosmology": [PYTHON, "src/cosmology/cosmic_mantle_expansion_scan.py"],
}

DOMAIN_HINTS = {
    "Proto-atom (stable structure)": [
        "figures/proto_atom_final_best.png",
        "figures/proto_atom_final_best_diagnostics.png",
    ],
    "Proto-periodic classification": [
        "figures/proto_periodic_table.png",
        "figures/proto_families_scatter.png",
        "figures/proto_family_counts.png",
    ],
    "Dipole / binding regime": [
        "figures/dipole_interaction_gallery.png",
        "figures/effective_dipole_orientation.png",
    ],
    "Magnetic alignment": [
        "figures/aligned_domain_curl_snapshots.png",
    ],
    "Orbital regime": [
        "figures/full_dynamics_v2_trajectories.png",
    ],
    "Two-scale geometry": [
        "figures/two_scale_total_geometry.png",
        "figures/two_scale_trajectories.png",
        "figures/two_scale_tail_fit.png",
    ],
    "Cosmology": [
        "figures/cosmic_scale_factor.png",
        "figures/cosmic_hubble_rate.png",
        "figures/cosmic_final_total_geometry.png",
    ],
}

DOMAIN_DESCRIPTIONS = {
    "Proto-atom (stable structure)": "Generate the canonical stable proto-atomic render used as the main atom-like reference.",
    "Proto-periodic classification": "Generate the emergent family / periodic-style classification outputs.",
    "Dipole / binding regime": "Generate dipolar and binding-oriented interaction outputs between structured configurations.",
    "Magnetic alignment": "Generate alignment-sensitive magnetic-like domain outputs.",
    "Orbital regime": "Generate orbital or quasi-orbital structured trajectories.",
    "Two-scale geometry": "Generate effective geometry outputs linking local and global structure.",
    "Cosmology": "Generate the large-scale expansion scan and cosmological outputs.",
}

st.set_page_config(page_title="OMM-SOT Explorer", layout="wide")

st.title("OMM-SOT Interactive Explorer")
st.markdown(
    "A lightweight interface for exploring representative regimes of the Oscillatory Mantle Model."
)

with st.sidebar:
    st.header("Explorer")
    domain = st.selectbox("Scientific domain", list(DOMAIN_COMMANDS.keys()))
    st.caption(DOMAIN_DESCRIPTIONS[domain])

    if domain == "Magnetic alignment":
        angle = st.slider("Initial alignment angle (preview parameter)", 0, 180, 45)
        st.caption("Current version: UI preset only. Full parameter wiring can be added in a dedicated wrapper script.")
    elif domain == "Dipole / binding regime":
        rot_x = st.slider("Render rotation X (preview)", 0, 360, 20)
        rot_y = st.slider("Render rotation Y (preview)", 0, 360, 35)
        rot_z = st.slider("Render rotation Z (preview)", 0, 360, 0)
        st.caption("Current version: UI preset only. Full parameter wiring can be added in a dedicated wrapper script.")
    elif domain == "Orbital regime":
        orbit_scale = st.slider("Orbit scale (preview)", 1, 10, 5)
        st.caption("Current version: UI preset only. Full parameter wiring can be added in a dedicated wrapper script.")
    elif domain == "Cosmology":
        complexity = st.slider("Scan scope (preview)", 1, 10, 5)
        st.caption("Current version: UI preset only. Full parameter wiring can be added in a dedicated wrapper script.")

    run = st.button("Run simulation", use_container_width=True)

st.subheader(domain)
st.write(DOMAIN_DESCRIPTIONS[domain])

st.subheader("Expected outputs")
for fig in DOMAIN_HINTS.get(domain, []):
    st.code(fig)

if run:
    st.info("Running simulation. Depending on the selected domain, this may take some time.")
    try:
        subprocess.run(DOMAIN_COMMANDS[domain], check=True)
        st.success("Simulation completed.")
    except subprocess.CalledProcessError as exc:
        st.error(f"Simulation failed with return code {exc.returncode}.")

st.subheader("Preview")
figures = DOMAIN_HINTS.get(domain, [])
cols = st.columns(2)

for i, fig in enumerate(figures[:2]):
    path = Path(fig)
    if path.exists():
        with cols[i % 2]:
            st.image(str(path), caption=path.name, use_container_width=True)

if len(figures) > 2:
    for fig in figures[2:]:
        path = Path(fig)
        if path.exists():
            st.image(str(path), caption=path.name, use_container_width=True)

st.markdown("---")
st.caption(
    "This explorer currently launches existing simulation scripts. Domain-specific parameter controls can be wired more deeply by introducing lightweight wrapper scripts for each regime."
)