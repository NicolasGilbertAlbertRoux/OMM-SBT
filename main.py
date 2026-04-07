#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys

PYTHON = sys.executable

COMMANDS = {
    # Core paper figures
    "chain": [PYTHON, "src/core/unified_chain_diagram.py"],
    "regime-map": [PYTHON, "src/core/regime_map_diagram.py"],
    "lifecycle": [PYTHON, "src/core/structure_lifecycle_figure_suite.py"],
    "mantle": [PYTHON, "src/core/two_scale_mantle_instability_test.py"],

    # Cosmology
    "cosmo": [PYTHON, "src/cosmology/cosmic_mantle_expansion_scan.py"],

    # Laws
    "laws-short-range": [PYTHON, "src/laws/short_range_attraction_test.py"],
    "laws-loop-bg": [PYTHON, "src/laws/loop_on_background_test.py"],
    "laws-loop-interaction": [PYTHON, "src/laws/loop_interaction_test.py"],
    "laws-loop-stability": [PYTHON, "src/laws/loop_stability_test.py"],
    "laws-flux": [PYTHON, "src/laws/flux_triptych_test.py"],

    # Geometry
    "geometry-effective": [PYTHON, "src/geometry/effective_geometry.py"],
    "geometry-energy": [PYTHON, "src/geometry/geodesic_energy_based_geometry.py"],
    "geometry-dynamic": [PYTHON, "src/geometry/geodesic_dynamic_geometry.py"],
    "geometry-two-scale": [PYTHON, "src/geometry/geodesic_two_scale_geometry.py"],
    "geometry-lensing": [PYTHON, "src/geometry/geodesic_lensing_scan.py"],

    # Magnetism
    "magnetic-alignment": [PYTHON, "src/magnetism/proto_atom_magnetic_alignment_test.py"],
    "magnetic-domain-emergence": [PYTHON, "src/magnetism/proto_atom_magnetic_domain_emergence.py"],
    "magnetic-domain-interaction": [PYTHON, "src/magnetism/proto_atom_magnetic_domain_interaction.py"],
    "magnetic-pair-orientation": [PYTHON, "src/magnetism/proto_atom_magnetic_pair_orientation_scan.py"],
    "magnetic-sector": [PYTHON, "src/magnetism/proto_atom_magnetic_sector_scan.py"],

    # Orbital
    "orbital-test": [PYTHON, "src/orbital/proto_atom_orbital_test.py"],
    "orbital-rotation": [PYTHON, "src/orbital/proto_atom_rotation_scan.py"],
    "orbital-true-rotation": [PYTHON, "src/orbital/proto_atom_true_rotation_test.py"],

    # Proto-atoms / molecules
    "proto-classifier": [PYTHON, "src/proto_atoms/proto_atom_classifier.py"],
    "proto-dipole": [PYTHON, "src/proto_atoms/proto_atom_dipole_interaction.py"],
    "proto-effective-dipole": [PYTHON, "src/proto_atoms/proto_atom_effective_dipole.py"],
    "proto-family-validation": [PYTHON, "src/proto_atoms/proto_atom_family_validation.py"],
    "proto-final-render": [PYTHON, "src/proto_atoms/proto_atom_final_render_best.py"],
    "proto-full-dynamics": [PYTHON, "src/proto_atoms/proto_atom_full_dynamics.py"],
    "proto-full-dynamics-v2": [PYTHON, "src/proto_atoms/proto_atom_full_dynamics_v2.py"],
    "proto-molecular": [PYTHON, "src/proto_atoms/proto_atom_molecular_dynamics.py"],
    "proto-molecule-builder": [PYTHON, "src/proto_atoms/proto_atom_molecule_builder.py"],
    "proto-pair": [PYTHON, "src/proto_atoms/proto_atom_pair_interaction.py"],
    "proto-periodic": [PYTHON, "src/proto_atoms/proto_atom_periodic_map.py"],
    "proto-three-body": [PYTHON, "src/proto_atoms/proto_atom_three_body_dynamics.py"],
    "proto-valence": [PYTHON, "src/proto_atoms/proto_atom_valence_map.py"],

    # App variants
    "app-proto-atom": [PYTHON, "src/app_variants/proto_atom_render_app.py"],
    "app-dipole": [PYTHON, "src/app_variants/proto_atom_dipole_interaction_app.py"],
    "app-magnetic": [PYTHON, "src/app_variants/proto_atom_magnetic_alignment_app.py"],
    "app-orbital": [PYTHON, "src/app_variants/proto_atom_orbital_app.py"],
}

GROUPS = {
    "core": ["chain", "regime-map", "lifecycle", "mantle"],
    "cosmology": ["cosmo"],
    "laws": [
        "laws-short-range",
        "laws-loop-bg",
        "laws-loop-interaction",
        "laws-loop-stability",
        "laws-flux",
    ],
    "geometry": [
        "geometry-effective",
        "geometry-energy",
        "geometry-dynamic",
        "geometry-two-scale",
        "geometry-lensing",
    ],
    "magnetism": [
        "magnetic-alignment",
        "magnetic-domain-emergence",
        "magnetic-domain-interaction",
        "magnetic-pair-orientation",
        "magnetic-sector",
    ],
    "orbital": [
        "orbital-test",
        "orbital-rotation",
        "orbital-true-rotation",
    ],
    "proto-atoms": [
        "proto-classifier",
        "proto-dipole",
        "proto-effective-dipole",
        "proto-family-validation",
        "proto-final-render",
        "proto-full-dynamics",
        "proto-full-dynamics-v2",
        "proto-molecular",
        "proto-molecule-builder",
        "proto-pair",
        "proto-periodic",
        "proto-three-body",
        "proto-valence",
    ],
    "showcase": [
        "chain",
        "regime-map",
        "proto-final-render",
        "proto-periodic",
        "proto-dipole",
        "magnetic-alignment",
        "orbital-test",
        "geometry-two-scale",
        "cosmo",
    ],
    "all": ["chain", "regime-map", "lifecycle", "mantle", "cosmo"],
}

def run_command(name: str) -> None:
    print(f"\n=== RUNNING: {name} ===")
    subprocess.run(COMMANDS[name], check=True)
    print(f"=== DONE: {name} ===")

def main() -> None:
    parser = argparse.ArgumentParser(description="OMM-SOT reproducibility launcher")
    parser.add_argument("--target", required=True, help="Single target or group to run")
    args = parser.parse_args()

    if args.target in COMMANDS:
        run_command(args.target)
    elif args.target in GROUPS:
        for name in GROUPS[args.target]:
            run_command(name)
    else:
        valid = sorted(list(COMMANDS.keys()) + list(GROUPS.keys()))
        print(f"[ERROR] Unknown target: {args.target}")
        print("Available targets:")
        for v in valid:
            print(f" - {v}")
        sys.exit(1)

if __name__ == "__main__":
    main()
