#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parent


COMMANDS = {
    # ========================================================
    # Core paper figures / conservative manuscript subset
    # ========================================================
    "chain": [PYTHON, "src/core/unified_chain_diagram.py"],
    "regime-map": [PYTHON, "src/core/regime_map_diagram.py"],
    "lifecycle": [PYTHON, "src/core/structure_lifecycle_figure_suite.py"],
    "mantle": [PYTHON, "src/core/two_scale_mantle_instability_test.py"],

    # ========================================================
    # Cosmology
    # ========================================================
    "cosmo": [PYTHON, "src/cosmology/cosmic_mantle_expansion_scan.py"],

    # ========================================================
    # Laws
    # ========================================================
    "laws-short-range": [PYTHON, "src/laws/short_range_attraction_test.py"],
    "laws-loop-bg": [PYTHON, "src/laws/loop_on_background_test.py"],
    "laws-loop-interaction": [PYTHON, "src/laws/loop_interaction_test.py"],
    "laws-loop-stability": [PYTHON, "src/laws/loop_stability_test.py"],
    "laws-flux": [PYTHON, "src/laws/flux_triptych_test.py"],

    # ========================================================
    # Geometry
    # ========================================================
    "geometry-effective": [PYTHON, "src/geometry/effective_geometry.py"],
    "geometry-energy": [PYTHON, "src/geometry/geodesic_energy_based_geometry.py"],
    "geometry-dynamic": [PYTHON, "src/geometry/geodesic_dynamic_geometry.py"],
    "geometry-two-scale": [PYTHON, "src/geometry/geodesic_two_scale_geometry.py"],
    "geometry-lensing": [PYTHON, "src/geometry/geodesic_lensing_scan.py"],

    # ========================================================
    # Magnetism
    # ========================================================
    "magnetic-alignment": [PYTHON, "src/magnetism/proto_atom_magnetic_alignment_test.py"],
    "magnetic-domain-emergence": [PYTHON, "src/magnetism/proto_atom_magnetic_domain_emergence.py"],
    "magnetic-domain-interaction": [PYTHON, "src/magnetism/proto_atom_magnetic_domain_interaction.py"],
    "magnetic-pair-orientation": [PYTHON, "src/magnetism/proto_atom_magnetic_pair_orientation_scan.py"],
    "magnetic-sector": [PYTHON, "src/magnetism/proto_atom_magnetic_sector_scan.py"],

    # ========================================================
    # Orbital
    # ========================================================
    "orbital-test": [PYTHON, "src/orbital/proto_atom_orbital_test.py"],
    "orbital-rotation": [PYTHON, "src/orbital/proto_atom_rotation_scan.py"],
    "orbital-true-rotation": [PYTHON, "src/orbital/proto_atom_true_rotation_test.py"],

    # ========================================================
    # Proto-atoms / molecules
    # ========================================================
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
}


GROUPS = {
    "core": [
        "chain",
        "regime-map",
        "lifecycle",
        "mantle",
    ],
    "cosmology": [
        "cosmo",
    ],
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

    # Conservative manuscript-facing subset
    "all": [
        "chain",
        "regime-map",
        "lifecycle",
        "mantle",
        "cosmo",
    ],

    # Compact cross-domain demonstration
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
}


def script_path_from_command(command: list[str]) -> Path:
    """
    Return the script path from a command specification such as:
    [PYTHON, "src/core/unified_chain_diagram.py"]
    """
    if len(command) < 2:
        raise ValueError(f"Invalid command specification: {command}")
    return ROOT / command[1]


def validate_commands() -> list[str]:
    """
    Check that all referenced scripts exist before execution.
    Returns a list of human-readable error messages.
    """
    errors = []

    for name, command in COMMANDS.items():
        try:
            path = script_path_from_command(command)
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            continue

        if not path.exists():
            errors.append(f"{name}: missing script -> {path}")

    return errors


def run_command(name: str) -> None:
    if name not in COMMANDS:
        print(f"[ERROR] Unknown target: {name}")
        sys.exit(1)

    command = COMMANDS[name]
    script_path = script_path_from_command(command)

    print(f"\n=== RUNNING: {name} ===")
    print(f"[SCRIPT] {script_path.relative_to(ROOT)}")

    subprocess.run(command, check=True, cwd=ROOT)

    print(f"=== DONE: {name} ===")


def list_targets() -> None:
    print("Available single targets:\n")
    for name in sorted(COMMANDS.keys()):
        print(f" - {name}")

    print("\nAvailable groups:\n")
    for name in sorted(GROUPS.keys()):
        print(f" - {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OMM-SOT reproducibility and domain launcher"
    )
    parser.add_argument(
        "--target",
        help="Single target or group to run",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available single targets and groups",
    )
    args = parser.parse_args()

    if args.list:
        list_targets()
        return

    if not args.target:
        parser.error("the following argument is required: --target (or use --list)")

    errors = validate_commands()
    if errors:
        print("[ERROR] Repository launcher configuration is not consistent.\n")
        for err in errors:
            print(f" - {err}")
        print("\nPlease fix the missing or invalid script references before running targets.")
        sys.exit(1)

    target = args.target

    if target in COMMANDS:
        run_command(target)
        return

    if target in GROUPS:
        for name in GROUPS[target]:
            run_command(name)
        return

    print(f"[ERROR] Unknown target: {target}\n")
    list_targets()
    sys.exit(1)


if __name__ == "__main__":
    main()
