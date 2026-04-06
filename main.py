#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys

COMMANDS = {
    "lifecycle": ["python", "src/core/structure_lifecycle_figure_suite.py"],
    "cosmo": ["python", "src/cosmology/cosmic_mantle_expansion_scan.py"],
    "mantle": ["python", "src/core/two_scale_mantle_instability_test.py"],
    "chain": ["python", "src/core/unified_chain_diagram.py"],
    "regime-map": ["python", "src/core/regime_map_diagram.py"],
}

def run_command(name: str):
    if name not in COMMANDS:
        print(f"[ERROR] Unknown target: {name}")
        print("Available targets:", ", ".join(COMMANDS.keys()))
        sys.exit(1)

    print(f"[RUN] {name}")
    subprocess.run(COMMANDS[name], check=True)

def main():
    parser = argparse.ArgumentParser(
        description="OMM-SOT reproducibility launcher"
    )
    parser.add_argument(
        "--target",
        choices=list(COMMANDS.keys()) + ["all"],
        required=True,
        help="Which reproducibility target to run"
    )

    args = parser.parse_args()

    if args.target == "all":
        for key in ["chain", "regime-map", "lifecycle", "mantle", "cosmo"]:
            run_command(key)
    else:
        run_command(args.target)

if __name__ == "__main__":
    main()