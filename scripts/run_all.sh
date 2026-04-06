#!/bin/bash

set -e

echo "=== OMM-SOT FULL PIPELINE ==="

echo "[1/5] Unified chain diagram"
python src/core/unified_chain_diagram.py

echo "[2/5] Regime map diagram"
python src/core/regime_map_diagram.py

echo "[3/5] Structure lifecycle figures"
python src/core/structure_lifecycle_figure_suite.py

echo "[4/5] Two-scale mantle instability"
python src/core/two_scale_mantle_instability_test.py

echo "[5/5] Cosmic mantle expansion"
python src/cosmology/cosmic_mantle_expansion_scan.py

echo "=== DONE ==="