# Oscillatory Mantle Model (OMM)

---

## Toward a Substantial Oscillation Theory (SOT)

This repository contains the code, figures and reproducibility pipeline associated with the paper:

**An Emergent Field Theory of Physical Structures: The Oscillatory Mantle Model (OMM)**

---

## Repository structure

- `src/core/` — core scripts used to reproduce the main conceptual figures
- `src/laws/` — law-emergence tests and reduced interaction regimes
- `src/geometry/` — geometric and geodesic emergence
- `src/cosmology/` — large-scale expansion-like behavior
- `src/proto_atoms/` — proto-atomic, dipolar and molecular regimes
- `src/magnetism/` — magnetic alignment and domain interactions
- `src/orbital/` — orbital and quasi-orbital tests
- `src/diagnostics/` — support and post-processing scripts
- `figures/` — generated figures retained for the paper and repository
- `results/` — numerical outputs and selected final states

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick start

Generate the main reproducibility targets:

```bash
python main.py --target all
```

Or run them individually:

```bash
python main.py --target chain
python main.py --target regime-map
python main.py --target lifecycle
python main.py --target mantle
python main.py --target cosmo
```

---

## Final states

Selected .npy files are provided in results/final_states/ as reference end states for some key regimes. These files are included to facilitate rapid figure regeneration, while the corresponding scripts can also recompute them from scratch.

---

## Notes

The framework explores the emergence of wave-like, structured, flux-like, geometric and cosmological regimes from a unified discrete energetic substrate.

This repository is organized to make both direct reproduction and further extension possible.

---

## License

MIT License

---

## Citation

If you use this code or build upon this framework, please cite the associated manuscript.