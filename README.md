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

## Showcase (recommended)

To explore the model across multiple physical regimes:

```bash
python main.py --target showcase
```

This generates representative results including:
	- proto-atomic structures
	- emergent periodic classification
	- molecular dynamics
	- dipole and magnetic interactions
	- orbital regimes
	- effective geometry
    - cosmological evolution

This mode is designed to provide an intuitive entry point into the unified framework.

---

## Interactive explorer

A lightweight local interface is also provided:

```bash
streamlit run app.py
```

This opens a small interactive application from which users can launch representative simulations by scientific domain.

---

## Quick start

Generate the main reproducibility targets used in the paper:

```bash
python main.py --target all
```

Or run grouped domains individually:

```bash
python main.py --target laws
python main.py --target geometry
python main.py --target proto-atoms
python main.py --target magnetism
python main.py --target orbital
python main.py --target cosmology
```

---

## Final states

Selected .npy files are provided in results/final_states/ as reference end states for some key regimes. These files are included to facilitate rapid figure regeneration, while the corresponding scripts can also recompute them from scratch.

---

## Calibration

Although the model is formulated in abstract units, it naturally supports physically meaningful calibration.

In particular, the framework provides sufficient structure to enable external quantitative testing against known physical scales. Calibration is not imposed a priori, but can be derived from emergent structures and propagation regimes.

---

## Notes

The framework explores the emergence of wave-like, structured, flux-like, geometric, proto-atomic, magnetic, orbital and cosmological regimes from a unified discrete energetic substrate.

This repository is organized to support both direct reproducibility and further extension.

---

## License

MIT License

---

## Citation

If you use this code or build upon this framework, please cite the associated manuscript.
---

## Notes

The framework explores the emergence of wave-like, structured, flux-like, geometric, proto-atomic, magnetic, orbital and cosmological regimes from a unified discrete energetic substrate.

This repository is organized to make both direct reproduction and further extension possible.

---

## License

MIT License

---

## Citation

If you use this code or build upon this framework, please cite the associated manuscript.
