# Oscillatory Mantle Model (OMM)

## Toward a Substantial Oscillation Theory (SOT)

This repository contains the code, figures, and reproducibility pipeline associated with the manuscript:

**An Emergent Field Theory of Physical Structures: The Oscillatory Mantle Model (OMM)**

Git repository:
`https://github.com/NicolasGilbertAlbertRoux/OMM-SOT.git`

---

## Scope of the repository

This repository serves **three distinct purposes**:

1. **Paper reproduction**  
   Reproduce the main figures and numerical outputs directly used in the manuscript.

2. **Domain exploration**  
   Run broader domain-specific experiments beyond the strict paper subset:
   laws, geometry, proto-atoms, magnetism, orbital regimes, and cosmology.

3. **Interactive local exploration**  
   Launch a lightweight Streamlit interface that executes real simulation scripts from the repository.

These three usages should not be confused:
- the **paper pipeline** is the most conservative and publication-oriented subset;
- the **domain groups** are broader research explorations;
- the **interactive app** is an exploration tool, not the canonical manuscript pipeline.

---

## Repository structure

- `src/core/` — core scripts used for the main conceptual figures and mantle tests
- `src/laws/` — law-emergence tests and reduced interaction regimes
- `src/geometry/` — geometric and geodesic emergence
- `src/cosmology/` — large-scale mantle / expansion-like behavior
- `src/proto_atoms/` — proto-atomic, dipolar, molecular, and classification regimes
- `src/magnetism/` — magnetic alignment and domain interactions
- `src/orbital/` — orbital and quasi-orbital tests
- `src/app_variants/` — parameterized local variants used by the interactive explorer
- `src/diagnostics/` — support and post-processing scripts
- `data/` — selected input data required for some exploratory regimes
- `figures/` — generated figures retained for the manuscript and repository
- `results/` — numerical outputs, summaries, and selected final states
- `main.py` — command-line launcher for reproducibility and grouped domain runs
- `app.py` — local Streamlit exploration interface

---

## Installation

Create a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

If you want to use the interactive interface, make sure Streamlit is installed from requirements.txt as well.

---

## Recommended entry points

1) Reproduce the paper subset

Run the conservative paper-oriented pipeline:

```bash
python main.py --target all
```

This launches the principal manuscript-facing targets:
	- unified chain diagram
	- regime map
	- lifecycle figure suite
	- mantle instability test
	- cosmology scan

This is the best starting point for a reader who wants to reproduce the main paper outputs.

2) Explore the project by scientific domain

You can run broader grouped explorations:

```bash
python main.py --target core
python main.py --target laws
python main.py --target geometry
python main.py --target proto-atoms
python main.py --target magnetism
python main.py --target orbital
python main.py --target cosmology
python main.py --target showcase
```

Domain meanings
	- core — main conceptual figures and mantle-level structural tests
	- laws — interaction laws and reduced force / flux regimes
	- geometry — effective, dynamic, and two-scale emergent geometry
	- proto-atoms — stable structures, dipoles, classification, molecular and family-level regimes
	- magnetism — magnetic alignment, domains, and sector-level effects
	- orbital — orbital and rotational exploratory tests
	- cosmology — large-scale mantle feedback and expansion-like behavior
	- showcase — a compact cross-domain demonstration run

3) Launch the interactive explorer

A lightweight local interface is also provided:

```bash
streamlit run app.py
```

This app launches real scripts from the repository through parameterized local variants and previews the figures they generate.

The app is intended for:
	- controlled parameter exploration,
	- pedagogical inspection of selected regimes,
	- local testing by readers and researchers.

It is not the canonical paper reproduction pipeline.

---

## Listing available targets

To list all available command-line targets:

```bash
python main.py --list
```

To run a single target:

```bash
python main.py --target chain
```

---

## Final states and cached outputs

Selected .npy files are provided in results/final_states/ for some key regimes.
These files are included to facilitate rapid figure regeneration, while the corresponding scripts can also recompute results from scratch when applicable.

This means the repository supports both:
	- direct replay from preserved states, and
	- fresh numerical regeneration for the relevant scripts.

---

## Calibration and relation to physical scales

The framework is formulated in abstract simulation units, but it supports physically meaningful calibration strategies.

In particular:
	- calibration is not imposed a priori;
	- emergent structures and propagation regimes can be used to define possible scale mappings;
	- the framework provides sufficient structure to enable external quantitative testing.

This repository therefore supports:
	- qualitative structural exploration,
	- reproducible numerical experimentation,
	- and possible downstream quantitative testing by external researchers.

The calibration logic should be understood as structured and physically motivated, but still open to further external testing and refinement.

---

## Reproducibility philosophy

The repository is organized around a simple principle:

first obtain the emergent structure, then investigate how physical calibration may be assigned to it.

In other words:
	- the model is not hard-wired to a specific physical scale from the start;
	- scale assignment is treated as a downstream interpretative and testable step.

This is intentional and central to the logic of the framework.

---

## Important caution for interpretation

This repository contains a large number of exploratory numerical experiments across several physical sectors.

Accordingly:
	- not every script has the same publication status;
	- the all target is the most conservative manuscript-oriented subset;
	- broader grouped targets and app variants should be interpreted as research exploration tools.

Users should therefore distinguish between:
	- paper-facing results,
	- extended exploratory runs,
	- and interactive pedagogical variants.

---

## License

MIT License

---

## Citation

If you use this code or build upon this framework, please cite the associated manuscript and this repository.

Suggested citation block can be added here once the manuscript metadata is finalized.
---

## Interactive explorer

A lightweight local interface is also provided:

```bash
streamlit run app.py
```

This interface launches actual simulation scripts from the repository and previews the real figures they generate.

Current domains include:
	- proto-atomic stable render
	- proto-periodic classification
	- dipole / binding regime
	- magnetic alignment
	- orbital regime
	- two-scale geometry
	- cosmology

The current version is intentionally conservative: it prioritizes honest execution of the real research code over decorative interactivity.

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
