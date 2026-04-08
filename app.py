#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import subprocess
import sys
from pathlib import Path

import streamlit as st

PYTHON = sys.executable

DOMAINS = {
    "Substantial oscillation": {
        "script": "src/app_variants/substantial_oscillation_app.py",
        "description": (
            "Interactive single-pair Dirac beat explorer based on the Research 1 "
            "single-pair scan. This page reconstructs the fundamental oscillatory "
            "regime and its fermionic periodicity."
        ),
        "params": {
            "case_index": {"type": "int", "min": 0, "max": 8, "default": 0, "step": 1},
            "pair_index": {"type": "int", "min": 1, "max": 8, "default": 1, "step": 1},
            "t_max": {"type": "float", "min": 20.0, "max": 400.0, "default": 200.0, "step": 10.0},
            "n_steps": {"type": "int", "min": 500, "max": 6000, "default": 4000, "step": 500},
        },
        "figures": [
            "figures/app_substantial_oscillation.png",
            "figures/app_substantial_oscillation_components.png",
        ],
        "presets": {
            "Reference scan A": {
                "beta_index": 0,
                "seed_index": 0,
                "pair_index": 1,
                "t_max": 200.0,
                "n_steps": 4000,
            },
        },
    },
    "Proto-atom (interactive)": {
        "script": "src/app_variants/proto_atom_render_app.py",
        "description": (
            "Interactive proto-atomic render using the real research code adapted "
            "for parameterized launch."
        ),
        "params": {
            "size": {"type": "int", "min": 64, "max": 256, "default": 128, "step": 32},
            "n_steps": {"type": "int", "min": 50, "max": 300, "default": 180, "step": 10},
            "seed": {"type": "int", "min": 1, "max": 20, "default": 3, "step": 1},
            "beta": {"type": "float", "min": 5.0, "max": 12.0, "default": 8.75, "step": 0.05},
            "center_gain": {"type": "float", "min": 0.0, "max": 0.05, "default": 0.012, "step": 0.001},
            "node_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.100, "step": 0.001},
            "matter_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.098, "step": 0.001},
            "omega_bg": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.22, "step": 0.01},
            "background_gain": {"type": "float", "min": 0.0, "max": 0.1, "default": 0.035, "step": 0.001},
            "omega_local": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.47, "step": 0.01},
            "local_beat_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.085, "step": 0.001},
            "flux_gain": {"type": "float", "min": 0.0, "max": 0.1, "default": 0.045, "step": 0.001},
            "edge_penalty": {"type": "float", "min": 0.0, "max": 0.3, "default": 0.10, "step": 0.01},
        },
        "figures": [
            "figures/app_proto_atom_render.png",
            "figures/app_proto_atom_render_diagnostics.png",
        ],
        "presets": {
            "Reference render A": {
                "size": 128,
                "n_steps": 180,
                "seed": 3,
                "beta": 8.75,
                "center_gain": 0.012,
                "node_gain": 0.100,
                "matter_gain": 0.098,
                "omega_bg": 0.22,
                "background_gain": 0.035,
                "omega_local": 0.47,
                "local_beat_gain": 0.085,
                "flux_gain": 0.045,
                "edge_penalty": 0.10,
            },
            "Reference render B": {
                "size": 128,
                "n_steps": 180,
                "seed": 4,
                "beta": 8.50,
                "center_gain": 0.014,
                "node_gain": 0.085,
                "matter_gain": 0.104,
                "omega_bg": 0.22,
                "background_gain": 0.035,
                "omega_local": 0.47,
                "local_beat_gain": 0.085,
                "flux_gain": 0.040,
                "edge_penalty": 0.12,
            },
        },
    },
    "Proto-atom dipole": {
        "script": "src/app_variants/proto_atom_dipole_interaction_app.py",
        "description": (
            "Interactive dipole interaction using the real dipole interaction "
            "research code adapted for parameterized launch."
        ),
        "params": {
            "size": {"type": "int", "min": 128, "max": 320, "default": 256, "step": 32},
            "n_steps": {"type": "int", "min": 120, "max": 480, "default": 360, "step": 20},
            "distance": {"type": "float", "min": 20.0, "max": 120.0, "default": 60.0, "step": 2.0},
            "angle_deg": {"type": "float", "min": 0.0, "max": 180.0, "default": 0.0, "step": 5.0},
            "flip_b": {"type": "bool", "default": False},
        },
        "figures": [
            "figures/app_proto_dipole.png",
            "figures/app_proto_dipole_snapshots.png",
        ],
        "presets": {
            "Parallel close": {
                "size": 256,
                "n_steps": 360,
                "distance": 40.0,
                "angle_deg": 32.0,
                "flip_b": False,
            },
            "Antiparallel close": {
                "size": 256,
                "n_steps": 360,
                "distance": 40.0,
                "angle_deg": 32.0,
                "flip_b": True,
            },
            "Parallel far": {
                "size": 256,
                "n_steps": 360,
                "distance": 70.0,
                "angle_deg": 32.0,
                "flip_b": False,
            },
        },
    },
}


def extract_float(text: str, name: str):
    m = re.search(fr"{name}=([0-9eE\.\-]+)", text)
    return float(m.group(1)) if m else None


def init_session_state() -> None:
    if "report" not in st.session_state:
        st.session_state.report = {
            "stdout": "",
            "stderr": "",
            "metrics": {},
            "domain": None,
            "error": None,
        }


st.set_page_config(page_title="OMM-SOT Explorer", layout="wide")
init_session_state()

st.title("OMM-SOT Interactive Explorer")
st.markdown(
    "This interface launches interactive variants of selected research scripts, "
    "so that users can modify parameters and regenerate actual outputs."
)

domain = st.selectbox("Scientific domain", list(DOMAINS.keys()))
entry = DOMAINS[domain]

st.subheader(domain)
st.write(entry["description"])

st.markdown("**Executed script**")
st.code(f"{PYTHON} {entry['script']} ...")

preset_names = ["Custom"] + list(entry.get("presets", {}).keys())
selected_preset = st.selectbox("Preset", preset_names)
preset_values = entry["presets"][selected_preset] if selected_preset != "Custom" else {}

st.markdown("**Parameters**")
values = {}

for key, spec in entry["params"].items():
    default_value = preset_values.get(key, spec["default"])

    if spec["type"] == "bool":
        values[key] = st.checkbox(key, value=bool(default_value))
    elif spec["type"] == "int":
        values[key] = st.slider(
            key,
            int(spec["min"]),
            int(spec["max"]),
            int(default_value),
            step=int(spec.get("step", 1)),
        )
    else:
        values[key] = st.slider(
            key,
            float(spec["min"]),
            float(spec["max"]),
            float(default_value),
            step=float(spec.get("step", 0.01)),
        )

if st.button("Run simulation", width="stretch"):
    cmd = [PYTHON, entry["script"]]

    for key, value in values.items():
        if isinstance(value, bool):
            cmd.extend([f"--{key}", "1" if value else "0"])
        else:
            cmd.extend([f"--{key}", str(value)])

    st.info("Running interactive research variant...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        st.session_state.report = {
            "stdout": stdout,
            "stderr": stderr,
            "metrics": {},
            "domain": domain,
            "error": f"Simulation failed with return code {result.returncode}.",
        }
        st.error(st.session_state.report["error"])
    else:
        metrics = {}
        if domain == "Substantial oscillation":
            metrics = {
                "beta": extract_float(stdout, "beta"),
                "seed": extract_float(stdout, "seed"),
                "pair_index": extract_float(stdout, "pair_index"),
                "abs_energy": extract_float(stdout, "abs_energy"),
                "pairing_error": extract_float(stdout, "pairing_error"),
                "dominant_frequency": extract_float(stdout, "dominant_frequency"),
                "expected_pair_frequency": extract_float(stdout, "expected_pair_frequency"),
                "frequency_ratio": extract_float(stdout, "frequency_ratio"),
                "relative_peak_strength": extract_float(stdout, "relative_peak_strength"),
                "recurrence_quality": extract_float(stdout, "recurrence_quality"),
                "dirac_dim": extract_float(stdout, "dirac_dim"),
                "n_nodes": extract_float(stdout, "n_nodes"),
                "n_edges": extract_float(stdout, "n_edges"),
            }
        elif domain == "Proto-atom (interactive)":
            metrics = {
                "stability_index": extract_float(stdout, "stability_index"),
                "final_std": extract_float(stdout, "final_std"),
                "centroid_offset": extract_float(stdout, "centroid_offset"),
                "anisotropy": extract_float(stdout, "anisotropy"),
                "core_mass_fraction": extract_float(stdout, "core_mass_fraction"),
                "split_score": extract_float(stdout, "split_score"),
            }
        elif domain == "Proto-atom dipole":
            metrics = {
                "interaction_score": extract_float(stdout, "interaction_score"),
                "final_distance": extract_float(stdout, "final_distance"),
                "dipole_amplitude": extract_float(stdout, "dipole_amplitude"),
                "anisotropy_ratio": extract_float(stdout, "anisotropy_ratio"),
                "major_axis_angle_deg": extract_float(stdout, "major_axis_angle_deg"),
                "dipole_angle_deg": extract_float(stdout, "dipole_angle_deg"),
            }

        st.session_state.report = {
            "stdout": stdout,
            "stderr": stderr,
            "metrics": metrics,
            "domain": domain,
            "error": None,
        }
        st.success("Simulation completed.")

report = st.session_state.report

st.subheader("Report")

if report["domain"] != domain or not report["metrics"]:
    st.info("Run a simulation to generate a report.")
else:
    if domain == "Substantial oscillation":
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Dominant frequency",
                f"{report['metrics'].get('dominant_frequency'):.4f}"
                if report["metrics"].get("dominant_frequency") is not None else "—",
            )
            st.metric(
                "Expected pair frequency",
                f"{report['metrics'].get('expected_pair_frequency'):.4f}"
                if report["metrics"].get("expected_pair_frequency") is not None else "—",
            )
            st.metric(
                "Frequency ratio",
                f"{report['metrics'].get('frequency_ratio'):.4f}"
                if report["metrics"].get("frequency_ratio") is not None else "—",
            )
        with col2:
            st.metric(
                "Recurrence quality",
                f"{report['metrics'].get('recurrence_quality'):.4f}"
                if report["metrics"].get("recurrence_quality") is not None else "—",
            )
            st.metric(
                "Relative peak strength",
                f"{report['metrics'].get('relative_peak_strength'):.4f}"
                if report["metrics"].get("relative_peak_strength") is not None else "—",
            )
            st.metric(
                "Abs energy",
                f"{report['metrics'].get('abs_energy'):.4f}"
                if report["metrics"].get("abs_energy") is not None else "—",
            )

        st.markdown("### Structural context")
        st.write({
            "beta": report["metrics"].get("beta"),
            "seed": report["metrics"].get("seed"),
            "pair_index": report["metrics"].get("pair_index"),
            "dirac_dim": report["metrics"].get("dirac_dim"),
            "n_nodes": report["metrics"].get("n_nodes"),
            "n_edges": report["metrics"].get("n_edges"),
        })

    elif domain == "Proto-atom (interactive)":
        col1, col2 = st.columns(2)
        with col1:
            score = report["metrics"].get("stability_index")
            st.metric("Stability score", f"{score:.3f}" if score is not None else "—")
            final_std = report["metrics"].get("final_std")
            st.metric("Final std", f"{final_std:.3f}" if final_std is not None else "—")
        with col2:
            offset = report["metrics"].get("centroid_offset")
            st.metric("Centroid offset", f"{offset:.3f}" if offset is not None else "—")
            anis = report["metrics"].get("anisotropy")
            st.metric("Anisotropy", f"{anis:.3f}" if anis is not None else "—")

        st.markdown("### Structural metrics")
        st.write({
            "core_mass_fraction": report["metrics"].get("core_mass_fraction"),
            "split_score": report["metrics"].get("split_score"),
        })

        st.markdown("### Stability gradient")
        if score is None:
            st.progress(0.0)
        else:
            normalized = max(0.0, min(1.0, 1.0 - score / 1.5))
            st.progress(normalized)
            st.caption("Higher bar = lower instability score")

    elif domain == "Proto-atom dipole":
        col1, col2 = st.columns(2)
        with col1:
            s = report["metrics"].get("interaction_score")
            st.metric("Interaction score", f"{s:.3f}" if s is not None else "—")
            d = report["metrics"].get("final_distance")
            st.metric("Final distance", f"{d:.3f}" if d is not None else "—")
        with col2:
            dip = report["metrics"].get("dipole_amplitude")
            st.metric("Dipole amplitude", f"{dip:.3f}" if dip is not None else "—")
            anis = report["metrics"].get("anisotropy_ratio")
            st.metric("Anisotropy ratio", f"{anis:.3f}" if anis is not None else "—")

        st.markdown("### Angular metrics")
        st.write({
            "major_axis_angle_deg": report["metrics"].get("major_axis_angle_deg"),
            "dipole_angle_deg": report["metrics"].get("dipole_angle_deg"),
        })

        st.markdown("### Interaction gradient")
        if s is None:
            st.progress(0.0)
        else:
            normalized = max(0.0, min(1.0, s))
            st.progress(normalized)
            st.caption("Higher bar = stronger interaction score")

st.subheader("Expected output figures")
for fig in entry["figures"]:
    st.code(fig)

st.subheader("Preview")
existing = [Path(fig) for fig in entry["figures"] if Path(fig).exists()]
if not existing:
    st.warning("No preview figure currently found.")
else:
    cols = st.columns(2)
    for i, path in enumerate(existing):
        with cols[i % 2]:
            st.image(str(path), caption=path.name, width="stretch")

with st.expander("Raw simulation output"):
    if report["stdout"]:
        st.text(report["stdout"])
    if report["stderr"]:
        st.text(report["stderr"])

st.markdown("---")
st.caption(
    "These interactive variants should remain faithful to the original research scripts. "
    "They are intended for controlled parameter exploration, not for replacing the reference paper pipeline."
)                "n_steps": 4000,
            },
        },
    },
    "Proto-atom (interactive)": {
        "script": "src/app_variants/proto_atom_render_app.py",
        "description": (
            "Interactive proto-atomic render using the real research code adapted "
            "for parameterized launch."
        ),
        "params": {
            "size": {"type": "int", "min": 64, "max": 256, "default": 128, "step": 32},
            "n_steps": {"type": "int", "min": 50, "max": 300, "default": 180, "step": 10},
            "seed": {"type": "int", "min": 1, "max": 20, "default": 3, "step": 1},
            "beta": {"type": "float", "min": 5.0, "max": 12.0, "default": 8.75, "step": 0.05},
            "center_gain": {"type": "float", "min": 0.0, "max": 0.05, "default": 0.012, "step": 0.001},
            "node_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.100, "step": 0.001},
            "matter_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.098, "step": 0.001},
            "omega_bg": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.22, "step": 0.01},
            "background_gain": {"type": "float", "min": 0.0, "max": 0.1, "default": 0.035, "step": 0.001},
            "omega_local": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.47, "step": 0.01},
            "local_beat_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.085, "step": 0.001},
            "flux_gain": {"type": "float", "min": 0.0, "max": 0.1, "default": 0.045, "step": 0.001},
            "edge_penalty": {"type": "float", "min": 0.0, "max": 0.3, "default": 0.10, "step": 0.01},
        },
        "figures": [
            "figures/app_proto_atom_render.png",
            "figures/app_proto_atom_render_diagnostics.png",
        ],
        "presets": {
            "Reference render A": {
                "size": 128,
                "n_steps": 180,
                "seed": 3,
                "beta": 8.75,
                "center_gain": 0.012,
                "node_gain": 0.100,
                "matter_gain": 0.098,
                "omega_bg": 0.22,
                "background_gain": 0.035,
                "omega_local": 0.47,
                "local_beat_gain": 0.085,
                "flux_gain": 0.045,
                "edge_penalty": 0.10,
            },
            "Reference render B": {
                "size": 128,
                "n_steps": 180,
                "seed": 4,
                "beta": 8.50,
                "center_gain": 0.014,
                "node_gain": 0.085,
                "matter_gain": 0.104,
                "omega_bg": 0.22,
                "background_gain": 0.035,
                "omega_local": 0.47,
                "local_beat_gain": 0.085,
                "flux_gain": 0.040,
                "edge_penalty": 0.12,
            },
        },
    },
    "Proto-atom dipole": {
        "script": "src/app_variants/proto_atom_dipole_interaction_app.py",
        "description": (
            "Interactive dipole interaction using the real dipole interaction "
            "research code adapted for parameterized launch."
        ),
        "params": {
            "size": {"type": "int", "min": 128, "max": 320, "default": 256, "step": 32},
            "n_steps": {"type": "int", "min": 120, "max": 480, "default": 360, "step": 20},
            "distance": {"type": "float", "min": 20.0, "max": 120.0, "default": 60.0, "step": 2.0},
            "angle_deg": {"type": "float", "min": 0.0, "max": 180.0, "default": 0.0, "step": 5.0},
            "flip_b": {"type": "bool", "default": False},
        },
        "figures": [
            "figures/app_proto_dipole.png",
            "figures/app_proto_dipole_snapshots.png",
        ],
        "presets": {
            "Parallel close": {
                "size": 256,
                "n_steps": 360,
                "distance": 40.0,
                "angle_deg": 32.0,
                "flip_b": False,
            },
            "Antiparallel close": {
                "size": 256,
                "n_steps": 360,
                "distance": 40.0,
                "angle_deg": 32.0,
                "flip_b": True,
            },
            "Parallel far": {
                "size": 256,
                "n_steps": 360,
                "distance": 70.0,
                "angle_deg": 32.0,
                "flip_b": False,
            },
        },
    },
}


def extract_float(text: str, name: str):
    m = re.search(fr"{name}=([0-9eE\.\-]+)", text)
    return float(m.group(1)) if m else None


def init_session_state() -> None:
    if "report" not in st.session_state:
        st.session_state.report = {
            "stdout": "",
            "stderr": "",
            "metrics": {},
            "domain": None,
            "error": None,
        }


st.set_page_config(page_title="OMM-SOT Explorer", layout="wide")
init_session_state()

st.title("OMM-SOT Interactive Explorer")
st.markdown(
    "This interface launches interactive variants of selected research scripts, "
    "so that users can modify parameters and regenerate actual outputs."
)

domain = st.selectbox("Scientific domain", list(DOMAINS.keys()))
entry = DOMAINS[domain]

st.subheader(domain)
st.write(entry["description"])

st.markdown("**Executed script**")
st.code(f"{PYTHON} {entry['script']} ...")

preset_names = ["Custom"] + list(entry.get("presets", {}).keys())
selected_preset = st.selectbox("Preset", preset_names)
preset_values = entry["presets"][selected_preset] if selected_preset != "Custom" else {}

repo_root_value = "../unified-emergent-framework"
if domain == "Substantial oscillation":
    repo_root_value = st.text_input(
        "Research 1 repository path",
        value="../unified-emergent-framework",
        help="Path to the unified-emergent-framework repository used for the Dirac beat scan.",
    )

st.markdown("**Parameters**")
values = {}

for key, spec in entry["params"].items():
    default_value = preset_values.get(key, spec["default"])

    if spec["type"] == "bool":
        values[key] = st.checkbox(key, value=bool(default_value))
    elif spec["type"] == "int":
        values[key] = st.slider(
            key,
            int(spec["min"]),
            int(spec["max"]),
            int(default_value),
            step=int(spec.get("step", 1)),
        )
    else:
        values[key] = st.slider(
            key,
            float(spec["min"]),
            float(spec["max"]),
            float(default_value),
            step=float(spec.get("step", 0.01)),
        )

if st.button("Run simulation", width="stretch"):
    cmd = [PYTHON, entry["script"]]
    if domain == "Substantial oscillation":
        cmd.extend(["--repo_root", repo_root_value])

    for key, value in values.items():
        if isinstance(value, bool):
            cmd.extend([f"--{key}", "1" if value else "0"])
        else:
            cmd.extend([f"--{key}", str(value)])

    st.info("Running interactive research variant...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        st.session_state.report = {
            "stdout": stdout,
            "stderr": stderr,
            "metrics": {},
            "domain": domain,
            "error": f"Simulation failed with return code {result.returncode}.",
        }
        st.error(st.session_state.report["error"])
    else:
        metrics = {}
        if domain == "Substantial oscillation":
            metrics = {
                "beta": extract_float(stdout, "beta"),
                "seed": extract_float(stdout, "seed"),
                "pair_index": extract_float(stdout, "pair_index"),
                "abs_energy": extract_float(stdout, "abs_energy"),
                "pairing_error": extract_float(stdout, "pairing_error"),
                "dominant_frequency": extract_float(stdout, "dominant_frequency"),
                "expected_pair_frequency": extract_float(stdout, "expected_pair_frequency"),
                "frequency_ratio": extract_float(stdout, "frequency_ratio"),
                "relative_peak_strength": extract_float(stdout, "relative_peak_strength"),
                "recurrence_quality": extract_float(stdout, "recurrence_quality"),
                "dirac_dim": extract_float(stdout, "dirac_dim"),
                "n_nodes": extract_float(stdout, "n_nodes"),
                "n_edges": extract_float(stdout, "n_edges"),
            }
        elif domain == "Proto-atom (interactive)":
            metrics = {
                "stability_index": extract_float(stdout, "stability_index"),
                "final_std": extract_float(stdout, "final_std"),
                "centroid_offset": extract_float(stdout, "centroid_offset"),
                "anisotropy": extract_float(stdout, "anisotropy"),
                "core_mass_fraction": extract_float(stdout, "core_mass_fraction"),
                "split_score": extract_float(stdout, "split_score"),
            }
        elif domain == "Proto-atom dipole":
            metrics = {
                "interaction_score": extract_float(stdout, "interaction_score"),
                "final_distance": extract_float(stdout, "final_distance"),
                "dipole_amplitude": extract_float(stdout, "dipole_amplitude"),
                "anisotropy_ratio": extract_float(stdout, "anisotropy_ratio"),
                "major_axis_angle_deg": extract_float(stdout, "major_axis_angle_deg"),
                "dipole_angle_deg": extract_float(stdout, "dipole_angle_deg"),
            }

        st.session_state.report = {
            "stdout": stdout,
            "stderr": stderr,
            "metrics": metrics,
            "domain": domain,
            "error": None,
        }
        st.success("Simulation completed.")

report = st.session_state.report

st.subheader("Report")

if report["domain"] != domain or not report["metrics"]:
    st.info("Run a simulation to generate a report.")
else:
    if domain == "Substantial oscillation":
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Dominant frequency",
                f"{report['metrics'].get('dominant_frequency'):.4f}"
                if report["metrics"].get("dominant_frequency") is not None else "—",
            )
            st.metric(
                "Expected pair frequency",
                f"{report['metrics'].get('expected_pair_frequency'):.4f}"
                if report["metrics"].get("expected_pair_frequency") is not None else "—",
            )
            st.metric(
                "Frequency ratio",
                f"{report['metrics'].get('frequency_ratio'):.4f}"
                if report["metrics"].get("frequency_ratio") is not None else "—",
            )
        with col2:
            st.metric(
                "Recurrence quality",
                f"{report['metrics'].get('recurrence_quality'):.4f}"
                if report["metrics"].get("recurrence_quality") is not None else "—",
            )
            st.metric(
                "Relative peak strength",
                f"{report['metrics'].get('relative_peak_strength'):.4f}"
                if report["metrics"].get("relative_peak_strength") is not None else "—",
            )
            st.metric(
                "Abs energy",
                f"{report['metrics'].get('abs_energy'):.4f}"
                if report["metrics"].get("abs_energy") is not None else "—",
            )

        st.markdown("### Structural context")
        st.write({
            "beta": report["metrics"].get("beta"),
            "seed": report["metrics"].get("seed"),
            "pair_index": report["metrics"].get("pair_index"),
            "dirac_dim": report["metrics"].get("dirac_dim"),
            "n_nodes": report["metrics"].get("n_nodes"),
            "n_edges": report["metrics"].get("n_edges"),
        })

    elif domain == "Proto-atom (interactive)":
        col1, col2 = st.columns(2)
        with col1:
            score = report["metrics"].get("stability_index")
            st.metric("Stability score", f"{score:.3f}" if score is not None else "—")
            final_std = report["metrics"].get("final_std")
            st.metric("Final std", f"{final_std:.3f}" if final_std is not None else "—")
        with col2:
            offset = report["metrics"].get("centroid_offset")
            st.metric("Centroid offset", f"{offset:.3f}" if offset is not None else "—")
            anis = report["metrics"].get("anisotropy")
            st.metric("Anisotropy", f"{anis:.3f}" if anis is not None else "—")

        st.markdown("### Structural metrics")
        st.write({
            "core_mass_fraction": report["metrics"].get("core_mass_fraction"),
            "split_score": report["metrics"].get("split_score"),
        })

        st.markdown("### Stability gradient")
        if score is None:
            st.progress(0.0)
        else:
            normalized = max(0.0, min(1.0, 1.0 - score / 1.5))
            st.progress(normalized)
            st.caption("Higher bar = lower instability score")

    elif domain == "Proto-atom dipole":
        col1, col2 = st.columns(2)
        with col1:
            s = report["metrics"].get("interaction_score")
            st.metric("Interaction score", f"{s:.3f}" if s is not None else "—")
            d = report["metrics"].get("final_distance")
            st.metric("Final distance", f"{d:.3f}" if d is not None else "—")
        with col2:
            dip = report["metrics"].get("dipole_amplitude")
            st.metric("Dipole amplitude", f"{dip:.3f}" if dip is not None else "—")
            anis = report["metrics"].get("anisotropy_ratio")
            st.metric("Anisotropy ratio", f"{anis:.3f}" if anis is not None else "—")

        st.markdown("### Angular metrics")
        st.write({
            "major_axis_angle_deg": report["metrics"].get("major_axis_angle_deg"),
            "dipole_angle_deg": report["metrics"].get("dipole_angle_deg"),
        })

        st.markdown("### Interaction gradient")
        if s is None:
            st.progress(0.0)
        else:
            normalized = max(0.0, min(1.0, s))
            st.progress(normalized)
            st.caption("Higher bar = stronger interaction score")

st.subheader("Expected output figures")
for fig in entry["figures"]:
    st.code(fig)

st.subheader("Preview")
existing = [Path(fig) for fig in entry["figures"] if Path(fig).exists()]
if not existing:
    st.warning("No preview figure currently found.")
else:
    cols = st.columns(2)
    for i, path in enumerate(existing):
        with cols[i % 2]:
            st.image(str(path), caption=path.name, width="stretch")

with st.expander("Raw simulation output"):
    if report["stdout"]:
        st.text(report["stdout"])
    if report["stderr"]:
        st.text(report["stderr"])

st.markdown("---")
st.caption(
    "These interactive variants should remain faithful to the original research scripts. "
    "They are intended for controlled parameter exploration, not for replacing the reference paper pipeline."
)                "t_max": 200.0,
                "n_steps": 4000,
            },
        },
    },
    "Proto-atom (interactive)": {
        "script": "src/app_variants/proto_atom_render_app.py",
        "description": (
            "Interactive proto-atomic render using the real research code adapted "
            "for parameterized launch."
        ),
        "params": {
            "size": {"type": "int", "min": 64, "max": 256, "default": 128, "step": 32},
            "n_steps": {"type": "int", "min": 50, "max": 300, "default": 180, "step": 10},
            "seed": {"type": "int", "min": 1, "max": 20, "default": 3, "step": 1},
            "beta": {"type": "float", "min": 5.0, "max": 12.0, "default": 8.75, "step": 0.05},
            "center_gain": {"type": "float", "min": 0.0, "max": 0.05, "default": 0.012, "step": 0.001},
            "node_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.100, "step": 0.001},
            "matter_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.098, "step": 0.001},
            "omega_bg": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.22, "step": 0.01},
            "background_gain": {"type": "float", "min": 0.0, "max": 0.1, "default": 0.035, "step": 0.001},
            "omega_local": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.47, "step": 0.01},
            "local_beat_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.085, "step": 0.001},
            "flux_gain": {"type": "float", "min": 0.0, "max": 0.1, "default": 0.045, "step": 0.001},
            "edge_penalty": {"type": "float", "min": 0.0, "max": 0.3, "default": 0.10, "step": 0.01},
        },
        "figures": [
            "figures/app_proto_atom_render.png",
            "figures/app_proto_atom_render_diagnostics.png",
        ],
        "presets": {
            "Reference render A": {
                "size": 128,
                "n_steps": 180,
                "seed": 3,
                "beta": 8.75,
                "center_gain": 0.012,
                "node_gain": 0.100,
                "matter_gain": 0.098,
                "omega_bg": 0.22,
                "background_gain": 0.035,
                "omega_local": 0.47,
                "local_beat_gain": 0.085,
                "flux_gain": 0.045,
                "edge_penalty": 0.10,
            },
            "Reference render B": {
                "size": 128,
                "n_steps": 180,
                "seed": 4,
                "beta": 8.50,
                "center_gain": 0.014,
                "node_gain": 0.085,
                "matter_gain": 0.104,
                "omega_bg": 0.22,
                "background_gain": 0.035,
                "omega_local": 0.47,
                "local_beat_gain": 0.085,
                "flux_gain": 0.040,
                "edge_penalty": 0.12,
            },
        },
    },
    "Proto-atom dipole": {
        "script": "src/app_variants/proto_atom_dipole_interaction_app.py",
        "description": (
            "Interactive dipole interaction using the real dipole interaction "
            "research code adapted for parameterized launch."
        ),
        "params": {
            "size": {"type": "int", "min": 128, "max": 320, "default": 256, "step": 32},
            "n_steps": {"type": "int", "min": 120, "max": 480, "default": 360, "step": 20},
            "distance": {"type": "float", "min": 20.0, "max": 120.0, "default": 60.0, "step": 2.0},
            "angle_deg": {"type": "float", "min": 0.0, "max": 180.0, "default": 0.0, "step": 5.0},
            "flip_b": {"type": "bool", "default": False},
        },
        "figures": [
            "figures/app_proto_dipole.png",
            "figures/app_proto_dipole_snapshots.png",
        ],
        "presets": {
            "Parallel close": {
                "size": 256,
                "n_steps": 360,
                "distance": 40.0,
                "angle_deg": 32.0,
                "flip_b": False,
            },
            "Antiparallel close": {
                "size": 256,
                "n_steps": 360,
                "distance": 40.0,
                "angle_deg": 32.0,
                "flip_b": True,
            },
            "Parallel far": {
                "size": 256,
                "n_steps": 360,
                "distance": 70.0,
                "angle_deg": 32.0,
                "flip_b": False,
            },
        },
    },
}


def extract_float(text: str, name: str):
    m = re.search(fr"{name}=([0-9eE\.\-]+)", text)
    return float(m.group(1)) if m else None


def init_session_state() -> None:
    if "report" not in st.session_state:
        st.session_state.report = {
            "stdout": "",
            "stderr": "",
            "metrics": {},
            "domain": None,
            "error": None,
        }


st.set_page_config(page_title="OMM-SOT Explorer", layout="wide")
init_session_state()

st.title("OMM-SOT Interactive Explorer")
st.markdown(
    "This interface launches interactive variants of selected research scripts, "
    "so that users can modify parameters and regenerate actual outputs."
)

domain = st.selectbox("Scientific domain", list(DOMAINS.keys()))
entry = DOMAINS[domain]

st.subheader(domain)
st.write(entry["description"])

st.markdown("**Executed script**")
st.code(f"{PYTHON} {entry['script']} ...")

preset_names = ["Custom"] + list(entry.get("presets", {}).keys())
selected_preset = st.selectbox("Preset", preset_names)
preset_values = entry["presets"][selected_preset] if selected_preset != "Custom" else {}

repo_root_value = "../unified-emergent-framework"
if domain == "Substantial oscillation":
    repo_root_value = st.text_input(
        "Research 1 repository path",
        value="../unified-emergent-framework",
        help="Path to the unified-emergent-framework repository used for the Dirac beat scan.",
    )

st.markdown("**Parameters**")
values = {}

for key, spec in entry["params"].items():
    default_value = preset_values.get(key, spec["default"])

    if spec["type"] == "bool":
        values[key] = st.checkbox(key, value=bool(default_value))
    elif spec["type"] == "int":
        values[key] = st.slider(
            key,
            int(spec["min"]),
            int(spec["max"]),
            int(default_value),
            step=int(spec.get("step", 1)),
        )
    else:
        values[key] = st.slider(
            key,
            float(spec["min"]),
            float(spec["max"]),
            float(default_value),
            step=float(spec.get("step", 0.01)),
        )

if st.button("Run simulation", width="stretch"):
    cmd = [PYTHON, entry["script"]]
    if domain == "Substantial oscillation":
        cmd.extend(["--repo_root", repo_root_value])

    for key, value in values.items():
        if isinstance(value, bool):
            cmd.extend([f"--{key}", "1" if value else "0"])
        else:
            cmd.extend([f"--{key}", str(value)])

    st.info("Running interactive research variant...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        st.session_state.report = {
            "stdout": stdout,
            "stderr": stderr,
            "metrics": {},
            "domain": domain,
            "error": f"Simulation failed with return code {result.returncode}.",
        }
        st.error(st.session_state.report["error"])
    else:
        metrics = {}
        if domain == "Substantial oscillation":
            metrics = {
                "beta": extract_float(stdout, "beta"),
                "seed": extract_float(stdout, "seed"),
                "pair_index": extract_float(stdout, "pair_index"),
                "abs_energy": extract_float(stdout, "abs_energy"),
                "pairing_error": extract_float(stdout, "pairing_error"),
                "dominant_frequency": extract_float(stdout, "dominant_frequency"),
                "expected_pair_frequency": extract_float(stdout, "expected_pair_frequency"),
                "frequency_ratio": extract_float(stdout, "frequency_ratio"),
                "relative_peak_strength": extract_float(stdout, "relative_peak_strength"),
                "recurrence_quality": extract_float(stdout, "recurrence_quality"),
                "dirac_dim": extract_float(stdout, "dirac_dim"),
                "n_nodes": extract_float(stdout, "n_nodes"),
                "n_edges": extract_float(stdout, "n_edges"),
            }
        elif domain == "Proto-atom (interactive)":
            metrics = {
                "stability_index": extract_float(stdout, "stability_index"),
                "final_std": extract_float(stdout, "final_std"),
                "centroid_offset": extract_float(stdout, "centroid_offset"),
                "anisotropy": extract_float(stdout, "anisotropy"),
                "core_mass_fraction": extract_float(stdout, "core_mass_fraction"),
                "split_score": extract_float(stdout, "split_score"),
            }
        elif domain == "Proto-atom dipole":
            metrics = {
                "interaction_score": extract_float(stdout, "interaction_score"),
                "final_distance": extract_float(stdout, "final_distance"),
                "dipole_amplitude": extract_float(stdout, "dipole_amplitude"),
                "anisotropy_ratio": extract_float(stdout, "anisotropy_ratio"),
                "major_axis_angle_deg": extract_float(stdout, "major_axis_angle_deg"),
                "dipole_angle_deg": extract_float(stdout, "dipole_angle_deg"),
            }

        st.session_state.report = {
            "stdout": stdout,
            "stderr": stderr,
            "metrics": metrics,
            "domain": domain,
            "error": None,
        }
        st.success("Simulation completed.")

report = st.session_state.report

st.subheader("Report")

if report["domain"] != domain or not report["metrics"]:
    st.info("Run a simulation to generate a report.")
else:
    if domain == "Substantial oscillation":
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Dominant frequency",
                f"{report['metrics'].get('dominant_frequency'):.4f}"
                if report["metrics"].get("dominant_frequency") is not None else "—",
            )
            st.metric(
                "Expected pair frequency",
                f"{report['metrics'].get('expected_pair_frequency'):.4f}"
                if report["metrics"].get("expected_pair_frequency") is not None else "—",
            )
            st.metric(
                "Frequency ratio",
                f"{report['metrics'].get('frequency_ratio'):.4f}"
                if report["metrics"].get("frequency_ratio") is not None else "—",
            )
        with col2:
            st.metric(
                "Recurrence quality",
                f"{report['metrics'].get('recurrence_quality'):.4f}"
                if report["metrics"].get("recurrence_quality") is not None else "—",
            )
            st.metric(
                "Relative peak strength",
                f"{report['metrics'].get('relative_peak_strength'):.4f}"
                if report["metrics"].get("relative_peak_strength") is not None else "—",
            )
            st.metric(
                "Abs energy",
                f"{report['metrics'].get('abs_energy'):.4f}"
                if report["metrics"].get("abs_energy") is not None else "—",
            )

        st.markdown("### Structural context")
        st.write({
            "beta": report["metrics"].get("beta"),
            "seed": report["metrics"].get("seed"),
            "pair_index": report["metrics"].get("pair_index"),
            "dirac_dim": report["metrics"].get("dirac_dim"),
            "n_nodes": report["metrics"].get("n_nodes"),
            "n_edges": report["metrics"].get("n_edges"),
        })

    elif domain == "Proto-atom (interactive)":
        col1, col2 = st.columns(2)
        with col1:
            score = report["metrics"].get("stability_index")
            st.metric("Stability score", f"{score:.3f}" if score is not None else "—")
            final_std = report["metrics"].get("final_std")
            st.metric("Final std", f"{final_std:.3f}" if final_std is not None else "—")
        with col2:
            offset = report["metrics"].get("centroid_offset")
            st.metric("Centroid offset", f"{offset:.3f}" if offset is not None else "—")
            anis = report["metrics"].get("anisotropy")
            st.metric("Anisotropy", f"{anis:.3f}" if anis is not None else "—")

        st.markdown("### Structural metrics")
        st.write({
            "core_mass_fraction": report["metrics"].get("core_mass_fraction"),
            "split_score": report["metrics"].get("split_score"),
        })

        st.markdown("### Stability gradient")
        if score is None:
            st.progress(0.0)
        else:
            normalized = max(0.0, min(1.0, 1.0 - score / 1.5))
            st.progress(normalized)
            st.caption("Higher bar = lower instability score")

    elif domain == "Proto-atom dipole":
        col1, col2 = st.columns(2)
        with col1:
            s = report["metrics"].get("interaction_score")
            st.metric("Interaction score", f"{s:.3f}" if s is not None else "—")
            d = report["metrics"].get("final_distance")
            st.metric("Final distance", f"{d:.3f}" if d is not None else "—")
        with col2:
            dip = report["metrics"].get("dipole_amplitude")
            st.metric("Dipole amplitude", f"{dip:.3f}" if dip is not None else "—")
            anis = report["metrics"].get("anisotropy_ratio")
            st.metric("Anisotropy ratio", f"{anis:.3f}" if anis is not None else "—")

        st.markdown("### Angular metrics")
        st.write({
            "major_axis_angle_deg": report["metrics"].get("major_axis_angle_deg"),
            "dipole_angle_deg": report["metrics"].get("dipole_angle_deg"),
        })

        st.markdown("### Interaction gradient")
        if s is None:
            st.progress(0.0)
        else:
            normalized = max(0.0, min(1.0, s))
            st.progress(normalized)
            st.caption("Higher bar = stronger interaction score")

st.subheader("Expected output figures")
for fig in entry["figures"]:
    st.code(fig)

st.subheader("Preview")
existing = [Path(fig) for fig in entry["figures"] if Path(fig).exists()]
if not existing:
    st.warning("No preview figure currently found.")
else:
    cols = st.columns(2)
    for i, path in enumerate(existing):
        with cols[i % 2]:
            st.image(str(path), caption=path.name, width="stretch")

with st.expander("Raw simulation output"):
    if report["stdout"]:
        st.text(report["stdout"])
    if report["stderr"]:
        st.text(report["stderr"])

st.markdown("---")
st.caption(
    "These interactive variants should remain faithful to the original research scripts. "
    "They are intended for controlled parameter exploration, not for replacing the reference paper pipeline."
)
