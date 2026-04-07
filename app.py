#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import subprocess
import sys
from pathlib import Path

import streamlit as st

PYTHON = sys.executable

DOMAINS = {
    "Proto-atom (interactive)": {
        "script": "src/app_variants/proto_atom_render_app.py",
        "description": "Interactive proto-atomic render using the real research code adapted for parameterized launch.",
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
}


def extract_float(text: str, name: str):
    m = re.search(fr"{name}=([0-9eE\.\-]+)", text)
    return float(m.group(1)) if m else None


def init_session_state() -> None:
    if "report" not in st.session_state:
        st.session_state.report = {
            "stability": "not yet run",
            "score": None,
            "final_std": None,
            "offset": None,
            "anis": None,
            "core": None,
            "split": None,
            "stdout": "",
            "stderr": "",
        }

    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = None


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

# Presets
preset_names = ["Custom"] + list(entry.get("presets", {}).keys())
selected_preset = st.selectbox("Preset", preset_names)

if selected_preset != "Custom":
    preset_values = entry["presets"][selected_preset]
else:
    preset_values = {}

st.markdown("**Parameters**")
values = {}

for key, spec in entry["params"].items():
    step = spec.get("step", None)
    default_value = preset_values.get(key, spec["default"])

    if spec["type"] == "int":
        values[key] = st.slider(
            key,
            int(spec["min"]),
            int(spec["max"]),
            int(default_value),
            step=int(step) if step is not None else 1,
        )
    else:
        values[key] = st.slider(
            key,
            float(spec["min"]),
            float(spec["max"]),
            float(default_value),
            step=float(step) if step is not None else None,
        )

if st.button("Run simulation", width="stretch"):
    cmd = [PYTHON, entry["script"]]
    for key, value in values.items():
        cmd.extend([f"--{key}", str(value)])

    st.info("Running interactive research variant...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        st.session_state.report = {
            "stability": "error",
            "score": None,
            "final_std": None,
            "offset": None,
            "anis": None,
            "core": None,
            "split": None,
            "stdout": stdout,
            "stderr": stderr,
        }
        st.error(f"Simulation failed with return code {result.returncode}.")
    else:
        stability = re.search(r"stability_class=(\w+)", stdout)
        st.session_state.report = {
            "stability": stability.group(1) if stability else "unknown",
            "score": extract_float(stdout, "stability_index"),
            "final_std": extract_float(stdout, "final_std"),
            "offset": extract_float(stdout, "centroid_offset"),
            "anis": extract_float(stdout, "anisotropy"),
            "core": extract_float(stdout, "core_mass_fraction"),
            "split": extract_float(stdout, "split_score"),
            "stdout": stdout,
            "stderr": stderr,
        }
        st.success("Simulation completed.")

report = st.session_state.report

st.subheader("Stability report")

col1, col2 = st.columns(2)

with col1:
    st.metric("Stability class", report["stability"])
    st.metric(
        "Stability index",
        f"{report['score']:.3f}" if report["score"] is not None else "—",
    )

with col2:
    st.metric(
        "Final std",
        f"{report['final_std']:.3f}" if report["final_std"] is not None else "—",
    )
    st.metric(
        "Offset",
        f"{report['offset']:.3f}" if report["offset"] is not None else "—",
    )

st.markdown("### Structural metrics")
st.write({
    "anisotropy": report["anis"],
    "core_mass_fraction": report["core"],
    "split_score": report["split"],
})

# Simple visual stability bar
st.markdown("### Stability gradient")
score = report["score"]
if score is None:
    st.progress(0)
    st.caption("Run a simulation to compute stability.")
else:
    normalized = max(0.0, min(1.0, score / 3.0))
    st.progress(normalized)
    if report["stability"] == "stable":
        st.caption("Stable regime")
    elif report["stability"] == "meta-stable":
        st.caption("Meta-stable regime")
    elif report["stability"] == "unstable":
        st.caption("Unstable regime")
    else:
        st.caption("Unknown regime")

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
st.set_page_config(page_title="OMM-SOT Explorer", layout="wide")

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

st.markdown("**Parameters**")
values = {}
for key, spec in entry["params"].items():
    step = spec.get("step", None)
    if spec["type"] == "int":
        values[key] = st.slider(
            key,
            int(spec["min"]),
            int(spec["max"]),
            int(spec["default"]),
            step=int(step) if step is not None else 1,
        )
    else:
        values[key] = st.slider(
            key,
            float(spec["min"]),
            float(spec["max"]),
            float(spec["default"]),
        )

if st.button("Run simulation", width="stretch"):
    cmd = [PYTHON, entry["script"]]
    for key, value in values.items():
        cmd.extend([f"--{key}", str(value)])

    st.info("Running interactive research variant...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        text = result.stdout
        import re

        def extract(name):
            m = re.search(fr"{name}=([0-9eE\.\-]+)", text)
            return float(m.group(1)) if m else None

        final_std = extract("final_std")
        offset = extract("centroid_offset")
        anis = extract("anisotropy")
        core = extract("core_mass_fraction")
        split = extract("split_score")
        score = extract("stability_index")

        m = re.search(r"stability_class=(\w+)", text)
        stability = m.group(1) if m else "unknown"
        st.success("Simulation completed.")
    except subprocess.CalledProcessError as exc:
        st.error(f"Simulation failed with return code {exc.returncode}.")

st.subheader("Stability report")

col1, col2 = st.columns(2)

with col1:
    st.metric("Stability class", stability)
    st.metric("Stability index", f"{score:.3f}")

with col2:
    st.metric("Final std", f"{final_std:.3f}")
    st.metric("Offset", f"{offset:.3f}")

st.markdown("### Structural metrics")
st.write({
    "anisotropy": anis,
    "core_mass_fraction": core,
    "split_score": split,
})

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

st.markdown("---")
st.caption(
    "These interactive variants should remain faithful to the original research scripts. "
    "They are intended for controlled parameter exploration, not for replacing the reference paper pipeline."
)
