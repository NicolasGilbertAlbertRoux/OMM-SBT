#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
            "beta": {"type": "float", "min": 5.0, "max": 12.0, "default": 8.75},
            "center_gain": {"type": "float", "min": 0.0, "max": 0.05, "default": 0.012},
            "node_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.100},
            "matter_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.098},
            "omega_bg": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.22},
            "background_gain": {"type": "float", "min": 0.0, "max": 0.1, "default": 0.035},
            "omega_local": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.47},
            "local_beat_gain": {"type": "float", "min": 0.0, "max": 0.2, "default": 0.085},
            "flux_gain": {"type": "float", "min": 0.0, "max": 0.1, "default": 0.045},
            "edge_penalty": {"type": "float", "min": 0.0, "max": 0.3, "default": 0.10},
        },
        "figures": [
            "figures/app_proto_atom_render.png",
            "figures/app_proto_atom_render_diagnostics.png",
        ],
    },
}

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
