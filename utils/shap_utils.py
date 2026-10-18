"""SHAP level kata. Masker berbasis kata dipakai agar adil antar tokenizer berbeda."""

import numpy as np
import shap
import streamlit as st

from .config import CLICKBAIT_LABEL


@st.cache_resource(show_spinner=False)
def get_explainer(_pipe, key: str):
    masker = shap.maskers.Text(tokenizer=r"\W+")
    return shap.Explainer(_pipe, masker)


def explain(explainer, pipe, text: str):
    """Kembalikan (daftar kata, nilai SHAP untuk kelas clickbait)."""
    sv = explainer([text])
    try:
        idx = list(sv.output_names).index(CLICKBAIT_LABEL)
    except Exception:
        idx = pipe.model.config.label2id.get(CLICKBAIT_LABEL, 1)
    tokens = list(sv.data[0])
    values = np.array(sv.values[0])[:, idx]
    return tokens, values
