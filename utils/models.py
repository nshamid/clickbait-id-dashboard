"""Memuat kedua model dari Hugging Face Hub (di-cache agar tidak dimuat ulang)."""

import streamlit as st
import torch
from transformers import pipeline

from .config import (
    INDOBERT_REPO,
    INDOBERT_REVISION,
    XLMR_REPO,
    XLMR_REVISION,
    MAX_LEN,
)


def _device() -> int:
    return 0 if torch.cuda.is_available() else -1


@st.cache_resource(
    show_spinner="Memuat model dari Hugging Face..."
)
def load_pipeline(repo_id: str, revision: str):
    return pipeline(
        "text-classification",
        model=repo_id,
        revision=revision,
        top_k=None,
        truncation=True,
        max_length=MAX_LEN,
        device=_device(),
    )


def get_models() -> dict:
    return {
        "IndoBERT": load_pipeline(
            INDOBERT_REPO, INDOBERT_REVISION
        ),
        "XLM-RoBERTa": load_pipeline(
            XLMR_REPO, XLMR_REVISION
        ),
    }


def predict(pipe, text: str):
    """Kembalikan (label, {label: probabilitas})."""
    out = pipe(text)
    if out and isinstance(out[0], list):
        out = out[0]
    scores = {d["label"]: float(d["score"]) for d in out}
    label = max(scores, key=scores.get)
    return label, scores
