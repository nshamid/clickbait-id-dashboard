"""Memuat kedua model dari Hugging Face Hub (di-cache agar tidak dimuat ulang)."""

import streamlit as st
import torch
from transformers import pipeline

from .config import INDOBERT_REPO, MAX_LEN, XLMR_REPO


def _device() -> int:
    return 0 if torch.cuda.is_available() else -1


@st.cache_resource(show_spinner="Memuat model dari Hugging Face (pertama kali bisa beberapa menit)...")
def load_pipeline(repo_id: str):
    return pipeline(
        "text-classification",
        model=repo_id,
        top_k=None,
        truncation=True,
        max_length=MAX_LEN,
        device=_device(),
    )


def get_models() -> dict:
    return {
        "IndoBERT": load_pipeline(INDOBERT_REPO),
        "XLM-RoBERTa": load_pipeline(XLMR_REPO),
    }


def predict(pipe, text: str):
    """Kembalikan (label, {label: probabilitas})."""
    out = pipe(text)
    if out and isinstance(out[0], list):
        out = out[0]
    scores = {d["label"]: float(d["score"]) for d in out}
    label = max(scores, key=scores.get)
    return label, scores
