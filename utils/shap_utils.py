"""Interpretasi SHAP dengan konfigurasi eksperimen baru."""

import numpy as np
import shap

from .config import CLICKBAIT_LABEL


def get_explainer(pipe, key: str):
    masker = shap.maskers.Text(tokenizer=r"\W+")

    model = shap.models.TransformersPipeline(
        pipe,
        rescale_to_logits=False,
    )

    return shap.Explainer(
        model,
        masker,
        algorithm="partition",
    )


def explain(explainer, pipe, text: str):
    """Kembalikan segmen dan atribusi probabilitas kelas clickbait."""
    sv = explainer(
        [text],
        max_evals=500,
        batch_size=16,
    )

    output_names = list(sv.output_names)

    if CLICKBAIT_LABEL not in output_names:
        raise ValueError(
            f"Label {CLICKBAIT_LABEL!r} tidak ditemukan: "
            f"{output_names}"
        )

    idx = output_names.index(CLICKBAIT_LABEL)

    tokens = list(sv.data[0])
    values = np.asarray(sv.values[0])[:, idx]

    return tokens, values
