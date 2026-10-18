"""Angka hasil penelitian (dari notebook 02-05). Ubah di sini bila ada eksperimen ulang."""

import pandas as pd

# Model final = seed terbaik (seed 123) pada test set (1.496 judul)
BEST_SEED_METRICS = {
    "IndoBERT":    {"Accuracy": 0.7266, "Precision": 0.7394, "Recall": 0.6953, "F1 (macro)": 0.6982},
    "XLM-RoBERTa": {"Accuracy": 0.8389, "Precision": 0.8378, "Recall": 0.8295, "F1 (macro)": 0.8327},
}

# Rata-rata ± std dari 3 seed (42, 123, 2024) pada test set
MULTI_SEED = {
    "IndoBERT":    {"Accuracy": (0.7219, 0.0041), "F1 (macro)": (0.6950, 0.0028)},
    "XLM-RoBERTa": {"Accuracy": (0.8316, 0.0066), "F1 (macro)": (0.8249, 0.0069)},
}

# McNemar's test (exact) pada prediksi model final
MCNEMAR = {
    "both_correct": 993,
    "only_indobert": 94,
    "only_xlmr": 262,
    "both_wrong": 147,
    "p_value": 1.79e-19,
}

# Proporsi kategori hasil per status kesepakatan annotator (baris = 100%)
ERROR_BY_AGREEMENT = pd.DataFrame(
    {
        "Keduanya benar":              [0.790, 0.503],
        "IndoBERT benar, XLM-R salah": [0.025, 0.112],
        "IndoBERT salah, XLM-R benar": [0.159, 0.194],
        "Keduanya salah":              [0.026, 0.191],
    },
    index=["Annotator sepakat", "Annotator tidak sepakat"],
)

SHAP_SUMMARY = {
    "n_sample": 181,
    "overlap": "11 dari 30 kata (37%)",
    "entropy_indobert": 1.9462,
    "entropy_xlmr": 1.9608,
}

DATASET = {
    "nama": "CLICK-ID",
    "total_awal": 15000,
    "setelah_dedup": 14959,
    "train": 11967,
    "val": 1496,
    "test": 1496,
    "clickbait": 6290,
    "non_clickbait": 8710,
}

HYPERPARAMS = pd.DataFrame(
    {
        "IndoBERT": ["indobenchmark/indobert-base-p1", "5e-5", "64", "16", "4", "0.01", "0.1", "42, 123, 2024"],
        "XLM-RoBERTa": ["xlm-roberta-base", "2e-5", "64", "16", "4", "0.01", "0.1", "42, 123, 2024"],
    },
    index=["Checkpoint", "Learning rate", "Max length", "Batch size", "Epoch maks.", "Weight decay", "Warmup ratio", "Seed"],
)
