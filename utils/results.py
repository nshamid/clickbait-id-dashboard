"""Ringkasan hasil eksperimen baru IndoBERT dan XLM-RoBERTa."""

import pandas as pd


# Model final dipilih berdasarkan F1 macro VALIDASI.
# IndoBERT: seed 42; XLM-RoBERTa: seed 2024.
# Angka berikut adalah hasil evaluasi pada test set.
BEST_SEED_METRICS = {
    "IndoBERT": {
        "Accuracy": 0.8255347593582888,
        "Precision": 0.8206211055459762,
        "Recall": 0.8223568346590743,
        "F1 (macro)": 0.8214231321750369,
    },
    "XLM-RoBERTa": {
        "Accuracy": 0.8435828877005348,
        "Precision": 0.8422645613771893,
        "Recall": 0.8348278493645249,
        "F1 (macro)": 0.8378201077418864,
    },
}


# Rata-rata dan simpangan baku sampel (ddof=1).
# Tiga seed: 42, 123, 2024; dibulatkan empat desimal.
MULTI_SEED = {
    "IndoBERT": {
        "Accuracy": (0.8269, 0.0012),
        "F1 (macro)": (0.8216, 0.0009),
    },
    "XLM-RoBERTa": {
        "Accuracy": (0.8351, 0.0104),
        "F1 (macro)": (0.8280, 0.0110),
    },
}


# McNemar exact pada prediksi dua model final.
MCNEMAR = {
    "both_correct": 1182,
    "only_indobert": 53,
    "only_xlmr": 80,
    "both_wrong": 181,
    "p_value": 0.023801099334891,
}


# Proporsi kategori dalam masing-masing kelompok annotator.
# Dibulatkan tiga desimal, mengikuti output notebook komparasi.
ERROR_BY_AGREEMENT = pd.DataFrame(
    {
        "Keduanya benar": [0.914, 0.630, 1.000],
        "IndoBERT benar, XLM-R salah": [0.023, 0.052, 0.000],
        "IndoBERT salah, XLM-R benar": [0.034, 0.078, 0.000],
        "Keduanya salah": [0.029, 0.240, 0.000],
    },
    index=[
        "Annotator sepakat",
        "Annotator tidak sepakat",
        "Status tidak diketahui",
    ],
)


# Sampel SHAP dipilih dengan memperbanyak kasus kesalahan.
# Ringkasan ini tidak mewakili distribusi seluruh test set.
SHAP_SUMMARY = {
    "n_sample": 200,
    "overlap": "13 dari 30 kata (43,33%)",
    "entropy_indobert": 1.918498,
    "entropy_xlmr": 1.917892,
}


DATASET = {
    "nama": "CLICK-ID",
    "total_awal": 15000,
    "setelah_dedup": 14959,
    "train": 11967,
    "val": 1496,
    "test": 1496,

    # Distribusi label dataset awal, sebelum deduplikasi.
    "clickbait": 6290,
    "non_clickbait": 8710,
}


HYPERPARAMS = pd.DataFrame(
    {
        "IndoBERT": [
            "indobenchmark/indobert-base-p1",
            "2e-5",
            "64",
            "16",
            "4",
            "0.01",
            "0.1",
            "42, 123, 2024",
        ],
        "XLM-RoBERTa": [
            "xlm-roberta-base",
            "3e-5",
            "64",
            "16",
            "4",
            "0.01",
            "0.1",
            "42, 123, 2024",
        ],
    },
    index=[
        "Checkpoint",
        "Learning rate",
        "Max length",
        "Batch size",
        "Epoch maks.",
        "Weight decay",
        "Warmup ratio",
        "Seed",
    ],
)
