"""Jalankan di NOTEBOOK KAGGLE (bukan di laptop).

Prasyarat:
1. Add Input: dataset indobert-best-model dan xlmr-best-model.
2. Add-ons > Secrets: buat secret bernama HF_TOKEN berisi token Hugging Face (tipe Write).
3. Ganti INDOBERT_DIR / XLMR_DIR bila path input Anda berbeda.
"""
# !pip install -q huggingface_hub    # jalankan di sel terpisah

from huggingface_hub import HfApi, login
from kaggle_secrets import UserSecretsClient

login(token=UserSecretsClient().get_secret("HF_TOKEN"))
api = HfApi()
username = api.whoami()["name"]
print("Login sebagai:", username)

INDOBERT_DIR = "/kaggle/input/datasets/nshamid/indobert-best-model"
XLMR_DIR = "/kaggle/input/datasets/nshamid/xlmr-best-model"

MODELS = {
    f"{username}/indobert-clickbait-id": dict(
        folder=INDOBERT_DIR, base="indobenchmark/indobert-base-p1", acc="0.7266", f1="0.6982"),
    f"{username}/xlmr-clickbait-id": dict(
        folder=XLMR_DIR, base="xlm-roberta-base", acc="0.8389", f1="0.8327"),
}


def model_card(base, acc, f1):
    return f"""---
language: id
library_name: transformers
pipeline_tag: text-classification
base_model: {base}
tags:
- clickbait
- indonesian
---

# Deteksi clickbait judul berita berbahasa Indonesia

Hasil fine-tuning `{base}` pada dataset CLICK-ID (klasifikasi biner: `clickbait` / `non-clickbait`).

- Max length: 64 token
- Test set (1.496 judul): accuracy {acc}, F1 (macro) {f1}

Model dibuat untuk keperluan penelitian skripsi. Label dataset berasal dari annotator manusia
yang tidak selalu sepakat, sehingga prediksi bukan kebenaran mutlak.
"""


for repo_id, cfg in MODELS.items():
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=cfg["folder"],
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["training_args.bin"],   # tidak dibutuhkan untuk inferensi
    )
    api.upload_file(
        path_or_fileobj=model_card(cfg["base"], cfg["acc"], cfg["f1"]).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print("Terunggah:", repo_id)

# Verifikasi: muat langsung dari Hub
from transformers import pipeline

for repo_id in MODELS:
    pipe = pipeline("text-classification", model=repo_id, top_k=None)
    print(repo_id, pipe("Viral! Pemandangan indah di gunung"))
