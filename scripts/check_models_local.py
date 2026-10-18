"""Uji cepat di laptop: pastikan kedua model dapat dimuat dari Hugging Face.

Jalankan dari folder utama proyek:  python scripts/check_models_local.py
(Unduhan pertama sekitar 1,6 GB, setelahnya memakai cache lokal.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import pipeline

from utils.config import HF_USERNAME, INDOBERT_REPO, MAX_LEN, XLMR_REPO

if HF_USERNAME.startswith("GANTI"):
    sys.exit("Ganti HF_USERNAME di utils/config.py terlebih dahulu.")

for repo in (INDOBERT_REPO, XLMR_REPO):
    pipe = pipeline("text-classification", model=repo, top_k=None, truncation=True, max_length=MAX_LEN)
    print(repo, pipe("Ini Alasan Harga Cabai Naik Terus, Nomor 3 Bikin Kaget!"))
