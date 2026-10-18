"""Konfigurasi dashboard. Ganti HF_USERNAME dengan username Hugging Face Anda."""

HF_USERNAME = "midlyn"

INDOBERT_REPO = f"{HF_USERNAME}/indobert-clickbait-id"
XLMR_REPO = f"{HF_USERNAME}/xlmr-clickbait-id"

MAX_LEN = 64                 # sama dengan saat training
CLICKBAIT_LABEL = "clickbait"
