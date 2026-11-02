"""Konfigurasi dashboard. Ganti HF_USERNAME dengan username Hugging Face Anda."""

HF_USERNAME = "midlyn"

INDOBERT_REPO = f"{HF_USERNAME}/indobert-clickbait-id"
XLMR_REPO = f"{HF_USERNAME}/xlmr-clickbait-id"

INDOBERT_REVISION = "6afd9af1d679889f0849e8d6739219d357f06fee"
XLMR_REVISION = "8a5919c0bab8558e790c4f693eef0d54810c3c0b"

MAX_LEN = 64                 # sama dengan saat training
CLICKBAIT_LABEL = "clickbait"
