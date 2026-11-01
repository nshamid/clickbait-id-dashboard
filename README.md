# Dashboard Deteksi Clickbait Berita Indonesia

Perbandingan IndoBERT dan XLM-RoBERTa (dengan SHAP) pada dataset CLICK-ID.

## Struktur
- `app.py` – Beranda
- `pages/` – Uji Coba, Riwayat, Perbandingan Model, Tentang
- `utils/` – pemuat model, SHAP, tampilan, riwayat, angka hasil penelitian
- `data/` – letakkan CSV hasil notebook 05 (`indobert_top30_words.csv`, `xlmr_top30_words.csv`)
- `scripts/` – unggah model dari Kaggle dan uji muat model di lokal

## Catatan deploy
- Streamlit Community Cloud: batas memori terbatas, dua model sekaligus berisiko melampaui batas.
- Hugging Face Spaces (SDK Streamlit): perlu `README.md` dengan header YAML (`sdk: streamlit`, `app_file: app.py`).
- Riwayat disimpan di `data/history.db` dan hilang saat aplikasi restart di hosting gratis.
