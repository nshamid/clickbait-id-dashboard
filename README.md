# Dashboard Deteksi Clickbait Berita Indonesia

Perbandingan IndoBERT dan XLM-RoBERTa (dengan SHAP) pada dataset CLICK-ID.

## Struktur
- `app.py` – Beranda
- `pages/` – Uji Coba, Riwayat, Perbandingan Model, Tentang
- `utils/` – pemuat model, SHAP, tampilan, riwayat, angka hasil penelitian
- `data/` – letakkan CSV hasil notebook 05 (`indobert_top30_words.csv`, `xlmr_top30_words.csv`)
- `scripts/` – unggah model dari Kaggle dan uji muat model di lokal

## Langkah
1. Unggah model ke Hugging Face: jalankan `scripts/upload_models_kaggle.py` di notebook Kaggle.
2. Ganti `HF_USERNAME` di `utils/config.py`.
3. Lokal:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   python scripts/check_models_local.py
   streamlit run app.py
   ```
4. Push ke GitHub (jangan menyertakan model atau rahasia; lihat `.gitignore`).

## Catatan deploy
- Streamlit Community Cloud: batas memori terbatas, dua model sekaligus berisiko melampaui batas.
- Hugging Face Spaces (SDK Streamlit): perlu `README.md` dengan header YAML (`sdk: streamlit`, `app_file: app.py`).
- Riwayat disimpan di `data/history.db` dan hilang saat aplikasi restart di hosting gratis.
