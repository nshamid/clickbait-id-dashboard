import streamlit as st

from utils.results import DATASET, HYPERPARAMS

st.set_page_config(page_title="Tentang", page_icon="ℹ️", layout="wide")
st.title("ℹ️ Tentang Penelitian")

st.subheader("Metodologi")
st.markdown(
    f"""
1. **Dataset**: {DATASET['nama']}, {DATASET['total_awal']:,} judul berlabel clickbait / non-clickbait
   ({DATASET['clickbait']:,} clickbait, {DATASET['non_clickbait']:,} non-clickbait).
2. **Pra-pemrosesan**: pembersihan duplikat (termasuk judul kembar berlabel konflik) menyisakan
   {DATASET['setelah_dedup']:,} judul, dibagi stratified 80/10/10 menjadi train {DATASET['train']:,},
   validasi {DATASET['val']:,}, dan test {DATASET['test']:,} dengan pembagian yang sama untuk kedua model.
3. **Model**: fine-tuning IndoBERT (monolingual) dan XLM-RoBERTa (multilingual), tiga seed per model,
   learning rate dipilih pada set validasi, model final = seed dengan F1 validasi tertinggi.
4. **Evaluasi**: accuracy, precision, recall, F1 (macro), McNemar's test, dan analisis kesalahan
   berdasarkan kesepakatan annotator.
5. **Interpretabilitas**: SHAP dengan masker level kata agar atribusi kedua model sebanding
   walaupun tokenizer internalnya berbeda.
""".replace(",", ".")
)

st.subheader("Konfigurasi pelatihan")
st.dataframe(HYPERPARAMS)

st.subheader("Batasan sistem")
st.markdown(
    """
- Label dataset berasal dari annotator manusia yang tidak selalu sepakat (sekitar 43% judul tidak
  disepakati penuh), sehingga akurasi tidak mendekati 100% dan hasil model bukan kebenaran mutlak.
- Model dilatih pada judul berita daring Indonesia dari sumber tertentu; performa pada gaya judul
  atau domain lain (media sosial, iklan) belum diuji.
- SHAP adalah perkiraan kontribusi kata, bukan penjelasan sebab-akibat; analisis kata dalam
  penelitian memakai sampel kecil (181 judul).
- Riwayat uji coba disimpan di file lokal dan dapat hilang ketika aplikasi di-restart oleh platform hosting.
- Hanya judul yang dianalisis, bukan isi berita.
"""
)
