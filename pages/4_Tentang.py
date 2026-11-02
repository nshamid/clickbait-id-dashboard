import streamlit as st

from utils.results import DATASET, HYPERPARAMS

st.set_page_config(page_title="Tentang", page_icon="ℹ️", layout="wide")
st.title("ℹ️ Tentang Penelitian")

st.subheader("Metodologi")
st.markdown(
    f"""
1. **Dataset**: {DATASET['nama']}, {format(DATASET['total_awal'], ',').replace(',', '.')} judul berlabel clickbait / non-clickbait
   ({format(DATASET['clickbait'], ',').replace(',', '.')} clickbait, {format(DATASET['non_clickbait'], ',').replace(',', '.')} non-clickbait).
2. **Pra-pemrosesan**: pembersihan duplikat (termasuk judul kembar berlabel konflik) menyisakan
   {format(DATASET['setelah_dedup'], ',').replace(',', '.')} judul, dibagi stratified 80/10/10 menjadi train {format(DATASET['train'], ',').replace(',', '.')},
   validasi {format(DATASET['val'], ',').replace(',', '.')}, dan test {format(DATASET['test'], ',').replace(',', '.')} dengan pembagian yang sama untuk kedua model.
3. **Model**: fine-tuning IndoBERT (monolingual) dan XLM-RoBERTa (multilingual), tiga seed per model,
   learning rate dipilih pada set validasi, model final = seed dengan F1 validasi tertinggi.
4. **Evaluasi**: accuracy, precision, recall, F1 (macro), McNemar's test, dan analisis kesalahan
   berdasarkan kesepakatan annotator.
5. **Interpretabilitas**: SHAP Partition menggunakan segmentasi teks
   berbasis regex yang sama untuk kedua model, dengan atribusi pada
   skala probabilitas. Analisis dilakukan pada 200 judul yang dipilih
   dengan memperbanyak kasus kesalahan.
"""
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
- SHAP memperkirakan kontribusi segmen teks terhadap prediksi,
  bukan hubungan sebab-akibat. Analisis penelitian menggunakan
  200 judul dengan memperbanyak kasus kesalahan, sehingga hasilnya
  tidak mewakili seluruh test set.
- Riwayat uji coba disimpan di file lokal dan dapat hilang ketika aplikasi di-restart oleh platform hosting.
- Hanya judul yang dianalisis, bukan isi berita.
"""
)
