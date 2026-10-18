import streamlit as st

from utils.results import BEST_SEED_METRICS, DATASET, MCNEMAR

st.set_page_config(page_title="Deteksi Clickbait Berita Indonesia", page_icon="📰", layout="wide")

st.title("📰 Deteksi Clickbait Judul Berita Berbahasa Indonesia")
st.caption("Perbandingan IndoBERT (monolingual) dan XLM-RoBERTa (multilingual) dengan interpretabilitas SHAP")

st.markdown(
    f"""
Clickbait adalah judul yang sengaja memancing rasa penasaran agar diklik, sering kali tanpa
menyampaikan isi berita yang sebenarnya. Penelitian ini membandingkan dua arsitektur transformer
untuk mendeteksinya pada dataset **{DATASET['nama']}** ({DATASET['total_awal']:,} judul berita
berlabel dari beberapa media daring Indonesia), lalu menjelaskan keputusan model dengan **SHAP**.
""".replace(",", ".")
)

st.subheader("Ringkasan performa (test set)")
c1, c2 = st.columns(2)
for col, name in zip((c1, c2), ("IndoBERT", "XLM-RoBERTa")):
    m = BEST_SEED_METRICS[name]
    with col:
        st.markdown(f"**{name}**")
        a, b = st.columns(2)
        a.metric("Accuracy", f"{m['Accuracy']:.2%}")
        b.metric("F1 (macro)", f"{m['F1 (macro)']:.2%}")

total_beda = MCNEMAR["only_indobert"] + MCNEMAR["only_xlmr"]
st.success(
    f"XLM-RoBERTa unggul signifikan secara statistik (McNemar's test, p < 0,001): "
    f"benar sendirian pada {MCNEMAR['only_xlmr']} judul, dibanding {MCNEMAR['only_indobert']} judul "
    f"untuk IndoBERT, dari {total_beda} judul yang diprediksi berbeda."
)

st.info(
    "Gunakan menu di sisi kiri: **Uji Coba** untuk menganalisis judul Anda sendiri, "
    "**Riwayat** untuk melihat hasil uji coba, **Perbandingan Model** untuk hasil penelitian lengkap, "
    "dan **Tentang** untuk metodologi serta batasan sistem."
)
