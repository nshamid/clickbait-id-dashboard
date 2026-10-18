import streamlit as st

from utils import history
from utils.models import get_models, predict
from utils.render import highlight_html, recommendation, top_positive_words
from utils.shap_utils import explain, get_explainer

st.set_page_config(page_title="Uji Coba", page_icon="🧪", layout="wide")
st.title("🧪 Uji Coba Judul")
st.caption("Masukkan judul berita, lalu bandingkan prediksi IndoBERT dan XLM-RoBERTa secara berdampingan.")

CONTOH = {
    "Contoh: kemungkinan clickbait": "Ini Alasan Harga Cabai Naik Terus, Nomor 3 Bikin Kaget!",
    "Contoh: kemungkinan non-clickbait": "Pemerintah Resmikan Jembatan Baru di Kabupaten Banyuasin",
}

pilihan = st.selectbox("Contoh judul (opsional)", ["(ketik sendiri)"] + list(CONTOH))
judul = st.text_area("Judul berita", value=CONTOH.get(pilihan, ""), height=90, max_chars=300)
pakai_shap = st.checkbox("Tampilkan highlight kata (SHAP), lebih lambat", value=True)

if st.button("Analisis", type="primary"):
    judul = judul.strip()
    if not judul:
        st.warning("Silakan isi judul terlebih dahulu.")
        st.stop()

    models = get_models()
    hasil = {name: predict(pipe, judul) for name, pipe in models.items()}

    shap_out = {}
    cols = st.columns(2)
    for col, (name, pipe) in zip(cols, models.items()):
        label, scores = hasil[name]
        p_click = scores.get("clickbait", 0.0)
        with col:
            st.subheader(name)
            if label == "clickbait":
                st.error("Prediksi: CLICKBAIT")
            else:
                st.success("Prediksi: NON-CLICKBAIT")
            st.progress(min(max(p_click, 0.0), 1.0), text=f"Probabilitas clickbait: {p_click:.1%}")

            if pakai_shap:
                with st.spinner("Menghitung SHAP..."):
                    explainer = get_explainer(pipe, name)
                    tokens, values = explain(explainer, pipe, judul)
                shap_out[name] = (tokens, values)
                st.markdown(highlight_html(tokens, values), unsafe_allow_html=True)

    if pakai_shap:
        st.caption("Merah = kata mendorong ke clickbait · Hijau = kata mendorong ke non-clickbait · Arahkan kursor ke kata untuk melihat nilai SHAP.")

    kata_pemicu = []
    for tokens, values in shap_out.values():
        for w in top_positive_words(tokens, values):
            if w.lower() not in [k.lower() for k in kata_pemicu]:
                kata_pemicu.append(w)

    jenis, pesan = recommendation({n: hasil[n][0] for n in hasil}, kata_pemicu)
    st.subheader("Saran")
    getattr(st, jenis)(pesan)

    history.add_record(
        judul,
        hasil["IndoBERT"][0], hasil["IndoBERT"][1].get("clickbait", 0.0),
        hasil["XLM-RoBERTa"][0], hasil["XLM-RoBERTa"][1].get("clickbait", 0.0),
    )
    st.caption("Hasil ini otomatis tersimpan di halaman Riwayat.")
