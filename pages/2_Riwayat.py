import streamlit as st

from utils import history

st.set_page_config(page_title="Riwayat", page_icon="🗂️", layout="wide")
st.title("🗂️ Riwayat Uji Coba")

df = history.load_history()

if df.empty:
    st.info("Belum ada uji coba. Silakan coba judul di halaman Uji Coba.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Total uji coba", len(df))
c2.metric("Kesepakatan kedua model", f"{df['sepakat'].mean():.0%}")
c3.metric("Judul diprediksi clickbait (XLM-R)", f"{(df['pred_xlmr'] == 'clickbait').mean():.0%}")

tampil = df.rename(columns={
    "waktu": "Waktu", "judul": "Judul",
    "pred_indobert": "IndoBERT", "prob_indobert": "P(clickbait) IndoBERT",
    "pred_xlmr": "XLM-R", "prob_xlmr": "P(clickbait) XLM-R",
    "sepakat": "Sepakat (1/0)",
})
st.dataframe(tampil, hide_index=True)

st.divider()
st.subheader("Hapus data")
ids = st.multiselect("Pilih ID yang dihapus", df["id"].tolist())
if st.button("Hapus yang dipilih") and ids:
    history.delete_ids(ids)
    st.rerun()

konfirmasi = st.checkbox("Saya yakin ingin menghapus seluruh riwayat")
if st.button("Hapus semua riwayat", disabled=not konfirmasi):
    history.delete_all()
    st.rerun()
