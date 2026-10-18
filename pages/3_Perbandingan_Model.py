from pathlib import Path

import pandas as pd
import streamlit as st

from utils.results import (BEST_SEED_METRICS, DATASET, ERROR_BY_AGREEMENT, MCNEMAR,
                           MULTI_SEED, SHAP_SUMMARY)

st.set_page_config(page_title="Perbandingan Model", page_icon="📊", layout="wide")
st.title("📊 Perbandingan Model")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.subheader("1. Metrik pada test set (model final, seed terbaik)")
dfm = pd.DataFrame(BEST_SEED_METRICS)
st.dataframe(dfm.style.format("{:.4f}"))
st.bar_chart(dfm)

st.subheader("2. Stabilitas antar seed (rata-rata ± std, 3 seed)")
baris = []
for model, metr in MULTI_SEED.items():
    row = {"Model": model}
    for nama, (mean, std) in metr.items():
        row[nama] = f"{mean:.4f} ± {std:.4f}"
    baris.append(row)
st.dataframe(pd.DataFrame(baris), hide_index=True)

st.subheader("3. Uji signifikansi: McNemar's test (exact)")
tabel = pd.DataFrame(
    [[MCNEMAR["both_correct"], MCNEMAR["only_indobert"]],
     [MCNEMAR["only_xlmr"], MCNEMAR["both_wrong"]]],
    index=["XLM-R benar", "XLM-R salah"],
    columns=["IndoBERT benar", "IndoBERT salah"],
)
st.dataframe(tabel)
st.markdown(
    f"Nilai p = **{MCNEMAR['p_value']:.2e}**, sehingga perbedaan kinerja kedua model "
    f"signifikan secara statistik (taraf 0,05). XLM-RoBERTa benar sendirian pada "
    f"**{MCNEMAR['only_xlmr']}** judul, IndoBERT pada **{MCNEMAR['only_indobert']}** judul."
)

st.subheader("4. Analisis kesalahan berdasarkan kesepakatan annotator")
st.dataframe(ERROR_BY_AGREEMENT.style.format("{:.1%}"))
st.caption(
    "Pada judul yang tidak disepakati annotator, porsi 'keduanya salah' jauh lebih besar "
    "(19,1% vs 2,6%). Sebagian kesalahan model berasal dari label yang memang ambigu."
)

st.subheader("5. Interpretabilitas SHAP")
st.markdown(
    f"- Sampel SHAP: **{SHAP_SUMMARY['n_sample']}** judul (kasus beda prediksi, kedua salah, dan sebagian kedua benar)\n"
    f"- Tumpang tindih 30 kata teratas (muncul minimal 3 kali): **{SHAP_SUMMARY['overlap']}**\n"
    f"- Entropi atribusi rata-rata: IndoBERT **{SHAP_SUMMARY['entropy_indobert']:.4f}**, "
    f"XLM-R **{SHAP_SUMMARY['entropy_xlmr']:.4f}**"
)

f_ind = DATA_DIR / "indobert_top30_words.csv"
f_xlm = DATA_DIR / "xlmr_top30_words.csv"
if f_ind.exists() and f_xlm.exists():
    a, b = st.columns(2)
    a.markdown("**Top 30 kata IndoBERT**")
    a.dataframe(pd.read_csv(f_ind), hide_index=True)
    b.markdown("**Top 30 kata XLM-RoBERTa**")
    b.dataframe(pd.read_csv(f_xlm), hide_index=True)
else:
    st.info("Salin indobert_top30_words.csv dan xlmr_top30_words.csv (dari notebook 05) ke folder data/ untuk menampilkan tabel kata.")

st.caption(
    f"Test set: {DATASET['test']:,} judul. Seluruh angka berasal dari notebook penelitian (02-05).".replace(",", ".")
)
