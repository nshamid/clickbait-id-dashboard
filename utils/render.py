"""Fungsi tampilan murni (tanpa Streamlit) agar mudah diuji."""

import html

import numpy as np


def highlight_html(tokens, values) -> str:
    """Merah = mendorong ke clickbait, hijau = mendorong ke non-clickbait."""
    values = np.asarray(values, dtype=float)
    vmax = max(float(np.max(np.abs(values))) if len(values) else 0.0, 1e-6)
    parts = []
    for tok, v in zip(tokens, values):
        if not str(tok).strip():
            parts.append(html.escape(str(tok)))
            continue
        alpha = min(abs(v) / vmax, 1.0) * 0.75
        color = f"rgba(220,53,69,{alpha:.2f})" if v > 0 else f"rgba(40,167,69,{alpha:.2f})"
        parts.append(
            f"<span title='SHAP {v:+.3f}' style='background:{color};"
            f"padding:2px 3px;border-radius:4px'>{html.escape(str(tok))}</span>"
        )
    return "<div style='line-height:2.2;font-size:1.1rem'>" + "".join(parts) + "</div>"


def top_positive_words(tokens, values, n: int = 3):
    """Kata dengan dorongan terbesar ke arah clickbait (nilai SHAP > 0)."""
    pairs = [
        (str(t).strip(), float(v))
        for t, v in zip(tokens, values)
        if str(t).strip() and float(v) > 0
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in pairs[:n]]


def recommendation(labels: dict, trigger_words):
    """Kembalikan (jenis, pesan); jenis: error | success | warning."""
    vals = list(labels.values())
    n_click = sum(v == "clickbait" for v in vals)
    if n_click == len(vals):
        msg = "Kedua model menilai judul ini **clickbait**."
        if trigger_words:
            msg += (
                " Kata yang paling mendorong ke arah clickbait: "
                + ", ".join(f"**{w}**" for w in trigger_words)
                + ". Pertimbangkan menggantinya dengan kata yang lebih informatif."
            )
        return "error", msg
    if n_click == 0:
        return "success", "Kedua model menilai judul ini **non-clickbait**. Judul tampak informatif."
    return (
        "warning",
        "Kedua model **tidak sepakat**. Pada pengujian penelitian ini, ketidaksepakatan model "
        "lebih sering muncul pada judul yang labelnya juga tidak seragam di antara annotator "
        "manusia, sehingga judul ini sebaiknya ditinjau manual.",
    )
