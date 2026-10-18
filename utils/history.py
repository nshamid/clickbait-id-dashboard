"""Riwayat uji coba (SQLite lokal). Catatan: di hosting gratis, file ini hilang saat aplikasi restart."""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS riwayat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    waktu TEXT,
    judul TEXT,
    pred_indobert TEXT,
    prob_indobert REAL,
    pred_xlmr TEXT,
    prob_xlmr REAL,
    sepakat INTEGER
)
"""


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def add_record(judul, pred_indobert, prob_indobert, pred_xlmr, prob_xlmr):
    sepakat = int(pred_indobert == pred_xlmr)
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO riwayat (waktu, judul, pred_indobert, prob_indobert, pred_xlmr, prob_xlmr, sepakat) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), judul, pred_indobert,
             float(prob_indobert), pred_xlmr, float(prob_xlmr), sepakat),
        )
        conn.commit()


def load_history() -> pd.DataFrame:
    with closing(_connect()) as conn:
        return pd.read_sql_query("SELECT * FROM riwayat ORDER BY id DESC", conn)


def delete_ids(ids):
    ids = [int(i) for i in ids]
    if not ids:
        return
    with closing(_connect()) as conn:
        conn.executemany("DELETE FROM riwayat WHERE id = ?", [(i,) for i in ids])
        conn.commit()


def delete_all():
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM riwayat")
        conn.commit()
