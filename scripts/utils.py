"""Shared utilities: chunking, attribute heuristics, text normalization, and small CSV helpers.
"""

import os
from pathlib import Path
from typing import List
import re
import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw_texts'
PROC_DIR = ROOT / 'data' / 'processed'
VECTOR_EXPORTS = ROOT / 'data' / 'vector_exports'


ATTRIBUTE_KEYS = [
    "source",
    "vendor_name",
    "vendor_type",
    "contract_number",
    "object",
    "start_date",
    "end_date",
    "value",
    "currency",
    "manager",
    "contact",
    "penalties",
    "termination_clause",
    "jurisdiction",
]


def chunk_text(text: str) -> List[str]:
    """Split contract text into reasonable chunks (clauses or ~1k chars)."""
    parts = re.split(r"(?i)(clausula|cláusula|clause|article|section)\s+\w+", text)
    if len(parts) <= 1:
        return [text[i:i+1000] for i in range(0, len(text), 1000)]
    chunks = [p.strip() for p in parts if p.strip()]
    return chunks


def extract_basic_attributes(text: str, filename: str) -> dict:
    a = {k: '' for k in ATTRIBUTE_KEYS}
    a['source'] = filename
    m = re.search(r"\b(OCS\s*)?(\d{2,4}\/\d{2,4})\b", text, re.I)
    if m:
        a['contract_number'] = m.group(2)
    dm = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", text)
    if dm:
        a['start_date'] = dm.group(1)
    vm = re.search(r"(R\$|\$|EUR|USD)?\s?([\d\.,]+)\s?(BRL|USD|EUR)?", text)
    if vm:
        a['value'] = vm.group(2)
        a['currency'] = (vm.group(1) or vm.group(3) or '').strip()
    vm = re.search(r"(Fornecedor|Fornecedor:|Supplier:|Contracted party:|Contratante|Contratada)\s*[:\-]?\s*(.+)", text, re.I)
    if vm:
        a['vendor_name'] = vm.group(2).split('\n')[0].strip()
    else:
        a['vendor_name'] = Path(filename).stem.split('_')[-1]
    om = re.search(r"(Objeto|Object)[:\-]?\s*(.{10,400})", text, re.I|re.S)
    if om:
        a['object'] = om.group(2).split('\n')[0].strip()
    mm = re.search(r"(Manager|Gestor|Gerente)[:\-]?\s*(.+)", text, re.I)
    if mm:
        a['manager'] = mm.group(2).split('\n')[0].strip()
    return a


def build_embeddings(texts: List[str], force_tfidf: bool = False, batch_size: int = 64) -> List[List[float]]:
    """Compute embeddings for texts. Prefer Sentence-Transformers, fallback to TF-IDF.
    Returns list of vectors (lists).
    """
    if (not force_tfidf) and HAS_ST:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [e.tolist() for e in embs]
    tf = TfidfVectorizer(stop_words='english', max_features=4096)
    mat = tf.fit_transform(texts)
    return [mat[i].toarray().ravel().tolist() for i in range(mat.shape[0])]


def normalize(text: str) -> str:
    return text.lower().strip()


def normalize_supplier(name: str) -> str:
    if not isinstance(name, str):
        return ''
    name = name.lower()
    for suffix in [" limited", " ltd", " private", " pvt", " services"]:
        name = name.replace(suffix, "")
    return name.strip()


def append_rows_to_csv(path: Path, rows: List[dict], cols: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_csv(path)
        new_df = pd.DataFrame(rows)
        out = pd.concat([df, new_df], ignore_index=True)
        out.to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


def ensure_vector_exports_dir():
    VECTOR_EXPORTS.mkdir(parents=True, exist_ok=True)


def save_vectors_parquet(df: pd.DataFrame, out_path: Path):
    ensure_vector_exports_dir()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
