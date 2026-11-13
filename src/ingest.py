"""Read datasheet PDFs and turn them into searchable chunks.

Two modes:
  plain        - extract the page text and cut it into pieces (the naive baseline)
  table_aware  - find tables and turn every row into a readable sentence, and prefix
                 every chunk with the drive name so a chunk always says which drive it is about
"""
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from .config import CHUNK_CHARS, CHUNK_OVERLAP


@dataclass
class Chunk:
    text: str
    doc: str    # file name without .pdf, e.g. "nimbus-n8"
    page: int   # 1-based page number
    kind: str   # "text" or "table"


def clean(cell):
    """Tidy one table cell: None -> '', collapse whitespace."""
    return re.sub(r"\s+", " ", cell or "").strip()


def split_text(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Cut text into overlapping pieces, preferring to cut at a line break or space."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt, end)  # start the next piece at a word boundary
        start = space + 1 if space != -1 else nxt
    return [p for p in pieces if p]


def table_to_chunks(table, doc, page, rows_per_chunk=8):
    """Turn a table (list of rows) into text chunks that keep the meaning of each row.

    2 columns  -> "Spec: value" lines (a key/value spec table)
    3+ columns -> one sentence per row using the header names
    """
    rows = [[clean(c) for c in row] for row in table]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    ncols = max(len(r) for r in rows)
    chunks = []
    if ncols == 2:
        lines = [f"{r[0]}: {r[1]}" for r in rows if len(r) >= 2 and r[0] and r[1]]
        for i in range(0, len(lines), rows_per_chunk):
            body = "\n".join(lines[i:i + rows_per_chunk])
            chunks.append(Chunk(f"Drive {doc} (page {page}) specifications:\n{body}", doc, page, "table"))
    else:
        header, data = rows[0], rows[1:]
        for r in data:
            label = r[0]
            parts = [f"{h} = {v}" for h, v in zip(header[1:], r[1:]) if v]
            if label and parts:
                chunks.append(Chunk(f"Drive {doc} (page {page}) {label}: " + "; ".join(parts),
                                    doc, page, "table"))
    return chunks


def ingest_pdf(path, mode="table_aware"):
    if mode not in ("plain", "table_aware"):
        raise ValueError("mode must be 'plain' or 'table_aware'")
    doc = Path(path).stem
    chunks = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            if mode == "plain":
                for piece in split_text(page.extract_text() or ""):
                    chunks.append(Chunk(piece, doc, page_no, "text"))
                continue
            tables = page.find_tables()
            text_page = page
            for t in tables:  # remove table areas so table text is not stored twice
                try:
                    text_page = text_page.outside_bbox(t.bbox)
                except Exception:
                    pass
            for piece in split_text(text_page.extract_text() or ""):
                chunks.append(Chunk(f"Drive {doc} (page {page_no}):\n{piece}", doc, page_no, "text"))
            for t in tables:
                chunks.extend(table_to_chunks(t.extract(), doc, page_no))
    return chunks


def ingest_folder(folder, mode="table_aware"):
    files = sorted(Path(folder).glob("*.pdf"))
    if not files:
        raise FileNotFoundError(f"No PDF files found in {folder}")
    chunks = []
    for f in files:
        chunks.extend(ingest_pdf(f, mode))
    return chunks
