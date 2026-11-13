"""Find the chunks most relevant to a question.

Modes:
  tfidf   - keyword matching (fast, no downloads)
  dense   - meaning matching with sentence-transformers (optional install)
  hybrid  - both combined with reciprocal rank fusion

Comparison questions ("compare A and B") are handled by pulling chunks from EACH drive
named in the question, so one drive cannot crowd out the other.
"""
import math
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import DENSE_MODEL, TOP_K

TOKEN = r"[a-z0-9]+(?:\.[0-9]+)?"


def _prep(text):
    return re.sub(r"[-_]", " ", text.lower())


def tokens(text):
    return re.findall(TOKEN, _prep(text))


def mentioned_docs(query, docs):
    """Documents whose name appears in the query (all name words must be present)."""
    q = set(tokens(query))
    return [d for d in docs if tokens(d) and set(tokens(d)) <= q]


@dataclass
class Hit:
    chunk: object
    score: float


@dataclass
class SearchResult:
    hits: list
    top_lexical: float  # best keyword score ignoring drive names; used to decide "I don't know"


class Retriever:
    def __init__(self, chunks, mode="tfidf", dense_model=DENSE_MODEL):
        if mode not in ("tfidf", "dense", "hybrid"):
            raise ValueError("mode must be tfidf, dense or hybrid")
        self.chunks, self.mode = chunks, mode
        self.docs = sorted({c.doc for c in chunks})
        self.vec = TfidfVectorizer(preprocessor=_prep, token_pattern=TOKEN, ngram_range=(1, 2),
                                   sublinear_tf=True, stop_words="english")
        self.matrix = self.vec.fit_transform([c.text for c in chunks])
        if mode in ("dense", "hybrid"):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError("Dense/hybrid retrieval needs: pip install sentence-transformers") from e
            self.encoder = SentenceTransformer(dense_model)
            self.dense = self.encoder.encode([c.text for c in chunks], normalize_embeddings=True)

    def _lexical(self, query):
        qv = self.vec.transform([query])
        return (self.matrix @ qv.T).toarray().ravel()  # rows are L2-normalised -> cosine

    @staticmethod
    def _rrf(*score_arrays, k=60):
        total = np.zeros(len(score_arrays[0]))
        for s in score_arrays:
            ranks = np.argsort(-s).argsort()  # 0 = best
            total += 1.0 / (k + ranks + 1)
        return total

    def search(self, query, k=TOP_K):
        docs = mentioned_docs(query, self.docs)
        # Drive names are used to FILTER chunks, so they are removed from the text used for
        # ranking; otherwise short chunks that repeat the drive name would always win.
        drop = {t for d in docs for t in tokens(d)}
        rest = " ".join(t for t in tokens(query) if t not in drop)
        lex_query = rest if (docs and rest) else query
        lex = self._lexical(lex_query)

        if self.mode == "tfidf":
            scores = lex
        else:
            q_emb = self.encoder.encode([query], normalize_embeddings=True)[0]
            dense = self.dense @ q_emb
            scores = dense if self.mode == "dense" else self._rrf(lex, dense)

        if docs:
            per_doc = max(2, math.ceil(k / len(docs)))
            picked = []
            for d in docs:
                idx = [i for i, c in enumerate(self.chunks) if c.doc == d]
                idx.sort(key=lambda i: -scores[i])
                picked += idx[:per_doc]
        else:
            picked = [int(i) for i in np.argsort(-scores)[:k]]
        picked.sort(key=lambda i: -scores[i])
        hits = [Hit(self.chunks[i], float(scores[i])) for i in picked]

        # Abstention score: how well the question (minus drive names) matches anything at all.
        top_lexical = float(self._lexical(rest).max()) if rest and len(self.chunks) else 0.0
        return SearchResult(hits, top_lexical)
