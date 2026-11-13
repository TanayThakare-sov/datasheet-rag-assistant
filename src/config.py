"""Shared settings. Change them here, not in the other files."""
import os

CHUNK_CHARS = 600          # max characters per text chunk
CHUNK_OVERLAP = 100        # characters shared between neighbouring text chunks
TOP_K = 5                  # passages given to the answerer
ABSTAIN_THRESHOLD = 0.10   # below this keyword-match score we answer "I don't know"
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "claude-sonnet-5-5")
DENSE_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers model for dense retrieval
