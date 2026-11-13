from src.ingest import Chunk
from src.retriever import Retriever, mentioned_docs


def make():
    return Retriever([
        Chunk("Drive nimbus-n8 idle power: 4.8 W", "nimbus-n8", 1, "table"),
        Chunk("Drive nimbus-n8 warranty: 3 years", "nimbus-n8", 2, "table"),
        Chunk("Drive orion-o12 idle power: 6.3 W", "orion-o12", 1, "table"),
        Chunk("Drive orion-o12 warranty: 5 years", "orion-o12", 2, "table"),
    ])


def test_mentioned_docs_matches_names_with_or_without_hyphen():
    docs = ["nimbus-n8", "nimbus-n16", "orion-o12"]
    assert mentioned_docs("Compare Nimbus N8 and Orion O12", docs) == ["nimbus-n8", "orion-o12"]
    assert mentioned_docs("tell me about the Nimbus N16", docs) == ["nimbus-n16"]


def test_search_finds_the_right_chunk():
    top = make().search("idle power of the Nimbus N8", k=2).hits[0]
    assert top.chunk.doc == "nimbus-n8" and "4.8 W" in top.chunk.text


def test_comparison_returns_chunks_from_both_drives():
    hits = make().search("Compare the idle power of the Nimbus N8 and the Orion O12", k=2).hits
    assert {h.chunk.doc for h in hits} == {"nimbus-n8", "orion-o12"}


def test_unknown_topic_has_low_lexical_score():
    r = make()
    assert r.search("price of the Nimbus N8", k=2).top_lexical < 0.1
    assert r.search("idle power of the Nimbus N8", k=2).top_lexical > 0.1
