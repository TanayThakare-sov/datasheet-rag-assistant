from src.ingest import split_text, table_to_chunks


def test_split_text_overlaps_and_respects_size():
    text = " ".join(f"word{i}" for i in range(300))
    pieces = split_text(text, size=200, overlap=40)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    assert pieces[0].split()[-1] in pieces[1]  # overlap keeps context across the cut


def test_two_column_table_becomes_key_value_lines():
    table = [["Specification", "Value"], ["Idle power", "4.8 W"], ["Capacity", None]]
    chunks = table_to_chunks(table, "nimbus-n8", 1)
    assert "Idle power: 4.8 W" in chunks[0].text
    assert "nimbus-n8" in chunks[0].text
    assert "Capacity" not in chunks[0].text  # empty value is skipped


def test_wide_table_becomes_one_sentence_per_row():
    table = [["Metric", "Value", "Condition"], ["MTBF", "2,000,000 hours", "8760 h/year"]]
    chunks = table_to_chunks(table, "orion-o12", 2)
    assert len(chunks) == 1
    assert "MTBF" in chunks[0].text and "Value = 2,000,000 hours" in chunks[0].text
