# Hard Drive Datasheet Assistant (RAG with measured accuracy)

Ask questions about hard drive datasheets in plain English, or compare drives, and get an answer with the **document and page** it came from. If the answer is not in the documents, the assistant says **"I don't know"** instead of guessing.

> Example: *"Compare the idle power consumption of the Nimbus N8 and the Orion O12."*
> -> "Nimbus N8: 4.8 W [nimbus-n8, page 1]. Orion O12: 6.3 W [orion-o12, page 1]."

Most "chat with a PDF" projects have no evidence that they work. This one includes an **evaluation script** with a question set, so every change to the system can be measured.

## Why datasheets are hard for basic RAG

Datasheets are mostly **tables** and the same fields repeat across many drives. A basic pipeline that cuts page text into pieces has three problems:

1. Table rows get separated from their column headers.
2. A chunk with "Idle power: 4.8 W" does not say which drive it belongs to.
3. A comparison question needs passages from **several** documents, but plain top-k search often returns passages from just one.

This project addresses each of them (see Design decisions).

## How it works

```
PDF datasheets
     |
 ingest.py     read pages, find tables, turn table rows into sentences,
               tag every chunk with the drive name and page number
     |
 retriever.py  find the best chunks (keyword / dense / hybrid);
               for "compare A and B" take chunks from each named drive
     |
 answer.py     LLM answers from those chunks only, with citations;
               "I don't know" if nothing relevant was found
     |
 app.py        Streamlit chat UI          evaluate.py   accuracy on a question set
```

## Tech stack

Python, pdfplumber (PDF and table extraction), scikit-learn (TF-IDF keyword retrieval), sentence-transformers (optional dense retrieval), Anthropic API (answer generation), Streamlit (UI), pytest.

## Project structure

```
app.py                       web UI
src/config.py                settings
src/ingest.py                PDF -> chunks (plain and table-aware modes)
src/retriever.py             keyword / dense / hybrid retrieval, multi-drive comparison
src/answer.py                prompt, LLM call, "I don't know" logic
src/evaluate.py              evaluation script
scripts/make_sample_pdfs.py  creates 4 FICTIONAL datasheets + 42 test questions
eval/real_questions_template.json   format for your own questions
tests/                       unit tests
```

## Quick start

```bash
git clone https://github.com/TanayThakare-sov/datasheet-rag-assistant.git
cd datasheet-rag-assistant
pip install -r requirements.txt
```

### 1. Try it on fictional sample data (no downloads)

```bash
python scripts/make_sample_pdfs.py
python -m src.evaluate --k 2
streamlit run app.py
```

The sample PDFs describe made-up "FakeDrive" products. They only prove the pipeline runs end to end, because the questions are generated from the same numbers that went into the PDFs. They do **not** show real-world accuracy.

### 2. Use real datasheets

1. Download public hard drive datasheets (PDF) from manufacturers' websites (check each site's terms of use) and put them in `data/raw/`. Name each file after the drive, for example `exos-x20.pdf`, because the file name is used as the drive name.
2. Write 40-50 questions with known answers, following `eval/real_questions_template.json`. Take the expected values directly from the PDFs. Include comparison questions and a few unanswerable ones.
3. Run:

```bash
python -m src.evaluate --pdf-dir data/raw --questions eval/questions.json
```

### Optional: LLM answers and dense retrieval

```bash
export ANTHROPIC_API_KEY=your_key      # Windows: set ANTHROPIC_API_KEY=your_key
pip install sentence-transformers      # large download
python -m src.evaluate --pdf-dir data/raw --questions eval/questions.json --llm --dense
```

Without a key the app shows the most relevant passages with sources instead of a written answer. The model can be changed with the `RAG_LLM_MODEL` environment variable.

## Design decisions

- **Table rows become sentences.** Two-column spec tables become "Spec: value" lines. Wider tables become one sentence per row using the column headers, so a value never loses its meaning.
- **Every chunk names its drive and page.** A chunk is stored as "Drive exos-x20 (page 3) ...", so it is understandable on its own and can be cited.
- **Comparison questions.** If the question names several drives, chunks are pulled from each of them. The drive names are used to filter, and removed from the text used to rank, so short chunks that merely repeat a drive name do not win.
- **Honest refusals.** A question that shares no real content words with the documents gets "I don't know" immediately. Otherwise the LLM is instructed to answer only from the passages and refuse if the answer is missing.
- **Evaluation first.** `evaluate.py` compares configurations side by side and reports retrieval hit rate, false refusals and correct refusals (and answer accuracy with `--llm`).

## Results

Fill this in from your own runs on real datasheets:

| configuration | doc_hit@k | value_hit@k | false_abstain | correct_refusal | answer_acc |
|---|---|---|---|---|---|
| plain + keyword | _your result_ | | | | |
| table-aware + keyword | _your result_ | | | | |
| table-aware + hybrid | _your result_ | | | | |

Notes on the fictional sample data: every configuration scores the same there, which shows that this tiny synthetic set cannot tell the configurations apart. Real datasheets, with their messy tables, are where the differences should appear.

## Limitations

- The keyword "I don't know" check can be fooled when a question reuses words from the documents (for example, a question about the manufacturer's CEO). The LLM prompt is the second line of defence.
- Drive names come from file names, so files must be named sensibly.
- Scanned PDFs (images) need OCR, which is not included.
- Tables with merged cells or unusual layouts may extract badly. Check extraction on your real PDFs.
- Manufacturer figures such as MTBF are published estimates, not measured lifetimes.

## Ideas to extend

Reranking, handling units and unit conversion in comparisons, a table-extraction quality check, and a hosted demo (for example on Hugging Face Spaces).
