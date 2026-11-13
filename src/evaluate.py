"""Measure how well the assistant works on a set of questions with known answers.

Usage:
    python -m src.evaluate --pdf-dir data/sample --questions eval/sample_questions.json --k 2
    python -m src.evaluate --pdf-dir data/raw --questions eval/questions.json --dense --llm

Question file format (JSON list):
    {"question": "...", "answerable": true, "expected_docs": ["nimbus-n8"], "expected_contains": ["4.8 W"]}
For unanswerable questions set "answerable": false (the right behaviour is "I don't know").

Metrics (per configuration):
    doc_hit@k        retrieved passages come from every drive the answer needs
    value_hit@k      retrieved passages contain the expected value(s)
    false_abstain    answerable questions where it wrongly said "I don't know" (lower is better)
    correct_refusal  unanswerable questions where it said "I don't know" (higher is better)
    answer_acc       (only with --llm) final answer contains the expected value(s), or refuses correctly
"""
import argparse
import json
import re
from pathlib import Path

from .answer import answer_question
from .config import ABSTAIN_THRESHOLD, TOP_K
from .ingest import ingest_folder
from .retriever import Retriever


def norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_all(text, needles):
    t = norm(text)
    return all(norm(n) in t for n in needles)


def evaluate_config(retriever, questions, k, threshold, use_llm):
    ans_q = [q for q in questions if q["answerable"]]
    un_q = [q for q in questions if not q["answerable"]]
    doc_hits, val_hits, false_abs, acc = [], [], [], []
    refusals = []
    for q in questions:
        out = answer_question(q["question"], retriever, k=k, threshold=threshold, use_llm=use_llm)
        if q["answerable"]:
            hits = out["result"].hits
            docs = {h.chunk.doc for h in hits}
            joined = " ".join(h.chunk.text for h in hits)
            doc_hits.append(set(q["expected_docs"]) <= docs)
            val_hits.append(contains_all(joined, q["expected_contains"]))
            false_abs.append(out["abstained"])
            acc.append((not out["abstained"]) and contains_all(out["answer"], q["expected_contains"]))
        else:
            refusals.append(out["abstained"])
            acc.append(out["abstained"])
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    res = {
        "doc_hit@k": mean(doc_hits),
        "value_hit@k": mean(val_hits),
        "false_abstain": mean(false_abs),
        "correct_refusal": mean(refusals),
    }
    if use_llm:
        res["answer_acc"] = mean(acc)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf-dir", default="data/sample")
    p.add_argument("--questions", default="eval/sample_questions.json")
    p.add_argument("--k", type=int, default=TOP_K)
    p.add_argument("--threshold", type=float, default=ABSTAIN_THRESHOLD)
    p.add_argument("--dense", action="store_true", help="also test dense and hybrid retrieval")
    p.add_argument("--llm", action="store_true", help="also test final LLM answers (needs ANTHROPIC_API_KEY)")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    n_ans = sum(q["answerable"] for q in questions)
    print(f"{len(questions)} questions ({n_ans} answerable, {len(questions) - n_ans} unanswerable), k={args.k}")

    chunks = {m: ingest_folder(args.pdf_dir, m) for m in ("plain", "table_aware")}
    configs = [("plain + keyword", "plain", "tfidf"), ("table-aware + keyword", "table_aware", "tfidf")]
    if args.dense:
        configs += [("table-aware + dense", "table_aware", "dense"),
                    ("table-aware + hybrid", "table_aware", "hybrid")]

    results = {}
    for name, ingest_mode, ret_mode in configs:
        retriever = Retriever(chunks[ingest_mode], mode=ret_mode)
        results[name] = evaluate_config(retriever, questions, args.k, args.threshold, args.llm)

    cols = list(next(iter(results.values())).keys())
    lines = ["| configuration | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(str(r[c]) for c in cols) + " |")
    table = "\n".join(lines)
    print("\n" + table)

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    (out / "eval_results.json").write_text(json.dumps(results, indent=2))
    (out / "eval_table.md").write_text(table + "\n")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
