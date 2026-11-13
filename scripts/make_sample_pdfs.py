"""Create 4 FICTIONAL hard drive datasheets (brand "FakeDrive") plus a matching question set.

Purpose: let you test the whole pipeline without downloading anything. The questions are
generated from the same numbers that go into the PDFs, so the expected answers are correct
by construction. Real evaluation needs real datasheets and questions you write yourself.

Usage:
    python scripts/make_sample_pdfs.py
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DRIVES = {
    "nimbus-n8": dict(name="Nimbus N8", capacity="8 TB", interface="SATA 6Gb/s", idle_power="4.8 W",
                      operating_power="7.9 W", noise_idle="20 dBA", form_factor="3.5-inch",
                      mtbf="1,800,000 hours", afr="0.55%", warranty="3 years"),
    "nimbus-n16": dict(name="Nimbus N16", capacity="16 TB", interface="SATA 6Gb/s", idle_power="5.6 W",
                       operating_power="9.4 W", noise_idle="22 dBA", form_factor="3.5-inch",
                       mtbf="2,000,000 hours", afr="0.44%", warranty="5 years"),
    "orion-o12": dict(name="Orion O12", capacity="12 TB", interface="SAS 12Gb/s", idle_power="6.3 W",
                      operating_power="10.8 W", noise_idle="24 dBA", form_factor="3.5-inch",
                      mtbf="2,500,000 hours", afr="0.35%", warranty="5 years"),
    "orion-o20": dict(name="Orion O20", capacity="20 TB", interface="SAS 12Gb/s", idle_power="7.2 W",
                      operating_power="12.1 W", noise_idle="26 dBA", form_factor="3.5-inch",
                      mtbf="2,500,000 hours", afr="0.31%", warranty="5 years"),
}
SPEC_ROWS = [("Capacity", "capacity"), ("Interface", "interface"),
             ("Idle power consumption", "idle_power"), ("Operating power consumption", "operating_power"),
             ("Idle acoustic noise", "noise_idle"), ("Form factor", "form_factor")]
ENV_ROWS = [("Operating temperature", "5 to 60 C"), ("Non-operating shock", "250 G, 2 ms"),
            ("Relative humidity", "5 to 95 % non-condensing"), ("Altitude", "-305 to 3,048 m")]
GRID = TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black),
                   ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)])
FILLER = ("This fictional datasheet exists only to test a retrieval pipeline. It describes a made-up "
          "storage product from a made-up company. Specifications are listed in the tables below. ")


def build_pdf(path, key, d):
    s = getSampleStyleSheet()
    story = [Paragraph(f"FakeDrive {d['name']} Product Datasheet", s["Title"]),
             Paragraph(FILLER * 3, s["BodyText"]), Spacer(1, 12),
             Paragraph("General specifications", s["Heading2"]),
             Table([["Specification", "Value"]] + [[label, d[k]] for label, k in SPEC_ROWS], style=GRID),
             PageBreak(),
             Paragraph("Reliability", s["Heading2"]),
             Table([["Metric", "Value", "Test condition"],
                    ["MTBF", d["mtbf"], "8,760 power-on hours per year"],
                    ["Annualized failure rate", d["afr"], "Measured at 25 C"],
                    ["Warranty", d["warranty"], "From date of purchase"]], style=GRID),
             PageBreak(),
             Paragraph("Environmental and features", s["Heading2"]),
             Paragraph(FILLER * 4, s["BodyText"]), Spacer(1, 12),
             Table([["Condition", "Limit"]] + [list(r) for r in ENV_ROWS], style=GRID)]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def build_questions():
    qs = []
    fact = [("capacity", "What is the capacity of the {n}?"),
            ("idle_power", "What is the idle power consumption of the {n}?"),
            ("operating_power", "What is the operating power consumption of the {n}?"),
            ("noise_idle", "How loud is the {n} when idle?"),
            ("mtbf", "What is the MTBF of the {n}?"),
            ("warranty", "What is the warranty period of the {n}?")]
    for key, d in DRIVES.items():
        for spec, template in fact:
            qs.append(dict(question=template.format(n=d["name"]), answerable=True,
                           expected_docs=[key], expected_contains=[d[spec]]))
    pairs = [("nimbus-n8", "nimbus-n16"), ("nimbus-n16", "orion-o12"),
             ("orion-o12", "orion-o20"), ("nimbus-n8", "orion-o20")]
    comp = [("idle_power", "idle power consumption"), ("operating_power", "operating power consumption"),
            ("capacity", "capacity")]
    for a, b in pairs:
        for spec, text in comp:
            qs.append(dict(question=f"Compare the {text} of the {DRIVES[a]['name']} and the {DRIVES[b]['name']}.",
                           answerable=True, expected_docs=[a, b],
                           expected_contains=[DRIVES[a][spec], DRIVES[b][spec]]))
    for q in ["What is the price of the Nimbus N8?", "What is the price of the Orion O20?",
              "Does the Nimbus N16 support Wi-Fi?", "What colour is the Orion O12?",
              "What is the weight of the Nimbus N8 in grams?", "Who is the CEO of FakeDrive?"]:
        qs.append(dict(question=q, answerable=False, expected_docs=[], expected_contains=[]))
    return qs


if __name__ == "__main__":
    out = Path("data/sample")
    out.mkdir(parents=True, exist_ok=True)
    for key, d in DRIVES.items():
        build_pdf(out / f"{key}.pdf", key, d)
    Path("eval").mkdir(exist_ok=True)
    qs = build_questions()
    Path("eval/sample_questions.json").write_text(json.dumps(qs, indent=2))
    print(f"Wrote {len(DRIVES)} PDFs to {out}/ and {len(qs)} questions to eval/sample_questions.json")
