"""Checks that REPORT.md is filled in. It does not grade your reasoning: it makes sure nothing is left blank."""
import pathlib
import re

REPORT = pathlib.Path(__file__).resolve().parent.parent / "REPORT.md"


def section(text, start, end=None):
    i = text.index(start)
    j = text.index(end, i) if end else len(text)
    return text[i:j]


def test_report_exists():
    assert REPORT.exists(), "REPORT.md is missing."


def test_required_answers_filled():
    text = REPORT.read_text(encoding="utf-8")
    required = section(text, "## Part A", "**Going further")
    left = required.count("TODO")
    assert left == 0, f"{left} required answer(s) still say TODO in REPORT.md (Parts A, B and C)."


def test_c2_has_a_results_table():
    text = REPORT.read_text(encoding="utf-8")
    c2 = section(text, "**C2.**", "**C3.**")
    rows = [line for line in c2.splitlines() if line.strip().startswith("|")]
    assert any("recall" in line.lower() for line in rows), (
        "C2: paste the markdown table printed by measure.py, with its header row (| Chunking | Search | recall@5 |)."
    )
    numbers = [line for line in rows if re.search(r"\b[01]\.\d+\b", line)]
    assert len(numbers) >= 6, (
        f"C2: the table should have one row per chunking x search pair (6 rows with a recall value), found {len(numbers)}."
    )


def test_c3_cites_specific_questions():
    text = REPORT.read_text(encoding="utf-8")
    c3 = section(text, "**C3.**", "**C4.**")
    cited = set(re.findall(r"\bq\d{2}\b", c3, flags=re.IGNORECASE))
    assert len(cited) >= 2, (
        "C3: explain the gaps with specific questions. Cite at least two question ids (for example q04 and q07)."
    )
