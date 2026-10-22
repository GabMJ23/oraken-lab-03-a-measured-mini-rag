"""Part C1: your evaluation set, data/questions.json. Run with: pytest -q"""
import json
import pathlib
import re

import pytest

import rag

QUESTIONS = pathlib.Path(__file__).resolve().parent.parent / "data" / "questions.json"


@pytest.fixture(scope="module")
def questions():
    assert QUESTIONS.exists(), "data/questions.json is missing."
    try:
        data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        pytest.fail(f"data/questions.json is not valid JSON: {e}")
    assert isinstance(data, list), "data/questions.json must be a JSON list of objects."
    return data


@pytest.fixture(scope="module")
def sections():
    return {s["id"]: s for s in rag.load_corpus()}


def test_at_least_ten_questions(questions):
    assert len(questions) >= 10, f"Write at least 10 questions in data/questions.json (you have {len(questions)})."


def test_each_question_is_well_formed(questions, sections):
    for i, q in enumerate(questions):
        for field in ("id", "question", "expected"):
            assert field in q, f"Question #{i + 1} has no '{field}' field."
        assert q["expected"] in sections, (
            f"{q['id']}: expected section {q['expected']!r} is not in data/corpus.json. Use an id such as '825.220'."
        )
        assert not re.search(r"\b825\.\d+", q["question"]), (
            f"{q['id']}: do not put a section number in the question. Ask it the way an employee would."
        )
    ids = [q["id"] for q in questions]
    assert len(set(ids)) == len(ids), "Question ids must be unique (q01, q02, ...)."
    texts = [q["question"].strip().lower() for q in questions]
    assert len(set(texts)) == len(texts), "Two questions are identical."


def test_stop_words_were_not_stretched():
    topic = {"employer", "leave", "job", "fire", "fired", "boss", "doctor", "sick", "baby", "pay", "work"}
    assert not topic & rag.STOP_WORDS, "Do not add topic words to STOP_WORDS: the zero-overlap check would mean nothing."
    assert len(rag.STOP_WORDS) < 200, "STOP_WORDS has grown too much: keep the list provided in rag.py."


def test_at_least_three_questions_share_no_word(questions, sections):
    zero = [q["id"] for q in questions if not rag.overlap(q["question"], sections[q["expected"]])]
    assert len(zero) >= 3, (
        f"Only {len(zero)} question(s) share no content word with their expected section ({zero}); you need 3. "
        "Check a question with: python -c \"import rag; s={x['id']: x for x in rag.load_corpus()}; "
        "print(rag.overlap('your question', s['825.220']))\""
    )
