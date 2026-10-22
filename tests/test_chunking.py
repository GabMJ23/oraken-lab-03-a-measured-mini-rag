"""Part A: the two chunkers. Run with: pytest -q"""
import inspect
import math

import pytest

import rag

TINY = [
    {"id": "1", "heading": "A b", "text": "c d e f"},  # 7 words with its number and heading
    {"id": "2", "heading": "G", "text": "h i"},        # 4 words
]


@pytest.fixture(scope="module")
def corpus():
    return rag.load_corpus()


def test_corpus_is_intact(corpus):
    ids = [s["id"] for s in corpus]
    assert len(corpus) == 30, f"data/corpus.json should hold 30 sections, it holds {len(corpus)}. Restore it from the template."
    assert "825.220" in ids, "Section 825.220 is missing from data/corpus.json. Restore the file from the template."


def test_sections_one_chunk_per_section(corpus):
    chunks = rag.chunk_by_section(corpus)
    assert len(chunks) == len(corpus), f"chunk_by_section should return one chunk per section ({len(corpus)}), not {len(chunks)}."
    for s, c in zip(corpus, chunks):
        assert c["ref"] == s["id"], f"The chunk of section {s['id']} should have ref '{s['id']}', not {c.get('ref')!r}."
        assert c["refs"] == [s["id"]], f"The chunk of section {s['id']} should have refs ['{s['id']}']."
        assert c["text"] == rag.section_text(s), "A section chunk's text should be section_text(section): number, heading and body."
    assert len({c["id"] for c in chunks}) == len(chunks), "Every chunk needs a unique id."


def test_words_default_size_is_120():
    size = inspect.signature(rag.chunk_by_words).parameters["size"].default
    assert size == 120, f"chunk_by_words should cut 120-word chunks by default, not {size}."


def test_words_on_a_tiny_corpus():
    chunks = rag.chunk_by_words(TINY, size=5)
    texts = [c["text"].split() for c in chunks]
    assert texts == [["1", "A", "b", "c", "d"], ["e", "f", "2", "G", "h"], ["i"]], (
        f"With size=5 the 11 words should be cut 5 + 5 + 1, ignoring the section boundary. Got {texts}."
    )
    assert [c["refs"] for c in chunks] == [["1"], ["1", "2"], ["2"]], (
        "refs must list every section a chunk has words from, in corpus order: the middle chunk spans sections 1 and 2."
    )
    assert [c["ref"] for c in chunks] == ["1", "1", "2"], "ref is the section of the chunk's first word."


def test_words_sizes_and_count(corpus):
    chunks = rag.chunk_by_words(corpus)
    total = sum(len(rag.section_text(s).split()) for s in corpus)
    assert len(chunks) == math.ceil(total / 120), (
        f"{total} words cut every 120 should give {math.ceil(total / 120)} chunks, not {len(chunks)}."
    )
    sizes = [len(c["text"].split()) for c in chunks]
    assert all(n == 120 for n in sizes[:-1]), "Every chunk except the last one must have exactly 120 words."
    assert 1 <= sizes[-1] <= 120, "The last chunk holds the words that are left (between 1 and 120)."
    stream = [w for s in corpus for w in rag.section_text(s).split()]
    assert [w for c in chunks for w in c["text"].split()] == stream, "Put together, the chunks must give back every word, in order, once."
    assert len({c["id"] for c in chunks}) == len(chunks), "Every chunk needs a unique id."


def test_words_refs_follow_the_text(corpus):
    chunks = rag.chunk_by_words(corpus)
    order = [s["id"] for s in corpus]
    assert chunks[0]["refs"] == ["825.100"], "The first chunk starts in section 825.100."
    spanning = [c for c in chunks if len(c["refs"]) > 1]
    assert spanning, "Some 120-word chunks must span two sections: refs should then list both."
    for c in chunks:
        assert c["ref"] == c["refs"][0], "ref should be the first item of refs (the section of the first word)."
        idx = [order.index(r) for r in c["refs"]]
        assert idx == sorted(idx), f"Chunk {c['id']}: refs should follow the corpus order, got {c['refs']}."
    holder = [c for c in chunks if "825.220 Protection for employees" in c["text"]]
    assert holder and "825.220" in holder[0]["refs"], "The chunk holding the heading of 825.220 must have 825.220 in its refs."
