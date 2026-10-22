"""Part B (search by meaning) and the keyword search of Part C.

These tests never download a model: they use rag.hashed_embed, a fake embedder,
or a hand-made one. Run with: pytest -q
"""
import numpy as np
import pytest

import rag

# A hand-made embedder: each text has a fixed vector, chosen so that the cosine
# order (b, a, c) differs from the raw dot-product order (a, b, c).
VECTORS = {
    "q": [1.0, 1.0],
    "a": [10.0, 0.0],   # long vector, 45 degrees away: big dot product, cosine 0.71
    "b": [1.0, 0.9],    # short vector, almost the same direction: cosine 0.999
    "c": [0.0, -1.0],   # opposite-ish: cosine -0.71
}
HAND = [{"id": t, "text": t, "ref": t, "refs": [t]} for t in "abc"]


def hand_embed(texts):
    return np.array([VECTORS[t] for t in texts])


@pytest.fixture(scope="module")
def corpus():
    return rag.load_corpus()


def test_normalize_gives_unit_rows():
    v = rag.normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0), "normalize must scale every row to length 1."
    assert np.allclose(v[0], [0.6, 0.8]), "normalize([3, 4]) should be [0.6, 0.8]."


def test_normalize_keeps_zero_rows():
    v = rag.normalize(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert not np.isnan(v).any(), "A zero vector must stay zero, not become NaN (do not divide by 0)."
    assert np.allclose(v[0], [0.0, 0.0])


def test_search_uses_cosine_not_raw_dot_product():
    got = [c["id"] for c in rag.search("q", HAND, hand_embed, k=3)]
    assert got == ["b", "a", "c"], (
        f"Expected ['b', 'a', 'c'] (best cosine first), got {got}. "
        "If you got ['a', 'b', 'c'], you ranked by raw dot product: normalize the vectors first."
    )


def test_search_returns_k_chunks():
    assert len(rag.search("q", HAND, hand_embed, k=2)) == 2, "search(..., k=2) must return 2 chunks."
    assert rag.search("q", HAND, hand_embed, k=1)[0]["id"] == "b"


def test_search_accepts_precomputed_vectors():
    vectors = rag.embed_chunks(HAND, hand_embed)
    got = [c["id"] for c in rag.search("q", HAND, hand_embed, k=3, vectors=vectors)]
    assert got == ["b", "a", "c"], "When vectors are passed, search should use them and give the same order."


def test_search_on_the_corpus_with_the_fake_embedder(corpus):
    chunks = rag.chunk_by_section(corpus)
    top = rag.search("leave for adoption or foster care", chunks, rag.hashed_embed, k=5)
    assert len(top) == 5
    assert top[0]["ref"] == "825.121", (
        "With the fake word-based embedder, the question 'leave for adoption or foster care' "
        f"should find section 825.121 first, not {top[0]['ref']}."
    )


def test_bm25_finds_an_exact_section_number(corpus):
    chunks = rag.chunk_by_section(corpus)
    top = rag.bm25_search("825.220", chunks, k=5)
    assert len(top) == 5, "bm25_search(..., k=5) must return 5 chunks."
    assert top[0]["ref"] == "825.220", (
        f"A keyword search for '825.220' should rank section 825.220 first, not {top[0]['ref']}. "
        "Tokenize the chunks and the question with rag.tokenize."
    )


def test_bm25_on_a_tiny_set():
    chunks = [
        {"id": "x", "text": "the cat sat on the mat", "refs": ["x"]},
        {"id": "y", "text": "dogs bark at night", "refs": ["y"]},
        {"id": "z", "text": "a cat and a dog", "refs": ["z"]},
        {"id": "w", "text": "bread and butter", "refs": ["w"]},
    ]
    got = [c["id"] for c in rag.bm25_search("dogs bark", chunks, k=2)]
    assert got[0] == "y", f"The only chunk containing 'dogs bark' should come first, got {got}."
