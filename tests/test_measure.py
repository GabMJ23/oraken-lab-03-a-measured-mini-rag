"""Part C: recall@k, Reciprocal Rank Fusion, and a reproducible measurement. Run with: pytest -q"""
import pytest

import measure
import rag


def ch(*refs):
    """A hand-made chunk that covers the given sections."""
    return {"refs": list(refs)}


def test_recall_counts_hits():
    results = [[ch("a"), ch("b")], [ch("c")]]
    assert rag.recall_at_k(results, ["b", "x"], k=5) == pytest.approx(0.5), (
        "Question 1 finds 'b', question 2 never finds 'x': recall should be 1/2 = 0.5."
    )


def test_recall_only_looks_at_the_first_k():
    results = [[ch("a"), ch("b"), ch("c"), ch("d"), ch("e"), ch("target")]]
    assert rag.recall_at_k(results, ["target"], k=5) == 0.0, "The 6th result does not count for recall@5."
    assert rag.recall_at_k(results, ["target"], k=6) == 1.0, "With k=6, the 6th result counts."
    assert rag.recall_at_k([[ch("a"), ch("b")]], ["b"], k=1) == 0.0, "With k=1, only the first result counts."


def test_recall_uses_all_refs_of_a_chunk():
    assert rag.recall_at_k([[ch("a", "b")]], ["b"], k=5) == 1.0, (
        "A chunk that spans sections a and b is a hit for b: look at every item of chunk['refs']."
    )


def test_recall_counts_a_question_once():
    results = [[ch("a"), ch("a"), ch("a")], [ch("z")]]
    assert rag.recall_at_k(results, ["a", "b"], k=5) == pytest.approx(0.5), (
        "Several chunks of the right section still count as one hit: recall is per question, at most 1."
    )


def test_rrf_rewards_agreement():
    assert rag.rrf([["a", "b", "c"], ["b", "c", "a"]]) == ["b", "a", "c"], (
        "b is ranked 2nd and 1st, so 1/62 + 1/61 beats a (1/61 + 1/63) and c (1/63 + 1/62)."
    )


def test_rrf_keeps_items_found_by_one_list_only():
    fused = rag.rrf([["a", "b"], ["c"]])
    assert sorted(fused) == ["a", "b", "c"], "rrf must return every id that appears in any list."
    assert fused[-1] == "b", "b (rank 2 in one list) scores less than a and c (rank 1 in one list each)."


def test_rrf_uses_k():
    lists = [["a", "b", "c"], ["d", "e", "c"]]
    assert rag.rrf(lists)[0] == "c", "With k=60, c (3rd in both lists) beats a (1st in one list)."
    assert rag.rrf(lists, k=0)[0] == "a", "With k=0, 1/1 for a beats 1/3 + 1/3 for c: use the k argument."


def test_hybrid_search_runs():
    chunks = rag.chunk_by_section(rag.load_corpus())
    top = rag.hybrid_search("leave for adoption or foster care", chunks, rag.hashed_embed, k=5)
    assert len(top) == 5 and top[0]["ref"] == "825.121", "hybrid_search should put 825.121 first for this question."


def test_measure_is_reproducible():
    first = measure.run(rag.hashed_embed)
    second = measure.run(rag.hashed_embed)
    assert first == second, "Running measure.py twice with the same embedder must give identical numbers."
    assert len(first["recall"]) == 6, "measure.run should report 2 chunkings x 3 searches = 6 recall values."
    for key, value in first["recall"].items():
        assert 0.0 <= value <= 1.0, f"{key}: recall must be between 0 and 1, got {value}."
