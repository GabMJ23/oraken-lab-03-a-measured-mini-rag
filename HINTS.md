# Lab 3 hints

Read one hint at a time.

<details>
<summary>Hint 1: cutting by words without losing track of sections</summary>

Do not cut each section separately: that would respect the boundaries. Build one list of pairs first, each word with the section it comes from:

```python
stream = [(word, s["id"]) for s in corpus for word in section_text(s).split()]
```

Then walk through it in steps of `size` (`range(0, len(stream), size)`). For each window, the text is the words joined with spaces, and `refs` is the list of section ids in the window without duplicates, in order. `list(dict.fromkeys(ids))` does exactly that.

For questions with no shared word, think of how people talk: "boss", "rehab", "chemo", "doctor's note" rarely appear in a regulation. Check each one with `rag.overlap(...)`, as shown in the failing check.

</details>

<details>
<summary>Hint 2: cosine similarity and top k with numpy</summary>

Divide each row by its length: `v / np.linalg.norm(v, axis=1, keepdims=True)`. Replace zero lengths by 1 first, or you divide by zero.

Once the chunk vectors (one row each) and the question vector are normalized, `scores = vectors @ q` gives one cosine per chunk. `np.argsort(-scores, kind="stable")[:k]` gives the positions of the k best, best first, ties in corpus order. Return `[chunks[i] for i in order]`.

`bm25_search` follows the same pattern: `BM25Okapi([tokenize(c["text"]) for c in chunks])`, then `index.get_scores(tokenize(question))` gives one score per chunk.

</details>

<details>
<summary>Hint 3: recall@k and RRF, almost the answer</summary>

```python
hits = 0
for results, exp in zip(results_per_question, expected):
    if any(exp in c["refs"] for c in results[:k]):
        hits += 1
return hits / len(expected)
```

For RRF, keep a dict `scores` from id to fused score. For each list, `for rank, item in enumerate(ranked, start=1)`, add `1 / (k + rank)` to `scores[item]` (start at 0.0). Return `sorted(scores, key=lambda item: -scores[item])`. Python's sort is stable and a dict remembers insertion order, so ties keep the order in which ids were first seen.

</details>
