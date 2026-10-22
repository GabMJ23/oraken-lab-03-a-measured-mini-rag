"""Lab 3: a measured mini-RAG over 30 sections of the FMLA regulations (29 CFR Part 825).

Your job: replace every `raise NotImplementedError` (Parts A, B and C).

A chunk is a dict:
    {"id": "sec-825.220" or "w-007",   # unique id of the chunk
     "text": "...",                     # what gets embedded and searched
     "ref": "825.220",                  # the section where the chunk starts
     "refs": ["825.219", "825.220"]}    # every section the chunk contains words from

Run the measurement: python measure.py --fake   (fast, no download)
                     python measure.py          (real embedding model, local only)
Test it:             pytest -q
"""
import hashlib
import json
import pathlib
import re

import numpy as np
from rank_bm25 import BM25Okapi

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE / "data" / "corpus.json"
QUESTIONS = HERE / "data" / "questions.json"

# ---------------------------------------------------------------------------
# Provided helpers (do not change them: the tests rely on them)
# ---------------------------------------------------------------------------

# Common English words that carry no topic. Used to decide whether a question
# "shares a word" with a section. Exact word forms, no stemming: "fire" and
# "fired" are different words.
STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing done down
during each even ever every few for from further get gets getting got had has
have having he her here hers herself him himself his how i if in into is it its
itself just let like may me might more most much must my myself no nor not now
of off on once only or other our ours ourselves out over own really same shall
she should so some still such than that the their theirs them themselves then
there these they this those through to too under until up upon us very was we
were what when where which while who whom why will with would yet you your
yours yourself yourselves s t ve ll re d m
""".split())


def load_corpus(path=CORPUS):
    """The 30 sections, as a list of {"id", "heading", "text"}."""
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_questions(path=QUESTIONS):
    """Your evaluation set, as a list of {"id", "question", "expected"}."""
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def tokenize(text):
    """Lowercase words. Section numbers such as 825.220 stay one token."""
    return re.findall(r"\d+(?:\.\d+)+|[a-z0-9]+", text.lower())


def content_words(text):
    """The set of words in `text` that are not stop-words and not pure numbers."""
    return {w for w in tokenize(text) if w not in STOP_WORDS and not w[0].isdigit()}


def overlap(question, section):
    """Content words shared by a question and a section (heading + text)."""
    return content_words(question) & content_words(section["heading"] + " " + section["text"])


def section_text(section):
    """The text of one section as the chunkers see it: number, heading, body."""
    return f"{section['id']} {section['heading']}\n{section['text']}"


def hashed_embed(texts, dim=512):
    """A FAKE embedder: hashed bag of words. Deterministic, instant, no download.

    It only knows words, not meaning: two texts are close if they share words
    (stop-words are skipped). The tests use it. Your report uses a real model
    (see measure.py). Returns an array of shape (len(texts), dim).
    """
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for w in tokenize(t):
            if w in STOP_WORDS:
                continue
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            out[i, h % dim] += 1.0
    return out


# ---------------------------------------------------------------------------
# Part A: two ways to cut the corpus
# ---------------------------------------------------------------------------

def chunk_by_section(corpus):
    """One chunk per section. id "sec-<section id>", ref = refs[0] = the section id."""
    # TODO Part A1: one dict per section, built with section_text(s).
    raise NotImplementedError


def chunk_by_words(corpus, size=120):
    """Fixed windows of `size` words that ignore section boundaries.

    All sections are read in order as one long stream of words (section_text,
    split on whitespace). Every chunk has exactly `size` words, except the last
    one, which has what is left. ids are "w-000", "w-001", ...
    ref  = the section of the chunk's first word.
    refs = every section the chunk has at least one word from, in corpus order.
    """
    # TODO Part A2: build the stream of (word, section id) pairs, then cut it every `size` words.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part B: search by meaning
# ---------------------------------------------------------------------------

def normalize(vectors):
    """Scale every row to length 1, so that a dot product is a cosine similarity."""
    # TODO Part B1: divide each row by its length (np.linalg.norm). Leave all-zero rows at zero.
    raise NotImplementedError


def embed_chunks(chunks, embed):
    """Embed every chunk once. Returns a normalized matrix, one row per chunk."""
    return normalize(embed([c["text"] for c in chunks]))


def search(question, chunks, embed, k=5, vectors=None):
    """The k chunks closest in meaning to the question, best first.

    embed: a function that takes a list of strings and returns one vector per string.
    vectors: optional precomputed embed_chunks(chunks, embed), to avoid re-embedding.
    Similarity is the cosine; ties keep the corpus order.
    """
    # TODO Part B2: embed the question, normalize it, dot product with every chunk vector,
    # then keep the k best (np.argsort(-scores, kind="stable") keeps ties in corpus order).
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part C: measure, then add keywords
# ---------------------------------------------------------------------------

def recall_at_k(results_per_question, expected, k=5):
    """Share of questions whose expected section is found in the first k results.

    results_per_question[i] is the ranked list of chunks returned for question i;
    expected[i] is the section id that question i should find. A question is a
    hit when expected[i] is in the "refs" of at least one of its first k chunks.
    Returns a float between 0 and 1.
    """
    # TODO Part C2
    raise NotImplementedError


def bm25_search(question, chunks, k=5, index=None):
    """The k chunks with the best BM25 keyword score, best first (ties: corpus order).

    index: optional prebuilt BM25Okapi over the same chunks, to avoid rebuilding it.
    """
    # TODO Part C3: BM25Okapi over tokenize(chunk text); index.get_scores(tokenize(question)).
    raise NotImplementedError


def rrf(lists, k=60):
    """Reciprocal Rank Fusion of several ranked lists of ids.

    Each id gets the sum over the lists of 1 / (k + rank), rank starting at 1.
    An id missing from a list gets nothing from that list.
    Returns all ids, best fused score first; ties keep first-seen order.
    """
    # TODO Part C4
    raise NotImplementedError


def hybrid_search(question, chunks, embed, k=5, vectors=None, index=None, depth=20):
    """Meaning + keywords: fuse the top `depth` of each search with rrf, keep k."""
    by_id = {c["id"]: c for c in chunks}
    dense = [c["id"] for c in search(question, chunks, embed, k=depth, vectors=vectors)]
    keyword = [c["id"] for c in bm25_search(question, chunks, k=depth, index=index)]
    return [by_id[i] for i in rrf([dense, keyword])[:k]]
