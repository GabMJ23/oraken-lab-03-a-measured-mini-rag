"""Lab 3: the measurement. You do not need to change this file.

    python measure.py --fake                 # fake embedder: instant, no download
    python measure.py                        # real model (needs requirements-local.txt)
    python measure.py --model BAAI/bge-m3    # the heavier model Nyambot uses
    python measure.py --ask "Can my employer fire me for taking medical leave?"

It cuts the corpus both ways, runs the three searches on every question of
data/questions.json and prints recall@5 for each pair, then the rank of the
expected section question by question (the material for your explanation).
"""
import argparse

import rag

CHUNKINGS = {"sections": rag.chunk_by_section, "words-120": rag.chunk_by_words}
SEARCHES = ("dense", "bm25", "hybrid")
DEPTH = 20  # how far down we look to report a rank


def sentence_transformer_embed(model_name="BAAI/bge-small-en-v1.5"):
    """A real embedder. Downloads the model the first time (local only, never in CI)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts):
        return model.encode(list(texts), normalize_embeddings=True, batch_size=16)

    return embed


def rank_of(expected, results):
    """1-based rank of the first chunk that contains the expected section, or None."""
    for i, c in enumerate(results, start=1):
        if expected in c["refs"]:
            return i
    return None


def run(embed, corpus=None, questions=None, k=5):
    """Run every (chunking, search) pair. Returns plain dicts, easy to compare and print."""
    corpus = corpus if corpus is not None else rag.load_corpus()
    questions = questions if questions is not None else rag.load_questions()
    expected = [q["expected"] for q in questions]
    recall, ranks = {}, {}
    for cname, chunker in CHUNKINGS.items():
        chunks = chunker(corpus)
        vectors = rag.embed_chunks(chunks, embed)
        index = rag.BM25Okapi([rag.tokenize(c["text"]) for c in chunks])
        for sname in SEARCHES:
            results = []
            for q in questions:
                if sname == "dense":
                    r = rag.search(q["question"], chunks, embed, k=DEPTH, vectors=vectors)
                elif sname == "bm25":
                    r = rag.bm25_search(q["question"], chunks, k=DEPTH, index=index)
                else:
                    r = rag.hybrid_search(q["question"], chunks, embed, k=DEPTH,
                                          vectors=vectors, index=index, depth=DEPTH)
                results.append(r)
            key = f"{cname} / {sname}"
            recall[key] = round(rag.recall_at_k(results, expected, k=k), 3)
            ranks[key] = {q["id"]: rank_of(q["expected"], r) for q, r in zip(questions, results)}
    return {"k": k, "recall": recall, "ranks": ranks}


def print_report(result, questions):
    k = result["k"]
    print(f"| Chunking | Search | recall@{k} |")
    print("|---|---|---|")
    for key, value in result["recall"].items():
        cname, sname = key.split(" / ")
        print(f"| {cname} | {sname} | {value:.2f} |")
    print()
    keys = list(result["ranks"])
    print(f"Rank of the expected section (- = not in the top {DEPTH}; a hit needs rank <= {k}):")
    print("| Question | Expected | " + " | ".join(keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for q in questions:
        cells = []
        for key in keys:
            r = result["ranks"][key][q["id"]]
            cells.append("-" if r is None else str(r))
        print(f"| {q['id']} | {q['expected']} | " + " | ".join(cells) + " |")


def ask(question, embed, k=5):
    corpus = rag.load_corpus()
    for cname, chunker in CHUNKINGS.items():
        chunks = chunker(corpus)
        print(f"\n{cname}: top {k} for {question!r}")
        for i, c in enumerate(rag.search(question, chunks, embed, k=k), start=1):
            preview = " ".join(c["text"].split()[:12])
            print(f"  {i}. {c['id']:<12} refs={','.join(c['refs'])}  {preview}...")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="use the fake hashed bag-of-words embedder")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="sentence-transformers model name")
    p.add_argument("--ask", help="print the top 5 chunks for one question, for both chunkings")
    args = p.parse_args()
    embed = rag.hashed_embed if args.fake else sentence_transformer_embed(args.model)
    print("embedder:", "fake (hashed bag of words)" if args.fake else args.model)
    if args.ask:
        ask(args.ask, embed)
        return
    questions = rag.load_questions()
    print_report(run(embed, questions=questions), questions)


if __name__ == "__main__":
    main()
