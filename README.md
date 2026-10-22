# Lab 3: A measured mini-RAG

**Part 2 · Lessons 5 and 6 · about 3 hours · automatically checked**

## Goals

- Build a search by meaning over a small legal corpus: 30 sections of the US Family and Medical Leave Act (FMLA) regulations.
- Compare two ways of cutting that corpus with a number, recall@5, rather than an impression.
- Add keyword search, fuse the two ranked lists, and find out on which questions it helps and on which it hurts.

## Before you start

You need Python 3.12 and Git. Create your own copy of this lab (**Use this template** on GitHub), clone it, then:

```bash
python -m venv .venv
# macOS / Linux: source .venv/bin/activate      Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q          # most checks fail at first: that is expected
```

Rules for every lab: one Git repository, one commit per step, your answers in `REPORT.md`, and no secret in Git.

For the real embedding model (Parts B and C), also run `pip install -r requirements-local.txt` on your own machine. It installs `sentence-transformers` and PyTorch (a few hundred MB), and the first run downloads the model (about 130 MB). The automated checks never download a model: they use `rag.hashed_embed`, a fake embedder that only counts words. That is why `search()` takes the embedder as a parameter.

### The corpus

`data/corpus.json` holds 30 sections of **29 CFR Part 825**, the Department of Labor regulations that implement the FMLA. The FMLA lets eligible employees take up to 12 weeks of unpaid, job-protected leave, for example for their own serious illness. This is the course's running example: an employee is sick and wants to know whether their job is safe.

- Each item is `{"id": "825.220", "heading": "...", "text": "..."}`. `data/corpus.txt` holds the same text, one section per paragraph, preceded by its number.
- Source: the Electronic Code of Federal Regulations (eCFR), [ecfr.gov](https://www.ecfr.gov/current/title-29/subtitle-B/chapter-V/subchapter-C/part-825), text up to date as of 2026-10-21, retrieved on 2026-10-23. Federal regulations are in the public domain.
- To fetch the whole part yourself (the API refuses uncompressed answers, so keep `--compressed`):
  ```bash
  curl --compressed -o part825.xml "https://www.ecfr.gov/api/versioner/v1/full/2026-10-21/title-29.xml?part=825"
  ```

Do not edit the corpus: the checks rely on its 30 sections.

## Part A: Prepare the corpus

The top of `rag.py` defines a chunk: a dict with an `id`, the `text` to search, and two pieces of metadata, `ref` and `refs`. You need them to measure: to know whether a retrieved chunk "is" the expected section, the chunk must remember where it came from.

1. Write `chunk_by_section(corpus)`: one chunk per section. The text is `section_text(section)` (number, heading, body); `ref` is the section id and `refs` is a one-item list.
2. Write `chunk_by_words(corpus, size=120)`: read the 30 sections, in order, as one long stream of words, and cut it every 120 words, whatever the section boundaries. `refs` lists every section the chunk has at least one word from (a chunk that straddles two sections has two refs). `ref` is the section of its first word.
3. Count your chunks. How many of each kind? How many word chunks straddle two sections? How long is the longest section, in words? Answer in `REPORT.md`.

## Part B: Search by meaning

1. Write `normalize(vectors)`, then `search(question, chunks, embed, k=5)`. Embed the question, compute its cosine similarity with every chunk, return the 5 best. Once every vector has length 1, a dot product **is** the cosine. `embed` is any function that turns a list of texts into one vector per text.
2. Try the real model on the course's question:
   ```bash
   python measure.py --ask "Can my employer fire me for taking medical leave?"
   ```
   Section 825.220 protects employees who take FMLA leave. Is it in the top 5, for each chunking? At which rank?

Two open embedding models work here. `BAAI/bge-small-en-v1.5` is the default: light, English, it reads at most 512 tokens per text (about 380 words). `BAAI/bge-m3` is heavier (about 2.3 GB), multilingual, reads up to 8,192 tokens, and is the one Nyambot uses: add `--model BAAI/bge-m3`. On a laptop CPU, a full `measure.py` run takes about a minute with bge-small and closer to ten with bge-m3. Keep the 512-token limit in mind when you look at long sections.

## Part C: Measure

1. Complete `data/questions.json`: at least **10 everyday questions**, each with the id of the section that answers it. Two are given; add eight. Write them the way a worried employee would, not a lawyer, and never put a section number in a question. **At least three** must share no content word with their expected section: those are the questions where keyword search is blind and meaning must do the work. `rag.overlap(question, section)` returns the shared words (it ignores the stop-words in `rag.STOP_WORDS`; it does not stem, so "fire" and "fired" are different words).
2. Write `recall_at_k(results_per_question, expected, k=5)`: the share of questions whose expected section appears in the `refs` of at least one of their first 5 chunks.
3. Write `bm25_search(question, chunks, k=5)` with `rank_bm25`: keyword search, the kind that finds "825.220" or "foster care" exactly.
4. Write `rrf(lists, k=60)`, Reciprocal Rank Fusion: every id gets the sum of `1 / (k + rank)` over the lists it appears in. `hybrid_search` (already written) uses it to fuse the top 20 of each search.
5. Run the measurement: `python measure.py --fake` first (instant, checks that everything runs), then `python measure.py` with the real model. It prints the table chunking × search → recall@5, then the rank of the expected section for every question. Run it twice: you get the same numbers. That is the point of a measurement.
6. Paste the table in `REPORT.md` and explain the gaps. Use the second table: name the questions that one setup finds and the other misses, and say why.

## Deliverable and success criteria

- `rag.py` complete, `data/questions.json` with 10 questions or more, and `REPORT.md` completed: the table chunking × search → recall@5, and an explanation of the gaps based on specific questions.
- **Success:** `pytest -q` is all green. `python measure.py` prints the same numbers every time. Your explanation cites questions by id, not impressions.
- Each time you push, GitHub runs the same checks (the **Actions** tab shows a green or red mark).

**Going further:** store the vectors in PostgreSQL with the pgvector extension, and let the database do the top 5. Start one with `docker run -d -e POSTGRES_PASSWORD=dev -p 5432:5432 pgvector/pgvector:pg16`, then `CREATE EXTENSION vector;` and a table `chunks (id text PRIMARY KEY, refs text[], text text, embedding vector(384))` (384 is the size of bge-small vectors). Query with `ORDER BY embedding <=> %s LIMIT 5` (`<=>` is the cosine distance) and check that you get the same top 5 as `search()`. You will meet pgvector again in Lab 4.

Stuck? Open `HINTS.md`: three hints, from gentle to almost the answer.
