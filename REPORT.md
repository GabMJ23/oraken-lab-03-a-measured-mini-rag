# Lab 3 report

Replace every `TODO` with your answer. The automated checks make sure nothing is left blank.

## Part A: Prepare the corpus

**A1.** How many chunks does each strategy produce? How many 120-word chunks straddle two sections?

- `chunk_by_section`: TODO
- `chunk_by_words`: TODO
- Word chunks with two refs: TODO

**A2.** What is the longest section, and how many words does it have? What is the shortest? Name one risk of each when you embed it as a single vector.

TODO

## Part B: Search by meaning

**B1.** Which embedding model did you use? (If you could not install it, say so and use `--fake`.)

TODO

**B2.** Run `python measure.py --ask "Can my employer fire me for taking medical leave?"`. Is 825.220 in the top 5? At which rank, for each chunking? Which sections came first?

- Sections: TODO
- 120-word chunks: TODO

**B3.** Pick one chunk in that top 5 that is *not* 825.220. Why did the model find it close to the question?

TODO

## Part C: Measure

**C1.** Which of your questions share no content word with their expected section (ids)? Give one of them and explain how you wrote it.

TODO

**C2.** Paste the recall@5 table printed by `python measure.py` (chunking × search, six rows).

TODO

**C3.** Explain the gap between section chunks and 120-word chunks. Cite the questions (by id, for example q05) that one finds and the other misses, and say why.

TODO

**C4.** Did keyword search and fusion (BM25, then RRF) change recall? On which questions did it help, and on which did it hurt? Why?

TODO

**C5.** A 120-word chunk can carry two refs and count as a hit for both sections. Does that make the comparison unfair? What would you change to make it stricter?

TODO

**Going further (optional).** Did you store the vectors in PostgreSQL with pgvector? Do you get the same top 5 as `search()`?

TODO
