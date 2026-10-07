# Evaluation

All numbers below come from real runs on this machine (AMD Ryzen 5 7530U, 16 GB RAM, CPU only).
Reproduce them with:

```bash
python -m scripts.download_corpus      # the 10 open-access arXiv papers, if data/ is empty
python -m scripts.eval retrieval       # retrieval metrics -> storage/eval/retrieval.json
python -m scripts.eval report          # print the tables below
```

## Setup

- **Corpus**: 10 open-access arXiv papers (Attention Is All You Need, BERT, RoBERTa, LoRA, ColBERT,
  DPR, RAG, Sentence-BERT, T5, ViT), 982 chunks. The **small corpus** is 3 of them (Attention,
  BERT, RAG; 227 chunks).
- **Questions**: `eval/questions.json`: 49 answerable questions spread over all 10 papers, each
  with the expected document and page(s), and 12 unanswerable questions. The questions were
  drafted with AI help from the extracted text, then reviewed by hand against the PDFs.
- **Hit**: a retrieved chunk comes from the expected document **and** one of the expected pages.
  "Doc hit" only requires the right document. **MRR** is the mean of 1/rank of the first correct
  chunk (0 if none is retrieved).
- **Latency** is retrieval only (no LLM), measured after one warm-up query.
- The baseline returns 3 chunks (as in the original script); all other methods return 5, so the
  "Dense top-5" row isolates the effect of k from the effect of hybrid search.

## Retrieval results

### Small corpus (3 PDFs, 227 chunks, 19 questions)

| Method | Hit@k (page) | Doc hit | MRR | Mean latency | p95 latency |
|---|---|---|---|---|---|
| Baseline: dense top-3 (original main.py) | 0.789 | 1.000 | 0.640 | 33 ms | 38 ms |
| Dense top-5 | 0.895 | 1.000 | 0.664 | 37 ms | 43 ms |
| Hybrid BM25 + dense (RRF) top-5, no rerank | 0.947 | 1.000 | 0.829 | 37 ms | 46 ms |
| Hybrid + cross-encoder rerank top-5 | 0.947 | 1.000 | 0.781 | 1905 ms | 2412 ms |

### Full corpus (10 PDFs, 982 chunks, 49 questions)

| Method | Hit@k (page) | Doc hit | MRR | Mean latency | p95 latency |
|---|---|---|---|---|---|
| Baseline: dense top-3 (original main.py) | 0.612 | 0.980 | 0.500 | 34 ms | 47 ms |
| Dense top-5 | 0.673 | 1.000 | 0.515 | 37 ms | 56 ms |
| Hybrid BM25 + dense (RRF) top-5, no rerank | 0.857 | 1.000 | 0.661 | 44 ms | 61 ms |
| Hybrid + cross-encoder rerank top-5 | 0.878 | 1.000 | 0.664 | 1935 ms | 2704 ms |

### What the numbers say

- **Hybrid search is the biggest win.** On the full corpus, page-level hit goes from 0.612
  (baseline) to 0.857 with BM25 + dense fusion, and MRR from 0.500 to 0.661. Many questions hinge
  on exact tokens ("warmup steps", "GELU", "C4", "WordPiece") that BM25 matches and MiniLM
  embeddings blur.
- **The gap grows with corpus size.** With 3 PDFs the baseline already finds the right page 79%
  of the time; with 10 PDFs it drops to 61%, while hybrid + rerank drops only from 0.947 to 0.878.
  More documents means more near-duplicate topics (BERT, RoBERTa and Sentence-BERT all discuss
  masking, pre-training and NLI), and dense-only search confuses them.
- **The reranker adds little to hit rate and MRR here**: +0.021 hit on the full corpus, and
  slightly *lower* MRR than plain fusion on the small corpus (0.781 vs 0.829, which is one or two
  questions out of 19). It costs about 1.9 s per query on this CPU (20 pairs at about 60 ms each;
  thread count tuning did not help). It is still kept because its score is a calibrated relevance
  signal: it powers the relevance gate in the next section, which plain RRF scores cannot do.
- **Document-level retrieval is essentially solved** (doc hit 1.000 for every method except the
  baseline); the hard part is finding the exact page.

### Limitations

- 49 questions is a small sample: one question is about 2 percentage points of hit rate, so
  differences of a few points are within noise.
- The questions were written by looking at the text, so they share vocabulary with the source
  pages. That favours BM25 somewhat compared with real user questions.
- Expected pages were checked by hand, but any remaining wrong label counts as a miss for every
  method.

## Hallucination guards

Reproduce with:

```bash
python -m scripts.calibrate            # relevance threshold from the saved reranker scores
python -m scripts.eval_guards          # full pipeline with the real LLM -> storage/eval/guards.json
```

### Relevance-gate calibration

For every question, the top cross-encoder score on the full corpus was recorded
(`scripts/eval.py retrieval`):

| | n | min | median | max |
|---|---|---|---|---|
| Answerable | 49 | -1.38 | 5.24 | 8.64 |
| Unanswerable | 12 | -11.07 | -9.10 | 4.55 |

`scripts/calibrate.py` picks the threshold with the best balanced accuracy and places it in the
middle of the gap between neighbouring observed scores: **-0.11**. At that threshold the gate
lets through 97.96% of answerable questions (48/49) and blocks 91.67% of unanswerable ones (11/12)
before any LLM call. The one unanswerable question that passes ("What learning rate was used to
train GPT-4?", score 4.55) matches passages about learning rates in other papers; the one
answerable question that is blocked ("Which activation function does BERT use instead of
ReLU?", score -1.38) is answered by a single "gelu" mention on a hyperparameter page.

Caveat: the threshold was calibrated on the same questions it is evaluated on (there is no
held-out set), so the gate numbers below are optimistic.

### End-to-end results (guarded single-pass pipeline)

`qwen2.5:3b` via Ollama, threshold -0.11, full corpus, 49 answerable + 12 unanswerable questions.

| Metric | Value |
|---|---|
| Abstention rate on unanswerable questions | 100.0% (12/12) |
| False abstention rate on answerable questions | 20.4% (10/49) |
| Answered questions with at least one valid citation | 79.5% (31/39) |
| Answered questions citing the expected page | 56.4% (22/39) |
| Invalid citations dropped (total) | 0 |
| Abstain reasons, unanswerable | relevance gate: 11, model said not found: 1 |
| Abstain reasons, answerable | groundedness check: 6, model said not found: 3, relevance gate: 1 |
| Latency per question (mean / median) | 27.3 s / 33.4 s |

### What the numbers say

- **Abstention works.** All 12 unanswerable questions got "Not found in the provided documents."
  11 were stopped by the relevance gate in about 2 s without calling the LLM; the GPT-4 question
  that slipped through was refused by the model itself.
- **The price is false abstentions**: 10 of 49 answerable questions were refused. 6 of those were
  rejected by the groundedness check, which with a 3B judge is sometimes wrong on correct
  answers. This is the main quality limitation, and the place a larger local model
  (e.g. `qwen2.5:7b`) would help most.
- **Citations**: 79.5% of answers carry a valid citation and 56.4% cite the exact expected page.
  Common failure: the model quotes a passage verbatim but doesn't add the passage id, or it
  cites a neighbouring passage from the same paper (e.g. rag.pdf p. 18 instead of p. 4). No
  answer cited a page outside the retrieved set, so the validator didn't need to drop anything
  in this run (it does drop invented ids in testing).
- **Answers that pass the guards can still be wrong.** Reading all 39 answers by hand shows the
  guards check *support by the retrieved passages*, not *correctness for the question*. For
  example, "How many warmup steps were used in the Transformer's learning rate schedule?" was
  answered with T5's warmup setting (10^4, rendered as "104"), which is supported by a retrieved
  T5 passage. The question is ambiguous across papers, and the answer cites t5.pdf, so the user
  can see where it came from. No reference answers exist in the eval set, so answer accuracy is
  not reported as a number.
- **Latency**: about 30 s per answered question on CPU, mostly generation (about 1,500 prompt
  tokens and a short answer). The groundedness check costs about 2 s because it shares its prompt
  prefix with the answer call, so Ollama reuses the processed context. Abstentions from the gate
  take about 2 s.

### Prompt iterations (on a handful of eval questions)

The first real run exposed three problems, each fixed and re-tested before the full run:
1. Passages were numbered `[1]`, `[2]`; the model cited `[1], p. 12` and confused passage numbers
   with the papers' own reference markers. Now passages carry `[S1]`-style ids that the code maps
   back to `[filename, p. N]`.
2. The groundedness judge said NO to correct, verbatim answers. Including the question and
   asking "is the proposed answer supported?" fixed 6 of 6 hand-made test cases (3 correct,
   3 fabricated). These cases reused eval questions, so this is light tuning on the eval set.
3. The groundedness call re-read the whole context (about 37 s). Sharing the system message and
   context prefix with the answer call cut it to about 2 s.

## LangGraph agent vs single-pass pipeline

Reproduce with `python -m scripts.eval_guards --agent` (-> `storage/eval/agent.json`). Same
questions, corpus, model and threshold as above.

| Metric | Single-pass pipeline | Agent |
|---|---|---|
| Abstention rate on unanswerable questions | 100% (12/12) | 100% (12/12) |
| False abstention rate on answerable questions | 20.4% (10/49) | **14.3%** (7/49) |
| Answered questions with a valid citation | 79.5% (31/39) | **81.0%** (34/42) |
| Answered questions citing the expected page | 56.4% (22/39) | **57.1%** (24/42) |
| Abstain reasons, answerable | groundedness: 6, model said not found: 3, gate: 1 | groundedness: 2, model said not found: 4, gate: 1 |
| Latency per question (mean / median) | 27.3 s / 33.4 s | **18.9 s / 17.5 s** |

### What the numbers say

- **The agent answers more questions without giving up abstention.** It answered 5 questions the
  pipeline refused (att-2, bert-8, lora-5, colbert-3, sbert-1) and refused 2 the pipeline
  answered (dpr-3, t5-3). The groundedness check rejected only 2 answers instead of 6.
- **Likely reason: it generates only from chunks that pass grading** (top-5 filtered by the
  relevance threshold) instead of all 5. Fewer, more relevant passages give the 3B model a
  shorter prompt and fewer distractors, which helps both the answer and the judge. Query
  rewriting can also change which candidates are retrieved (they are still scored against the
  original question). This run doesn't separate the two effects; an ablation would.
- **It is also faster on average** despite extra calls (route, rewrite): shorter generation
  prompts outweigh two short LLM calls. Unanswerable questions are slower than in the pipeline
  (11-20 s instead of about 2 s) because the agent rewrites and retries twice before
  abstaining.
- **Routing**: all 61 eval questions were routed to `search_documents` (none misrouted to
  `list_documents` or `summarise_document`) after the router prompt was tightened (search as
  the explicit default tool, real filenames listed in the prompt).
- The same ambiguity failure remains: the Transformer warmup question is answered with T5's
  setting, cited to t5.pdf.
- Caveat: about 2 minutes of API tests ran on the same CPU during the agent run, which can only
  have made its latency slightly worse.
