# Findings

Every result below carries one of two labels:
- *(published run: `results/…`)*: the run files are in this repository, and the numbers reproduce offline (see
  `results/README.md`);
- *(development measurement)*: measured while the design was being developed. The raw runs are not included.

Common setup:
- **Model:** MiniCPM5-2B (INT4, OpenVINO build) on vLLM 0.26, with prefix caching in 16-token blocks.
- **Pieces:** documents are split into pieces of 350–600 tokens (about 500) at structural boundaries.
- **Documents:** excerpts of Analog Devices' FY2024 Form 10-K are the evaluated documents. Excerpts of Home Depot's
  FY2023 Form 10-K are used only to build worked examples and are never evaluated.
- **Cost:** weighted cost = (uncached input + 4 × output) / document tokens.

## 1. Cache mechanics

*(development measurement)* A mechanics probe on an evaluated document passed all 11 of its checks; the ones about
reading and keeping are below. In every request, the server's cached-token count equalled the local cache model's
prediction:

| Check | Uncached tokens |
|---|---|
| Reading a piece | exactly its own ids plus framing (489) |
| A second question on the same piece | only the question suffix (8) |
| Keeping a piece in full | free: the next read paid only its own ids (473) |
| Removing a piece | free (489, the next piece's own ids) |
| A summary | its question suffix (8) plus 40 output tokens; later the summary plus framing once (521 = 32 + own ids) |
| Keeping the head of a piece | one block (16) |
| Control: a keep without block padding | re-computes the unaligned tail (12) |
| Control: padding placed before the question header | leaves the memory 7 ids off the block edge (7) |
| Trimming a kept piece | re-computes exactly the survivors appended again (352 + own ids) |

- **No cache resets.** A unique run id placed right after the fixed prefix makes each run cold, while the fixed prefix
  (instructions and worked examples) stays shared across runs.
- **Per-request counts are reliable.** The server's aggregate cache counters cannot be attributed to one run when other
  requests use the server; the per-request cached-token counts stay exact.
- **The block edge must fall in the right place.** One block-sized mismatch (16 tokens) appeared when the end of the
  fixed prefix shared a block with the per-run header. Padding the fixed prefix to a block edge, so that the run id
  starts a fresh block, removed it.

*(published runs: `results/final_*`, `results/final_fr_*`)* 0 mismatches in all 16 runs.

## 2. Reading cost

- *(development measurement)* Full or remove at 3K and 5K tokens: 1.18–1.23 weighted per document token. At 20K tokens
  (41 pieces): 1.17–1.18.
- *(development measurement)* The 20K-token runs in detail:
  - **Uncached input:** 22.9–23.2K tokens, the document once plus about 3.1K of question and framing tokens
    (×1.15–1.17).
  - **Output:** 170–230 tokens, one per decision plus the final answer.
  - **Weighted cost, including the final answer:** 1.19–1.21 × the document.
  - **Sent vs computed:** on one question, the requests sent 1.99M prompt tokens, but the server computed only about
    23K of them. The KV cache in use stayed at 3.5–4.6% of the pool.
- *(published runs)* Full or remove: 1.12–1.17. With summaries on: 1.33–1.82 on the six tasks whose answer is in the
  text, and 1.12 and 1.17 on the two that keep nothing, as with full or remove. A summary costs its output (up to 80
  tokens, weighted ×4), and that cost is paid even when the model then judges the summary not enough.

*(development measurement)* Speed is reported, not counted. Reading slows linearly with the context already in the
prompt (the fixed prefix plus the memory). A least-squares fit over 72 reads:

    seconds ≈ 0.23 + 3.8 ms × new tokens + 1.9 ms × new tokens × (context / 1K)

| Context | Read of ~470 new tokens | Best tokens/s |
|---|---|---|
| ~0 | 3.2 s | 264 |
| ~2K | 3.9 s | 132 |
| ~5K | 6.5 s | 79 |
| ~10K | 12.5 s | 41 |
| ~20K | 19 s | 25 |

The times were measured under concurrent load and are only indicative. The growth with context shows in the minimum
times as well.

## 3. Decision quality

### Worked examples in the cached prefix
*(development measurement)* These runs are full or remove, over 4 lookup questions per document. Six worked examples
(3 useful, 3 not, built from agreed labels on Home Depot text) were added to the fixed prefix:

| | Needless keeps (3K / 5K) | Facts lost (3K / 5K) | Weighted cost per document token |
|---|---|---|---|
| Without worked examples | 15 of 23 kept / 26 of 32 kept | 0 / 0 | 1.19–1.23 |
| With worked examples | 3 of 10 kept / 3 of 8 kept | 2 / 0 | 1.18–1.22 |

The examples sit in the cached prefix, so they add no cost. P(useful) moved from a strong "yes" bias to needed pieces
at 0.52–0.76 and the others at 0.23–0.50.

### Separation
- *(development measurement)* AUROC of P(useful) against the needed pieces:
  - 0.83–0.91 on the 3K/5K lookup questions;
  - 0.73 at 20K tokens.
  - A 0–100 relevance score asked alongside separated less well on the small documents (0.56–0.80) and about as well
    at 20K (0.739).
- *(published runs)* Relevance tasks, full or remove: 0.818 (MD&A excerpt) and 0.752 (statements excerpt).

### First-option bias
*(development measurement)* Each answer order was logged separately (306 piece decisions against the reference labels):
- **"(yes or no)":** needed pieces have median P(yes) 0.96, the others 0.47. AUROC 0.861.
- **"(no or yes)":** 0.10 against 0.02. The model almost always answers "no".
- **Averaging both orders:** needed pieces land around 0.53, and the ranking gets worse (AUROC 0.781). A log-odds sum
  does slightly better (0.806).

The reader therefore asks once, as "(yes or no)", and compares P(yes) with a strictness threshold.

### The decision limit
*(development measurement)* These figures are offline, over 992 logged piece decisions (135 needed, 857 not).

Recall and false keeps by decision rule:

| Rule | Needed pieces kept | Other pieces kept |
|---|---|---|
| Average of both orders ≥ 0.5 | 0.87 | 0.32 |
| Log-odds sum > 0 | 0.87 | 0.32 |
| "(yes or no)" alone ≥ 0.5 | 1.00 | 0.68 |
| "(no or yes)" alone ≥ 0.5 | 0.20 | 0.03 |
| Average, and relevance ≥ 50 | 0.80 | 0.28 |
| Average, and relevance ≥ 70 | 0.59 | 0.21 |

The single "(yes or no)" question at each strictness:

| Strictness | Needed pieces kept | Other pieces kept |
|---|---|---|
| 0.5 | 1.00 | 0.68 |
| 0.7 | 0.95 | 0.51 |
| 0.8 | 0.90 | 0.37 |
| 0.9 | 0.70 | 0.23 |
| 0.95 | 0.51 | 0.11 |

No reading of the logged answers gives both high recall and few needless keeps. Separation stays around AUROC 0.8
whatever the wording or the examples. When only a few percent of a long document is relevant, the memory fills with
needless pieces. The mechanics hold; the bottleneck is the per-piece decision quality of a 2B INT4 model in this format.

### Variants not adopted
*(development measurement)*
- **The question word** ("useful", "helps", "relevant"), with 12 worked examples, on the relevance tasks:

  | Word | Context kept vs. essential (5K MD&A / 5K statements) | Essential pieces missed | Needless keeps |
  |---|---|---|---|
  | useful | ×2.4 / ×3.0 | 0 / 0 | 10 / 7 |
  | helps | ×2.0 / ×2.7 | 0 / 2 | 5 / 8 |
  | relevant | ×2.3 / ×3.3 | 0 / 0 | 8 / 9 |

  Relevance-style worked examples with hard negatives gave mixed results, within the noise of single runs on 9–10
  pieces.
- **Restating the question after every piece**, to keep the question next to each decision at any memory size.
  - On the 3K / 5K / 20K documents: AUROC 0.75 / 0.873 / 0.697 (against 0.863 / 0.833 / 0.73 without), and facts lost
    5 / 3 / 6 (against 2 / 0 / 14).
  - Mixed: fewer facts lost at 20K (6 against 14) but more on the small documents (5 and 3, against 2 and 0); AUROC rose
    at 5K and fell at 3K and 20K; and it cost 6–11% more per document token. Not adopted.
- **Drift at 20K tokens.** In one 20K question, once the memory passed about 10K tokens, nearly every piece scored
  0.55–0.65. A question far from its answer drifts towards the middle.

## 4. Partial keeping

*(development measurement)*
- **Per-line selection** (a "needed?" question per line of a kept piece) failed.
  - The per-line answers barely separated (nearly all 0.50–0.67). The reader kept headings and table header rows,
    dropped key value rows, and removed pieces judged useful when no line reached 0.5.
  - Facts lost were 15 and 10, against 2 and 0 for full or remove.
  - The cost rose to 1.3–2.0 per document token, because two questions per line cost more than the piece itself.
- **Free-form summaries** failed in their first design.
  - Within one question, summaries of different pieces were copied word for word from the first summary in memory.
  - Some summaries answered the question instead of summarising the piece.
  - Facts lost were 31 and 16, at 1.23–1.93 per document token.
- **Summaries checked by code** (every number must appear in the piece, and no copying of an earlier summary) kept
  34 of 65 key points, against 55 for full or remove in the same development run. 7 summaries were rejected, mostly
  over rounded figures. The design keeps every content judgement with the model, so this check was replaced by the
  model's own "summary enough" answer.
- **Summaries judged by the model, summary threshold 0.5:**
  - P(summary) was 0.946–0.988 on every kept piece, so a summary was written for every useful piece.
  - P(summary enough) spread from 0.44 to 0.87. 5 summaries were kept, and the rest of those pieces were kept in full.
  - **All 5 kept summaries were unfaithful, per both judges.** One credited a fact to the wrong line item, one invented
    a causal link, and three were copies of one summary that reversed three directions.
  - The model had rated them all "enough" (0.80–0.87).
- **At summary threshold 0.985:** no piece of the statements excerpt reached the threshold. On the MD&A excerpt the
  model judged 9 of 11 summaries not enough, and those pieces were kept in full.

*(published runs: `results/final_mini5k`, `results/final_mini5s`)* Summaries on (summary threshold 0.98, enough 0.7):
- With the shipped worked examples, P(summary) was 0.978–0.996, so almost every kept piece was summarised. 23 summaries
  were kept.
- **12 of the 23 were unfaithful.** The two judges agreed on all 23.
- **The main failure:** a summary repeats facts already in memory instead of summarising its own piece. On one task,
  five summaries carry the same income-statement figures and call cost of sales and operating expenses "higher" when
  both fell. The model had rated them "enough" (0.74–0.83).
- **Summaries are capped at 80 tokens.** 19 of the 29 summaries written hit the cap (14 of the 23 kept), and some end
  mid-sentence.
- Key points kept: 40 of 65, against 56 of 65 for full or remove *(published runs: `results/final_fr_*`)*.

The model's "summary enough" answer judges whether a summary sounds complete, not whether it is true to the piece.

## 5. The answer stage

- *(development measurement)* 14 answers to lookup questions, full or remove, judged by two independent LLM judges
  against the reference answers: 9 correct, 2 partial, 3 wrong. The judges agreed on 13 of 14. **Every failure came
  from a needed piece removed during reading, none from the answer stage.**
- *(published runs)* On the relevance tasks the answer stage is weak. The model often says "not stated" even when the
  kept pieces contain the answer, and occasionally refers to the worked examples' company.
- **Format fixes:**
  - The first answers copied the instructions' `[a.p] | summary:` format. The answer now has its own `Answer:` lead,
    under a grammar that bars the `|` separator.
  - One answer cited a worked example's facts as if they came from the document. The answer instruction now says the
    worked examples are not part of it.

## 6. Evaluation method

- **Lookup questions** (3K, 5K and 20K documents; 4, 4 and 6 questions): a lookup, a change and its driver, a list, and
  one question not answered in the text; the 20K document adds a cross-section question and an accounting-policy
  question. Two independent LLM judges listed every fact with all its locations. They wrote equivalent answers and gave
  the same needed pieces, except for one piece in one question.
- **Relevance tasks** (the 5K MD&A excerpt and the 5K statements excerpt; 4 tasks each): explain, gather, assess, and
  one not covered.
  - Two independent LLM judges labelled every piece essential, supporting or not relevant, and wrote the key points with
    their anchors. They agreed on 7–10 of the 9–10 pieces in each task.
  - The 65 key points are the two judges' key points counted together.
- **Scoring** (`score_rel.py`): recall of essential pieces, key points kept, context kept against the essential text,
  needless keeps (both judges: not relevant), and AUROC.
- **Summaries:** two independent LLM judges reviewed every kept summary for faithfulness and for the key points it
  preserves (`summary_items.py`). A key point counts through a summary only when both judges say it is preserved.
- **Worked examples** (`data/examples/reader_examples.json`, 17 examples, Home Depot text): 8 pieces not useful, 6
  useful with a summary judged enough, and 3 useful with a summary judged not enough. They are built from the agreed
  labels by `examples.py`.
- **None of the labels has been verified by a human.**
