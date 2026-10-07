# adaptive-reader

A small local language model reads a long document once, piece by piece, for a question known in advance, and keeps
only what the question needs. Every decision is a single-token answer read from the model's logits, and the server's
prefix cache makes both keeping and removing a piece free. Each document token is computed once, and the context grows
only by what the model chooses to keep.

**Status: research prototype.** It has been measured on one model (MiniCPM5-2B, INT4) and on small excerpts of one
annual report (Analog Devices' FY2024 Form 10-K), with reference labels from LLM judges. Treat the numbers as
indicative, not as benchmarks. The results include the negative ones.

## How it works

```
cached once for every run   fixed prefix: instructions + worked examples
per run                     question + document outline
grows only by what is kept  memory M
read once                   next piece (~500 tokens), appended right after M
                              │
                              ├─ "[a.p] | useful (yes or no):"  P(yes) < strictness ──▶ remove: M rolls back (free)
                              │
                              └─ "[a.p] | keep:"  full or summary
                                   ├─ full ──▶ M continues through the piece (already computed: free)
                                   └─ summary ──▶ the model writes a one-line summary, then
                                                  "[a.p] | summary enough (yes or no):"
                                                    yes ──▶ the summary replaces the piece
                                                    no  ──▶ the piece is kept in full
```

Invariants:
- **Read once.** Keeping a piece means continuing from the prefix that already contains it; removing it means rolling
  back to the prefix before it. Neither recomputes anything.
- **No size rules.** The memory holds whatever the decisions keep. Its size is measured, never budgeted or trimmed.
- **The model judges content; code does arithmetic.** Code only compares the model's probabilities with thresholds.
  Regular expressions are used only as generation grammars that constrain the output format.
- **Checked cost.** A local model of the server's prefix cache predicts the cached tokens of every request and checks
  the prediction against the server's own count.

The only controls are three thresholds at the top of `reader.py`:

| Threshold | Default | Effect |
|---|---|---|
| `STRICTNESS` | 0.7 | Keep a piece when P(useful) reaches it. Higher keeps less. |
| `SUMMARY_THRESHOLD` | 1 | Summarise when P(summary) reaches it. 1 turns summaries off (full or remove). |
| `ENOUGH_THRESHOLD` | 0.7 | Keep a summary when the model's P(summary enough) reaches it; otherwise keep the piece in full. |

The single-token decision technique is inspired by [jevfire](https://github.com/kikoncuo/jevfire).
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes the request layout and the cache model.

## Headline result

Eight relevance tasks ("explain", "gather", "assess" and one whose answer is not in the text) over two excerpts of
about 4.5K tokens each from Analog Devices' FY2024 annual report on Form 10-K: part of the MD&A section, and the
financial statements. Strictness 0.8.

| Setting | Key points kept | Unfaithful summaries | Context kept vs. essential text | Weighted cost per document token |
|---|---|---|---|---|
| **Full or remove** (summary threshold 1) | **56 / 65** | – (no summaries) | ×1.5 / ×1.4 | **1.12–1.17** |
| Summaries on (summary threshold 0.98, enough 0.7) | 40 / 65 | 12 of 23 | ×0.8 / ×0.7 | 1.12–1.82 |

Caveats:
- **n = 8 tasks, single runs, one 2B INT4 model.**
- **LLM judges.** Two independent LLM judges wrote the reference key points and judged every kept summary; their
  labels are not human-verified. A key point counts as kept when its piece was kept in full, or when both judges say a
  kept summary preserves it.
- **Measured at strictness 0.8.** The demo default of 0.7 has not been measured.
- **The answer stage is the weakest step.** Writing the final answer from what was kept, the model often says
  "not stated" even when the kept pieces contain the answer.

Weighted cost = (uncached input + 4 × output) / document tokens, for reading the document; writing the final answer
adds a little more (the replay page's per-document-token figure includes it). A value of 1.0 would mean that every
document token is computed once and nothing else is computed or generated. With summaries on, the six tasks whose
answer is in the text cost 1.33–1.82, and the two that keep nothing cost 1.12 and 1.17, as with full or remove.

## What was measured

Details, with provenance for each result, are in [docs/FINDINGS.md](docs/FINDINGS.md).
- **The cache mechanics hold.** Every request's cached-token count matched the local cache model (0 mismatches in every
  published run), and reading costs about 1.1–1.2 weighted per document token at 3K, 5K and 20K tokens.
- **Decisions are the bottleneck.** P(useful) separates needed from unneeded pieces with an AUROC of about 0.75–0.9 on
  small documents and 0.73 at 20K tokens. No threshold gives both high recall and few needless keeps.
- **Worked examples in the cached prefix fix the format.** They cut needless keeps sharply at no extra token cost.
- **Yes/no answers have a strong first-option bias.** Asking "(yes or no)" in one order, with a strictness threshold,
  ranks better than averaging both orders.
- **Partial keeping did not work on this model.** Per-line selection did not separate lines. Free-form summaries copied
  earlier summaries or answered the question instead of summarising the piece. The model's own "summary enough"
  check judges whether a summary sounds complete, not whether it is true to the piece.

## Requirements

- Python 3.11 or newer. Standard library only; no packages to install.
- To run the reader: a [vLLM](https://github.com/vllm-project/vllm) 0.26 server with an OpenAI-compatible API, serving
  MiniCPM5-2B, with prefix caching (16-token blocks) and `--enable-prompt-tokens-details`, which supplies the
  per-request cached-token counts the cost check compares against. `VLLM_SERVER_DEV_MODE=1` is needed only for the
  cache-reset endpoint used in cold-cost measurements. The address comes from `CPM_URL` (default
  `http://localhost:8000`) and the served model name from `CPM_MODEL` (default `MiniCPM5-2B`). The published runs used
  an INT4 OpenVINO build; other builds or quantisations may make different decisions.
- To view the recorded runs: nothing. `viz/replay.html` is a self-contained page.

## Quickstart

Watch the recorded runs, with no server needed: open `viz/replay.html` in a browser. `?ex=N&speed=K&autoplay=1`
selects an example.

Ask your own question (needs the model server):

```
python serve.py                      # then open http://localhost:8765
```

Pick a document, type a question, set the thresholds and watch the model read it live. Each run is saved under
`results/web_<bundle>/`.

From the command line:

```
python reader.py mini5k data/questions/mini5k_rel.json --ids mini5k-r1 --tag mine
python score_rel.py mini5k mine
```

Reproduce the published numbers offline:

```
python score_rel.py mini5k final_fr && python score_rel.py mini5s final_fr
python summary_items.py score final_fr mini5k mini5s     # key points kept: 56/65
python summary_items.py score final mini5k mini5s        # key points kept: 40/65; unfaithful summaries: 12/23
python viz.py                                            # rebuilds viz/replay.html
```

## Tests

```
python -m unittest discover tests            # offline
CPM_LIVE=1 python -m unittest discover tests # also the live checks against the model server
```

## Cost

Uncached input is what the server computes and output is what it generates; cached input is reused for free. Cost
counts uncached input plus four times output, a common price ratio for hosted models. Speed is reported but never
counted. The fixed prefix (instructions and worked examples) is computed once for every run, the document is computed
once per run, and each decision adds a few uncached tokens and one output token.

## Repository layout

| Path | What it is |
|---|---|
| `reader.py` | The reader: the per-piece loop, its three thresholds, the answer stage, the per-run trace pages |
| `chat.py` | Token-level requests: chat-turn ids, block padding, single-token answers, grammar-constrained generation |
| `cpm.py` | The model-server client: tokenizer calls, completions, prefix-cache metrics |
| `ledger.py` | A local model of the server's prefix cache that checks every request |
| `memory.py` | The memory stack: keep = prefix, remove = roll back, summaries appended |
| `corpus.py`, `adapters/` | Splits documents into ~500-token pieces at structural boundaries, with anchors `[a.p]` |
| `examples.py` | Builds the worked examples for the cached prefix from agreed labels |
| `score_rel.py`, `score.py`, `summary_items.py` | Scoring against the reference labels |
| `serve.py` | The local live demo |
| `viz.py`, `viz/` | The replay page |
| `data/` | Document excerpts (not Apache-licensed), questions, reference labels, worked examples |
| `results/` | The published runs |
| `docs/` | `ARCHITECTURE.md` and `FINDINGS.md` |

## Limitations and known issues

- Decision quality is the bottleneck. When only a few percent of a long document matters, the memory fills with
  needless pieces.
- The 2B model's summaries are often unfaithful, and its own "summary enough" check does not catch it.
- Summaries are capped at 80 tokens. In the published runs 19 of the 29 summaries written hit the cap (14 of the 23
  kept), and some end mid-sentence.
- The worked examples contain no "full" decisions, so P(summary) is skewed high (0.978–0.996). A summaries-on threshold
  therefore has to sit close to 1.
- The answer stage often says "not stated" even when the kept pieces hold the answer.
- The samples are small and every result is a single run.
- `serve.py` runs one reader at a time and is a local tool only (see [SECURITY.md](SECURITY.md)).

## Contributing and licence

Contributions: [CONTRIBUTING.md](CONTRIBUTING.md). Security: [SECURITY.md](SECURITY.md).

The code is licensed under the Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The document excerpts
in `data/`, and the passages quoted from them in `results/` and `viz/replay.html`, are not covered by that licence; see
[data/README.md](data/README.md).
