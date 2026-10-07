# Data

## Document excerpts (not covered by the Apache License)

The bundles in `data/corpus/` contain verbatim excerpts of two public SEC filings:

| Bundle | Source | Lines | Tokens | Pieces | Role |
|---|---|---|---|---|---|
| `mini3k` | Analog Devices, Inc., Form 10-K, fiscal year 2024 | 1225–1367 | 2,853 | 6 | evaluated: lookup questions |
| `mini5k` | Analog Devices, Inc., Form 10-K, fiscal year 2024 | 1225–1451 | 4,490 | 9 | evaluated: lookup questions and relevance tasks |
| `mini5s` | Analog Devices, Inc., Form 10-K, fiscal year 2024 | 1673–1900 | 4,477 | 10 | evaluated: relevance tasks |
| `mini20k` | Analog Devices, Inc., Form 10-K, fiscal year 2024 | 1225–2065 | 19,845 | 41 | evaluated: lookup questions |
| `examples` | The Home Depot, Inc., Form 10-K, fiscal year 2023 | 957–1106 | 3,145 | 6 | worked examples only, never evaluated |
| `examples2` | The Home Depot, Inc., Form 10-K, fiscal year 2023 | 1288–1558 | 4,619 | 10 | worked examples only, never evaluated |

The line ranges refer to a Markdown rendering of each filing.
- **Licensing:** the excerpts are reproduced for research and reproducibility. The copyright in the filings remains
  with the respective companies, and the excerpts, including the passages quoted from them in `results/` and
  `viz/replay.html`, are **not** licensed under this project's Apache License. Rights holders who want an excerpt
  removed can open an issue.
- **Rebuilding:** the full source files are not redistributed. `data/bundles.json` records the expected source file for
  each bundle (`sources/<id>.md`, not included), its SHA-256 and the line range used. Given an identical file and the
  model server (for its tokenizer), `python corpus.py build <bundle>` rebuilds that bundle. The bundles shipped here
  are ready to use without it.

Each bundle folder holds:
- `pieces.jsonl`: one piece per line, with its token ids, anchor `[a.p]`, character span, units and section path;
- `tree.json`: the scope tree;
- `stats.json`: sizes and settings;
- `view/1.md`: the normalised text;
- `view/1.anchored.md`: the same text with anchors.

## Questions and reference labels

- `questions/<bundle>.json`: lookup questions (a value, a change and its driver, a list, and one not answered in the
  text; `mini20k` adds a cross-section question and an accounting-policy question).
- `questions/<bundle>_rel.json`: relevance tasks (explain, gather, assess, and one not covered).
- `gold/`: **reference labels written by two independent LLM judges, `a` and `b`, and not verified by a human.**
  - `<bundle>_{a,b}.json`: for each lookup question, the answer and every fact with all its locations.
  - `<bundle>_rel_{a,b}.json`: for each relevance task, a label per piece (essential, supporting or not relevant) and
    the key points with their anchors.
  - `examples_rel_labels.json`, `examples2_rel_labels.json`: both judges' relevance labels on the worked-example
    documents. `examples_labels.json` and `examples2_labels.json` are the earlier lookup-style labels for the same
    documents.
  - `examples_rel_notes.json`: summaries of worked-example pieces, written and then independently verified.
    `examples_rel_bad_notes.json` holds summaries verified as not enough. They teach the "summary enough" answer.

## Worked examples

`examples/reader_examples.json` holds the 17 worked examples placed in the reader's cached prefix: 8 pieces not useful,
6 useful with a summary judged enough, and 3 useful with a summary judged not enough. To rebuild it:

```
python examples.py examples=data/gold/examples_rel_labels.json examples2=data/gold/examples2_rel_labels.json \
  --n 8 --yes-share 0.375 --hard --notes data/gold/examples_rel_notes.json \
  --bad-notes data/gold/examples_rel_bad_notes.json --out data/examples/reader_examples.json
```
