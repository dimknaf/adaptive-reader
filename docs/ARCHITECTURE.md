# Architecture

The reader turns one long document into a short memory for one question, with every document token computed once.
Two things make that possible: the server's prefix cache, and decisions that are single tokens read from the model's
logits.

## The prefix cache

vLLM caches the key/value state of whole 16-token blocks, keyed by the chain of all blocks before them, so a cached
entry is always a prefix. A request reuses the longest chain of cached blocks, up to `((len - 1) // 16) * 16` tokens
(the last token is always computed). After a request, every full block of what the model computed is cached: the
prompt plus every sampled token except the last.

Three consequences shape the design:
- **Going back to any earlier point of a prompt is free**, because that prefix is cached.
- **Appending pays only for the new tokens**, provided the existing part ends on a block edge.
- **Removing something from the middle re-computes everything after it.** So the memory is a stack, and every segment
  of it is padded to a block edge.

`ledger.py` holds a local model of these rules. For every request it predicts the cached tokens, compares the
prediction with the server's own `cached_tokens` (vLLM's `--enable-prompt-tokens-details`), and records any difference
as a mismatch. A mismatch is never absorbed silently. Uncached tokens, cached tokens and output are totalled per
request type. The weighted cost used throughout is uncached input plus four times output.

## The request layout

```
[ fixed prefix L0 ][ run header ][ memory M ................ ][ piece ][ question ] -> one token
  instructions,      run id,       what was kept, as a stack    framed     "[a.p] | useful (yes or no):"
  worked examples    question,     of scopes                    and read
  (shared by every   outline                                    once
  run, cached once)
```

- **Fixed prefix (L0).**
  - The instructions (`GUIDE` in `reader.py`) and the worked examples, rendered as chat turns and padded to a block
    edge.
  - It is identical for every run, so after the first run it is always cached.
- **Run header.**
  - A random run id, the question, and an outline of the document's parts, padded to a block edge.
  - Because the run id starts a fresh block, each run is cold without resetting a shared cache, while L0 stays shared.
- **Memory M** (`memory.py`): one growing token-id prefix, shaped as a stack of scopes.
  - Each part of the document opens a scope with a short header.
  - A scope whose content was entirely removed is rolled back when it closes.
- **The piece** is appended right after M, with its anchor `[a.p]` (part a, piece p) and, when the piece does not start
  with its heading, the name of its section. The piece and M together are padded to a block edge: this is the
  **keep point**.
- **The questions** branch from the keep point, padded so that the assistant-turn header ends on a block edge. Each
  question is a short suffix such as `[a.p] | useful (yes or no):`, sent with `max_tokens=1`. Its answer is the
  probability of each option token at the next position, normalised over the options. Each option counts in both
  spellings (` yes` and ` Yes`).

## One piece

1. **Read** the piece (computed once), and ask `useful (yes or no)`.
   - If P(yes) is below the strictness, **remove** it: M stays as it was, and the next piece is appended after the old
     M. Nothing is recomputed.
2. **Ask `keep:`** (full or summary), only when the summary threshold is below 1.
   - If P(summary) is below the summary threshold, **keep in full**: M becomes the keep point. The piece was already
     computed in this prefix, so keeping it is free.
3. **Summary.** The model writes a one-line summary after `[a.p] (section) | summary:`, under a grammar that constrains
   only its format. Then it answers `[a.p] | summary enough (yes or no):` on the same branch, where the summary is
   already cached.
   - If P(yes) reaches the enough threshold, the summary is appended to M in place of the piece. Its tokens are
     generated once (output) and prefilled once (input), when the next piece is read.
   - Otherwise the piece is kept in full.

Code never judges the content of a piece or a summary. Every content decision is the model's answer, and code only
compares the probabilities with the three thresholds.

## The worked examples

The model is a 2B-parameter model, so the format of each decision matters more than its wording. The fixed prefix
therefore contains 17 worked examples in exactly the reader's format, built by `examples.py` from labels that two
independent judges agreed on. They use a separate document (Home Depot) that is never evaluated, under the anchor part
`x`, which no document part uses, and the answer instruction says they are not part of the document. The 2B model still
cites them occasionally (see FINDINGS §5). Because they sit in the cached prefix, they cost nothing after the first
run.

## The answer

After the last piece, the question is restated after M, followed by its own `Answer:` lead. The answer is generated
under a grammar that bars the `|` separator, so it cannot imitate the decision format. It is written only from the
fixed prefix and what was kept, never from the whole document.

## Documents and pieces

`corpus.py` turns a source file into pieces, through an adapter (`adapters/markdown.py`; `adapters/chatlog.py` is an
example for chat logs).
- **Scopes** come from heading depth alone.
- **Units** end at newline tokens.
- **Pieces** of 350–600 tokens are cut at the strongest structural boundary, with ties broken by the least padding.
- **Lossless:** the builder checks on every tokenizer call and every piece that nothing is lost or reordered.
- **Output:** `data/corpus/<bundle>/` holds `pieces.jsonl` (token ids, anchors, character spans, units and section path
  per piece), `tree.json` (the scope tree), `view/1.md` (the normalised text) and `view/1.anchored.md` (the same text
  with anchors).

## Traces and the replay page

Each run writes `results/<tag>_<bundle>/<question id>/trace.json`, recording every decision with its probabilities,
any summary, the tokens it cost and the memory size before and after, plus the answer. It also writes a readable
`trace.html` next to it.
- `viz.py` builds `viz/replay.html` from selected traces: a self-contained page that replays a run on the real
  document text, with the context bar, cost counters and the answer.
- `serve.py` serves the same page with a panel that starts a live run, one at a time, and streams its decisions to the
  page.
