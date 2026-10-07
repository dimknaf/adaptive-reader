# Contributing

Thank you for your interest. This is a research prototype, and issues are welcome:
- bug reports;
- questions;
- measurements on other models, documents or servers.

**Pull requests:** please open an issue first, and send a pull request only after agreeing the change there. This
keeps the design coherent and the measurements comparable.

## Setup

Python 3.11 or newer; the standard library only, so there is nothing to install. The live parts need a model server;
see the README's Requirements.

## Tests

```
python -m unittest discover tests            # offline: the cache model, the memory stack and the corpus builder
CPM_LIVE=1 python -m unittest discover tests # also the live checks against the model server
```

The scripts that measure decisions on the model server are not tests: their results depend on the model, the build
and the server. Report such results with the model, its quantisation, the server version and the thresholds.

## Working principles

- **Read once.** No change may make the reader re-read or re-compute document text: keeping is a prefix, removing is a
  rollback.
- **No size rules.** The memory holds what the decisions keep. Do not add budgets, watermarks or size-based trimming.
- **The model judges content.** In the reader, code compares the model's probabilities with thresholds and handles
  tokens and the cache. It never judges content itself (no keyword or regex checks of meaning), and regular expressions
  belong only in generation grammars that constrain format. The evaluation scripts match the reader's output against
  the reference labels; that is scoring, not reading.
- **Measure before changing behaviour.** Start with the small documents, and change one thing at a time. A change must
  clearly help on the reference labels before it is adopted.
- **Every request is checked.** Keep the cache ledger at 0 mismatches.

## Licence of contributions

Under section 5 of the Apache License 2.0, any contribution you submit for inclusion is licensed under the same
licence, unless you state otherwise.
