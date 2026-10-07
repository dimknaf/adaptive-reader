# Published runs

All runs: MiniCPM5-2B (INT4, OpenVINO build) on vLLM 0.26, the 8 relevance tasks over `mini5k` and `mini5s`, and
strictness 0.8.

| Folder | Setting |
|---|---|
| `final_fr_mini5k/`, `final_fr_mini5s/` | Full or remove: summary threshold 1 |
| `final_mini5k/`, `final_mini5s/` | Summaries on: summary threshold 0.98, enough threshold 0.7 |

Each task folder holds:
- `trace.json`: every decision with its probabilities, any summary, the tokens each step cost, the memory size before
  and after, the answer, and the cache-ledger result;
- `trace.html`: the same trace as a readable page.

`scores_rel.json` in each run folder is the output of `score_rel.py`.

The summary judgements belong to the summaries-on runs:
- `final_summaries.json`: every kept summary, with its piece, task and key points.
- `final_judge_1.json`, `final_judge_2.json`: two independent LLM judges' verdicts on faithfulness and preserved key
  points.
- `final_summary_verdicts.json`: both verdicts merged. A key point counts through a summary only when both judges say
  the summary preserves it.

Reproduce the published numbers:

```
python score_rel.py mini5k final_fr && python score_rel.py mini5s final_fr
python score_rel.py mini5k final && python score_rel.py mini5s final
python summary_items.py score final_fr mini5k mini5s    # key points kept: 56/65
python summary_items.py score final mini5k mini5s       # key points kept: 40/65; unfaithful summaries: 12/23
python viz.py                                           # rebuilds viz/replay.html from these runs
```

Runs made with `serve.py` are saved under `results/web_<bundle>/`, which git ignores.
