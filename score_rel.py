"""Score reader traces on relevance tasks against reference labels from two independent judges (per-piece labels + key
points).

Per task:
  essential   pieces either judge labels essential; recall = kept / essential (strict: pieces both judges call essential)
  key points  a judge's key point survives when one of its anchors is kept; coverage per judge
  bloat       the context kept against the tokens of the essential pieces (either judge)
  needless    kept pieces both judges call not_relevant
  not covered tasks: the context kept (ideally nothing)
Over all tasks: AUROC of the logged readouts (P(useful), log-odds, each order) against essential.

  python score_rel.py <bundle> <tag>     (results/<tag>_<bundle>/*/trace.json, data/gold/<bundle>_rel_{a,b}.json)
"""
import json
import os
import sys

from score import auroc

HERE = os.path.dirname(os.path.abspath(__file__))


def score_task(trace, golds):
    tid = trace["question"]["id"]
    kept = set(trace["trace"][-1]["memory"]) if trace["trace"] else set()
    size = {r["anchor"]: r["n_tokens"] for r in trace["trace"]}
    labels = [{p["anchor"]: p["label"] for p in next(x for x in g["answers"] if x["id"] == tid)["pieces"]}
              for g in golds.values()]
    answers = [next(x for x in g["answers"] if x["id"] == tid) for g in golds.values()]
    ess_any = {a for lab in labels for a, v in lab.items() if v == "essential"}
    ess_both = {a for a in ess_any if all(lab.get(a) == "essential" for lab in labels)}
    not_rel = {a for a in size if all(lab.get(a, "not_relevant") == "not_relevant" for lab in labels)}
    out = {"id": tid, "covered": answers[0]["covered"], "kept": sorted(kept, key=lambda a: int(a.split(".")[1])),
           "essential": sorted(ess_any, key=lambda a: int(a.split(".")[1])),
           "recall_any": round(len(kept & ess_any) / len(ess_any), 3) if ess_any else None,
           "recall_both": round(len(kept & ess_both) / len(ess_both), 3) if ess_both else None,
           "missed": sorted(ess_both - kept), "needless": sorted(kept & not_rel),
           "kept_context": trace["memory_tokens"] - trace["root_tokens"],
           "essential_tokens": sum(size.get(a, 0) for a in ess_any)}
    for k, a in zip("ab", answers):
        pts = a["key_points"]
        out[f"points_{k}"] = round(sum(bool(set(p["anchors"]) & kept) for p in pts) / len(pts), 3) if pts else None
    out["readouts"] = {r["anchor"]: [r["anchor"] in ess_any, r.get("useful"), r.get("useful_logodds"),
                                     *(r.get("useful_orders") or [None, None])] for r in trace["trace"]}
    out["answer"] = trace["answer"]["text"]
    rd = trace["reading"]
    out["cost"] = {"weighted_per_doc_token": trace["weighted_per_doc_token"], "mismatches": len(trace["mismatches"]),
                   "uncached": rd["uncached"], "output": rd["output"]}
    return out


def main():
    bundle, tag = sys.argv[1], sys.argv[2]
    golds = {k: json.load(open(os.path.join(HERE, "data", "gold", f"{bundle}_rel_{k}.json"), encoding="utf-8"))
             for k in "ab"}
    rdir = os.path.join(HERE, "results", f"{tag}_{bundle}")
    rows = [score_task(json.load(open(os.path.join(rdir, d, "trace.json"), encoding="utf-8")), golds)
            for d in sorted(os.listdir(rdir)) if os.path.exists(os.path.join(rdir, d, "trace.json"))]
    for r in rows:
        head = (f"recall {r['recall_any']} (strict {r['recall_both']}), key points a {r['points_a']} b {r['points_b']}"
                if r["covered"] else "not covered")
        print(f"{r['id']}: {head}; kept {r['kept']} essential {r['essential']} missed {r['missed']} needless "
              f"{r['needless']}; context {r['kept_context']} vs essential {r['essential_tokens']}; "
              f"weighted/doc {r['cost']['weighted_per_doc_token']}, mismatches {r['cost']['mismatches']}")
    sep = {}
    for i, name in ((1, "useful"), (2, "logodds"), (3, "yes_or_no"), (4, "no_or_yes")):
        vals = [(v[0], v[i]) for r in rows if r["covered"] for v in r["readouts"].values() if v[i] is not None]
        sep[name] = auroc([x for e, x in vals if e], [x for e, x in vals if not e])
    ctx = sum(r["kept_context"] for r in rows)
    ess = sum(r["essential_tokens"] for r in rows)
    missed = sum(len(r["missed"]) for r in rows)
    needless = sum(len(r["needless"]) for r in rows)
    print(f"== {bundle} {tag}: context {ctx} vs essential {ess} (x{ctx / max(ess, 1):.1f}); missed essential "
          f"(both judges) {missed}; needless (both judges not relevant) {needless}; AUROC {sep}")
    json.dump({"separation": sep, "tasks": rows}, open(os.path.join(rdir, "scores_rel.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
