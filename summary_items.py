"""Collect every summary a run kept, with its piece, task and the gold key points anchored to that piece, for two
independent summary judges; then fold their verdicts back into key-point coverage.

  python summary_items.py collect <tag> <bundle>...  -> results/<tag>_summaries.json (items for the judges)
  python summary_items.py score <tag> <bundle>...    -> key points kept, counting a summarised piece only for the key
                                                        points both judges say its summary preserves
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def gold(bundle):
    return [json.load(open(os.path.join(HERE, "data", "gold", f"{bundle}_rel_{k}.json"), encoding="utf-8")) for k in "ab"]


def collect(tag, bundles):
    items = []
    for b in bundles:
        g = gold(b)
        view = open(os.path.join(HERE, "data", "corpus", b, "view", "1.md"), encoding="utf-8").read()
        rdir = os.path.join(HERE, "results", f"{tag}_{b}")
        for q in sorted(d for d in os.listdir(rdir) if os.path.isdir(os.path.join(rdir, d))):
            t = json.load(open(os.path.join(rdir, q, "trace.json"), encoding="utf-8"))
            for r in t["trace"]:
                if r["decision"] != "summary":
                    continue
                pts = [p["point"] for j in g for a in j["answers"] if a["id"] == q for p in a["key_points"]
                       if r["anchor"] in p["anchors"]]
                items.append({"key": f"{tag}|{b}|{q}|{r['anchor']}", "task": t["question"]["question"],
                              "piece": view[r["char"][0]:r["char"][1]], "note": r["summary"], "key_points": pts})
    path = os.path.join(HERE, "results", f"{tag}_summaries.json")
    json.dump(items, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(items)} summaries -> {path}")


def score(tag, bundles):
    path = os.path.join(HERE, "results", f"{tag}_summary_verdicts.json")
    verdicts = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}   # no summaries: none needed
    kept_pts = {}                 # (bundle, task, point) covered
    for b in bundles:
        g = gold(b)
        rdir = os.path.join(HERE, "results", f"{tag}_{b}")
        for q in sorted(d for d in os.listdir(rdir) if os.path.isdir(os.path.join(rdir, d))):
            t = json.load(open(os.path.join(rdir, q, "trace.json"), encoding="utf-8"))
            full = {r["anchor"] for r in t["trace"] if r["decision"] == "full"}
            summ = {r["anchor"] for r in t["trace"] if r["decision"] == "summary"}
            for j, jg in enumerate(g):
                a = next(x for x in jg["answers"] if x["id"] == q)
                for p in a["key_points"]:
                    ok = bool(set(p["anchors"]) & full)
                    for anc in set(p["anchors"]) & summ:
                        v = verdicts.get(f"{tag}|{b}|{q}|{anc}", {})
                        ok = ok or p["point"] in v.get("preserved_by_both", [])
                    kept_pts[(b, q, j, p["point"])] = ok
    n = len(kept_pts)
    unfaithful = sum(1 for v in verdicts.values() if not v.get("faithful_both", True))
    print(f"{tag}: key points kept {sum(kept_pts.values())}/{n} ({sum(kept_pts.values()) / max(n, 1):.2f}); "
          f"unfaithful summaries {unfaithful}/{len(verdicts)}")


if __name__ == "__main__":
    {"collect": collect, "score": score}[sys.argv[1]](sys.argv[2], sys.argv[3:])
