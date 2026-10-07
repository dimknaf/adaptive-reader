"""Score reader traces against reference labels from two independent judges.

Per question:
  facts     a gold fact survives when one of its locations is still in memory at the end: the piece kept in full, or
            a summary containing all the quote's numbers. Coverage = surviving facts / facts, per judge (a fact both
            judges list counts for each).
  keeps     pieces kept that hold no fact of either judge (needless keeps), and the tokens kept.
  answer    the numbers of the judges' answer found in the reader's answer; for a question the text does not answer,
            whether the reader says so.
  cost      uncached input, output, weighted (uncached + 4 x output) per document token, ledger mismatches.

  python score.py <bundle> <tag>        (reads results/<tag>_<bundle>/*/trace.json and data/gold/<bundle>_{a,b}.json)
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NUM = re.compile(r"\d[\d,.]*\d|\d")
NOT_STATED = re.compile(r"not (stated|provided|mentioned|available|found|included|disclosed|given|specified)|"
                        r"no information|does not (state|provide|mention|include|contain)|cannot be determined", re.I)


def norm(s):
    return " ".join(s.split()).lower()


def numbers(s):
    return {n.rstrip(".,") for n in NUM.findall(s)}


def kept_text(trace, views):
    """anchor -> (form, text kept in memory) for everything still in memory at the end."""
    final = set(trace["trace"][-1]["memory"]) if trace["trace"] else set()
    out = {}
    for r in trace["trace"]:
        if r["anchor"] not in final:
            continue
        if r["decision"] == "full":
            c0, c1 = r["char"]
            out[r["anchor"]] = ("full", views[r["anchor"].split(".")[0]][c0:c1])
        elif r["decision"] == "summary":
            out[r["anchor"]] = ("summary", r["summary"])
    return out


def survives(loc, kept):
    if loc["anchor"] not in kept:
        return False
    form, text = kept[loc["anchor"]]
    if norm(loc["quote"]) in norm(text):
        return True
    return form == "summary" and numbers(loc["quote"]) and numbers(loc["quote"]) <= numbers(text)


def score_question(trace, golds, views):
    kept = kept_text(trace, views)
    qid = trace["question"]["id"]
    out = {"id": qid, "kept": sorted(kept), "kept_tokens": sum(r["n_tokens"] for r in trace["trace"]
                                                              if r["anchor"] in kept and r["decision"] == "full")}
    needed = set()
    for k, g in golds.items():
        a = next(x for x in g["answers"] if x["id"] == qid)
        facts = a["facts"]
        ok = [any(survives(l, kept) for l in f["locations"]) for f in facts]
        out[f"coverage_{k}"] = round(sum(ok) / len(facts), 3) if facts else None
        out[f"lost_{k}"] = [f["fact"] for f, s in zip(facts, ok) if not s]
        needed |= {l["anchor"] for f in facts for l in f["locations"]}
        out["answerable"] = a["answerable"]
        out[f"gold_answer_{k}"] = a["answer"]
    out["needless_keeps"] = sorted(set(kept) - needed)
    out["needed_pieces"] = sorted(needed)
    # bloat: the context kept against the smallest context the answer needs (a greedy cover of every gold fact by
    # pieces, fewest tokens per fact covered first)
    size = {r["anchor"]: r["n_tokens"] for r in trace["trace"]}
    facts = [{l["anchor"] for l in f["locations"]} for g in golds.values()
             for f in next(x for x in g["answers"] if x["id"] == qid)["facts"]]
    cover, left = [], [f for f in facts if f]
    while left:
        best = max({a for f in left for a in f}, key=lambda a: (sum(a in f for f in left) / size.get(a, 1), -size.get(a, 1)))
        cover.append(best)
        left = [f for f in left if best not in f]
    out["needed_tokens"] = sum(size.get(a, 0) for a in cover)
    out["kept_context"] = trace["memory_tokens"] - trace["root_tokens"]
    answer = trace["answer"]["text"]
    gold = golds["a"]
    ga = next(x for x in gold["answers"] if x["id"] == qid)
    if ga["answerable"]:
        want = numbers(ga["answer"])
        got = numbers(answer)
        out["answer_numbers"] = f"{len(want & got)}/{len(want)}"
        out["answer_numbers_missing"] = sorted(want - got)
    else:
        out["answer_says_not_stated"] = bool(NOT_STATED.search(answer))
    rd = trace["reading"]
    out["cost"] = {"uncached": rd["uncached"], "output": rd["output"], "weighted": rd["weighted"],
                   "weighted_per_doc_token": trace["weighted_per_doc_token"], "mismatches": len(trace["mismatches"])}
    out["decisions"] = {r["anchor"]: [r["decision"], r["useful"], r.get("relevance"), r.get("useful_logodds"),
                                      *(r.get("useful_orders") or [None, None])] for r in trace["trace"]}
    out["memory"] = {"root": trace["root_tokens"], "final": trace["memory_tokens"],
                     "max": max((r["m_after"] for r in trace["trace"]), default=trace["memory_tokens"])}
    out["answer"] = answer
    return out


def auroc(pos, neg):
    """Probability that a needed piece scores above an unneeded one (ties count half); None without both kinds."""
    if not pos or not neg:
        return None
    return round(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)), 3)


def separation(rows):
    """AUROC of each readout against the gold needed pieces, over all pieces of the answerable questions (check that
    the model separates the answers at all before tuning any threshold)."""
    out = {}
    for i, name in ((1, "useful"), (2, "relevance"), (3, "logodds"), (4, "yes_or_no"), (5, "no_or_yes")):
        pick = lambda need: [v[i] for r in rows if r["answerable"] for a, v in r["decisions"].items()
                             if (a in r["needed_pieces"]) == need and len(v) > i and v[i] is not None]
        out[name] = auroc(pick(True), pick(False))
    return out


def main():
    bundle, tag = sys.argv[1], sys.argv[2]
    golds = {k: json.load(open(os.path.join(HERE, "data", "gold", f"{bundle}_{k}.json"), encoding="utf-8")) for k in "ab"}
    d = os.path.join(HERE, "data", "corpus", bundle, "view")
    views = {f.split(".")[0]: open(os.path.join(d, f), encoding="utf-8").read() for f in os.listdir(d)
             if f.endswith(".md") and not f.endswith(".anchored.md")}
    rdir = os.path.join(HERE, "results", f"{tag}_{bundle}")
    rows = []
    for q in sorted(os.listdir(rdir)):
        path = os.path.join(rdir, q, "trace.json")
        if os.path.exists(path):
            rows.append(score_question(json.load(open(path, encoding="utf-8")), golds, views))
    for r in rows:
        dec = " ".join(f"{a}:{v[0][0]}({v[1]:.2f}" + (f"/{v[2]:.0f}" if v[2] is not None else "") + ")"
                       for a, v in r["decisions"].items())
        cov = f"coverage a {r['coverage_a']} b {r['coverage_b']}" if r["answerable"] else "not in text"
        ans = f"answer numbers {r['answer_numbers']}" if r["answerable"] else f"says not stated: {r['answer_says_not_stated']}"
        print(f"{r['id']}: {cov}; needed {r['needed_pieces']} kept {r['kept']} needless {r['needless_keeps']}; {ans}; "
              f"weighted/doc {r['cost']['weighted_per_doc_token']}, mismatches {r['cost']['mismatches']}; memory "
              f"{r['memory']['final'] - r['memory']['root']} kept tokens at the end\n    {dec}")
    sep = separation(rows)
    kept = sum(len(r["kept"]) for r in rows)
    needless = sum(len(r["needless_keeps"]) for r in rows)
    lost = sum(len(r[f"lost_{k}"]) for r in rows for k in "ab" if r["answerable"])
    ctx, need = sum(r["kept_context"] for r in rows), sum(r["needed_tokens"] for r in rows)
    print(f"== {bundle} {tag}: kept context {ctx} tokens vs needed {need} (bloat x{ctx / max(need, 1):.1f}); "
          f"kept {kept} pieces, {needless} needless; facts lost (both judges' lists) {lost}; AUROC {sep}")
    json.dump({"separation": sep, "questions": rows}, open(os.path.join(rdir, "scores.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
