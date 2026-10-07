"""Build the replay page: real reader traces, animated step by step (viz/replay.html, self-contained).

Each example is one run on one question: every piece with its text, P(useful), the decision (and the summary, if any),
the tokens it cost (uncached input, cached input, output), the memory before and after, whether gold says it is
needed, and the final answer. The page replays it: pieces stream in, the context grows only by what is kept, and the
counters show what was computed (uncached), what was reused (cached) and what was generated (output).

  python viz.py            -> viz/replay.html
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# (label, bundle, tag, question id, gold kind)
EXAMPLES = [
    ("Why did net income halve? (full or remove)", "mini5k", "final_fr", "mini5k-r1", "rel"),
    ("Liquidity and cash (full or remove)", "mini5k", "final_fr", "mini5k-r3", "rel"),
    ("Debt and borrowing (full or remove)", "mini5s", "final_fr", "mini5s-r2", "rel"),
    ("Why did net income halve? (summaries on)", "mini5k", "final", "mini5k-r1", "rel"),
    ("Liquidity and cash (summaries on)", "mini5k", "final", "mini5k-r3", "rel"),
    ("Debt and borrowing (summaries on)", "mini5s", "final", "mini5s-r2", "rel"),
    ("Not in the text: Q1 outlook (keeps nothing)", "mini5k", "final_fr", "mini5k-r4", "rel"),
]


def gold_pieces(bundle, qid, kind):
    out = set()
    for k in "ab":
        name = f"{bundle}_rel_{k}.json" if kind == "rel" else f"{bundle}_{k}.json"
        for a in json.load(open(os.path.join(HERE, "data", "gold", name), encoding="utf-8"))["answers"]:
            if a["id"] != qid:
                continue
            if kind == "rel":
                out |= {p["anchor"] for p in a["pieces"] if p["label"] == "essential"}
            else:
                out |= {l["anchor"] for f in a["facts"] for l in f["locations"]}
    return out


def gold_answer(bundle, qid, kind):
    name = f"{bundle}_rel_a.json" if kind == "rel" else f"{bundle}_a.json"
    a = next(x for x in json.load(open(os.path.join(HERE, "data", "gold", name), encoding="utf-8"))["answers"]
             if x["id"] == qid)
    if kind == "rel":
        return " ".join(p["point"] for p in a["key_points"][:3]) or "Not covered in the excerpt."
    return a["answer"]


def piece(p, view):
    """A piece as the page shows it before any decision: anchor, heading, size, text."""
    c0, c1 = p["char"]
    return {"a": p["anchor"], "h": (p.get("path") or [""])[-1][:70], "n": p["n_tokens"], "t": view[c0:c1].strip("\n"),
            "g": False}


def decision(r):
    """What the reader decided on a piece and what it cost (one row of its trace)."""
    return {"d": r["decision"], "p": r["useful"], "unc": r["cost"]["uncached"], "cac": r["cost"]["cached"],
            "out": r["cost"]["output"], "m0": r["m_before"], "m1": r["m_after"], "note": r.get("summary") or "",
            "ps": r.get("p_summary"), "pe": r.get("p_enough")}


def settings(t):
    """The thresholds a run used, as the page labels them."""
    su = t.get("summary_threshold")
    return {"ask": t.get("ask") or "useful", "strictness": t.get("strictness", 0.5), "summary_threshold": su,
            "enough_threshold": t.get("enough_threshold"),
            "forms": list(t.get("forms") or ["full"]) if (su or 0) < 1 else ["full"]}


def answer(t):
    ans = t["answer"]
    return {"answer": ans["text"], "a_unc": ans["cost"]["uncached"], "a_out": ans["cost"]["output"],
            "a_cac": ans["cost"]["cached"], "mismatches": len(t["mismatches"])}


def example(label, bundle, tag, qid, kind):
    t = json.load(open(os.path.join(HERE, "results", f"{tag}_{bundle}", qid, "trace.json"), encoding="utf-8"))
    view = open(os.path.join(HERE, "data", "corpus", bundle, "view", "1.md"), encoding="utf-8").read()
    gold = gold_pieces(bundle, qid, kind)
    steps = [dict(piece(r, view), g=r["anchor"] in gold, **decision(r)) for r in t["trace"]]
    return {"label": label, "doc": f"{bundle} ({t['doc_tokens']:,} tokens, {t['pieces']} pieces)",
            "question": t["question"]["question"], "doc_tokens": t["doc_tokens"], "root": t["root_tokens"],
            "steps": steps, "gold": gold_answer(bundle, qid, kind), **answer(t), **settings(t)}


def page(data, live=None):
    """The replay page with its examples (and, when served by serve.py, the live panel's documents)."""
    tpl = open(os.path.join(HERE, "viz", "replay_template.html"), encoding="utf-8").read()
    js = lambda v: json.dumps(v, ensure_ascii=False).replace("</", "<\\/")
    return tpl.replace("/*DATA*/[]", js(data)).replace("/*LIVE*/null", js(live))


def examples():
    return [example(*e) for e in EXAMPLES
            if os.path.exists(os.path.join(HERE, "results", f"{e[2]}_{e[1]}", e[3], "trace.json"))]


def main():
    data = examples()
    out = os.path.join(HERE, "viz", "replay.html")
    open(out, "w", encoding="utf-8").write(page(data))
    print(f"wrote {out} with {len(data)} examples")


if __name__ == "__main__":
    main()
