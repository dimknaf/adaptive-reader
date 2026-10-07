"""Worked examples for the cached L0, built from judge labels on a document used only for examples, never evaluated.

Each example is one reading step as a chat turn pair, mirroring a live step: the user turn holds the question and one
piece with its anchor; the assistant turn holds the answers the reader is asked for, in the same "subject | field:"
form ("[x.3] | useful (yes or no): no"). Only (question, piece) pairs on which both judges agree are used, balanced
between useful and not useful. The anchors use the part name "x", which no document part uses.

  python examples.py <bundle>=<labels.json> [<bundle>=<labels.json> ...] [--n 6]
                     [--out data/examples/reader_examples.json]
Each source (a corpus bundle and its labels) gives up to n balanced examples; the sources are interleaved.
labels.json: {"questions": [...], "judges": [{"labels": [...]}, {"labels": [...]}]} (two independent judges).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def agreed(labels_a, labels_b):
    """(question id, anchor) -> the label both judges gave (relevance within 20). Two label formats: lookup labels
    (useful yes/no + keep) must agree on both; relevance labels (essential / supporting / not_relevant) must agree,
    essential becomes useful + keep full, not_relevant becomes not useful, and supporting (ambiguous to keep or not)
    is never used as an example."""
    b = {(x["question_id"], x["anchor"]): x for x in labels_b}
    out = {}
    for x in labels_a:
        y = b.get((x["question_id"], x["anchor"]))
        if not y or abs(x["relevance"] - y["relevance"]) > 20:
            continue
        rel = round((x["relevance"] + y["relevance"]) / 20) * 10
        if "label" in x:
            if x["label"] == y["label"] and x["label"] != "supporting":
                yes = x["label"] == "essential"
                out[(x["question_id"], x["anchor"])] = {"useful": "yes" if yes else "no", "keep": "full" if yes else "none",
                                                        "relevance": rel, "needed_lines": []}
        elif x["useful"] == y["useful"] and x["keep"] == y["keep"]:
            out[(x["question_id"], x["anchor"])] = dict(x, relevance=rel,
                                                        needed_lines=[l for l in x["needed_lines"] if l in y["needed_lines"]])
    return out


def render(question, piece, text, label):
    anchor = "x." + str(piece["p"])
    heading = f"({piece['path'][-1]})\n" if piece.get("path") and not text.lstrip().startswith("#") else ""
    state = f"Question: {question}\n\n[{anchor}] {heading}{text.rstrip()}"
    ans = [f"[{anchor}] | useful (yes or no): {label['useful']}"]
    if label["useful"] == "yes":
        ans.append(f"[{anchor}] | keep: {label['keep']}")
        if label["keep"] == "summary" and label.get("note"):
            # the note's lead names the piece exactly as the reader asks it: "[a.p] (heading) | summary:"; then the
            # model's own verdict on the note, as the reader asks it
            name = f"[{anchor}]" + (f" ({piece['path'][-1][:60]})" if piece.get("path") else "")
            ans.append(f"{name} | summary: {label['note']}")
            if label.get("enough"):
                ans.append(f"[{anchor}] | summary enough (yes or no): {label['enough']}")
    return {"state": state, "answers": "\n".join(ans), "question": question, "anchor": anchor, "label": label}


def build(bundle, labels_path, n=6, yes_share=0.5, hard=False, notes=None, bad_notes=None):
    """n examples, a yes_share of them useful; with hard, the not-useful ones are the most on-subject (highest judged
    relevance) first: pieces near the boundary teach it best. notes: {"<bundle>|<question id>|<anchor>": {form, note,
    verified}}: a verified note turns a useful example into "keep: <form>" with its summary, ending with the model's
    verdict on the note ("summary enough: yes"). bad_notes: {key: note} verified as not enough: each adds an example
    ending "summary enough: no",
    placed after the same piece's good example when there is one."""
    data = json.load(open(labels_path, encoding="utf-8"))
    qs = {q["id"]: q["question"] for q in data["questions"]}
    d = os.path.join(HERE, "data", "corpus", bundle)
    pieces = {p["anchor"]: p for p in (json.loads(l) for l in open(os.path.join(d, "pieces.jsonl"), encoding="utf-8"))}
    view = open(os.path.join(d, "view", "1.md"), encoding="utf-8").read()
    ok = agreed(data["judges"][0]["labels"], data["judges"][1]["labels"])
    for (qid, anchor), lab in ok.items():
        nt = (notes or {}).get(f"{bundle}|{qid}|{anchor}")
        if lab["useful"] == "yes" and nt and nt.get("verified"):
            lab.update(keep=nt["form"], note=nt["note"], enough="yes")
    yes = sorted((k for k, v in ok.items() if v["useful"] == "yes"), key=lambda k: (k[1], k[0]))
    no = [k for k, v in ok.items() if v["useful"] == "no"]
    used, asked, pick_yes, pick_no = set(), {}, [], []
    while len(pick_yes) < round(n * yes_share):                          # useful ones, spread over the questions
        free = [k for k in yes if k[1] not in used]                      # one example per piece
        if not free:
            break
        k = min(free, key=lambda k: (asked.get(k[0], 0), k[1], k[0]))
        pick_yes.append(k)
        used.add(k[1])
        asked[k[0]] = asked.get(k[0], 0) + 1
    while len(pick_no) < n - len(pick_yes):                              # the rest not useful, spread likewise
        free = [k for k in no if k[1] not in used]
        if not free:
            break
        k = min(free, key=lambda k: (asked.get(k[0], 0), -ok[k]["relevance"] if hard else 0, k[1], k[0]))
        pick_no.append(k)
        used.add(k[1])
        asked[k[0]] = asked.get(k[0], 0) + 1
    picked = [x for pair in zip(pick_yes, pick_no) for x in pair] + pick_yes[len(pick_no):] + pick_no[len(pick_yes):]
    out = []
    for qid, anchor in picked:
        p = pieces[anchor]
        out.append(render(qs[qid], p, view[p["char"][0]:p["char"][1]], ok[(qid, anchor)]))
    for key, note in (bad_notes or {}).items():
        b, qid, anchor = key.split("|")
        if b != bundle or (qid, anchor) not in ok:
            continue
        p = pieces[anchor]
        ex = render(qs[qid], p, view[p["char"][0]:p["char"][1]],
                    dict(ok[(qid, anchor)], useful="yes", keep="summary", note=note, enough="no"))
        at = next((j + 1 for j, e in enumerate(out) if e["question"] == qs[qid] and e["anchor"] == "x." + str(p["p"])),
                  len(out))
        out.insert(at, ex)
    return out, len(ok), len(yes), len(no)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="<bundle>=<labels.json>")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--yes-share", type=float, default=0.5, help="share of the examples that are useful")
    ap.add_argument("--hard", action="store_true", help="not-useful examples: the most on-subject first")
    ap.add_argument("--notes", help="verified summary notes (data/gold/examples_rel_notes.json)")
    ap.add_argument("--bad-notes", help="notes verified as not enough ({key: note}): examples ending in 'no'")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "examples", "reader_examples.json"))
    args = ap.parse_args()
    per = []
    for src in args.sources:
        bundle, labels = src.split("=", 1)
        notes = json.load(open(args.notes, encoding="utf-8")) if args.notes else None
        bad = json.load(open(args.bad_notes, encoding="utf-8")) if args.bad_notes else None
        out, n_ok, n_yes, n_no = build(bundle, labels, args.n, args.yes_share, args.hard, notes, bad)
        print(f"{bundle}: {n_ok} agreed pairs ({n_yes} useful, {n_no} not); {len(out)} examples")
        per.append(out)
    merged = [e for group in zip(*per) for e in group] + [e for g in per for e in g[min(map(len, per)):]]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(merged, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {len(merged)} examples to {args.out}")
    for e in merged:
        print(f"  {e['anchor']} {e['label']['useful']:3} {e['label']['keep']:7} {e['question'][:70]}")


if __name__ == "__main__":
    sys.exit(main())
