"""The adaptive reader: read a document once, piece by piece, for a question known before reading, and keep only what
is needed. Every decision is the model's, one token read from the logits; code only compares it to a threshold.

For each piece, in document order:
  1. it is appended right after the memory M and read once;
  2. "[a.p] | useful (yes or no):"      removed when P(yes) is below STRICTNESS (M rolls back: free)
  3. "[a.p] | keep:" full or summary     summary when P(summary) reaches SUMMARY_THRESHOLD (1: not asked, kept full)
       full    -> M continues through the piece: it was already computed, so it is free
       summary -> the model writes a short summary of the piece, then judges it:
                  "[a.p] | summary enough (yes or no):"  the summary is kept (and the piece leaves the context) when
                  P(yes) reaches ENOUGH_THRESHOLD; otherwise the piece is kept in full
M is a stack: the fixed instructions and worked examples (shared by every run, computed once), the question and the
outline, then one scope per document part with what was kept from it. M's size is whatever the decisions keep.

  python reader.py <bundle> <questions.json> [--ids q1,q2] [--tag t]
                   [--strictness 0.7] [--summary-threshold 1] [--enough-threshold 0.7]   (summary threshold 1: off;
                   0.98 turns summaries on)
Writes results/<tag>_<bundle>/<question id>/{trace.json, trace.html}.
"""
import argparse
import html
import json
import os
import sys
import time
import uuid

import cpm
import chat
from ledger import Ledger
from memory import Memory

HERE = os.path.dirname(os.path.abspath(__file__))
STRICTNESS = 0.7           # keep a piece when P(useful) reaches this (higher = keep less)
SUMMARY_THRESHOLD = 1.0    # write a summary when P(summary) reaches this; 1 = off (full or remove, the measured best)
ENOUGH_THRESHOLD = 0.7     # keep the summary when the model's P(summary enough) reaches this
EXAMPLES = os.path.join(HERE, "data", "examples", "reader_examples.json")
FORMS = ("full", "summary")
# generation grammars (format only): the answer and the summary cannot contain the "|" of the one-token questions
_ANS_LINE = "[ -{}-~" + chr(0xA0) + "-" + chr(0xFFFF) + "]{1,400}"
ANSWER_RE = _ANS_LINE + "(\n" + _ANS_LINE + "){0,7}"
LINE_RE = "[ -9;-Z\\\\^-{}-~" + chr(0xA0) + "-" + chr(0xFFFF) + "]{1,400}"
GUIDE = """You read a long document one piece at a time, to answer one question. Your memory holds the question, an \
outline of the document, and what you kept so far, each with its anchor [a.p] (part a, piece p). After each piece you \
are asked about it, one question at a time:
"[a.p] | useful (yes or no):" - is anything in this piece needed to answer the question? yes or no.
"[a.p] | keep:" - full or summary: full when most of the piece is needed, summary when a short note of it is enough.
"[a.p] (heading) | summary:" - one short line noting what this piece says that matters for the question: facts from \
this piece only, numbers exactly as written. It is a note of the piece, not an answer.
"[a.p] | summary enough (yes or no):" - does the note keep what the question needs from this piece, so the piece itself \
is no longer needed?
Judge only from the text. Something already in your memory does not need to be kept again."""


def load_bundle(bundle):
    d = os.path.join(HERE, "data", "corpus", bundle)
    pieces = [json.loads(line) for line in open(os.path.join(d, "pieces.jsonl"), encoding="utf-8")]
    views = {}
    for p in pieces:
        if p["a"] not in views:
            views[p["a"]] = open(os.path.join(d, "view", f"{p['a']}.md"), encoding="utf-8").read()
    return pieces, views


class Reader:
    def __init__(self, bundle, question, strictness=STRICTNESS, summary_threshold=SUMMARY_THRESHOLD,
                 enough_threshold=ENOUGH_THRESHOLD, examples=None, client=None, ledger=None):
        self.bundle, self.question = bundle, question
        self.strictness, self.summary_threshold, self.enough_threshold = strictness, summary_threshold, enough_threshold
        self.examples = json.load(open(EXAMPLES, encoding="utf-8")) if examples is None else examples
        self.pieces, self.views = load_bundle(bundle)
        self.L = ledger or Ledger()
        self.c = client or chat.Client(observer=self.L)
        self.yn = self.c.words(["yes", "no"])
        self.form = self.c.words(list(FORMS))
        self.titles = {p["a"]: p["title"] for p in self.pieces}
        self.trace = []

    # ---- ids --------------------------------------------------------------------------------------------------------
    def root(self):
        """L0 (the guide and worked examples: shared by every run) + a run id (so the run is cold without resetting a
        shared cache) + the question + the outline, padded to a block edge."""
        m = [{"role": "system", "content": GUIDE}]
        for e in self.examples:
            m += [{"role": "user", "content": e["state"]}, {"role": "assistant", "content": e["answers"]}]
        outline = "\n".join(f"[{a}] {t}" for a, t in sorted(self.titles.items()))
        self.run_id = uuid.uuid4().hex[:12]
        # the prefix every run shares is padded to a block edge, so the run id starts a fresh block: what is shared
        # across runs is exactly l0, and the ledger learns exactly that
        self.l0 = self.c.pad_to(self.c.open_user_turn(m + [{"role": "user", "content": ""}]))
        header = f"Run {self.run_id}\nQuestion: {self.question['question']}\nThe document:\n{outline}\n"
        return self.c.pad_to(self.l0 + cpm.enc(header))

    def frame(self, piece):
        """The anchor, and the section heading when the piece does not start with it (where the piece is)."""
        text = f"\n[{piece['anchor']}] "
        path = piece.get("path") or []
        starts = bool(piece.get("units")) and piece["units"][0][1] == "heading"
        if path and not starts:
            text += f"({path[-1]})\n"
        return cpm.enc(text)

    # ---- one piece --------------------------------------------------------------------------------------------------
    def step(self, mem, piece):
        m = mem.ids
        k_ids = self.c.pad_to(m + self.frame(piece) + piece["ids"])            # keep point: M through the piece
        base = self.c.pad_to(k_ids, len(self.c.header)) + self.c.header        # branch point for the questions
        mark = self.L.mark()
        lead = cpm.enc(f"[{piece['anchor']}]")
        d, mass, _ = self.c.ask(base + lead + cpm.enc(" | useful (yes or no):"), self.yn, "useful")
        row = {"anchor": piece["anchor"], "path": piece.get("path"), "char": piece["char"], "n_tokens": len(piece["ids"]),
               "useful": round(d[0], 4), "useful_mass": round(mass, 4), "m_before": len(m)}
        if d[0] < self.strictness:
            row["decision"] = "removed"
        else:
            if self.summary_threshold < 1:                                   # at 1 summaries are off: not asked
                f, _, _ = self.c.ask(base + lead + cpm.enc(" | keep:"), self.form, "keep")
                row["p_summary"] = round(f[1], 4)
            if row.get("p_summary", 0) >= self.summary_threshold:
                row.update(self.summary(mem, piece, base, k_ids))
            else:
                mem.keep_prefix(k_ids, len(k_ids), piece["anchor"], d[0])
                row["decision"] = "full"
        row["cost"] = self.L.since(mark)
        row["m_after"] = len(mem)
        row["memory"] = [it.key for it in mem.items if it.kind in ("piece", "note")]
        return row

    def summary(self, mem, piece, base, k_ids):
        """The model writes a short summary of the piece (its lead names the piece), then judges it on the same
        branch, where the summary is already computed: "[a.p] | summary enough (yes or no):". The summary is kept when
        P(yes) reaches the enough threshold; otherwise the piece, judged useful, is kept in full (free)."""
        path = piece.get("path") or []
        name = f"[{piece['anchor']}]" + (f" ({path[-1][:60]})" if path else "")
        lead = base + cpm.enc(f"{name} | summary:")
        gen, text, finish, _ = self.c.generate(lead, 80, LINE_RE, "summary")
        dist, mass, _ = self.c.ask(lead + gen + cpm.enc(f"\n[{piece['anchor']}] | summary enough (yes or no):"),
                                   self.yn, "enough")
        out = {"summary": text.strip(), "finish": finish, "p_enough": round(dist[0], 4), "enough_mass": round(mass, 4)}
        if dist[0] >= self.enough_threshold:
            mem.append("note", piece["anchor"], self.c.pad_to(cpm.enc(f"\n{name} summary: ") + gen), None)
            return dict(out, decision="summary")
        mem.keep_prefix(k_ids, len(k_ids), piece["anchor"], None)
        return dict(out, decision="full")

    # ---- the answer -------------------------------------------------------------------------------------------------
    def answer(self, mem):
        """The answer is written after its own lead ("Answer:"), from what was kept."""
        ask = cpm.enc(f"\n\nQuestion: {self.question['question']}\nAnswer from what you kept of this document above, "
                      f"citing its anchors like [1.2]; the worked examples ([x.*]) are not part of it. If what you "
                      f"kept does not contain the answer, say it is not stated.")
        base = self.c.pad_to(mem.ids + ask, len(self.c.header)) + self.c.header + cpm.enc("Answer:")
        mark = self.L.mark()
        gen, text, finish, _ = self.c.generate(base, 200, ANSWER_RE, "answer")
        return {"text": text.strip(), "finish": finish, "cost": self.L.since(mark)}

    # ---- the run ----------------------------------------------------------------------------------------------------
    def run(self, log=print):
        t0 = time.time()
        root = self.root()
        self.root_tokens = len(root)
        self.c.send(self.l0 + [self.c.nl], {"max_tokens": 1, "temperature": 0}, "warm")   # the ledger learns L0
        mark = self.L.mark()
        mem = Memory(root)
        part = None
        for piece in self.pieces:
            if piece["a"] != part:
                if part is not None:
                    mem.close_scope()
                part = piece["a"]
                mem.open_scope(f"[{part}]", self.c.pad_to(cpm.enc(f"\n## [{part}] {self.titles[part]}\n")))
            row = self.step(mem, piece)
            self.trace.append(row)
            if log:
                log(f"  {row['anchor']:>6} useful {row['useful']:.2f} -> {row['decision']:7} |M| {row['m_after']:6}"
                    f"  uncached {row['cost']['uncached']:5} out {row['cost']['output']:3}")
        if part is not None:
            mem.close_scope()
        reading = self.L.since(mark)
        ans = self.answer(mem)
        doc = sum(len(p["ids"]) for p in self.pieces)
        return {"bundle": self.bundle, "question": self.question, "run_id": self.run_id,
                "strictness": self.strictness, "summary_threshold": self.summary_threshold,
                "enough_threshold": self.enough_threshold, "forms": list(FORMS) if self.summary_threshold < 1 else ["full"],
                "pieces": len(self.pieces), "doc_tokens": doc, "root_tokens": len(root),
                "memory": [{"key": it.key, "kind": it.kind, "tokens": len(it.ids)} for it in mem.items],
                "memory_tokens": len(mem), "answer": ans, "reading": reading, "total": self.L.total(),
                "uncached_per_doc_token": round(reading["uncached"] / doc, 4) if doc else None,
                "weighted_per_doc_token": round(reading["weighted"] / doc, 4) if doc else None,
                "mismatches": self.L.mismatches, "secs": round(time.time() - t0, 1), "trace": self.trace}


def write_html(res, path):
    """A standalone page to read the decisions: one row per piece, the memory at the end, the answer, the cost."""
    e = html.escape
    rows = []
    for r in res["trace"]:
        extra = ""
        if r.get("summary"):
            label = "(kept" if r["decision"] == "summary" else "(summary not enough, piece kept in full"
            extra = f"{label}, P(enough) {r.get('p_enough', 0):.2f}): " + e(r["summary"])
        rows.append(f"<tr class='{e(r['decision'])}'><td>{e(r['anchor'])}</td><td>{e(' › '.join(r.get('path') or []))}</td>"
                    f"<td class=n>{r['n_tokens']}</td><td class=n>{r['useful']:.2f}</td><td><b>{e(r['decision'])}</b> {extra}</td>"
                    f"<td class=n>{r['cost']['uncached']}</td><td class=n>{r['cost']['output']}</td><td class=n>{r['m_after']}</td></tr>")
    rd = res["reading"]
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reader trace</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#e5e5ea;--full:#e6f4ea;--summary:#e8f0fe}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1c1e;--fg:#f2f2f7;--mut:#a1a1a6;--line:#38383a;--full:#1f3a28;--summary:#1d2d4a}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;max-width:1200px}}
h1{{font-size:20px;margin:0 0 4px}} .mut{{color:var(--mut)}} table{{border-collapse:collapse;width:100%;margin:12px 0}}
td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}} .n{{text-align:right;font-variant-numeric:tabular-nums}}
tr.full{{background:var(--full)}} tr.summary{{background:var(--summary)}}
.wrap{{overflow-x:auto}} pre{{white-space:pre-wrap;background:var(--line);padding:10px;border-radius:6px}}
</style></head><body>
<h1>{e(res['question']['question'])}</h1>
<div class=mut>{e(res['bundle'])} · {res['pieces']} pieces · {res['doc_tokens']} document tokens · strictness {res['strictness']} ·
summary threshold {res['summary_threshold']} · enough threshold {res['enough_threshold']} · run {e(res['run_id'])}</div>
<h2>Answer</h2><pre>{e(res['answer']['text'])}</pre>
<h2>Cost of reading</h2>
<div>uncached {rd['uncached']} · output {rd['output']} · weighted {rd['weighted']} (= {res['weighted_per_doc_token']} per document token) ·
memory at the end {res['memory_tokens']} tokens · ledger mismatches {len(res['mismatches'])}</div>
<h2>Pieces</h2><div class=wrap><table><tr><th>anchor</th><th>section</th><th class=n>tokens</th><th class=n>P(useful)</th>
<th>decision</th><th class=n>uncached</th><th class=n>output</th><th class=n>|M| after</th></tr>
{''.join(rows)}</table></div>
<h2>Memory at the end</h2><div>{e(', '.join(f"{m['key']} ({m['kind']}, {m['tokens']})" for m in res['memory']))}</div>
</body></html>"""
    open(path, "w", encoding="utf-8").write(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("questions")
    ap.add_argument("--ids")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--strictness", type=float, default=STRICTNESS)
    ap.add_argument("--summary-threshold", type=float, default=SUMMARY_THRESHOLD)
    ap.add_argument("--enough-threshold", type=float, default=ENOUGH_THRESHOLD)
    args = ap.parse_args()
    qs = json.load(open(args.questions, encoding="utf-8"))
    if args.ids:
        qs = [q for q in qs if q["id"] in args.ids.split(",")]
    for q in qs:
        out = os.path.join(HERE, "results", f"{args.tag}_{args.bundle}", q["id"])
        os.makedirs(out, exist_ok=True)
        print(f"{q['id']}: {q['question']}", flush=True)
        res = Reader(args.bundle, q, args.strictness, args.summary_threshold, args.enough_threshold).run()
        json.dump(res, open(os.path.join(out, "trace.json"), "w", encoding="utf-8"), indent=1)
        write_html(res, os.path.join(out, "trace.html"))
        print(f"{q['id']}: kept {[m['key'] for m in res['memory'] if m['kind'] in ('piece', 'note')]}, "
              f"weighted/doc {res['weighted_per_doc_token']}, mismatches {len(res['mismatches'])}\n"
              f"  answer: {res['answer']['text'][:300]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
