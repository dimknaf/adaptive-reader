"""Corpus builder: documents -> parts -> pieces -> units, every piece with an anchor [a.p].

The core is document-agnostic: it receives a Document from an adapter (adapters/markdown.py: the normalised view text,
its heading lines with their depth, its tables) and never looks at what the document is about.

  Scopes      a heading opens a scope that ends at the next heading of the same or smaller depth. Flat headings give
              one level. (No heading-style heuristics: add them only when a document needs them.)
  Tokens      each section (a heading line and its body) is tokenised by ONE tokenizer call. Token strings are
              byte-level (GPT-2 alphabet) and are inverted to exact bytes, so every token has an exact char span and
              the ids decode back to the exact view text (checked on every call).
  Units       a unit ends after a token whose text ends with a newline: a line, a table row, a heading, a blank line.
              A line longer than long_line tokens is split at sentence ends, and cut hard at hard_cut tokens.
  Pieces      units are walked in order, never across a document. Each cut lies in [min_tokens, max_tokens] at the
              strongest boundary available (heading start > blank line > table end > row end > line end > sentence
              end; never right after a heading); ties go to the least padding (-(anchor ids + piece ids)) % 16, then to
              the size closest to target_tokens. A cut that would leave a remainder smaller than min_tokens is avoided
              when another cut is possible. A piece starting inside a table repeats the table's header row first
              (n_repeat ids; its char span covers only the new text).
  Anchors     a = document (part) index in the bundle, p = piece index in the part; both <= 999, so each is one token.

  python corpus.py build <bundle>        -> data/corpus/<bundle>/{pieces.jsonl, tree.json, view/, stats.json}
Bundles are defined in data/bundles.json: an adapter and a list of documents (a file, optionally a 1-based line range).
"""
import argparse
import hashlib
import json
import os
import statistics as st
import sys
from pathlib import Path

import adapters

ROOT = Path(__file__).resolve().parent
BUNDLES = ROOT / "data" / "bundles.json"
CORPUS = ROOT / "data" / "corpus"
CONFIG = {"min_tokens": 350, "max_tokens": 600, "target_tokens": 500, "long_line": 250, "hard_cut": 600, "block": 16}

# ---------------------------------------------------------------- byte-level token strings -> bytes


# The byte-to-unicode table of GPT-2's byte-level BPE (OpenAI GPT-2, Modified MIT License; see THIRD_PARTY_NOTICES.md).
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_ENCODER = _bytes_to_unicode()                       # byte -> printable char (GPT-2 alphabet)
BYTE_DECODER = {c: b for b, c in BYTE_ENCODER.items()}   # the inversion


def token_bytes(s):
    try:
        return bytes(BYTE_DECODER[c] for c in s)
    except KeyError as e:
        raise ValueError(f"token string {s!r} is not byte-level (GPT-2 alphabet): {e}") from None


def anchor_text(a, p):
    return f"\n[{a}.{p}] "


class Tokenizer:
    """Wraps a tokenize function text -> (ids, byte-level token strings); default: the server's (cpm.tokenize_strs).
    Every call is checked lossless; the id -> bytes map it learns decodes pieces during validation."""

    def __init__(self, fn=None):
        if fn is None:
            import cpm
            fn = cpm.tokenize_strs
        self.fn, self.vocab, self._anchors, self.calls = fn, {}, {}, 0

    def tokenize(self, text):
        ids, strs = self.fn(text)
        self.calls += 1
        if len(ids) != len(strs):
            raise ValueError("tokenizer returned ids and strings of different lengths")
        bs = [token_bytes(s) for s in strs]
        for i, b in zip(ids, bs):
            if self.vocab.setdefault(i, b) != b:
                raise ValueError(f"token id {i} maps to two different byte strings")
        if b"".join(bs) != text.encode("utf-8"):
            raise ValueError("tokenizer output does not rebuild the input text")
        return list(ids), bs

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8")

    def anchor_ids(self, a, p):
        if (a, p) not in self._anchors:
            self._anchors[(a, p)] = self.tokenize(anchor_text(a, p))[0]
        return self._anchors[(a, p)]


# ---------------------------------------------------------------- tokens with char spans


def char_spans(text, bs):
    """[(c0, c1)] of each token in text. A token that splits a multi-byte character is given the character it starts
    in (so spans stay monotonic and together cover the text)."""
    char_at, n = [], 0
    for ch in text:                                   # char index of every byte offset
        char_at.extend([n] * len(ch.encode("utf-8")))
        n += 1
    char_at.append(n)
    out, b = [], 0
    for t in bs:
        out.append((char_at[b], char_at[b + len(t)]))
        b += len(t)
    return out


class Tok:
    __slots__ = ("id", "b", "c0", "c1")

    def __init__(self, i, b, c0, c1):
        self.id, self.b, self.c0, self.c1 = i, b, c0, c1


def tokenise_doc(doc, tok):
    """All tokens of the document, one tokenizer call per section (sections start at heading lines)."""
    starts = sorted({0} | {h["char_start"] for h in doc.headings})
    bounds = [s for s in starts if s < len(doc.text)] + [len(doc.text)]
    toks = []
    for s, e in zip(bounds, bounds[1:]):
        ids, bs = tok.tokenize(doc.text[s:e])
        for i, b, (c0, c1) in zip(ids, bs, char_spans(doc.text[s:e], bs)):
            toks.append(Tok(i, b, s + c0, s + c1))
    return toks


# ---------------------------------------------------------------- units

HEADING, BLANK, ROW, LINE, SENTENCE = "heading", "blank", "row", "line", "sentence"
STRENGTH = {"heading": 6, "blank": 5, "table_end": 4, "row": 3, "line": 2, "sentence": 1, "hard": 0}
_CLOSERS = (b'"', b"'", b")", b"]", "”".encode(), "’".encode())


def _sentence_end(b):
    b = b.rstrip()
    while b.endswith(_CLOSERS):
        b = b[:-1] if b[-1:] in (b'"', b"'", b")", b"]") else b[:-3]
    return b.endswith((b".", b"!", b"?"))


def build_units(doc, toks, cfg):
    """[(t0, t1, kind, line_kind)] over the document's tokens. kind is the unit's own kind (heading/blank/row/line or
    sentence for a piece of a long line); line_kind is the kind of the line it belongs to."""
    heads = {h["char_start"] for h in doc.headings}
    units, t0 = [], 0
    for t in range(len(toks)):
        if not toks[t].b.endswith(b"\n") and t != len(toks) - 1:
            continue
        line = doc.text[toks[t0].c0:toks[t].c1]
        kind = HEADING if toks[t0].c0 in heads else BLANK if not line.strip() else \
            ROW if line.strip().startswith("|") else LINE
        if t + 1 - t0 <= cfg["long_line"] or kind != LINE:
            units.append((t0, t + 1, kind, kind))
        else:                                          # a long line: sentence units, hard cut when there is no end
            s = t0
            for k in range(t0, t + 1):
                end = k == t or (_sentence_end(toks[k].b) and toks[k + 1].b[:1].isspace())
                if end or k + 1 - s >= cfg["hard_cut"]:
                    units.append((s, k + 1, SENTENCE, LINE))
                    s = k + 1
        t0 = t + 1
    return units


def boundary(units, i):
    """Strength of a cut before unit i (None: not allowed)."""
    prev, cur = units[i - 1], units[i]
    if prev[2] == HEADING:
        return None                                    # never separate a heading from what follows it
    if cur[2] == HEADING:
        return STRENGTH["heading"]
    if prev[2] == BLANK:
        return STRENGTH["blank"]
    if prev[2] == ROW:
        return STRENGTH["row"] if cur[2] == ROW else STRENGTH["table_end"]
    if prev[2] == SENTENCE and prev[3] == LINE and cur[2] == SENTENCE:
        return STRENGTH["sentence"]
    return STRENGTH["line"]


# ---------------------------------------------------------------- pieces


def table_header(doc, units, toks, i):
    """Ids of the header row of the table that unit i continues (None if unit i does not continue a table)."""
    if units[i][2] != ROW or i == 0 or units[i - 1][2] != ROW:
        return None
    c = toks[units[i][0]].c0
    for t in doc.tables:
        if t["char_start"] < c <= t["char_end"]:
            h0, h1 = t["header"]
            return [x.id for x in toks if h0 <= x.c0 and x.c1 <= h1 + 1]
    return None


def cut_pieces(doc, toks, units, tok, a, cfg):
    """[(u0, u1, repeat_ids)] unit ranges of the pieces of one document."""
    size = lambda u0, u1: units[u1 - 1][1] - units[u0][0]
    pieces, u0, p = [], 0, 1
    while u0 < len(units):
        rep = table_header(doc, units, toks, u0) or []
        total = size(u0, len(units))
        if total + len(rep) <= cfg["max_tokens"]:
            pieces.append((u0, len(units), rep))
            break
        cands = []
        for u1 in range(u0 + 1, len(units)):
            n = size(u0, u1) + len(rep)
            if n > cfg["max_tokens"]:
                break
            s = boundary(units, u1)
            if n < cfg["min_tokens"] or s is None:
                continue
            pad = (-(len(tok.anchor_ids(a, p)) + n)) % cfg["block"]
            tail_ok = size(u1, len(units)) >= cfg["min_tokens"]
            cands.append((tail_ok, s, -pad, -abs(n - cfg["target_tokens"]), u1))
        if cands:
            u1 = max(cands)[-1]
        else:                                          # no allowed cut in the window: the first allowed one after it
            u1 = next((u for u in range(u0 + 1, len(units)) if boundary(units, u) is not None
                       and size(u0, u) >= cfg["min_tokens"]), len(units))
        pieces.append((u0, u1, rep))
        u0, p = u1, p + 1
    return pieces


def heading_path(doc, c):
    """Texts of the headings whose scope contains char c, outermost first (scopes by markdown depth)."""
    stack = []
    for h in doc.headings:
        if h["char_start"] > c:
            break
        d = h.get("depth") or 1
        while stack and (stack[-1].get("depth") or 1) >= d:
            stack.pop()
        stack.append(h)
    return [h["text"] for h in stack]


def build_doc(doc, tok, a, cfg=CONFIG):
    toks = tokenise_doc(doc, tok)
    units = build_units(doc, toks, cfg)
    out = []
    for p, (u0, u1, rep) in enumerate(cut_pieces(doc, toks, units, tok, a, cfg), 1):
        t0, t1 = units[u0][0], units[u1 - 1][1]
        ids = rep + [x.id for x in toks[t0:t1]]
        c0, c1 = toks[t0].c0, toks[t1 - 1].c1
        anchor_ids = tok.anchor_ids(a, p)
        out.append({"anchor": f"{a}.{p}", "a": a, "p": p, "doc_id": doc.doc_id, "title": doc.title,
                    "path": heading_path(doc, c0),
                    "headings_inside": [h["text"] for h in doc.headings if c0 <= h["char_start"] < c1],
                    "char": [c0, c1], "n_tokens": len(ids), "ids": ids,
                    "units": [[len(rep) + units[u][0] - t0, units[u][2], toks[units[u][0]].c0] for u in range(u0, u1)],
                    "n_repeat": len(rep), "anchor_ids": anchor_ids,
                    "pad": (-(len(anchor_ids) + len(ids))) % cfg["block"],
                    "sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]})
    assert len(out) <= 999, "more than 999 pieces in one part: split the document into parts"
    return out


def validate(doc, pieces, tok):
    """Pieces tile the text exactly (no gap, no overlap) and every piece decodes to its char span."""
    pos = 0
    for pc in pieces:
        c0, c1 = pc["char"]
        assert c0 == pos, f"{pc['anchor']}: gap or overlap at char {pos}"
        assert tok.decode(pc["ids"][pc["n_repeat"]:]) == doc.text[c0:c1], f"{pc['anchor']}: ids do not decode to the span"
        pos = c1
    assert pos == len(doc.text), "the pieces do not reach the end of the text"


def scope_tree(doc, pieces):
    """Nested scopes of one document (by heading depth) with the anchors of the pieces that start in each."""
    root = {"title": doc.title, "depth": 0, "char": [0, len(doc.text)], "pieces": [], "children": []}
    stack = [root]
    for h in doc.headings:
        d = h.get("depth") or 1
        while len(stack) > 1 and stack[-1]["depth"] >= d:
            stack.pop()
        node = {"title": h["text"], "depth": d, "char": [h["char_start"], None], "pieces": [], "children": []}
        stack[-1]["children"].append(node)
        stack.append(node)
    flat = []

    def walk(n):
        flat.append(n)
        for ch in n["children"]:
            walk(ch)
    walk(root)
    for n in flat[1:]:
        later = [m["char"][0] for m in flat if m["char"][0] > n["char"][0] and m["depth"] <= n["depth"]]
        n["char"][1] = min(later) if later else len(doc.text)
    for pc in pieces:
        best = max((n for n in flat if n["char"][0] <= pc["char"][0] < n["char"][1]), key=lambda n: n["depth"])
        best["pieces"].append(pc["anchor"])
    return root


# ---------------------------------------------------------------- bundles


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_docs(name, bundles_path=BUNDLES):
    spec = json.load(open(bundles_path, encoding="utf-8"))[name]
    ad = adapters.get(spec["adapter"])
    docs = []
    for k, entry in enumerate(spec["docs"], 1):
        path = ROOT / entry["path"]                      # a relative path is relative to the repository root
        if not path.exists():
            raise FileNotFoundError(f"{entry['path']} not found. The source filings are not redistributed (see "
                                    f"data/README.md); the built bundle in data/corpus/{name} is ready to use.")
        if entry.get("sha256") and file_sha256(path) != entry["sha256"]:
            raise ValueError(f"{entry['path']} changed since the bundle was defined")
        docs.append(ad.parse(path, entry))
    return docs


def build_bundle(name, tok=None, cfg=CONFIG, out_root=CORPUS, bundles_path=BUNDLES):
    tok = tok or Tokenizer()
    docs = load_docs(name, bundles_path)
    assert len(docs) <= 999, "more than 999 documents in one bundle"
    out = Path(out_root) / name
    (out / "view").mkdir(parents=True, exist_ok=True)
    pieces, tree = [], {"bundle": name, "parts": []}
    for a, doc in enumerate(docs, 1):
        ps = build_doc(doc, tok, a, cfg)
        validate(doc, ps, tok)
        pieces += ps
        tree["parts"].append({"a": a, "doc_id": doc.doc_id, "title": doc.title, "meta": doc.meta,
                              "tokens": sum(p["n_tokens"] - p["n_repeat"] for p in ps), "scopes": scope_tree(doc, ps)})
        (out / "view" / f"{a}.md").write_text(doc.text, encoding="utf-8", newline="\n")
        marked, pos = [], 0
        for pc in ps:
            marked += [doc.text[pos:pc["char"][0]], f"⟦{pc['anchor']}⟧"]
            pos = pc["char"][0]
        marked.append(doc.text[pos:])
        (out / "view" / f"{a}.anchored.md").write_text("".join(marked), encoding="utf-8", newline="\n")
    with open(out / "pieces.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for pc in pieces:
            f.write(json.dumps(pc, ensure_ascii=False) + "\n")
    json.dump(tree, open(out / "tree.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sizes = [p["n_tokens"] for p in pieces]
    stats = {"bundle": name, "docs": len(docs), "pieces": len(pieces), "tokens": sum(sizes),
             "piece_tokens": {"min": min(sizes), "median": st.median(sizes), "max": max(sizes)},
             "mean_pad": round(st.mean(p["pad"] for p in pieces), 2), "config": cfg,
             "units": {k: sum(1 for p in pieces for u in p["units"] if u[1] == k)
                       for k in (HEADING, BLANK, ROW, LINE, SENTENCE)}}
    json.dump(stats, open(out / "stats.json", "w", encoding="utf-8"), indent=1)
    return pieces, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("bundle")
    args = ap.parse_args(argv)
    try:
        pieces, stats = build_bundle(args.bundle)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in stats.items() if k != "config"}, indent=1))
    for p in pieces:
        print(f"  [{p['anchor']}] {p['n_tokens']:4} tokens, pad {p['pad']:2}, repeat {p['n_repeat']}, "
              f"path {' > '.join(p['path'])[-70:]!r}")


if __name__ == "__main__":
    sys.exit(main())
