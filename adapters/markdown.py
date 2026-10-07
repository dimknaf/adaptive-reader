"""Markdown adapter: a .md file -> Document.

Format-specific parsing only:
  - YAML front matter ("---" ... "---" at the top) -> meta (flat "key: value" lines), removed from the view text;
  - the title is a generic join of the identity fields that are present (title/name/company, document_type/type,
    period, published_at/date), else the first heading, else the doc_id;
  - ATX headings ("#" .. "######", outside code fences) -> heading lines with their markdown depth;
  - table rows ("| ... |" lines) -> tables, the header row being the row before a separator row, else the first row.
Setext headings (a line underlined with === or ---) are not used: in converted documents a "----" line under a number
row is common and is not a heading.
"""
import re
from pathlib import Path

from adapters import Document, code_mask, find_tables, normalise

_FRONT = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)
_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
TITLE_FIELDS = (("title", "name", "company"), ("document_type", "type"), ("period",), ("published_at", "date"))


def parse_front_matter(raw):
    """(meta, rest). Only flat "key: value" lines are read; other lines are kept verbatim under their key as text."""
    m = _FRONT.match(raw)
    if not m:
        return {}, raw
    meta, key = {}, None
    for line in m.group(1).split("\n"):
        kv = re.match(r"^([A-Za-z0-9_][A-Za-z0-9_ .-]*):(?:[ \t]+(.*))?$", line)
        if kv and not line[:1].isspace():
            key, v = kv.group(1).strip(), (kv.group(2) or "").strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            meta[key] = None if v in ("", "null", "~", "Null", "NULL") else v
        elif key is not None and line.strip():
            prev = meta.get(key)
            meta[key] = (prev + "\n" if prev else "") + line.strip()
    return meta, raw[m.end():]


def title_from(meta, headings, doc_id):
    parts = []
    for group in TITLE_FIELDS:
        for k in group:
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
                break
    if parts:
        return " · ".join(parts)
    return headings[0]["text"] if headings else doc_id


def headings_of(text):
    lines = text.split("\n")
    out, pos = [], 0
    for line, code in zip(lines, code_mask(lines)):
        if not code:
            m = _ATX.match(line)
            if m and m.group(2).strip():
                out.append({"char_start": pos, "char_end": pos + len(line), "text": m.group(2).strip(),
                            "depth": len(m.group(1))})
        pos += len(line) + 1
    return out


def from_text(raw, doc_id, title=None, meta=None):
    fm, body = parse_front_matter(raw.lstrip("﻿"))
    fm.update(meta or {})
    norm = normalise(body)
    headings = headings_of(norm.text)
    return Document(doc_id=doc_id, title=title or title_from(fm, headings, doc_id), text=norm.text, meta=fm,
                    headings=headings, tables=find_tables(norm.text, norm.header_rows), pages=norm.pages)


def parse(path, entry=None):
    """entry may hold "lines": [first, last] (1-based, inclusive) to take a slice of the file's lines as the body; the
    file's front matter still gives the meta and the title."""
    entry = entry or {}
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    meta = {"source": path.name}
    if entry.get("lines"):
        first, last = entry["lines"]
        fm, _ = parse_front_matter(raw.lstrip("﻿"))
        meta.update(fm, lines=f"{first}-{last}")
        raw = "\n".join(raw.split("\n")[first - 1:last]) + "\n"
    return from_text(raw, entry.get("doc_id") or path.stem, title=entry.get("title"), meta=meta)
