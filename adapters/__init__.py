"""Adapters turn one source file into a Document: the normalised view text plus its heading lines and tables.

Everything format-specific lives in an adapter module (markdown.py, chatlog.py). The corpus core (corpus.py) sees only a
Document and never learns what the document is about. A new document type means a new adapter; the core never changes.

This package also holds the generic, meaning-preserving normalisation that every adapter applies to its text:
  - line endings to "\\n"; control characters dropped (form feed becomes a line break)
  - zero-width characters dropped (ZWSP, word joiner, BOM, soft hyphen); non-breaking and other unicode spaces -> " "
  - HTML comments removed (page markers such as <!-- PAGE 12 --> are kept as metadata only, never as text)
  - HTML entities decoded (only complete "&name;" / "&#123;" forms, decoded once)
  - markdown table separator rows (|---|:---:|) dropped, cell padding stripped ("|   a   |  b |" -> "| a | b |")
  - trailing spaces stripped, 3+ newlines collapsed to 2, no leading blank lines, exactly one final "\\n"
Fenced code blocks are left alone by the comment, entity and table rules. No cell or value is ever removed: equal
adjacent cells such as "(10) | (10)" are real values and stay.
"""
import html
import importlib
import re
from dataclasses import dataclass, field


@dataclass
class Document:
    """One source document. All char positions index `text` (the view text) in Python str characters."""
    doc_id: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)
    headings: list = field(default_factory=list)   # {char_start, char_end, text, depth?, level?}; the core adds style/level
    tables: list = field(default_factory=list)     # {char_start, char_end, header: [s, e]} (header = the header row line)
    pages: list = field(default_factory=list)      # [char_pos, page_no]: page page_no starts at char_pos


def get(name):
    """The adapter module for a name in bundles.json ("markdown", "chatlog")."""
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name or ""):
        raise ValueError(f"bad adapter name {name!r}")
    return importlib.import_module(f"adapters.{name}")


# ---------------------------------------------------------------- generic normalisation

_ZERO_WIDTH = {ord(c): None for c in "​⁠﻿­"}
_SPACES = {ord(c): " " for c in "             "
                              "  　"}
_CONTROL = {c: None for c in list(range(0, 9)) + [11] + list(range(14, 32)) + [127]}   # keeps \t \n \r \f
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_COMMENT = re.compile(r"<!--(.*?)-->", re.S)
_PAGE_COMMENT = re.compile(r"\s*page\s*[:#]?\s*(\d{1,6})\s*", re.I)
_ENTITY = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_SEPARATOR_CELL = re.compile(r":?-+:?")
PAGE_MARK = "\x00"          # sentinels used only inside normalise(); raw NULs are dropped first
HEADER_MARK = "\x01"


@dataclass
class Normalised:
    text: str
    pages: list          # [char_pos, page_no]
    header_rows: list    # char positions of table rows that were followed by a separator row (markdown header rows)


def code_mask(lines):
    """For each line, True if it is inside a fenced code block (fence lines included)."""
    mask, fence = [], None
    for line in lines:
        m = _FENCE.match(line)
        if fence is None:
            if m:
                fence = m.group(1)
            mask.append(fence is not None)
        else:
            mask.append(True)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not line.strip()[len(m.group(1)):]:
                fence = None
    return mask


def is_row(line):
    s = line.strip()
    return len(s) >= 2 and s[0] == "|" and s[-1] == "|"


def _cells(line):
    s = line.strip()
    return [c.strip() for c in _CELL_SPLIT.split(s[1:-1])]


def _decode_entity(m):
    out = html.unescape(m.group(0))
    return out if out != m.group(0) else m.group(0)


def _comment(m):
    pm = _PAGE_COMMENT.fullmatch(m.group(1))
    return f"{PAGE_MARK}{int(pm.group(1))}{PAGE_MARK}" if pm else ""


def normalise(raw):
    """Generic, meaning-preserving normalisation of one text (see the module docstring)."""
    t = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    t = t.translate(_CONTROL).translate(_ZERO_WIDTH).translate(_SPACES)
    lines = t.split("\n")
    mask = code_mask(lines)
    out_lines = []
    i = 0
    while i < len(lines):                          # process maximal runs of non-code lines as one block
        if mask[i]:
            out_lines.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and not mask[j]:
            j += 1
        block = "\n".join(lines[i:j])
        block = _COMMENT.sub(_comment, block)
        block = _ENTITY.sub(_decode_entity, block)
        rows = block.split("\n")
        for k, line in enumerate(rows):
            if not is_row(line):
                out_lines.append(line)
                continue
            cells = _cells(line)
            if any(cells) and all(c == "" or _SEPARATOR_CELL.fullmatch(c) for c in cells):
                # a separator row: drop it, and mark the row just before it (if any) as a header row
                if out_lines and is_row(out_lines[-1].lstrip(HEADER_MARK)) and not out_lines[-1].startswith(HEADER_MARK):
                    out_lines[-1] = HEADER_MARK + out_lines[-1]
                continue
            out_lines.append("|" + "|".join(f" {c} " if c else " " for c in cells) + "|")
        i = j
    return _finish(out_lines)


def _finish(lines):
    """Strip trailing spaces, collapse blank runs, drop leading blanks, resolve sentinels into positions."""
    out, pages, headers, pending = [], [], [], []
    pos, blank = 0, False
    for line in lines:
        header = line.startswith(HEADER_MARK)
        if header:
            line = line[1:]
        marks = []
        if PAGE_MARK in line:
            parts = line.split(PAGE_MARK)
            kept = ""
            for k, part in enumerate(parts):
                if k % 2 and part.isdigit():
                    marks.append((len(kept), int(part)))
                else:
                    kept += part
            line = kept
        line = line.rstrip(" \t")
        if not line.strip():
            pending.extend(n for _, n in marks)
            if out:
                blank = True
            continue
        if blank:
            out.append("")
            pos += 1
            blank = False
        for n in pending:
            pages.append([pos, n])
        pending = []
        for k, n in marks:
            pages.append([pos + min(k, len(line)), n])
        if header:
            headers.append(pos)
        out.append(line)
        pos += len(line) + 1
    for n in pending:
        pages.append([pos, n])
    text = "\n".join(out) + "\n" if out else ""
    for p in pages:
        p[0] = min(p[0], len(text))
    return Normalised(text, pages, headers)


def find_tables(text, header_rows=(), offset=0):
    """Tables = maximal runs of table-row lines outside code fences. A row marked as a markdown header row starts a new
    table. Each table's header is its first row. Positions are shifted by `offset`."""
    header_rows = set(header_rows)
    lines = text.split("\n")
    mask = code_mask(lines)
    tables, cur, pos = [], None, 0
    for line, code in zip(lines, mask):
        end = pos + len(line)
        if not code and is_row(line):
            if cur is None or pos in header_rows:
                if cur is not None:
                    tables.append(cur)
                cur = {"char_start": pos + offset, "char_end": end + offset, "header": [pos + offset, end + offset]}
            else:
                cur["char_end"] = end + offset
        elif cur is not None:
            tables.append(cur)
            cur = None
        pos = end + 1
    if cur is not None:
        tables.append(cur)
    return tables
