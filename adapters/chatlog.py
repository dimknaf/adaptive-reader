"""Chat-log adapter: a JSON chat log -> Document.

Accepted shapes (format-specific parsing only):
  - a list of batches/sessions {batch_number?, time_anchor?, turns: [[message, ...], ...] or [message, ...]}, as in
    the BEAM benchmark;
  - {"sessions": [...]} with the same session shape, or {"messages": [...]} / a plain list of messages (one session).
A message is {role, content, time_anchor?}. The view text has one heading line per session,
"## Session <n> · <time anchor>", and each message as "<role>: <content>" lines, messages separated by a blank line.
A message whose own time anchor differs from its session's gets it in brackets: "user [April-02-2024]: ...".
Message content gets the same generic normalisation as any document; markdown headings inside messages are plain lines
here (only sessions are scopes). Session headings carry an explicit level 1.
"""
import json
from pathlib import Path

from adapters import Document, find_tables, normalise


def _sessions(data):
    if isinstance(data, dict):
        if "sessions" in data:
            return data["sessions"]
        if "messages" in data:
            return [{"turns": data["messages"]}]
        raise ValueError("chat log: expected a list, or a dict with 'sessions' or 'messages'")
    if data and all(isinstance(m, dict) and "role" in m and "content" in m for m in data):
        return [{"turns": data}]
    return data


def _messages(session):
    out = []
    for turn in session.get("turns", session.get("messages", [])):
        out.extend(turn if isinstance(turn, list) else [turn])
    return out


def from_data(data, doc_id, title=None):
    chunks, headings, tables, pos = [], [], [], 0
    sessions = _sessions(data)
    n_msgs = 0
    for k, session in enumerate(sessions, 1):
        msgs = _messages(session)
        anchor = session.get("time_anchor") or next((m.get("time_anchor") for m in msgs if m.get("time_anchor")), None)
        htext = f"Session {session.get('batch_number', k)}" + (f" · {anchor}" if anchor else "")
        line = "## " + htext
        headings.append({"char_start": pos, "char_end": pos + len(line), "text": htext, "level": 1})
        chunks.append(line + "\n\n")
        pos += len(line) + 2
        for m in msgs:
            ta = m.get("time_anchor")
            prefix = f"{m.get('role', 'unknown')}" + (f" [{ta}]" if ta and ta != anchor else "") + ":"
            norm = normalise(str(m.get("content", "")))
            body = norm.text.rstrip("\n")
            block = (prefix + " " + body if body else prefix) + "\n\n"
            shift = len(prefix) + 1 if body else 0
            tables.extend(find_tables(block, [h + shift for h in norm.header_rows], offset=pos))
            chunks.append(block)
            pos += len(block)
            n_msgs += 1
    text = "".join(chunks)
    text = text[:-1] if text.endswith("\n\n") else text
    tables = [t for t in tables if t["char_end"] <= len(text)]
    return Document(doc_id=doc_id, title=title or f"Chat log {doc_id}", text=text,
                    meta={"format": "chatlog", "sessions": len(sessions), "messages": n_msgs},
                    headings=headings, tables=tables, pages=[])


def parse(path, entry=None):
    entry = entry or {}
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    doc = from_data(data, entry.get("doc_id") or path.stem, title=entry.get("title"))
    doc.meta["source"] = path.name
    return doc
