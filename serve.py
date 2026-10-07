"""Try the reader yourself: a local page where you pick a document, type a question, set the three thresholds and
watch the model read it live. It is the replay page (viz/replay.html) fed by a running reader, one piece at a time.

  python serve.py [--port 8765]        -> http://localhost:8765
The model server (CPM_URL, default http://localhost:8000) must be up. One run at a time. Each run is also saved to
results/web_<bundle>/<run>/{trace.json, trace.html}.
"""
import argparse
import itertools
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import reader
import viz

HERE = os.path.dirname(os.path.abspath(__file__))
# the documents offered (Analog Devices FY2024 10-K slices; the worked examples' Home Depot text is not offered)
DOCS = {"mini5k": "MD&A", "mini5s": "financial statements", "mini3k": "MD&A, first part",
        "mini20k": "MD&A and more (slow)"}
QUESTIONS = {"mini5k": "mini5k_rel.json", "mini5s": "mini5s_rel.json", "mini3k": "mini3k.json",
             "mini20k": "mini20k.json"}
JOBS = {}
BUSY = threading.Lock()
CURRENT = None                                     # the run holding BUSY (a reloaded page can follow it)
COUNT = itertools.count(1)


def documents():
    """What the page's live panel offers: each document with its size and the questions already asked of it."""
    out = []
    for b, what in DOCS.items():
        pieces, views = reader.load_bundle(b)
        n = sum(p["n_tokens"] for p in pieces)
        qs = json.load(open(os.path.join(HERE, "data", "questions", QUESTIONS[b]), encoding="utf-8"))
        out.append({"bundle": b, "label": f"Analog Devices 10-K, {what} ({len(pieces)} pieces, {n:,} tokens)",
                    "samples": [q["question"] for q in qs]})
    return out


class Job:
    def __init__(self, bundle, question, strictness, summary_threshold, enough_threshold):
        self.name = time.strftime("%Y%m%d-%H%M%S") + f"-{next(COUNT)}"
        self.bundle, self.question = bundle, question
        self.args = (strictness, summary_threshold, enough_threshold)
        self.reader, self.result, self.error, self.status = None, None, None, "starting"

    def skeleton(self):
        """The document as the page shows it before any decision, and the run's settings."""
        pieces, views = reader.load_bundle(self.bundle)
        st, su, en = self.args
        return {"doc": f"{self.bundle} ({sum(p['n_tokens'] for p in pieces):,} tokens, {len(pieces)} pieces)",
                "doc_tokens": sum(p["n_tokens"] for p in pieces), "steps": [viz.piece(p, views[p["a"]]) for p in pieces],
                **viz.settings({"strictness": st, "summary_threshold": su, "enough_threshold": en,
                                "forms": list(reader.FORMS)})}

    def state(self):
        rows = list(self.reader.trace) if self.reader else []
        out = {"status": self.status, "error": self.error, "root": getattr(self.reader, "root_tokens", None),
               "steps": [viz.decision(r) for r in rows]}
        if self.result:
            out["final"] = dict(viz.answer(self.result), trace=f"/trace/{self.name}")
        return out

    def work(self):
        try:
            self.reader = reader.Reader(self.bundle, {"id": f"web-{self.name}", "question": self.question}, *self.args)
            self.status = "reading"
            res = self.reader.run(log=None)
            out = os.path.join(HERE, "results", f"web_{self.bundle}", self.name)
            os.makedirs(out, exist_ok=True)
            json.dump(res, open(os.path.join(out, "trace.json"), "w", encoding="utf-8"), indent=1)
            reader.write_html(res, os.path.join(out, "trace.html"))
            self.result, self.status = res, "done"
        except Exception as e:                       # shown on the page (e.g. the model server is down)
            self.error, self.status = f"{type(e).__name__}: {e}", "error"
        finally:
            BUSY.release()


def number(v, lo=0.0, hi=1.0):
    if v is None or v == "":
        raise ValueError("fill in all three thresholds")
    v = float(v)
    if not lo <= v <= hi:
        raise ValueError(f"{v} is outside [{lo}, {hi}]")
    return v


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, kind="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            live = {"docs": documents(), "defaults": {"strictness": reader.STRICTNESS,
                    "summary_threshold": reader.SUMMARY_THRESHOLD, "enough_threshold": reader.ENOUGH_THRESHOLD}}
            return self.reply(200, viz.page(viz.examples(), live), "text/html; charset=utf-8")
        if url.path == "/api/job":
            job = JOBS.get(parse_qs(url.query).get("id", [""])[0])
            return self.reply(200, job.state()) if job else self.reply(404, {"error": "no such run"})
        if url.path.startswith("/trace/"):
            job = JOBS.get(url.path[len("/trace/"):])
            path = job and os.path.join(HERE, "results", f"web_{job.bundle}", job.name, "trace.html")
            if path and os.path.exists(path):
                return self.reply(200, open(path, encoding="utf-8").read(), "text/html; charset=utf-8")
        return self.reply(404, {"error": "not found"})

    def do_POST(self):
        global CURRENT
        if urlparse(self.path).path != "/api/run":
            return self.reply(404, {"error": "not found"})
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self.reply(415, {"error": "JSON only"})   # a cross-site form cannot start a run
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("expected a JSON object")
            bundle, question = body.get("bundle"), " ".join(str(body.get("question") or "").split())
            if bundle not in DOCS:
                raise ValueError("unknown document")
            if not 3 <= len(question) <= 1000:
                raise ValueError("the question must be 3 to 1000 characters")
            job = Job(bundle, question, number(body.get("strictness")), number(body.get("summary_threshold")),
                      number(body.get("enough_threshold")))
        except (ValueError, TypeError) as e:
            return self.reply(400, {"error": str(e)})
        if not BUSY.acquire(blocking=False):
            cur = CURRENT
            live = cur and cur.status in ("starting", "reading")          # never a finished run
            follow = dict(cur.skeleton(), job=cur.name, question=cur.question) if live else {}
            return self.reply(409, dict(follow, error="a run is in progress: wait for it to finish"))
        CURRENT = job
        JOBS[job.name] = job
        threading.Thread(target=job.work, daemon=True).start()
        self.reply(200, dict(job.skeleton(), job=job.name))

    def log_message(self, fmt, *args):             # quiet: the page polls every second
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Adaptive reader: http://localhost:{args.port}  (model server {reader.cpm.URL}; Ctrl+C to stop)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
