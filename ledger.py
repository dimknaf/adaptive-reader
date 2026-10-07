"""A local model of the server's prefix cache, so the cost of every request is predicted and checked, not estimated.

The rules it models (vLLM 0.26, verified in the server's source and checked on every request against its cached_tokens):
  - the cache holds whole 16-token blocks only, keyed by the chain of everything before them (a cache is a prefix);
  - a request hits the longest chain of cached blocks, but never more than ((len-1)//16)*16 tokens: vLLM always
    computes at least the last token, and hits count whole blocks, so a fully cached block-aligned prompt pays its
    whole last block again;
  - after a request, every full block of what the model computed is cached: the prompt plus every sampled token
    except the last (the last is sampled, never fed back).
Eviction is not modelled: the reader's memory is hit by every request and stays warm. Any difference from the server's
own cached_tokens is recorded as a mismatch (an eviction, another client's reset, or a bug) - never silently absorbed.

Cost = uncached input + 4 x output (the cost weighting used throughout); cached input is shown for information only.
"""
import hashlib
import struct
import threading
from collections import defaultdict

BLOCK = 16
OUTPUT_WEIGHT = 4


def _chain(ids):
    """Chained hashes of the full blocks of ids."""
    h, out = b"", []
    for i in range(0, len(ids) - len(ids) % BLOCK, BLOCK):
        h = hashlib.sha256(h + struct.pack(f"<{BLOCK}q", *ids[i:i + BLOCK])).digest()
        out.append(h)
    return out


class Ledger:
    def __init__(self, learn_tags=("warm",)):
        """Requests tagged in learn_tags teach the ledger what the server already had cached (a shared prefix warmed by
        an earlier run) instead of being checked: their server-reported hit is taken as known cached blocks."""
        self.learn_tags = set(learn_tags)
        self.lock = threading.Lock()
        self.blocks = set()
        self.totals = defaultdict(lambda: {"requests": 0, "prompt": 0, "uncached": 0, "cached": 0, "output": 0})
        self.mismatches = []
        self.log = []                   # one row per request: tag, prompt length, predicted hit, server hit, output

    # ---- the model --------------------------------------------------------------------------------------------------
    def reset(self):
        with self.lock:
            self.blocks = set()

    def predict(self, prompt):
        """Tokens of this prompt the server will serve from its cache."""
        n = 0
        for h in _chain(prompt):
            if h not in self.blocks:
                break
            n += BLOCK
        return min(n, ((len(prompt) - 1) // BLOCK) * BLOCK)

    def add(self, fed):
        for h in _chain(fed):
            self.blocks.add(h)

    # ---- observing requests ------------------------------------------------------------------------------------------
    def __call__(self, prompt, out, tag):
        """Observer for chat.Client: predict before adding, compare with the server, then cache what was computed."""
        c = out["choices"][0]
        sampled = [int(t.split(":")[1]) for t in (c.get("logprobs") or {}).get("tokens") or []]
        u = out.get("usage") or {}
        server = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        output = u.get("completion_tokens", len(sampled))
        with self.lock:
            if tag in self.learn_tags and server:
                self.add(list(prompt[:server]))
            hit = self.predict(prompt)
            if u.get("prompt_tokens", len(prompt)) != len(prompt):
                raise RuntimeError(f"server counted {u.get('prompt_tokens')} prompt tokens, sent {len(prompt)}")
            if server is not None and server != hit:
                self.mismatches.append({"tag": tag, "prompt": len(prompt), "predicted": hit, "server": server})
            self.add(list(prompt) + sampled[:-1])
            t = self.totals[tag]
            t["requests"] += 1
            t["prompt"] += len(prompt)
            t["uncached"] += len(prompt) - hit
            t["cached"] += hit
            t["output"] += output
            self.log.append((tag, len(prompt), hit, server, output))

    # ---- reporting ---------------------------------------------------------------------------------------------------
    def total(self, tags=None):
        keys = ("requests", "prompt", "uncached", "cached", "output")
        s = {k: sum(t[k] for tag, t in self.totals.items() if tags is None or tag in tags) for k in keys}
        s["weighted"] = s["uncached"] + OUTPUT_WEIGHT * s["output"]
        return s

    def mark(self):
        """A position in the log, to cost a span of requests: ledger.since(mark)."""
        return len(self.log)

    def since(self, mark):
        rows = self.log[mark:]
        s = {"requests": len(rows), "prompt": sum(r[1] for r in rows), "uncached": sum(r[1] - r[2] for r in rows),
             "cached": sum(r[2] for r in rows), "output": sum(r[4] for r in rows),
             "server_uncached": (sum(r[1] - r[3] for r in rows) if all(r[3] is not None for r in rows) else None)}
        s["weighted"] = s["uncached"] + OUTPUT_WEIGHT * s["output"]
        return s
