"""The MiniCPM server, in one place: its address, requests, tokenizer calls and prefix-cache counters.

The server is vLLM (OpenVINO backend) serving MiniCPM5-2B with prefix caching in 16-token blocks. The cost check needs
--enable-prompt-tokens-details (per-request cached_tokens). VLLM_SERVER_DEV_MODE=1 enables POST /reset_prefix_cache,
which only cold-cost measurements use. The backend proved unstable under 8 concurrent long requests, so callers use 4
and every request is retried after a dropped connection.

  CPM_URL    default http://localhost:8000
  CPM_MODEL  default MiniCPM5-2B
"""
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request

URL = os.getenv("CPM_URL", "http://localhost:8000")
MODEL = os.getenv("CPM_MODEL", "MiniCPM5-2B")
BLOCK = 16                                                  # the server's prefix-cache block, in tokens


def post(path, body, timeout=900):
    """One request; a dropped connection or a 5xx is retried up to 3 times after a pause (the answer does not change)."""
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == 3:
                raise
        except (urllib.error.URLError, http.client.RemoteDisconnected, ConnectionError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(15 * (attempt + 1))


def enc(text):
    """Token ids of a text, special-token strings included, no BOS added (the server's own tokenizer)."""
    return post("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False})["tokens"]


def tokenize_strs(text):
    """(ids, token strings) of a text, no BOS. The strings are byte-level (GPT-2 alphabet): see corpus.byte_decoder."""
    out = post("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False, "return_token_strs": True})
    return out["tokens"], out["token_strs"]


def detokenize(ids):
    return post("/detokenize", {"model": MODEL, "tokens": list(ids)})["prompt"]


def chat_ids(messages):
    """Token ids of a conversation rendered by the server's chat template, ending with the assistant header."""
    return post("/tokenize", {"model": MODEL, "messages": messages, "add_generation_prompt": True})["tokens"]


def common(a, b):
    """Length of the shared prefix of two token lists."""
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _metrics_text():
    return urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode()


def metrics():
    """The server's prefix-cache counters (tokens queried / tokens served from cache) and its queue."""
    txt = _metrics_text()
    get = lambda k: float(re.search(r"^" + k + r"\{[^}]*\} ([0-9.e+]+)$", txt, re.M).group(1))
    return {"queries": get("vllm:prefix_cache_queries_total"), "hits": get("vllm:prefix_cache_hits_total"),
            "running": get("vllm:num_requests_running"), "waiting": get("vllm:num_requests_waiting")}


def cache_config():
    """The labels of vllm:cache_config_info (block_size, num_gpu_blocks, kv_cache_size_tokens, ...)."""
    m = re.search(r"^vllm:cache_config_info\{([^}]*)\}", _metrics_text(), re.M)
    return dict(re.findall(r'(\w+)="([^"]*)"', m.group(1))) if m else {}


def idle():
    m = metrics()
    return m["running"] == 0 and m["waiting"] == 0


def warm(ids):
    """Compute (and so cache) a prefix."""
    return post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0})


def reset_prefix_cache():
    """Empty the prefix cache. vLLM answers 200 even when it could not reset (any block in use): callers must verify,
    e.g. by re-warming a prefix and checking that it hit nothing."""
    req = urllib.request.Request(URL + "/reset_prefix_cache", data=b"", method="POST")
    urllib.request.urlopen(req, timeout=60).read()


def reset_and_warm(ids, send=None):
    """Empty the prefix cache, then re-warm only the shared prefix, so what follows is measured as brand new.
    send(ids) replaces the plain warm request (so a ledger can observe it)."""
    reset_prefix_cache()
    before = metrics()
    (send or warm)(ids)
    if metrics()["hits"] - before["hits"] > 0:
        raise RuntimeError("prefix cache was not reset")
