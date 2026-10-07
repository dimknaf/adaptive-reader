"""Request patterns for single-token decisions at the token-id level: the chat-turn ids, block padding, one-token
answers read from the logits, grammar-forced generation that keeps its ids as history, and parallel branches from one
cached point. Nothing here knows what is being asked: callers pass the ids and the options.

Every request goes through Client.send(), which hands the prompt, the ids the model computed and the server's usage to
an optional observer (the ledger), so cost is checked on every request, not estimated.
"""
import concurrent.futures as cf
import math
import os
import time

import cpm

NL = "\n"
SWITCH = "<|im_end|>\n<|im_start|>user\n"                                      # end of a turn, start of a user turn
HEADER = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"           # end of the user turn, assistant answers
STOP_STRINGS = ("<|im_end|>", "</s>", "<|endoftext|>")
MAX_OPTION_IDS = 128                                                           # the server's logprob_token_ids limit


class Client:
    def __init__(self, observer=None, workers=None):
        self.observer = observer
        self.workers = workers or int(os.getenv("CPM_WORKERS", "4"))
        self.nl = cpm.enc(NL)
        assert len(self.nl) == 1, "newline is not one token"
        self.nl = self.nl[0]
        self.space = cpm.enc(" 0")[0]                     # this tokenizer writes " 70" as a space token + "70"
        assert cpm.enc(" 70") == [self.space] + cpm.enc("70"), "numbers are not written as space + number"
        self.switch, self.header = cpm.enc(SWITCH), cpm.enc(HEADER)
        # stop tokens (only those that are one special token each) are not history
        self.special = {x[0] for x in (cpm.enc(s) for s in STOP_STRINGS) if len(x) == 1}
        assert self.switch[0] in self.special and self.header[0] in self.special

    # ---- ids --------------------------------------------------------------------------------------------------------
    def assert_template(self, messages):
        """Appending turns by hand at the id level must equal the server's own rendering of the continued conversation."""
        mine = cpm.chat_ids(messages) + cpm.enc("NAMES") + self.switch + cpm.enc("Q") + self.header
        ref = cpm.chat_ids(messages + [{"role": "assistant", "content": "NAMES"}, {"role": "user", "content": "Q"}])
        assert mine == ref, "template assembly differs from the chat template"

    def open_user_turn(self, messages):
        """Ids of a conversation whose LAST message is a user turn left open (no end-of-turn), so text can be appended
        to it: the rendering up to and including that message's content."""
        assert messages[-1]["role"] == "user"
        a = cpm.chat_ids(messages[:-1] + [dict(messages[-1], content="\x01")])
        b = cpm.chat_ids(messages[:-1] + [dict(messages[-1], content="\x02")])
        n = cpm.common(a, b)
        return a[:n] + cpm.enc(messages[-1]["content"])

    def pad_to(self, ids, tail=0):
        """Newlines so that ids + pad + (tail tokens) ends on a block edge."""
        return ids + [self.nl] * ((-(len(ids) + tail)) % cpm.BLOCK)

    def pad_len(self, n, tail=0):
        return (-(n + tail)) % cpm.BLOCK

    # ---- options ----------------------------------------------------------------------------------------------------
    def words(self, words):
        """Option groups for words, each read in both spellings (" keep" + " Keep"); every option must be one token in
        its first spelling, and a second spelling that is not one token is left out."""
        groups = []
        for w in words:
            spellings = [" " + w] + ([" " + w.capitalize()] if w.capitalize() != w else [])
            ids = [cpm.enc(s) for s in spellings]
            assert len(ids[0]) == 1, f"option {w!r} is not one token"
            groups.append([x[0] for x in ids if len(x) == 1])
        return Options(groups)

    def numbers(self, values):
        """Option groups for numbers read AFTER the space token (the prompt must end with self.space)."""
        groups = []
        for v in values:
            ids = cpm.enc(str(v))
            assert len(ids) == 1, f"number {v} is not one token"
            groups.append(ids)
        return Options(groups)

    # ---- requests ---------------------------------------------------------------------------------------------------
    def send(self, prompt, body, tag):
        """One completion request; the observer sees the prompt, the ids the model computed and the usage."""
        t0 = time.time()
        out = cpm.post("/v1/completions", dict(body, model=cpm.MODEL, prompt=prompt))
        out["_secs"] = time.time() - t0
        if self.observer:
            self.observer(prompt, out, tag)
        return out

    def ask(self, prompt, options, tag="ask"):
        """One single-token answer: max_tokens=1, the option tokens' probabilities, normalised over the options.
        Returns (distribution, mass on the options, the request's output)."""
        out = self.send(prompt, {"max_tokens": 1, "temperature": 0, "logprobs": 0,
                                 "logprob_token_ids": options.ids, "return_tokens_as_token_ids": True}, tag)
        top = out["choices"][0]["logprobs"]["top_logprobs"][0]
        missing = [i for i in options.ids if f"token_id:{i}" not in top]
        if missing:
            raise RuntimeError(f"logprob ids missing: {missing}")
        p = [sum(math.exp(top[f"token_id:{i}"]) for i in g) for g in options.groups]
        s = sum(p)
        return [x / s for x in p], s, out

    def generate(self, prompt, max_tokens, regex=None, tag="generate"):
        """Free or grammar-forced generation (the grammar only for generated text, never on answers).
        Returns (generated ids without the stop token, text, finish reason, the request's output)."""
        body = {"max_tokens": max_tokens, "temperature": 0, "logprobs": 0, "return_tokens_as_token_ids": True}
        if regex:
            body["structured_outputs"] = {"regex": regex}
        out = self.send(prompt, body, tag)
        c = out["choices"][0]
        gen = [int(t.split(":")[1]) for t in c["logprobs"]["tokens"]]
        while gen and gen[-1] in self.special:
            gen.pop()
        return gen, c["text"], c["finish_reason"], out

    def branches(self, base, jobs):
        """jobs = [(suffix ids, options, tag)]. The first is sent alone so the server caches the base; the rest go out
        in parallel. Returns [(distribution, mass, output)] in job order."""
        if not jobs:
            return []
        res = [self.ask(base + jobs[0][0], jobs[0][1], jobs[0][2])]
        if len(jobs) > 1:
            with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
                res += list(ex.map(lambda j: self.ask(base + j[0], j[1], j[2]), jobs[1:]))
        return res


class Options:
    """Option groups: each option is a list of single-token ids whose probabilities add up."""

    def __init__(self, groups):
        self.groups = groups
        self.ids = sorted({i for g in groups for i in g})
        assert len(self.ids) <= MAX_OPTION_IDS, "too many option ids for one request"


def expected(dist, values):
    return sum(p * v for p, v in zip(dist, values))


def argmax(dist):
    return max(range(len(dist)), key=dist.__getitem__)
