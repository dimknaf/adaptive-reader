"""Offline tests for the stack memory and the prefix-cache ledger (no server)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ledger import Ledger, BLOCK                                               # noqa: E402
from memory import Memory                                                     # noqa: E402


def blk(tag, n=1):
    """n blocks of distinct ids tagged by a number."""
    return [tag * 1000 + i for i in range(n * BLOCK)]


def out(sampled, cached=None, prompt=None):
    u = {"completion_tokens": len(sampled)}
    if prompt is not None:
        u["prompt_tokens"] = prompt
    if cached is not None:
        u["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"choices": [{"logprobs": {"tokens": [f"token_id:{t}" for t in sampled]}}], "usage": u}


class TestLedger(unittest.TestCase):
    def test_hits_whole_blocks_and_never_the_last_token(self):
        L = Ledger()
        p = blk(1, 3)                                     # 48 ids, aligned
        self.assertEqual(L.predict(p), 0)
        L(p, out([5]), "warm")
        # the same aligned prompt again: its whole last block is recomputed
        self.assertEqual(L.predict(p), 32)
        # extended by a partial block: all three blocks hit
        self.assertEqual(L.predict(p + [7, 8, 9]), 48)
        # a prompt that differs inside block 2 hits only block 1
        q = list(p)
        q[20] = -1
        self.assertEqual(L.predict(q + [1]), 16)

    def test_generated_tokens_are_cached_except_the_last(self):
        L = Ledger()
        p = blk(1, 1)
        gen = list(range(500, 517))                       # 17 sampled: 16 fed back, the 17th never computed
        L(p, out(gen), "gen")
        self.assertEqual(L.predict(p + gen[:16] + [1]), 32)
        L2 = Ledger()
        L2(p, out(gen[:16]), "gen")                       # 16 sampled: only 15 fed, the second block is partial
        self.assertEqual(L2.predict(p + gen[:16] + [1]), 16)

    def test_mismatch_is_recorded_not_absorbed(self):
        L = Ledger()
        p = blk(1, 2)
        L(p + [1], out([2], cached=0, prompt=33), "a")
        L(p + [3], out([2], cached=0, prompt=33), "b")    # the server says 0 (e.g. an eviction): we predicted 32
        self.assertEqual(L.mismatches, [{"tag": "b", "prompt": 33, "predicted": 32, "server": 0}])

    def test_totals_and_weight(self):
        L = Ledger()
        L(blk(1, 1) + [1], out([9]), "read")
        L(blk(1, 1) + [2, 3], out([9]), "branch")
        t = L.total()
        self.assertEqual((t["uncached"], t["cached"], t["output"]), (17 + 2, 16, 2))
        self.assertEqual(t["weighted"], 19 + 4 * 2)


class TestMemory(unittest.TestCase):
    def test_keep_prefix_is_an_extension_and_aligned(self):
        m = Memory(blk(1))
        prompt = m.ids + blk(2, 2) + [7, 7, 7]
        m.keep_prefix(prompt, len(m) + 2 * BLOCK, "1.1", 80)
        self.assertEqual(m.ids, prompt[:3 * BLOCK])
        with self.assertRaises(AssertionError):
            m.keep_prefix([0] + prompt, 4 * BLOCK, "x")          # not an extension of M
        with self.assertRaises(AssertionError):
            m.append("units", "x", [1, 2, 3])                    # not whole blocks

    def test_empty_scope_rolls_back_free(self):
        m = Memory(blk(1))
        root = m.ids
        m.open_scope("s", blk(2))
        self.assertFalse(m.close_scope())
        self.assertEqual(m.ids, root)
        m.open_scope("s", blk(2))
        m.append("units", "1.1", blk(3), 50)
        self.assertTrue(m.close_scope())
        self.assertEqual(len(m), 3 * BLOCK)

    def test_remove_counts_the_reread_of_everything_after(self):
        m = Memory(blk(1))
        m.open_scope("A", blk(2))
        a1 = m.append("piece", "1.1", blk(3, 2), 90)
        a2 = m.append("piece", "1.2", blk(4, 2), 10)
        m.close_scope()
        m.open_scope("B", blk(5))
        m.append("piece", "2.1", blk(6, 2), 60)
        self.assertEqual(m.remove(a2), 3 * BLOCK)                  # scope B header + 2.1 are computed again
        self.assertEqual(m.remove(a1), 3 * BLOCK)                  # A is closed and now empty: its header goes too
        self.assertEqual([it.key for it in m.items], ["root", "B", "2.1"])
        self.assertEqual(m.rereads, 6 * BLOCK)

    def test_remove_the_last_item_is_free(self):
        m = Memory(blk(1))
        m.open_scope("A", blk(2))
        last = m.append("piece", "1.1", blk(3), 50)
        self.assertEqual(m.remove(last), 0)
        with self.assertRaises(AssertionError):
            m.remove(m.stack[-1])                                  # an open scope cannot be removed

if __name__ == "__main__":
    unittest.main()
