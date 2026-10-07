"""Offline tests for the corpus builder, with a fake byte-level tokenizer (no server). Live checks run only with
CPM_LIVE=1."""
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import corpus                                                                  # noqa: E402
from adapters import markdown                                                  # noqa: E402

_PIECES = re.compile(r"[^\s]+\n?| +|\n")


def fake_tokenize(text):
    """Words (with a trailing newline merged in), runs of spaces, newlines; multi-byte characters split in two tokens
    so a token can end inside a character, like a real byte-level tokenizer."""
    ids, strs = [], []
    for m in _PIECES.finditer(text):
        b = m.group(0).encode("utf-8")
        parts = [b] if len(b) == len(m.group(0)) else [b[:1], b[1:]]
        for p in parts:
            strs.append("".join(corpus.BYTE_ENCODER[x] for x in p))
            ids.append(zlib.crc32(p))
    return ids, strs


def doc_from(text, doc_id="d", title="T"):
    return markdown.from_text(text, doc_id, title=title)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


class TestPieces(unittest.TestCase):
    cfg = dict(corpus.CONFIG, min_tokens=40, max_tokens=80, target_tokens=60, long_line=30, hard_cut=50)

    def build(self, text):
        tok = corpus.Tokenizer(fake_tokenize)
        d = doc_from(text)
        ps = corpus.build_doc(d, tok, 1, self.cfg)
        corpus.validate(d, ps, tok)
        return d, ps

    def test_lossless_tiling_and_unit_boundaries(self):
        text = "".join(f"## Head {k}\n\n" + "\n\n".join(words(7, f"s{k}p{j}") + "." for j in range(4)) + "\n\n"
                       for k in range(5))
        d, ps = self.build(text)
        self.assertGreater(len(ps), 2)
        for p in ps[:-1]:                      # every cut is at a unit boundary: the piece text ends with a newline
            self.assertTrue(d.text[p["char"][0]:p["char"][1]].endswith("\n"))
            self.assertLessEqual(p["n_tokens"], self.cfg["max_tokens"])
        self.assertEqual(ps[-1]["char"][1], len(d.text))

    def test_prefers_a_heading_boundary(self):
        text = "## A\n" + "\n".join(words(6, f"a{i}") for i in range(7)) + "\n## B\n" + \
               "\n".join(words(6, f"b{i}") for i in range(12)) + "\n"
        d, ps = self.build(text)
        starts = [d.text[p["char"][0]:p["char"][0] + 4] for p in ps]
        self.assertIn("## B", starts)

    def test_never_cuts_right_after_a_heading(self):
        text = "\n".join(f"## H{i}\n" + words(12, f"x{i}") for i in range(12)) + "\n"
        d, ps = self.build(text)
        for p in ps:
            self.assertFalse(d.text[p["char"][0]:p["char"][1]].rstrip("\n").endswith(tuple(f"H{i}" for i in range(12))))

    def test_table_continuation_repeats_header(self):
        rows = "\n".join(f"| r{i} | {i} | {i * 2} |" for i in range(60))
        text = "## T\n\n| name | x | y |\n|---|---|---|\n" + rows + "\n\nafter.\n"
        tok = corpus.Tokenizer(fake_tokenize)
        d = doc_from(text)
        ps = corpus.build_doc(d, tok, 1, self.cfg)
        corpus.validate(d, ps, tok)
        cont = [p for p in ps if p["n_repeat"]]
        self.assertTrue(cont)
        head_ids = fake_tokenize("| name | x | y |\n")[0]
        for p in cont:
            self.assertEqual(p["ids"][:p["n_repeat"]], head_ids)

    def test_long_line_split_and_hard_cut(self):
        text = "## L\n" + " ".join(f"s{i} word word." for i in range(40)) + "\n" + words(120, "z") + "\n"
        d, ps = self.build(text)
        kinds = {u[1] for p in ps for u in p["units"]}
        self.assertIn("sentence", kinds)

    def test_non_ascii_and_no_headings(self):
        text = "\n".join("Größe € naïve 日本 " + words(5, f"u{i}") for i in range(40)) + "\n"
        d, ps = self.build(text)
        self.assertEqual(ps[0]["path"], [])

    def test_deterministic(self):
        text = "## A\n" + "\n".join(words(9, f"d{i}") for i in range(40)) + "\n"
        self.assertEqual([p["sha"] for p in self.build(text)[1]], [p["sha"] for p in self.build(text)[1]])

    def test_heading_path_by_depth(self):
        text = "# Top\n## Sub\ntext\n### Deep\nmore\n## Sub2\nend\n"
        d = doc_from(text)
        self.assertEqual(corpus.heading_path(d, d.text.index("more")), ["Top", "Sub", "Deep"])
        self.assertEqual(corpus.heading_path(d, d.text.index("end")), ["Top", "Sub2"])


class TestNormaliseAndSlice(unittest.TestCase):
    def test_equal_adjacent_cells_are_kept(self):
        d = doc_from("| a | (10) | (10) |\n|---|---|---|\n| b | 1 | 1 |\n")
        self.assertIn("(10) | (10)", d.text)
        self.assertIn("| b | 1 | 1 |", d.text)
        self.assertNotIn("---", d.text)

    def test_slice_keeps_front_matter_meta(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "f.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("---\ncompany: \"Acme\"\nperiod: \"FY 1\"\n---\nl1\nl2\nl3\nl4\n")
            d = markdown.parse(path, {"lines": [5, 6]})                   # file lines 5-6: after the front matter
            self.assertEqual(d.text, "l1\nl2\n")
            self.assertIn("Acme", d.title)
        finally:
            shutil.rmtree(tmp)


@unittest.skipUnless(os.getenv("CPM_LIVE") == "1", "live server tests: set CPM_LIVE=1")
class TestLive(unittest.TestCase):
    def test_server_tokens_rebuild_text(self):
        tok = corpus.Tokenizer()
        text = "## Revenue Trends\n\n| Industrial | $ 4,779,466 | 50 % |\nNaïve “quotes” — 日本.\n"
        ids, bs = tok.tokenize(text)
        self.assertEqual(tok.decode(ids), text)
        spans = corpus.char_spans(text, bs)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(text))


if __name__ == "__main__":
    unittest.main()
