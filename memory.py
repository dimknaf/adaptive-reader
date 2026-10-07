"""The working memory M: one growing token-id prefix, shaped as a stack of scopes, always ending on a block edge.

Why a stack: the server's cache is a prefix. Going back to any earlier point of M is free (that prefix is cached), and
appending pays only for the new tokens, but removing something from the middle re-computes everything after it. So:
  - a piece kept in full is committed as the prefix of the prompt it was read in (already computed: free);
  - a scope (a section of the document) pushes its header when it opens, and at close the header is rolled back for
    free if nothing inside it was kept;
  - removing an item already kept is a decision of the reader, never a size rule; it re-computes every item after it,
    and that re-read is returned and counted.
M's size is whatever the decisions keep: it is measured, never set. Every segment of M (root, scope header, piece, units,
note) is a whole number of blocks, so any cut between segments is a block edge. Pure code: no server calls; the ledger measures what each change costs.
"""
BLOCK = 16


class Item:
    __slots__ = ("kind", "key", "ids", "score", "parent", "closed")

    def __init__(self, kind, key, ids, score=None, parent=None):
        assert len(ids) % BLOCK == 0, f"{kind} {key}: {len(ids)} ids is not a whole number of blocks"
        self.kind, self.key, self.ids, self.score, self.parent, self.closed = kind, key, list(ids), score, parent, False


class Memory:
    def __init__(self, root_ids):
        self.items = [Item("root", "root", root_ids)]
        self.stack = []                                   # open scope items, outermost first
        self.rereads = 0                                  # tokens computed again after a removal (the re-read cost)

    # ---- views ------------------------------------------------------------------------------------------------------
    @property
    def ids(self):
        return [t for it in self.items for t in it.ids]

    def __len__(self):
        return sum(len(it.ids) for it in self.items)

    def offsets(self):
        """[(item, start, end)] in M."""
        out, n = [], 0
        for it in self.items:
            out.append((it, n, n + len(it.ids)))
            n += len(it.ids)
        return out

    def scope(self):
        return self.stack[-1] if self.stack else None

    def children(self, scope):
        return [it for it in self.items if it.parent is scope]

    # ---- scopes -----------------------------------------------------------------------------------------------------
    def open_scope(self, key, header_ids):
        it = Item("scope", key, header_ids, parent=self.scope())
        self.items.append(it)
        self.stack.append(it)
        return it

    def close_scope(self):
        """Close the innermost scope; its header is rolled back (free) when nothing inside it was kept."""
        it = self.stack.pop()
        it.closed = True
        if self.items[-1] is it:                           # nothing after the header: M shrinks back to a cached prefix
            self.items.pop()
            return False
        return True

    # ---- commits ----------------------------------------------------------------------------------------------------
    def keep_prefix(self, prompt, k, key, score=None):
        """Keep in full: M becomes prompt[:k], a prefix of a prompt the server already computed (free)."""
        m = self.ids
        assert k % BLOCK == 0 and k > len(m) and list(prompt[:len(m)]) == m, "keep point is not an aligned extension of M"
        it = Item("piece", key, prompt[len(m):k], score, self.scope())
        self.items.append(it)
        return it

    def append(self, kind, key, ids, score=None):
        """Append new text (part of a piece, or a note): the next request pays for these ids once."""
        it = Item(kind, key, ids, score, self.scope())
        self.items.append(it)
        return it

    # ---- removal ----------------------------------------------------------------------------------------------------
    def remove(self, item):
        """Remove an item the reader decided is no longer needed. Everything after it is computed again by the next
        request: that re-read is returned and counted. A closed scope left with nothing under it goes too."""
        i = self.items.index(item)
        assert item.kind not in ("root",) and item not in self.stack, "the root and open scopes cannot be removed"
        gone = {i}
        parent = item.parent
        if parent is not None and parent.closed and all(c is item for c in self.children(parent)):
            gone.add(self.items.index(parent))
        first = min(gone)
        reread = sum(len(it.ids) for j, it in enumerate(self.items) if j > first and j not in gone)
        self.items = [it for j, it in enumerate(self.items) if j not in gone]
        self.rereads += reread
        return reread
