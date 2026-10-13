#!/usr/bin/env python3
"""
Find sections shared between Matthew, Mark, and Luke (MorphGNT format).

Definition of a match:
  Two windows (from different books) match if they share at least THRESHOLD
  words, counted as a multiset intersection (a word repeated twice in both
  windows counts twice; once in either counts once). Word order is ignored.

Window size:
  By default (CONTENT_WINDOW = True, with CONTENT_ONLY = True) a window is
  WINDOW *non-excluded* words. Excluded words (articles, conjunctions,
  prepositions, particles, pronouns) are skipped when counting, so a window
  may span more than WINDOW raw words (about 2x WINDOW if half the words are
  excluded). With --no-content-window a window is WINDOW consecutive raw
  words, and excluded words occupy slots but can never match.

Grouping:
  1. Matching window pairs for each pair of books are grouped when they sit
     close together in BOTH books, so unrelated matches elsewhere are never
     chained in. Sections whose passages overlap in both books are merged.
  2. A passage that matches two different books appears once in each pair's
     report, tagged '[also shares text with X]'.

Usage:
  python synoptic_overlap.py --matthew PATH --mark PATH --luke PATH
                             [--out PATH] [--dir DIR] [--window 10]
                             [--threshold 5] [--field norm|lemma]
                             [--content-only | --no-content-only]
                             [--content-window | --no-content-window]
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# SETTINGS: edit these, then press Run in VS Code. No terminal needed.
# Use raw strings (r"...") for Windows paths.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = os.path.join(SCRIPT_DIR, "..", "Text", "sblgnt")  # relative to this script

MATTHEW_PATH = os.path.join(TEXT_DIR, "61-Mt-morphgnt.txt")
MARK_PATH = os.path.join(TEXT_DIR, "62-Mk-morphgnt.txt")
LUKE_PATH = os.path.join(TEXT_DIR, "63-Lk-morphgnt.txt")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "synoptic_overlaps.txt")  # saved next to this script

WINDOW = 10            # words per block (see CONTENT_WINDOW)
THRESHOLD = 7          # shared words required within a block
FIELD = "lemma"        # "lemma" (dictionary form) or "norm" (normalized form)
CONTENT_ONLY = True    # True ignores articles, conjunctions, prepositions, etc.
CONTENT_WINDOW = True  # True: a window is WINDOW non-excluded words, so it may
                       # span more raw words. False: WINDOW consecutive raw words.
                       # Only has an effect when CONTENT_ONLY is True.
# ---------------------------------------------------------------------------

BOOKS = [
    ("Matthew", "61-Mt-morphgnt.txt"),
    ("Mark", "62-Mk-morphgnt.txt"),
    ("Luke", "63-Lk-morphgnt.txt"),
]
# MorphGNT fields: ref pos parse text word norm lemma
FIELD_INDEX = {"norm": 5, "lemma": 6}
# Articles (RA), conjunctions (C-), prepositions (P-), particles (X-), and
# pronouns: personal incl. reflexive (RP), demonstrative (RD),
# interrogative/indefinite (RI), relative (RR)
FUNCTION_POS = {"RA", "C-", "P-", "X-", "RP", "RD", "RI", "RR"}


def load(path, field_idx, content_only, vocab):
    refs, texts, ids = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 7:
                continue
            ref = "%d:%d" % (int(p[0][2:4]), int(p[0][4:6]))
            refs.append(ref)
            texts.append(p[3])
            if content_only and p[1] in FUNCTION_POS:
                ids.append(-1)  # excluded: can never match
            else:
                ids.append(vocab.setdefault(p[field_idx], len(vocab)))
    return refs, texts, np.array(ids, dtype=np.int32)


def window_overlap(a_slice, b_slice):
    ca = Counter(x for x in a_slice if x >= 0)
    cb = Counter(x for x in b_slice if x >= 0)
    return sum(min(n, cb[w]) for w, n in ca.items() if w in cb)


def find_pairs(a, b, win, thr):
    """Return list of (i, j) where a[i:i+win] and b[j:j+win] share >= thr words."""
    na, nb = len(a), len(b)
    if na < win or nb < win:
        return []
    zeros = np.zeros(nb, dtype=np.int16)

    def eq(k):
        return (b == a[k]).astype(np.int16) if a[k] >= 0 else zeros

    s = np.zeros(nb, dtype=np.int16)
    for k in range(win):
        s += eq(k)
    pairs = []
    for i in range(na - win + 1):
        c = np.concatenate(([0], np.cumsum(s, dtype=np.int32)))
        w = c[win:] - c[:-win]  # pair-count upper bound for each j
        for j in np.nonzero(w >= thr)[0]:
            # exact multiset check (pair-count overcounts repeated words)
            if window_overlap(a[i:i + win], b[j:j + win]) >= thr:
                pairs.append((i, int(j)))
        if i + win < na:
            s += eq(i + win)
            s -= eq(i)
    return pairs


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x, y):
        self.p[self.find(x)] = self.find(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matthew", default=MATTHEW_PATH, help="path to the Matthew file")
    ap.add_argument("--mark", default=MARK_PATH, help="path to the Mark file")
    ap.add_argument("--luke", default=LUKE_PATH, help="path to the Luke file")
    ap.add_argument("--dir", default=".",
                    help="fallback directory for any book path not given "
                         "explicitly (default: current directory)")
    ap.add_argument("--out", default=OUTPUT_PATH,
                    help="path of the output file")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--threshold", type=int, default=THRESHOLD)
    ap.add_argument("--field", choices=FIELD_INDEX, default=FIELD,
                    help="compare normalized forms (norm) or lemmas "
                         "(default: %s)" % FIELD)
    ap.add_argument("--content-only", action=argparse.BooleanOptionalAction, default=CONTENT_ONLY,
                    help="ignore articles, conjunctions, prepositions, "
                         "particles, and pronouns when counting matches")
    ap.add_argument("--content-window", action=argparse.BooleanOptionalAction, default=CONTENT_WINDOW,
                    help="make each window WINDOW non-excluded words, so it may "
                         "span more raw words (needs --content-only; on by default)")
    args, _ = ap.parse_known_args()

    if args.content_window and not args.content_only:
        print("Note: --content-window has no effect without --content-only "
              "(no words are excluded).")
    use_cw = args.content_window and args.content_only

    vocab = {}
    data = {}   # name -> (refs, texts, ids) in original word positions
    mids = {}   # name -> ids used for matching (excluded words removed if use_cw)
    cmap = {}   # name -> position in the original text of each matching word
    explicit = {"Matthew": args.matthew, "Mark": args.mark, "Luke": args.luke}
    for name, fn in BOOKS:
        path = explicit[name] or os.path.join(args.dir, fn)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            sys.exit("Missing file: " + path)
        data[name] = load(path, FIELD_INDEX[args.field], args.content_only, vocab)
        ids = data[name][2]
        if use_cw:
            keep = np.nonzero(ids >= 0)[0]
            mids[name], cmap[name] = ids[keep], keep
        else:
            mids[name], cmap[name] = ids, np.arange(len(ids))
        print("Loaded %s: %d words (%d used for matching)"
              % (name, len(ids), len(mids[name])))

    names = [n for n, _ in BOOKS]
    pair_names = [(names[x], names[y]) for x in range(len(names))
                  for y in range(x + 1, len(names))]

    # Pass 1: matching window pairs and per-book coverage, for each book pair.
    # Windows are found in matching-word space; coverage is recorded over the
    # original text, so excluded words inside a window count as covered.
    window_pairs = {}
    covs = {}  # (book, other_book) -> words of `book` covered by matches with `other_book`
    last = args.window - 1
    for A, B in pair_names:
        print("Comparing %s and %s ..." % (A, B))
        pairs = find_pairs(mids[A], mids[B], args.window, args.threshold)
        print("  %d matching window pairs" % len(pairs))
        window_pairs[(A, B)] = pairs
        covs[(A, B)] = np.zeros(len(data[A][2]), dtype=bool)
        covs[(B, A)] = np.zeros(len(data[B][2]), dtype=bool)
        for i, j in pairs:
            covs[(A, B)][cmap[A][i]:cmap[A][i + last] + 1] = True
            covs[(B, A)][cmap[B][j]:cmap[B][j + last] + 1] = True

    APPARATUS = re.compile('[\u2e00-\u2e7f]')

    def describe(book, span, partner):
        refs, texts, _ = data[book]
        st, en = span
        rng = refs[st] if refs[st] == refs[en] else "%s-%s" % (refs[st], refs[en])
        also = [o for o in names if o not in (book, partner) and (book, o) in covs
                and covs[(book, o)][st:en + 1].any()]
        return rng, APPARATUS.sub("", " ".join(texts[st:en + 1])), also

    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    total = 0
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("Window: %d %s | Threshold: %d shared words | Compared on: %s%s\n"
                  % (args.window,
                     "non-excluded words (may span more raw words)" if use_cw else "raw words",
                     args.threshold, args.field,
                     " (content words only)" if args.content_only else ""))
        out.write("Each section pairs the passages that matched each other.\n")
        out.write("'[also shares text with X]' means part of that passage also "
                  "matches the third book X in a separate comparison.\n\n")

        # Pass 2: group matching window pairs into local alignments. Two window
        # pairs belong together only if they sit close together in BOTH books,
        # so unrelated matches elsewhere can never be chained in.
        for A, B in pair_names:
            pairs = set(window_pairs[(A, B)])
            dsu = DSU()
            for (i, j) in pairs:
                dsu.find((i, j))
                for di in range(0, 3):
                    for dj in range(-2, 3):
                        if (di or dj > 0) and (i + di, j + dj) in pairs:
                            dsu.union((i, j), (i + di, j + dj))
            groups = defaultdict(list)
            for p in pairs:
                groups[dsu.find(p)].append(p)
            sections = []
            for members in groups.values():
                a0 = min(i for i, _ in members)
                a1 = max(i for i, _ in members) + last
                b0 = min(j for _, j in members)
                b1 = max(j for _, j in members) + last
                sections.append(((a0, a1), (b0, b1)))
            # Merge sections whose passages overlap in BOTH books (repeated
            # phrases can otherwise split one parallel into overlapping pieces).
            merged = True
            while merged:
                merged = False
                sections.sort()
                for x in range(len(sections)):
                    for y in range(x + 1, len(sections)):
                        (a0, a1), (b0, b1) = sections[x]
                        (c0, c1), (d0, d1) = sections[y]
                        if c0 > a1:
                            break
                        if c0 <= a1 and d0 <= b1 and b0 <= d1:
                            sections[x] = ((min(a0, c0), max(a1, c1)),
                                           (min(b0, d0), max(b1, d1)))
                            del sections[y]
                            merged = True
                            break
                    if merged:
                        break
            total += len(sections)

            out.write("#" * 70 + "\n")
            out.write("# %s vs %s: %d sections\n" % (A, B, len(sections)))
            out.write("#" * 70 + "\n\n")
            for n_, (sa, sb) in enumerate(sections, 1):
                out.write("=== %s vs %s, section %d ===\n" % (A, B, n_))
                for book, mspan, partner in ((A, sa, B), (B, sb, A)):
                    # convert from matching-word positions to original positions
                    span = (int(cmap[book][mspan[0]]), int(cmap[book][mspan[1]]))
                    rng, text, also = describe(book, span, partner)
                    tag = "  [also shares text with %s]" % ", ".join(also) if also else ""
                    if use_cw:
                        counts = "%d words, %d non-excluded" % (span[1] - span[0] + 1,
                                                                mspan[1] - mspan[0] + 1)
                    else:
                        counts = "%d words" % (span[1] - span[0] + 1)
                    out.write("%s %s (%s) @%d-%d%s:\n  %s\n"
                              % (book, rng, counts, span[0], span[1], tag, text))
                out.write("\n")
    print("Wrote %d sections to %s" % (total, out_path))


if __name__ == "__main__":
    main()