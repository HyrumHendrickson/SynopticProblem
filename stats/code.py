#!/usr/bin/env python3
"""
Statistics on the overlap file produced by synoptic_overlap.py.

Inputs
  - The overlap file (must come from the current synoptic_overlap.py, whose
    passage lines carry exact word positions such as "@925-944"). Both line
    formats the overlap script can write are accepted:
        Matthew 2:13 (19 words) @667-685:
        Matthew 2:13 (19 words, 10 non-excluded) @667-685:
    The second appears when the overlap script's content-window option is on
    (its default). The overlap file's first line records the settings used,
    including whether word order was ignored; it is echoed in the report.
  - The three MorphGNT files, used only to get each book's total words,
    chapters, and verses so percentages have a denominator.

Edit the SETTINGS block below and press Run in VS Code, or pass the same
settings as command-line options (see --help).

Charts
  PNG images are saved to MEDIA_DIR. They need matplotlib (pip install matplotlib).
  21 images are written. File names keep their original numbers (01-18, 20-22);
  number 19 (the former radar chart) is no longer produced.

  Every chart adapts to the data, so any window and threshold works: section
  length bins are chosen from the observed range, length axes switch to a log
  scale when lengths span a wide range, and marker size and transparency
  shrink as the number of sections grows.

Terminology
  "Covered" words are words of a book that fall inside any overlap section.
  Coverage is counted by unique word position, so a word inside several
  sections is never counted twice.
  All percentages are in raw words (including articles, conjunctions, etc.),
  since the denominators are the full books. A section's word count is the
  raw span, so it includes any excluded words (articles, etc.) inside it.
"""
import argparse
import csv
import os
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# SETTINGS: edit these paths, then press Run. Use raw strings (r"...").
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = os.path.join(SCRIPT_DIR, "..", "Text", "sblgnt")

# synoptic_overlap.py saves its output next to itself by default. If that file
# is not found, the "overlap" folder beside this script's parent is tried.
OVERLAP_PATH = os.path.join(SCRIPT_DIR, "synoptic_overlaps.txt")   # change filename if needed
FALLBACK_OVERLAP_PATH = os.path.join(SCRIPT_DIR, "..", "overlap", "synoptic_overlaps.txt")
MATTHEW_PATH = os.path.join(TEXT_DIR, "61-Mt-morphgnt.txt")
MARK_PATH = os.path.join(TEXT_DIR, "62-Mk-morphgnt.txt")
LUKE_PATH = os.path.join(TEXT_DIR, "63-Lk-morphgnt.txt")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "synoptic_stats.txt")       # the report, next to this script
SECTIONS_CSV_PATH = ""                                              # optional per-section CSV; "" = off
MEDIA_DIR = os.path.join(SCRIPT_DIR, "media")                       # folder for the chart images
MAKE_IMAGES = True                                                  # False skips the charts (matplotlib not needed)
# ---------------------------------------------------------------------------

BOOKS = ["Matthew", "Mark", "Luke"]
PAIRS = [("Matthew", "Mark"), ("Matthew", "Luke"), ("Mark", "Luke")]
ABBR = {"Matthew": "Mt", "Mark": "Mk", "Luke": "Lk"}

HEADER_RE = re.compile(r"^=== (\w+) vs (\w+), section (\d+) ===$")
# Group 4 (non-excluded word count) is present only when the overlap script
# ran with its content-window option on.
PASSAGE_RE = re.compile(
    r"^(Matthew|Mark|Luke) (\S+) \((\d+) words(?:, (\d+) non-excluded)?\) @(\d+)-(\d+)"
    r"(?:\s+\[also shares text with ([^\]]+)\])?:$")
OLD_PASSAGE_RE = re.compile(r"^(Matthew|Mark|Luke) (\S+) \((\d+) words[,)]")


# ----------------------------------------------------------------- loading
def read_settings_line(path):
    """First line of the overlap file ('Window: ... | Threshold: ...'), or ''."""
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip()
    return first if first.startswith("Window:") else ""


def parse_overlap(path):
    sections = []
    cur = None
    pending = None
    old_format = False
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = HEADER_RE.match(line)
            if m:
                cur = {"a": m.group(1), "b": m.group(2),
                       "n": int(m.group(3)), "passages": []}
                sections.append(cur)
                pending = None
                continue
            m = PASSAGE_RE.match(line)
            if m and cur is not None:
                pending = {"book": m.group(1), "ref": m.group(2),
                           "words": int(m.group(3)),
                           "content": int(m.group(4)) if m.group(4) else None,
                           "start": int(m.group(5)), "end": int(m.group(6)),
                           "text": ""}
                cur["passages"].append(pending)
                continue
            if cur is not None and OLD_PASSAGE_RE.match(line):
                old_format = True
            if pending is not None and line.startswith("  "):
                pending["text"] = line.strip()
                pending = None
    if old_format:
        sys.exit("The overlap file lacks word positions. Re-run the current "
                 "synoptic_overlap.py to regenerate it, then run this again.")
    good = [s for s in sections if len(s["passages"]) == 2]
    if not good:
        sys.exit("No sections could be read from: " + path)
    for s in good:
        if (s["passages"][0]["book"] != s["a"]
                or s["passages"][1]["book"] != s["b"]):
            sys.exit("Unexpected passage order in section %s vs %s #%d"
                     % (s["a"], s["b"], s["n"]))
    return good


def load_book(path):
    chap, verse = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 7:  # same filter the overlap program uses
                continue
            chap.append(int(p[0][2:4]))
            verse.append(int(p[0][4:6]))
    return chap, verse


# ----------------------------------------------------------------- helpers
def pct(x, y):
    return 100.0 * x / y if y else 0.0


def bar(p, width=30):
    return "#" * int(round(width * p / 100.0))


def norm_words(text):
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = "".join(c if c.isalpha() else " " for c in t)
    return t.split()


def lexical_overlap(wa, wb):
    from collections import Counter
    ca, cb = Counter(wa), Counter(wb)
    shared = sum(min(n, cb[w]) for w, n in ca.items() if w in cb)
    return pct(shared, min(len(wa), len(wb)))


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


# Candidate bin edges, from finest to coarsest. The first set that gives no
# more than max_bins bins over the observed range is used.
NICE_CUT_SETS = [
    [6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000],
    [10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 5000],
    [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
    [10, 100, 1000, 10000, 100000],
]


def length_bins(lengths, max_bins=9):
    """Bins for section lengths chosen from the observed range.

    Returns (bins, names): bins are inclusive (lo, hi) pairs that together
    cover every length from min(lengths) to max(lengths)."""
    if not lengths:
        return [(0, 0)], ["-"]
    lo, hi = min(lengths), max(lengths)
    for cand in NICE_CUT_SETS:
        cuts = [c for c in cand if lo < c <= hi]
        if len(cuts) + 1 <= max_bins:
            break
    while len(cuts) + 1 > max_bins:
        cuts = cuts[1::2]
    edges = [lo] + cuts
    bins = [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)] + [(edges[-1], hi)]
    names = [str(a) if a == b else "%d-%d" % (a, b) for a, b in bins]
    return bins, names


def bin_counts(values, bins):
    return [sum(1 for v in values if a <= v <= b) for a, b in bins]


def wide_range(values, ratio=8.0):
    """True when values span enough of a range that a log axis reads better."""
    return bool(values) and min(values) > 0 and max(values) / min(values) >= ratio


def point_style(count, size=24.0, alpha=0.6, ref=300, min_size=6.0, min_alpha=0.2):
    """Marker size and alpha that shrink as the number of points grows."""
    if count <= ref:
        return size, alpha
    f = (ref / float(count)) ** 0.5
    return max(min_size, size * f), max(min_alpha, alpha * f)


def set_length_axis(ax, values, axis="x"):
    """Use a log axis with plain-number tick labels when lengths span widely."""
    if not wide_range(values):
        return
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    lo, hi = min(values), max(values)
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
                         10000, 20000, 50000) if lo * 0.8 <= t <= hi * 1.25]
    ticks = sorted(set([lo] + ticks + [hi]))
    # drop interior ticks that would crowd an end tick
    ticks = [t for t in ticks if t in (lo, hi) or (t / lo >= 1.3 and hi / t >= 1.3)]
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(lambda v, _pos: "%d" % round(v))
    a.set_minor_locator(NullLocator())
    a.set_minor_formatter(NullFormatter())


def table(rows, header, aligns=None):
    """Plain-text table. rows: list of lists of strings."""
    cols = len(header)
    widths = [max(len(str(r[c])) for r in rows + [header]) for c in range(cols)]
    aligns = aligns or ["<"] + [">"] * (cols - 1)
    fmt = "  ".join("{:%s%d}" % (aligns[c], widths[c]) for c in range(cols))
    lines = [fmt.format(*header), fmt.format(*["-" * w for w in widths])]
    lines += [fmt.format(*[str(v) for v in r]) for r in rows]
    return lines


# ------------------------------------------------------------------ charts
def make_charts(media_dir, n, chap, cov, covcount, venn, metrics):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Patch, Wedge, FancyBboxPatch
        import matplotlib.patheffects as pe
        import numpy as np
    except ImportError:
        print("\nmatplotlib is not installed, so no images were made.\n"
              "Install it with:  pip install matplotlib")
        return []

    os.makedirs(media_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    book_col = {"Matthew": "#2F5D8C", "Mark": "#B8473A", "Luke": "#4C8C5A"}
    pair_col = {("Matthew", "Mark"): "#8E5BA8", ("Matthew", "Luke"): "#2A9D9D",
                ("Mark", "Luke"): "#D98E2B"}
    saved = []

    def save(fig, name):
        path = os.path.join(media_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(path)

    def pair_key(x, y):
        return tuple(sorted((x, y), key=BOOKS.index))

    union = {b: [max(x, y) for x, y in zip(*cov[b].values())] for b in BOOKS}
    category = {b: [x + y for x, y in zip(*cov[b].values())] for b in BOOKS}  # 0, 1, or 2

    # 1. directional matrix
    mat = np.full((3, 3), np.nan)
    for i, x in enumerate(BOOKS):
        for j, y in enumerate(BOOKS):
            if x != y:
                mat[i, j] = pct(covcount[x][y], n[x])
    fig, ax = plt.subplots(figsize=(6.2, 5))
    vmax = max(float(np.nanmax(mat)), 1.0)
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax * 1.15)
    ax.set_xticks(range(3), BOOKS)
    ax.set_yticks(range(3), BOOKS)
    ax.set_xlabel("...is found in")
    ax.set_ylabel("Percent of this book's words")
    for i in range(3):
        for j in range(3):
            txt = "-" if i == j else "%.1f%%" % mat[i, j]
            dark = i != j and mat[i, j] > vmax * 0.6
            ax.text(j, i, txt, ha="center", va="center", fontsize=13,
                    color="white" if dark else "black")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("How much of each book appears in each other book")
    fig.colorbar(im, ax=ax, shrink=0.8, label="% of row book's words")
    save(fig, "01_directional_overlap.png")

    # 2. composition of each book
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for yi, b in enumerate(BOOKS):
        o1, o2, only1, only2, both, none = venn[b]
        left = 0.0
        for val, color in ((only1, pair_col[pair_key(b, o1)]),
                           (only2, pair_col[pair_key(b, o2)]),
                           (both, "#333333"), (none, "#E3E3E3")):
            w = pct(val, n[b])
            ax.barh(yi, w, left=left, color=color, height=0.6)
            if w >= 3:
                ax.text(left + w / 2, yi, "%.1f%%" % w, ha="center", va="center",
                        fontsize=9, color="black" if color == "#E3E3E3" else "white")
            left += w
    ax.set_yticks(range(3), BOOKS)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of the book's words")
    ax.set_title("Composition of each book by overlap")
    ax.legend(handles=[Patch(color=pair_col[("Matthew", "Mark")], label="Matthew + Mark only"),
                       Patch(color=pair_col[("Matthew", "Luke")], label="Matthew + Luke only"),
                       Patch(color=pair_col[("Mark", "Luke")], label="Mark + Luke only"),
                       Patch(color="#333333", label="All three"),
                       Patch(color="#E3E3E3", label="No overlap")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, frameon=False)
    save(fig, "02_book_composition.png")

    # 3. coverage by chapter
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    for ax, b in zip(axes, BOOKS):
        words, covered = defaultdict(int), defaultdict(int)
        for i in range(n[b]):
            words[chap[b][i]] += 1
            covered[chap[b][i]] += union[b][i]
        chs = sorted(words)
        ax.bar(chs, [pct(covered[c], words[c]) for c in chs], color=book_col[b])
        ax.set_xticks(chs)
        ax.set_ylabel("% covered")
        ax.set_title(b, loc="left", fontweight="bold")
        ax.set_ylim(0, 100)
    axes[-1].set_xlabel("Chapter")
    fig.suptitle("Share of each chapter that overlaps another Gospel", y=0.995)
    fig.tight_layout()
    save(fig, "03_coverage_by_chapter.png")

    # 4. overlap barcode along each book
    cmap = LinearSegmentedColormap.from_list("ov", ["#F2F2F2", "#F2A541", "#B2182B"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 4.8))
    nbins = 400
    for ax, b in zip(axes, BOOKS):
        vals = [float(np.mean(chunk)) for chunk in np.array_split(np.array(category[b]), nbins)]
        ax.imshow(np.array(vals)[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=2,
                  interpolation="nearest")
        starts = {}
        for i, c in enumerate(chap[b]):
            starts.setdefault(c, i)
        ax.set_xticks([i / n[b] * nbins for i in starts.values()], list(starts.keys()), fontsize=7)
        ax.set_yticks([])
        ax.set_ylabel(b, rotation=0, ha="right", va="center", fontweight="bold")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[-1].set_xlabel("Position in book (tick marks = chapter starts)")
    fig.legend(handles=[Patch(color="#F2F2F2", label="No overlap"),
                        Patch(color="#F2A541", label="Shared with one other Gospel"),
                        Patch(color="#B2182B", label="Shared with both")],
               loc="upper center", ncol=3, frameon=False)
    save(fig, "04_overlap_barcode.png")

    # 5. section length distribution (bins chosen from the observed lengths)
    bins, names = length_bins([m["length"] for m in metrics])
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(bins) + 2), 4.5))
    width = 0.26
    peak = 0
    for k, pr in enumerate(PAIRS):
        counts = bin_counts([m["length"] for m in metrics if m["pair"] == pr], bins)
        peak = max([peak] + counts)
        xs = [x + (k - 1) * width for x in range(len(bins))]
        ax.bar(xs, counts, width, color=pair_col[pr],
               label="%s - %s" % (ABBR[pr[0]], ABBR[pr[1]]))
        for x, c in zip(xs, counts):
            if c:
                ax.text(x, c, format(c, ","), ha="center", va="bottom", fontsize=7,
                        color="#555555")
    ax.set_ylim(0, max(peak, 1) * 1.12)
    ax.set_xticks(range(len(bins)), names)
    ax.set_xlabel("Words in the longer passage")
    ax.set_ylabel("Number of sections")
    ax.set_title("Length of shared sections")
    ax.legend(frameon=False)
    save(fig, "05_section_lengths.png")

    # 6. order agreement
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (a, b) in zip(axes, PAIRS):
        ms = [m for m in metrics if m["pair"] == (a, b)]
        xs = [m["pa"]["start"] for m in ms]
        ys = [m["pb"]["start"] for m in ms]
        size, alpha = point_style(len(xs), size=16, alpha=0.7, min_size=4)
        ax.scatter([x / n[a] * 100 for x in xs], [y / n[b] * 100 for y in ys],
                   s=size, alpha=alpha, color=pair_col[(a, b)], linewidths=0)
        ax.plot([0, 100], [0, 100], ls="--", color="#999999", lw=1)
        rho = spearman(xs, ys)
        ax.set_title("%s vs %s%s" % (a, b, "\nSpearman rho = %.3f" % rho if rho is not None else ""))
        ax.set_xlabel("Position in %s (%%)" % a)
        ax.set_ylabel("Position in %s (%%)" % b)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
    fig.suptitle("Do parallel passages occur in the same order? (dashed line = identical order)")
    fig.tight_layout()
    save(fig, "06_order_agreement.png")

    # 7. verbal agreement
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, key, label in ((axes[0], "lex", "Lexical overlap (%)"),
                           (axes[1], "seq", "Sequence similarity (%)")):
        present = [(i, pr, [m[key] for m in metrics if m["pair"] == pr])
                   for i, pr in enumerate(PAIRS)]
        present = [t for t in present if t[2]]
        bp = ax.boxplot([t[2] for t in present], positions=[t[0] for t in present],
                        widths=0.55, patch_artist=True, medianprops={"color": "black"},
                        flierprops={"markersize": 2.5, "alpha": 0.35,
                                    "markeredgewidth": 0.5})
        for patch, pr in zip(bp["boxes"], [t[1] for t in present]):
            patch.set_facecolor(pair_col[pr])
            patch.set_alpha(0.8)
        ax.set_xticks(range(len(PAIRS)), ["%s - %s" % (ABBR[a], ABBR[b]) for a, b in PAIRS])
        ax.set_title(label)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Agreement within sections")
    fig.suptitle("How closely the wording matches inside shared sections")
    fig.tight_layout()
    save(fig, "07_verbal_agreement.png")

    # 8. relationship diagram: three circles (after the classic synoptic diagram)
    code = {("Matthew", "Mark"): "MtMk", ("Matthew", "Luke"): "MtLk", ("Mark", "Luke"): "MkLk"}
    share = {}
    for b in BOOKS:
        o1, o2, only1, only2, both, none = venn[b]
        d = {"unique": pct(none, n[b]), "triple": pct(both, n[b])}
        d[code[pair_key(b, o1)]] = pct(only1, n[b])
        d[code[pair_key(b, o2)]] = pct(only2, n[b])
        share[b] = d
    slice_col = {"triple": "#8B1FA9", "MtLk": "#14147A", "MkLk": "#F28C28", "MtMk": "#A3122A"}
    unique_col = {"Mark": "#7B4B13", "Luke": "#1E8E8E", "Matthew": "#1F8A33"}
    centers = {"Mark": (50, 100), "Luke": (27, 36), "Matthew": (73, 36)}
    group = {"Mark": (270, ["MkLk", "triple", "MtMk"]),
             "Luke": (45, ["MtLk", "triple", "MkLk"]),
             "Matthew": (135, ["MtMk", "triple", "MtLk"])}
    RAD, BUS_Y = 13.0, 78.0
    outline = [pe.withStroke(linewidth=2.2, foreground="black", alpha=0.35)]

    fig, ax = plt.subplots(figsize=(8, 11.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 140)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(50, 137, "Relationships between the Synoptic Gospels", ha="center",
            fontsize=16, fontweight="bold")
    ax.text(50, 132.8, "Percent of each Gospel's words in each overlap category",
            ha="center", fontsize=10, color="#555555")

    mid = {}
    for b in BOOKS:
        cx, cy = centers[b]
        centre_angle, order = group[b]
        total = sum(share[b][c] for c in order) * 3.6
        start = centre_angle - total / 2.0
        a = start
        for c in order:
            span = share[b][c] * 3.6
            ax.add_patch(Wedge((cx, cy), RAD, a, a + span, fc=slice_col[c], ec="white", lw=1.2, zorder=3))
            mid[(b, c)] = a + span / 2.0
            a += span
        ax.add_patch(Wedge((cx, cy), RAD, a, start + 360, fc=unique_col[b], ec="white", lw=1.2, zorder=3))
        mid[(b, "unique")] = centre_angle + 180

    def edge(b, c, extra=0.0):
        cx, cy = centers[b]
        t = np.radians(mid[(b, c)])
        return (cx + (RAD + extra) * np.cos(t), cy + (RAD + extra) * np.sin(t))

    # circle names and percentages
    for b in BOOKS:
        cx, cy = centers[b]
        ax.text(cx, cy, b.upper(), ha="center", va="center", color="white", fontsize=12,
                fontweight="bold", zorder=6, path_effects=outline)
        ux, uy = edge(b, "unique", -RAD * 0.42)
        ax.text(ux, uy, "%.1f%%" % share[b]["unique"], ha="center", va="center", color="white",
                fontsize=12, fontweight="bold", zorder=6, path_effects=outline)
        prev_angle, bump = None, 0
        for c in sorted(group[b][1], key=lambda k: mid[(b, k)]):
            v = share[b][c]
            if v >= 9:
                x, y = edge(b, c, -RAD * 0.4)
                ax.text(x, y, "%.1f%%" % v, ha="center", va="center", color="white",
                        fontsize=10, fontweight="bold", zorder=6, path_effects=outline)
            else:
                if prev_angle is not None and mid[(b, c)] - prev_angle < 16:
                    bump += 1
                else:
                    bump = 0
                x, y = edge(b, c, 3.2 + 3.6 * (bump % 2))
                dx, dy = {("Mark", "triple"): (4.0, 0.0)}.get((b, c), (0.0, 0.0))
                x, y = x + dx, y + dy
                ax.text(x, y, "%.1f%%" % v, ha="center", va="center", fontsize=8.5, zorder=6,
                        bbox={"fc": "white", "ec": "none", "pad": 0.8, "alpha": 0.9})
            prev_angle = mid[(b, c)]

    # connectors
    def link(pts, color):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=2.2, solid_capstyle="round", zorder=2)

    L_BOX_X, R_BOX_X = 25.0, 75.0
    for b in ("Mark", "Luke"):
        x, y = edge(b, "MkLk")
        link([(x, y), (x, BUS_Y), (L_BOX_X, BUS_Y)], slice_col["MkLk"])
    for b in ("Mark", "Matthew"):
        x, y = edge(b, "MtMk")
        link([(x, y), (x, BUS_Y), (R_BOX_X, BUS_Y)], slice_col["MtMk"])
    x, y = edge("Mark", "triple")
    link([(x, y), (48.5, 68)], slice_col["triple"])
    x, y = edge("Luke", "triple")
    link([(x, y), (41, 60)], slice_col["triple"])
    x, y = edge("Matthew", "triple")
    link([(x, y), (59, 60)], slice_col["triple"])
    for b in ("Luke", "Matthew"):
        x, y = edge(b, "MtLk")
        link([(x, y), (50, y), (50, 18)], slice_col["MtLk"])
    x, y = edge("Mark", "unique")
    link([(x, y), (x, 120.5)], unique_col["Mark"])
    ulx, _ = edge("Luke", "unique")
    urx, _ = edge("Matthew", "unique")
    for b, bx in (("Luke", ulx), ("Matthew", urx)):
        x, y = edge(b, "unique")
        link([(x, y), (x, 18)], unique_col[b])

    # labelled boxes
    def box(text, xc, yc, w, h, color):
        ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                                    boxstyle="round,pad=0.2,rounding_size=0.9",
                                    fc=color, ec=color, zorder=4))
        ax.text(xc, yc, text, ha="center", va="center", color="white", fontsize=9.5,
                fontweight="bold", zorder=5)

    box("Unique to Mark", 50, 124, 24, 7, unique_col["Mark"])
    box("Mark and Luke", 13, BUS_Y, 24, 7.5, slice_col["MkLk"])
    box("Mark and Matthew", 87, BUS_Y, 24, 7.5, slice_col["MtMk"])
    box("Triple Tradition", 50, 64, 22, 8, slice_col["triple"])
    box("Double Tradition", 50, 14, 22, 7.5, slice_col["MtLk"])
    box("Unique to Luke", ulx, 14, 24, 7.5, unique_col["Luke"])
    box("Unique to Matthew", urx, 14, 24, 7.5, unique_col["Matthew"])
    ax.text(50, -1.5, "Each circle is one Gospel; slices show the share of its words that also appear in the "
            "other Gospel(s).", ha="center", fontsize=8, color="#666666")
    save(fig, "08_synoptic_relationships.png")

    return saved


def make_extra_charts(media_dir, n, chap, verse, cov, covcount, metrics):
    """A second set of charts in different visual forms (images 09 onward)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Patch, Wedge, Rectangle, PathPatch
        from matplotlib.path import Path
        from matplotlib.lines import Line2D
        from matplotlib.collections import PatchCollection
        import numpy as np
    except ImportError:
        return []

    os.makedirs(media_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    book_col = {"Matthew": "#2F5D8C", "Mark": "#B8473A", "Luke": "#4C8C5A"}
    pair_col = {("Matthew", "Mark"): "#8E5BA8", ("Matthew", "Luke"): "#2A9D9D",
                ("Mark", "Luke"): "#D98E2B"}
    cat_col = {"unique": "#E3E3E3", "MtMk": "#A3122A", "MtLk": "#14147A",
               "MkLk": "#F28C28", "triple": "#8B1FA9"}
    cat_name = {"unique": "No overlap", "MtMk": "Matthew + Mark only",
                "MtLk": "Matthew + Luke only", "MkLk": "Mark + Luke only",
                "triple": "All three"}
    cat_short = {"unique": "Unique", "MtMk": "Mt+Mk", "MtLk": "Mt+Lk",
                 "MkLk": "Mk+Lk", "triple": "All 3"}
    overlap_cats = ["MtMk", "MtLk", "MkLk", "triple"]
    code = {("Matthew", "Mark"): "MtMk", ("Matthew", "Luke"): "MtLk", ("Mark", "Luke"): "MkLk"}
    saved = []

    def save(fig, name):
        path = os.path.join(media_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(path)

    def pair_key(x, y):
        return tuple(sorted((x, y), key=BOOKS.index))

    def cat_legend(cats=None):
        return [Patch(color=cat_col[c], label=cat_name[c]) for c in (cats or list(cat_col))]

    union = {b: [max(x, y) for x, y in zip(*cov[b].values())] for b in BOOKS}
    wordcat, counts = {}, {}
    for b in BOOKS:
        o1, o2 = [o for o in BOOKS if o != b]
        c1, c2 = cov[b][o1], cov[b][o2]
        k1, k2 = code[pair_key(b, o1)], code[pair_key(b, o2)]
        labs = []
        for i in range(n[b]):
            if c1[i] and c2[i]:
                labs.append("triple")
            elif c1[i]:
                labs.append(k1)
            elif c2[i]:
                labs.append(k2)
            else:
                labs.append("unique")
        wordcat[b] = labs
        counts[b] = {c: labs.count(c) for c in cat_col}

    # ---- 9. chord diagram
    gap = 8.0
    scale = (360 - 3 * gap) / sum(n.values())  # degrees per word
    arcs, a = {}, 90.0
    for b in BOOKS:
        arcs[b] = (a, a + n[b] * scale)
        a += n[b] * scale + gap
    seg = {}
    for i, b in enumerate(BOOKS):
        nxt, prv = BOOKS[(i + 1) % 3], BOOKS[(i - 1) % 3]
        seg[(b, nxt)] = (arcs[b][0], arcs[b][0] + covcount[b][nxt] * scale)
        seg[(b, prv)] = (arcs[b][1] - covcount[b][prv] * scale, arcs[b][1])

    def pt(deg, r):
        t = np.radians(deg)
        return (r * np.cos(t), r * np.sin(t))

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.4, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    for b in BOOKS:
        ax.add_patch(Wedge((0, 0), 1.0, arcs[b][0], arcs[b][1], width=0.07,
                           fc=book_col[b], ec="white"))
        mid = np.radians((arcs[b][0] + arcs[b][1]) / 2)
        cx_, sy_ = np.cos(mid), np.sin(mid)
        ax.text(1.08 * cx_, 1.08 * sy_, "%s\n%s words" % (b, format(n[b], ",")),
                ha="left" if cx_ > 0.25 else "right" if cx_ < -0.25 else "center",
                va="top" if sy_ < -0.5 else "center", fontsize=11, fontweight="bold",
                color=book_col[b])
    r_in = 0.93
    for A, B in PAIRS:
        a0, a1 = seg[(A, B)]
        b0, b1 = seg[(B, A)]

        def arc_pts(t0, t1):
            return [pt(t, r_in) for t in np.linspace(t0, t1, 14)]
        pa_, pb_ = arc_pts(a0, a1), arc_pts(b0, b1)
        verts = pa_ + [(0, 0), pb_[0]] + pb_[1:] + [(0, 0), pa_[0]]
        codes = ([Path.MOVETO] + [Path.LINETO] * (len(pa_) - 1) + [Path.CURVE3, Path.CURVE3]
                 + [Path.LINETO] * (len(pb_) - 1) + [Path.CURVE3, Path.CURVE3])
        ax.add_patch(PathPatch(Path(verts, codes), fc=pair_col[(A, B)], ec="white", lw=0.6,
                               alpha=0.65))
    ax.legend(handles=[Patch(color=pair_col[p], alpha=0.65, label="%s - %s" % (ABBR[p[0]], ABBR[p[1]]))
                       for p in PAIRS], loc="lower center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, -0.04))
    ax.set_title("Shared words between the Gospels (ribbon width = words in overlap)", y=1.02)
    save(fig, "09_chord_diagram.png")

    # ---- 10. alignment ribbons
    ymap = {"Matthew": 2.0, "Mark": 1.0, "Luke": 0.0}
    H = 0.16
    maxn = max(n.values())
    fig, ax = plt.subplots(figsize=(14, 6.2))
    for b in BOOKS:
        ax.add_patch(Rectangle((0, ymap[b] - H / 2), n[b], H, fc=book_col[b], ec="none", zorder=3))
        ax.text(-maxn * 0.012, ymap[b], b, ha="right", va="center", fontweight="bold",
                color=book_col[b], fontsize=12)
    rib_codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    for A, B in PAIRS:
        ms = [m for m in metrics if m["pair"] == (A, B)]
        if not ms:
            continue
        ya, yb = ymap[A] - H / 2, ymap[B] + H / 2
        ym = (ya + yb) / 2
        patches = []
        for m in ms:
            xa0, xa1 = m["pa"]["start"], m["pa"]["end"] + 1
            xb0, xb1 = m["pb"]["start"], m["pb"]["end"] + 1
            verts = [(xa0, ya), (xa0, ym), (xb0, ym), (xb0, yb), (xb1, yb), (xb1, ym),
                     (xa1, ym), (xa1, ya), (xa0, ya)]
            patches.append(PathPatch(Path(verts, rib_codes)))
        _, alpha = point_style(len(ms), alpha=0.4, ref=200, min_alpha=0.03)
        ax.add_collection(PatchCollection(patches, fc=pair_col[(A, B)], ec=pair_col[(A, B)],
                                          lw=0.3, alpha=alpha, zorder=2,
                                          match_original=False))
    ax.set_xlim(-maxn * 0.1, maxn * 1.01)
    ax.set_ylim(-0.3, 2.3)
    ax.set_yticks([])
    ax.set_xlabel("Word position in the book (bars are drawn to scale)")
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Patch(color=pair_col[p], alpha=0.6, label="%s - %s" % (ABBR[p[0]], ABBR[p[1]]))
                       for p in PAIRS], loc="upper center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.08))
    save(fig, "10_alignment_ribbons.png")

    # ---- 11. chapter-by-chapter heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, (A, B) in zip(axes, PAIRS):
        nca, ncb = max(chap[A]), max(chap[B])
        M = np.zeros((nca, ncb))
        for m in metrics:
            if m["pair"] == (A, B):
                M[chap[A][m["pa"]["start"]] - 1, chap[B][m["pb"]["start"]] - 1] += (
                    m["pa"]["words"] + m["pb"]["words"]) / 2.0
        cmap = LinearSegmentedColormap.from_list("c", ["#FFFFFF", pair_col[(A, B)]])
        im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(0, ncb, 2), [str(i + 1) for i in range(0, ncb, 2)])
        ax.set_yticks(range(0, nca, 2), [str(i + 1) for i in range(0, nca, 2)])
        ax.set_xlabel("Chapter in %s" % B)
        ax.set_ylabel("Chapter in %s" % A)
        ax.set_title("%s vs %s" % (A, B))
        fig.colorbar(im, ax=ax, shrink=0.8, label="shared words")
    fig.suptitle("Which chapters parallel which? (shared words by chapter pair)")
    fig.tight_layout()
    save(fig, "11_chapter_heatmaps.png")

    # ---- 12. verse grid
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), gridspec_kw={"height_ratios": [28, 16, 24]})
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#EFEFEF")
    for ax, b in zip(axes, BOOKS):
        nch, maxv = max(chap[b]), max(verse[b])
        words, covw = np.zeros((nch, maxv)), np.zeros((nch, maxv))
        for i in range(n[b]):
            words[chap[b][i] - 1, verse[b][i] - 1] += 1
            covw[chap[b][i] - 1, verse[b][i] - 1] += union[b][i]
        frac = np.where(words > 0, covw / np.maximum(words, 1), np.nan)
        im = ax.imshow(np.ma.masked_invalid(frac), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(nch), [str(i + 1) for i in range(nch)], fontsize=7)
        ax.set_xticks(range(0, maxv, 5), [str(i + 1) for i in range(0, maxv, 5)], fontsize=8)
        ax.set_ylabel("Chapter")
        ax.set_title(b, loc="left", fontweight="bold")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[-1].set_xlabel("Verse")
    fig.colorbar(im, ax=axes, shrink=0.35, label="share of the verse's words in overlap")
    fig.suptitle("Verse-by-verse overlap map (grey = no such verse)", y=0.915)
    save(fig, "12_verse_grid.png")

    # ---- 13. donuts
    order = ["MtMk", "MtLk", "MkLk", "triple", "unique"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    for ax, b in zip(axes, BOOKS):
        ax.pie([counts[b][c] for c in order], colors=[cat_col[c] for c in order],
               startangle=90, counterclock=False,
               wedgeprops={"width": 0.38, "edgecolor": "white"})
        ov = 100 - pct(counts[b]["unique"], n[b])
        ax.text(0, 0, "%s\n%.1f%%\nin overlap" % (b, ov), ha="center", va="center",
                fontweight="bold", fontsize=11)
        ax.text(0, -1.45, "\n".join("%s: %.1f%%" % (cat_name[c], pct(counts[b][c], n[b]))
                                    for c in order
                                    if c in ("triple", "unique") or ABBR[b] in cat_short[c]),
                ha="center", va="top", fontsize=9)
    fig.legend(handles=cat_legend(order), loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Overlap composition of each Gospel", y=1.09, fontsize=13)
    save(fig, "13_donuts.png")

    # ---- 14. waffle charts
    def waffle_counts(vals):
        fl = [int(v) for v in vals]
        for i in sorted(range(len(vals)), key=lambda k: vals[k] - fl[k], reverse=True)[:100 - sum(fl)]:
            fl[i] += 1
        return fl
    wf_order = ["triple", "MkLk", "MtLk", "MtMk", "unique"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    for ax, b in zip(axes, BOOKS):
        sq = waffle_counts([pct(counts[b][c], n[b]) for c in wf_order])
        seq = [c for c, k in zip(wf_order, sq) for _ in range(k)]
        for idx, c in enumerate(seq):
            r, col = divmod(idx, 10)
            ax.add_patch(Rectangle((col, 9 - r), 0.9, 0.9, fc=cat_col[c], ec="none"))
        ax.set_xlim(-0.1, 10)
        ax.set_ylim(-0.1, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(b, fontweight="bold")
    fig.legend(handles=cat_legend(wf_order), loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Each square = 1% of the Gospel's words", y=1.02, fontsize=13)
    save(fig, "14_waffle.png")

    # ---- 15. treemap
    fig, ax = plt.subplots(figsize=(13, 7.2))
    total = float(sum(n.values()))
    x0 = 0.0
    for b in BOOKS:
        w = n[b] / total * 100
        y0 = 0.0
        for c in ["unique"] + overlap_cats:
            h = counts[b][c] / n[b] * 100
            ax.add_patch(Rectangle((x0, y0), w, h, fc=cat_col[c], ec="white", lw=1.5))
            if h >= 4 and w >= 12:
                ax.text(x0 + w / 2, y0 + h / 2, "%s\n%s words (%.1f%%)" % (
                    cat_short[c], format(counts[b][c], ","), h), ha="center", va="center",
                    fontsize=9, color="black" if c == "unique" else "white")
            y0 += h
        ax.text(x0 + w / 2, 102, "%s (%s words)" % (b, format(n[b], ",")), ha="center",
                fontweight="bold", color=book_col[b])
        x0 += w
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 108)
    ax.axis("off")
    ax.legend(handles=cat_legend(), loc="lower center", ncol=5, frameon=False,
              bbox_to_anchor=(0.5, -0.07))
    ax.set_title("All three Gospels at a glance (width = book size, height = share of its words)",
                 y=1.04)
    save(fig, "15_treemap.png")

    # ---- 16. asymmetry dumbbell
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for yi, (A, B) in enumerate(PAIRS):
        pa_, pb_ = pct(covcount[A][B], n[A]), pct(covcount[B][A], n[B])
        ax.plot([pa_, pb_], [yi, yi], color="#BBBBBB", lw=4, zorder=1)
        ax.scatter([pa_], [yi], s=170, color=book_col[A], zorder=3, clip_on=False)
        ax.scatter([pb_], [yi], s=170, color=book_col[B], zorder=3, clip_on=False)
        # one label above its dot and one below, so close values never collide
        for val, bk, up in ((pa_, A, True), (pb_, B, False)):
            ax.annotate("%.1f%%" % val, (val, yi), xytext=(0, 11 if up else -11),
                        textcoords="offset points", va="bottom" if up else "top",
                        ha="left" if val < 4 else "right" if val > 96 else "center",
                        color=book_col[bk], fontsize=9)
    ax.set_yticks(range(len(PAIRS)), ["%s - %s" % p for p in PAIRS])
    ax.invert_yaxis()
    allv = [pct(covcount[A][B], n[A]) for A in BOOKS for B in BOOKS if A != B]
    ax.set_xlim(max(0, 10 * int((min(allv) - 12) // 10)), 100)
    ax.set_ylim(len(PAIRS) - 0.4, -0.6)
    ax.set_xlabel("% of the book's words found in the other book")
    ax.set_title("Overlap is directional: a shorter book is covered more heavily")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=book_col[b], label=b, markersize=9)
                       for b in BOOKS], frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3)
    save(fig, "16_asymmetry_dumbbell.png")

    # ---- 17. stacked area along each book
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    nb = 40
    for ax, b in zip(axes, BOOKS):
        edges = np.linspace(0, n[b], nb + 1).astype(int)
        ys = {c: [] for c in overlap_cats}
        for k in range(nb):
            piece = wordcat[b][edges[k]:edges[k + 1]]
            for c in ys:
                ys[c].append(100.0 * piece.count(c) / max(len(piece), 1))
        xs = (np.arange(nb) + 0.5) / nb * 100
        ax.stackplot(xs, [ys[c] for c in overlap_cats], colors=[cat_col[c] for c in overlap_cats])
        ax.set_ylabel("% of words")
        ax.set_title(b, loc="left", fontweight="bold")
        ax.set_xlim(0, 100)
    axes[-1].set_xlabel("Position in book (%)")
    fig.legend(handles=cat_legend(overlap_cats), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("What kind of overlap occurs where", y=1.02, fontsize=13)
    save(fig, "17_overlap_along_book_area.png")

    # ---- 18. cumulative coverage and section-length ECDF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for b in BOOKS:
        u = np.array(union[b], dtype=float)
        ax1.plot(np.arange(n[b]) / n[b] * 100, np.cumsum(u) / max(u.sum(), 1) * 100,
                 color=book_col[b], lw=2, label=b)
    ax1.plot([0, 100], [0, 100], ls="--", color="#999999", lw=1)
    ax1.set_xlabel("Position in book (%)")
    ax1.set_ylabel("Cumulative share of the book's overlap (%)")
    ax1.set_title("How evenly is the overlap spread? (dashed = perfectly even)")
    ax1.legend(frameon=False)
    for pr in PAIRS:
        L = np.sort([m["length"] for m in metrics if m["pair"] == pr])
        if len(L):
            ax2.step(L, np.arange(1, len(L) + 1) / len(L) * 100, where="post", lw=2,
                     color=pair_col[pr], label="%s - %s" % (ABBR[pr[0]], ABBR[pr[1]]))
    set_length_axis(ax2, [m["length"] for m in metrics])
    ax2.set_xlabel("Section length (words)%s" % (
        ", log scale" if wide_range([m["length"] for m in metrics]) else ""))
    ax2.set_ylabel("% of sections this long or shorter")
    ax2.set_title("Cumulative distribution of section length")
    ax2.legend(frameon=False)
    fig.tight_layout()
    save(fig, "18_cumulative_and_ecdf.png")

    # ---- 20. section length vs similarity
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    # Draw all pairs in one shuffled pass so no pair hides another.
    order_ = np.random.default_rng(0).permutation(len(metrics))
    size, alpha = point_style(len(metrics))
    ax.scatter([metrics[i]["length"] for i in order_], [metrics[i]["seq"] for i in order_],
               s=size, alpha=alpha, linewidths=0,
               c=[pair_col[metrics[i]["pair"]] for i in order_])
    handles = [Line2D([], [], marker="o", ls="", color=pair_col[pr], markersize=6,
                      label="%s - %s" % (ABBR[pr[0]], ABBR[pr[1]]))
               for pr in PAIRS if any(m["pair"] == pr for m in metrics)]
    xs_all = np.array([m["length"] for m in metrics], dtype=float)
    ys_all = np.array([m["seq"] for m in metrics], dtype=float)
    if len(metrics) > 2 and np.ptp(xs_all) > 0 and np.ptp(ys_all) > 0:
        coef = np.polyfit(xs_all, ys_all, 1)
        r = np.corrcoef(xs_all, ys_all)[0, 1]
        grid = np.linspace(xs_all.min(), xs_all.max(), 200)
        handles += ax.plot(grid, np.polyval(coef, grid), color="#333333", ls="--", lw=1.5,
                           label="linear trend (r = %.2f)" % r)
        # median similarity per length bin, which is robust when points pile up
        bins_, _names = length_bins(list(xs_all.astype(int)), max_bins=12)
        mx, my = [], []
        for lo_, hi_ in bins_:
            sel_ = ys_all[(xs_all >= lo_) & (xs_all <= hi_)]
            if len(sel_):
                mx.append((lo_ + hi_) / 2.0)
                my.append(float(np.median(sel_)))
        if len(mx) > 1:
            handles += ax.plot(mx, my, color="black", marker="D", ms=4, lw=1.2,
                               label="median by length bin")
    set_length_axis(ax, [m["length"] for m in metrics])
    ax.set_xlabel("Section length (words, longer passage%s)" % (
        "; log scale" if wide_range([m["length"] for m in metrics]) else ""))
    ax.set_ylabel("Sequence similarity (%)")
    ax.set_title("Do longer sections agree more closely?")
    ax.set_ylim(0, 105)
    ax.legend(handles=handles, frameon=False)
    save(fig, "20_length_vs_similarity.png")

    # ---- 21. similarity distributions
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    # Bins are centred on multiples of 5 so that the coarse values short sections
    # produce (e.g. 50%, 66.7%, 100%) fall inside a bin rather than on an edge.
    bins = np.arange(-2.5, 105, 5)
    for ax, key, label in ((axes[0], "lex", "Lexical overlap (%)"),
                           (axes[1], "seq", "Sequence similarity (%)")):
        for pr in PAIRS:
            vals = [m[key] for m in metrics if m["pair"] == pr]
            if vals:
                ax.hist(vals, bins=bins, histtype="step", lw=2.2, color=pair_col[pr],
                        weights=[100.0 / len(vals)] * len(vals),
                        label="%s - %s (%s)" % (ABBR[pr[0]], ABBR[pr[1]], format(len(vals), ",")))
        ax.set_xlabel(label)
        ax.set_xlim(-2.5, 102.5)
    axes[0].set_ylabel("% of the pair's sections")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Shape of the agreement distributions inside shared sections")
    fig.tight_layout()
    save(fig, "21_similarity_distribution.png")

    # ---- 22. longest sections lollipop
    top = sorted(metrics, key=lambda m: -m["length"])[:15]
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    for i, m in enumerate(top):
        c = pair_col[m["pair"]]
        ax.hlines(i, 0, m["length"], color=c, lw=2.2)
        ax.scatter(m["length"], i, s=70, color=c, zorder=3)
        ax.annotate("%d words, %.0f%% sim." % (m["length"], m["seq"]), (m["length"], i),
                    xytext=(7, 0), textcoords="offset points", va="center", fontsize=8,
                    color="#555555")
    ax.set_yticks(range(len(top)), ["%s %s  /  %s %s" % (ABBR[m["pair"][0]], m["pa"]["ref"],
                                                       ABBR[m["pair"][1]], m["pb"]["ref"])
                                    for m in top])
    ax.invert_yaxis()
    ax.set_xlabel("Words in the longer passage")
    ax.set_title("The 15 longest shared sections")
    ax.legend(handles=[Patch(color=pair_col[p], label="%s - %s" % (ABBR[p[0]], ABBR[p[1]]))
                       for p in PAIRS], frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_xlim(0, max(m["length"] for m in top) * 1.3 if top else 1)
    save(fig, "22_longest_sections.png")

    return saved


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Stats on a synoptic overlap file.")
    ap.add_argument("--overlap", default=OVERLAP_PATH, help="overlap file")
    ap.add_argument("--matthew", default=MATTHEW_PATH)
    ap.add_argument("--mark", default=MARK_PATH)
    ap.add_argument("--luke", default=LUKE_PATH)
    ap.add_argument("--out", default=OUTPUT_PATH, help="report file to write")
    ap.add_argument("--csv", default=SECTIONS_CSV_PATH,
                    help="optional CSV of every section with its metrics")
    ap.add_argument("--media", default=MEDIA_DIR, help="folder for chart images")
    ap.add_argument("--images", action=argparse.BooleanOptionalAction, default=MAKE_IMAGES,
                    help="create chart images (needs matplotlib)")
    args, _ = ap.parse_known_args()

    # If the overlap file is not where expected, try the fallback location.
    if (not os.path.exists(os.path.expanduser(args.overlap))
            and args.overlap == OVERLAP_PATH
            and os.path.exists(FALLBACK_OVERLAP_PATH)):
        args.overlap = FALLBACK_OVERLAP_PATH

    paths = {"overlap file": args.overlap, "Matthew": args.matthew,
             "Mark": args.mark, "Luke": args.luke}
    for label, p in paths.items():
        if not os.path.exists(os.path.expanduser(p)):
            sys.exit("Cannot find %s: %s\nEdit the SETTINGS block at the top "
                     "of this script." % (label, p))

    sections = parse_overlap(os.path.expanduser(args.overlap))
    settings_line = read_settings_line(os.path.expanduser(args.overlap))
    book_files = {"Matthew": args.matthew, "Mark": args.mark, "Luke": args.luke}
    chap, verse, n = {}, {}, {}
    for b in BOOKS:
        chap[b], verse[b] = load_book(os.path.expanduser(book_files[b]))
        n[b] = len(chap[b])

    # ---- coverage arrays: cov[book][partner][i] = 1 if word i is in a section with partner
    cov = {b: {o: bytearray(n[b]) for o in BOOKS if o != b} for b in BOOKS}
    for s in sections:
        for p, partner in ((s["passages"][0], s["b"]), (s["passages"][1], s["a"])):
            if p["end"] >= n[p["book"]]:
                sys.exit("Overlap file does not match the %s file given "
                         "(position %d is out of range)." % (p["book"], p["end"]))
            cov[p["book"]][partner][p["start"]:p["end"] + 1] = (
                b"\x01" * (p["end"] - p["start"] + 1))

    union = {b: bytearray(max(c1, c2) for c1, c2 in zip(*cov[b].values()))
             for b in BOOKS}
    covcount = {b: {o: sum(cov[b][o]) for o in cov[b]} for b in BOOKS}
    unioncount = {b: sum(union[b]) for b in BOOKS}

    # ---- per-section metrics
    metrics = []
    for s in sections:
        pa, pb = s["passages"]
        wa, wb = norm_words(pa["text"]), norm_words(pb["text"])
        seq = 100.0 * SequenceMatcher(None, wa, wb, autojunk=False).ratio()
        metrics.append({
            "pair": (s["a"], s["b"]), "n": s["n"], "pa": pa, "pb": pb,
            "length": max(pa["words"], pb["words"]),
            "lex": lexical_overlap(wa, wb), "seq": seq})

    R = []  # report lines

    def title(t):
        R.extend(["", "=" * 74, t, "=" * 74])

    R.append("SYNOPTIC OVERLAP STATISTICS")
    R.append("Overlap file: %s" % args.overlap)
    if settings_line:
        R.append("Overlap settings: %s" % settings_line)
    R.append("Sections read: %d (%s)" % (len(sections), ", ".join(
        "%s-%s %d" % (ABBR[a], ABBR[b], sum(1 for s in sections if (s['a'], s['b']) == (a, b)))
        for a, b in PAIRS)))

    # ---- 1. book sizes
    title("1. BOOK SIZES")
    rows = []
    for b in BOOKS:
        rows.append([b, n[b], len(set(chap[b])), len(set(zip(chap[b], verse[b])))])
    rows.append(["Total", sum(n.values()), "", ""])
    R += table(rows, ["Book", "Words", "Chapters", "Verses"])

    # ---- 2. directional percentages
    title("2. DIRECTIONAL OVERLAP: % of one book found in another")
    R.append("Cell = share of the ROW book's words that lie inside a section "
             "shared with the COLUMN book.")
    R.append("")
    rows = []
    for x in BOOKS:
        rows.append([x] + ["-" if x == y else "%.2f%%" % pct(covcount[x][y], n[x])
                           for y in BOOKS])
    R += table(rows, ["% of ... in ..."] + BOOKS)
    R.append("")
    R.append("Headline figures:")
    for x, y in (("Luke", "Mark"), ("Mark", "Matthew"), ("Matthew", "Luke"),
                 ("Mark", "Luke"), ("Matthew", "Mark"), ("Luke", "Matthew")):
        R.append("  %-8s in %-8s: %6.2f%%  (%d of %d words)"
                 % (x, y, pct(covcount[x][y], n[x]), covcount[x][y], n[x]))

    # ---- 3. total overlap
    title("3. OVERALL OVERLAP (unique words, counted once per book)")
    rows = []
    for b in BOOKS:
        rows.append([b, n[b], unioncount[b], "%.2f%%" % pct(unioncount[b], n[b]),
                     "%.2f%%" % pct(unioncount[b], sum(unioncount.values()))])
    tot_cov, tot_words = sum(unioncount.values()), sum(n.values())
    rows.append(["All three", tot_words, tot_cov, "%.2f%%" % pct(tot_cov, tot_words), "100.00%"])
    R += table(rows, ["Book", "Words", "Words in any overlap", "% of book",
                      "Share of all overlap"])
    R.append("")
    R.append("Share of all overlap = that book's covered words / covered words in all three books.")

    # ---- 4. Venn-style breakdown
    title("4. WHO SHARES WHAT: overlap with one other book vs both")
    R.append("For each book, its words split by which other book(s) also contain them.")
    R.append("")
    venn = {}
    for b in BOOKS:
        o1, o2 = [o for o in BOOKS if o != b]
        c1, c2 = cov[b][o1], cov[b][o2]
        both = sum(1 for i in range(n[b]) if c1[i] and c2[i])
        only1 = sum(1 for i in range(n[b]) if c1[i] and not c2[i])
        only2 = sum(1 for i in range(n[b]) if c2[i] and not c1[i])
        none = n[b] - both - only1 - only2
        venn[b] = (o1, o2, only1, only2, both, none)
        R.append("%s (%d words)" % (b, n[b]))
        rows = [["Shared with %s only" % o1, only1, "%.2f%%" % pct(only1, n[b])],
                ["Shared with %s only" % o2, only2, "%.2f%%" % pct(only2, n[b])],
                ["Shared with both (triple tradition)", both, "%.2f%%" % pct(both, n[b])],
                ["No overlap (unique to %s)" % b, none, "%.2f%%" % pct(none, n[b])]]
        R += ["  " + l for l in table(rows, ["Category", "Words", "% of book"])]
        a1 = covcount[b][o1]
        a2 = covcount[b][o2]
        R.append("  Of %s's words shared with %s, %.1f%% are also in %s."
                 % (b, o1, pct(both, a1), o2))
        R.append("  Of %s's words shared with %s, %.1f%% are also in %s."
                 % (b, o2, pct(both, a2), o1))
        R.append("")

    # ---- 5. pair summary
    title("5. PAIR SUMMARY")
    rows = []
    for a, b in PAIRS:
        ms = [m for m in metrics if m["pair"] == (a, b)]
        lens = [m["length"] for m in ms]
        ca, cb = covcount[a][b], covcount[b][a]
        rows.append(["%s - %s" % (a, b), len(ms),
                     "%.1f%%" % pct(len(ms), len(metrics)), ca, cb,
                     "%.2f%%" % pct(ca + cb, n[a] + n[b]),
                     "%.1f" % statistics.mean(lens) if lens else "-",
                     "%.0f" % statistics.median(lens) if lens else "-",
                     max(lens) if lens else "-"])
    R += table(rows, ["Pair", "Sections", "% of sections", "Words (1st)", "Words (2nd)",
                      "Similarity*", "Mean len", "Median len", "Longest"])
    R.append("")
    R.append("* Similarity = (covered words in both books) / (total words in both books),")
    R.append("  a Dice-style symmetric measure. Section length = the longer of its two passages.")

    # ---- 6. length histogram
    title("6. SECTION LENGTH DISTRIBUTION (words in the longer passage)")
    bins, names = length_bins([m["length"] for m in metrics])
    rows = []
    for (lo, hi), nm in zip(bins, names):
        row = [nm]
        for pr in PAIRS:
            row.append(sum(1 for m in metrics if m["pair"] == pr and lo <= m["length"] <= hi))
        row.append(sum(1 for m in metrics if lo <= m["length"] <= hi))
        rows.append(row)
    rows.append(["Total"] + [sum(1 for m in metrics if m["pair"] == pr) for pr in PAIRS]
                + [len(metrics)])
    R += table(rows, ["Words"] + ["%s-%s" % (ABBR[a], ABBR[b]) for a, b in PAIRS] + ["All"])
    if metrics:
        L = [m["length"] for m in metrics]
        R.append("")
        R.append("All sections: mean %.1f, median %.0f, st. dev. %.1f, min %d, max %d"
                 % (statistics.mean(L), statistics.median(L),
                    statistics.pstdev(L), min(L), max(L)))

    # ---- 7. verbal agreement
    title("7. VERBAL AGREEMENT WITHIN SECTIONS")
    R.append("Lexical overlap = shared words / words in the shorter passage (order ignored).")
    R.append("Sequence similarity = order-sensitive match ratio between the two word lists.")
    R.append("Accents and punctuation are ignored in both.")
    R.append("")
    rows = []
    for label, ms in ([("%s - %s" % pr, [m for m in metrics if m["pair"] == pr])
                       for pr in PAIRS] + [("All", metrics)]):
        if not ms:
            continue
        rows.append([label,
                     "%.1f%%" % statistics.mean(m["lex"] for m in ms),
                     "%.1f%%" % statistics.median(m["lex"] for m in ms),
                     "%.1f%%" % statistics.mean(m["seq"] for m in ms),
                     "%.1f%%" % statistics.median(m["seq"] for m in ms),
                     sum(1 for m in ms if m["seq"] >= 90),
                     sum(1 for m in ms if m["seq"] < 60)])
    R += table(rows, ["Pair", "Mean lexical", "Median lexical", "Mean sequence",
                      "Median sequence", "Near-verbatim (>=90%)", "Loose (<60%)"])

    # ---- 8. order agreement
    title("8. ORDER AGREEMENT: do parallel passages appear in the same order?")
    R.append("Spearman rho compares the order of sections in the two books "
             "(1.0 = identical order, 0 = unrelated).")
    R.append("In-order steps = share of consecutive sections (by first book) whose "
             "second-book position also moves forward.")
    R.append("")
    rows = []
    for pr in PAIRS:
        ms = sorted([m for m in metrics if m["pair"] == pr],
                    key=lambda m: m["pa"]["start"])
        xs = [m["pa"]["start"] for m in ms]
        ys = [m["pb"]["start"] for m in ms]
        rho = spearman(xs, ys)
        steps = [ys[i + 1] > ys[i] for i in range(len(ys) - 1)]
        rows.append(["%s - %s" % pr, len(ms),
                     "%.3f" % rho if rho is not None else "-",
                     "%.1f%%" % pct(sum(steps), len(steps)) if steps else "-"])
    R += table(rows, ["Pair", "Sections", "Spearman rho", "In-order steps"])

    # ---- 9. verse-level coverage
    title("9. VERSE-LEVEL COVERAGE (any overlap, either partner)")
    rows = []
    for b in BOOKS:
        vw, vc = defaultdict(int), defaultdict(int)
        for i in range(n[b]):
            key = (chap[b][i], verse[b][i])
            vw[key] += 1
            vc[key] += union[b][i]
        total = len(vw)
        anyv = sum(1 for k in vw if vc[k] > 0)
        half = sum(1 for k in vw if vc[k] * 2 >= vw[k])
        full = sum(1 for k in vw if vc[k] == vw[k])
        rows.append([b, total, "%d (%.1f%%)" % (anyv, pct(anyv, total)),
                     "%d (%.1f%%)" % (half, pct(half, total)),
                     "%d (%.1f%%)" % (full, pct(full, total))])
    R += table(rows, ["Book", "Verses", "Touched by overlap", ">=50% covered",
                      "Fully covered"])

    # ---- 10. per-chapter
    title("10. COVERAGE BY CHAPTER (any overlap, either partner)")
    for b in BOOKS:
        R.append("")
        R.append(b)
        cw, cc = defaultdict(int), defaultdict(int)
        for i in range(n[b]):
            cw[chap[b][i]] += 1
            cc[chap[b][i]] += union[b][i]
        rows = []
        for c in sorted(cw):
            p = pct(cc[c], cw[c])
            rows.append([c, cw[c], cc[c], "%.1f%%" % p, bar(p)])
        R += ["  " + l for l in table(rows, ["Ch", "Words", "Covered", "%", ""],
                                       ["<", ">", ">", ">", "<"])]
        ranked = sorted(cw, key=lambda c: -pct(cc[c], cw[c]))
        R.append("  Highest: " + ", ".join("%d (%.0f%%)" % (c, pct(cc[c], cw[c]))
                                          for c in ranked[:3]))
        R.append("  Lowest:  " + ", ".join("%d (%.0f%%)" % (c, pct(cc[c], cw[c]))
                                          for c in ranked[::-1][:3]))

    # ---- 11. position within book
    title("11. WHERE IN THE BOOK? coverage by tenth of the text")
    for b in BOOKS:
        R.append("")
        R.append(b)
        rows = []
        for d in range(10):
            lo, hi = n[b] * d // 10, n[b] * (d + 1) // 10
            c = sum(union[b][lo:hi])
            p = pct(c, hi - lo)
            rows.append(["%d0-%d0%%" % (d, d + 1) if d else "0-10%", "%.1f%%" % p, bar(p)])
        R += ["  " + l for l in table(rows, ["Part", "Covered", ""], ["<", ">", "<"])]

    # ---- 12. top lists
    def describe(m):
        return "%s %s <-> %s %s  (%d / %d words, lexical %.0f%%, sequence %.0f%%)" % (
            m["pair"][0], m["pa"]["ref"], m["pair"][1], m["pb"]["ref"],
            m["pa"]["words"], m["pb"]["words"], m["lex"], m["seq"])

    title("12. NOTABLE SECTIONS")
    R.append("Longest sections:")
    for m in sorted(metrics, key=lambda m: -m["length"])[:10]:
        R.append("  " + describe(m))
    eligible = [m for m in metrics if m["length"] >= 15]
    R.append("")
    R.append("Most similar (sections of 15+ words):")
    for m in sorted(eligible, key=lambda m: -m["seq"])[:10]:
        R.append("  " + describe(m))
    R.append("")
    R.append("Least similar (sections of 15+ words; worth checking by hand):")
    for m in sorted(eligible, key=lambda m: m["seq"])[:10]:
        R.append("  " + describe(m))

    report = "\n".join(R) + "\n"
    print(report)

    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("Report written to", out_path)

    if args.images:
        media = os.path.expanduser(args.media)
        for img in (make_charts(media, n, chap, cov, covcount, venn, metrics)
                    + make_extra_charts(media, n, chap, verse, cov, covcount, metrics)):
            print("Image written to", img)

    if args.csv:
        csv_path = os.path.expanduser(args.csv)
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair", "section", "book_a", "ref_a", "words_a", "nonexcluded_a",
                        "start_a", "end_a",
                        "book_b", "ref_b", "words_b", "nonexcluded_b", "start_b", "end_b",
                        "lexical_overlap_pct", "sequence_similarity_pct"])
            for m in metrics:
                pa, pb = m["pa"], m["pb"]
                w.writerow(["%s-%s" % m["pair"], m["n"],
                            pa["book"], pa["ref"], pa["words"],
                            "" if pa["content"] is None else pa["content"],
                            pa["start"], pa["end"],
                            pb["book"], pb["ref"], pb["words"],
                            "" if pb["content"] is None else pb["content"],
                            pb["start"], pb["end"],
                            "%.1f" % m["lex"], "%.1f" % m["seq"]])
        print("Section CSV written to", csv_path)


if __name__ == "__main__":
    main()
