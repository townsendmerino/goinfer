#!/usr/bin/env python3
"""doc_id_collisions.py — find a queue-item ID (P24, G23, B8, ...) defined twice in the same file.

This repo tracks open work in docs/queue-*.md (and docs/QUEUE.md itself) as short bulleted items —
a letter per queue plus a number, always in the shape `**ID · text**` (a plain bold bullet in
docs/queue-engineering.md's convention, or `- **ID · text**` as a list item in
docs/queue-performance.md's). Two different findings landing under the same ID is a real, recurring
bug here, not a hypothetical: as of 2026-09-13 this script found **G23** defined twice in
docs/queue-engineering.md (a `.repowise/` git-history purge decision at one location, an unrelated
T3 cosine-bar calibration at another), **B8** defined twice in the same file (two unrelated
findings), and **P24** defined twice in docs/queue-performance.md (an attn_fused occupancy finding
and an unrelated KV-depth decode-falloff finding) — the second one already half-caught in the
surrounding prose ("Filed as P24, not P21: P21-P23 were taken by another session's filing...")
without anyone then noticing P24 itself collided too.

TWO THINGS THAT LOOK LIKE COLLISIONS AND ARE NOT, LEARNED FROM THIS REPO'S OWN CONVENTIONS:

  1. A `**ID (original) · text**` bullet next to a plain `**ID · text**` one is a DELIBERATE
     retraction pair — the corrected/current item and the superseded original kept side by side for
     history (docs/queue-release.md's D3/D3b, docs/queue-engineering.md's B4, docs/queue-correctness.md's
     G8 all do this). Any hit whose parenthetical contains "original" is excluded from the collision
     count; it's expected to share an ID with exactly one non-original sibling.
  2. AN INLINE MENTION DOES NOT DEFINE THE ITEM. "**E2's obligation is unchanged...**" and
     "- **D3** has no expression-rewriting exposure..." both look like item openers to a loose regex,
     but neither is a definition — real items in this tree always put " · " directly after the ID,
     inline mentions never do. Requiring that separator is what tells the two apart mechanically,
     without a hand-maintained exclude list.

Markdown HEADINGS (`## ID ...`, `#### ID ...`) are deliberately NOT checked at all: this repo reuses
the same ID across multiple headings on purpose to mark a finding's own lifecycle
(docs/QUEUE.md's `## G33 · SCOPE — ...` followed later by `## G33 RESULT · ...` is the SAME item,
scoped then resolved, not two). A heading-based version of this check would need the same kind of
learned exclusion as above, or a progression-word allowlist, before it would be trustworthy — not
attempted here.

This is a read-only report, not a push gate. Run it by hand before filing a new item (to see
whether the next number you're about to use is already taken) or periodically to catch a collision
that already happened.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# "**P24 · ..." or "- **P24 · ..." (the leading dash is optional; both are real conventions here),
# with an optional "(original)"-style parenthetical annotation before the required " · " separator
# that distinguishes a real definition from an inline mention of the same ID.
BULLET_RE = re.compile(r"^(?:-\s+)?\*\*([A-Z]{1,4}-?\d{1,4}[a-z]?)\s*(\([^)]*\))?\s*·")


def scoped_files():
    """docs/queue-*.md and docs/QUEUE.md itself — where this ID convention actually lives.
    Excludes docs/completed/ (frozen; a duplicate there is historical, not something to fix)."""
    out = list(sorted((ROOT / "docs").glob("queue-*.md")))
    queue_md = ROOT / "docs" / "QUEUE.md"
    if queue_md.exists():
        out.append(queue_md)
    return out


def find_collisions(path: pathlib.Path):
    """id -> [(line_no, full_line_text), ...] for every id with 2+ NON-"(original)" definitions."""
    hits = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = BULLET_RE.match(line)
        if not m:
            continue
        item_id, paren = m.group(1), m.group(2)
        is_original = bool(paren) and "original" in paren.lower()
        hits.setdefault(item_id, []).append((lineno, line.strip(), is_original))

    collisions = {}
    for item_id, entries in hits.items():
        current = [(ln, txt) for ln, txt, is_orig in entries if not is_orig]
        if len(current) > 1:
            collisions[item_id] = current
    return collisions


def main() -> int:
    any_collision = False
    for path in scoped_files():
        collisions = find_collisions(path)
        if not collisions:
            continue
        any_collision = True
        rel = path.relative_to(ROOT)
        print(f"{rel}:")
        for item_id, hits in sorted(collisions.items()):
            print(f"  {item_id} defined {len(hits)} times:")
            for lineno, line in hits:
                shown = line if len(line) <= 100 else line[:97] + "..."
                print(f"    :{lineno}  {shown}")
        print()

    if not any_collision:
        print("doc_id_collisions: no duplicate queue-item IDs found in docs/QUEUE.md or docs/queue-*.md.")
        return 0

    print(
        "Each ID above names two different findings under the same number, within one file. "
        "Rename the newer one to the next free number in its series (or fold the two together if "
        "they're actually the same finding) — do not just pick one to keep silently, since a live "
        "cross-reference may point at either. If a hit above is actually a deliberate retraction "
        "pair or lifecycle update that this script's exclusions don't yet recognize, that's worth "
        "teaching the script, not just ignoring the report."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
