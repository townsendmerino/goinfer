#!/usr/bin/env python3
"""PW3: move CHANGELOG.md's [Unreleased] entries into changelog.d/ fragments, once, keeping their order.

    python3 docs/measurements/pw3-2026-10/migrate_changelog.py

Each entry is dated by the commit that added its first line (git blame), the dates are held non-increasing down the list,
and a two-digit index breaks ties, so scripts/changelog.py's newest-first order (file names descending) gives the list back
in its order. The [Unreleased] body becomes the pointer line.
"""
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
POINTER = ("Each unreleased change is a file in [`changelog.d/`](changelog.d/), collected into the release's section when it is "
           "tagged (`scripts/changelog.py release`).\n")


def main():
    path = ROOT / "CHANGELOG.md"
    doc = path.read_text(encoding="utf-8")
    head = "## [Unreleased]\n"
    i = doc.index(head) + len(head)
    j = doc.index("\n## [", i) + 1
    body = doc[i:j]
    assert body.startswith("\n") and body.endswith("\n\n")
    lines = body[1:-1].splitlines(keepends=True)
    entries, cur = [], None
    for k, ln in enumerate(lines):
        if ln.startswith("- "):
            cur = [k, [ln]]
            entries.append(cur)
        else:
            cur[1].append(ln)
    start = doc[:i].count("\n") + 2  # 1-based line number of the first entry line
    dates = []
    for k, _ in entries:
        n = start + k
        out = subprocess.run(["git", "blame", "-L", f"{n},{n}", "--porcelain", "HEAD", "--", "CHANGELOG.md"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        t = int(re.search(r"^committer-time (\d+)", out, re.M).group(1))
        dates.append(subprocess.run(["date", "-r", str(t), "+%Y-%m-%d"], capture_output=True, text=True).stdout.strip())
    for x in range(1, len(dates)):
        dates[x] = min(dates[x], dates[x - 1])
    frag = ROOT / "changelog.d"
    frag.mkdir(exist_ok=True)
    seen = {}
    for (k, text), d in zip(entries, dates):
        seen[d] = seen.get(d, 0) + 1
    idx = {}
    for (k, text), d in zip(entries, dates):
        idx[d] = idx.get(d, 0) + 1
        n = seen[d] - idx[d] + 1  # descending within a day, so name order descending is list order
        title = re.sub(r"[^a-z0-9]+", "-", re.sub(r"`|\*", "", text[0][2:]).lower()).strip("-")
        slug = "-".join(title.split("-")[:6])
        (frag / f"{d}-{n:02d}-{slug}.md").write_text("".join(text), encoding="utf-8")
    path.write_text(doc[:i] + "\n" + POINTER + "\n" + doc[j:], encoding="utf-8")
    print(f"{len(entries)} entries into changelog.d/")


if __name__ == "__main__":
    main()
