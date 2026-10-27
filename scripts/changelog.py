#!/usr/bin/env python3
"""The CHANGELOG's unreleased entries, one file per change in changelog.d/, and the release step that collects them.

    python3 scripts/changelog.py render                          # print the unreleased entries, newest first
    python3 scripts/changelog.py release vX.Y.Z --date YYYY-MM-DD  # write CHANGELOG.md's [vX.Y.Z] section, delete the fragments

A fragment is one entry exactly as it reads in the CHANGELOG (a `- **...**` bullet and its indented lines), in
changelog.d/YYYY-MM-DD-<slug>.md. Entries are rendered newest first, by file name descending, and joined verbatim, so two
pull requests that each add an entry touch different files and never conflict. CHANGELOG.md's [Unreleased] section holds only
the line pointing here; `release` puts the rendered entries under a dated header right after it, which is the move a release
used to make by hand, and removes the fragment files (task-subsystems-2026-10.md, PW3).
"""
import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRAGMENTS = "changelog.d"
NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9._-]*\.md$")
UNRELEASED = "## [Unreleased]\n"


def fragments(root: pathlib.Path) -> list[pathlib.Path]:
    """The fragment files, newest first. A file in changelog.d/ that is not README.md and not named YYYY-MM-DD-<slug>.md is
    an error: it would be skipped otherwise, and its entry lost at release."""
    d = root / FRAGMENTS
    out, bad = [], []
    for p in sorted(d.iterdir()) if d.is_dir() else []:
        if p.name == "README.md":
            continue
        (out if NAME_RE.match(p.name) else bad).append(p)
    if bad:
        sys.exit("changelog: not a fragment name (want YYYY-MM-DD-<slug>.md): " + ", ".join(p.name for p in bad))
    return sorted(out, key=lambda p: p.name, reverse=True)


def render(root: pathlib.Path) -> str:
    """The unreleased entries as the [Unreleased] section's body read before fragments: a blank line, then each fragment
    verbatim, then a blank line. Empty when there are none."""
    texts = []
    for p in fragments(root):
        t = p.read_text(encoding="utf-8")
        if not t.startswith("- ") or not t.endswith("\n"):
            sys.exit(f"changelog: {p.name}: a fragment is one `- ` entry ending in a newline")
        texts.append(t)
    return "\n" + "".join(texts) + "\n" if texts else ""


def release(root: pathlib.Path, version: str, date: str) -> None:
    """Write `## [version] — date` with the rendered entries under the [Unreleased] section, and delete the fragments."""
    if not re.fullmatch(r"v\d+\.\d+\.\d+", version) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        sys.exit("changelog: release wants vX.Y.Z and --date YYYY-MM-DD")
    body = render(root)
    if not body:
        sys.exit("changelog: no fragments in changelog.d/; nothing to release")
    path = root / "CHANGELOG.md"
    doc = path.read_text(encoding="utf-8")
    if f"## [{version}]" in doc:
        sys.exit(f"changelog: CHANGELOG.md already has a [{version}] section")
    i = doc.find(UNRELEASED)
    if i < 0:
        sys.exit("changelog: CHANGELOG.md has no `## [Unreleased]` section")
    j = doc.find("\n## [", i + len(UNRELEASED))
    j = len(doc) if j < 0 else j + 1
    doc = doc[:j] + f"## [{version}] — {date}\n" + body + doc[j:]
    path.write_text(doc, encoding="utf-8")
    for p in fragments(root):
        subprocess.run(["git", "rm", "-q", "--", str(p.relative_to(root))], cwd=root, check=True)
    print(f"changelog: wrote [{version}] — {date} from {body.count(chr(10) + '- ')} entries; the fragments are removed (staged)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("render", help="print the unreleased entries")
    r = sub.add_parser("release", help="write the release's section and delete the fragments")
    r.add_argument("version")
    r.add_argument("--date", required=True)
    a = ap.parse_args()
    if a.cmd == "render":
        sys.stdout.write(render(ROOT))
    else:
        release(ROOT, a.version, a.date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
