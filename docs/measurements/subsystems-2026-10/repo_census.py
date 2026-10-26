#!/usr/bin/env python3
"""repo_census.py — the git and source facts behind docs/tasks/task-subsystems-2026-10.md §2, §6 and §7.

Read-only. Run from a goinfer checkout (a worktree of origin/main, so another session's tree is not read):

  python3 docs/measurements/subsystems-2026-10/repo_census.py [--since YYYY-MM-DD] [--aikit PATH]

--aikit names an aikit git checkout; its files are read at --aikit-ref (default origin/main) through `git archive`
into a temporary directory, so the checkout's own branch does not matter. Without --aikit the aikit sections are
skipped.

Sections: commit rate and the most-touched files; the parity manifest's shared sets and how often their files change;
aikit bumps in goinfer by what drove them (by commit subject, so approximate); test and non-test files that import
`decoder`, per module; the memory-related file groups; and, with --aikit, the moving packages' sizes, importers,
churn and aikit's tag count.
"""
import argparse
import collections
import glob
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile

DECODER = '"github.com/townsendmerino/goinfer/decoder"'
MEMORY_GROUPS = {
    "residency": ["decoder/residency.go"],
    "fit guard": ["decoder/fitguard.go"],
    "fit plan, query, Metal fit": ["decoder/fitplan.go", "decoder/fitquery.go", "decoder/metal_fitctx.go"],
    "prefill scratch": ["decoder/prefill_budget.go", "decoder/scratch.go", "decoder/prefill_chunk_scratch.go"],
    "weight and resident bytes": ["decoder/weightbytes.go", "decoder/residentneed.go"],
    "KV plan": ["decoder/kvresolve.go", "decoder/kvgeometry.go", "decoder/kvpolicy.go", "decoder/kvscope.go"],
    "pagers": ["decoder/moepool.go", "decoder/moepaging.go", "decoder/moeworkingset.go", "decoder/layerpaging.go"],
    "probes": ["decoder/hostram_*.go", "decoder/memwatch_*.go", "decoder/swapwatch.go", "decoder/faultcount_*.go",
               "decoder/coldload*.go", "decoder/giwmap_*.go"],
    "internal/hostmem": ["internal/hostmem/*.go"],
    "internal/swapguard": ["internal/swapguard/*.go"],
    "internal/fitcmd": ["internal/fitcmd/*.go"],
    "internal/kvmgr": ["internal/kvmgr/*.go"],
}
BUMP_CLASSES = [
    ("towers (vision, audio)", r"vision|tower|siglip|pixtral|\bvl\b|image|audio|whisper|voxtral|asr|mmproj|bicubic"),
    ("GPU module", r"gpu/v|qwencuda|visioncuda|gpu v0|anncuda|annmetal"),
    ("linalg and kernels", r"linalg|kernel|quantiz|matmul|neon|avx|int4|int8|bf16|dotprod|fallback|scales|w4a8|widen|binary16"),
    ("embed, formats, mmap", r"embed|gguf|q4_k|q5k|q6k|dequant|safetensors|mmap|split"),
]
MOVING = ["vision", "audio", "gpu", "gpu/qwencuda", "gpu/qwenmetal", "gpu/visioncuda", "gpu/visionmetal", "gpu/webgpu",
          "gpu/anncuda", "gpu/annmetal", "gpu/enccuda", "gpu/encmetal"]


def git(*args, cwd="."):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout


def count_commits(since, paths, cwd="."):
    out = git("log", f"--since={since}", "--no-merges", "--full-history", "--format=%H", "--", *paths, cwd=cwd)
    return len([x for x in out.splitlines() if x])


def expand(pats):
    files = []
    for p in pats:
        files += [f for f in glob.glob(p) if not f.endswith("_test.go")]
    return sorted(set(files))


def lines(files):
    return sum(sum(1 for _ in open(f, errors="replace")) for f in files)


def go_files(root, test):
    for d, _, fs in os.walk(root):
        if "/." in d or "_to_delete" in d:
            continue
        for f in fs:
            if f.endswith(".go") and f.endswith("_test.go") == test:
                yield os.path.join(d, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-26")
    ap.add_argument("--aikit")
    ap.add_argument("--aikit-ref", default="origin/main")
    a = ap.parse_args()
    head = git("log", "-1", "--format=%h %ad", "--date=short").strip()
    print(f"# repository census at {head}, window since {a.since}\n")

    shas = [x for x in git("log", f"--since={a.since}", "--no-merges", "--format=%ad", "--date=short").splitlines() if x]
    print(f"## rate: {len(shas)} non-merge commits; last ten days:")
    for day, n in sorted(collections.Counter(shas).items())[-10:]:
        print(f"  {day} {n}")
    names = [x for x in git("log", f"--since={a.since}", "--no-merges", "--name-only", "--format=").splitlines() if x]
    print("\n## most-touched files")
    for f, n in collections.Counter(names).most_common(25):
        print(f"  {n:4d}  {f}")

    m = json.load(open("testdata/parity_manifest.json"))
    uses = collections.Counter(u for v in m["families"].values() for u in v.get("uses", []))
    print(f"\n## parity manifest: {len(m['families'])} families; sets used: {dict(uses)}")
    for name, files in m["shared_sets"].items():
        for f in files:
            if os.path.exists(f):
                print(f"  {name:10s} {f:32s} {lines([f]):5d} lines {count_commits(a.since, [f]):4d} commits")
    adapters = len(re.findall(r"^func \w+Architecture\(cfg \*Config\)", open("decoder/registry.go").read(), re.M))
    print(f"  decoder/registry.go holds {adapters} family adapters")

    subjects = git("log", f"--since={a.since}", "--no-merges", "--full-history", "-G", "townsendmerino/aikit",
                   "--format=%s", "--", "go.mod").splitlines()
    cls = collections.Counter()
    for s in subjects:
        cls[next((k for k, rx in BUMP_CLASSES if re.search(rx, s.lower())), "other")] += 1
    print(f"\n## aikit requirement changes in go.mod: {len(subjects)}; by subject: {dict(cls)}")

    print("\n## files importing decoder")
    for mod in ("cuda", "metal", "gpu", "internal/serveapp"):
        t = list(go_files(mod, True))
        n = list(go_files(mod, False))
        ti = sum(1 for f in t if DECODER in open(f, errors="replace").read())
        ni = [f for f in n if DECODER in open(f, errors="replace").read()]
        ids = set()
        for f in n:
            ids |= set(re.findall(r"\bdecoder\.[A-Z]\w*", open(f, errors="replace").read()))
        print(f"  {mod:18s} test files {len(t):4d}, importing decoder {ti:4d}; non-test importing {len(ni):3d}; "
              f"distinct decoder identifiers {len(ids)}")

    print("\n## memory-related file groups")
    allf = []
    for name, pats in MEMORY_GROUPS.items():
        fs = expand(pats)
        allf += [f for f in fs if f.startswith("decoder/")]
        print(f"  {name:28s} {len(fs):3d} files {lines(fs):6d} lines {count_commits(a.since, fs):4d} commits")
    print(f"  decoder files above: {len(allf)}, {lines(allf)} lines, {count_commits(a.since, allf)} commits")

    if not a.aikit:
        return
    repo = os.path.expanduser(a.aikit)
    ref = a.aikit_ref
    tags = git("for-each-ref", "--format=%(creatordate:short) %(refname:short)", "refs/tags", cwd=repo).splitlines()
    recent = [t.split()[1] for t in tags if t.split()[0] >= a.since]
    print(f"\n## aikit at {ref} ({git('log', '-1', '--format=%h', ref, cwd=repo).strip()}): tags since {a.since}: "
          f"{sum(1 for t in recent if re.match(r'v[0-9]', t))} root, {sum(1 for t in recent if t.startswith('gpu/'))} gpu/*")
    with tempfile.TemporaryDirectory() as ak:
        tar = subprocess.run(["git", "archive", ref], cwd=repo, capture_output=True).stdout
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(ak, filter="data")
        imp = re.compile(r'"github\.com/townsendmerino/aikit([^"]*)"')
        graph = collections.defaultdict(set)
        for f in go_files(ak, False):
            rel = os.path.relpath(os.path.dirname(f), ak)
            for x in imp.findall(open(f, errors="replace").read()):
                graph[rel].add(x or "/")
        for pkg in MOVING:
            src = [f for f in glob.glob(os.path.join(ak, pkg, "*.go")) if not f.endswith("_test.go")]
            tst = glob.glob(os.path.join(ak, pkg, "*_test.go"))
            users = sorted(k for k, v in graph.items() if "/" + pkg in v and k != pkg)
            log = lambda *p: len([x for x in git("log", ref, f"--since={a.since}", "--no-merges", "--format=%H", "--",
                                                 *p, cwd=repo).splitlines() if x])
            print(f"  {pkg:16s} {len(src):3d} files {lines(src):6d} lines {len(tst):3d} tests; imports "
                  f"{sorted(graph.get(pkg, []))}; aikit importers {users}; commits {log(pkg)} "
                  f"({log(pkg + '/*.go')} touching .go)")


if __name__ == "__main__":
    main()
