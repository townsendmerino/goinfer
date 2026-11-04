#!/usr/bin/env bash
# scripts/layout/tier2-2.sh — PR 2 of repo layout Tier 2: internal/ regroup (MOVE ONLY).
#
# Moves 34 top-level internal/ package directories into four group directories:
#   - internal/app/      <- serveapp chatapp cliutil fitcmd decidecmd pullcmd inproc connect loadflags
#   - internal/guard/    <- swapguard prezerolint paritykey movedpaths servecheck memgate
#   - internal/diag/     <- hwcensus hwreport kernelprint sb2census towerid
#   - internal/runtime/  <- giw mem kvmgr f16 mamba2 whisper scheduler adapter hybridattn program modelload prequant diskfree sampling
#
# Package clauses do NOT change (directory basenames are kept).
# Must run from a clean checkout (before internal/app exists).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Idempotency guard: refuse if any of the target group directories already exists.
for grp in app guard diag runtime; do
  if [[ -e "internal/$grp" ]]; then
    echo "ERROR: internal/$grp already exists; refusing to run (script is idempotent-guarded)." >&2
    exit 1
  fi
done

APP_PKGS=(serveapp chatapp cliutil fitcmd decidecmd pullcmd inproc connect loadflags)
GUARD_PKGS=(swapguard prezerolint paritykey movedpaths servecheck memgate)
DIAG_PKGS=(hwcensus hwreport kernelprint sb2census towerid)
RUNTIME_PKGS=(giw mem kvmgr f16 mamba2 whisper scheduler adapter hybridattn program modelload prequant diskfree sampling)

# Verify all source directories exist before moving anything.
for pkg in "${APP_PKGS[@]}" "${GUARD_PKGS[@]}" "${DIAG_PKGS[@]}" "${RUNTIME_PKGS[@]}"; do
  if [[ ! -d "internal/$pkg" ]]; then
    echo "ERROR: expected source directory internal/$pkg not found." >&2
    exit 1
  fi
done

mkdir -p internal/app internal/guard internal/diag internal/runtime

for pkg in "${APP_PKGS[@]}"; do
  git mv "internal/$pkg" "internal/app/$pkg"
done
for pkg in "${GUARD_PKGS[@]}"; do
  git mv "internal/$pkg" "internal/guard/$pkg"
done
for pkg in "${DIAG_PKGS[@]}"; do
  git mv "internal/$pkg" "internal/diag/$pkg"
done
for pkg in "${RUNTIME_PKGS[@]}"; do
  git mv "internal/$pkg" "internal/runtime/$pkg"
done

python3 - <<'PY'
import json
import os
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path.cwd()

MAPPING = {
    # internal/app/
    "serveapp": "app",
    "chatapp": "app",
    "cliutil": "app",
    "fitcmd": "app",
    "decidecmd": "app",
    "pullcmd": "app",
    "inproc": "app",
    "connect": "app",
    "loadflags": "app",
    # internal/guard/
    "swapguard": "guard",
    "prezerolint": "guard",
    "paritykey": "guard",
    "movedpaths": "guard",
    "servecheck": "guard",
    "memgate": "guard",
    # internal/diag/
    "hwcensus": "diag",
    "hwreport": "diag",
    "kernelprint": "diag",
    "sb2census": "diag",
    "towerid": "diag",
    # internal/runtime/
    "giw": "runtime",
    "mem": "runtime",
    "kvmgr": "runtime",
    "f16": "runtime",
    "mamba2": "runtime",
    "whisper": "runtime",
    "scheduler": "runtime",
    "adapter": "runtime",
    "hybridattn": "runtime",
    "program": "runtime",
    "modelload": "runtime",
    "prequant": "runtime",
    "diskfree": "runtime",
    "sampling": "runtime",
}

NAMES_SORTED = sorted(MAPPING.keys(), key=len, reverse=True)
NAMES_ALT = "|".join(re.escape(k) for k in NAMES_SORTED)


def fail(file_path: str, expected: str) -> None:
    sys.stderr.write(
        f"ERROR: expected string not found in {file_path}:\n  {expected!r}\n"
    )
    sys.exit(1)


MISSES: list[tuple[str, str]] = []


def replace_exact(rel_path: str, pairs: list[tuple[str, str]]) -> None:
    p = REPO_ROOT / rel_path
    if not p.is_file():
        MISSES.append((rel_path, "<file does not exist>"))
        return
    text = p.read_text(encoding="utf-8")
    changed = False
    for old, new in pairs:
        if old not in text:
            MISSES.append((rel_path, old))
            continue
        text = text.replace(old, new)
        changed = True
    if changed:
        p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Rewrite quoted Go import strings across all .go files (including tests
#    and build-tagged files), check per-file replacement counts, and handle
#    any stdlib "runtime" vs "github.com/townsendmerino/goinfer/internal/runtime"
#    package-name collision by aliasing the internal import as `rt`.
# ---------------------------------------------------------------------------
IMPORT_RE = re.compile(
    r'"github\.com/townsendmerino/goinfer/internal/(' + NAMES_ALT + r')(/[^"]*)?"'
)
NEW_IMPORT_RE = re.compile(
    r'"github\.com/townsendmerino/goinfer/internal/(?:app|guard|diag|runtime)/('
    + NAMES_ALT
    + r')(/[^"]*)?"'
)
STDLIB_RUNTIME_RE = re.compile(r'(?m)^\s*"runtime"\s*(?://.*)?$')
INTERNAL_RUNTIME_DIRECT_RE = re.compile(
    r'(?m)^(\s*)"github\.com/townsendmerino/goinfer/internal/runtime"\s*(?://.*)?$'
)


def rewrite_import_match(m: re.Match) -> str:
    name = m.group(1)
    rest = m.group(2) or ""
    group = MAPPING[name]
    return f'"github.com/townsendmerino/goinfer/internal/{group}/{name}{rest}"'


total_rewritten_imports = 0
total_go_files_modified = 0
runtime_alias_files: list[str] = []

for root, dirs, files in os.walk(REPO_ROOT):
    dirs[:] = [d for d in dirs if d not in (".git", ".claude", "vendor", "node_modules")]
    for fname in files:
        if not fname.endswith(".go"):
            continue
        fpath = pathlib.Path(root) / fname
        rel = fpath.relative_to(REPO_ROOT).as_posix()
        content = fpath.read_text(encoding="utf-8")

        before_old_count = len(IMPORT_RE.findall(content))
        before_new_count = len(NEW_IMPORT_RE.findall(content))

        if before_old_count > 0:
            new_content, n_subs = IMPORT_RE.subn(rewrite_import_match, content)
            if n_subs != before_old_count:
                fail(rel, f"import count mismatch: expected {before_old_count}, got {n_subs}")
            after_old_count = len(IMPORT_RE.findall(new_content))
            after_new_count = len(NEW_IMPORT_RE.findall(new_content))
            if after_old_count != 0 or after_new_count != before_old_count + before_new_count:
                fail(
                    rel,
                    f"post-rewrite import count check failed: remaining={after_old_count}, new={after_new_count}",
                )
            content = new_content
            total_rewritten_imports += n_subs
            total_go_files_modified += 1
            fpath.write_text(content, encoding="utf-8")

        # Trap 1: If a file imports both stdlib "runtime" and directly imports
        # "github.com/townsendmerino/goinfer/internal/runtime" (package runtime),
        # alias the internal one as `rt` and update its selector uses.
        if STDLIB_RUNTIME_RE.search(content) and INTERNAL_RUNTIME_DIRECT_RE.search(content):
            STDLIB_SYMS = {
                "GC", "Goexit", "GOMAXPROCS", "GOROOT", "Gosched", "KeepAlive",
                "LockOSThread", "MemProfile", "MemProfileRate", "MemStats",
                "NumCPU", "NumCgoCall", "NumGoroutine", "ReadMemStats",
                "SetFinalizer", "Stack", "StartTrace", "StopTrace",
                "ThreadCreateProfile", "UnlockOSThread", "Version", "Caller",
                "Callers", "CallersFrames", "FuncForPC", "Frame", "Frames",
                "Pinner", "AddCleanup", "GOARCH", "GOOS", "BlockProfile",
                "CPUProfile", "GoroutineProfile", "MutexProfile",
                "SetBlockProfileRate", "SetCPUProfileRate",
                "SetMutexProfileFraction", "SetCgoTraceback",
            }
            content = INTERNAL_RUNTIME_DIRECT_RE.sub(
                r'\1rt "github.com/townsendmerino/goinfer/internal/runtime"', content
            )

            def replace_rt_ident(m: re.Match) -> str:
                ident = m.group(1)
                if ident in STDLIB_SYMS:
                    return m.group(0)
                return f"rt.{ident}"

            content = re.sub(r"\bruntime\.([A-Z][A-Za-z0-9_]*)\b", replace_rt_ident, content)
            fpath.write_text(content, encoding="utf-8")
            runtime_alias_files.append(rel)

if total_rewritten_imports == 0:
    fail("*.go", "no Go import strings were rewritten")


# ---------------------------------------------------------------------------
# 2. Update internal/layout/* repository-shape ratchets and path-dependent
#    Go test / gate strings.
# ---------------------------------------------------------------------------
INTERNAL_SUBPATH_RE = re.compile(r"\binternal/(" + NAMES_ALT + r")\b(/[^\s\"'`)]*)?")


def map_internal_path(s: str) -> str:
    def _sub(m: re.Match) -> str:
        name = m.group(1)
        rest = m.group(2) or ""
        return f"internal/{MAPPING[name]}/{name}{rest}"

    return INTERNAL_SUBPATH_RE.sub(_sub, s)


# 2a. internal/layout/imports.json (schema: _about, rules, arrows, test_arrows)
imports_json_path = REPO_ROOT / "internal/layout/imports.json"
if not imports_json_path.is_file():
    fail("internal/layout/imports.json", "<file does not exist>")
raw_imports = imports_json_path.read_text(encoding="utf-8")
if '"internal/serveapp"' not in raw_imports or '"internal/mem"' not in raw_imports:
    fail("internal/layout/imports.json", '"internal/serveapp" / "internal/mem"')

imp_data = json.loads(raw_imports)
for rule in imp_data.get("rules", []):
    if "from" in rule:
        rule["from"] = [map_internal_path(x) for x in rule["from"]]
    if "forbid_goinfer" in rule:
        rule["forbid_goinfer"] = [map_internal_path(x) for x in rule["forbid_goinfer"]]
    if "allow_goinfer" in rule:
        rule["allow_goinfer"] = {
            map_internal_path(k): [map_internal_path(x) for x in v]
            for k, v in rule["allow_goinfer"].items()
        }
for arrow_key in ("arrows", "test_arrows"):
    if arrow_key in imp_data:
        new_arrows = {}
        for k, v in imp_data[arrow_key].items():
            nk = map_internal_path(k)
            new_arrows[nk] = sorted(map_internal_path(x) for x in v)
        imp_data[arrow_key] = {k: new_arrows[k] for k in sorted(new_arrows)}
imports_json_path.write_text(
    json.dumps(imp_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

# 2b. internal/layout/arrows_test.go & internal/layout/decoder_imports_test.go
replace_exact(
    "internal/layout/arrows_test.go",
    [
        (
            '{"internal/audio", g("internal/mem"), true},',
            '{"internal/audio", g("internal/runtime/mem"), true},',
        ),
    ],
)
replace_exact(
    "internal/layout/decoder_imports_test.go",
    [
        (
            'if f.dir != "internal/serveapp" || !f.test {',
            'if f.dir != "internal/app/serveapp" || !f.test {',
        ),
    ],
)

# 2c. Explicit checks on known path-dependent Go tests
replace_exact(
    "internal/app/serveapp/flagsdoc_test.go",
    [
        (
            "go test ./internal/serveapp -run TestFlagsDoc -update",
            "go test ./internal/app/serveapp -run TestFlagsDoc -update",
        ),
    ],
)
# 2d. Rewrite any remaining quoted string literals in .go files that embed
#     moved `internal/<pkg>` paths (e.g. cmd/gate/*.go, internal/runtime/mem/*,
#     decoder/*), without touching regular comments, PLUS update `go test ./internal/<pkg>`
#     in package doc comments enforced by TestPackageDocs_namedTestsExist in
#     internal/layout/docs_test.go.
QUOTED_INTERNAL_RE = re.compile(
    r'("(?:[^"\\]|\\.)*"|`[^`]*`)'
)
STRING_INTERNAL_PATH_RE = re.compile(
    r'(^|(?<=[\s/"\'`]))internal/(' + NAMES_ALT + r')(?=[/"\'`\s)]|$)'
)
PKGDOC_GOTEST_RE = re.compile(
    r"(?m)^(\s*//[^\n]*?\bgo\s+test\b[^\n]*?\s\./)internal/(" + NAMES_ALT + r")\b"
)


def rewrite_quoted_literal(m: re.Match) -> str:
    lit = m.group(1)
    return STRING_INTERNAL_PATH_RE.sub(
        lambda sm: f"{sm.group(1)}internal/{MAPPING[sm.group(2)]}/{sm.group(2)}",
        lit,
    )


for root, dirs, files in os.walk(REPO_ROOT):
    dirs[:] = [d for d in dirs if d not in (".git", ".claude", "vendor", "node_modules")]
    for fname in files:
        if not fname.endswith(".go"):
            continue
        fpath = pathlib.Path(root) / fname
        text = fpath.read_text(encoding="utf-8")
        # Rewrite string literals line-by-line (skipping full-line // comments)
        lines = text.splitlines(keepends=True)
        changed = False
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("//"):
                # Only update `go test ... ./internal/<moved>` in // comments so
                # TestPackageDocs_namedTestsExist in internal/layout/docs_test.go resolves
                # the moved package directory.
                new_line = PKGDOC_GOTEST_RE.sub(
                    lambda gm: f"{gm.group(1)}internal/{MAPPING[gm.group(2)]}/{gm.group(2)}",
                    line,
                )
                if new_line != line:
                    lines[i] = new_line
                    changed = True
                continue
            # Split before any trailing // comment outside quotes
            new_line = QUOTED_INTERNAL_RE.sub(rewrite_quoted_literal, line)
            if new_line != line:
                lines[i] = new_line
                changed = True
        if changed:
            fpath.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. Embedded/relative paths (`..` depth adjustments in moved packages)
# ---------------------------------------------------------------------------
replace_exact(
    "internal/app/cliutil/cliutil_test.go",
    [
        (
            'filepath.Join("..", "..")',
            'filepath.Join("..", "..", "..")',
        ),
    ],
)

replace_exact(
    "internal/app/fitcmd/defaultctx_test.go",
    [
        (
            'os.ReadFile("../../backend/cuda/resident.go")',
            'os.ReadFile("../../../backend/cuda/resident.go")',
        ),
    ],
)

replace_exact(
    "internal/app/serveapp/flagsdoc_test.go",
    [
        (
            'filepath.Join("..", "..")',
            'filepath.Join("..", "..", "..")',
        ),
    ],
)

replace_exact(
    "internal/guard/movedpaths/movedpaths_test.go",
    [
        (
            'filepath.WalkDir("../..",',
            'filepath.WalkDir("../../..",',
        ),
    ],
)


# ---------------------------------------------------------------------------
# 4. Non-Go references: GitHub workflows, scripts, parity manifests, webui, docs
# ---------------------------------------------------------------------------

# 4a. .github/workflows/*.yml
replace_exact(
    ".github/workflows/ci.yml",
    [
        (
            "./internal/sampling/",
            "./internal/runtime/sampling/",
        ),
        (
            "^github.com/townsendmerino/goinfer/internal/sampling$",
            "^github.com/townsendmerino/goinfer/internal/runtime/sampling$",
        ),
    ],
)

# 4b. scripts/touched.py
replace_exact(
    "scripts/touched.py",
    [
        ('"internal/sampling"', '"internal/runtime/sampling"'),
        ('"internal/serveapp"', '"internal/app/serveapp"'),
    ],
)

# 4c. scripts/touched_scans.py & scripts/test_touched.py
for script_rel in ("scripts/touched_scans.py", "scripts/test_touched.py"):
    sp = REPO_ROOT / script_rel
    if not sp.is_file():
        fail(script_rel, "<file does not exist>")
    stext = sp.read_text(encoding="utf-8")
    new_stext, n_subs = INTERNAL_SUBPATH_RE.subn(
        lambda m: f"internal/{MAPPING[m.group(1)]}/{m.group(1)}{m.group(2) or ''}",
        stext,
    )
    if n_subs == 0:
        fail(script_rel, "internal/<moved_pkg>")
    sp.write_text(new_stext, encoding="utf-8")

# 4e. testdata/parity/shared_sets.json
replace_exact(
    "testdata/parity/shared_sets.json",
    [
        (
            '"internal/program/program.go"',
            '"internal/runtime/program/program.go"',
        ),
        (
            '"internal/hybridattn/deltanet.go"',
            '"internal/runtime/hybridattn/deltanet.go"',
        ),
    ],
)

# 4f. WebUI gate scripts
replace_exact(
    "scripts/webui-gate/md-harness.html",
    [
        (
            "../../internal/serveapp/webui/ui/markdown.js",
            "../../internal/app/serveapp/webui/ui/markdown.js",
        ),
    ],
)

for webui_script in ("scripts/webui_md_gate.mjs", "scripts/webui_app_gate.mjs"):
    wp = REPO_ROOT / webui_script
    if not wp.is_file():
        fail(webui_script, "<file does not exist>")
    wtext = wp.read_text(encoding="utf-8")
    orig_wtext = wtext
    wtext = wtext.replace("internal/serveapp", "internal/app/serveapp")
    wtext = wtext.replace("'internal', 'serveapp'", "'internal', 'app', 'serveapp'")
    wtext = wtext.replace('"internal", "serveapp"', '"internal", "app", "serveapp"')
    if wtext == orig_wtext:
        fail(webui_script, "internal/serveapp")
    wp.write_text(wtext, encoding="utf-8")

# 4g. docs/layout.md
replace_exact(
    "docs/layout.md",
    [
        (
            "- `gpu/` -> `backend/webgpu/` (`module github.com/townsendmerino/goinfer/gpu` -> `github.com/townsendmerino/goinfer/backend/webgpu`, `package gpu` -> `package webgpu`)",
            "- `gpu/` -> `backend/webgpu/` (`module github.com/townsendmerino/goinfer/gpu` -> `github.com/townsendmerino/goinfer/backend/webgpu`, `package gpu` -> `package webgpu`)\n"
            "- `internal/{serveapp,chatapp,cliutil,fitcmd,decidecmd,pullcmd,inproc,connect,loadflags}` -> `internal/app/<pkg>` (package clause unchanged)\n"
            "- `internal/{swapguard,prezerolint,paritykey,movedpaths,servecheck,memgate}` -> `internal/guard/<pkg>` (package clause unchanged)\n"
            "- `internal/{hwcensus,hwreport,kernelprint,sb2census,towerid}` -> `internal/diag/<pkg>` (package clause unchanged)\n"
            "- `internal/{giw,mem,kvmgr,f16,mamba2,whisper,scheduler,adapter,hybridattn,program,modelload,prequant,diskfree,sampling}` -> `internal/runtime/<pkg>` (`internal/mem/probe` -> `internal/runtime/mem/probe`; package clause unchanged)",
        ),
        (
            "| `internal/` | packages private to the root module (45 of them: guards, codecs, scheduler, census and fit logic, the app wiring behind `cmd/`); `internal/layout` holds repository-shape checks as tests |",
            "| `internal/` | packages private to the root module, grouped under `internal/app/` (CLI/server apps and wiring), `internal/guard/` (lint/CI/memory/parity guards), `internal/diag/` (hardware census and reports), and `internal/runtime/` (execution, memory, KV, scheduler, model loading, sampling, and runtime kernels), plus top-level domain packages (`audio`, `batchio`, `clef`, `codec`, `confidence`, `config`, `cost`, `decide`, `fidelity`, `layout`, `vision`, `wrappedupload`); `internal/layout` holds repository-shape checks as tests |",
        ),
    ],
)

# 4h. docs/flags.md
replace_exact(
    "docs/flags.md",
    [
        (
            "go test ./internal/serveapp -run TestFlagsDoc -update",
            "go test ./internal/app/serveapp -run TestFlagsDoc -update",
        ),
    ],
)

# 4i. Regenerate docs/ARCHITECTURE.md import appendix from updated imports.json
#     (matching TestArchitecture_importAppendix in internal/layout/docs_test.go)
#     and update live internal/<pkg> references in the rest of docs/ARCHITECTURE.md.
APPENDIX_BEGIN = "<!-- imports:begin (generated from internal/layout/imports.json; go test -run TestArchitecture_importAppendix ./internal/layout/ -update) -->"
APPENDIX_END = "<!-- imports:end -->"


def render_import_appendix(data: dict) -> str:
    lines: list[str] = [APPENDIX_BEGIN, ""]
    lines.append(
        "_Generated from `internal/layout/imports.json`; "
        "`go test ./internal/layout/` checks both files against the tree._"
    )
    lines.append("")
    lines.append("### Layer rules")
    lines.append("")
    lines.append("| Rule | From | Forbidden `goinfer/…` imports | Exceptions |")
    lines.append("|---|---|---|---|")
    for r in data.get("rules", []):
        exc = "—"
        allow = r.get("allow_goinfer") or {}
        if allow:
            parts = []
            for k in sorted(allow.keys()):
                parts.append(f"`{k}` → " + ", ".join(f"`{x}`" for x in allow[k]))
            exc = "; ".join(parts)
        from_s = ", ".join(f"`{x}`" for x in r.get("from", []))
        forbid_s = ", ".join(f"`{x}`" for x in r.get("forbid_goinfer", []))
        lines.append(f"| {r['name']} | {from_s} | {forbid_s} | {exc} |")
    lines.append("")
    lines.append("### Package import arrows (`goinfer/…` only)")
    lines.append("")
    lines.append("| Package directory | Production imports | Test-only imports |")
    lines.append("|---|---|---|")
    arrows = data.get("arrows", {})
    test_arrows = data.get("test_arrows", {})
    all_keys = sorted(set(arrows.keys()) | set(test_arrows.keys()))

    def code_list(xs: list[str]) -> str:
        if not xs:
            return "—"
        return ", ".join(f"`{x}`" for x in xs)

    for k in all_keys:
        prod = code_list(arrows.get(k, []))
        tst = code_list(test_arrows.get(k, []))
        lines.append(f"| `{k}` | {prod} | {tst} |")
    lines.append("")
    lines.append(APPENDIX_END)
    return "\n".join(lines)


arch_path = REPO_ROOT / "docs/ARCHITECTURE.md"
if not arch_path.is_file():
    fail("docs/ARCHITECTURE.md", "<file does not exist>")
arch_text = arch_path.read_text(encoding="utf-8")
bi = arch_text.find(APPENDIX_BEGIN)
ei = arch_text.find(APPENDIX_END)
if bi < 0 or ei < 0 or ei < bi:
    fail("docs/ARCHITECTURE.md", APPENDIX_BEGIN)
ei += len(APPENDIX_END)
before_app = map_internal_path(arch_text[:bi])
after_app = map_internal_path(arch_text[ei:])
arch_path.write_text(
    before_app + render_import_appendix(imp_data) + after_app, encoding="utf-8"
)

# 4j. Live docs: README.md, CLAUDE.md, cmd/gate/README.md, models/README.md,
#     docs/*.md, docs/code-notes/*.md, docs/tasks/*.md
#     Explicitly skip all forbidden historical/audit files and directories:
#       docs/completed/, docs/measurements/, CHANGELOG.md, changelog.d/,
#       THIRD_PARTY_LICENSES.md, testdata/gate/goldens/,
#       docs/audit-security-2026-10-12.md, docs/tasks/task-flint-2026-10.md, .claude/
FORBIDDEN_DOC_REL = {
    "CHANGELOG.md",
    "THIRD_PARTY_LICENSES.md",
    "docs/audit-security-2026-10-12.md",
    "docs/tasks/task-flint-2026-10.md",
    "docs/layout.md",        # already updated with exact historical + live entries above
    "docs/ARCHITECTURE.md",  # already updated above
    "docs/flags.md",         # already updated above
}

candidate_docs: list[pathlib.Path] = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / "cmd/gate/README.md",
    REPO_ROOT / "models/README.md",
]
for sub in ("docs", "docs/code-notes", "docs/tasks"):
    subdir = REPO_ROOT / sub
    if subdir.is_dir():
        for child in sorted(subdir.iterdir()):
            if child.is_file() and child.suffix == ".md":
                candidate_docs.append(child)

for doc_path in candidate_docs:
    if not doc_path.is_file():
        continue
    rel = doc_path.relative_to(REPO_ROOT).as_posix()
    if rel in FORBIDDEN_DOC_REL:
        continue
    if rel.startswith(("docs/completed/", "docs/measurements/", "changelog.d/", ".claude/")):
        continue
    dtext = doc_path.read_text(encoding="utf-8")
    new_dtext = map_internal_path(dtext)
    if new_dtext != dtext:
        doc_path.write_text(new_dtext, encoding="utf-8")

# ---------------------------------------------------------------------------
# 5. Create changelog fragment changelog.d/2026-11-04-layout-tier2-internal.md
# ---------------------------------------------------------------------------
changelog_path = REPO_ROOT / "changelog.d/2026-11-04-layout-tier2-internal.md"
changelog_path.write_text(
    "### Changed\n"
    "- Regrouped `internal/` packages under `internal/app/`, `internal/guard/`, "
    "`internal/diag/`, and `internal/runtime/` (Tier 2 repository layout move; "
    "no behavior or public API change).\n",
    encoding="utf-8",
)

if MISSES:
    sys.stderr.write(f"ERROR: {len(MISSES)} replace_exact miss(es):\n")
    for file_path, expected in MISSES:
        sys.stderr.write(
            f"  {file_path}: expected string not found:\n    {expected!r}\n"
        )
    sys.exit(1)

print(
    f"tier2-2.sh complete: rewrote {total_rewritten_imports} imports across "
    f"{total_go_files_modified} .go files (runtime-alias files: {len(runtime_alias_files)})."
)
PY

# The parity manifest's deps_hash (testdata/parity/*.json) is refreshed by hand, after
# the goldens pass: scripts/refresh_parity_hashes.sh. Not run here.
echo "next: scripts/refresh_parity_hashes.sh (goldens-gated), then the layout ratchets"
