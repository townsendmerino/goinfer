#!/usr/bin/env bash
# tier2-1.sh — PR 1 of repo layout Tier 2: move GPU backend modules under backend/
#
# Moves:
#   cuda/  -> backend/cuda/    (module github.com/townsendmerino/goinfer/backend/cuda)
#   metal/ -> backend/metal/   (module github.com/townsendmerino/goinfer/backend/metal)
#   gpu/   -> backend/webgpu/  (module github.com/townsendmerino/goinfer/backend/webgpu, package webgpu)
#
# Usage:
#   bash scripts/layout/tier2-1.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -e backend ]]; then
	echo "error: backend/ already exists; refusing to run" >&2
	exit 1
fi

for dir in cuda metal gpu; do
	if [[ ! -d "$dir" ]]; then
		echo "error: expected directory $dir/ at repository root" >&2
		exit 1
	fi
done

# ---------------------------------------------------------------------------
# Step 1: git mv cuda -> backend/cuda, metal -> backend/metal, gpu -> backend/webgpu
# ---------------------------------------------------------------------------
mkdir -p backend
git mv cuda backend/cuda
git mv metal backend/metal
git mv gpu backend/webgpu

# ---------------------------------------------------------------------------
# Steps 2-9: mechanical rewrites via Python
# ---------------------------------------------------------------------------
python3 - <<'PY'
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path.cwd()


MISSES: list[tuple[str, str, str]] = []


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _nearest_line_for_miss(txt: str, old: str) -> str:
    lines = txt.splitlines()
    if not lines:
        return "<empty file>"
    raw_tokens = re.findall(r"[A-Za-z0-9_./:-]{3,}", old)
    seen: set[str] = set()
    tokens: list[str] = []
    for t in sorted(raw_tokens, key=len, reverse=True):
        if t not in seen:
            seen.add(t)
            tokens.append(t)
    if not tokens:
        return f"L1: {lines[0].strip()}"
    best_idx = -1
    best_score = 0
    for idx, line in enumerate(lines):
        score = sum(len(t) for t in tokens if t in line)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx >= 0:
        return f"L{best_idx + 1}: {lines[best_idx].strip()}"
    return "<no matching token in file>"


def replace_exact(path: Path, old: str, new: str, *, allow_already_applied: bool = False) -> bool:
    rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    if not path.exists():
        MISSES.append((rel, old, "<file does not exist>"))
        return False
    txt = read_text(path)
    if old in txt:
        write_text(path, txt.replace(old, new))
        return True
    if allow_already_applied and new in txt:
        return False
    MISSES.append((rel, old, _nearest_line_for_miss(txt, old)))
    return False


# ===========================================================================
# Step 2: go.mod files
# ===========================================================================
replace_exact(
    ROOT / "backend/cuda/go.mod",
    "module github.com/townsendmerino/goinfer/cuda\n",
    "module github.com/townsendmerino/goinfer/backend/cuda\n",
)
replace_exact(
    ROOT / "backend/metal/go.mod",
    "module github.com/townsendmerino/goinfer/metal\n",
    "module github.com/townsendmerino/goinfer/backend/metal\n",
)
replace_exact(
    ROOT / "backend/webgpu/go.mod",
    "module github.com/townsendmerino/goinfer/gpu\n",
    "module github.com/townsendmerino/goinfer/backend/webgpu\n",
)

GOMOD_PATHS = [
    ROOT / "go.mod",
    ROOT / "backend/cuda/go.mod",
    ROOT / "backend/metal/go.mod",
    ROOT / "backend/webgpu/go.mod",
    ROOT / "demo/agent/go.mod",
    ROOT / "apps/sdktest/go.mod",
    ROOT / "apps/site/go.mod",
]

GOMOD_MOD_RE = re.compile(
    r'(github\.com/townsendmerino/goinfer/)(cuda|metal|gpu)(\b)'
)
GOMOD_MAP = {"cuda": "backend/cuda", "metal": "backend/metal", "gpu": "backend/webgpu"}

for gm in GOMOD_PATHS:
    txt = read_text(gm)
    lines = txt.splitlines(keepends=True)
    new_lines = []
    for ln in lines:
        if ln.startswith("module "):
            new_lines.append(ln)
            continue
        new_lines.append(
            GOMOD_MOD_RE.sub(lambda m: m.group(1) + GOMOD_MAP[m.group(2)] + m.group(3), ln)
        )
    new_txt = "".join(new_lines)
    if new_txt != txt:
        write_text(gm, new_txt)


# ===========================================================================
# Step 3 & Step 4: Go import paths + package gpu -> webgpu + gpu. -> webgpu.
# ===========================================================================
IMPORT_BLOCK_RE = re.compile(r'(?m)^import\s+(?:\([^)]*\)|[^\n]+)')
IMPORT_PATH_RE = re.compile(
    r'"github\.com/townsendmerino/goinfer/(cuda|metal|gpu)((?:/[^"]*)?)"'
)
IMPORT_SPEC_RE = re.compile(
    r'(?m)^\s*(?:([A-Za-z_][A-Za-z0-9_]*|\.)\s+)?"github\.com/townsendmerino/goinfer/backend/webgpu"\s*(?://.*)?$'
)
PKG_GPU_RE = re.compile(r'(?m)^package (gpu|gpu_test)(\s|$)')

# Tokenizer for Go source to rewrite `gpu.` -> `webgpu.` only in code tokens
# (never inside line comments, block comments, interpreted strings, raw strings, or rune literals).
GO_TOKEN_RE = re.compile(
    r'//[^\n]*'
    r'|/\*[\s\S]*?\*/'
    r'|`[^`]*`'
    r'|"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'"
    r'|\bgpu\s*\.\s*[A-Za-z_][A-Za-z0-9_]*'
)


def rewrite_gpu_qualifiers(src: str) -> str:
    def _sub(m: re.Match) -> str:
        tok = m.group(0)
        if tok.startswith(("//", "/*", "`", '"', "'")):
            return tok
        return re.sub(r'^\bgpu\b', 'webgpu', tok)

    return GO_TOKEN_RE.sub(_sub, src)


all_go_files: list[Path] = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d != ".git"]
    for fn in sorted(filenames):
        if fn.endswith(".go"):
            all_go_files.append(Path(dirpath) / fn)

webgpu_root = ROOT / "backend/webgpu"

for go_file in all_go_files:
    orig = read_text(go_file)
    txt = orig

    # Step 3: rewrite quoted import strings inside import declarations only.
    def _rewrite_import_block(m: re.Match) -> str:
        blk = m.group(0)
        return IMPORT_PATH_RE.sub(
            lambda im: f'"github.com/townsendmerino/goinfer/{GOMOD_MAP[im.group(1)]}{im.group(2)}"',
            blk,
        )

    txt = IMPORT_BLOCK_RE.sub(_rewrite_import_block, txt)

    # Step 4a: in backend/webgpu/**/*.go, change `package gpu` -> `package webgpu`
    # and `package gpu_test` -> `package webgpu_test`.
    if go_file.is_relative_to(webgpu_root):
        txt = PKG_GPU_RE.sub(
            lambda pm: f"package {'webgpu' if pm.group(1) == 'gpu' else 'webgpu_test'}{pm.group(2)}",
            txt,
            count=1,
        )

    # Step 4b: in files that import "github.com/townsendmerino/goinfer/backend/webgpu"
    # under no alias or alias `gpu`, rewrite `gpu.` -> `webgpu.`.
    imports_goinfer_webgpu_as_gpu = False
    for im_block in IMPORT_BLOCK_RE.finditer(txt):
        for spec in IMPORT_SPEC_RE.finditer(im_block.group(0)):
            alias = spec.group(1)
            if alias is None or alias == "gpu":
                imports_goinfer_webgpu_as_gpu = True

    if imports_goinfer_webgpu_as_gpu:
        txt = rewrite_gpu_qualifiers(txt)

    if txt != orig:
        write_text(go_file, txt)


# ===========================================================================
# Step 5: Relative `../` paths in moved backend trees
# ===========================================================================
# 5a. backend/cuda/build_ptx.sh relative path to repo-root .cuda_tools
replace_exact(
    ROOT / "backend/cuda/build_ptx.sh",
    '"../.cuda_tools/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"',
    '"../../.cuda_tools/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"',
)
replace_exact(
    ROOT / "backend/cuda/build_ptx.sh",
    '"../.cuda_tools/nvidia/cuda_runtime/include/cuda_fp16.h"',
    '"../../.cuda_tools/nvidia/cuda_runtime/include/cuda_fp16.h"',
)

# 5b. Relative string literals in backend/{cuda,metal,webgpu}/**/*.go that reach
# up to the repository root (depth d inside backend/, so k == d leading `../` segments).
REPO_ROOT_TARGETS = (
    "testdata",
    "decoder",
    "internal",
    "docs",
    "scripts",
    "cmd",
    "apps",
    "demo",
    "README.md",
    "CLAUDE.md",
    "RELEASING.md",
    "go.mod",
    "go.work.example",
)
REPO_TARGETS_ALT = "|".join(re.escape(t) for t in REPO_ROOT_TARGETS)

backend_root = ROOT / "backend"
for go_file in all_go_files:
    if not go_file.is_relative_to(backend_root):
        continue
    rel_to_backend = go_file.relative_to(backend_root)
    # depth d = number of directory components under backend/
    # e.g., backend/cuda/foo_test.go -> parts = ('cuda', 'foo_test.go') -> d = 1
    # e.g., backend/cuda/s4siglip/siglip_test.go -> d = 2
    # e.g., backend/webgpu/internal/kernels/vision_test.go -> d = 3
    d = len(rel_to_backend.parts) - 1
    dotdots = "../" * d

    orig = read_text(go_file)
    txt = orig

    # Rewrite string literals starting with exactly d "../" followed by a repo-root target
    # (and NOT already d+1 "../").
    pat = re.compile(
        r'"(' + re.escape(dotdots) + r')(?:' + REPO_TARGETS_ALT + r')(?:/[^"]*)?"'
    )
    txt = pat.sub(lambda m: '"../' + m.group(0)[1:], txt)

    # Also handle dynamic prefix concatenations at d=1 (`"../" + fixture`, `"../"+fixture`, `"../" + it.Image`)
    # in top-level backend/{cuda,metal,webgpu}/*.go (NOT in subpackages like backend/cuda/kernels where
    # `"../" + fn` refers to parent directory `backend/cuda/`).
    if d == 1:
        txt = re.sub(r'"\.\./"\s*\+\s*(fixture|it\.Image)\b', lambda m: '"../../" + ' + m.group(1), txt)

    if txt != orig:
        write_text(go_file, txt)

# Also update backend/webgpu/internal/kernels/bufaccount_test.go package list passed to `go list`
replace_exact(
    ROOT / "backend/webgpu/internal/kernels/bufaccount_test.go",
    'pkgs := []string{"github.com/townsendmerino/goinfer/gpu/internal/kernels", "github.com/townsendmerino/goinfer/gpu"}',
    'pkgs := []string{"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels", "github.com/townsendmerino/goinfer/backend/webgpu"}',
)


# ===========================================================================
# Step 6: CI workflows, scripts/, demo/chat/build-embed.sh, cmd/lint, cmd/gate,
#         internal/layout/imports.json
# ===========================================================================

# 6a. .github/workflows/ci.yml
ci_yml = ROOT / ".github/workflows/ci.yml"
ci_replacements = [
    (
        "        run: cd cuda && go mod download\n",
        "        run: cd backend/cuda && go mod download\n",
    ),
    (
        "          leak='webgpu|eitamring/gocudrv|ebitengine/purego|townsendmerino/aikit/gpu|townsendmerino/goinfer/(gpu|cuda|metal)'\n",
        "          leak='webgpu|eitamring/gocudrv|ebitengine/purego|townsendmerino/aikit/gpu|townsendmerino/goinfer/(backend/(webgpu|cuda|metal)|gpu|cuda|metal)'\n",
    ),
    (
        "          cache-dependency-path: gpu/go.sum\n",
        "          cache-dependency-path: backend/webgpu/go.sum\n",
    ),
    (
        "        run: go work init . ./gpu ./demo/agent\n",
        "        run: go work init . ./backend/webgpu ./demo/agent\n",
    ),
    (
        "        run: go work init . ./gpu\n",
        "        run: go work init . ./backend/webgpu\n",
    ),
    (
        "        run: go build -tags 'gpu goinfer_testhooks' ./gpu/...\n",
        "        run: go build -tags 'gpu goinfer_testhooks' ./backend/webgpu/...\n",
    ),
    (
        "        run: go build -tags 'gpu goinfer_testhooks' ./gpu/cmd/serve ./gpu/cmd/chat\n",
        "        run: go build -tags 'gpu goinfer_testhooks' ./backend/webgpu/cmd/serve ./backend/webgpu/cmd/chat\n",
    ),
    (
        "        run: go vet -tags 'gpu goinfer_testhooks' ./gpu/...\n",
        "        run: go vet -tags 'gpu goinfer_testhooks' ./backend/webgpu/...\n",
    ),
    (
        "        run: go vet -tags gpu ./gpu/...\n",
        "        run: go vet -tags gpu ./backend/webgpu/...\n",
    ),
    (
        "        run: staticcheck -tags 'gpu goinfer_testhooks' ./gpu/...\n",
        "        run: staticcheck -tags 'gpu goinfer_testhooks' ./backend/webgpu/...\n",
    ),
    (
        "        run: go test -short -tags 'gpu goinfer_testhooks' ./gpu/...\n",
        "        run: go test -short -tags 'gpu goinfer_testhooks' ./backend/webgpu/...\n",
    ),
    (
        "          cache-dependency-path: cuda/go.sum\n",
        "          cache-dependency-path: backend/cuda/go.sum\n",
    ),
    (
        "        run: go work init . ./cuda\n",
        "        run: go work init . ./backend/cuda\n",
    ),
    (
        "        run: CGO_ENABLED=0 go build -tags 'cuda goinfer_testhooks' ./cuda/...\n",
        "        run: CGO_ENABLED=0 go build -tags 'cuda goinfer_testhooks' ./backend/cuda/...\n",
    ),
    (
        "        run: CGO_ENABLED=0 go build -tags 'cuda goinfer_testhooks' ./cuda/cmd/serve ./cuda/cmd/chat\n",
        "        run: CGO_ENABLED=0 go build -tags 'cuda goinfer_testhooks' ./backend/cuda/cmd/serve ./backend/cuda/cmd/chat\n",
    ),
    (
        "        run: go vet -tags 'cuda goinfer_testhooks' ./cuda/...\n",
        "        run: go vet -tags 'cuda goinfer_testhooks' ./backend/cuda/...\n",
    ),
    (
        "        run: go vet ./cuda/...\n",
        "        run: go vet ./backend/cuda/...\n",
    ),
    (
        "        run: go vet -tags cuda ./cuda/...\n",
        "        run: go vet -tags cuda ./backend/cuda/...\n",
    ),
    (
        "        run: staticcheck -tags 'cuda goinfer_testhooks' ./cuda/...\n",
        "        run: staticcheck -tags 'cuda goinfer_testhooks' ./backend/cuda/...\n",
    ),
    (
        "        run: CGO_ENABLED=0 go test -short -tags 'cuda goinfer_testhooks' ./cuda/...\n",
        "        run: CGO_ENABLED=0 go test -short -tags 'cuda goinfer_testhooks' ./backend/cuda/...\n",
    ),
    (
        "          cache-dependency-path: metal/go.sum\n",
        "          cache-dependency-path: backend/metal/go.sum\n",
    ),
    (
        "        run: go work init . ./metal\n",
        "        run: go work init . ./backend/metal\n",
    ),
    (
        "        run: go build ./metal/...\n",
        "        run: go build ./backend/metal/...\n",
    ),
    (
        "        run: go build ./metal/cmd/serve ./metal/cmd/chat\n",
        "        run: go build ./backend/metal/cmd/serve ./backend/metal/cmd/chat\n",
    ),
    (
        "        run: go vet ./metal/...\n",
        "        run: go vet ./backend/metal/...\n",
    ),
    (
        "        run: go vet -tags goinfer_testhooks ./metal/...\n",
        "        run: go vet -tags goinfer_testhooks ./backend/metal/...\n",
    ),
    (
        "          matched=$(go test -list 'TestParity_NaNCosineFailsTheGate' ./metal/... | grep -E '^[A-Za-z_][A-Za-z0-9_]*$' || true)\n",
        "          matched=$(go test -list 'TestParity_NaNCosineFailsTheGate' ./backend/metal/... | grep -E '^[A-Za-z_][A-Za-z0-9_]*$' || true)\n",
    ),
    (
        "          matched=$(go test -list 'TestBuildDemand_matchesDecoderDefault' ./metal/ | grep -E '^[A-Za-z_][A-Za-z0-9_]*$' || true)\n",
        "          matched=$(go test -list 'TestBuildDemand_matchesDecoderDefault' ./backend/metal/ | grep -E '^[A-Za-z_][A-Za-z0-9_]*$' || true)\n",
    ),
    (
        "          matched=$(go test -list 'TestLedger_everyMetalAllocationNamesALine' ./metal/ | grep -E '^[A-Za-z_][A-Za-z0-9_]*$' || true)\n",
        "          matched=$(go test -list 'TestLedger_everyMetalAllocationNamesALine' ./backend/metal/ | grep -E '^[A-Za-z_][A-Za-z0-9_]*$' || true)\n",
    ),
    (
        "        run: go test -run 'TestParity_NaNCosineFailsTheGate|TestBuildDemand_matchesDecoderDefault|TestLedger_everyMetalAllocationNamesALine' ./metal/...\n",
        "        run: go test -run 'TestParity_NaNCosineFailsTheGate|TestBuildDemand_matchesDecoderDefault|TestLedger_everyMetalAllocationNamesALine' ./backend/metal/...\n",
    ),
]
for old, new in ci_replacements:
    replace_exact(ci_yml, old, new)

# 6b. .github/workflows/govulncheck.yml
govuln_yml = ROOT / ".github/workflows/govulncheck.yml"
for old, new in [
    ("          cache-dependency-path: gpu/go.sum\n", "          cache-dependency-path: backend/webgpu/go.sum\n"),
    ("        run: go work init . ./gpu ./demo/agent\n", "        run: go work init . ./backend/webgpu ./demo/agent\n"),
    ("        working-directory: gpu\n", "        working-directory: backend/webgpu\n"),
    ("          cache-dependency-path: cuda/go.sum\n", "          cache-dependency-path: backend/cuda/go.sum\n"),
    ("        run: go work init . ./cuda\n", "        run: go work init . ./backend/cuda\n"),
    ("        working-directory: cuda\n", "        working-directory: backend/cuda\n"),
    ("          cache-dependency-path: metal/go.sum\n", "          cache-dependency-path: backend/metal/go.sum\n"),
    ("        run: go work init . ./metal\n", "        run: go work init . ./backend/metal\n"),
    ("        working-directory: metal\n", "        working-directory: backend/metal\n"),
]:
    replace_exact(govuln_yml, old, new)

# 6c. .github/workflows/release-assets.yml
rel_yml = ROOT / ".github/workflows/release-assets.yml"
replace_exact(
    rel_yml,
    '          ( cd "$root" && GOWORK="$gowork" go work init . ./cuda ./metal )\n',
    '          ( cd "$root" && GOWORK="$gowork" go work init . ./backend/cuda ./backend/metal )\n',
)
replace_exact(
    rel_yml,
    "          build metal metal darwin arm64  ./cmd/chat\n"
    "          build metal metal darwin amd64  ./cmd/chat\n"
    "          build cuda  cuda  linux  amd64  ./cmd/chat\n"
    "          build cuda  cuda  linux  arm64  ./cmd/chat\n",
    "          build backend/metal metal darwin arm64  ./cmd/chat\n"
    "          build backend/metal metal darwin amd64  ./cmd/chat\n"
    "          build backend/cuda  cuda  linux  amd64  ./cmd/chat\n"
    "          build backend/cuda  cuda  linux  arm64  ./cmd/chat\n",
)
replace_exact(
    rel_yml,
    '          ( cd "$PWD" && GOWORK="$gowork" go work init . ./cuda ./metal )\n',
    '          ( cd "$PWD" && GOWORK="$gowork" go work init . ./backend/cuda ./backend/metal )\n',
)
replace_exact(
    rel_yml,
    "          build metal metal darwin arm64\n"
    "          build metal metal darwin amd64\n"
    "          build cuda  cuda  linux  amd64\n"
    "          build cuda  cuda  linux  arm64\n",
    "          build backend/metal metal darwin arm64\n"
    "          build backend/metal metal darwin amd64\n"
    "          build backend/cuda  cuda  linux  amd64\n"
    "          build backend/cuda  cuda  linux  arm64\n",
)
replace_exact(
    rel_yml,
    " | grep -qE '^\\s*mod\\s+github.com/townsendmerino/goinfer/metal\\b' \\\n",
    " | grep -qE '^\\s*mod\\s+github.com/townsendmerino/goinfer/backend/metal\\b' \\\n",
)
replace_exact(
    rel_yml,
    " | grep -qE '^\\s*mod\\s+github.com/townsendmerino/goinfer/cuda\\b' \\\n",
    " | grep -qE '^\\s*mod\\s+github.com/townsendmerino/goinfer/backend/cuda\\b' \\\n",
)

# 6e. .github/workflows/standalone-build.yml
sb_yml = ROOT / ".github/workflows/standalone-build.yml"
replace_exact(
    sb_yml,
    """      - 'gpu/v*'
      - 'cuda/v*'
      - 'metal/v*'
      - 'demo/agent/v*'""",
    """      - 'backend/webgpu/v*'
      - 'backend/cuda/v*'
      - 'backend/metal/v*'
      - 'gpu/v*'
      - 'cuda/v*'
      - 'metal/v*'
      - 'demo/agent/v*'""",
)
replace_exact(
    sb_yml,
    """          - module: gpu
            tags: gpu
          - module: cuda
            tags: cuda
          - module: metal
            tags: metal""",
    """          - module: backend/webgpu
            tags: gpu
          - module: backend/cuda
            tags: cuda
          - module: backend/metal
            tags: metal""",
)
replace_exact(
    sb_yml,
    "          GOOS: ${{ matrix.module == 'metal' && 'darwin' || 'linux' }}",
    "          GOOS: ${{ matrix.module == 'backend/metal' && 'darwin' || 'linux' }}",
)

# 6f. demo/chat/build-embed.sh
replace_exact(
    ROOT / "demo/chat/build-embed.sh",
    '    darwin) moduledir="$ROOT/metal"; entry="./cmd/chat"; buildtags="$TAGS,metal" ;;\n'
    '    linux)  moduledir="$ROOT/cuda";  entry="./cmd/chat"; buildtags="$TAGS,cuda" ;;',
    '    darwin) moduledir="$ROOT/backend/metal"; entry="./cmd/chat"; buildtags="$TAGS,metal" ;;\n'
    '    linux)  moduledir="$ROOT/backend/cuda";  entry="./cmd/chat"; buildtags="$TAGS,cuda" ;;',
)

# 6g. scripts/touched.py, scripts/touched_scans.py, scripts/test_touched.py
replace_exact(
    ROOT / "scripts/touched.py",
    'SLOW = {"decoder", "internal/sampling", "cuda", "internal/serveapp"}',
    'SLOW = {"decoder", "internal/sampling", "backend/cuda", "internal/serveapp"}',
)
replace_exact(
    ROOT / "scripts/touched.py",
    'TAGS = {"cuda": "cuda goinfer_testhooks", "gpu": "gpu goinfer_testhooks", "": "goinfer_testhooks"}',
    'TAGS = {"backend/cuda": "cuda goinfer_testhooks", "backend/webgpu": "gpu goinfer_testhooks", "": "goinfer_testhooks"}',
)
replace_exact(
    ROOT / "scripts/touched.py",
    '        if mod == "metal" and sys.platform != "darwin":\n'
    '            print("  skip metal: builds on darwin only")\n'
    '            skips.append("metal")\n'
    '            continue\n'
    '        env = dict(os.environ, CGO_ENABLED="0") if mod == "cuda" else dict(os.environ)',
    '        if mod == "backend/metal" and sys.platform != "darwin":\n'
    '            print("  skip metal: builds on darwin only")\n'
    '            skips.append("metal")\n'
    '            continue\n'
    '        env = dict(os.environ, CGO_ENABLED="0") if mod == "backend/cuda" else dict(os.environ)',
)

# scripts/touched_scans.py
ts_path = ROOT / "scripts/touched_scans.py"
ts_txt = read_text(ts_path)
ts_txt = re.sub(r'"(cuda|metal)(?=/|")', lambda m: f'"backend/{m.group(1)}', ts_txt)
ts_txt = re.sub(r'"gpu(?=/|")', '"backend/webgpu', ts_txt)
write_text(ts_path, ts_txt)

# scripts/test_touched.py
replace_exact(
    ROOT / "scripts/test_touched.py",
    '        got = selected(["cuda/moe.cu"])\n'
    '        self.assertIn(("cuda/kernels", "TestKernelFMALint"), got)',
    '        got = selected(["backend/cuda/moe.cu"])\n'
    '        self.assertIn(("backend/cuda/kernels", "TestKernelFMALint"), got)',
)

# 6h. Other scripts in scripts/
replace_exact(
    ROOT / "scripts/queue_citation_lint.py",
    '                           (ROOT / "cuda" / "go.mod", "github.com/townsendmerino/aikit/gpu")):',
    '                           (ROOT / "backend" / "cuda" / "go.mod", "github.com/townsendmerino/aikit/gpu")):',
)
replace_exact(
    ROOT / "scripts/night.py",
    '    if backend == "cuda":\n'
    '        return ["go", "run", "-tags", "cuda", "./cuda/cmd/chat"]\n'
    '    if backend == "metal":\n'
    '        return ["go", "run", "./metal/cmd/chat"]',
    '    if backend == "cuda":\n'
    '        return ["go", "run", "-tags", "cuda", "./backend/cuda/cmd/chat"]\n'
    '    if backend == "metal":\n'
    '        return ["go", "run", "./backend/metal/cmd/chat"]',
)
replace_exact(
    ROOT / "scripts/test_cuda_device_suite.py",
    '"github.com/townsendmerino/goinfer/cuda"',
    '"github.com/townsendmerino/goinfer/backend/cuda"',
)

# 6i. cmd/lint/lint.go and cmd/lint/lint_test.go
replace_exact(
    ROOT / "cmd/lint/lint.go",
    '\t{name: "gpu", args: []string{"-tags", "gpu goinfer_testhooks", "./gpu/..."}, workspace: []string{"gpu", "demo/agent"}, cgo: true},\n'
    '\t{name: "demo/agent", args: []string{"./demo/agent/..."}, workspace: []string{"gpu", "demo/agent"}},\n'
    '\t{name: "cuda", args: []string{"-tags", "cuda goinfer_testhooks", "./cuda/..."}, workspace: []string{"cuda"}},',
    '\t{name: "gpu", args: []string{"-tags", "gpu goinfer_testhooks", "./backend/webgpu/..."}, workspace: []string{"backend/webgpu", "demo/agent"}, cgo: true},\n'
    '\t{name: "demo/agent", args: []string{"./demo/agent/..."}, workspace: []string{"backend/webgpu", "demo/agent"}},\n'
    '\t{name: "cuda", args: []string{"-tags", "cuda goinfer_testhooks", "./backend/cuda/..."}, workspace: []string{"backend/cuda"}},',
)
replace_exact(
    ROOT / "cmd/lint/lint_test.go",
    '  *./cuda/...*) [ "$GOOS" = linux ] && { echo "cuda/x.go:1:1: bad (SA4006)"; exit 1; } ;;',
    '  *./backend/cuda/...*) [ "$GOOS" = linux ] && { echo "cuda/x.go:1:1: bad (SA4006)"; exit 1; } ;;',
)
replace_exact(
    ROOT / "cmd/lint/lint_test.go",
    '  *./cuda/...*) echo "cuda/x.go:1:1: bad (SA4006)"; exit 1 ;;',
    '  *./backend/cuda/...*) echo "cuda/x.go:1:1: bad (SA4006)"; exit 1 ;;',
)

# 6j. cmd/gate/ live runners and non-golden-pinned tests
# cmd/gate/gpu.go
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\tout, err := exec.Command("go", "test", "-tags", "gpu", "./gpu/internal/kernels/", "-run", "TestAdapterProbe", "-v").CombinedOutput()',
    '\tout, err := exec.Command("go", "test", "-tags", "gpu", "./backend/webgpu/internal/kernels/", "-run", "TestAdapterProbe", "-v").CombinedOutput()',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\tfiles, _ := filepath.Glob(filepath.Join("cuda", "*_test.go"))',
    '\tfiles, _ := filepath.Glob(filepath.Join("backend", "cuda", "*_test.go"))',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\treturn cell{Name: "cuda-drain", Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},',
    '\treturn cell{Name: "cuda-drain", Pkgs: []string{"./backend/cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\treturn cell{Name: "cuda-iso-" + name, Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},',
    '\treturn cell{Name: "cuda-iso-" + name, Pkgs: []string{"./backend/cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\tlisted, err := listCUDATests("./cuda/")',
    '\tlisted, err := listCUDATests("./backend/cuda/")',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    'const cudaKernelsPkg = "./cuda/kernels/"',
    'const cudaKernelsPkg = "./backend/cuda/kernels/"',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\trefs, err := testReachableLiterals("cuda", []string{"cuda", "goinfer_testhooks"}, filepath.Join("testdata", "assets.json"))',
    '\trefs, err := testReachableLiterals("backend/cuda", []string{"cuda", "goinfer_testhooks"}, filepath.Join("testdata", "assets.json"))',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\tif pkg == "" {\n'
    '\t\tpkg = "./cuda/"\n'
    '\t}',
    '\tif pkg == "" {\n'
    '\t\tpkg = "./backend/cuda/"\n'
    '\t}',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\t\tcmd.Dir = "cuda"',
    '\t\tcmd.Dir = filepath.Join("backend", "cuda")',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\tst, err := os.Stat(filepath.Join("cuda", "build_ptx.sh"))',
    '\tst, err := os.Stat(filepath.Join("backend", "cuda", "build_ptx.sh"))',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\tptxFiles, _ := filepath.Glob(filepath.Join("cuda", "testdata", "*.ptx"))',
    '\tptxFiles, _ := filepath.Glob(filepath.Join("backend", "cuda", "testdata", "*.ptx"))',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\t\tif _, err := os.Stat(filepath.Join("cuda", base+".cu")); err != nil {',
    '\t\tif _, err := os.Stat(filepath.Join("backend", "cuda", base+".cu")); err != nil {',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\t\t\tif _, err := os.Stat(filepath.Join("cuda", kernel+".cu")); kernel == base || err != nil {',
    '\t\t\tif _, err := os.Stat(filepath.Join("backend", "cuda", kernel+".cu")); kernel == base || err != nil {',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\tbuild.Dir = "metal"',
    '\tbuild.Dir = filepath.Join("backend", "metal")',
)
replace_exact(
    ROOT / "cmd/gate/gpu.go",
    '\tbuild.Dir = "cuda"',
    '\tbuild.Dir = filepath.Join("backend", "cuda")',
)

# cmd/gate/gpu_test.go
replace_exact(
    ROOT / "cmd/gate/gpu_test.go",
    '\tst, err := os.Stat(filepath.Join("..", "..", "cuda", "build_ptx.sh"))',
    '\tst, err := os.Stat(filepath.Join("..", "..", "backend", "cuda", "build_ptx.sh"))',
)
replace_exact(
    ROOT / "cmd/gate/gpu_test.go",
    '\tif _, err := os.Stat(filepath.Join("cuda", "build_ptx.sh")); err != nil {',
    '\tif _, err := os.Stat(filepath.Join("backend", "cuda", "build_ptx.sh")); err != nil {',
)
replace_exact(
    ROOT / "cmd/gate/gpu_test.go",
    '\tall, _ := filepath.Glob(filepath.Join("cuda", "testdata", "*.ptx"))',
    '\tall, _ := filepath.Glob(filepath.Join("backend", "cuda", "testdata", "*.ptx"))',
)

# cmd/gate/grade_gpu_cuda.go
replace_exact(
    ROOT / "cmd/gate/grade_gpu_cuda.go",
    '\tcudaSuiteCell = cell{Name: "cuda-suite", Pkgs: []string{"./cuda/", "./cuda/kernels/"}, Tags: []string{"cuda"},',
    '\tcudaSuiteCell = cell{Name: "cuda-suite", Pkgs: []string{"./backend/cuda/", "./backend/cuda/kernels/"}, Tags: []string{"cuda"},',
)
replace_exact(
    ROOT / "cmd/gate/grade_gpu_cuda.go",
    '\tcudaParityCell = cell{Name: "cuda-parity", Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},',
    '\tcudaParityCell = cell{Name: "cuda-parity", Pkgs: []string{"./backend/cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},',
)
replace_exact(
    ROOT / "cmd/gate/grade_gpu_cuda.go",
    '\tcudaGraphsCell = cell{Name: "cuda-graphs", Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},',
    '\tcudaGraphsCell = cell{Name: "cuda-graphs", Pkgs: []string{"./backend/cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},',
)

# cmd/gate/grade_gpu_metal.go
replace_exact(
    ROOT / "cmd/gate/grade_gpu_metal.go",
    'Pkgs: []string{"./metal/"}',
    'Pkgs: []string{"./backend/metal/"}',
)

# cmd/gate/grade_gpu_groups.go
replace_exact(
    ROOT / "cmd/gate/grade_gpu_groups.go",
    '\twebgpuSuiteCell = cell{Name: "webgpu-suite", Pkgs: []string{"./gpu/", "./gpu/internal/kernels/"}, Tags: []string{"gpu"},',
    '\twebgpuSuiteCell = cell{Name: "webgpu-suite", Pkgs: []string{"./backend/webgpu/", "./backend/webgpu/internal/kernels/"}, Tags: []string{"gpu"},',
)
replace_exact(
    ROOT / "cmd/gate/grade_gpu_groups.go",
    '\twebgpuParityCell = cell{Name: "webgpu-parity", Pkgs: []string{"./gpu/", "./gpu/internal/kernels/"},',
    '\twebgpuParityCell = cell{Name: "webgpu-parity", Pkgs: []string{"./backend/webgpu/", "./backend/webgpu/internal/kernels/"},',
)

# cmd/gate/parity.go
replace_exact(
    ROOT / "cmd/gate/parity.go",
    'var metalDirs = []string{"metal"}',
    'var metalDirs = []string{"backend/metal"}',
)
replace_exact(
    ROOT / "cmd/gate/parity.go",
    'var webgpuDirs = []string{"gpu", "gpu/internal/kernels"}',
    'var webgpuDirs = []string{"backend/webgpu", "backend/webgpu/internal/kernels"}',
)

# cmd/gate/quick_select.go
replace_exact(
    ROOT / "cmd/gate/quick_select.go",
    '\t\t\tName: "gpu", Dir: "gpu", Work: []string{".", "gpu"},',
    '\t\t\tName: "gpu", Dir: "backend/webgpu", Work: []string{".", "backend/webgpu"},',
)
replace_exact(
    ROOT / "cmd/gate/quick_select.go",
    '\t\t\tName: "cuda", Dir: "cuda", Work: []string{".", "cuda"},',
    '\t\t\tName: "cuda", Dir: "backend/cuda", Work: []string{".", "backend/cuda"},',
)
replace_exact(
    ROOT / "cmd/gate/quick_select.go",
    '\t\t\tName: "metal", Dir: "metal", Work: []string{".", "metal"},',
    '\t\t\tName: "metal", Dir: "backend/metal", Work: []string{".", "backend/metal"},',
)
replace_exact(
    ROOT / "cmd/gate/quick_select.go",
    '\t\t\tName: "demo/agent", Dir: "demo/agent", Work: []string{".", "gpu", "demo/agent"},',
    '\t\t\tName: "demo/agent", Dir: "demo/agent", Work: []string{".", "backend/webgpu", "demo/agent"},',
)

# cmd/gate/heavy_groups.go and cmd/gate/heavy_groups_test.go
replace_exact(
    ROOT / "cmd/gate/heavy_groups.go",
    '\tlisted, err := listCUDATests("./cuda/")',
    '\tlisted, err := listCUDATests("./backend/cuda/")',
)
replace_exact(
    ROOT / "cmd/gate/heavy_groups_test.go",
    '\trefs, err := testReachableLiterals(filepath.Join(root, "cuda"), []string{"cuda", "goinfer_testhooks"}, filepath.Join(root, "testdata", "assets.json"))',
    '\trefs, err := testReachableLiterals(filepath.Join(root, "backend", "cuda"), []string{"cuda", "goinfer_testhooks"}, filepath.Join(root, "testdata", "assets.json"))',
)

# cmd/gate/composition.go
replace_exact(
    ROOT / "cmd/gate/composition.go",
    'var compSearchDirs = []string{"decoder", "cuda", "metal", "tokenizer", "internal"}',
    'var compSearchDirs = []string{"decoder", "backend/cuda", "backend/metal", "tokenizer", "internal"}',
)

# cmd/gate/identity.go, identity_test.go
replace_exact(
    ROOT / "cmd/gate/identity.go",
    '\tcase "metal":\n'
    '\t\treturn "dumper in metal/ (registers the backend), ephemeral go.work: . ./metal"\n'
    '\tcase "webgpu":\n'
    '\t\treturn "dumper in gpu/ with -tags gpu, ephemeral go.work: . ./gpu"\n'
    '\tcase "cuda":\n'
    '\t\treturn "dumper in cuda/ with -tags cuda, ephemeral go.work: . ./cuda"',
    '\tcase "metal":\n'
    '\t\treturn "dumper in backend/metal/ (registers the backend), ephemeral go.work: . ./backend/metal"\n'
    '\tcase "webgpu":\n'
    '\t\treturn "dumper in backend/webgpu/ with -tags gpu, ephemeral go.work: . ./backend/webgpu"\n'
    '\tcase "cuda":\n'
    '\t\treturn "dumper in backend/cuda/ with -tags cuda, ephemeral go.work: . ./backend/cuda"',
)
replace_exact(
    ROOT / "cmd/gate/identity.go",
    '\tswitch o.Backend {\n'
    '\tcase "metal":\n'
    '\t\tmodDir, register = filepath.Join(s.WT, "metal"), `_ "github.com/townsendmerino/goinfer/metal"`\n'
    '\tcase "webgpu":\n'
    '\t\tmodDir, register, tags = filepath.Join(s.WT, "gpu"), `_ "github.com/townsendmerino/goinfer/gpu"`, "gpu"\n'
    '\tcase "cuda":\n'
    '\t\tmodDir, register, tags = filepath.Join(s.WT, "cuda"), `_ "github.com/townsendmerino/goinfer/cuda"`, "cuda"\n'
    '\t}',
    '\tswitch o.Backend {\n'
    '\tcase "metal":\n'
    '\t\tmodDir, register = filepath.Join(s.WT, "backend", "metal"), `_ "github.com/townsendmerino/goinfer/backend/metal"`\n'
    '\tcase "webgpu":\n'
    '\t\tmodDir, register, tags = filepath.Join(s.WT, "backend", "webgpu"), `_ "github.com/townsendmerino/goinfer/backend/webgpu"`, "gpu"\n'
    '\tcase "cuda":\n'
    '\t\tmodDir, register, tags = filepath.Join(s.WT, "backend", "cuda"), `_ "github.com/townsendmerino/goinfer/backend/cuda"`, "cuda"\n'
    '\t}',
)
replace_exact(
    ROOT / "cmd/gate/identity.go",
    '\tbody := fmt.Sprintf("go %s\\n\\nuse (\\n\\t.\\n\\t./%s\\n)\\n", gov, filepath.Base(modDir))',
    '\tbody := fmt.Sprintf("go %s\\n\\nuse (\\n\\t.\\n\\t./backend/%s\\n)\\n", gov, filepath.Base(modDir))',
)
replace_exact(
    ROOT / "cmd/gate/identity_test.go",
    '\tfor _, reg := range []string{"", `_ "github.com/townsendmerino/goinfer/metal"`} {',
    '\tfor _, reg := range []string{"", `_ "github.com/townsendmerino/goinfer/backend/metal"`} {',
)

# cmd/gate/comments_only.go
replace_exact(
    ROOT / "cmd/gate/comments_only.go",
    '\tcase "cuda":\n'
    '\t\t// tagged only: prefill_deltanet.go has no build line but uses types the tagged files define, so an untagged vet cannot build\n'
    '\t\treturn []string{"cuda goinfer_testhooks"}\n'
    '\tcase "gpu":\n'
    '\t\treturn []string{"", "gpu goinfer_testhooks"}',
    '\tcase "backend/cuda", "cuda":\n'
    '\t\t// tagged only: prefill_deltanet.go has no build line but uses types the tagged files define, so an untagged vet cannot build\n'
    '\t\treturn []string{"cuda goinfer_testhooks"}\n'
    '\tcase "backend/webgpu", "gpu":\n'
    '\t\treturn []string{"", "gpu goinfer_testhooks"}',
)
replace_exact(
    ROOT / "cmd/gate/comments_only.go",
    '\t\tif mod == "metal" {',
    '\t\tif mod == "backend/metal" || mod == "metal" {',
)

# 6k. internal/layout/imports.json, internal/layout/decoder_imports_test.go, internal/fitcmd/defaultctx_test.go
imports_json_path = ROOT / "internal/layout/imports.json"
imp_data = json.loads(read_text(imports_json_path))


def map_pkg_name(p: str) -> str:
    for old_p, new_p in (("cuda", "backend/cuda"), ("metal", "backend/metal"), ("gpu", "backend/webgpu")):
        if p == old_p:
            return new_p
        if p.startswith(old_p + "/"):
            return new_p + p[len(old_p):]
    return p


def map_gomod_path(m: str) -> str:
    return GOMOD_MOD_RE.sub(lambda match: match.group(1) + GOMOD_MAP[match.group(2)] + match.group(3), m)


for rule in imp_data.get("rules", []):
    if "from" in rule:
        rule["from"] = [map_pkg_name(x) for x in rule["from"]]
    if "forbid_goinfer" in rule:
        rule["forbid_goinfer"] = [map_pkg_name(x) for x in rule["forbid_goinfer"]]
    if "allow_goinfer" in rule:
        rule["allow_goinfer"] = {
            map_pkg_name(k): [map_pkg_name(x) for x in v]
            for k, v in rule["allow_goinfer"].items()
        }
    if "gomod_forbid" in rule:
        rule["gomod_forbid"] = [map_gomod_path(x) for x in rule["gomod_forbid"]]

for arrow_key in ("arrows", "test_arrows"):
    if arrow_key in imp_data:
        new_arrows = {}
        for k, v in imp_data[arrow_key].items():
            nk = map_pkg_name(k)
            new_arrows[nk] = sorted(map_pkg_name(x) for x in v)
        imp_data[arrow_key] = {k: new_arrows[k] for k in sorted(new_arrows)}

write_text(imports_json_path, json.dumps(imp_data, indent=2, ensure_ascii=False) + "\n")

replace_exact(
    ROOT / "internal/layout/decoder_imports_test.go",
    'var decoderRatchetExempt = map[string]bool{\n'
    '\t"cuda/sb2_census_test.go":  true,\n'
    '\t"gpu/sb2_census_test.go":   true,\n'
    '\t"metal/sb2_census_test.go": true,\n'
    '}',
    'var decoderRatchetExempt = map[string]bool{\n'
    '\t"backend/cuda/sb2_census_test.go":   true,\n'
    '\t"backend/webgpu/sb2_census_test.go": true,\n'
    '\t"backend/metal/sb2_census_test.go":  true,\n'
    '}',
)
replace_exact(
    ROOT / "internal/layout/decoder_imports_test.go",
    '\t\tmod := strings.SplitN(f.dir, "/", 2)[0]\n'
    '\t\tif _, ok := decoderRatchet[mod]; !ok {',
    '\t\tmod := strings.SplitN(f.dir, "/", 2)[0]\n'
    '\t\tif mod == "backend" {\n'
    '\t\t\tif parts := strings.SplitN(f.dir, "/", 3); len(parts) >= 2 {\n'
    '\t\t\t\tmod = parts[1]\n'
    '\t\t\t\tif mod == "webgpu" {\n'
    '\t\t\t\t\tmod = "gpu"\n'
    '\t\t\t\t}\n'
    '\t\t\t}\n'
    '\t\t}\n'
    '\t\tif _, ok := decoderRatchet[mod]; !ok {',
)
replace_exact(
    ROOT / "internal/fitcmd/defaultctx_test.go",
    '\tsrc, err := os.ReadFile("../../cuda/resident.go")',
    '\tsrc, err := os.ReadFile("../../backend/cuda/resident.go")',
)


# ===========================================================================
# Step 7: go.work.example (idempotent if already edited directly)
# ===========================================================================
replace_exact(
    ROOT / "go.work.example",
    "use (\n\t.\n\t./gpu\n\t./cuda\n\t./metal\n\t./demo/agent\n\t./apps/sdktest\n\t./apps/site\n)",
    "use (\n\t.\n\t./backend/webgpu\n\t./backend/cuda\n\t./backend/metal\n\t./demo/agent\n\t./apps/sdktest\n\t./apps/site\n)",
    allow_already_applied=True,
)


# ===========================================================================
# Step 9: changelog.d/2026-11-03-layout-tier2-backends.md
# ===========================================================================
changelog_frag = ROOT / "changelog.d/2026-11-03-layout-tier2-backends.md"
write_text(
    changelog_frag,
    "- **BREAKING — GPU backend modules moved under `backend/`**: the three GPU backend satellite modules have moved under `backend/` as part of the Tier 2 repository layout reorganization, and the WebGPU package clause is now `package webgpu` instead of `package gpu`:\n"
    "  - `github.com/townsendmerino/goinfer/cuda` (and subpaths) -> `github.com/townsendmerino/goinfer/backend/cuda`\n"
    "  - `github.com/townsendmerino/goinfer/metal` (and subpaths) -> `github.com/townsendmerino/goinfer/backend/metal`\n"
    "  - `github.com/townsendmerino/goinfer/gpu` (and subpaths) -> `github.com/townsendmerino/goinfer/backend/webgpu` (`package gpu` -> `package webgpu`)\n",
)

if MISSES:
    print(f"tier2-1.sh: {len(MISSES)} replacement miss(es):", file=sys.stderr)
    for idx, (rel, old, nearest) in enumerate(MISSES, 1):
        print(f"  [{idx}] {rel}\n      expected: {old!r}\n      nearest:  {nearest}", file=sys.stderr)
    sys.exit(1)

print("tier2-1.sh: all mechanical rewrites completed successfully.")
PY
