# goinfer: instructions for every coding agent

`CLAUDE.md` in this directory is the project's instruction file, and it binds every agent that works in this tree, whatever
model or tool it runs under (Claude Code, Gemini in Antigravity, any other). Read it in full before changing anything. This
page adds no rule. It lists the ones an agent that did not load `CLAUDE.md` has skipped before.

- **Product code reaches `main` through a pull request** that you arm at once (`CLAUDE.md`, "Working in the tree"). Only a
  docs-only change (`scripts/ci_docs_only.sh`) or a release goes straight to `main`.
- **Before a push:** `gofmt -l .` in every module you touched prints nothing; `go run ./cmd/lint` and
  `python3 scripts/touched.py` each end in a VERDICT line that says PASS; never `--no-verify`.
- **Stage explicit paths.** Never `git add -A` or `git add .`. No rebase and no force-push: merge `origin/main`.
- **A change that lands structure or closes work changes its task doc's ledger row in the same commit** (`CLAUDE.md`, "Task
  docs: the ledger"). If no row owns the work, add one.
- **A golden is never regenerated to turn a test green.** A moved decision census is a red run first, and the commit says
  which groups moved and why before `-census.update` (`internal/runtime/mem/CLAUDE.md`). A ratchet or pin that may only
  shrink is not raised without the owner's word.
- **CI runs no CUDA or Metal device test.** A change to a GPU backend is not verified by a green CI run or by goldens run on
  another architecture: say which device tests ran, on which box, or say that none did.
- **A number enters a doc only with its provenance,** and a comparison with a peer goes through `scripts/bench_peer.py`
  (`CLAUDE.md`, "Models and benchmarks"). Describe a dataset from the file, not from memory.
- **A gate, a default, a guard threshold and a tag are the owner's.** Do not promote a parity gate, change a default or add a
  family to a delivered-precision list without the gate that list names.
