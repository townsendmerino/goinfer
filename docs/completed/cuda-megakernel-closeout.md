# Brief — close out `cuda-megakernel-spec.md` and the dead `megakernel.cu` scaffold

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.

> **Status: EXECUTED 2026-09-12.** Every numbered step below landed: the scaffold deleted, the
> spec archived to `docs/completed/cuda-megakernel-spec.md`, `cuda/doc.go` and
> `cuda/kernel_local_memory_test.go`'s census comment rewritten, the two live references fixed,
> and this brief itself moved here alongside its own subject rather than staying in
> `docs/prompts/` — everything below is a record of what was asked, not a task.

> Written by Cowork 2026-09-12 against `a3879e2` on `aikit-v1.41.0-bump`, for vscode-claude to
> execute on the Mac. Docs-and-dead-code cleanup only: no kernel changes, no re-opening K2, no
> change to what `G35` in `QUEUE.md` concluded.

## What is true today (verified, not to be re-derived)

- `docs/cuda-megakernel-spec.md` has carried **Status: CLOSED** since 2026-09-02 but still lives
  among the live docs. Its companion, `task-cuda-cgofree-spike.md`, already moved to
  `docs/completed/`. The spec was never listed in `docs/README.md`'s map.
- The work the spec described landed as `cuda/fused_qkv.cu` (K1 + K3a, behind `fuseQKV` /
  `GOINFER_CUDA_NO_FUSE`); K2 was built, measured ~0%, and reverted. Both facts are in the spec's
  status block and in `docs/completed/task-cuda-cgofree-spike.md`.
- `cuda/megakernel.cu` is the 48-line July scaffold, header still reads "SCAFFOLD — NOT YET
  FUNCTIONAL". `cuda/megakernel.ptx` sits beside it. **Neither is `go:embed`-ed** (43 embed
  directives in `cuda/`, none names megakernel), no Go file loads the PTX, and
  `cuda/build_ptx.sh:70-71` already skips the file unless asked by name. Both are git-tracked.
- `cuda/doc.go` (the package comment) still describes the backend as "the Phase-1 skeleton of the
  megakernel spike", points at the pre-move path of the spike doc and at the spec, and calls
  Layer B "one fused decode-layer megakernel (megakernel.cu)". None of that has been true since
  v0.10; the N-34 note at the bottom already half-admits it.
- References that will break or go stale on the move:
  - `cuda/doc.go:5`
  - `docs/decode-fusion-next.md:104`
  - `docs/QUEUE.md:2839` — prose inside the G35 entry ("the spec … still reads as a live prep
    artifact and does not say the spike it belongs to has closed")
  - `docs/QUEUE.md:1998-2002` — five citation-index rows keyed `docs/cuda-megakernel-spec.md|…`
  - `docs/completed/cuda-decode-headroom-audit.md:11,28` — bare name, already archived; becomes a
    sibling reference after the move. Leave it.
  - `docs/internal/metal-decode-review-package.md:16` — gitignored, this machine only. Fix if it
    costs nothing; not part of the commit.
  - `.repowise/knowledge-graph.json` — untracked. Ignore.
- `cuda/kernel_local_memory_test.go:150-156` carries a hand-audit comment dated 2026-08-13 that
  says "22 .ptx blobs are go:embed-ed … 10 of the difference (…, megakernel.ptx) are referenced
  ONLY from _test.go". The 22 is already stale (43 today) and megakernel.ptx is about to go.
- Other `megakernel` mentions (`gpu/staged_bench_test.go:21`, `cuda/gemv_bw_test.go:22`,
  `cuda/gocudrv_proof_test.go:26,147`, `cuda/decode_projection_test.go:17,138`) describe the
  *thesis* — launch amortization — not the file. **Leave them.**

## Steps

1. **Delete the scaffold.** `git rm cuda/megakernel.cu cuda/megakernel.ptx`. Remove the two-line
   skip clause at `cuda/build_ptx.sh:70-71` so the script has nothing to say about a file that no
   longer exists. Re-read the loop around it to make sure the `$#`-based branch still does what it
   did for every other kernel.

2. **Rewrite `cuda/doc.go`.** Describe the package as it is: an opt-in, cgo-free CUDA backend;
   Layer A = the gocudrv shim (keep that paragraph, it is still accurate); Layer B = the kernel
   set in `cuda/*.cu` compiled to PTX by `go generate` and `go:embed`-ed — name the production
   modules from `ptxModules()` in `cuda/kernels.go`, don't list from memory. One sentence of
   history pointing at `docs/completed/task-cuda-cgofree-spike.md` and
   `docs/completed/cuda-megakernel-spec.md`. Fold the N-34 note into the prose so the comment no
   longer contradicts itself; the "Build with `-tags cuda`" / BuildResident paragraph stays.

3. **Re-run the census audit** the comment at `cuda/kernel_local_memory_test.go:150-156` asks for:
   `grep -n go:embed cuda/*.go`, then which embedded vars non-test files use. Rewrite the
   parenthetical with today's numbers and date. Change the `ptxModules()` denominator **only** if
   the audit says a production PTX is missing from it — and if so, say which in the commit message.

4. **Move the spec.** `git mv docs/cuda-megakernel-spec.md docs/completed/cuda-megakernel-spec.md`.
   Then, per the "archiving a doc strips its imperatives" rule in `docs/parity-coverage-policy.md`:
   - Prepend the standard archival header — copy the one at the top of
     `docs/completed/task-cuda-cgofree-spike.md` verbatim.
   - Keep the existing **Status: CLOSED** block, but change its `cuda/megakernel.cu … is now dead`
     sentence to say the scaffold was deleted in this closeout (cite the commit after you have it,
     or say "deleted 2026-09-12").
   - §0 ("read first"), §5's "Spike plan: start with (2)…", §7 "the deliverable", and §8's "box
     fills" list are imperatives. Lightest correct touch: one sentence in the status block saying
     every instruction below is a record of what the spike was asked to do, and that §5.2's option
     (2) is what shipped. Do not rewrite the sections.

5. **Leave a pointer stub** at `docs/cuda-megakernel-spec.md`, in the shape of
   `docs/plan-still-slow.md` (title, **Moved.**, new path, one line on what became of it, "pointer
   kept because other pages link here"). The stub is a *live* doc and the citation lint scans it,
   so it must contain **no `path:line` citations** — prose and one markdown link only.

6. **Fix the two live references.** `docs/decode-fusion-next.md:104` → the `completed/` path.
   `docs/QUEUE.md:2839` → the `completed/` path, and move the clause about the spec "still
   reading as a live prep artifact" into the past tense — that was G35's reason for existing, so
   it stays as history, e.g. "(the spec then still read as live; archived 2026-09-12)".

7. **Run the citation lint** exactly as `CLAUDE.md` §"queue_citation_lint" says — read the exit
   code directly, never through a pipe. Expect the `EXAMINED … live markdown document(s)` line to
   show one fewer live doc and one more `excluded as docs/completed/` than before the move (the
   stub adds one live doc back — so net live count is unchanged; the archived count is +1). Then
   look at the five index rows at `QUEUE.md:1998-2002`: if the lint prunes rows for a doc that
   left the live set, fine; if it leaves them, delete the five by hand and say so in the commit.
   `--update` if it asks for a re-key; never delete a citation to make a red go green.

8. **`docs/README.md` counts.** Recount `ls docs/completed | wc -l` for the "Archive —
   `completed/` (63)" heading (it was already 66 on disk before this move) and `ls docs/prompts |
   wc -l` for "`prompts/` (24)" (already 25 before this brief). Fix both to the real numbers.
   **Note:** `docs/README.md` already has an unrelated uncommitted hunk (the doc-count bump for
   `task-llamacpp-inproc.md`). Stage this file with `git add -p` so only the count lines you
   changed go into this commit, unless Francis says that hunk should ride along.

9. **Verify.** `go build ./...`; `go vet -tags cuda ./cuda/` and `go test -tags cuda -count=1
   -run NONE ./cuda/` (compile-only — no GPU on the Mac); `bash -n cuda/build_ptx.sh`;
   `grep -rn megakernel --include='*.go' --include='*.sh' --include='*.md' . | grep -v
   _to_delete | grep -v testdata` and confirm every remaining hit is one of the thesis-comments
   listed above or a `completed/` doc.

10. **Commit** — one commit, staged by path (the tree has other uncommitted work:
    `docs/task-int4-layout-2026-09.md`, `docs/task-halt-2026-09.md`,
    `docs/task-llamacpp-inproc.md`; none of it is yours). Suggested subject:
    `docs(cuda): archive megakernel spec, delete dead scaffold, fix package comment`. Do not push
    if the lint is red.

## Out of scope

Anything in `cuda/fused_qkv.cu`; revisiting K2; the G35 conclusion; the thesis-comments listed
above; `docs/internal/` beyond a courtesy fix.
