# Task: Go comments that say what is true now — guardrails kept, history moved out (2026-10)

> **Status: SCOPED 2026-10-09, not started.** Owner decisions taken 2026-10-09: do it; a `CLAUDE.md` rule (landed with
> this doc as § "Code comments"); change how docs link into code first (CC0), because that removes most of the cost
> of everything after it. Branch `comment-diet-2026-10`; merge when the track is done. Steps CC0–CC6. Nothing here
> needs a night run.
>
> **CC0 DONE 2026-10-09/10 on `comment-diet-2026-10` (not merged, not pushed):** lint support (symbol citations, pinned records) with tests
> first; 284 live-doc citations now name declarations; 35 records pinned (`docs/measurements/code-comments-2026-10/pins.tsv`); the door is
> closed (`path:line` in an unpinned doc is red); `scripts/remap_gate_citations.py` is gone. CC2 (`gate comments-only`) built and
> mutation-checked; CC3 pilot done and the rule adjusted (§6); CC4 wave A done (§7). Waves B and C, CC5 and CC6 not started.

## 1. Why

The Go comments have become a second lab notebook. Measured 2026-10-09 on `main` with a throwaway census script (CC6
keeps one):

| | |
|---|---|
| Go files / non-blank lines | 1,977 / ~386k |
| comment lines | ~73k — 19% overall, 26% outside `_test.go` |
| comments of 1–3 lines | 87% carry no history marker |
| comments of 10+ lines | ~1,660 blocks, ~27k lines; 69% (10–19 lines) to 97% (40+ lines) carry one |
| exported top-level doc comments in public packages that carry one | 214 of 1,111 |

A *history marker* is a date, a commit id, an audit or tracker id (`M-35`, `R10`, `P24`), a `docs/` or `task-`
reference, a measured figure with a unit, or a phrase such as "used to", "landed", "shipped", "found by". The count is
a ceiling on what moves, not a forecast: a marked comment is often a guardrail plus a pointer, and that stays.

Two examples of the long form:

- `gpu/prefillrunner.go:runModelToModelW` — 81 comment lines over the function, about 60 of them the 2026-09-12
  debugging story of the bias divergence: two causes, their fixes, a fused-residual attempt that regressed and was
  removed, and advice to whoever picks it up next.
- the fast-attention guard in `decoder/forwardn.go:Model.runLayersFromEmbedN` — 78 lines, including a timing table and
  "which took three tries to get right".

What it costs:

- **Every read.** A session that opens `decoder/residency.go` (941 comment lines of 1,816) or `cuda/resident.go`
  (1,559 of 4,330) reads the history before it reaches the code, on most turns of most campaigns.
- **Stale claims.** About 600 comment lines quote measured figures and about two dozen state open work ("STILL OPEN",
  "whoever picks this up next", "UNVERIFIED"). `CLAUDE.md`'s retraction rule says to grep for a struck figure with its
  unit; code comments are where nobody looks.
- **Duplication.** Most long comments summarize a `docs/measurements/` or `docs/completed/` record they already cite.
- **The library's face.** pkg.go.dev shows exported doc comments to the embedding Go developer the README now
  addresses. Audit ids and session history read badly there.

What the current style gets right, and must survive: these comments exist to stop the next session repeating a mistake.
`fastAttn` is a parameter rather than a global so spec-decode verify cannot turn it on by accident; the fast-attention
floor keys on the suffix length deliberately, and re-keying it needs a measurement at the new shape first; bias prefill
is admitted on Vulkan only. Each of those stays, as a sentence.

## 2. The rule

The working rule is `CLAUDE.md` § "Code comments". In full:

**Stays in a comment:** what the code does; its contract — ownership, units, concurrency, what a caller must check; the
reason for a non-obvious shape, in a sentence or two; every guardrail ("do not change X without Y"), with a pointer to
the evidence; a known limitation a caller would trip on, in one line.

**Moves out:** how something was found; dates; commit ids; which session or reviewer; audit and tracker ids used as
content (one may stay as the label on a pointer); measured figures and tables; retractions; open work.

**The test for a sentence:** would someone changing this code next week act differently because of it? Keep it. Does it
explain how we came to know? Move it.

**Length is part of the rule** (added 2026-10-10 after the CC3 pilot, §6). Say each thing once, at the declaration it
binds, in as many lines as the contract needs: an optional-interface or accessor comment is about six lines. Do not
restate the signature or a sibling's comment, do not list which backends, families or callers currently take a path (the
code is that list), and do not narrate the body. What a tightening drops that carries information (a per-backend list, a
rationale beyond a sentence or two) moves to `docs/code-notes/`; wording that only got shorter does not, and the CC2
moved-text report lists it as not found for the reviewer to read.

**Pointers name things, not lines.** A comment points at a doc by path and heading
(`docs/code-notes/gpu.md#runModelToModelW`) and at code by declaration (`BuildResident`'s `prefillLast` closure). There
are about 50 `file.go:NNN`-style references inside Go comments today; each goes stale on the next edit above its line.
They are converted as their package is worked.

**Directive comments are code.** `//go:build`, `//go:embed`, `//go:generate`, `//nolint` and `//lint:ignore` (15 today),
and an Example's `// Output:` block (one file) are never edited by this task; CC2 refuses a diff that touches them.

Before and after, `gpu/prefillrunner.go:runModelToModelW` (81 lines → 12; the Vulkan-only bias check it describes is
current, in `gpu/residency.go:webgpuBackend.BuildResident`):

```go
// runModelToModelW narrows a resident runModel to the plain dense-W8A8 ModelW
// shape PrefillLastW8A8 accepts, returning ok=false if the model uses anything
// outside it (MoE, MLA, SSM, DeltaNet, sliding window, int8 KV, partial RoPE).
//
// q/k/v bias is accepted, but callers must also check ModelW.hasBias() against
// the backend: bias prefill is proven bit-exact on Vulkan only and is declined
// elsewhere until proven there. See BuildResident's prefillLast gate in
// residency.go; history in docs/code-notes/gpu.md#runModelToModelW.
//
// The result is a zero-copy view over rm's buffers; callers must not Close it.
// hd is the head dimension, used only to tell genuine partial RoPE (declined)
// from ropeHalf set explicitly to hd/2.
```

## 3. CC0 — docs link to code by declaration; dated records are pinned to their commit

**Do this first.** Today a comment edit above a cited line shifts or breaks doc citations. After CC0 it does neither,
and the rest of this task stops paying a citation cost on every commit.

What is there (2026-10-09, tracked markdown outside `docs/completed/`):

| where | `path:line` links |
|---|---|
| `docs/citation-index.md` (generated by the lint) | ~1,050 |
| dated records — audits, reviews, `docs/measurements/` — 34 docs | ~1,200 |
| … of which name their tree commit in their first 15 lines | 12 docs, ~1,080 links |
| live docs — task docs, reference docs, `QUEUE.md`, `CLAUDE.md` — 36 docs | ~340 |

Since 2026-09-01, 225 of 2,854 commit subjects mention citation work, and 88 commits exist only to re-point citations
("citations re-pointed through … numbers only"). The line-content check does not test whether a doc's claim is true; it
notices that a line moved, and the routine answer is a renumbering. `scripts/remap_gate_citations.py` must also be run
exactly once per base, because a second run silently shifts links onto wrong lines.

The dated records already say what their links mean — the 2026-09-12 Metal audit's header says every `path:line` "is
keyed to" its tree commit — and the lint checks them against HEAD anyway. When the audited code is fixed, the lint goes
red, and the routine fix rewrites the record to describe code it never audited.

**CC0.a Dated records are pinned.** A record carries `<!-- citations-at: <commit> -->` near its top. The lint resolves
each `path:line` in that doc at that commit (`git show <commit>:<path>`), checks the file existed there and has the
line, and never rewrites it: `--update` skips pinned docs. Set the marker from the tree commit the doc already names.
For the 22 records that name none (~124 links), pin to the commit that added the doc's citing lines (`git blame` on
them); where a doc's links were written across several commits, convert that doc to CC0.b instead. A record that still
carries open findings (`docs/audit-2026-09-10.md`'s open half) stays pinned: whether an open finding still holds is
checked by whoever works it.

**CC0.b Live docs name the declaration.** The form is `path.go:Name`, or `path.go:Type.Method` for a method —
`decoder/forwardn.go:Model.forwardN`, `gpu/residency.go:residentDecoder.PrefillLast`. The lint checks that the file
declares that name (func, method, type, or a const or var, including inside a block). That survives every edit that does
not rename or move the declaration, and a rename is exactly when the doc should go red. For one statement inside a long
function, name the function and quote a short fragment in the prose — `gpu/residency.go:webgpuBackend.BuildResident`,
"`mw.hasBias() && c.Backend() != "vulkan"`"; the lint checks the name, the reader finds the fragment. The lint's
`anchor_for` already derives a line's enclosing declaration, so the migration is mostly mechanical: rewrite each live
`path:line` to `path:Name`, and list the ones outside any declaration (file-level comments, import blocks) for hand
conversion. `docs/use-from-go.md` carries a block that `examples/embed` must match byte for byte; convert both together.

**CC0.c Close the door.** After migration, a `path:line` in an unpinned doc is a red: "pin this doc (`citations-at`) or
name the declaration". The index keeps its commit table and drops its path section. Retire
`scripts/remap_gate_citations.py`. Update `CLAUDE.md` § "Citations and the pre-push hook" (its second bullet describes
the line scheme) and the lint's own docstring.

**Gate (mutation-checked, extending `scripts/test_queue_citation_lint.py`):** a renamed func → red; a removed const
inside a block → red; a pinned doc citing a path absent at its commit → red; a pinned line past the end of the file at
its commit → red; a `path:line` in an unpinned doc → red; moving a cited func within its file → green; rewriting the
comments above a cited func → green; a pinned doc whose code changed at HEAD → green. Then the whole tree: green, with
the counts of links rewritten, pinned and hand-converted in the commit message.

The lint needs Python 3.12+, so it runs on the Mac (Cowork's VM has 3.10).

## 4. CC1 — where moved text goes

1. **The doc the comment already cites holds it** (the common case): delete the story, keep the pointer. Read the doc
   first. A comment that summarizes a record sometimes carries the only copy of a detail; if so, that detail goes to 2.
2. **Otherwise move it verbatim** to `docs/code-notes/<package>.md` under a heading named for the declaration
   (`## runModelToModelW`), with one line above the text giving the file it came from and the date it moved. Verbatim
   means unedited — these are records, and the retraction rule applies to them as to any other page. A `path:line`
   inside moved text is converted to `path:Name` on the way (§3). The comment's pointer is
   `docs/code-notes/<package>.md#<name>`.
3. **Open work** ("STILL OPEN", "whoever picks this up next", "UNVERIFIED") goes to the task doc or queue entry that owns
   it, as a line pointing at the code-notes section. If nothing owns it, list it in the package's commit message and in
   §10 below for the owner, rather than inventing a home.

`docs/code-notes/` holds one file per package directory (`decoder.md`, `cuda.md`, `metal.md`, `gpu.md`,
`internal-serveapp.md`, …). It is off the read path on purpose: a session reads it when a pointer sends it there.

## 5. CC2 — prove the diff is comments only

A `gate` subcommand, since `cmd/gate` is the repo's tool for checks that must be able to go red:
`gate comments-only <base>`, with `--worktree` for uncommitted edits.

- For every `.go` file changed since `<base>`, parse both versions with `go/parser` without comments and print each with
  `go/printer`; the two must be byte-identical. String literals are tokens, so a change to an embedded MSL, CUDA or WGSL
  kernel string is a code change and red.
- **Protected comments are byte-identical or red:** `//go:` directives, `// +build`, `//nolint`, `//lint:ignore`, and
  any comment inside an `Example` function.
- **Only `.go` comments and `docs/` may change** in a comments commit; anything else is red.
- **Moved-text report** (listed, not red): every removed comment block, `//` stripped, is searched for in the same
  commit's added `docs/` text. Blocks not found are listed, so the reviewer can check each is covered by the doc the new
  comment cites (CC1.1) or was meant to go. This is what makes "move, don't delete" checkable.
- Then the usual cheap gates for the touched modules: `gofmt -l`, `go vet` with the tags the package uses, and the
  citation lint.

**Mutation check before first use:** an identifier rename, a changed string literal, a removed `//go:build`, an edited
`// Output:` and a non-doc file in the diff each go red; a pure comment rewrite goes green. A gate that has never been
seen red has not been shown to work.

## 6. CC3 — pilot, with the expectation written first

> **Pilot done 2026-10-10: the result is below the band.** Comment lines `gpu/prefillrunner.go` 248 → 154 (−38%),
> `decoder/residency.go` 941 → 776 (−18%), together 1,189 → 930 (−22%): under the 25% line below, so the owner decides (§10.1)
> before any rollout. What the numbers say: only 32 of the 248 and 97 of the 941 comment lines carry a history marker at all (the §1
> ceiling), so the two files are mostly contract, and `residency.go` is mostly interface documentation. The history that was there
> moved verbatim (`docs/code-notes/gpu.md`, `docs/code-notes/decoder.md`); the rest of the cut is wording tightened to the rule's
> "a sentence or two" with nothing dropped, listed as NOT FOUND by the moved-text report. Going further means cutting contract.
>
> **Rule adjusted, pass 2 (2026-10-10, owner: "adjust the rule").** The rule gained the "Length is part of the rule" paragraph
> (§2, `CLAUDE.md`), and the pilot files were tightened to it: `gpu/prefillrunner.go` 248 → 132 (−47%), `decoder/residency.go` 941 → 705
> (−25%), together 1,189 → 837 (−30%). The band is met for `gpu`; `decoder/residency.go`, interface documentation almost
> throughout, stays under it, and what remains there is contract. Expect a rollout to land between the 25% line and the
> band for contract-heavy packages and inside it for the ones the §1 census names (`cuda`, `metal`, `gpu`, `cmd/gate`).

Files: `gpu/prefillrunner.go` (248 comment lines of 1,039) and `decoder/residency.go` (941 of 1,816). Neither has an open
branch against it on 2026-10-09. `decoder/forwardn.go` was the first choice; it is out because `s10-pixtral` edits it.

**Expected band:** comment lines in the two files fall 40–65%.

- Below 25%: the rule is keeping the story. Stop and revisit the rule with the owner before any rollout.
- Above 75%: check that no guardrail went. The reviewer reads every removed block against the notes file and the
  pointer left behind.

The owner reads the pilot diff and the notes file, then decides (§10.1).

## 7. CC4 — rollout, one package per commit

Ordered by lines in long history-marked comments (the §1 census), with packages held back where an open branch touches
them on 2026-10-09:

| wave | package | comment lines | in long marked comments |
|---|---|---|---|
| A | `metal` | 4,324 | 1,213 |
| A | `cuda` | 4,155 | 1,115 |
| A | `gpu` | 2,946 | 749 |
| A | `cmd/gate` | 1,680 | 498 |
| A | `constrain`, `tokenizer`, `pull`, `internal/prequant` | ~2,080 | ~580 |
| B | `decoder`, by file groups, one commit each | 15,708 | 4,902 |
| B | `internal/serveapp` | 3,730 | 749 |
| B | `chat`, `multimodal` | ~1,270 | ~300 |
| C | each package's `_test.go` files, after its non-test files (CC5) | ~34k | ~9.5k |

Wave A is what no open branch touches today. Wave B waits for `s10-pixtral` and `libsurface-harness-2026-10` to merge.
**Before each package:** list open branches and worktrees (`git branch --no-merged main`, `git worktree list`) and run
`git diff --name-only main...<branch> -- <pkg>` for each. A file an open branch touches is skipped and picked up after
that branch merges; rewriting a comment block someone else is editing guarantees a conflict for no gain.

> **Wave A done 2026-10-10 (branches merged into `comment-diet-2026-10`, not pushed).** Comment lines, whole non-test package, before →
> after (`scripts/comment_diet.py census`; the "skipped" column is the files left alone because an unmerged branch edits them):
>
> | package | before → after | cut | history-marked | skipped (open branch) |
> |---|---|---|---|---|
> | `cuda` | 4,112 → 3,293 | −20% | 477 → 262 | `resident.go` (`q4k-narrow`) |
> | `metal` | 3,471 → 3,349 | −4% | 586 → 547 | `alias`, `backend`, `gemma4_moe`, `model`, `moe`, `prefill`, `gumbel`, `gumbel_sample`, `kernels` (`config-phase4-metal`, `r7b-metal-verify-mac`) |
> | `gpu` | 2,739 → 2,456 | −11% | 307 → 244 | `decoderunner.go`, `residency.go` (`webgpu-nogqa-second-pass`); `prefillrunner.go` was the pilot |
> | `cmd/gate` | 1,719 → 1,341 | −22% | 137 → 33 | none (`comments_only.go` is new, under the rule) |
> | `constrain` | 678 → 600 | −12% | 55 → 12 | none |
> | `tokenizer` | 756 → 652 | −14% | 51 → 8 | none |
> | `pull` | 388 → 360 | −8% | 33 → 13 | none |
> | `internal/prequant` | 258 → 217 | −16% | 33 → 8 | none |
> | **total** | **14,121 → 12,268** | **−14%** | **1,679 → 1,127** | |
>
> Over only the files that were worked the cut is larger (`cuda` −32%, `cmd/gate` −24%, `gpu` −18%, `metal` −16%). The four skipped
> branches are 2-3 weeks old and one to four commits each; `metal` is where it costs most (about 3,400 of its 4,341 comment lines are in
> skipped files). The combined diff against the pre-wave base is `gate comments-only` GREEN over 117 `.go` files, with `gofmt`,
> `go vet` (tagged variants) and the citation lint. History-marked lines that remain are mostly `docs/` pointers, which the marker
> regex counts. Moved history is in `docs/code-notes/{cuda,metal,gpu,cmd-gate,constrain,tokenizer,pull,internal-prequant}.md`.

One commit per package, or per file group in `decoder`. Subject:
`comments(<pkg>): history to docs/code-notes, guardrails kept (<before> → <after> comment lines)`. Body: the CC2
moved-text report, and any open work found (CC1.3).

## 8. CC5 — test files

The same rule, plus:

- **Test names are not touched** (`TestAuditT14_…` and the like). Gate manifests and the ledger reference them;
  renaming is a separate decision.
- A test's doc comment says what the test pins and why that matters. `CLAUDE.md`'s "a doc comment claiming coverage is
  not coverage" applies: if a trimmed comment claims coverage the body does not assert, list it in the commit message;
  do not fix it in this pass.

## 9. CC6 — keep it from growing back

The `CLAUDE.md` rule is the main defence. Sessions copy the style of the comments around them, and once CC4 is done the
surroundings agree with the rule.

`scripts/comment_census.py`:

- **report mode** prints the §1 census per package; this doc's before/after table comes from it;
- **diff mode**, run by the pre-push hook as a warning, prints added or changed comment lines in pushed `.go` files that
  carry a history marker (a date, a commit id, a tracker id, a figure with a unit, a `file.go:NNN` reference).

A warning, not a refusal, to start: it will have false positives ("timeout in ms" is a contract, not history).

## 10. Owner decisions, and open work found

1. **After CC3:** roll out as is, adjust the rule, or stop at CC0 (which pays for itself without the rest).
2. **After CC4:** CC6's diff mode stays a warning or becomes a refusal.
3. **Open work found in comments with no owning doc** — listed here as packages are done:
   - **`gpu/kv_slots.go:slotsBeforeContext`**: context shrinking on a discrete Vulkan GPU waits on a measurement of how a failed
     allocation behaves on real Vulkan hardware (`docs/QUEUE.md` says only "clamp-only"). `docs/code-notes/gpu.md#slotsBeforeContext`.
   - **`gpu/attention.go` (top of file)**: Metal's `float(pos)*invf[dd]` sin/cos pattern is unmeasured at long context.
     `docs/code-notes/gpu.md`, section "Stage 3 attention primitives: long-context RoPE".
   - **`metal/batch.go:promptStepAboveFloor`** is off "until the owner decides", and **`metal/paged_fence.go:pagedFenceOn`** is off
     (measured 0.85-0.88x); neither has a queue entry. The `pagedFenceOn` "re-measure on M26 before turning it on" is the wave-A
     agent's gloss, not the original text.
   - **`cuda/vision_encoder.go:addInPlaceHost`**: the tower's residual add round-trips through the host; a device-side kernel was
     noted as a natural follow-on. `docs/code-notes/cuda.md`.
   - **`cmd/gate/parity.go:whyNoResult`** checks `-run` selection only, not build-tag reachability, so a tagged test that cannot
     compile is reported as "selected … but reported nothing". Recorded only in `docs/completed/audit-2026-09-02.md` (N-41).
   - **`tokenizer`, `byteLevelKnobs`**: a GGUF with `pre="default"` (or none) is walked with GPT-2's shape; llama.cpp's default is a
     different multi-pass shape and needs a new `splitShape` and goldens. Recorded only in `docs/completed/audit-2026-09-10.md` (N-72).
   - Already owned, listed so they are not re-found: `cuda` `PrefillSeedArgmax` cannot be cancelled (`docs/queue-performance.md`);
     `constrain` N-79 control tokens in constrained JSON strings (`docs/audit-2026-09-10.md`, deferred); `metal` qGate LoRA gap
     (`docs/tasks/task-gpu-paths-2026-09.md`).
   - **`gpu/prefillrunner.go:runModelToModelW`** (CC3 pilot): bias prefill on Metal still diverges past nKeys~15 (cosine
     ~0.997-0.999 against sequential decode, not float noise), while Vulkan is bit-exact. The comment named the next step: a
     fused residual-epilogue kernel for O-proj/down-proj (the first attempt regressed Vulkan and was removed), isolated
     tests before wiring, and a check of whether the §3.2 pooled fidelity gate (`docs/completed/task-prefill-gap.md`)
     already clears Metal. `BuildResident` declines bias prefill off Vulkan meanwhile. No doc or queue entry owns it; the
     text is at `docs/code-notes/gpu.md#runModelToModelW`.

## 11. Not in scope

- Changing code, including names. Comments only.
- Comments inside embedded kernel source (MSL, CUDA, WGSL held in Go strings): they are code to the build, and editing
  one changes the kernel text.
- Rewriting dated records or `docs/completed/`.
- aikit: the same problem in a separate repo. CC0 and CC2 are reusable there.

## 12. Done when

CC0 is merged and the lint is green with no unpinned `path:line`; every package is worked, or skipped with a named
reason; the before/after census is recorded here; CC6 runs in the hook; `CLAUDE.md` § "Citations and the pre-push hook"
describes the new scheme.

<!-- doc-reviewed: 2026-10-09 -->
