# cmd/gate: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `cmd/gate`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## gpu gate header

Moved from `cmd/gate/gpu.go` (the comment above `gpuGate`) on 2026-10-09.

```text
The pre-tag GPU correctness gate. Run it on EACH GPU box, paste the verdict, then tag.

WHY THIS IS THE GATE. CI never runs the GPU backends: ci.yml only BUILDS and VETS under -tags
cuda, because the runners have no GPU. Metal is darwin-only and CUDA is Linux+NVIDIA, so no single
machine — and no CI job — can cover both. Every GPU correctness claim this project makes therefore
rests on a human running tests by hand on two boxes. This makes that reproducible: one command,
one verdict, provenance attached.

It is deliberately NOT "run everything". A gate that takes two hours gets skipped, and a skipped
gate is worse than no gate because it still implies assurance. It runs the checks that map to bugs
we actually shipped.

HONESTY RULES, all learned the hard way:

  - A skip is NOT a pass. `go test` prints "ok" for a package whose tests all skipped, so real
    runs are counted and SKIPPED is reported separately. Green here must mean tested.
  - Stray GPU processes invalidate results. A clean-tree control once "confirmed" a pre-existing
    failure while 3.4 GB of leaked serve processes held the card — both sides equally poisoned.
  - The CUDA suite must run SEQUENTIALLY (-p 1). Its tests each build a context and several load
    real models; parallel packages contend for VRAM and the failures come back as bogus numerics
    ("cosine 0.000000") rather than "you are out of memory".
  - GROUP ACCOUNTING (audit G-01). A tally computed from what EMITTED can never detect what did
    not: a block that dies mid-way emits nothing, and "ran 3" and "ran 4" are both plausible
    numbers. The expected groups are DECLARED up front and reconciled at the end — a group that
    emits no verdict, or an unexpected group id, is itself a FAIL.
```

## vramNote

Moved from `cmd/gate/gpu.go` (the comment above `vramNote`) on 2026-10-09.

```text
vramNote fires on a cosine of EXACTLY zero.

A cosine of 0.000000 is not a parity result — an all-zero buffer is what a failed allocation
leaves behind, and this gate's own history records an OOM wearing a parity bug's clothes long
enough that two people concluded "the tests just interfere" and moved on.

WORDING IS DELIBERATE: it states the READING and points at the entry; it does NOT name a
mechanism. "Suspect retention" was the obvious phrasing and is now DISPROVEN — A12 measured
Close() returning all 4892 MiB synchronously in 123 ms with a 0 MiB asynchronous tail. Naming a
mechanism a gate cannot see is how the last three explanations became someone's wasted afternoon.
```

## run: empty-cell note

Moved from `cmd/gate/gpu.go` (the comment inside `gpuGate.run`) on 2026-10-09.

```text
A FILTERED CELL THAT MATCHED NOTHING IS NOT A PASS, and the aggregate `g.ran == 0` check at
the end cannot see it: one empty cell among several full ones leaves g.ran > 0, so the cell
reports clean and its coverage is simply gone. Every -run pattern in this file is a literal
test-name prefix or alternation, so renaming a test silently empties its cell — the same
shape as the qwen3next oracle, where a -run pattern that could not match a required gate
produced "DID NOT RUN" for five weeks (docs/tasks/task-verification-surface-audit.md).
```

## noteIfAllSkipped

Moved from `cmd/gate/gpu.go` (the comment above `noteIfAllSkipped`) on 2026-10-09.

```text
noteIfAllSkipped records a FILTERED cell whose tests all SKIPPED. noteIfEmpty
cannot see this: a skip is a result, so `len(res.final) > 0` and it returns
early — which is precisely how G-01 survived. Two Metal groups printed PASS,
with their specific claim text, across at least two archived release logs while
executing zero tests, because every test in them skipped for want of
GOINFER_HEAVY_TESTS and `go test` exits 0 on an all-skip package.

This is the gate's own rule applied to the gate: A SKIP IS NOT A PASS. It is a
note rather than a hard failure because a cell CAN legitimately be all-skip on
a box without the assets — but it must say so loudly, since the alternative is
a green that vouches for nothing.
```

## noteIfEmpty

Moved from `cmd/gate/gpu.go` (the comment above `noteIfEmpty`) on 2026-10-09.

```text
noteIfEmpty records a FILTERED cell that matched no test at all. Split out from run so it can be
tested without spawning `go test`: pointing a cell at "./" from inside cmd/gate re-runs this very
suite, which re-runs the cell, and the first draft of that test sat for ten minutes.

An unfiltered cell is exempt — an empty -run means "everything", so emptiness there is a
different bug (no packages, or a build failure) that runCell's own policies already cover.
```

## failLineRe

Moved from `cmd/gate/gpu.go` (the comment above `failLineRe`) on 2026-10-09.

```text
failLineRe matches a test-level failure header. It deliberately does NOT match
bare `file.go:N:` lines: every passing t.Log emits one, and when this pattern
included them a crashed run reported a dozen `cosine=1.0000000 — PARITY` lines
as its "failure detail" while the actual SIGSEGV went unshown — the breadth
defeated detail()'s own crash fallback, which only fires when nothing matches.
Assertion lines are still shown, but only the ones BELOW a failing test (see
failureLines).
```

## crashExcerpt: head and frames

Moved from `cmd/gate/gpu.go` (the comment inside `crashExcerpt`) on 2026-10-09.

```text
The head (signal, address, goroutine) plus — crucially — the first frames
belonging to OUR code. A fixed prefix does not reach them: the real Metal
crash puts ~8 runtime/purego/reflect frames above the goinfer frame, so a
24-line cap truncated exactly before the test name, which is the one fact
worth printing.
```

## detectWebGPU

Moved from `cmd/gate/gpu.go` (the comment above `detectWebGPU`) on 2026-10-09.

```text
detectWebGPU probes for a real WebGPU adapter via a subprocess (gpu/adapter_probe_test.go's
TestAdapterProbe) — cmd/gate stays free of the gpu build tag and its cgo dependency, matching
detectBackend's nvidia-smi/uname style. Independent of detectBackend's cuda|metal|none choice on
purpose (G-09): a CUDA Linux box can ALSO have a working Vulkan adapter, and before this fix
nothing ever checked — the gate saw nvidia-smi, picked "cuda", and the WebGPU resident-parity
gates (qwen3.5 DeltaNet, granite/nemotron Mamba-2) ran only when a human remembered a private
env var. A software adapter (CI's lavapipe/llvmpipe) still counts as present: the group's own
tests already skip hardware-sensitive cases on one (newOrSkipHW), the same way a heavy tier's
tests self-skip on a missing checkpoint rather than the gate deciding for them.

V-21 (docs/review-2026-09-04.md): the subprocess's error was discarded and a REGEX MISS on the
success line was read as "no adapter" regardless of WHY it missed — a genuine "TestAdapterProbe
ran and found nothing" (which prints its OWN "ADAPTER_PROBE: none" line) looked identical to a
build break of ./gpu/, a panic, or any other reason the subprocess never reached that line at
all. An operator debugging "why didn't WebGPU get detected" saw "no adapter" and had no way to
tell a driver/hardware question from a build failure. w gets a visible note when the probe's
own output shows NEITHER recognized line — present/backend still resolve to false/"" either
way, because there is genuinely nothing to report as detected, but now it says so instead of
looking exactly like the hardware case.
```

## classifyAdapterProbe

Moved from `cmd/gate/gpu.go` (the comment above `classifyAdapterProbe`) on 2026-10-09.

```text
classifyAdapterProbe is detectWebGPU's pure classification of the probe subprocess's captured
output, pulled out so a test can drive it with synthetic output instead of needing a real
build break to reproduce (V-21, docs/review-2026-09-04.md). note is non-empty exactly when
out shows NEITHER the found-adapter line nor TestAdapterProbe's own explicit no-adapter line —
meaning the subprocess never reached either print, which is what a build break, a panic, or
any other reason the test body never ran looks like, as opposed to the test genuinely running
and finding nothing.
```

## runGPU: group environment

Moved from `cmd/gate/gpu.go` (the comment inside `runGPU`) on 2026-10-09.

```text
THE GROUPS DEFINE THEIR OWN ENVIRONMENT. Each group sets what it needs; nothing unsets them
for the groups that must NOT have them, so an operator who exports one "to be helpful"
silently changes what the other groups MEAN. Three consecutive red runs came from invoking
the gate as `GOINFER_HEAVY_TESTS=1 bash scripts/gpu_gate.sh`, which pulled the real-model
tests into the parity group as well: it then ran 608s against go's DEFAULT 600s timeout and
failed with no assertion line at all. Neutralised, and REPORTED rather than silently ignored —
an operator who set one deliberately must see that it did not take effect.
```

## cudaSuite

Moved from `cmd/gate/gpu.go` (the comment above `cudaSuite`) on 2026-10-09.

```text
---- 2a. CUDA kernel-level suite ----

The header used to read "CUDA kernels + parity" while running NEITHER the resident parity gates
NOR anything that asserts a forward. Every resident parity gate is behind `goinfer_testhooks`, so
for the whole of v0.10.x/v0.11.0 this block ran 53 kernel-level tests and the release record said
"full cuda suite" — while parity_manifest.json's shared_sets cover decoder/*.go ONLY, so deps_hash
could not go stale on resident.go either. A change to CUDA forward numerics had no enforced signal
anywhere in the gate. TWO groups now, because they answer different questions and one is not
evidence for the other (audit G-01: the artifact must not be adjacent to what it is read as).
```

## drainingTests

Moved from `cmd/gate/gpu.go` (the comment above `drainingTests`) on 2026-10-09.

```text
drainingTests derives the drain group FROM A MARKER rather than a list.

The draining tests take the device to refusal, which is A13's only reproducible poisoning
stimulus, so they run in their OWN process after the main tier and an exhausted device cannot
reach anything else. `drainsDevice(t, why)` in cuda/drain_marker_test.go is the marker; this walks
the test files tracking the enclosing `func TestX` and returns X for each one that calls it. A
hand-kept -run list would be a constant restating a property — the same drift shape the census
denominators keep making visible.
```

## isolatedTests

Moved from `cmd/gate/gpu.go` (the comment above `isolatedTests`) on 2026-10-09.

```text
isolatedTests derives the fresh-process group from needsFreshProcess(t, why) in
cuda/isolated_marker_test.go, the same way: edge-of-card tests that fit in a fresh process and not
after a few hundred others (measured 2026-09-28, see the marker's comment).
```

## cudaHeavy

Moved from `cmd/gate/gpu.go` (the comment above `cudaHeavy`) on 2026-10-09.

```text
---- 2c. heavy tier: the real-model group NOTHING has ever run ----

This tier existed and was never executed by anything: no script set the variable, so the tests
behind it were written, committed, and skipped forever. Declared here so it cannot quietly stop
running again, and TIMED into the verdict so its cost is visible up front rather than discovered
by someone waiting 28 minutes for a gate they thought took one.
```

## cudaHeavy: main-tier timeout

Moved from `cmd/gate/gpu.go` (the comment inside `cudaHeavy`) on 2026-10-09.

```text
TIMEOUT 90m, RAISED FROM 60m ON 2026-09-01 BECAUSE 60m HAD NO MARGIN LEFT.
Two runs of this tier on the same box, the same day, at adjacent commits:

	11:48 run   3320 s (55.3 min)   PASSED, by 4.7 min
	13:48 run   3612 s (60.2 min)   PROCESS DIED — "panic: test timed out after 1h0m0s"

Nothing about the tier changed between them; it simply drifted across the
line. A timeout that close reports a HANG when what happened was a slow
afternoon, and it reports it as a dead process with no test-level failure —
the most expensive kind of red to diagnose, because the crash head names
whichever test was merely unlucky enough to be running (TestSplitKVCrossover,
31 s in, entirely innocent).

This is the same fragility the retired scripts/heavy_gate.sh hit and fixed by
going to 120m — measured there as "the decoder tier loads ~15+ big real
checkpoints sequentially and needs ~50-60 min". cmd/gate did not inherit that
lesson when it replaced the script. 90m is ~50% headroom over the observed
60.2 min, which is enough for drift without letting a genuine hang sit for
two hours.

120m SINCE 2026-09-28: the tier measured 4,699 s (78 min) that day, 12 min under 90m — the
no-margin state the note above describes. The fresh-process tests below moved ~15 min out of
this process at the same time, so 120m is headroom, not a new expectation.
```

## ptxVersionRe (group 4 header)

Moved from `cmd/gate/gpu.go` (the comment above `ptxVersionRe`) on 2026-10-09.

```text
---- 4. PTX reproduces from source, each at the NVRTC it records ----

INTEGRITY: this block must ALWAYS reach pass/fail/skip. An earlier shell revision died on a bash
error midway and the gate still reported PASS — a check that can neither pass nor fail is the same
defect as one that can only fail (audit G-01). In Go the block cannot exit early without
returning, and the group reconciliation at the end catches it if it ever does.

Every .ptx states the toolchain that produced it in its own header:

	// Cuda compilation tools, release 12.6, V12.6.85

That is the artifact's provenance, and it is what we rebuild against — NOT whatever NVRTC this box
happens to default to. The tree legitimately carries a MIX (kernels added after a toolchain bump
were built at the newer one, and the audited ones are deliberately pinned), so a single-toolchain
rebuild reports a false FAIL on every file from the other era. That is what made this check
unpassable for the whole of v0.10.x/v0.11.0.

NOTHING IS EXEMPTED BY NAME. A file is only skipped when the NVRTC version IT RECORDS is not
installed here, and the skip names the version so it is actionable.
```

## metalCgoFree: build target

Moved from `cmd/gate/gpu.go` (the comment inside `metalCgoFree`) on 2026-10-09.

```text
G-03: build the METAL submodule entrypoint, not the root one. Without
`build.Dir` this compiled ./cmd/serve from the repo root -- a binary that
imports no Metal code at all, as cmd/serve/backendtag_guard_metal.go says in
so many words ("`-tags metal` does nothing on the root cmd/serve since
v0.10.0 (it builds no backend)"). It therefore built fine forever and the
group asserted "Metal is dlopen'd via purego-objc" about a binary with no
Metal in it. The CUDA half of this was fixed at d2c4858 (build.Dir = "cuda");
this half was not, so the gate has been passing on the wrong artifact.
```

## metalParity

Moved from `cmd/gate/gpu.go` (the comment above `metalParity`) on 2026-10-09.

```text
---- Metal resident PARITY gates — the forward is asserted here ----

G-02: no Metal cell passed `-tags goinfer_testhooks`, so the ritual's "full
metal suite" was the kernel tier plus the snapshot golden, and 59 files / 64
test funcs -- every Metal resident-parity gate among them -- were never
COMPILED, let alone run. RELEASING.md said the Metal run vouches for G10 and
G11; neither was built by the command it named. This is the hole cudaParity
describes closing for CUDA, mirrored.

Filtered to the resident-parity gates rather than the whole tagged tree: the
tag also selects long device tests that belong to other groups, and a cell
that quietly runs everything is how a timeout becomes indistinguishable from a
crash. -timeout is declared for the same reason cudaParity declares it.
```

## metalParity: Run pattern

Moved from `cmd/gate/gpu.go` (the comment inside `metalParity`) on 2026-10-09.

```text
metalParityRun (parity.go): named so this cell and
TestMetalGateIsListedOrExplicitlyNotRequired read the same string (V-07,
docs/review-2026-09-04.md) — a hand-copied pattern here is exactly how KernelParity/
metalParity/residentIdxParity went unmatched despite being gate-shaped.
```

## metalLifecycle

Moved from `cmd/gate/gpu.go` (the comment above `metalLifecycle`) on 2026-10-09.

```text
Metal HAD the same hole CUDA did — Close() froze a channel and freed nothing, leaking ~267 MB per
Load+Close on a 0.5B (aacec89). These run WITHOUT -short (the suite above uses it) because they
load real models, and they cover BOTH conditions: the sequential sawtooth, and a second model
alive — the case that made CUDA's first fix look correct when it was not.
```

## metalLifecycle: heavy env

Moved from `cmd/gate/gpu.go` (the comment inside `metalLifecycle`) on 2026-10-09.

```text
G-01: all four tests in the two Metal cells call requireHeavyModel, and
the gate deliberately UNSETS GOINFER_HEAVY_TESTS above so no ambient
value can change what a group means. The cells therefore have to set it
themselves, exactly as cuda-heavy does -- without it every test skips,
`go test` exits 0, and the group printed PASS while running nothing.
Measured 2026-09-01: skip/skip in 0.4 s before, pass/pass in 27 s after.
```

## metalPrefill

Moved from `cmd/gate/gpu.go` (the comment above `metalPrefill`) on 2026-10-09.

```text
The newest bug that maps to this doctrine. PrefillLast — the f16 simdgroup_matrix TTFT path —
emitted NaN logits at EVERY prompt length (including the minimal single-tile M=8) after the LM
head was pinned to int8: prefill still ran the int8 head weights through the int4 gemm_w4f16,
misreading them as packed nibbles (19ef47d). It hit the DENSE control, a model Metal ships.
Nothing exercised it against a real checkpoint until a hand-run, so it was invisible on push.
```

## metalPrefill: vacuous check

Moved from `cmd/gate/gpu.go` (the comment inside `metalPrefill`) on 2026-10-09.

```text
V-21 (docs/review-2026-09-04.md): the sibling cells above (metal-parity, metal-lifecycle)
already check cr.vacuous() alongside cr.RC != 0 — a cell whose named tests ALL skipped
(Pass==0 && Fail==0 && Skip>0, e.g. TestPrefillParity/TestPrefillNoNaN both declining for a
reason unrelated to "the gate needs a checkpoint," which os.Stat above already handles) has
RC==0 and would otherwise print PASS despite verifying nothing. This one lacked the check.
```

## webgpu

Moved from `cmd/gate/gpu.go` (the comment above `webgpu`) on 2026-10-09.

```text
---- 5. repo hygiene: run what CI runs, DERIVED rather than duplicated (B0) ----

---- W. WebGPU: adapter-independent of the primary backend (audit G-09) ----

Before this, the gate never ran ./gpu/ at all on a CUDA or "none" box (only on darwin, and even
then only by accident of detectBackend never being asked about WebGPU), and the resident-parity
gates it DOES have — qwen3.5 DeltaNet, granite/nemotron Mamba-2 — required a human to remember
GOINFER_DNET_PARITY / GOINFER_SSM_PARITY. This group sets both itself, the same way every other
group here sets what it needs (see runGPU's header comment on that rule).
```

## webgpu: resident-parity cell

Moved from `cmd/gate/gpu.go` (the comment inside `webgpu`) on 2026-10-09.

```text
The resident-parity gates G-09 found opt-in-by-private-env-var. Every qwen3.5 fixture
(dense AND MoE) has its model.safetensors gitignored — only config.json is tracked — same
as the granite/nemotron Mamba-2 fixtures; all skip gracefully when absent (their own tests
stat the weights file, not the directory).

V-06 (docs/review-2026-09-04.md): this comment used to claim the dense qwen3.5 fixture WAS
tracked, "so this has real coverage on every clone" — wrong, and it hid a real bug: on a
clone without the fixtures, TestQwen35ResidentParity's two t.Run subtests both skip, and Go
reports a parent whose subtests ALL skipped as a top-level PASS, not SKIP. This cell counts
top-level results only (TopLevelOnly: true, gpu.go's gateConfig above), so that vacuous pass
registered as Pass=1 and made cr.vacuous() (Pass==0 && Fail==0 && Skip>0) false — a webgpu
forward could be broken with nothing ever forwarded here and this cell would still print
PASS. Fixed at the source: TestQwen35ResidentParity now tracks each subtest's own
t.Skipped() and skips itself when none of them ran, so the tally sees a real Skip instead of
a vacuous Pass.
```

## repoHygiene

Moved from `cmd/gate/gpu.go` (the comment above `repoHygiene`) on 2026-10-09.

```text
This block used to run `gofmt -l .` and `go vet ./decoder/ ./cmd/...` — a hand-written list that
was a strict SUBSET of CI's: no staticcheck at all, vet without the goinfer_testhooks tag and over
narrower packages, no build. So CI went red on `staticcheck -tags cuda` and stayed red for three
commits, and running this gate — the thing you run INSTEAD of remembering — would not have caught
it either. Adding staticcheck would fix the instance and leave the class open: the next check CI
gains reopens the gap. So the list is DERIVED from .github/workflows/ci.yml by ci_checks.py, and a
check CI adds appears here with no edit to this file.
```

## repoHygiene: citation lint

Moved from `cmd/gate/gpu.go` (the comment inside `repoHygiene`) on 2026-10-09.

```text
The queue's citations, commit AND path:line. A state document is cited without being
re-derived, so a wrong reference in it propagates with more confidence than the same error in
conversation — 9e5f8fa was cited several times, from the file, without anyone opening it, and
cuda/resident.go (a line there) kept an audit critical listed as open for weeks after it was fixed.
```

## withGoBin

Moved from `cmd/gate/gpu.go` (the comment above `withGoBin`) on 2026-10-09.

```text
withGoBin puts `go env GOPATH`/bin first on PATH. CI installs its tools (staticcheck v0.8.1, built by .github/actions/staticcheck) with `go
install` and then calls them by bare name, which works there because setup-go puts GOPATH/bin on PATH.
A developer shell need not, and on nobara it did not: the gate reported "staticcheck: command not
found" as two failed CI checks (2026-09-28) while the pinned binary sat in ~/go/bin. Prepending the
same directory reproduces CI's environment; a missing binary still fails, now with the install hint.
```

## verdict: dirty tree

Moved from `cmd/gate/gpu.go` (the comment inside `gpuGate.verdict`) on 2026-10-09.

```text
THREE STATES, NOT TWO. Every check is green here. A dirty tree is not a failure of the CHECKS
— it is a failure of PROVENANCE: this verdict names a commit, and an uncommitted edit means it
does not describe what that commit contains. Collapsing the two loses the distinction a reader
actually needs: is the CODE broken, or is the EVIDENCE broken? It used to print
"repo <sha>+dirty" in the provenance block and then PASS as normal — so the gate could emit a
verdict reading "PASS at <sha>" for a tree that is not <sha>, with the whole distinction
carried by a three-character suffix in a different block. Verdicts get pasted into tag
messages; that is what this gate is FOR.
```

## parityGates: int4-forward

Moved from `cmd/gate/parity.go` (the comment above the `int4-forward` entry of `parityGates`) on 2026-10-09.

```text
The int4 FORWARD gate (23 fixtures / 16 architectures) is the broadest quant check here, and
it was missing from this list until 2026-08-26 -- so when it went red in the v0.15.0-prep
sweep it surfaced only as an anonymous "1 fail" with no gate name, and attributing it took
hours. A required gate that is not named here is a gate whose failure nobody can read.
```

## parityRealckptGates: canonical gates

Moved from `cmd/gate/parity.go` (the comment inside `parityRealckptGates`) on 2026-10-09.

```text
ONE OR TWO CANONICAL GATES FOR SIX FAMILIES THAT HAD NONE (2026-09-02, audit G-05 follow-up).
gpt_oss, granite, laguna, glm4_moe, cohere, cohere2 and dense qwen3.8 were shipped families
with no required gate ANYWHERE in the checkset — not in parityGates either, since none has a
tiny fixture. "Not required" was never a decision about them; it was the absence of one, and
the sweep's own report could not distinguish the two. Every asset below is registered in
testdata/assets.json and verified present on the sweep box, so none of these is a permanent
SKIP-blocker.
```

## parityRealckptGates: smollm3

Moved from `cmd/gate/parity.go` (the comment inside `parityRealckptGates`) on 2026-10-09.

```text
smollm3's asset (testdata/assets.json GOINFER_SMOLLM3_3B) and gate
(decoder/smollm3_real_test.go) landed together; this line did not, leaving the family
with a real gate the sweep could not even report on (TestRealckptGateIsListedOrExplicitly
NotRequired). Caught by CI, not found by inspection — the two files can look complete on
their own and still not be reachable.
```

## parityRealckptGates: lfm2 and mistral3

Moved from `cmd/gate/parity.go` (the comment inside `parityRealckptGates`) on 2026-10-09.

```text
lfm2 and mistral3 repeated the exact smollm3 gap the comment above describes — their
gates/assets landed (commit 493897af) without this line, so TestRealckptGateIsListed
OrExplicitlyNotRequired was red on main until this fix. Landing three MORE real-checkpoint
families in the same session as this fix is the reason to trust it's not a one-off: the
check is doing its job, the discipline of registering in the SAME change is what was
missing, not the check itself.
```

## parityRealckptGates: internlm2

Moved from `cmd/gate/parity.go` (the comment inside `parityRealckptGates`) on 2026-10-09.

```text
internlm2 repeated the exact smollm3/lfm2/mistral3 registration gap: its gate and asset
(commit d1449f4) landed without this line, leaving TestRealckptGateIsListedOrExplicitly
NotRequired red on main. First run had FAILED anyway (cosine 0.87 — since found to be a
corrupt reference, not a goinfer defect; see docs/parity-coverage-policy.md's RESOLVED
note) so the missing registration was doubly invisible until the fix made the gate green.
```

## parityRealckptGates: cuda qwen25vl gate

Moved from `cmd/gate/parity.go` (the comment below `parityRealckptGates`) on 2026-10-09.

```text
Not in parityRealckptGates/realckptNotRequired (and not found by realckptDirs' decoder-only
scan): cuda/qwen25vl_resident_real_test.go's TestQwen25VLResidentReal_gate — gap 0's real-
checkpoint continuation of qwen25vl-real above (that gate is prefill-only; this one exercises
GenerateQwenVL's actual resident-decode path, one CPU-vs-hybrid step on the real image, via
UploadKV + ForwardMRoPE). It lives in package cuda (needs the cuda backend's init() to
register "cuda" with decoder — decoder itself cannot import cuda, an import cycle) and is
tagged `cuda && goinfer_testhooks`, the same convention as its siblings
(uploadkv_parity_test.go, forwardmrope_parity_test.go) — not `realckpt`, so it is outside this
list's discipline by construction, the same way those two already are.
```

## emitGates

Moved from `cmd/gate/parity.go` (the comment above `emitGates`) on 2026-10-09.

```text
emitGates are the numeric-oracle gates expected to record a manifest row under EMIT_MANIFEST.
Family here is the manifest family the gate writes, which is why the pair is the other way round
from the lists above — a detail that once produced six rows with the columns swapped.
```

## assetNeverBuilt

Moved from `cmd/gate/parity.go` (the comment above `assetNeverBuilt`) on 2026-10-09.

```text
assetNeverBuilt names required gates whose asset has NEVER been built anywhere, so no invocation
can make them green. They are reported and counted as coverage gaps, not blockers.

THE LIST IS EMPTY, AND IT GOT THERE THE ONLY CORRECT WAY (2026-08-18, v1.0 gate 1.3):
TestW4A8DecodeParity was the sole entry — it needs a MATCHED int4+int8 .giw pair, and only int4
bundles had ever been produced — until the pair was built from one source GGUF (so "matched" is by
construction, not by belief) and the gate ran green on first invocation. The machinery stays: the
next gate whose asset has never been built belongs here, and an empty list is the honest current
state rather than a reason to delete the classification. The only correct way OFF this list is to
build the asset.
```

## gateRunFilter

Moved from `cmd/gate/parity.go` (the comment above `gateRunFilter`) on 2026-10-09.

```text
gateRunFilter reads GATE_RUN — an optional narrowing filter for a granular re-run of the sweep
(e.g. after fixing a specific blocker, without paying for the full ~2h two-cell run again to get
the SAME checkset classification, ledger/neverConfirmed handling and verdict logic a targeted
`go test -run` alone would skip). "" (the default) means the full sweep, unchanged.
```

## parityCells: realckpt Run

Moved from `cmd/gate/parity.go` (the comment inside `parityCells`) on 2026-10-09.

```text
DERIVED FROM THE TAGGED FILES, not hand-written. The hand-written pattern could
not reach TestQwen3NextReal_oracle for weeks — the sweep reported it "DID NOT RUN
(blocker)" and every diagnosis went looking at the 163GB asset, which was present
and resolving the whole time — and after that was patched it still missed five more
(audit-2026-09-02 G-05). A -run filter that cannot reach a gate is not a skip; it is
a gate that silently does not exist. The cell is still filtered rather than
unfiltered because the realckpt tag also carries perf and diagnostic tests that the
release sweep is not for; the filter selects on SHAPE now, not on a name list.
```

## assetPreflight

Moved from `cmd/gate/parity.go` (the comment above `assetPreflight`) on 2026-10-09.

```text
assetPreflight resolves the asset environment from the SHARED REGISTRY (testdata/assets.json) and
reports what it resolved.

This exists because the same invocation error produced a false "15 BLOCKER(S)" three separate
times while the tree was fine every time: the gates skip-if-absent and a skip is reported as a
blocker, so an unset variable and a genuinely missing checkpoint were indistinguishable in the
output. The registry is the single implementation of "is this asset present" — an earlier table
inside the sweep tested `[ -e "$path" ]`, which a DIRECTORY satisfies, so it reported resolved for
three entries where the loader wanted the .gguf FILE inside.

A preflight that does not run is announced LOUDLY rather than left to become a blocker cascade:
without python3 every asset-gated gate skips and the count is about the failure, not the tree.
```

## runParity: composition

Moved from `cmd/gate/parity.go` (the comment inside `runParity`) on 2026-10-09.

```text
THE COMPOSITION, NOT JUST THE VERDICT. This gate's axes are family × quant × loader, and a
pass COUNT alone cannot distinguish "the axes are covered" from "an axis collapsed to one
value" — which is exactly how the forward goldens stayed f32-only through nine refreshes
behind an accurate count.
```

## classifyChecks: neverConfirmed skip

Moved from `cmd/gate/parity.go` (the comment inside `classifyChecks`) on 2026-10-09.

```text
neverConfirmed's own doc comment already promises "never blocks a tag" —
but until this branch existed that promise covered only a FAIL outcome
(via the ledger classify path below), never a SKIP. Measured 2026-09-18:
TestNemotron35LightningReal_oracle sat in neverConfirmed since 2026-09-13
specifically because its asset was absent on this box, and its SKIP still
counted as a blocker every run since — the deferral was written down but
never actually took effect. This is the fix.
```

## whyNoResult

Moved from `cmd/gate/parity.go` (the comment above `whyNoResult`) on 2026-10-09.

```text
whyNoResult distinguishes the two causes of "no result" that look identical in a report and are
fixed in completely different places.

This exists because the distinction cost five weeks. TestQwen3NextReal_oracle was reported as
"DID NOT RUN (blocker)" sweep after sweep; the realckpt cell selected on -run "Qwen35|Real_gate"
and the test is named ...Real_oracle, so no pattern could ever select it. The wording implied a
missing asset, so that is where three sessions looked — one of them verifying all 41 shards of a
163 GB checkpoint that was present and resolving the whole time.

A gate no pattern selects cannot be fixed by any machine, asset or environment. A gate that IS
selected and still produced nothing is a different problem entirely. Saying which halves the
search.

KNOWN GAP (N-41, audit-2026-09-02.md, found 2026-09-11): this only checks -run REGEX selection,
not build-tag reachability. parityCells' base cell has an empty Run (deliberately — see
TestBaseCellIsUnfiltered) and no Tags, so the loop's first branch below fires for EVERY test name
via the sweep's one real call site, and the UNREACHABLE fallthrough is provably unreachable
there — see TestParity_missingGateSaysWhichCause. Worse, that means a realckpt-tagged test base
cannot even COMPILE would be diagnosed as "selected (unfiltered) but reported nothing" (implying
an asset/build problem) instead of UNREACHABLE (implying a pattern/name problem) — reproducing
the original TestQwen3NextReal_oracle misdiagnosis this function was built to stop, for the exact
shape of bug that caused it. A real fix needs per-cell reachability (e.g. `go test -tags <cell's
Tags> -list '^<test>$' <cell's Pkgs>` and checking for a match), not a -run string match against
Run alone. Filed, not fixed, to avoid rushing a change to a pre-push-adjacent gate's own logic.
```

## neverConfirmed

Moved from `cmd/gate/parity.go` (the comment above `neverConfirmed`) on 2026-10-09.

```text
neverConfirmed names a REQUIRED gate that is deliberately absent from the ledger, with the reason.
A gate here stays permanently FIRST-RUN: its failure is reported as an ITEM and never blocks a
tag, so an entry is a decision to accept that, not a formality.

EMPTY, AND EMPTY IS THE HONEST STATE (2026-09-02, audit G-04). The ledger was bulk-seeded once on
2026-08-14 and never touched again, so five required gates — TestInt4_forwardParity ("the broadest
quant check here"), TestW4A8DecodeParity, TestNemotron_textParity, TestNemotron3NanoMoE_textParity
and TestQwen3NextReal_oracle — sat FIRST-RUN for two and a half weeks WHILE A CONFIRMED PASS FOR
EACH SAT IN THE v0.15.0 SWEEP LOG. Nothing turned the one into the other: `reconcile` is advisory
by design, and no test asserted `required ⊆ ledger`. TestParity_everyRequiredGateIsConfirmed is
that assertion, and this map is its only escape hatch — deliberately a code change with a written
reason rather than a state the ledger can drift into by nobody doing anything.
```

## neverConfirmed: removals of 2026-09-18

Moved from `cmd/gate/parity.go` (the comment inside `neverConfirmed`) on 2026-10-09.

```text
2026-09-18: 9 of the 10 v0.18.0-deferral entries above this line were REMOVED here, per
this map's own instruction two paragraphs up ("move each back... the moment a real sweep
on a box with the checkpoints actually runs it; do not let this entry persist past that").
The v0.19.0 §C1 sweep ran all 10 for real: 9 PASSED and were promoted to the ledger
(TestQwen3MoeReal_oracle, TestQwen38GGUF_weightDiff, TestSmolLM3_3bReal_gate,
TestLFM2Real_gate, TestMinistral3Real_gate, TestGraniteDenseReal_gate,
TestOlmoHybridReal_gate, TestInternLM2_1_8bReal_gate, and TestQwen2MoeReal_oracle/
TestQwen25VLReal_gate promoted earlier the same day) — leaving them here would have kept
asserting "not run this release" about a release that just ran them. TestOlmo3Real_gate
also ran (FIRST-RUN, no confirmed prior result) and moved to awaitingFirstConfirmation
instead, which is what running-but-unconfirmed actually means per this file's own
three-state contract. Only TestNemotron35LightningReal_oracle's asset is still absent from
this box, so only it stays.
```

## neverConfirmed: TestQwen3NextReal_oracle

Moved from `cmd/gate/parity.go` (the comment inside `neverConfirmed`) on 2026-10-09.

```text
2026-10-01: TestQwen35Real_gate2FullModel, the other gate this block named, left the list. It fit and passed on this
box in the v0.20.0 sweep run 2 (bcf50a49) and the scoped re-validation (70be7081), and is in the ledger.

v0.19.0 §C1 SWEEP FINDING (2026-09-18, Francis via Claude): these two gates genuinely ran
(not asset-missing) and genuinely cannot fit THIS BOX under the fit-guard's 70% budget —
not a flake, not a code defect. Both load Qwen3.6-35B-A3B or Qwen3Next-80B at
full/pinned context (not auto-capped), and both now Skip (not Fatalf) on a
decoder.ErrWontFitResident decline (decoder/real_oracle_test.go, decoder/qwen35_gate2_test.go)
rather than treating capacity refusal as a test failure. Measured: this box has 62GB RAM,
so the fit-guard's 70% ceiling never authorizes more than ~43.4GB even fully idle.
```

## awaitingFirstConfirmation

Moved from `cmd/gate/parity.go` (the comment above `awaitingFirstConfirmation`) on 2026-10-09.

```text
awaitingFirstConfirmation names a required gate that has NEVER produced a confirmed result, with
the date it became required and what will confirm it. It is the third state, and it is not the
same as either neighbour:

	ledger entry          a person looked at a value and said it is correct.
	neverConfirmed        we have decided to accept a permanently non-blocking gate.
	awaitingFirstConfirmation   nothing has been decided yet, because the gate has not run.

COLLAPSING THIS INTO EITHER NEIGHBOUR WOULD BE A LIE IN A DIFFERENT DIRECTION. Promoting these
from "it did not appear in the sweep's SKIP list, so it must have passed" would bank an inferred
value as a baseline — exactly the auto-promotion `gate ledger` refuses to do — and the inference
is not even sound, since the sweep that ran them discarded unlisted FAIL counts (G-05). Filing
them under neverConfirmed would assert a decision to leave them non-blocking forever, which is
the opposite of the intent: they were made required BECAUSE their families need cover.

So they are first-run, which is the correct and honest outcome — their failures are ITEMS until a
sweep produces a value a person promotes. The date is required so an entry that quietly becomes
permanent is visible as one.
EMPTY, 2026-09-18. The four gates that sat here since 2026-09-08 (TestQwen2MoeReal_oracle,
TestLagunaReal_oracle, TestQwen38Real_oracle, TestQwen25VLReal_gate) have all now produced a
confirmed PASS and been promoted to the ledger. Laguna and Qwen3.8's first-run FAILs (recorded
in the git history of this map) turned out to be genuine int4 near-tie sensitivity, not a code
defect — see docs/measurements/int4-neartie-laguna-qwen38-2026-09-18.md — and both now run
(and PASS) at int8 instead of int4; that document is also the retraction record for an earlier,
wrong plan to move both into neverConfirmed permanently on an unverified "int8 doesn't fit"
premise.
EMPTY again, 2026-10-01. TestOlmo3Real_gate, here since 2026-09-18, was confirmed and promoted to the ledger. Its
0.992789 cosine was a wrong REFERENCE, not quantization (the gate is f32): the golden had been pinned under
transformers 5.12, whose Olmo3 applies YaRN to every layer, while the Olmo 3 paper and transformers 5.15 put it on
full-attention layers only, as goinfer does. Re-pinned under 5.15 it passes at cosine 1.000000
(docs/measurements/olmo3-golden-repin-2026-10-01/).
```

## realckptNotRequired

Moved from `cmd/gate/parity.go` (the comment above `realckptNotRequired`) on 2026-10-09.

```text
realckptNotRequired names a gate-shaped test in a `//go:build realckpt` file that the sweep RUNS
but does not require, with the reason. Every such test must be here or in parityRealckptGates —
TestRealckptGateIsListedOrExplicitlyNotRequired fails otherwise.

WHAT AN ENTRY COSTS, EXACTLY. Since the sweep counts an unlisted FAIL as a blocker, an entry here
does NOT make a failure harmless. What it forgoes is the other two outcomes: a SKIP does not block
(the asset may not exist on this box), and the gate gets no named row in the checkset table. That
is a much smaller claim than "not required" used to be, and it is the one being made.

EVERY ENTRY IS NOW "THE FAMILY IS COVERED ELSEWHERE", AND THAT IS THE ONLY ACCEPTABLE REASON.
The second kind this map briefly held — "this family has no required gate anywhere" — was not a
reason, it was the absence of a decision: gpt_oss, granite, laguna, glm4_moe, cohere, cohere2 and
dense qwen3.8 were shipped families the checkset said nothing about. All seven are required gates
now, their assets registered and verified present, so what remains here is genuinely extra depth
on a family that already has a canonical gate. A new entry claiming anything else is a coverage
hole wearing a reason, and reviewing it is the point of making it a code change.
```

## metalGateTests

Moved from `cmd/gate/parity.go` (the comment above `metalGateTests`) on 2026-10-09.

```text
metalGateTests scans metal/'s goinfer_testhooks-tagged test files for gate-shaped top-level
tests — the Metal analogue of realckptGateTests, which V-07 found had no equivalent: a
regression of the exact class G-08 repaired (TestBatchedVerifyKernelParity, the Metal
decode==verify bit-identity gate) could pass `gate gpu` on the Mac simply by not being matched
by any cell's -run pattern, with nothing to say the cell had nothing to say about it.
```

## webgpuDirs

Moved from `cmd/gate/parity.go` (the comment above `webgpuDirs`) on 2026-10-09.

```text
webgpuDirs, webgpuGateTests, webgpuParityRun and webgpuNotRequired are the WebGPU twin of the
Metal scan above (audit-2026-09-10 G-10). webgpu-parity used to select only "ResidentParity",
so gate-shaped goinfer_testhooks-tagged gpu/ tests like the staged-int4 matmul gate, ForwardN
parity, both DeltaNet kernel parities, MoE route and the int4 expert GEMV ran in no gate cell
and no CI runner.
```

## realckptRun

Moved from `cmd/gate/parity.go` (the comment above `realckptRun`) on 2026-10-09.

```text
realckptRun derives the realckpt cell's -run from the tree, and returns a note saying how.

THE PATTERN THAT COULD NOT REACH A REQUIRED GATE, GENERALISED. legacyRealckptRun was widened by
hand each time someone noticed a miss, and on 2026-09-02 five gates still matched nothing:
TestGemma4_26B_gate, TestGlm4MoeAir_gate, TestLagunaGGUF_gate, TestQwen38GGUF_gate and
TestGptOssReal_logitParity. Being unlisted as well as unselected, they were not even reported as
DID NOT RUN — the sweep had no way to say a word about them. Deriving the pattern from the tagged
files makes "a gate exists" and "the sweep can reach it" the same fact.

The union with parityRealckptGates is not belt-and-braces: TestQwen35GGUF_weightDiff is required
and is NOT gate-shaped, so the scan alone would drop it.
```

## unlistedFailures

Moved from `cmd/gate/parity.go` (the comment above `unlistedFailures`) on 2026-10-09.

```text
unlistedFailures returns the tests that FAILED and are not one of the named gates.

THE FAIL COUNT THE SWEEP USED TO THROW AWAY. `blockers` came only from the checkset, so a FAIL in
any of the ~36 family parity tests outside it changed nothing: the cell line printed "N fail" and
the verdict still read ALL REQUIRED GATES GREEN, exit 0. The checkset is still the thing that says
a NAMED gate is green — that is the decision this gate exists to make — but "nothing else in the
sweep failed" is a separate and much cheaper claim, and it was not being made at all.

EXACT match, unlike catchAllSkips' containment: a test whose name merely CONTAINS a gate name is a
different test, and hiding its failure is the defect, not the feature. Exactness is also what
keeps B14 intact — a named gate's FIRST-RUN failure is excluded here because its name matches
exactly, so it stays an ITEM.
```

## extraBlockers

Moved from `cmd/gate/parity.go` (the comment above `extraBlockers`) on 2026-10-09.

```text
extraBlockers is everything blocking that the CHECKSET CANNOT SEE, and it returns the count so
there is exactly one place to get the arithmetic wrong.

`blockers` used to come only from classifyChecks over the named gates, which meant the sweep
discarded two whole categories: a FAIL in any of the ~36 family parity tests outside the checkset,
and a cell that died without printing a single --- FAIL line. Both left the verdict reading ALL
REQUIRED GATES GREEN, exit 0 (audit-2026-09-02 G-05).
```

## gate quick selection header

Moved from `cmd/gate/quick_select.go` (the comment above `quickModule`) on 2026-10-09.

```text
`gate quick`'s SELECTION (TE7(a), docs/tasks/task-test-efficiency-2026-09.md): changed files ->
their packages -> every test binary, in any of the five modules, that can observe them.

THREE TIERS, because "imports the change" is not the only way a test here observes one:

  - AFFECTED: the test binary compiles a changed file (the file's package is in its
    `go list -deps -test` closure, or it embeds the file). Its binary is new, so it re-runs.
    This is the tier the equivalence property pins (quick_test.go): it never drops a package
    `go list -deps -test` says depends on the change.
  - OBSERVER: the tests can READ the change as data. Measured 2026-09-28, not assumed: three
    decoder tests (TestEnvVars_docAndCodeAgree, TestBackendBanner_usesTheReport,
    TestGoldenNames_matchTheFileOnDisk) open 4,883 files across the whole tree — metal/, cuda/,
    gpu/, internal/, docs/, even the gitignored vendor/ and .claude/ — so an edit to a metal
    .go file is observable by decoder's tests although decoder imports nothing in metal. An
    import-graph selection alone would report GREEN on a change CI then fails.
  - CACHE-CHECKED: every other test binary. A test can open a path it builds at run time, which
    no static scan sees, so nothing is dropped: these run WITHOUT -count=1 and Go's test cache —
    which records every file, directory listing and env var a test binary actually read —
    replays each one whose recorded inputs did not change (TE7(b): decoder replays in 2 s).
    That is sound for reads inside the test's own module root, and the cache's one blind spot
    (cmd/go does not re-check files OUTSIDE the module root) is closed where determinable by
    forcing -count=1 on a cross-module reader (see crossModuleReader).
```

## realQuickConfig: cuda cross target

Moved from `cmd/gate/quick_select.go` (the comment inside `realQuickConfig`) on 2026-10-09.

```text
Measured 2026-09-28 on darwin/arm64: `CGO_ENABLED=0 go build -tags 'cuda
goinfer_testhooks' ./cuda/...` fails (undefined: gpu.MappedHostBuffer, gpu.Graph,
gpu.Event — aikit/gpu's CUDA surface is linux-only), while the same vet for
GOOS=linux GOARCH=amd64 passes. So it is linted cross-target and its tests are NOT
RUN off linux.
```

## cmdEnv

Moved from `cmd/gate/quick_select.go` (the comment above `cmdEnv`) on 2026-10-09.

```text
cmdEnv is the environment for a go command about module m, built for target ("" = this host).
overlay, when set, is passed to every go command (the equivalence tests mutate through it, so no
real file is ever edited).

forTest builds a TEST cell's environment, and differs in two ways, both measured on this Mac:

  - GOWORK is left to auto-discovery wherever the repo's own go.work (or its absence, for the
    root) already resolves the module the way its CI job does. An explicit GOWORK is inherited by
    every `go` a test spawns: gpu's TestParentSelfSkipsWhenNoSubtestRan runs `go test` in a scratch
    module and fails with "setup failed" under an explicit workspace, and decoder's
    TestGoDoc_listsEveryFieldOfOptionsAndSamplingParams runs `go doc`, which under GOWORK=off hits
    the stale vendor/ below. Both pass under a hand-typed `go test`, so the harness must not
    be what turns them red.
  - GOINFER_HEAVY_TESTS is removed: a shell that exported it would turn the day loop into the
    90-minute heavy tier.
```

## cmdEnv: vendor

Moved from `cmd/gate/quick_select.go` (the comment inside `cmdEnv`) on 2026-10-09.

```text
A gitignored, stale vendor/ (this Mac's root has one from 2026-09-20: aikit v1.46.0
against go.mod's v1.50.1) puts module-mode builds in vendor mode and fails them with
"inconsistent vendoring". CI's checkout has no vendor/, so -mod=readonly is what CI runs.
```

## scan: walker split

Moved from `cmd/gate/quick_select.go` (the comment inside `testRoot.scan`) on 2026-10-09.

```text
Only the tree walkers split off. A fixture helper that globs ../testdata is called by
hundreds of decoder tests (measured: counting every enumerator put 337 of them in the
"small" cell), and those re-run only when a fixture moves, which the rest cell's cache
already handles.
```

## minCheckpointBytes

Moved from `cmd/gate/quick_select.go` (the comment above `minCheckpointBytes`) on 2026-10-09.

```text
A real checkpoint is a model file of at least minCheckpointBytes. The tiny fixtures (the largest
is a few MB) load beside anything; a real one is what the fit guard refuses when other test
binaries hold memory. Measured 2026-09-28, the cold `gate quick` on the MacBook:
examples/confidence's f32 load of the 0.5B needed 3.6 GB against 3.2 GB available while decoder
and metal were loading the same checkpoint, and failed; alone it passes.
```

## crossModuleReader

Moved from `cmd/gate/quick_select.go` (the comment above `crossModuleReader`) on 2026-10-09.

```text
crossModuleReader reports why the test cache cannot be trusted to see a change to rel on this
root's behalf, or "" when it can. cmd/go re-checks a test's recorded inputs only INSIDE the
module root ("Do not recheck files outside the module, GOPATH, or GOROOT root" — Go's
computeTestInputsID), so a metal test that read ../docs/x.md replays a stale result after x.md
changes. Where the read is determinable, the root is forced to re-run:

  - its source names the file (see needle);
  - a "../…" string literal, resolved against the package directory, IS the file or one of its
    directories (metal's "../docs/audit-metal-2026-09-12.md" forces on that file only, not on
    every docs/ edit);
  - it has a bare ".." literal and every component of the file's path as a literal
    (filepath.Join("..", "testdata", "llama-tiny") plus "config.json"). Requiring every
    component, not just the top directory, is what stops metal's one
    Join("..", "docs", "measurements", …, "tickets.jsonl") from forcing it on every docs/ edit.

A read by a path assembled at run time outside the module root — including a walk of the whole
parent from a computed root — stays the cache's blind spot, exactly as it is for a hand-typed
`go test`; that is the risk TE7(b) accepted for day loops. At 2026-09-28 no submodule test does
that: cuda enumerates only its own directory, metal only ../testdata (both caught above).
```

## minSplitRest

Moved from `cmd/gate/quick_select.go` (the comment above `minSplitRest`) on 2026-10-09.

```text
minSplitRest is how many non-walking tests a package needs before its walkers get a cell of
their own. The split exists for decoder (hundreds of tests, ~5 min, and three census tests that
read the whole tree); for a ten-test package the second process costs more than it saves.
```

## package gate doc

Moved from `cmd/gate/event.go` (the comment at the top of the package (`package main`)) on 2026-10-09.

```text
Package main implements `gate` — one runner over `go test -json` for the tallying
gates and censuses that used to be six separate shell/Python scripts (QUEUE E8).

THE INSIGHT E8 IS BUILT ON. Three shell gates (parity_sweep, gpu_gate, heavy_gate) and
three Python censuses (skip_census, sweep_composition, selector_coverage) are one program
wearing six hats: each runs `go test -json` over a matrix of package × family × quant ×
build-tag, tallies PASS/SKIP/FAIL with SKIPs bucketed by reason, and applies a decision.
They differ only in WHICH matrix and WHICH decision. So the matrix and the decision become
config; the tallying core is written once, here.

WHY GO IS STRICTLY BETTER HERE, not merely same-language:

  - The `-e`/tally tension disappears. The shell gates omit `set -e` deliberately — running
    N cells and tallying is the whole point, and `-e` aborts on the first failure and loses
    the count — which is a discipline you must remember not to "fix". Here, running every
    cell and keeping every exit code is the natural shape of the code, not a restraint.
  - PIPESTATUS capture vanishes: os/exec hands back each subprocess's code directly.
  - The silent-skip anti-pattern cannot recur. `command -v tool && tool` PASSES when the
    tool is absent; exec.LookPath returning not-found is an error you must handle. Same for
    asset detection: "assets absent → refuse a verdict" is a decision, and it lives in code
    that cannot fail open.
  - The tallying layer stops SCRAPING TEXT and starts consuming events. skip_census.py had
    already made this move ("a reader, not a parser"); the shell gates had not — heavy_gate
    counted `grep -cE '^--- PASS: '`, which is correct only as long as nothing else in the
    stream starts a line that way.

It orchestrates `go test`; it does not reimplement it. Stdlib only — per E7's constraint,
a consumer's module graph must not grow because a gate changed language.
```

## testKey

Moved from `cmd/gate/event.go` (the comment above `testKey`) on 2026-10-09.

```text
testKey identifies one test result.

ALL THREE FIELDS ARE LOAD-BEARING, and the third was added because a mutation test caught its
absence. Package is in the key because the same test NAME legitimately exists in several
packages. Cell is in the key because a matrix legitimately runs the SAME package AND test more
than once — parity_sweep and gpu_gate run one package under several tag/quant combinations,
which is the entire point of a matrix. Keying on (Pkg, Test) alone let a later cell overwrite an
earlier one, so N runs of a test reported as one: an UNDERCOUNT that still printed a confident
verdict, which is the failure shape this program exists to eliminate.
```

## results: liveRun

Moved from `cmd/gate/event.go` (the comment above `results.liveRun`) on 2026-10-09.

```text
liveRun holds the tests that have started and not yet finished, keyed by name, valued by
start time. "Last finished" alone cannot answer the question a stalled run actually raises:
during the v0.15.0 sweep the count sat at 430 for four minutes while the line kept naming a
test that had already completed, which says nothing about what is holding the cell up.
sync.Map for the same reason the two atomics above exist — the heartbeat goroutine reads
this while consume() writes it.
```

## noteOutput

Moved from `cmd/gate/event.go` (the comment above `noteOutput`) on 2026-10-09.

```text
noteOutput folds one output line into the stream-wide accumulators.

runLines counts EVERY `=== RUN` line, top-level and subtest alike. That is deliberate and it is
not the same unit as the skip count beside it: go test does NOT indent a subtest's `=== RUN` line
(only its `--- PASS` result line is indented), so the shell gate's `grep -cE '^=== RUN'` counted
subtests while its `grep -cE '^--- SKIP'` did not. The pair "ran 238 tests, skipped 22" therefore
mixes units — 238 tests-and-subtests started against 22 top-level skips. Reproduced exactly,
because E8 changes the substrate and not what a gate reports; flagged in
docs/completed/task-gate-runner.md §10 as a number that should probably say which unit it is in.
```

## lookupTop

Moved from `cmd/gate/event.go` (the comment above `lookupTop`) on 2026-10-09.

```text
lookupTop returns the LAST terminal action recorded for a TOP-LEVEL test of this exact name,
across every cell and package, and whether it was seen at all.

"Last" and "exact" both reproduce `grep -E "^--- (PASS|FAIL|SKIP): NAME \(" | tail -1`: the sweep
runs some gates twice (the plain cell and the realckpt cell), and the trailing `(` in that grep is
what stops TestFoo from matching TestFooBar. Not-seen is a FOURTH outcome, not a flavour of skip —
a required gate that never ran is the one case where the sweep learned nothing at all.
```

## isSubtest

Moved from `cmd/gate/event.go` (the comment above `isSubtest`) on 2026-10-09.

```text
isSubtest reports whether the key names a subtest (`TestFoo/case`) rather than a top-level test.

THIS DISTINCTION IS LOAD-BEARING AND THE TWO MIGRATED SCRIPTS DISAGREED ON IT. heavy_gate.sh
counted `^--- PASS:` anchored at column 0, so it tallied TOP-LEVEL tests only (go test indents
subtest result lines). skip_census.py keyed on (Package, Test) from the JSON, which counts
every subtest as its own result. Both are defensible; they are not the same number, and E8's
acceptance (a) requires each migrated gate to reproduce ITS OWN tally — so this is a per-config
choice (topLevelOnly), not a house style.
```

## gate mutation header

Moved from `cmd/gate/mutation.go` (the comment above `runMutation`) on 2026-10-09.

```text
A gate's own gate, committed rather than typed.

The policy requires that a gate land with a demonstration it can FAIL. Running that demonstration
as an ad-hoc one-liner produced two defects of its own in this repo, and BOTH reported a mutation
as verified while nothing had been exercised:

  - `command -v staticcheck >/dev/null && staticcheck …` — the binary was not on PATH, the &&
    short-circuited, the whole check evaluated to nothing, and it was reported as clean.
  - `python3 lint.py 2>&1 | head -3; echo "exit=$?"` — $? read head's status, not the lint's, so a
    red mutation printed exit=0.

A mutation check that silently reads the wrong status certifies a gate as falsifiable when nothing
ran: G-01 inside the mechanism built to prevent G-01.

THE SHELL VERSION DEFENDED AGAINST THAT BY DISCIPLINE — "the status path here contains NO PIPES
and no && chains" — which is a rule someone has to keep remembering. Here there is no status path
to get wrong: exec.Cmd.Run returns the command's own error, and a missing `sed` is an explicit
LookPath failure rather than a short-circuit that evaluates to success. The class is gone by
construction, which is the whole argument for the migration (E8 §2).
```

## runMutation: no sed -i

Moved from `cmd/gate/mutation.go` (the comment inside `runMutation`) on 2026-10-09.

```text
2. MUTATE, and assert the mutation actually CHANGED something. A sed expression that matches
   nothing leaves a green run that looks like a verified mutation check.
	NO `sed -i`, DELIBERATELY, AND THIS IS NOT A STYLE CHOICE. GNU sed takes an OPTIONAL suffix
	attached to the flag (`-i.bak`), so `sed -i EXPR file` edits in place; BSD sed (macOS) takes a
	REQUIRED separate suffix, so the same argv makes EXPR the backup suffix and `file` the
	expression — "sed: 1: \"subject.txt\": unterminated substitute pattern". The shell script this
	replaces carried that bug for its whole life and nobody saw it, because it is a hand-run
	operator tool that nobody ran on the Mac. Moving it into Go put it under CI's darwin job,
	which failed on it within one push.

	Reading sed's STDOUT and writing the file from Go is portable across both, and it puts the
	file write on the side of the line that owns state anyway.
```

## gate mutation header: sed-expr line

Moved from `cmd/gate/mutation.go` (the Usage text under the comment above `runMutation`) on 2026-10-09.

```text
<sed-expr>  is applied in place; it MUST change the file (asserted — a no-op mutation is the
            defect that makes a mutation check vacuous, and it happened: float32(v/sc) where
            both operands were already float32).
```

## gateConfig: PkgFailIsFailure

Moved from `cmd/gate/run.go` (the comment above `gateConfig.PkgFailIsFailure`) on 2026-10-09.

```text
PkgFailIsFailure counts a package-level fail (build error / native crash) toward the verdict.

FALSE FOR THE CENSUS ON PURPOSE, AND IT IS NOT A TYPO. skip_census.py computes `rc = 1 if
nfail else 0` and prints package-level fails without counting them — so a build error in one
package exits 0 today. E8 changes the SUBSTRATE, not what a gate decides (acceptance a), so
the runner reproduces that. The knob exists so flipping it later is one bool rather than an
archaeology exercise; see the warning the census report prints when it suppresses one.
```

## runCell: heartbeat

Moved from `cmd/gate/run.go` (the comment inside `runCell`) on 2026-10-09.

```text
HEARTBEAT while the cell runs (~/.claude/rules/long-tests.md). `go test -json` reports a
cell only when it finishes, and the realckpt cells run 55-90 minutes — so without this a
reader cannot tell a working gate from a hung one without ps'ing the box, which is exactly
what happened on 2026-08-26. Prints elapsed, tests finished so far, and the most recent
test name. There is no done-of-TOTAL because `go test` never announces a total; claiming
one would be inventing it.

stderr, not the report writer: the report is a verdict document and a progress line is not
part of it. Interval is env-tunable per the rule's "make it configurable rather than
removing it" — 0 or a bad value disables.
```

## gate identity header

Moved from `cmd/gate/identity.go` (the comment above `identityDumperTmpl`) on 2026-10-09.

```text
`gate identity <old-rev> <new-rev>` — inherit validation by identity (TE6(b),
docs/tasks/task-test-efficiency-2026-09.md).

For a change meant to be numerically neutral, the cheapest complete proof is byte-identical logits
against the last validated build on the family's parity prompt. This formalizes what L1 did by hand
(task-cpu-decode-peer-gap-2026-09.md, "L1 build: the Mac half"): build ONE small dumper at each
revision, dump full logits (prefill + N greedy steps, raw little-endian float32) for each family's
parity prompt(s), and compare bytes.

  - Two temporary `git worktree`s, detached, outside the repo; removed afterwards (also on SIGINT).
    The main working tree is never touched. The dumper is written into each worktree and built
    there with -trimpath, so both builds compile the same dumper against their own revision.
  - DETERMINISM FIRST: the NEW build runs every cell twice, in separate processes, and must be
    byte-identical to itself. A cell that is not is compared against the old build in tolerance
    mode (identityTolFactor × its own run-to-run max |diff|, argmax and tokens exact) and reported
    WITHIN TOLERANCE, never IDENTICAL. On the CPU that nondeterminism is TE6's kill criterion.
  - A SKIP IS NOT A PASS: a family whose asset is missing, whose load fails in both builds, or whose
    GPU cell fell back to the CPU is NOT RUN, listed separately with the reason.
  - BOTH SIDES SEE ONE ENVIRONMENT. Every GOINFER_* variable is removed from the dumpers' env. L1's
    manual check hit the trap this closes: an env diagnostic given to the OLD side only
    (GOINFER_INT4_F16_SCALES) did not reach the .giw reader, so the "old" build ran something other
    than what the comparison claimed, and the dumps differed from byte 1 for a reason that was not
    the change. Here the only thing that differs between the two runs is the code; the decode path
    each side reports is printed, and a path difference is flagged.
  - The parity manifest is NOT written: what an "identity-inherited" row would record is printed,
    and whether to make it a manifest method is the owner's decision.
```

## defaultIdentityQuants

Moved from `cmd/gate/identity_assets.go` (the comment above `defaultIdentityQuants`) on 2026-10-09.

```text
defaultIdentityQuants: on CPU the tiny fixtures run at f32 (what their goldens load) AND at the two
shipped quantized paths, because a numerically-neutral claim about a W8A8 or int4 kernel is not
exercised by an f32 run at all. A GPU backend declines f32 residency, so it runs the quantized
paths only: both on Metal (the whole tiny set takes ~40 s there), int4 alone on WebGPU, where most
tiny families run staged — one dispatch per matmul — and both quants took 13 min on the M1 Pro
(2026-09-28); -quant int4,int8int8 restores the second. Real checkpoints run int4, the served quant.
```

## selector census header

Moved from `cmd/gate/selector.go` (the comment above the regexps in `selector.go`) on 2026-10-09.

```text
Tests that EXIST versus tests any selector actually RUNS.

Three coverage gaps this campaign found were PLUMBING, not authorship — every one a test that
existed, passed when invoked, and was simply never selected:

	the three int8int8 goldens      — skipped on GOINFER_HEAVY_TESTS being unset
	gpt_oss's int8 golden           — behind //go:build realckpt, AND a missing checkpoint
	eleven GGUF quant-format gates  — outside the goldens selector's regexp

Each was found by someone asking a different question and noticing in passing. This asks it
directly: enumerate what exists, enumerate what the selectors reach, print the difference.

DESIGN, from what the other censuses learned:

  - DERIVE BOTH SIDES. The selectors come from refresh_parity_hashes.sh's GOLDEN_RE and from the
    sweep's own gate list, never restated here. (The Python read the gate list by regexping
    `GATES=(…)` out of parity_sweep.sh — which is why this had to migrate in the same commit as
    the sweep rather than "later as a config": deleting that file would have broken it outright.
    It now reads the same Go slice the sweep checks, so the second copy is gone rather than
    re-implemented.)
  - SEPARATE THE REASONS. never-selected, build-tag-excluded, env-gated and asset-blocked have
    different remedies and different costs; collapsing them into one "uncovered" number is what
    made the int8int8 rows look as expensive as authoring new fixtures when they cost one env var.
  - ERR TOWARD FLAGGING. The env detection matches any GOINFER_* read in the file, so it flags
    TestInt4_forwardParity for GOINFER_INT4_GOLDEN_UPDATE — which gates REGENERATION, not the
    test. That false positive is deliberate: a census that UNDER-reports is the failure mode that
    produced all three gaps above, and a reader dismisses a flagged line in seconds where a
    missing one costs weeks.
  - PRINT THE DIFFERENCE, NOT A VERDICT. A test can be selected and still vacuous, so a green
    here means "nothing became unreachable since a person last looked", not "coverage is
    adequate".
```

## composition header

Moved from `cmd/gate/composition.go` (the comment above the regexps in `composition.go`) on 2026-10-09.

```text
The release gate's coverage COMPOSITION along its axes — family × quant × loader.

THE RULE THIS IMPLEMENTS: a gate whose value depends on an axis must print its composition along
that axis. The sweep reported pass/fail per gate with nothing saying what the set SPANNED, and
that is the shape that let the forward goldens report "19 passed" through nine deps_hash refreshes
while every one of the 19 was f32 — an accurate count that could not distinguish "the axis is
covered" from "the axis collapsed to one value".

DERIVED, NOT DECLARED. Quant comes from grepping each gate's own test source for `Quant: "..."`,
and loader from the test name. Hand-maintained axis metadata beside the gate list would be a
second copy to drift, which is the defect this repo keeps finding. A gate whose test source
cannot be located is reported as UNKNOWN rather than defaulted to f32 — defaulting would inflate
the f32 count with gates nobody checked, which is the opposite of the point.

MIGRATED FROM scripts/sweep_composition.py (E8). It had to move in the same commit as the sweep
itself, not "later as a config": it PARSED the `GATES=(…)` array out of parity_sweep.sh with a
regexp, so deleting that shell script would have broken it outright. The gate list is now the Go
slice both sides read, which removes the parse rather than reimplementing it.
```

## runJobs: checkpoint loader alone

Moved from `cmd/gate/quick_run.go` (the comment inside `runJobs`) on 2026-10-09.

```text
A checkpoint loader runs alone. Measured 2026-09-28 on the MacBook: with only the
other LOADERS held back, examples/confidence's f32 load (3.6 GB) still met 3.4 GB
available beside decoder, metal and their compiles, and was refused; alone it had 5.8.
```

## configs: models dirs

Moved from `cmd/gate/configs.go` (the comment inside `heavyConfig`) on 2026-10-09.

```text
Both names, not just one (audit-2026-09-02.md N-41): GOINFER_MODELS_DIR is
realckpt-tagged tests' own root (decoder/modelsdir_test.go's modelPath, G-06),
GOINFER_MODELS is the shared asset registry's (decoder/assets.go's modelsRoot(),
what assetPath/GOINFER_MELLUM_CKPT-style tests resolve through) — the SAME cell
runs both kinds of heavy test, so GOINFER_GATE_MODELS pointing elsewhere used to
reach only the first kind, silently splitting one run across two roots.
```

## ledger header: port

Moved from `cmd/gate/ledger.go` (the comment above `ledgerDoc`) on 2026-10-09.

```text
The gate ledger (B14): the record of gate results a PERSON has confirmed. Ported from
scripts/gate_ledger.py (2026-09-25) so the parity sweep no longer shells out to Python; the file
format, the source key and every verdict are byte-for-byte what the script produced — existing
confirmations stay valid (TestLedger_matchesThePythonImplementation pins it).
```

## Provenance wording replaced in cmd/gate/configs.go

Moved from `cmd/gate/configs.go` (comments that named retired scripts, dates or audit ids; the new comments state the same contracts) on 2026-10-09.

```text
The committed matrix configs. One per migrated script — this is where "six scripts are one
program" becomes literal: each former script is a value, not a file.

censusConfig is skip_census.py: PASS/SKIP/FAIL over the whole tree with every SKIP bucketed by
why. passthrough replaces the default cell entirely (`gate census -- -tags cuda ./cuda/`).

Subtests count. skip_census.py keyed on (Package, Test) straight out of the JSON, which
includes them; heavy_gate did not. See gateConfig.TopLevelOnly.
```

## Provenance wording replaced in cmd/gate/run.go

Moved from `cmd/gate/run.go` (comments that named retired scripts, dates or audit ids; the new comments state the same contracts) on 2026-10-09.

```text
TopLevelOnly counts top-level tests only, excluding subtests. heavy_gate.sh did this by
anchoring its grep at column 0; skip_census.py did not. See isSubtest.

  "no-pass"  — zero PASSES is red even if tests ran and skipped (heavy_gate)
  "no-tests" — zero test EVENTS at all is red (skip_census: an empty stream is the
               absence of a census, not a clean one)

RCIsFailure counts a non-zero `go test` exit with zero --- FAIL lines as a failure. This is
heavy_gate's hard-won rc-awareness: a panic in a goroutine, a fatal error, a timeout or a
zero-match all abort the binary WITHOUT a per-test FAIL line, and counting only FAIL lines
reports GREEN on a crashed package.

Precondition refuses a verdict rather than reporting one. heavy_gate exits 2 when the models
dir is missing: with no assets, both "green" and "red" would be lies.

runCell executes one cell and folds its events into res. It returns a cellResult and NEVER a
fatal error: a cell that fails to start is a red cell, not an abandoned matrix. That property —
run every cell, keep every count — is the one `set -e` would have broken, and here it is the
shape of the code rather than a comment asking you not to add `-e`.

`go` itself must be present. exec.LookPath failing is an ERROR here — the shell idiom
`command -v go && go test` would have passed silently, which is the exact fail-open this
migration exists to make impossible.
```

## Provenance wording replaced in cmd/gate/skips.go

Moved from `cmd/gate/skips.go` (comments that named retired scripts, dates or audit ids; the new comments state the same contracts) on 2026-10-09.

```text
Skip bucketing — ported verbatim in BEHAVIOUR from scripts/skip_census.py, whose rule order is
itself load-bearing (first match wins: fixture before device before heavy). The buckets come
from docs/parity-coverage-policy.md, "A gate must be able to run, and able to fail":
```

## Provenance wording replaced in cmd/gate/parity.go

Moved from `cmd/gate/parity.go` (comments that named retired scripts, dates or audit ids; the new comments state the same contracts) on 2026-10-09.

```text
2026-10-08: Qwen3-ASR's decoder from the checkpoint's own layout, against transformers

2026-10-09: Voxtral's Llama decoder from the checkpoint's own layout (language_model.*, untied head, head_dim != hidden/heads), against transformers

2026-10-08: Qwen3-VL-2B text forward against HF f32 (the image prompt is TestQwen3VLImageReal, which needs its own pin output)
```

## Provenance wording replaced in cmd/gate/gpu.go

Moved from `cmd/gate/gpu.go` (comments that named retired scripts, dates or audit ids; the new comments state the same contracts) on 2026-10-09.

```text
Build the CUDA SUBMODULE entrypoint. The root ./cmd/serve has been a DELIBERATE compile error
under -tags cuda since v0.10.0 (the root command builds no backend, and failing loudly beats
silently producing a CPU-only binary named as though it had CUDA). This check pointed at the
root command for that entire period, so it could not pass — see audit G-01.
```

## Provenance wording replaced in cmd/gate/event.go

Moved from `cmd/gate/event.go` (comments that named retired scripts, dates or audit ids; the new comments state the same contracts) on 2026-10-09.

```text
stream, when set, is called for each terminal test result as it arrives. Liveness is
load-bearing for a 28-minute group: buffering means a running tier and a HUNG one produce
byte-identical output (none), so progress has to be visible as it happens.

runLines counts top-level `=== RUN` lines. heavy_gate reported this next to its tally so
that "0 passed" could be told apart from "0 attempted", which are different bugs.
```

## Provenance wording replaced in cmd/gate/identity_record.go

Moved from `cmd/gate/identity_record.go` (comments that named retired scripts, dates or audit ids; the new comments state the same contracts) on 2026-10-09.

```text
`gate identity -record FILE` (TE6(b), docs/tasks/task-test-efficiency-2026-09.md; owner decision 2026-09-28: a
family's validation may be inherited by identity). It writes PARITY_ROW lines for the decoder package's
TestParityManifest_merge, the manifest's one writer, and never writes the manifest itself.
```
