package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// The pre-tag GPU correctness gate. Run it on EACH GPU box, paste the verdict, then tag.
//
// WHY THIS IS THE GATE. CI never runs the GPU backends: ci.yml only BUILDS and VETS under -tags
// cuda, because the runners have no GPU. Metal is darwin-only and CUDA is Linux+NVIDIA, so no single
// machine — and no CI job — can cover both. Every GPU correctness claim this project makes therefore
// rests on a human running tests by hand on two boxes. This makes that reproducible: one command,
// one verdict, provenance attached.
//
// It is deliberately NOT "run everything". A gate that takes two hours gets skipped, and a skipped
// gate is worse than no gate because it still implies assurance. It runs the checks that map to bugs
// we actually shipped.
//
// HONESTY RULES, all learned the hard way:
//
//   - A skip is NOT a pass. `go test` prints "ok" for a package whose tests all skipped, so real
//     runs are counted and SKIPPED is reported separately. Green here must mean tested.
//   - Stray GPU processes invalidate results. A clean-tree control once "confirmed" a pre-existing
//     failure while 3.4 GB of leaked serve processes held the card — both sides equally poisoned.
//   - The CUDA suite must run SEQUENTIALLY (-p 1). Its tests each build a context and several load
//     real models; parallel packages contend for VRAM and the failures come back as bogus numerics
//     ("cosine 0.000000") rather than "you are out of memory".
//   - GROUP ACCOUNTING (audit G-01). A tally computed from what EMITTED can never detect what did
//     not: a block that dies mid-way emits nothing, and "ran 3" and "ran 4" are both plausible
//     numbers. The expected groups are DECLARED up front and reconciled at the end — a group that
//     emits no verdict, or an unexpected group id, is itself a FAIL.

// gpuGate carries the tally, the declared/emitted group sets and the notes block.
type gpuGate struct {
	w       io.Writer
	backend string
	models  string
	logDir  string
	commit  string
	dirty   bool

	expect  []string
	emitted map[string]bool
	cur     string

	pass, fail, skipped, ran int
	vacuousCells             []string // filtered cells whose tests ALL skipped — a pass that ran nothing

	// emptyCells are filtered cells whose -run matched no test at all. Tracked separately from
	// `fail` because nothing failed: the cell simply did not exist, which the aggregate ran==0
	// check cannot see once any other cell has run.
	emptyCells []string
	notes      []string
}

func (g *gpuGate) grp(name string) { g.cur = name }
func (g *gpuGate) mark() {
	if g.cur != "" {
		g.emitted[g.cur] = true
	}
}
func (g *gpuGate) hdr(s string)  { fmt.Fprintf(g.w, "\n%s== %s ==%s\n", bold, s, off) }
func (g *gpuGate) note(s string) { g.notes = append(g.notes, s) }

func (g *gpuGate) ok(format string, a ...any) {
	fmt.Fprintf(g.w, "  %sPASS%s  %s\n", green, off, fmt.Sprintf(format, a...))
	g.pass++
	g.mark()
}

func (g *gpuGate) bad(format string, a ...any) {
	fmt.Fprintf(g.w, "  %sFAIL%s  %s\n", red, off, fmt.Sprintf(format, a...))
	g.fail++
	g.mark()
}

func (g *gpuGate) skip(format string, a ...any) {
	s := fmt.Sprintf(format, a...)
	fmt.Fprintf(g.w, "  %sSKIP%s  %s\n", amber, off, s)
	g.skipped++
	g.note("SKIPPED: " + s)
	g.mark()
}

// detail prints the matching assertion lines, or the RAW TAIL if nothing matched.
//
// A group that fails and explains nothing is the silence-reads-as-health shape this gate exists to
// prevent, one level up in the tooling. `go test` killed by a signal, an OOM or a timeout emits
// neither "--- FAIL" nor a "file.go:N:" line — only "FAIL <pkg> <secs>" — so every filter reported
// an EMPTY explanation for exactly the failures that are hardest to reproduce.
func (g *gpuGate) detail(out string, re *regexp.Regexp) {
	// A process death is reported FIRST and on its own terms. It has no
	// `--- FAIL` to match, and its stack head names the test that died — which a
	// tail-based excerpt would bury under a register dump.
	if ex := crashExcerpt(out); len(ex) > 0 {
		fmt.Fprintf(g.w, "      PROCESS DIED — no test-level failure was reported. Crash head:\n")
		for _, ln := range ex {
			fmt.Fprintf(g.w, "      | %s\n", strings.TrimRight(ln, "\r"))
		}
		return
	}
	var hits []string
	if re.String() == failLineRe.String() {
		hits = failureLines(out)
	} else {
		for _, ln := range strings.Split(out, "\n") {
			if re.MatchString(ln) {
				hits = append(hits, ln)
			}
		}
	}
	if len(hits) > 0 {
		for i, h := range hits {
			if i >= 12 {
				break
			}
			fmt.Fprintf(g.w, "      %s\n", strings.TrimRight(h, "\r"))
		}
		return
	}
	fmt.Fprintf(g.w, "      NO ASSERTION LINE MATCHED — this run failed without a test-level failure.\n")
	fmt.Fprintf(g.w, "      That is a signal, an OOM kill, or a timeout. Raw tail (last 15 lines):\n")
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if len(lines) > 15 {
		lines = lines[len(lines)-15:]
	}
	for _, ln := range lines {
		fmt.Fprintf(g.w, "      | %s\n", ln)
	}
}

// vramNote fires on a cosine of EXACTLY zero.
//
// A cosine of 0.000000 is not a parity result — an all-zero buffer is what a failed allocation
// leaves behind, and this gate's own history records an OOM wearing a parity bug's clothes long
// enough that two people concluded "the tests just interfere" and moved on.
//
// WORDING IS DELIBERATE: it states the READING and points at the entry; it does NOT name a
// mechanism. "Suspect retention" was the obvious phrasing and is now DISPROVEN — A12 measured
// Close() returning all 4892 MiB synchronously in 123 ms with a 0 MiB asynchronous tail. Naming a
// mechanism a gate cannot see is how the last three explanations became someone's wasted afternoon.
func (g *gpuGate) vramNote(out string) {
	if !strings.Contains(out, "cosine 0.000000") {
		return
	}
	free := "unknown"
	if b, err := exec.Command("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader").Output(); err == nil {
		free = strings.TrimSpace(string(b))
	}
	fmt.Fprintf(g.w, "      NOTE: a cosine of EXACTLY 0.000000 is an all-zero buffer, which is what a failed\n")
	fmt.Fprintf(g.w, "            allocation leaves behind — not necessarily a numerics defect.\n")
	fmt.Fprintf(g.w, "            free VRAM on the card right now: %s\n", free)
	fmt.Fprintf(g.w, "            (measured AFTER the run, so it is a bound, not the value at failure)\n")
	fmt.Fprintf(g.w, "            See docs/QUEUE.md A12. Mechanism NOT established: parallelism (-p 1, one\n")
	fmt.Fprintf(g.w, "            package, no t.Parallel) and async teardown (Close is synchronous) are both\n")
	fmt.Fprintf(g.w, "            REFUTED, and there is no leak. Do not assume; measure.\n")
}

// gpuRun executes one `go test` check and returns its results plus the reconstructed -v text.
// It never aborts the gate: a check that cannot start is a red check, not an abandoned run.
func (g *gpuGate) run(c cell, stream bool) (*results, cellResult, string) {
	cfg := &gateConfig{Name: "gpu", TopLevelOnly: true, RCIsFailure: true}
	res := newResults()
	if stream {
		// One line per completed test, as it completes.
		res.stream = func(_, test, action string) {
			fmt.Fprintf(g.w, "        · %s: %s\n", strings.ToUpper(action), test)
		}
	}
	cr := runCell(c, cfg, res, g.logDir)
	// A FILTERED CELL THAT MATCHED NOTHING IS NOT A PASS, and the aggregate `g.ran == 0` check at
	// the end cannot see it: one empty cell among several full ones leaves g.ran > 0, so the cell
	// reports clean and its coverage is simply gone. Every -run pattern in this file is a literal
	// test-name prefix or alternation, so renaming a test silently empties its cell — the same
	// shape as the qwen3next oracle, where a -run pattern that could not match a required gate
	// produced "DID NOT RUN" for five weeks (docs/task-verification-surface-audit.md).
	g.noteIfEmpty(c, res)
	g.noteIfAllSkipped(c, cr)
	return res, cr, res.text()
}

// noteIfAllSkipped records a FILTERED cell whose tests all SKIPPED. noteIfEmpty
// cannot see this: a skip is a result, so `len(res.final) > 0` and it returns
// early — which is precisely how G-01 survived. Two Metal groups printed PASS,
// with their specific claim text, across at least two archived release logs while
// executing zero tests, because every test in them skipped for want of
// GOINFER_HEAVY_TESTS and `go test` exits 0 on an all-skip package.
//
// This is the gate's own rule applied to the gate: A SKIP IS NOT A PASS. It is a
// note rather than a hard failure because a cell CAN legitimately be all-skip on
// a box without the assets — but it must say so loudly, since the alternative is
// a green that vouches for nothing.
func (g *gpuGate) noteIfAllSkipped(c cell, cr cellResult) {
	if c.Run == "" || cr.Pass != 0 || cr.Fail != 0 || cr.Skip == 0 {
		return
	}
	fmt.Fprintf(g.w, "\n  !! CELL RAN NOTHING — ALL %d TEST(S) SKIPPED: %s  -run %q\n"+
		"     `go test` exits 0 on an all-skip package, so this cell would otherwise read as a\n"+
		"     PASS that vouches for nothing. A skip is not a pass. Usually a missing asset or an\n"+
		"     opt-in env var the cell forgot to set for itself.\n", cr.Skip, c.Name, c.Run)
	g.note(fmt.Sprintf("%s: all %d test(s) skipped — cell proves nothing", c.Name, cr.Skip))
	g.vacuousCells = append(g.vacuousCells, fmt.Sprintf("%s (-run %q, %d skipped)", c.Name, c.Run, cr.Skip))
}

// noteIfEmpty records a FILTERED cell that matched no test at all. Split out from run so it can be
// tested without spawning `go test`: pointing a cell at "./" from inside cmd/gate re-runs this very
// suite, which re-runs the cell, and the first draft of that test sat for ten minutes.
//
// An unfiltered cell is exempt — an empty -run means "everything", so emptiness there is a
// different bug (no packages, or a build failure) that runCell's own policies already cover.
func (g *gpuGate) noteIfEmpty(c cell, res *results) {
	if c.Run == "" || len(res.final) > 0 {
		return
	}
	fmt.Fprintf(g.w, "\n  !! CELL MATCHED NO TESTS: %s  -run %q\n"+
		"     Zero tests ran, so this cell proves nothing. Usually a test was renamed out from\n"+
		"     under the pattern. Not a pass.\n", c.Name, c.Run)
	g.emptyCells = append(g.emptyCells, fmt.Sprintf("%s (-run %q)", c.Name, c.Run))
}

// skipCensus lists the skips INSIDE a passing run, by name. "ok" hides them, and a skip is not a
// pass — a tier whose value is "it runs the real models" is worth nothing if the real-model tests
// inside it skipped.
func (g *gpuGate) skipCensus(res *results, indent string) int {
	sk := res.topLevel("skip")
	for _, k := range sk {
		fmt.Fprintf(g.w, "%s· %s\n", indent, k.Test)
	}
	return len(sk)
}

// evidence prints the lines a passing group offered as its own proof — the `ok <pkg> <secs>` line,
// a lifecycle trajectory, a prefill's argmax comparison. The shell greps these out after a PASS and
// they are part of the scope line, not decoration: "PASS metal suite" says a verdict, while
// `ok github.com/…/metal 31.2s` says what actually ran and for how long. Acceptance (d) is that
// whatever the script printed about what it validated, the runner prints too.
func (g *gpuGate) evidence(out string, re *regexp.Regexp) {
	for _, ln := range strings.Split(out, "\n") {
		if re.MatchString(ln) {
			fmt.Fprintf(g.w, "      %s\n", strings.TrimRight(ln, "\r"))
		}
	}
}

var okLineRe = regexp.MustCompile(`^ok\s`)

// failLineRe matches a test-level failure header. It deliberately does NOT match
// bare `file.go:N:` lines: every passing t.Log emits one, and when this pattern
// included them a crashed run reported a dozen `cosine=1.0000000 — PARITY` lines
// as its "failure detail" while the actual SIGSEGV went unshown — the breadth
// defeated detail()'s own crash fallback, which only fires when nothing matches.
// Assertion lines are still shown, but only the ones BELOW a failing test (see
// failureLines).
var failLineRe = regexp.MustCompile(`^(---|    ---) FAIL`)

// assertionLineRe is a test's own `file.go:N: …` output. Shown only while inside
// a failing test.
var assertionLineRe = regexp.MustCompile(`^\s*[\w./-]+\.go:[0-9]+:`)

// testBoundaryRe ends a failing test's block: anything that starts a new test or
// closes the package.
var testBoundaryRe = regexp.MustCompile(`^(=== RUN|=== CONT|=== PAUSE|(---|    ---) (PASS|SKIP)|ok\s|PASS$|FAIL\s)`)

// crashHeadRe is a process death: a signal, a panic, or a runtime fatal. These
// carry no `--- FAIL`, which is exactly why detail() must recognize them.
var crashHeadRe = regexp.MustCompile(`^(SIGSEGV|SIGBUS|SIGABRT|SIGILL|panic:|fatal error:)|signal arrived during`)

// ourFrameRe matches a stack frame in code this repo owns or vendors — the frames
// that identify WHICH test or function died, as opposed to the runtime and FFI
// scaffolding above them.
var ourFrameRe = regexp.MustCompile(`townsendmerino/(goinfer|aikit)|_test\.go:[0-9]+`)

// registerDumpRe is the tail of a Go crash report — `r14  0x8000…`, `pc 0x…`.
// Machine state, useless for identifying WHICH test died, and long enough to push
// the part that matters off the end of a fixed-size excerpt.
var registerDumpRe = regexp.MustCompile(`^(r[0-9]+|x[0-9]+|sp|pc|lr|fp|fault)\s+0x`)

// failureLines picks the lines that explain a failure: every test-level FAIL
// header, the assertion lines belonging to it, and nothing from tests that
// passed.
func failureLines(out string) []string {
	var hits []string
	inFail := false
	for _, ln := range strings.Split(out, "\n") {
		switch {
		case failLineRe.MatchString(ln):
			inFail = true
			hits = append(hits, ln)
		case testBoundaryRe.MatchString(ln):
			inFail = false
		case inFail && assertionLineRe.MatchString(ln):
			hits = append(hits, ln)
		}
	}
	return hits
}

// crashExcerpt returns the head of a process-death report: the signal line and
// the frames just below it, which name the test that died. It stops at the
// register dump — for the Metal `fault 0x10` tail the last 15 lines are
// registers, so a tail-based excerpt shows machine state and hides the answer.
func crashExcerpt(out string) []string {
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	start := -1
	for i, ln := range lines {
		if crashHeadRe.MatchString(ln) {
			start = i
			break
		}
	}
	if start < 0 {
		return nil
	}
	// The head (signal, address, goroutine) plus — crucially — the first frames
	// belonging to OUR code. A fixed prefix does not reach them: the real Metal
	// crash puts ~8 runtime/purego/reflect frames above the goinfer frame, so a
	// 24-line cap truncated exactly before the test name, which is the one fact
	// worth printing.
	var head, ours []string
	for _, ln := range lines[start:] {
		if registerDumpRe.MatchString(ln) {
			break
		}
		if len(head) < 6 {
			head = append(head, ln)
			continue
		}
		if ourFrameRe.MatchString(ln) && len(ours) < 8 {
			ours = append(ours, strings.TrimRight(ln, "\r"))
		}
	}
	if len(ours) == 0 {
		return head
	}
	return append(append(head, "      ...(frames from this repo)..."), ours...)
}

var failTestRe = regexp.MustCompile(`^(---|    ---) FAIL|^panic:`)

// detectBackend: nvidia-smi ⇒ cuda, darwin ⇒ metal, else none.
func detectBackend() string {
	if b := os.Getenv("GOINFER_GATE_BACKEND"); b != "" {
		return b
	}
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		return "cuda"
	}
	if out, err := exec.Command("uname", "-s").Output(); err == nil && strings.TrimSpace(string(out)) == "Darwin" {
		return "metal"
	}
	return "none"
}

func runGPU(w io.Writer, logDir string) int {
	g := &gpuGate{w: w, logDir: logDir, emitted: map[string]bool{}}
	g.backend = detectBackend()
	g.models = env("GOINFER_GATE_MODELS", filepath.Join(home(), "models"))

	// THE GROUPS DEFINE THEIR OWN ENVIRONMENT. Each group sets what it needs; nothing unsets them
	// for the groups that must NOT have them, so an operator who exports one "to be helpful"
	// silently changes what the other groups MEAN. Three consecutive red runs came from invoking
	// the gate as `GOINFER_HEAVY_TESTS=1 bash scripts/gpu_gate.sh`, which pulled the real-model
	// tests into the parity group as well: it then ran 608s against go's DEFAULT 600s timeout and
	// failed with no assertion line at all. Neutralised, and REPORTED rather than silently ignored —
	// an operator who set one deliberately must see that it did not take effect.
	for _, v := range []string{"GOINFER_HEAVY_TESTS", "GOINFER_DRAIN_GROUP"} {
		if val := os.Getenv(v); val != "" {
			fmt.Fprintf(w, "note: %s=%s was set in the calling environment — UNSET.\n", v, val)
			fmt.Fprintf(w, "      Every group sets what it needs itself; ambient values change what a group means.\n")
			g.note(v + " was set in the caller's environment and was neutralised — groups set their own")
			os.Unsetenv(v)
		}
	}

	switch g.backend {
	case "cuda":
		g.expect = []string{"cleangpu", "seam", "suite", "parity", "heavy", "graphsforced", "cgofree", "ptx", "repo"}
	case "metal":
		g.expect = []string{"cleangpu", "seam", "suite", "parity", "cgofree", "lifecycle", "prefill", "repo"}
	default:
		g.expect = []string{"cleangpu", "seam", "suite", "repo"}
	}

	prov := gatherProvenance(nil)
	g.commit, g.dirty = prov.Commit, prov.Dirty
	g.hdr("provenance")
	d := ""
	if g.dirty {
		d = " +dirty"
	}
	fmt.Fprintf(w, "  repo        %s%s\n", g.commit, d)
	fmt.Fprintf(w, "  date (UTC)  %s\n", prov.Date)
	fmt.Fprintf(w, "  host        %s\n", prov.Host)
	fmt.Fprintf(w, "  backend     %s\n", g.backend)
	fmt.Fprintf(w, "  models      %s\n", g.models)
	switch g.backend {
	case "cuda":
		if b, err := exec.Command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader").Output(); err == nil {
			fmt.Fprintf(w, "  gpu         %s\n", strings.SplitN(strings.TrimSpace(string(b)), "\n", 2)[0])
		}
	case "metal":
		cpu := "Apple Silicon"
		if b, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output(); err == nil {
			cpu = strings.TrimSpace(string(b))
		}
		fmt.Fprintf(w, "  gpu         %s\n", cpu)
	}
	if g.dirty {
		g.note("WORKING TREE DIRTY — this verdict does not describe a committed state.")
	}

	g.cleanGPU()
	g.seam()
	switch g.backend {
	case "cuda":
		g.cudaSuite()
		g.cudaParity()
		g.cudaHeavy()
		g.cudaGraphsForced()
		g.cudaCgoFree()
		g.cudaPTX()
	case "metal":
		g.metalSuite()
		g.metalParity() // G-02: the tagged tier the suite above cannot see
		g.metalCgoFree()
		g.metalLifecycle()
		g.metalPrefill()
	default:
		g.grp("suite")
		g.hdr("2-4. backend suites")
		g.skip("no GPU backend detected on this host — only the seam gate ran")
	}
	g.repoHygiene()
	return g.verdict()
}

// ---- 0. the card must be quiet, or every memory-sensitive result below is noise ----
func (g *gpuGate) cleanGPU() {
	g.grp("cleangpu")
	g.hdr("0. clean GPU")
	if g.backend != "cuda" {
		g.skip("clean-GPU check (no nvidia-smi; on Metal check Activity Monitor by hand)")
		return
	}
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		g.skip("clean-GPU check (no nvidia-smi; on Metal check Activity Monitor by hand)")
		return
	}
	procs := ""
	if b, err := exec.Command("nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader").Output(); err == nil {
		procs = strings.TrimSpace(string(b))
	}
	used := "?"
	if b, err := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader").Output(); err == nil {
		used = strings.TrimSpace(strings.ReplaceAll(string(b), "MiB", ""))
	}
	if procs == "" {
		g.ok("no compute processes on the GPU (%s MiB baseline)", used)
		return
	}
	fmt.Fprintf(g.w, "  processes holding the GPU:\n")
	for _, ln := range strings.Split(procs, "\n") {
		fmt.Fprintf(g.w, "    %s\n", ln)
	}
	// Only OUR leftovers are a problem to call out; a display server legitimately holds some.
	low := strings.ToLower(procs)
	if strings.Contains(low, "serve") || strings.Contains(low, "goinfer") || strings.Contains(low, "gi_serve") {
		g.bad("stray goinfer/serve processes hold the GPU — kill them and re-run; leaked processes have\n" +
			"        silently poisoned a control run before (both sides equally, which made a real bug look\n" +
			"        pre-existing). Try: pkill -f '[s]erve'")
		return
	}
	g.ok("no stray goinfer processes (%s MiB in use by others)", used)
}

// ---- 1. seam: no GPU needed, and it is the class that cost five weeks ----
func (g *gpuGate) seam() {
	g.grp("seam")
	g.hdr("1. seam (runs anywhere — no GPU, no model download)")
	res, cr, out := g.run(cell{Name: "seam", Pkgs: []string{"./decoder/"}, Run: "TestSeam_"}, false)
	if cr.RC != 0 {
		g.bad("seam gate — GPU serve may be silently CPU-only (see 7557723 / 727f198)")
		g.detail(out, failLineRe)
		return
	}
	// A `-run` pattern that matches NOTHING is a FAIL, not a zero-test pass: `go test -run
	// NoSuchTest` exits 0 and prints "ok", so renaming a test away silently deletes a check while
	// the gate stays green.
	if res.ranCount() == 0 {
		g.bad("seam gate: -run 'TestSeam_' matched NO tests — the pattern or the test names moved, and a\n" +
			"        zero-test run exits 0. This gate reported PASS for a check it never executed.")
		return
	}
	g.ran++
	g.ok("serve↔decoder↔backend seam: residency is actually reached, backend names validate")
}

// ---- 2a. CUDA kernel-level suite ----
//
// The header used to read "CUDA kernels + parity" while running NEITHER the resident parity gates
// NOR anything that asserts a forward. Every resident parity gate is behind `goinfer_testhooks`, so
// for the whole of v0.10.x/v0.11.0 this block ran 53 kernel-level tests and the release record said
// "full cuda suite" — while parity_manifest.json's shared_sets cover decoder/*.go ONLY, so deps_hash
// could not go stale on resident.go either. A change to CUDA forward numerics had no enforced signal
// anywhere in the gate. TWO groups now, because they answer different questions and one is not
// evidence for the other (audit G-01: the artifact must not be adjacent to what it is read as).
func (g *gpuGate) cudaSuite() {
	g.grp("suite")
	g.hdr("2a. CUDA kernel-level suite (no testhooks: kernels, admission, lint)")
	res, cr, out := g.run(cell{
		Name: "cuda-suite", Pkgs: []string{"./cuda/"}, Tags: []string{"cuda"},
		Serial: true, Extra: []string{"-short"}, Env: map[string]string{"CGO_ENABLED": "0"},
	}, false)
	if cr.RC != 0 {
		g.bad("cuda kernel-level suite")
		g.detail(out, failLineRe)
	} else {
		g.ran++
		g.ok("cuda kernel-level suite")
		g.evidence(out, okLineRe)
	}
	// Census the skips INSIDE the passing suite. "ok" hides them, and a skip is not a pass.
	sk := len(res.topLevel("skip"))
	if sk == 0 {
		// A zero here is far more likely to mean "the census broke" than "nothing skipped" — the
		// shell version needed -v for this to work at all, and got an empty census the first time.
		fmt.Fprintf(g.w, "      skip census: 0 — this suite is known to skip several; verify before believing it\n")
		return
	}
	fmt.Fprintf(g.w, "      skipped within it: %d (all GOINFER_HEAVY_TESTS=1 — bandwidth benchmarks,\n", sk)
	fmt.Fprintf(g.w, "      TestRealWeightGemvParity (real q4_K_M weights), TestResidentSpecServe (loads a 1.5B model))\n")
	g.skipCensus(res, "        ")
}

// ---- 2b. resident PARITY gates — the forward is asserted here ----
func (g *gpuGate) cudaParity() {
	g.grp("parity")
	g.hdr("2b. resident PARITY gates (-tags goinfer_testhooks — the forward is asserted here)")
	// -timeout DECLARED, not defaulted. Without it this group inherits go's 10m default, and a group
	// that overruns reports "FAIL <pkg> 608.417s" with no test named — indistinguishable from a
	// crash until detail() dumps the tail. State the budget so an overrun reads as an overrun.
	_, cr, out := g.run(cell{
		Name: "cuda-parity", Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},
		Serial: true, Timeout: "10m", Env: map[string]string{"CGO_ENABLED": "0"},
	}, false)
	if cr.RC != 0 {
		g.bad("resident parity gates — a CUDA forward moved. This is the group 2a cannot see.")
		g.detail(out, failLineRe)
		g.vramNote(out)
		return
	}
	g.ran++
	g.ok("resident parity gates (gemma4 dense/two-geom/MoE+router, GLM partial-rotary, mixtral MoE, sliding-window, rope-partial)")
	g.evidence(out, okLineRe)
}

// drainingTests derives the drain group FROM A MARKER rather than a list.
//
// The draining tests take the device to refusal, which is A13's only reproducible poisoning
// stimulus, so they run in their OWN process after the main tier and an exhausted device cannot
// reach anything else. `drainsDevice(t, why)` in cuda/drain_marker_test.go is the marker; this walks
// the test files tracking the enclosing `func TestX` and returns X for each one that calls it. A
// hand-kept -run list would be a constant restating a property — the same drift shape the census
// denominators keep making visible.
func drainingTests() []string {
	files, _ := filepath.Glob(filepath.Join("cuda", "*_test.go"))
	sort.Strings(files)
	set := map[string]bool{}
	fnRe := regexp.MustCompile(`^func (Test[A-Za-z0-9_]*)\(`)
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		name := ""
		for _, ln := range strings.Split(string(b), "\n") {
			if m := fnRe.FindStringSubmatch(ln); m != nil {
				name = m[1]
				continue
			}
			if strings.Contains(ln, "drainsDevice(t,") && name != "" {
				set[name] = true
				name = ""
			}
		}
	}
	return sortedSet(set)
}

// ---- 2c. heavy tier: the real-model group NOTHING has ever run ----
//
// This tier existed and was never executed by anything: no script set the variable, so the tests
// behind it were written, committed, and skipped forever. Declared here so it cannot quietly stop
// running again, and TIMED into the verdict so its cost is visible up front rather than discovered
// by someone waiting 28 minutes for a gate they thought took one.
func (g *gpuGate) cudaHeavy() {
	g.grp("heavy")
	g.hdr("2c. heavy tier (GOINFER_HEAVY_TESTS=1 — real models; ~28 min)")
	if os.Getenv("GOINFER_GATE_SKIP_HEAVY") != "" {
		g.skip("heavy tier (GOINFER_GATE_SKIP_HEAVY set) — the real-model gates did NOT run: 26B expert\n" +
			"        streaming, real-weight GEMV parity, resident spec-serve, the bandwidth benchmarks")
		return
	}
	t0 := time.Now()

	drain := drainingTests()
	if len(drain) == 0 {
		g.bad("heavy tier partition: the marker derivation found ZERO draining tests")
		g.note("drainsDevice() marker matched nothing — the derivation is broken, not the tree clean")
	}
	total := 0
	lst := exec.Command("go", "test", "-tags", "cuda goinfer_testhooks", "./cuda/", "-list", ".*")
	lst.Env = append(os.Environ(), "CGO_ENABLED=0")
	if b, err := lst.Output(); err == nil {
		for _, ln := range strings.Split(string(b), "\n") {
			if strings.HasPrefix(ln, "Test") {
				total++
			}
		}
	}
	fmt.Fprintf(g.w, "      partition (derived from drainsDevice() in cuda/drain_marker_test.go):\n")
	fmt.Fprintf(g.w, "        package total     %d test(s)   [go test -list '.*']\n", total)
	fmt.Fprintf(g.w, "        drain group        %d test(s)   %s\n", len(drain), strings.Join(drain, " "))
	fmt.Fprintf(g.w, "        main group         %d test(s)   [complement, by construction]\n", total-len(drain))

	fmt.Fprintf(g.w, "      streaming (one line per test):\n")
	// TIMEOUT 90m, RAISED FROM 60m ON 2026-09-01 BECAUSE 60m HAD NO MARGIN LEFT.
	// Two runs of this tier on the same box, the same day, at adjacent commits:
	//
	//	11:48 run   3320 s (55.3 min)   PASSED, by 4.7 min
	//	13:48 run   3612 s (60.2 min)   PROCESS DIED — "panic: test timed out after 1h0m0s"
	//
	// Nothing about the tier changed between them; it simply drifted across the
	// line. A timeout that close reports a HANG when what happened was a slow
	// afternoon, and it reports it as a dead process with no test-level failure —
	// the most expensive kind of red to diagnose, because the crash head names
	// whichever test was merely unlucky enough to be running (TestSplitKVCrossover,
	// 31 s in, entirely innocent).
	//
	// This is the same fragility the retired scripts/heavy_gate.sh hit and fixed by
	// going to 120m — measured there as "the decoder tier loads ~15+ big real
	// checkpoints sequentially and needs ~50-60 min". cmd/gate did not inherit that
	// lesson when it replaced the script. 90m is ~50% headroom over the observed
	// 60.2 min, which is enough for drift without letting a genuine hang sit for
	// two hours.
	mainRes, mainCR, mainOut := g.run(cell{
		Name: "cuda-heavy", Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},
		Serial: true, Timeout: "90m",
		Env: map[string]string{"CGO_ENABLED": "0", "GOINFER_HEAVY_TESTS": "1"},
	}, true)

	// INVOCATION 2: the drain group, its own process, after everything else. GOINFER_DRAIN_GROUP is
	// what un-skips the marker; -run restricts it to the derived set so the rest of the package is
	// not paid for twice.
	fmt.Fprintf(g.w, "      drain group, separate process:\n")
	drainRE := "^(" + strings.Join(drain, "|") + ")$"
	_, drainCR, drainOut := g.run(cell{
		Name: "cuda-drain", Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},
		Serial: true, Timeout: "20m", Run: drainRE,
		Env: map[string]string{"CGO_ENABLED": "0", "GOINFER_HEAVY_TESTS": "1", "GOINFER_DRAIN_GROUP": "1"},
	}, true)

	// RECONCILIATION: a partition that silently drops a test is the failure mode. Counted from the
	// marker's own tokens in BOTH directions — main tier: every marked test must have SKIPPED; drain
	// tier: every marked test must have RUN. A derivation miss shows up as a main-tier
	// DRAIN-GROUP-SKIP with no matching drain-tier run, and fails here rather than being quietly
	// absent from both halves.
	mainSkipped := strings.Count(mainOut, "DRAIN-GROUP-SKIP")
	drainRan := strings.Count(drainOut, "DRAIN-GROUP-RUN")
	fmt.Fprintf(g.w, "      reconciliation: marked=%d  skipped-in-main=%d  ran-in-drain=%d\n",
		len(drain), mainSkipped, drainRan)
	if mainSkipped != len(drain) || drainRan != len(drain) {
		g.bad("heavy tier partition does not reconcile — a test is in neither half or in both")
		g.note(fmt.Sprintf("partition mismatch: %d marked, %d skipped in main, %d ran in drain",
			len(drain), mainSkipped, drainRan))
	}

	if drainCR.RC == 0 {
		g.ran++
		g.ok("drain group (%d test(s), separate process)", len(drain))
	} else {
		g.bad("drain group (%d test(s), separate process)", len(drain))
		g.detail(drainOut, failTestRe)
		fmt.Fprintf(g.w, "      full output: %s\n", drainCR.LogPath)
	}

	secs := int(time.Since(t0).Seconds())
	if mainCR.RC == 0 {
		g.ran++
		g.ok("heavy tier (real models) — %ds", secs)
		g.evidence(mainOut, okLineRe)
	} else {
		g.bad("heavy tier (real models) — %ds", secs)
		// Name the failing TESTS first, then their assertion lines. Never a bare "file.go:N:" match,
		// which under -v is every log line in the run — the filter copied from the non-verbose
		// groups matched everything and `head` truncated the actual "--- FAIL" away.
		g.detail(mainOut, failTestRe)
		fmt.Fprintf(g.w, "      full output: %s\n", mainCR.LogPath)
	}
	// Census its skips BY NAME. A tier whose value is "it runs the real models" is worth nothing if
	// the real-model tests inside it skipped.
	hsk := len(mainRes.topLevel("skip"))
	fmt.Fprintf(g.w, "      ran %d tests, skipped %d\n", mainRes.runLines, hsk)
	if hsk > 0 {
		g.skipCensus(mainRes, "        ")
		g.note(fmt.Sprintf("heavy tier skipped %d test(s) — see the 2c census for which", hsk))
	}
}

// ---- 2d. CUDA graphs bit-exactness, FORCED ----
//
// SEPARATE AND LABELLED, deliberately. admitGraphs declines under DEFAULT compute mode without MPS,
// which is correct production behaviour and must stay that way — so on this box the graph
// capture/replay path is never exercised at all. Forcing it here tests the CODE without changing the
// admission POLICY. Keeping it out of 2b matters: a forced result must never be read as evidence
// that graphs are admitted in production (audit G-01).
func (g *gpuGate) cudaGraphsForced() {
	g.grp("graphsforced")
	g.hdr("2d. CUDA graphs bit-exactness, FORCED (GOINFER_CUDA_GRAPHS_UNSAFE=1)")
	res, cr, out := g.run(cell{
		Name: "cuda-graphs", Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},
		Serial: true, Run: "TestGemma4Graphs_",
		Env: map[string]string{"CGO_ENABLED": "0", "GOINFER_CUDA_GRAPHS_UNSAFE": "1"},
	}, false)
	if cr.RC != 0 {
		g.bad("graphs bit-exactness FAILED under forced capture — replay diverges from live launches")
		g.detail(out, failLineRe)
	} else if res.ranCount() == 0 {
		g.bad("graphs (forced): -run 'TestGemma4Graphs_' matched NO tests — zero-test runs exit 0")
	} else {
		g.ran++
		gsk := len(res.topLevel("skip"))
		g.ok("graphs replay == live launches, FORCED capture (%d tests, %d skipped)", res.ranCount(), gsk)
		if gsk > 0 {
			g.skipCensus(res, "        ")
		}
	}
	g.note("graphs are FORCED in 2d (GOINFER_CUDA_GRAPHS_UNSAFE): this proves the capture/replay CODE, " +
		"not that graphs are admitted in production — admitGraphs still declines here (DEFAULT compute mode, no MPS).")
}

// ---- 3. cgo-free: the whole premise — verify, never assume ----
func (g *gpuGate) cudaCgoFree() {
	g.grp("cgofree")
	g.hdr("3. cgo-free (the whole premise — verify, never assume)")
	// Build the CUDA SUBMODULE entrypoint. The root ./cmd/serve has been a DELIBERATE compile error
	// under -tags cuda since v0.10.0 (the root command builds no backend, and failing loudly beats
	// silently producing a CPU-only binary named as though it had CUDA). This check pointed at the
	// root command for that entire period, so it could not pass — see audit G-01.
	bin := filepath.Join(os.TempDir(), "gpu_gate_serve")
	build := exec.Command("go", "build", "-tags", "cuda", "-o", bin, "./cmd/serve")
	build.Dir = "cuda"
	build.Env = append(os.Environ(), "CGO_ENABLED=0")
	if out, err := build.CombinedOutput(); err != nil {
		g.bad("cuda/cmd/serve does not build under -tags cuda (CGO_ENABLED=0)")
		g.detail(string(out), failLineRe)
		return
	}
	defer os.Remove(bin)
	g.ran++
	linked := ""
	if b, err := exec.Command("ldd", bin).Output(); err == nil {
		for _, ln := range strings.Split(string(b), "\n") {
			l := strings.ToLower(ln)
			if strings.Contains(l, "libcuda") || strings.Contains(l, "libnvrtc") || strings.Contains(l, "libcudart") {
				linked += ln + "\n"
			}
		}
	}
	if linked != "" {
		g.bad("binary links CUDA libraries — the cgo-free claim is false:")
		for _, ln := range strings.Split(strings.TrimRight(linked, "\n"), "\n") {
			fmt.Fprintf(g.w, "      %s\n", ln)
		}
		return
	}
	g.ok("serve builds CGO_ENABLED=0 and links no CUDA toolkit (driver is dlopen'd at runtime)")
}

// ---- 4. PTX reproduces from source, each at the NVRTC it records ----
//
// INTEGRITY: this block must ALWAYS reach pass/fail/skip. An earlier shell revision died on a bash
// error midway and the gate still reported PASS — a check that can neither pass nor fail is the same
// defect as one that can only fail (audit G-01). In Go the block cannot exit early without
// returning, and the group reconciliation at the end catches it if it ever does.
//
// Every .ptx states the toolchain that produced it in its own header:
//
//	// Cuda compilation tools, release 12.6, V12.6.85
//
// That is the artifact's provenance, and it is what we rebuild against — NOT whatever NVRTC this box
// happens to default to. The tree legitimately carries a MIX (kernels added after a toolchain bump
// were built at the newer one, and the audited ones are deliberately pinned), so a single-toolchain
// rebuild reports a false FAIL on every file from the other era. That is what made this check
// unpassable for the whole of v0.10.x/v0.11.0.
//
// NOTHING IS EXEMPTED BY NAME. A file is only skipped when the NVRTC version IT RECORDS is not
// installed here, and the skip names the version so it is actionable.
var ptxVersionRe = regexp.MustCompile(`Cuda compilation tools, release [0-9.]*, V([0-9.]*)`)

func recordedNVRTC(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	if m := ptxVersionRe.FindStringSubmatch(string(b)); m != nil {
		return m[1]
	}
	return ""
}

// probeNVRTC maps version -> (libdir, includedir) by COMPILING a trivial kernel with each candidate
// and reading the version out of the PTX it emits. Exact, and it exercises the same path the real
// build uses — filename/soname heuristics only give major.minor, and the patch matters.
func probeNVRTC(w io.Writer) map[string][2]string {
	out := map[string][2]string{}
	var cands []string
	// GOINFER_NVRTC_DIRS is an OVERRIDE, not an addition: set it and ONLY those toolchains are used.
	// That makes the "toolchain absent" path reachable on a box that happens to have it, which is
	// how the counted-skip behaviour below is tested rather than assumed.
	if v := os.Getenv("GOINFER_NVRTC_DIRS"); v != "" {
		cands = strings.Split(v, ":")
	} else {
		for _, pat := range []string{
			filepath.Join(home(), "nvrtc-*", "lib", "python*", "site-packages", "nvidia"),
			filepath.Join(home(), ".venv*", "lib", "python*", "site-packages", "nvidia"),
		} {
			m, _ := filepath.Glob(pat)
			for _, c := range m {
				if st, err := os.Stat(filepath.Join(c, "cuda_nvrtc", "lib")); err == nil && st.IsDir() {
					cands = append(cands, c)
				}
			}
		}
	}
	probe, err := os.MkdirTemp("", "gate_ptx")
	if err != nil {
		return out
	}
	defer os.RemoveAll(probe)
	src := filepath.Join(probe, "p.cu")
	if err := os.WriteFile(src, []byte("extern \"C\" __global__ void p(float* o){ o[0]=1.f; }\n"), 0o644); err != nil {
		return out
	}
	dst := filepath.Join(probe, "p.ptx")
	arch := env("ARCH", "compute_75")
	for _, c := range cands {
		if c == "" {
			continue
		}
		lib, inc := filepath.Join(c, "cuda_nvrtc", "lib"), filepath.Join(c, "cuda_runtime", "include")
		if _, err := os.Stat(filepath.Join(lib, "libnvrtc.so.12")); err != nil {
			lib, inc = filepath.Join(c, "lib"), filepath.Join(c, "include")
		}
		if _, err := os.Stat(filepath.Join(lib, "libnvrtc.so.12")); err != nil {
			continue
		}
		cmd := exec.Command("python3", "nvrtc_compile.py", src, dst, arch, inc)
		cmd.Dir = "cuda"
		cmd.Env = append(os.Environ(),
			"LD_LIBRARY_PATH="+lib, "NVRTC_SO="+filepath.Join(lib, "libnvrtc.so.12"))
		if err := cmd.Run(); err != nil {
			continue
		}
		if v := recordedNVRTC(dst); v != "" {
			if _, seen := out[v]; !seen {
				out[v] = [2]string{lib, inc}
			}
		}
	}
	return out
}

func (g *gpuGate) cudaPTX() {
	g.grp("ptx")
	g.hdr("4. PTX reproduces from source, each at the NVRTC it records")
	if st, err := os.Stat(filepath.Join("cuda", "build_ptx.sh")); err != nil || st.Mode()&0o111 == 0 {
		g.skip("PTX reproducibility (cuda/build_ptx.sh missing)")
		return
	}
	nvrtc := probeNVRTC(g.w)
	avail := sortedSet(func() map[string]bool {
		m := map[string]bool{}
		for k := range nvrtc {
			m[k] = true
		}
		return m
	}())
	tcs := "none"
	if len(avail) > 0 {
		tcs = strings.Join(avail, " ")
	}
	fmt.Fprintf(g.w, "  toolchains available: %s\n", tcs)

	ptxFiles, _ := filepath.Glob(filepath.Join("cuda", "testdata", "*.ptx"))
	sort.Strings(ptxFiles)
	// Back the committed artifacts up: this check must NOT mutate the tree.
	backup, err := os.MkdirTemp("", "gate_ptx_before")
	if err != nil {
		g.skip("PTX reproducibility (cannot stage a backup: %v)", err)
		return
	}
	defer os.RemoveAll(backup)
	saved := map[string][]byte{}
	for _, f := range ptxFiles {
		if b, err := os.ReadFile(f); err == nil {
			saved[f] = b
		}
	}

	diff, okN, total, nUnavail := 0, 0, 0, 0
	var unavail []string
	for _, f := range ptxFiles {
		base := strings.TrimSuffix(filepath.Base(f), ".ptx")
		if _, err := os.Stat(filepath.Join("cuda", base+".cu")); err != nil {
			continue // no source ⇒ not ours to reproduce
		}
		total++
		want := recordedNVRTC(f)
		if want == "" {
			unavail = append(unavail, base+"(no recorded version)")
			nUnavail++
			continue
		}
		ent, ok := nvrtc[want]
		if !ok {
			unavail = append(unavail, base+"(needs V"+want+")")
			nUnavail++
			continue
		}
		cmd := exec.Command("./build_ptx.sh", base)
		cmd.Dir = "cuda"
		cmd.Env = append(os.Environ(), "NVRTC_LIB="+ent[0], "CUDA_INC="+ent[1])
		if err := cmd.Run(); err != nil {
			unavail = append(unavail, base+"(build failed at V"+want+")")
			nUnavail++
			continue
		}
		now, _ := os.ReadFile(f)
		if string(now) == string(saved[f]) {
			okN++
		} else {
			diff++
			fmt.Fprintf(g.w, "      DIFFERS: %s.ptx (rebuilt at its recorded V%s)\n", base, want)
		}
	}
	// Restore; this check must not mutate the tree.
	for f, b := range saved {
		_ = os.WriteFile(f, b, 0o644)
	}

	switch {
	case diff == 0 && okN > 0:
		g.ran++
		g.ok("%d/%d PTX regenerate byte-identically at their recorded NVRTC", okN, total)
	case diff > 0:
		g.ran++
		g.bad("%d PTX differ from their committed form — the shipped kernels do not match their .cu", diff)
	default:
		g.skip("PTX reproducibility (no usable NVRTC for any recorded version)")
	}
	// A partial verification must never read as a full one: name the count AND the files, and make
	// it a counted SKIP so it appears in the verdict's skipped tally and the notes block.
	// "21/21 verified" and "11/21 verified" must be visibly different outcomes.
	if len(unavail) > 0 {
		g.skip("%d/%d PTX NOT verified on this box (toolchain absent): %s", nUnavail, total, strings.Join(unavail, " "))
	}
}

// ---- Metal groups ----

func (g *gpuGate) metalSuite() {
	g.grp("suite")
	g.hdr("2. Metal suite")
	_, cr, out := g.run(cell{Name: "metal-suite", Pkgs: []string{"./metal/"}, Serial: true, Extra: []string{"-short"}}, false)
	if cr.RC != 0 {
		g.bad("metal suite")
		g.detail(out, failLineRe)
		return
	}
	g.ran++
	g.ok("full metal suite")
	g.evidence(out, okLineRe)
}

func (g *gpuGate) metalCgoFree() {
	g.grp("cgofree")
	g.hdr("3. cgo-free")
	bin := filepath.Join(os.TempDir(), "gpu_gate_serve")
	// G-03: build the METAL submodule entrypoint, not the root one. Without
	// `build.Dir` this compiled ./cmd/serve from the repo root -- a binary that
	// imports no Metal code at all, as cmd/serve/backendtag_guard_metal.go says in
	// so many words ("`-tags metal` does nothing on the root cmd/serve since
	// v0.10.0 (it builds no backend)"). It therefore built fine forever and the
	// group asserted "Metal is dlopen'd via purego-objc" about a binary with no
	// Metal in it. The CUDA half of this was fixed at d2c4858 (build.Dir = "cuda");
	// this half was not, so the gate has been passing on the wrong artifact.
	build := exec.Command("go", "build", "-o", bin, "./cmd/serve")
	build.Dir = "metal"
	build.Env = append(os.Environ(), "CGO_ENABLED=0")
	if err := build.Run(); err != nil {
		g.bad("metal/cmd/serve does not build CGO_ENABLED=0")
		return
	}
	// The otool analogue of CUDA's ldd check: purego-objc resolves the Metal
	// framework at RUNTIME, so a link-time reference to it would falsify the
	// cgo-free claim exactly as a libcuda link falsifies the CUDA one. Checking the
	// build succeeds proves only that it compiles.
	linked := ""
	if b, err := exec.Command("otool", "-L", bin).Output(); err == nil {
		for _, ln := range strings.Split(string(b), "\n") {
			l := strings.ToLower(ln)
			if strings.Contains(l, "metal.framework") || strings.Contains(l, "metalperformanceshaders") {
				linked += ln + "\n"
			}
		}
	}
	os.Remove(bin)
	if linked != "" {
		g.bad("binary links the Metal framework — the cgo-free claim is false:")
		for _, ln := range strings.Split(strings.TrimRight(linked, "\n"), "\n") {
			fmt.Fprintf(g.w, "        %s\n", strings.TrimSpace(ln))
		}
		return
	}
	g.ran++
	g.ok("metal/cmd/serve builds CGO_ENABLED=0 and links no Metal framework (dlopen'd via purego-objc)")
}

// ---- Metal resident PARITY gates — the forward is asserted here ----
//
// G-02: no Metal cell passed `-tags goinfer_testhooks`, so the ritual's "full
// metal suite" was the kernel tier plus the snapshot golden, and 59 files / 64
// test funcs -- every Metal resident-parity gate among them -- were never
// COMPILED, let alone run. RELEASING.md said the Metal run vouches for G10 and
// G11; neither was built by the command it named. This is the hole cudaParity
// describes closing for CUDA, mirrored.
//
// Filtered to the resident-parity gates rather than the whole tagged tree: the
// tag also selects long device tests that belong to other groups, and a cell
// that quietly runs everything is how a timeout becomes indistinguishable from a
// crash. -timeout is declared for the same reason cudaParity declares it.
func (g *gpuGate) metalParity() {
	g.grp("parity")
	g.hdr("2c. resident PARITY gates (-tags goinfer_testhooks — the forward is asserted here)")
	_, cr, out := g.run(cell{
		Name: "metal-parity", Pkgs: []string{"./metal/"}, Tags: []string{"goinfer_testhooks"},
		Run:     "ResidentParity|residentParity|_bitExact|matchesNonPaged|cpuParity",
		Serial:  true,
		Timeout: "20m",
		Env:     map[string]string{"GOINFER_HEAVY_TESTS": "1"},
	}, false)
	if cr.RC != 0 || cr.vacuous() {
		g.bad("resident parity gates — a Metal forward moved. This is the group the suite cannot see.")
		g.detail(out, failLineRe)
		return
	}
	g.ran++
	g.ok("resident parity gates (dense, gemma3, gemma4 MoE, qwen3.5, mellum, gpt-oss, paging bit-exactness)")
}

// metalModel is the one checkpoint the Metal lifecycle and prefill gates need. Keeping both on the
// same file keeps the gate's asset footprint at one download.
func (g *gpuGate) metalModel() string {
	return filepath.Join(home(), "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
}

// Metal HAD the same hole CUDA did — Close() froze a channel and freed nothing, leaking ~267 MB per
// Load+Close on a 0.5B (aacec89). These run WITHOUT -short (the suite above uses it) because they
// load real models, and they cover BOTH conditions: the sequential sawtooth, and a second model
// alive — the case that made CUDA's first fix look correct when it was not.
func (g *gpuGate) metalLifecycle() {
	g.grp("lifecycle")
	g.hdr("4. lifecycle")
	m := g.metalModel()
	if _, err := os.Stat(m); err != nil {
		g.skip("Close() lifecycle gate needs %s", m)
		return
	}
	_, cr, out := g.run(cell{
		Name: "metal-lifecycle", Pkgs: []string{"./metal/"},
		Run: "TestMetal_CloseFreesMemory|TestMetal_CloseWithSecondModelAlive",
		// G-01: all four tests in the two Metal cells call requireHeavyModel, and
		// the gate deliberately UNSETS GOINFER_HEAVY_TESTS above so no ambient
		// value can change what a group means. The cells therefore have to set it
		// themselves, exactly as cuda-heavy does -- without it every test skips,
		// `go test` exits 0, and the group printed PASS while running nothing.
		// Measured 2026-09-01: skip/skip in 0.4 s before, pass/pass in 27 s after.
		Env: map[string]string{"GOINFER_HEAVY_TESTS": "1"},
	}, false)
	if cr.RC != 0 || cr.vacuous() {
		g.bad("Close() leaks memory")
		g.detail(out, regexp.MustCompile(`^--- FAIL|LEAK|did NOT free|USE-AFTER-FREE|\.go:[0-9]+:`))
		return
	}
	g.ran++
	g.ok("Close() frees — sawtooth not staircase, and frees with a second model resident")
	g.evidence(out, regexp.MustCompile(`trajectory|A\+B alive`))
}

// The newest bug that maps to this doctrine. PrefillLast — the f16 simdgroup_matrix TTFT path —
// emitted NaN logits at EVERY prompt length (including the minimal single-tile M=8) after the LM
// head was pinned to int8: prefill still ran the int8 head weights through the int4 gemm_w4f16,
// misreading them as packed nibbles (19ef47d). It hit the DENSE control, a model Metal ships.
// Nothing exercised it against a real checkpoint until a hand-run, so it was invisible on push.
func (g *gpuGate) metalPrefill() {
	g.grp("prefill")
	g.hdr("4b. prefill (f16-MMA TTFT — a shipped path, and it shipped NaN)")
	m := g.metalModel()
	if _, err := os.Stat(m); err != nil {
		g.skip("prefill gate needs %s", m)
		return
	}
	_, cr, out := g.run(cell{
		Name: "metal-prefill", Pkgs: []string{"./metal/"},
		Run: "TestPrefillParity|TestPrefillNoNaN",
		// GOINFER_HEAVY_TESTS for the same reason as metal-lifecycle above (G-01).
		Env: map[string]string{"GOINFER_METAL_MODEL": m, "GOINFER_HEAVY_TESTS": "1"},
	}, false)
	if cr.RC != 0 {
		g.bad("prefill parity/NaN gate — the f16-MMA TTFT path is wrong on a shipped model")
		g.detail(out, regexp.MustCompile(`^--- FAIL|parity FAIL|contain NaN|\.go:[0-9]+:`))
		return
	}
	g.ran++
	g.ok("prefill matches sequential decode and emits finite logits (no NaN)")
	g.evidence(out, regexp.MustCompile(`argmax matches|faster TTFT`))
}

// ---- 5. repo hygiene: run what CI runs, DERIVED rather than duplicated (B0) ----
//
// This block used to run `gofmt -l .` and `go vet ./decoder/ ./cmd/...` — a hand-written list that
// was a strict SUBSET of CI's: no staticcheck at all, vet without the goinfer_testhooks tag and over
// narrower packages, no build. So CI went red on `staticcheck -tags cuda` and stayed red for three
// commits, and running this gate — the thing you run INSTEAD of remembering — would not have caught
// it either. Adding staticcheck would fix the instance and leave the class open: the next check CI
// gains reopens the gap. So the list is DERIVED from .github/workflows/ci.yml by ci_checks.py, and a
// check CI adds appears here with no edit to this file.
func (g *gpuGate) repoHygiene() {
	g.grp("repo")
	g.hdr("5. repo hygiene (derived from .github/workflows/ci.yml)")

	// The queue's citations, commit AND path:line. A state document is cited without being
	// re-derived, so a wrong reference in it propagates with more confidence than the same error in
	// conversation — 9e5f8fa was cited several times, from the file, without anyone opening it, and
	// cuda/resident.go:244 kept an audit critical listed as open for weeks after it was fixed.
	lint := exec.Command("python3", "scripts/queue_citation_lint.py")
	lintOut, lintErr := lint.CombinedOutput()
	if lintErr == nil {
		lines := strings.Split(strings.TrimRight(string(lintOut), "\n"), "\n")
		g.ok("%s", lines[len(lines)-1])
	} else {
		g.bad("docs/QUEUE.md SHA citations")
		for i, ln := range strings.Split(string(lintOut), "\n") {
			if i >= 6 {
				break
			}
			fmt.Fprintf(g.w, "      %s\n", ln)
		}
	}

	rows, err := exec.Command("python3", "scripts/ci_checks.py").Output()
	if err != nil || len(strings.TrimSpace(string(rows))) == 0 {
		// A derivation that fails must FAIL, not silently degrade to the old hand-written list. That
		// would be the very substitution this block exists to prevent, and it would look like a pass.
		msg := ""
		if ee, ok := err.(*exec.ExitError); ok {
			msg = strings.ReplaceAll(strings.TrimSpace(string(ee.Stderr)), "\n", " ")
		} else if err != nil {
			msg = err.Error()
		}
		g.bad("cannot derive CI's check set: %s", msg)
		return
	}

	// This host runs the linux jobs; the *-darwin ones are a COUNTED SKIP naming why, never dropped.
	// A check that is skipped and a check that passed must not look the same (B0a).
	mine := regexp.MustCompile(`^(root|gpu|cuda)$`)
	other := "darwin"
	if b, e := exec.Command("uname", "-s").Output(); e == nil && strings.TrimSpace(string(b)) == "Darwin" {
		mine = regexp.MustCompile(`-darwin$`)
		other = "linux"
	}
	ciOK, ciBad, ciSkipped := 0, 0, 0
	for _, line := range strings.Split(strings.TrimRight(string(rows), "\n"), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		f := strings.Split(line, "\t")
		if len(f) < 5 {
			continue
		}
		job, name, kind, envSpec, cmdStr := f[0], f[1], f[2], f[3], f[4]
		if !mine.MatchString(job) {
			ciSkipped++
			continue
		}
		if strings.HasPrefix(kind, "runner:") {
			g.skip("CI[%s] %s — %s", job, name, strings.TrimPrefix(kind, "runner:"))
			continue
		}
		// The ENVIRONMENT is part of the check. CI's root job has no go.work, so the module-boundary
		// guard sees the root module graph in isolation; a developer box with a committed go.work
		// unions every submodule and the guard reports a false red. Derived from whether the job
		// sets up a workspace, not hardcoded here. "-" means no override.
		cmd := exec.Command("bash", "-c", unescapeCI(cmdStr))
		cmd.Env = os.Environ()
		if envSpec != "-" && envSpec != "" {
			cmd.Env = append(cmd.Env, envSpec)
		}
		if out, err := cmd.CombinedOutput(); err != nil {
			ciBad++
			g.bad("CI[%s] %s", job, name)
			for i, ln := range strings.Split(string(out), "\n") {
				if i >= 5 {
					break
				}
				fmt.Fprintf(g.w, "      %s\n", ln)
			}
		} else {
			ciOK++
		}
	}
	if ciSkipped > 0 {
		g.skip("%d %s-only CI hygiene step(s) — wrong platform for this host", ciSkipped, other)
	}
	if ciBad == 0 {
		g.ok("%d CI hygiene check(s) reproduced locally, derived from ci.yml", ciOK)
	}
}

// unescapeCI expands the \n escapes ci_checks.py packs a multi-line step into, matching the shell's
// `printf '%b'`.
func unescapeCI(s string) string {
	r := strings.NewReplacer(`\n`, "\n", `\t`, "\t", `\\`, `\`)
	return r.Replace(s)
}

// ---- 6. group reconciliation + verdict ----
func (g *gpuGate) verdict() int {
	// The generalisation of the per-block guard. A block that dies mid-way emits nothing, and a
	// tally computed from what emitted cannot see the hole — "ran 3" and "ran 4" are both
	// plausible-looking numbers. Reconciling against the DECLARED set is what makes silence
	// detectable (audit G-01).
	g.cur = "" // reconciliation failures belong to no group
	declared := map[string]bool{}
	for _, e := range g.expect {
		declared[e] = true
	}
	var missing, unexpected []string
	for _, e := range g.expect {
		if !g.emitted[e] {
			missing = append(missing, e)
		}
	}
	for _, e := range sortedSet(g.emitted) {
		if !declared[e] {
			unexpected = append(unexpected, e)
		}
	}
	if len(missing) > 0 {
		g.bad("check group(s) declared but emitted NO verdict: %s — the gate tested less than it reports (audit G-01)",
			strings.Join(missing, " "))
	}
	if len(unexpected) > 0 {
		g.bad("check group(s) emitted but not declared: %s — update the declared set so the tally stays meaningful",
			strings.Join(unexpected, " "))
	}

	g.hdr("verdict")
	// ONE UNIT: check groups. "6 declared / 4 ran" previously sat next to an unrelated count and a
	// reader deciding whether to ship could not tell at a glance whether something was missing.
	fmt.Fprintf(g.w, "  check groups: %d declared -> %d reported   |   verdicts within them: %d pass, %d skip, %d fail\n",
		len(g.expect), len(g.emitted), g.pass, g.skipped, g.fail)
	// The release record turns on this distinction: "the suite passed" is NOT "the forward is
	// gated". Say which of the two actually happened, by name, so neither can be read as the other.
	if g.backend == "cuda" {
		s2a, s2b := "not run", "not run"
		if g.emitted["suite"] {
			s2a = "reported"
		}
		if g.emitted["parity"] {
			s2b = "reported"
		}
		fmt.Fprintf(g.w, "  of which: kernel-level suite = %s   |   resident PARITY gates (forward asserted) = %s\n", s2a, s2b)
	}
	if len(g.expect) != len(g.emitted) {
		fmt.Fprintf(g.w, "  (declared != reported: %d group(s) produced no verdict — see the FAIL above)\n",
			len(g.expect)-len(g.emitted))
	}
	if g.skipped > 0 {
		fmt.Fprintf(g.w, "\n  %sSkipped — a skip is not a pass; this gate does NOT cover:%s\n", amber, off)
		for _, n := range g.notes {
			fmt.Fprintf(g.w, "    - %s\n", n)
		}
	}
	if g.ran == 0 {
		fmt.Fprintf(g.w, "\n  %sNO GATE%s — nothing actually ran. Do not read this as a pass.\n", red, off)
		return 1
	}
	// A FILTERED CELL WHOSE TESTS ALL SKIPPED IS THE SAME CLASS AS AN EMPTY ONE:
	// coverage the verdict is vouching for did not run. It is separate from
	// emptyCells because the failure LOOKS different -- the tests exist and were
	// selected, they simply all opted out -- and because `go test` exits 0 on an
	// all-skip package, so nothing upstream notices. This is G-01: two Metal
	// groups printed PASS with their specific claim text across at least two
	// archived release logs while executing nothing.
	if len(g.vacuousCells) > 0 {
		fmt.Fprintf(g.w, "\n  %sVACUOUS CELL(S)%s — every test skipped, so the PASS above vouches for nothing:\n", red, off)
		for _, c := range g.vacuousCells {
			fmt.Fprintf(g.w, "    - %s\n", c)
		}
		fmt.Fprintf(g.w, "    A skip is not a pass. Usually a missing asset, or an opt-in env var\n"+
			"    the cell must set for itself because the gate deliberately unsets ambient ones.\n")
		return 1
	}
	if len(g.emptyCells) > 0 {
		fmt.Fprintf(g.w, "\n  %sEMPTY CELL(S)%s — a -run pattern matched no test, so that coverage is gone:\n", red, off)
		for _, c := range g.emptyCells {
			fmt.Fprintf(g.w, "    - %s\n", c)
		}
		fmt.Fprintf(g.w, "    Nothing FAILED; the tests were not there to run. Fix the pattern or the name.\n")
		return 1
	}
	host := "?"
	if b, err := exec.Command("uname", "-s").Output(); err == nil {
		host = strings.TrimSpace(string(b))
	}
	if g.fail > 0 {
		d := ""
		if g.dirty {
			d = " +dirty"
		}
		fmt.Fprintf(g.w, "\n  %sFAIL%s — %s on %s @ %s%s. Do not tag.\n", red, off, g.backend, host, g.commit, d)
		return 1
	}
	// THREE STATES, NOT TWO. Every check is green here. A dirty tree is not a failure of the CHECKS
	// — it is a failure of PROVENANCE: this verdict names a commit, and an uncommitted edit means it
	// does not describe what that commit contains. Collapsing the two loses the distinction a reader
	// actually needs: is the CODE broken, or is the EVIDENCE broken? It used to print
	// "repo <sha>+dirty" in the provenance block and then PASS as normal — so the gate could emit a
	// verdict reading "PASS at <sha>" for a tree that is not <sha>, with the whole distinction
	// carried by a three-character suffix in a different block. Verdicts get pasted into tag
	// messages; that is what this gate is FOR.
	if g.dirty {
		fmt.Fprintf(g.w, "\n  %sINCONCLUSIVE%s — %d/%d groups green, but the working tree is DIRTY.\n",
			amber, off, len(g.expect), len(g.expect))
		fmt.Fprintf(g.w, "  The verdict names %s and the tree is not %s. Commit, then re-run before tagging.\n", g.commit, g.commit)
		if b, err := exec.Command("git", "status", "--porcelain").Output(); err == nil {
			for i, ln := range strings.Split(strings.TrimRight(string(b), "\n"), "\n") {
				if i >= 10 {
					break
				}
				fmt.Fprintf(g.w, "    %s\n", ln)
			}
		}
		return 1
	}
	fmt.Fprintf(g.w, "\n  %sPASS%s — %s on %s @ %s (%s)\n", green, off, g.backend, host, g.commit,
		time.Now().UTC().Format("2006-01-02T15:04:05Z"))
	fmt.Fprintf(g.w, "  Paste this block for the tag. The OTHER box must pass its own run: no machine has both GPUs.\n")
	return 0
}
