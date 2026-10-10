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

// The pre-tag GPU correctness gate (gate gpu). Run it on EACH GPU box, paste the verdict, then tag.
//
// CI never runs the GPU backends (ci.yml only builds and vets under -tags cuda), and no single machine
// covers both Metal (darwin) and CUDA (Linux+NVIDIA), so GPU correctness rests on running tests by hand
// on two boxes; this makes that one command, one verdict, provenance attached. It is deliberately NOT
// "run everything": a gate that takes two hours gets skipped, and a skipped gate still implies assurance.
// It runs the checks that map to bugs that shipped.
//
// Honesty rules; do not loosen them:
//
//   - A skip is NOT a pass. `go test` prints "ok" for a package whose tests all skipped, so real runs
//     are counted and SKIPPED is reported separately. Green must mean tested.
//   - Stray GPU processes invalidate results, so group 0 refuses to proceed past them.
//   - The CUDA suite runs SEQUENTIALLY (-p 1): parallel packages contend for VRAM and the failures come
//     back as bogus numerics ("cosine 0.000000") rather than "out of memory".
//   - GROUP ACCOUNTING. A tally computed from what EMITTED cannot detect what did not (a block that dies
//     mid-way emits nothing), so the expected groups are DECLARED up front and reconciled in verdict: a
//     group that emits no verdict, or an unexpected group id, is itself a FAIL.
//
// History: docs/code-notes/cmd-gate.md, "gpu gate header".

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

// detail prints the matching assertion lines, or the RAW TAIL if nothing matched, so a failing group
// never explains nothing: `go test` killed by a signal, an OOM or a timeout emits neither "--- FAIL" nor a
// "file.go:N:" line, only "FAIL <pkg> <secs>".
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
		for ln := range strings.SplitSeq(out, "\n") {
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

// vramNote fires on a cosine of EXACTLY zero: an all-zero buffer is what a failed allocation leaves
// behind, so it is not by itself a parity result.
//
// The wording is deliberate: it states the READING and points at the docs/queue-engineering.md A12
// entry; it does NOT name a mechanism. The obvious one (retention on Close) is refuted there, and a gate
// cannot see a mechanism. History: docs/code-notes/cmd-gate.md, "vramNote".
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

// run executes one `go test` cell and returns its results plus the reconstructed -v text. It never
// aborts the gate: a cell that cannot start is a red check, not an abandoned run.
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
	// A FILTERED CELL THAT MATCHED NOTHING IS NOT A PASS, and the aggregate `g.ran == 0` check at the end
	// cannot see it: one empty cell among full ones leaves g.ran > 0. Every -run pattern in this file is a
	// literal test-name prefix or alternation, so renaming a test silently empties its cell.
	g.noteIfEmpty(c, res)
	g.noteIfAllSkipped(c, cr)
	return res, cr, res.text()
}

// noteIfAllSkipped records a FILTERED cell whose tests all SKIPPED, which noteIfEmpty cannot see (a skip
// is a result, so len(res.final) > 0). `go test` exits 0 on an all-skip package, so without this the cell
// reads as a PASS that ran nothing. The cell goes into vacuousCells, which verdict turns into a failure;
// it is a note here rather than a hard failure only because a cell CAN be all-skip on a box without the
// assets, and must then say so loudly.
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

// noteIfEmpty records a FILTERED cell that matched no test at all. Split out from run so a test can drive
// it without spawning `go test` (a cell pointed at "./" from inside cmd/gate re-runs this suite, which
// re-runs the cell).
//
// An unfiltered cell is exempt: an empty -run means "everything", so emptiness there is a different bug
// (no packages, or a build failure) that runCell's own policies already cover.
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

// evidence prints the lines a passing group offered as its own proof: the `ok <pkg> <secs>` line, a
// lifecycle trajectory, a prefill's argmax comparison. They are part of the scope line, not decoration:
// "PASS metal suite" says a verdict, `ok github.com/…/metal 31.2s` says what ran and for how long.
func (g *gpuGate) evidence(out string, re *regexp.Regexp) {
	for ln := range strings.SplitSeq(out, "\n") {
		if re.MatchString(ln) {
			fmt.Fprintf(g.w, "      %s\n", strings.TrimRight(ln, "\r"))
		}
	}
}

var okLineRe = regexp.MustCompile(`^ok\s`)

// failLineRe matches a test-level failure header. It deliberately does NOT match bare `file.go:N:` lines:
// every passing t.Log emits one, and that breadth defeats detail()'s crash fallback, which fires only when
// nothing matches. Assertion lines are still shown, but only the ones below a failing test (failureLines).
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
	for ln := range strings.SplitSeq(out, "\n") {
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
	// The head (signal, address, goroutine) plus the first frames belonging to OUR code: a fixed prefix does
	// not reach them (runtime/purego/reflect frames sit above the goinfer frame), and the test name is the one
	// fact worth printing.
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

var adapterProbeRe = regexp.MustCompile(`ADAPTER_PROBE: backend=(\S+) software=(\S+)`)

// adapterProbeNoneRe matches TestAdapterProbe's OWN explicit "no adapter" line
// (gpu/adapter_probe_test.go: `fmt.Printf("ADAPTER_PROBE: none (%v)\n", err)`) — the genuine,
// the-test-ran-and-found-nothing case, as opposed to the probe subprocess never reaching that
// line at all (see detectWebGPU).
var adapterProbeNoneRe = regexp.MustCompile(`ADAPTER_PROBE: none`)

// detectWebGPU probes for a real WebGPU adapter via a subprocess (gpu/adapter_probe_test.go's
// TestAdapterProbe), so cmd/gate stays free of the gpu build tag and its cgo dependency. It is independent
// of detectBackend's cuda|metal|none choice on purpose: a CUDA Linux box can also have a working Vulkan
// adapter. A software adapter (CI's lavapipe/llvmpipe) counts as present: the group's own tests skip
// hardware-sensitive cases on one (newOrSkipHW).
//
// A miss is not read as "no adapter" regardless of why: when the probe's output shows NEITHER recognized
// line (found-adapter, or TestAdapterProbe's own "ADAPTER_PROBE: none"), w gets a visible note, so a
// driver/hardware question is not mistaken for a build break of ./gpu/. present and backend still resolve
// to false and "" in that case.
func detectWebGPU(w io.Writer) (present bool, backend string) {
	if os.Getenv("GOINFER_GATE_SKIP_WEBGPU") != "" {
		return false, ""
	}
	out, err := exec.Command("go", "test", "-tags", "gpu", "./gpu/", "-run", "TestAdapterProbe", "-v").CombinedOutput()
	present, backend, note := classifyAdapterProbe(out, err)
	if note != "" {
		fmt.Fprint(w, note)
	}
	return present, backend
}

// classifyAdapterProbe is detectWebGPU's pure classification of the probe's captured output, split out so
// a test can drive it with synthetic output. note is non-empty exactly when out shows neither the
// found-adapter line nor TestAdapterProbe's explicit no-adapter line, i.e. the test body never ran.
func classifyAdapterProbe(out []byte, err error) (present bool, backend, note string) {
	if m := adapterProbeRe.FindStringSubmatch(string(out)); m != nil {
		return true, m[1], ""
	}
	if adapterProbeNoneRe.Match(out) {
		return false, "", ""
	}
	return false, "", fmt.Sprintf("note: WebGPU adapter probe produced neither a found-adapter nor "+
		"an explicit no-adapter line — the subprocess likely failed to build or run (err=%v), not "+
		"\"no hardware\"; treating as no adapter present but this is NOT the same finding (V-21).\n", err)
}

func runGPU(w io.Writer, logDir string) int {
	g := &gpuGate{w: w, logDir: logDir, emitted: map[string]bool{}}
	g.backend = detectBackend()
	g.models = env("GOINFER_GATE_MODELS", filepath.Join(home(), "models"))

	// THE GROUPS DEFINE THEIR OWN ENVIRONMENT. Each group sets what it needs, so an ambient value changes what
	// the other groups MEAN: GOINFER_HEAVY_TESTS exported "to be helpful" pulls the real-model tests into the
	// parity group, which then overruns go's default 600s timeout and fails with no assertion line. The
	// variables are unset and REPORTED, not silently ignored: an operator who set one deliberately must see
	// that it did not take effect.
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
		g.expect = []string{"cleangpu", "seam", "suite", "parity", "heavy", "graphsforced", "cgofree", "ptx", "webgpu", "repo"}
	case "metal":
		g.expect = []string{"cleangpu", "seam", "suite", "parity", "cgofree", "lifecycle", "prefill", "webgpu", "repo"}
	default:
		g.expect = []string{"cleangpu", "seam", "suite", "webgpu", "repo"}
	}
	// WebGPU is detected and declared unconditionally, independent of the cuda|metal|none switch above (a CUDA
	// box can also have a Vulkan adapter). It is in g.expect on every branch, so a box with no adapter reads
	// as an explicit SKIP rather than a silent omission.
	hasWebGPU, wgBackend := detectWebGPU(w)

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
	if hasWebGPU {
		fmt.Fprintf(w, "  webgpu      adapter present (backend=%s)\n", wgBackend)
	} else {
		fmt.Fprintf(w, "  webgpu      no adapter detected\n")
	}
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
	g.webgpu(hasWebGPU, wgBackend)
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
	for ln := range strings.SplitSeq(procs, "\n") {
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
// This group asserts no forward: every resident parity gate is behind goinfer_testhooks and runs in 2b
// (cudaParity). The two are separate groups because they answer different questions and one is not
// evidence for the other; "the suite passed" must never be read as "the forward is gated".
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

// drainingTests derives the drain group from a MARKER rather than a list: `drainsDevice(t, why)` in
// cuda/drain_marker_test.go. Those tests take the device to refusal, so they run in their OWN process
// after the main tier. markedTests walks the test files tracking the enclosing `func TestX`; a hand-kept
// -run list would be a constant restating a property, and would drift.
func drainingTests() []string { return markedTests("drainsDevice(t,") }

// isolatedTests derives the fresh-process group from needsFreshProcess(t, why) in
// cuda/isolated_marker_test.go, the same way: edge-of-card tests that fit in a fresh process and not after
// a few hundred others (see the marker's comment).
func isolatedTests() []string { return markedTests("needsFreshProcess(t,") }

// markedTests returns every top-level test in cuda/*_test.go whose body calls marker.
func markedTests(marker string) []string {
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
		for ln := range strings.SplitSeq(string(b), "\n") {
			if m := fnRe.FindStringSubmatch(ln); m != nil {
				name = m[1]
				continue
			}
			if strings.Contains(ln, marker) && name != "" {
				set[name] = true
				name = ""
			}
		}
	}
	return sortedSet(set)
}

// ---- 2c. heavy tier: the real-model group ----
//
// Declared here so it cannot quietly stop running, and TIMED into the verdict so its cost is visible up
// front rather than discovered by someone waiting on a gate they thought took minutes.
func (g *gpuGate) cudaHeavy() {
	g.grp("heavy")
	g.hdr("2c. heavy tier (GOINFER_HEAVY_TESTS=1 — real models; measured 78 min on 2026-09-28)")
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
		for ln := range strings.SplitSeq(string(b), "\n") {
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
	// The timeout needs real headroom over the tier's observed wall (the group header states the last
	// measured one). A timeout that close to the wall reports a dead process with no test-level failure, and
	// the crash head names whichever test was merely running: the most expensive red to diagnose. Do not lower
	// it without re-measuring; the fresh-process tests below moved their share of the wall out of this process.
	// History and the measurements: docs/code-notes/cmd-gate.md, "cudaHeavy: main-tier timeout".
	mainRes, mainCR, mainOut := g.run(cell{
		Name: "cuda-heavy", Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},
		Serial: true, Timeout: "120m",
		Env: map[string]string{"CGO_ENABLED": "0", "GOINFER_HEAVY_TESTS": "1", "GOINFER_ISOLATE_DEFER": "1"},
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

	// INVOCATION 3: each fresh-process test as its own process (needsFreshProcess). The main tier ran
	// with GOINFER_ISOLATE_DEFER=1, so each one skipped there with ISOLATED-SKIP; here it runs alone
	// and logs ISOLATED-RUN. Reconciled both ways, exactly as the drain group is.
	iso := isolatedTests()
	fmt.Fprintf(g.w, "      fresh-process tests (%d, derived from needsFreshProcess() in cuda/isolated_marker_test.go), one process each:\n", len(iso))
	isoRan, isoBad := 0, 0
	for _, name := range iso {
		_, isoCR, isoOut := g.run(cell{
			Name: "cuda-iso-" + name, Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},
			Serial: true, Timeout: "30m", Run: "^" + name + "$",
			Env: map[string]string{"CGO_ENABLED": "0", "GOINFER_HEAVY_TESTS": "1"},
		}, true)
		isoRan += strings.Count(isoOut, "ISOLATED-RUN")
		if isoCR.RC != 0 {
			isoBad++
			g.bad("%s (fresh process)", name)
			g.detail(isoOut, failTestRe)
			fmt.Fprintf(g.w, "      full output: %s\n", isoCR.LogPath)
		}
	}
	isoSkipped := strings.Count(mainOut, "ISOLATED-SKIP")
	fmt.Fprintf(g.w, "      reconciliation: marked=%d  skipped-in-main=%d  ran-alone=%d\n", len(iso), isoSkipped, isoRan)
	if isoSkipped != len(iso) || isoRan != len(iso) {
		g.bad("fresh-process partition does not reconcile — a test is in neither half or in both")
		g.note(fmt.Sprintf("fresh-process mismatch: %d marked, %d skipped in main, %d ran alone", len(iso), isoSkipped, isoRan))
	} else if isoBad == 0 && len(iso) > 0 {
		g.ran++
		g.ok("fresh-process tests (%d, one process each)", len(iso))
	}

	secs := int(time.Since(t0).Seconds())
	if mainCR.RC == 0 {
		g.ran++
		g.ok("heavy tier (real models) — %ds", secs)
		g.evidence(mainOut, okLineRe)
	} else {
		g.bad("heavy tier (real models) — %ds", secs)
		// Name the failing TESTS first, then their assertion lines. Never a bare "file.go:N:" match, which under
		// -v is every log line in the run.
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
// Separate and labelled, deliberately. admitGraphs declines under DEFAULT compute mode without MPS, which
// is correct production behaviour and must stay, so on this box capture/replay is otherwise never
// exercised. Forcing it tests the CODE without changing the admission POLICY. Keep it out of 2b: a forced
// result must never be read as evidence that graphs are admitted in production.
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
	// Build the CUDA SUBMODULE entrypoint. The root ./cmd/serve is a DELIBERATE compile error under -tags cuda
	// (it builds no backend, and failing loudly beats a CPU-only binary named as though it had CUDA), so a
	// check pointed at it could never pass.
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
	var linked strings.Builder
	if b, err := exec.Command("ldd", bin).Output(); err == nil {
		for ln := range strings.SplitSeq(string(b), "\n") {
			l := strings.ToLower(ln)
			if strings.Contains(l, "libcuda") || strings.Contains(l, "libnvrtc") || strings.Contains(l, "libcudart") {
				linked.WriteString(ln + "\n")
			}
		}
	}
	if linked.String() != "" {
		g.bad("binary links CUDA libraries — the cgo-free claim is false:")
		for ln := range strings.SplitSeq(strings.TrimRight(linked.String(), "\n"), "\n") {
			fmt.Fprintf(g.w, "      %s\n", ln)
		}
		return
	}
	g.ok("serve builds CGO_ENABLED=0 and links no CUDA toolkit (driver is dlopen'd at runtime)")
}

// ---- 4. PTX reproduces from source, each at the NVRTC it records ----
//
// This block must ALWAYS reach pass/fail/skip: a check that can neither pass nor fail is the same defect
// as one that can only fail. In Go it cannot exit early without returning, and the group reconciliation
// in verdict catches it if it ever does.
//
// Every .ptx states the toolchain that produced it in its own header
// (`// Cuda compilation tools, release 12.6, V12.6.85`). That provenance is what we rebuild against, NOT
// whatever NVRTC this box defaults to: the tree legitimately carries a MIX (kernels added after a
// toolchain bump were built at the newer one, and the audited ones are pinned), so a single-toolchain
// rebuild reports a false FAIL on every file from the other era.
//
// NOTHING IS EXEMPTED BY NAME. A file is skipped only when the NVRTC version IT RECORDS is not installed
// here, and the skip names the version so it is actionable.
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
		// Zero artifacts verified is not a pass with a footnote. The device-free
		// TestPTX_matchesSourcesAndBindings still guards kernel names and signatures, but this group exists to
		// prove byte-identity, and here it proved nothing.
		g.ran++
		g.bad("PTX reproducibility: no usable NVRTC for any recorded version — zero of %d artifacts verified", total)
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
	// Build the METAL submodule entrypoint (build.Dir), not the root one: the root ./cmd/serve builds no
	// backend (cmd/serve/backendtag_guard_metal.go says so), so building it would succeed forever and the
	// group would assert "Metal is dlopen'd via purego-objc" about a binary with no Metal in it.
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
	var linked strings.Builder
	if b, err := exec.Command("otool", "-L", bin).Output(); err == nil {
		for ln := range strings.SplitSeq(string(b), "\n") {
			l := strings.ToLower(ln)
			if strings.Contains(l, "metal.framework") || strings.Contains(l, "metalperformanceshaders") {
				linked.WriteString(ln + "\n")
			}
		}
	}
	os.Remove(bin)
	if linked.String() != "" {
		g.bad("binary links the Metal framework — the cgo-free claim is false:")
		for ln := range strings.SplitSeq(strings.TrimRight(linked.String(), "\n"), "\n") {
			fmt.Fprintf(g.w, "        %s\n", strings.TrimSpace(ln))
		}
		return
	}
	g.ran++
	g.ok("metal/cmd/serve builds CGO_ENABLED=0 and links no Metal framework (dlopen'd via purego-objc)")
}

// ---- Metal resident PARITY gates: the forward is asserted here ----
//
// The Metal mirror of cudaParity. Without -tags goinfer_testhooks the "full metal suite" is the kernel
// tier plus the snapshot golden, and no resident-parity gate is even compiled. The cell is filtered to
// the resident-parity gates (metalParityRun) rather than the whole tagged tree: the tag also selects long
// device tests that belong to other groups, and a cell that quietly runs everything makes a timeout
// indistinguishable from a crash. -timeout is declared for the same reason cudaParity declares it.
func (g *gpuGate) metalParity() {
	g.grp("parity")
	g.hdr("2c. resident PARITY gates (-tags goinfer_testhooks — the forward is asserted here)")
	_, cr, out := g.run(cell{
		Name: "metal-parity", Pkgs: []string{"./metal/"}, Tags: []string{"goinfer_testhooks"},
		// metalParityRun (parity.go) is named so this cell and TestMetalGateIsListedOrExplicitlyNotRequired read
		// the same string; a hand-copied pattern here is how gate-shaped tests go unmatched.
		Run:     metalParityRun,
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

// metalLifecycle gates Close() actually freeing memory on Metal. It runs WITHOUT -short (the suite uses it)
// because it loads real models, and covers both conditions: the sequential sawtooth, and a second model
// alive, the case that made CUDA's first fix look correct when it was not.
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
		// The cell sets GOINFER_HEAVY_TESTS itself: its tests call requireHeavyModel and the gate unsets the
		// ambient value (see runGPU), so without it every test skips, `go test` exits 0 and the group would print
		// PASS while running nothing.
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

// metalPrefill gates PrefillLast, the f16 simdgroup_matrix TTFT path, against a real checkpoint: it is a
// shipped path that once emitted NaN logits at every prompt length (the int8 LM head was read as packed
// int4), and nothing else exercises it on real weights.
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
		// GOINFER_HEAVY_TESTS for the same reason as metal-lifecycle above.
		Env: map[string]string{"GOINFER_METAL_MODEL": m, "GOINFER_HEAVY_TESTS": "1"},
	}, false)
	// A cell whose named tests ALL skipped has RC==0 and would print PASS while verifying nothing, so check
	// cr.vacuous() as the sibling cells do.
	if cr.RC != 0 || cr.vacuous() {
		g.bad("prefill parity/NaN gate — the f16-MMA TTFT path is wrong on a shipped model, " +
			"or every test in it skipped and verified nothing")
		g.detail(out, regexp.MustCompile(`^--- FAIL|parity FAIL|contain NaN|\.go:[0-9]+:`))
		return
	}
	g.ran++
	g.ok("prefill matches sequential decode and emits finite logits (no NaN)")
	g.evidence(out, regexp.MustCompile(`argmax matches|faster TTFT`))
}

// ---- W. WebGPU: adapter-independent of the primary backend ----
//
// Runs ./gpu/ whenever an adapter is detected, whatever the primary backend. The resident-parity gates
// (qwen3.5 DeltaNet, granite/nemotron Mamba-2) need GOINFER_DNET_PARITY / GOINFER_SSM_PARITY; this group
// sets both itself, like every group here (see runGPU's rule on the environment).
func (g *gpuGate) webgpu(present bool, backend string) {
	g.grp("webgpu")
	g.hdr("W. WebGPU suite + resident parity (adapter-detected, -tags 'gpu goinfer_testhooks')")
	if !present {
		g.skip("no WebGPU adapter detected — set GOINFER_GATE_SKIP_WEBGPU to silence this note")
		return
	}
	_, cr, out := g.run(cell{
		Name: "webgpu-suite", Pkgs: []string{"./gpu/"}, Tags: []string{"gpu"},
		Serial: true, Extra: []string{"-short"},
	}, false)
	if cr.RC != 0 {
		g.bad("webgpu suite (backend=%s)", backend)
		g.detail(out, failLineRe)
		return
	}
	g.ran++
	g.ok("webgpu suite (backend=%s)", backend)
	g.evidence(out, okLineRe)

	// The resident-parity gates. The qwen3.5 fixtures (dense and MoE) and the granite/nemotron Mamba-2
	// fixtures have model.safetensors gitignored (only config.json is tracked), so on a clone without them
	// the tests skip; do not describe them as tracked. This cell counts top-level results only (TopLevelOnly),
	// and Go reports a parent whose subtests ALL skipped as a top-level PASS, which would make cr.vacuous()
	// false. TestQwen35ResidentParity therefore skips itself when none of its subtests ran, so the tally sees a
	// real Skip.
	_, cr2, out2 := g.run(cell{
		Name: "webgpu-parity", Pkgs: []string{"./gpu/"}, Tags: []string{"gpu", "goinfer_testhooks"},
		Run:     webgpuParityRun, // parity.go; checked by TestWebGPUGateIsListedOrExplicitlyNotRequired (G-10)
		Serial:  true,
		Timeout: "10m",
		Env:     map[string]string{"GOINFER_DNET_PARITY": "1", "GOINFER_SSM_PARITY": "1"},
	}, false)
	if cr2.RC != 0 || cr2.vacuous() {
		g.bad("webgpu resident parity gates — a WebGPU forward moved, or nothing ran at all")
		g.detail(out2, failLineRe)
		return
	}
	g.ran++
	g.ok("webgpu resident parity gates (qwen3.5 DeltaNet dense; MoE/granite/nemotron opt-in fixtures)")
	g.evidence(out2, okLineRe)
}

// repoHygiene runs the checks CI runs, DERIVED from .github/workflows/ci.yml by scripts/ci_checks.py
// rather than listed by hand, so a check CI gains appears here with no edit to this file. A hand-written
// list is a subset of CI's, and CI then goes red on something this gate passed.
func (g *gpuGate) repoHygiene() {
	g.grp("repo")
	g.hdr("5. repo hygiene (derived from .github/workflows/ci.yml)")

	// The queue's citations, commit AND path:line: a state document is cited without being re-derived, so a
	// wrong reference in it propagates with more confidence than the same error in conversation.
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
	for line := range strings.SplitSeq(strings.TrimRight(string(rows), "\n"), "\n") {
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
		if after, ok := strings.CutPrefix(kind, "runner:"); ok {
			g.skip("CI[%s] %s — %s", job, name, after)
			continue
		}
		// The ENVIRONMENT is part of the check. CI's root job has no go.work, so the module-boundary
		// guard sees the root module graph in isolation; a developer box with a committed go.work
		// unions every submodule and the guard reports a false red. Derived from whether the job
		// sets up a workspace, not hardcoded here. "-" means no override.
		cmd := exec.Command("bash", "-c", unescapeCI(cmdStr))
		cmd.Env = withGoBin(os.Environ())
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

// withGoBin puts `go env GOPATH`/bin first on PATH. CI installs its tools (staticcheck, built by
// .github/actions/staticcheck) with `go install` and calls them by bare name, which works there because
// setup-go puts GOPATH/bin on PATH; a developer shell need not. Prepending the same directory reproduces
// CI's environment; a missing binary still fails, now with the install hint.
func withGoBin(env []string) []string {
	out, err := exec.Command("go", "env", "GOPATH").Output()
	gp := strings.TrimSpace(string(out))
	if err != nil || gp == "" {
		return env
	}
	bin := filepath.Join(strings.Split(gp, string(os.PathListSeparator))[0], "bin")
	res := make([]string, 0, len(env)+1)
	found := false
	for _, kv := range env {
		if v, ok := strings.CutPrefix(kv, "PATH="); ok {
			kv, found = "PATH="+bin+string(os.PathListSeparator)+v, true
		}
		res = append(res, kv)
	}
	if !found {
		res = append(res, "PATH="+bin)
	}
	return res
}

// unescapeCI expands the \n escapes ci_checks.py packs a multi-line step into, matching the shell's
// `printf '%b'`.
func unescapeCI(s string) string {
	r := strings.NewReplacer(`\n`, "\n", `\t`, "\t", `\\`, `\`)
	return r.Replace(s)
}

// ---- 6. group reconciliation + verdict ----
func (g *gpuGate) verdict() int {
	// Reconcile against the DECLARED group set: a block that dies mid-way emits nothing, and a tally computed
	// from what emitted cannot see the hole.
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
	// ONE UNIT: check groups, so a reader deciding whether to ship sees at a glance whether one is missing.
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
	// A FILTERED CELL WHOSE TESTS ALL SKIPPED IS THE SAME CLASS AS AN EMPTY ONE: coverage the verdict vouches
	// for did not run. It is separate from emptyCells because the tests exist and were selected, they all
	// opted out, and `go test` exits 0 on an all-skip package, so nothing upstream notices.
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
	// THREE STATES, NOT TWO. Every check is green here, but a dirty tree is a failure of PROVENANCE, not of
	// the checks: the verdict names a commit, and an uncommitted edit means it does not describe what that
	// commit contains. Collapsing the two loses the distinction a reader needs (is the CODE broken, or the
	// EVIDENCE?). Verdicts get pasted into tag messages, which is what this gate is for.
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
