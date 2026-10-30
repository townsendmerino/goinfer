package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
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

	heavySkipped bool     // the real-model tier was skipped: the run is not the whole gate, and its record says so
	heavyOnly    []string // -heavy-group: the heavy groups this run is narrowed to (none: every group)
	recordDir    string   // where the run's record is written (gpu_record.go); empty writes none
	gpuName      string   // the device line printed in the provenance block, for the record
	ptxKeepDir   string   // where a rebuilt .ptx that differs from its committed form is written; empty writes none

	heavyProcs []heavyProcRecord // each heavy-tier main-group process's memory, for the record
	heavyT0    time.Time         // when the live heavy tier started, for its wall time

	cells []cellResult // every cell run, in order, for the run record (gate_gpu_run.json)
	prov  provenance   // the run's provenance, for the run record
	runIn gpuRunInputs // the side inputs the moved groups' graders read, for the run record
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

// detail prints detailLines(out, re).
func (g *gpuGate) detail(out string, re *regexp.Regexp) {
	for _, ln := range detailLines(out, re) {
		fmt.Fprintln(g.w, ln)
	}
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
	g.cells = append(g.cells, cr)
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

// webgpuProbe is what the gate knows about this host's WebGPU adapter before group W runs. failed is non-empty when the
// probe never answered (it failed to build, crashed, or exited before printing either of its lines); optOut is
// GOINFER_GATE_SKIP_WEBGPU.
type webgpuProbe struct {
	present bool
	backend string
	failed  string
	optOut  bool
}

// record is the probe as a run record holds it.
func (p webgpuProbe) record() *webgpuProbeRecord {
	return &webgpuProbeRecord{OptOut: p.optOut, Present: p.present, Backend: p.backend, Failed: p.failed}
}

// detectWebGPU probes for a real WebGPU adapter via a subprocess (gpu/internal/kernels' TestAdapterProbe), so cmd/gate
// stays free of the gpu build tag and its cgo dependency. It is independent of detectBackend's cuda|metal|none choice on
// purpose: a CUDA Linux box can also have a working Vulkan adapter. A software adapter (CI's lavapipe/llvmpipe) counts as
// present: the group's own tests skip hardware-sensitive cases on one (newOrSkipHW). A probe that never answered is not
// "no adapter": group W grades it as a failure (V-21, docs/code-notes/cmd-gate.md#detectWebGPU).
func detectWebGPU(w io.Writer) webgpuProbe {
	if os.Getenv("GOINFER_GATE_SKIP_WEBGPU") != "" {
		return webgpuProbe{optOut: true}
	}
	out, err := exec.Command("go", "test", "-tags", "gpu", "./gpu/internal/kernels/", "-run", "TestAdapterProbe", "-v").CombinedOutput()
	present, backend, note := classifyAdapterProbe(out, err)
	if note != "" {
		fmt.Fprint(w, note)
	}
	return webgpuProbe{present: present, backend: backend, failed: note}
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
		"an explicit no-adapter line — the subprocess failed to build or run (err=%v), not "+
		"\"no hardware\"; the WebGPU group fails on it (V-21).\n", err)
}

func runGPU(w io.Writer, logDir, recordDir string, heavyOnly []string) int {
	for _, l := range heavyOnly {
		if !slices.Contains(heavyGroupLabels(), l) {
			fmt.Fprintf(w, "gate gpu: -heavy-group: no heavy group %q (groups: %s)\n", l, strings.Join(heavyGroupLabels(), " "))
			return 2
		}
	}
	g := &gpuGate{w: w, logDir: logDir, recordDir: recordDir, emitted: map[string]bool{}, heavyOnly: heavyOnly}
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

	g.expect = gpuExpect(g.backend)
	// WebGPU is detected and declared unconditionally, independent of the cuda|metal|none switch above (a CUDA
	// box can also have a Vulkan adapter). It is in g.expect on every branch, so a box with no adapter reads
	// as an explicit SKIP rather than a silent omission.
	wg := detectWebGPU(w)
	g.runIn.WebGPU = wg.record()

	prov := gatherProvenance(nil)
	g.prov = prov
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
	switch {
	case wg.optOut:
		fmt.Fprintf(w, "  webgpu      not probed (GOINFER_GATE_SKIP_WEBGPU)\n")
	case wg.failed != "":
		fmt.Fprintf(w, "  webgpu      adapter probe FAILED (V-21)\n")
	case wg.present:
		fmt.Fprintf(w, "  webgpu      adapter present (backend=%s)\n", wg.backend)
	default:
		fmt.Fprintf(w, "  webgpu      no adapter detected\n")
	}
	switch g.backend {
	case "cuda":
		if b, err := exec.Command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader").Output(); err == nil {
			g.gpuName = strings.SplitN(strings.TrimSpace(string(b)), "\n", 2)[0]
			fmt.Fprintf(w, "  gpu         %s\n", g.gpuName)
		}
	case "metal":
		cpu := "Apple Silicon"
		if b, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output(); err == nil {
			cpu = strings.TrimSpace(string(b))
		}
		g.gpuName = cpu
		fmt.Fprintf(w, "  gpu         %s\n", cpu)
	}
	if g.dirty {
		g.note("WORKING TREE DIRTY — this verdict does not describe a committed state.")
	}

	g.runIn.RunnerNotes = slices.Clone(g.notes)
	src := liveInputs{g} // a live source never errs
	g.cleanGPUGroup(src)
	g.seamGroup(src)
	switch g.backend {
	case "cuda":
		g.cudaGroups(src, true)
	case "metal":
		m := g.metalModel()
		_, err := os.Stat(m)
		g.runIn.MetalModel, g.runIn.MetalModelPresent = m, err == nil
		g.metalGroups(src)
	default:
		g.grp("suite")
		g.hdr("2-4. backend suites")
		g.skip("no GPU backend detected on this host — only the seam gate ran")
	}
	g.webgpuGroup(src)
	g.repoGroup(src)
	g.runIn.Verdict = g.verdictInputs()
	g.writeGPURunRecord()
	return g.verdict()
}

// ---- 0. the card must be quiet, or every memory-sensitive result below is noise ----

// cleanGPUGroup prints gradeCleanGPU's verdict on the card's processes.
func (g *gpuGate) cleanGPUGroup(src gpuInputs) error {
	in, err := src.cleanGPU()
	if err != nil || in == nil {
		return err
	}
	g.grp("cleangpu")
	g.hdr("0. clean GPU")
	g.apply(gradeCleanGPU(*in))
	return nil
}

// readCleanGPU queries nvidia-smi for the compute processes and the memory in use, on CUDA only.
func readCleanGPU(backend string) cleanGPUInput {
	in := cleanGPUInput{Backend: backend}
	if backend != "cuda" {
		return in
	}
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return in
	}
	in.HaveSMI, in.Used = true, "?"
	if b, err := exec.Command("nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader").Output(); err == nil {
		in.Procs = strings.TrimSpace(string(b))
	}
	if b, err := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader").Output(); err == nil {
		in.Used = strings.TrimSpace(strings.ReplaceAll(string(b), "MiB", ""))
	}
	return in
}

// ---- 1. seam: no GPU needed, and it is the class that cost five weeks ----
func (g *gpuGate) seamGroup(src gpuInputs) error {
	g.grp("seam")
	g.hdr("1. seam (runs anywhere — no GPU, no model download)")
	cr, res, out, err := src.cell(seamCell)
	if err != nil {
		return err
	}
	g.apply(gradeSeam(cr, res.ranCount(), out))
	return nil
}

// cudaGroups runs the CUDA backend's groups in the gate's order: the kernel-level suite (2a), the resident parity gates (2b),
// the heavy tier (2c), forced-capture graphs (2d), cgofree (3) and the PTX rebuild (4). The kernel-level suite asserts no
// forward: every resident parity gate is behind goinfer_testhooks and runs in 2b, and "the suite passed" must never be read
// as "the forward is gated". The heavy tier and the PTX rebuild grade inside their runners still, so they run only live.
func (g *gpuGate) cudaGroups(src gpuInputs, live bool) error {
	steps := []func() error{
		func() error {
			g.grp("suite")
			g.hdr(cudaSuiteHeader)
			cr, res, out, err := src.cell(cudaSuiteCell)
			if err != nil {
				return err
			}
			g.apply(gradeCUDASuite(cr, testNames(res.topLevel("skip")), out))
			return nil
		},
		func() error {
			g.grp("parity")
			g.hdr(cudaParityHeader)
			cr, _, out, err := src.cell(cudaParityCell)
			if err != nil {
				return err
			}
			g.apply(gradeCUDAParity(cr, out, src.parityFreeVRAM(cr, out)))
			return nil
		},
	}
	steps = append(steps, func() error { return g.heavyTier(src) })
	steps = append(steps,
		func() error {
			g.grp("graphsforced")
			g.hdr(cudaGraphsHeader)
			cr, res, out, err := src.cell(cudaGraphsCell)
			if err != nil {
				return err
			}
			g.apply(gradeCUDAGraphs(cr, res.ranCount(), testNames(res.topLevel("skip")), out))
			return nil
		},
		func() error {
			in, err := src.cgoFree()
			if err != nil || in == nil {
				return err
			}
			g.grp("cgofree")
			g.hdr(cudaCgoHeader)
			g.apply(gradeCUDACgoFree(*in))
			return nil
		})
	steps = append(steps, func() error { return g.ptxGroup(src) })
	for _, step := range steps {
		if err := step(); err != nil {
			return err
		}
	}
	return nil
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
// Declared here so it cannot quietly stop running, and TIMED into the verdict so its cost is visible up front rather than
// discovered by someone waiting on a gate they thought took minutes. The main group runs as one process per model family, so
// no process holds two large checkpoints, each under its own timeout (runHeavyGroup); the drain group and each fresh-process
// test then run in a process of their own. Both halves of each split are reconciled.

// heavyTier runs the heavy tier through src, printing what the heavy graders decide: the partition, each main-group process
// (its streamed results and memory line), the split's reconciliation, the drain group, the fresh-process tests and the tier's
// verdict with its skip census. A recorded run without the tier's input does not regrade it.
func (g *gpuGate) heavyTier(src gpuInputs) error {
	in, err := src.heavy()
	if err != nil || in == nil {
		return err
	}
	g.grp("heavy")
	g.hdr(heavyHeader)
	g.heavySkipped = in.Skipped
	vs, run := gradeHeavyStart(*in)
	for _, v := range vs {
		g.apply(v)
	}
	if !run {
		return nil
	}
	groups, listed, drain, iso := heavyRunSet(*in)
	var mainRes []*results
	var mainOut string
	mainRC := 0
	for i, h := range groups {
		fmt.Fprintf(g.w, "      main group, %s (%d test(s)), streaming (one line per test):\n", h.Label, len(h.Tests))
		cr, res, out, peak, err := src.heavyCell(i, h)
		if err != nil {
			return err
		}
		mainRes = append(mainRes, res)
		mainOut += out
		lines, notes, rec := heavyProcLines(h, cr, peak)
		g.heavyProcs = append(g.heavyProcs, rec)
		g.apply(groupVerdict{State: groupNone, Lines: lines, Notes: notes})
		if cr.RC != 0 {
			mainRC = cr.RC
		}
	}
	missing, doubled := reconcileGroups(listed, mainRes)
	g.apply(gradeHeavySplit(len(listed), missing, doubled))

	// The drain group, its own process, after everything else. GOINFER_DRAIN_GROUP is what un-skips the marker; -run
	// restricts it to the derived set so the rest of the package is not paid for twice.
	if v, none := heavyNoDrain(in.Run, drain); none {
		g.apply(v) // a narrowed run with no drain test inside its groups runs no drain process
	} else {
		fmt.Fprintf(g.w, "      drain group, separate process:\n")
		drainCR, _, drainOut, err := src.streamCell(heavyDrainCell(drain))
		if err != nil {
			return err
		}
		for _, v := range gradeHeavyDrain(len(drain), strings.Count(mainOut, "DRAIN-GROUP-SKIP"), strings.Count(drainOut, "DRAIN-GROUP-RUN"), drainCR, drainOut) {
			g.apply(v)
		}
	}

	// Each fresh-process test as its own process (needsFreshProcess). The main tier ran with GOINFER_ISOLATE_DEFER=1, so each
	// skipped there with ISOLATED-SKIP; here it runs alone and logs ISOLATED-RUN.
	fmt.Fprintf(g.w, "      fresh-process tests (%d, derived from needsFreshProcess() in cuda/isolated_marker_test.go), one process each:\n", len(iso))
	isoRan, isoBad := 0, 0
	for _, name := range iso {
		cr, _, out, err := src.streamCell(heavyIsoCell(name))
		if err != nil {
			return err
		}
		isoRan += strings.Count(out, "ISOLATED-RUN")
		if v, bad := gradeHeavyIsoTest(name, cr, out); bad {
			isoBad++
			g.apply(v)
		}
	}
	g.apply(gradeHeavyIso(len(iso), strings.Count(mainOut, "ISOLATED-SKIP"), isoRan, isoBad))

	runLines := 0
	var skipped [][]string
	for _, r := range mainRes {
		runLines += r.runLines
		skipped = append(skipped, testNames(r.topLevel("skip")))
	}
	g.apply(gradeHeavyTier(mainRC, src.heavySecs(), mainOut, runLines, skipped))
	return nil
}

// heavyDrainCell and heavyIsoCell are the drain group's process and one fresh-process test's.
func heavyDrainCell(drain []string) cell {
	return cell{Name: "cuda-drain", Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},
		Serial: true, Timeout: "20m", Run: "^(" + strings.Join(drain, "|") + ")$",
		Env: map[string]string{"CGO_ENABLED": "0", "GOINFER_HEAVY_TESTS": "1", "GOINFER_DRAIN_GROUP": "1"}}
}

func heavyIsoCell(name string) cell {
	return cell{Name: "cuda-iso-" + name, Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},
		Serial: true, Timeout: "30m", Run: "^" + name + "$",
		Env: map[string]string{"CGO_ENABLED": "0", "GOINFER_HEAVY_TESTS": "1"}}
}

// readHeavy is the heavy tier's input before any process runs: whether it is skipped, the markers' tests, ./cuda/'s and the
// kernels package's listings and the model-family partition of them.
func (g *gpuGate) readHeavy() heavyInput {
	if os.Getenv("GOINFER_GATE_SKIP_HEAVY") != "" {
		return heavyInput{Skipped: true}
	}
	in := heavyInput{Drain: drainingTests(), Iso: isolatedTests()}
	listed, err := listCUDATests("./cuda/")
	in.Total = len(listed)
	if err != nil || in.Total == 0 {
		if err != nil {
			in.ListErr = err.Error()
		}
		return in
	}
	groups, problems, err := partitionHeavy(listed, g.models)
	if err != nil {
		in.PartitionErr = err.Error()
		return in
	}
	// The kernel-level tests in ./cuda/kernels/ load no model, so they are one more process, run whole.
	kernels, err := listCUDATests(cudaKernelsPkg)
	in.Kernels = len(kernels)
	if err != nil || len(kernels) == 0 {
		if err != nil {
			in.KernelsErr = err.Error()
		}
		return in
	}
	in.Groups = append(groups, heavyGroup{Label: "kernels", Tests: kernels, Pkg: cudaKernelsPkg})
	in.Listed = append(listed, kernels...)
	in.Problems = problems
	in.Run = g.heavyOnly
	return in
}

// cudaKernelsPkg holds the CUDA kernel-level tests that need nothing from package cuda but its PTX.
const cudaKernelsPkg = "./cuda/kernels/"

// listCUDATests lists pkg's top-level tests under the heavy tier's tags.
func listCUDATests(pkg string) ([]string, error) {
	lst := exec.Command("go", "test", "-tags", "cuda goinfer_testhooks", pkg, "-list", ".*")
	lst.Env = append(os.Environ(), "CGO_ENABLED=0")
	b, err := lst.Output()
	if err != nil {
		return nil, err
	}
	var listed []string
	for ln := range strings.SplitSeq(string(b), "\n") {
		if strings.HasPrefix(ln, "Test") {
			listed = append(listed, strings.TrimSpace(ln))
		}
	}
	return listed, nil
}

// partitionHeavy derives the heavy tier's main-group processes for the listed ./cuda/ tests from the source (heavyPartition).
func partitionHeavy(listed []string, modelsDir string) ([]heavyGroup, []string, error) {
	refs, err := testReachableLiterals("cuda", []string{"cuda", "goinfer_testhooks"}, filepath.Join("testdata", "assets.json"))
	if err != nil {
		return nil, nil, err
	}
	groups, problems := heavyPartition(listed, refs, modelsDirEntries(modelsDir))
	return groups, problems, nil
}

// runHeavyGroup runs one main-group process with its test binary's memory sampled. An all-skip group is not marked
// vacuous here: a family whose checkpoints are absent skips whole, and the tier's skip census names each test.
func (g *gpuGate) runHeavyGroup(h heavyGroup) (*results, cellResult, string, procPeak) {
	cfg := &gateConfig{Name: "gpu", TopLevelOnly: true, RCIsFailure: true}
	res := newResults()
	res.stream = func(_, test, action string) {
		fmt.Fprintf(g.w, "        · %s: %s\n", strings.ToUpper(action), test)
	}
	var peak procPeak
	pkg := h.Pkg
	if pkg == "" {
		pkg = "./cuda/"
	}
	c := cell{
		Name: "cuda-heavy-" + h.Label, Pkgs: []string{pkg}, Tags: []string{"cuda", "goinfer_testhooks"},
		Serial: true, Timeout: "120m", Run: h.Run,
		Env: map[string]string{"CGO_ENABLED": "0", "GOINFER_HEAVY_TESTS": "1", "GOINFER_ISOLATE_DEFER": "1"},
		OnStart: func(pid int) func() {
			stop := startProcSampler(pid, res.inFlight)
			return func() { peak = stop() }
		},
	}
	cr := runCell(c, cfg, res, g.logDir)
	g.cells = append(g.cells, cr)
	g.noteIfEmpty(c, res)
	return res, cr, res.text(), peak
}

// reportPeak prints one process's memory line and returns it as a record row.
func (g *gpuGate) reportPeak(h heavyGroup, cr cellResult, p procPeak) heavyProcRecord {
	rec := heavyProcRecord{Group: h.Label, Tests: len(h.Tests), WallS: int(cr.Dur.Seconds()), Samples: p.Samples}
	if p.Samples == 0 {
		fmt.Fprintf(g.w, "      %s memory: NOT SAMPLED (no test binary found under /proc)\n", h.Label)
		return rec
	}
	rec.VmHWMMB, rec.VmSwapPkMB = p.HWMKB/1024, p.SwapPeakKB/1024
	var top []string
	for _, n := range p.topTests(3) {
		top = append(top, fmt.Sprintf("%s %d MB", n, p.ByTest[n]/1024))
	}
	fmt.Fprintf(g.w, "      %s memory: VmHWM %d MB, peak VmSwap %d MB, wall %ds (pid %d, %d samples); largest RSS in flight: %s\n",
		h.Label, rec.VmHWMMB, rec.VmSwapPkMB, rec.WallS, p.PID, p.Samples, strings.Join(top, ", "))
	return rec
}

// reconcileGroups returns the listed tests with no top-level result in any process, and those with a result in
// more than one.
func reconcileGroups(listed []string, procs []*results) (missing, doubled []string) {
	seen := map[string]int{}
	for _, r := range procs {
		inThis := map[string]bool{}
		for _, k := range r.order {
			if !isSubtest(k) && r.final[k] != "" && !inThis[k.Test] {
				inThis[k.Test] = true
				seen[k.Test]++
			}
		}
	}
	for _, n := range listed {
		switch seen[n] {
		case 0:
			missing = append(missing, n)
		case 1:
		default:
			doubled = append(doubled, n)
		}
	}
	return missing, doubled
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
func probeNVRTC() map[string][2]string {
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

// readPTX rebuilds each committed PTX file that has a source of its own with cuda/build_ptx.sh, at the NVRTC version the file
// records, and records its outcome for gradePTX. The committed files are backed up first and restored after, so the check
// does not mutate the tree. A file that differs is written to keepDir when it is set (`gate ptx -keep`). A file is left
// unverified only when the NVRTC version it records is not installed here or its build failed; nothing is exempted by name.
func readPTX(keepDir string) ptxInput {
	st, err := os.Stat(filepath.Join("cuda", "build_ptx.sh"))
	if why := ptxScriptDecline(st, err, runtime.GOOS); why != "" {
		return ptxInput{Decline: why}
	}
	nvrtc := probeNVRTC()
	in := ptxInput{Toolchains: sortedSet(func() map[string]bool {
		m := map[string]bool{}
		for k := range nvrtc {
			m[k] = true
		}
		return m
	}())}

	ptxFiles, _ := filepath.Glob(filepath.Join("cuda", "testdata", "*.ptx"))
	sort.Strings(ptxFiles)
	backup, err := os.MkdirTemp("", "gate_ptx_before")
	if err != nil {
		in.Backup = err.Error()
		return in
	}
	defer os.RemoveAll(backup)
	saved := map[string][]byte{}
	for _, f := range ptxFiles {
		if b, err := os.ReadFile(f); err == nil {
			saved[f] = b
		}
	}
	for _, f := range ptxFiles {
		base := strings.TrimSuffix(filepath.Base(f), ".ptx")
		// kernel is what build_ptx.sh is asked for. <k>_kv16.ptx has no source of its own name: it is <k>.cu built with
		// the KV cache as f16, and <k>_cap.ptx and <k>_kv16_cap.ptx are <k>.cu built with the attention softcap (at f32 and at f16
		// KV); build_ptx.sh writes each whenever it builds <k>.
		kernel := base
		if _, err := os.Stat(filepath.Join("cuda", base+".cu")); err != nil {
			for _, suffix := range []string{"_kv16_cap", "_cap", "_kv16"} {
				if k := strings.TrimSuffix(base, suffix); k != base {
					kernel = k
					break
				}
			}
			if _, err := os.Stat(filepath.Join("cuda", kernel+".cu")); kernel == base || err != nil {
				continue // no source ⇒ not ours to reproduce
			}
		}
		want := recordedNVRTC(f)
		pf := ptxFile{Base: base, Want: want}
		ent, ok := nvrtc[want]
		switch {
		case want == "":
			pf.Outcome, pf.Why = "unverified", "no recorded version"
		case !ok:
			pf.Outcome, pf.Why = "unverified", "needs V"+want
		default:
			cmd := exec.Command("./build_ptx.sh", kernel)
			cmd.Dir = "cuda"
			cmd.Env = append(os.Environ(), "NVRTC_LIB="+ent[0], "CUDA_INC="+ent[1])
			if err := cmd.Run(); err != nil {
				pf.Outcome, pf.Why = "unverified", "build failed at V"+want
				break
			}
			now, _ := os.ReadFile(f)
			pf.Outcome = "ok"
			if string(now) != string(saved[f]) {
				pf.Outcome = "differs"
				if keepDir != "" {
					kerr := os.MkdirAll(keepDir, 0o755)
					if kerr == nil {
						kerr = os.WriteFile(filepath.Join(keepDir, base+".ptx"), now, 0o644)
					}
					if kerr != nil {
						pf.KeepErr = kerr.Error()
					}
				}
			}
		}
		in.Files = append(in.Files, pf)
	}
	// Restore; this check must not mutate the tree.
	for f, b := range saved {
		_ = os.WriteFile(f, b, 0o644)
	}
	return in
}

// ptxGroup prints gradePTX's verdicts on the ptx group's input; a recorded run without one does not regrade it.
func (g *gpuGate) ptxGroup(src gpuInputs) error {
	in, err := src.ptx()
	if err != nil || in == nil {
		return err
	}
	g.grp("ptx")
	g.hdr(ptxHeader)
	for _, v := range gradePTX(*in) {
		g.apply(v)
	}
	return nil
}

// ptxScriptDecline says why cuda/build_ptx.sh cannot be run, "" when it can. The execute bit is checked only on an OS that
// has one: Windows reports none for any file, so checking it there calls a script that is present "missing" and ends the
// group before it counts the artifacts it did not verify.
func ptxScriptDecline(st os.FileInfo, statErr error, goos string) string {
	switch {
	case statErr != nil:
		return "is missing"
	case goos != "windows" && st.Mode()&0o111 == 0:
		return "is not executable"
	}
	return ""
}

// runPTX is `gate ptx`: the PTX group alone, for a machine that has NVRTC and no GPU. It is stricter than the group
// inside `gate gpu`, where a file whose toolchain is absent is a counted skip: here every recorded toolchain is
// installed on purpose, so a file that was not verified fails the run.
func runPTX(w io.Writer, keepDir string) int {
	g := &gpuGate{w: w, emitted: map[string]bool{}, ptxKeepDir: keepDir}
	g.ptxGroup(liveInputs{g}) // a live source never errs
	if g.fail > 0 || g.skipped > 0 || g.pass == 0 {
		fmt.Fprintf(w, "\n%sgate ptx: RED%s  (%d failed, %d not verified)\n", red, off, g.fail, g.skipped)
		return 1
	}
	fmt.Fprintf(w, "\n%sgate ptx: GREEN%s\n", green, off)
	return 0
}

// ---- Metal groups ----

// buildCgoFree builds the METAL submodule's entrypoint (build.Dir), not the root one: the root ./cmd/serve builds no backend
// (cmd/serve/backendtag_guard_metal.go says so), so building it would succeed forever and the group would assert "Metal is
// dlopen'd via purego-objc" about a binary with no Metal in it. Then otool -L, the analogue of CUDA's ldd check, reads the
// binary's link-time libraries for gradeMetalCgoFree.
func buildCgoFree() cgoFreeInput {
	bin := filepath.Join(os.TempDir(), "gpu_gate_serve")
	build := exec.Command("go", "build", "-o", bin, "./cmd/serve")
	build.Dir = "metal"
	build.Env = append(os.Environ(), "CGO_ENABLED=0")
	if err := build.Run(); err != nil {
		return cgoFreeInput{}
	}
	defer os.Remove(bin)
	in := cgoFreeInput{Built: true}
	if b, err := exec.Command("otool", "-L", bin).Output(); err == nil {
		in.OtoolOK, in.Otool = true, string(b)
	}
	return in
}

// buildCUDACgoFree builds the CUDA SUBMODULE's entrypoint under -tags cuda with CGO_ENABLED=0, then reads ldd's output on the
// binary for gradeCUDACgoFree. The root ./cmd/serve is a DELIBERATE compile error under -tags cuda (it builds no backend, and
// failing loudly beats a CPU-only binary named as though it had CUDA), so a check pointed at it could never pass.
func buildCUDACgoFree() cgoFreeInput {
	bin := filepath.Join(os.TempDir(), "gpu_gate_serve")
	build := exec.Command("go", "build", "-tags", "cuda", "-o", bin, "./cmd/serve")
	build.Dir = "cuda"
	build.Env = append(os.Environ(), "CGO_ENABLED=0")
	out, err := build.CombinedOutput()
	if err != nil {
		return cgoFreeInput{BuildOut: string(out)}
	}
	defer os.Remove(bin)
	// ldd on a static binary, which CGO_ENABLED=0 makes, prints "not a dynamic executable" and exits 1. Its output is kept
	// either way, so the record shows which; only a run that exited 0 lists linked libraries for the grader.
	b, err := exec.Command("ldd", bin).CombinedOutput()
	return cgoFreeInput{Built: true, LddOK: err == nil, Ldd: string(b)}
}

// metalModel is the one checkpoint the Metal lifecycle and prefill gates need. Keeping both on the
// same file keeps the gate's asset footprint at one download.
func (g *gpuGate) metalModel() string {
	return filepath.Join(home(), "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
}

// gpuInputs is where the moved groups' cells and side inputs come from: the live gate runs them and records what it got
// (liveInputs), the replay reads them from a run record (recordedInputs), so both print the groups through the same code. A
// side input that is nil was not recorded, and its group is not regraded.
type gpuInputs interface {
	cell(c cell) (cellResult, *results, string, error)
	cgoFree() (*cgoFreeInput, error)
	webgpuProbe() *webgpuProbeRecord
	parityFreeVRAM(cr cellResult, out string) string
	ptx() (*ptxInput, error)
	heavy() (*heavyInput, error)
	heavyCell(i int, h heavyGroup) (cellResult, *results, string, procPeak, error)
	streamCell(c cell) (cellResult, *results, string, error) // a cell whose results print as they arrive
	heavySecs() int
	cleanGPU() (*cleanGPUInput, error)
	repo() (*repoInput, error)
	ciStep(r ciRow) (ciStepResult, error) // the repo group's next planned row
}

// liveInputs runs each cell and side input now, keeping the side inputs for the run record.
type liveInputs struct{ g *gpuGate }

func (l liveInputs) cell(c cell) (cellResult, *results, string, error) {
	res, cr, out := l.g.run(c, false)
	return cr, res, out, nil
}

func (l liveInputs) cleanGPU() (*cleanGPUInput, error) {
	in := readCleanGPU(l.g.backend)
	l.g.runIn.CleanGPU = &in
	return &in, nil
}

func (l liveInputs) repo() (*repoInput, error) {
	in := readRepo()
	l.g.runIn.Repo = &in
	return &in, nil
}

func (l liveInputs) ciStep(r ciRow) (ciStepResult, error) {
	res := runCIStep(r)
	l.g.runIn.Repo.Steps = append(l.g.runIn.Repo.Steps, res)
	return res, nil
}

func (l liveInputs) cgoFree() (*cgoFreeInput, error) {
	var in cgoFreeInput
	if l.g.backend == "cuda" {
		in = buildCUDACgoFree()
	} else {
		in = buildCgoFree()
	}
	l.g.runIn.CgoFree = &in
	return &in, nil
}

// parityFreeVRAM reads the card's free memory after a failed parity cell whose text holds a cosine of exactly zero, for
// gradeCUDAParity's note, and keeps it for the run record; it reads nothing otherwise.
func (l liveInputs) parityFreeVRAM(cr cellResult, out string) string {
	if cr.RC == 0 || !parityZeroCosine(out) {
		return ""
	}
	if b, err := exec.Command("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader").Output(); err == nil {
		l.g.runIn.ParityFreeVRAM = strings.TrimSpace(string(b))
	}
	return l.g.runIn.ParityFreeVRAM
}

func (l liveInputs) ptx() (*ptxInput, error) {
	in := readPTX(l.g.ptxKeepDir)
	l.g.runIn.PTX = &in
	return &in, nil
}

func (l liveInputs) heavy() (*heavyInput, error) {
	l.g.heavyT0 = time.Now()
	in := l.g.readHeavy()
	l.g.runIn.Heavy = &in
	return &in, nil
}

func (l liveInputs) heavyCell(_ int, h heavyGroup) (cellResult, *results, string, procPeak, error) {
	res, cr, out, peak := l.g.runHeavyGroup(h)
	l.g.runIn.Heavy.Peaks = append(l.g.runIn.Heavy.Peaks, peak)
	return cr, res, out, peak, nil
}

func (l liveInputs) streamCell(c cell) (cellResult, *results, string, error) {
	res, cr, out := l.g.run(c, true)
	return cr, res, out, nil
}

func (l liveInputs) heavySecs() int {
	l.g.runIn.Heavy.Secs = int(time.Since(l.g.heavyT0).Seconds())
	return l.g.runIn.Heavy.Secs
}

func (l liveInputs) webgpuProbe() *webgpuProbeRecord { return l.g.runIn.WebGPU }

// metalGroups runs the Metal backend's groups in the gate's order: suite, parity (the tagged tier the suite cannot see),
// cgofree, lifecycle, prefill.
func (g *gpuGate) metalGroups(src gpuInputs) error {
	groups := metalCellGroups(g.runIn.MetalModel)
	for _, step := range []func() error{
		func() error { return g.metalCell(src, groups[0]) },
		func() error { return g.metalCell(src, groups[1]) },
		func() error { return g.metalCgoFree(src) },
		func() error { return g.metalCell(src, groups[2]) },
		func() error { return g.metalCell(src, groups[3]) },
	} {
		if err := step(); err != nil {
			return err
		}
	}
	return nil
}

// metalCell runs one Metal cell group and prints what gradeMetalCell decides. A group that needs the checkpoint runs no cell
// when it is absent. The lifecycle group runs WITHOUT -short because it loads real models; it covers the sequential sawtooth
// and a second model alive, the case that made CUDA's first fix look correct when it was not. The prefill group gates
// PrefillLast, the f16 simdgroup_matrix TTFT path, on a real checkpoint: it once emitted NaN logits at every prompt length.
func (g *gpuGate) metalCell(src gpuInputs, s metalCellGroup) error {
	g.grp(s.Group)
	g.hdr(s.Header)
	if s.NeedsModel && !g.runIn.MetalModelPresent {
		g.apply(gradeMetalCell(s, g.runIn.MetalModel, false, cellResult{}, ""))
		return nil
	}
	cr, _, out, err := src.cell(s.Cell)
	if err != nil {
		return err
	}
	g.apply(gradeMetalCell(s, g.runIn.MetalModel, g.runIn.MetalModelPresent, cr, out))
	return nil
}

// metalCgoFree prints gradeMetalCgoFree's verdict on the build and its linkage.
func (g *gpuGate) metalCgoFree(src gpuInputs) error {
	in, err := src.cgoFree()
	if err != nil || in == nil {
		return err
	}
	g.grp("cgofree")
	g.hdr("3. cgo-free")
	g.apply(gradeMetalCgoFree(*in))
	return nil
}

// webgpuGroup runs group W whenever an adapter is detected, whatever the primary backend: the probe's verdict, then the
// suite cell, then the resident-parity cell when the suite passed. The parity gates (qwen3.5 DeltaNet, granite/nemotron
// Mamba-2) need GOINFER_DNET_PARITY and GOINFER_SSM_PARITY, which the cell sets itself, as every group here does.
func (g *gpuGate) webgpuGroup(src gpuInputs) error {
	p := src.webgpuProbe()
	if p == nil {
		return nil
	}
	g.grp("webgpu")
	g.hdr(webgpuHeader)
	if v, run := gradeWebGPUProbe(*p); !run {
		g.apply(v)
		return nil
	}
	cr, _, out, err := src.cell(webgpuSuiteCell)
	if err != nil {
		return err
	}
	v := gradeWebGPUSuite(p.Backend, cr, out)
	g.apply(v)
	if v.State != groupPass {
		return nil
	}
	cr, _, out, err = src.cell(webgpuParityCell)
	if err != nil {
		return err
	}
	g.apply(gradeWebGPUParity(cr, out))
	return nil
}

// apply prints a group's verdict: its Pre lines, its line, its detail or evidence lines, then adds its notes. A pass counts
// toward ran unless Uncounted, and so does a verdict that says it ran.
func (g *gpuGate) apply(v groupVerdict) {
	for _, ln := range v.Pre {
		fmt.Fprintln(g.w, ln)
	}
	if (v.State == groupPass && !v.Uncounted) || v.Ran {
		g.ran++
	}
	switch v.State {
	case groupPass:
		g.ok("%s", v.Line)
	case groupFail:
		g.bad("%s", v.Line)
	case groupSkip:
		g.skip("%s", v.Line)
	}
	for _, ln := range v.Lines {
		fmt.Fprintln(g.w, ln)
	}
	for _, n := range v.Notes {
		g.note(n)
	}
}

// repoHygiene runs the checks CI runs, DERIVED from .github/workflows/ci.yml by scripts/ci_checks.py
// rather than listed by hand, so a check CI gains appears here with no edit to this file. A hand-written
// list is a subset of CI's, and CI then goes red on something this gate passed.
func (g *gpuGate) repoGroup(src gpuInputs) error {
	in, err := src.repo()
	if err != nil || in == nil {
		return err
	}
	g.grp("repo")
	g.hdr("5. repo hygiene (derived from .github/workflows/ci.yml)")
	// The queue's citations, commit AND path:line: a state document is cited without being re-derived, so a wrong reference in
	// it propagates with more confidence than the same error in conversation.
	g.apply(gradeRepoLint(in.LintOK, in.LintOut))
	if v, ok := gradeCIDerivation(*in); !ok {
		g.apply(v)
		return nil
	}
	plan, skipped, other := ciPlan(in.Rows, in.Darwin)
	ok, bad := 0, 0
	for _, r := range plan {
		var res ciStepResult
		if r.Runner == "" {
			if res, err = src.ciStep(r); err != nil {
				return err
			}
			if res.OK {
				ok++
			} else {
				bad++
			}
		}
		if v, has := gradeCIStep(r, res); has {
			g.apply(v)
		}
	}
	for _, v := range gradeCISummary(skipped, other, ok, bad) {
		g.apply(v)
	}
	return nil
}

// readRepo runs the citation lint and derives CI's step list, and reads whether this host is Darwin.
func readRepo() repoInput {
	var in repoInput
	out, err := exec.Command("python3", "scripts/queue_citation_lint.py").CombinedOutput()
	in.LintOK, in.LintOut = err == nil, string(out)
	rows, err := exec.Command("python3", "scripts/ci_checks.py").Output()
	in.RowsOK, in.Rows = err == nil, string(rows)
	if ee, ok := err.(*exec.ExitError); ok {
		in.RowsErr = strings.ReplaceAll(strings.TrimSpace(string(ee.Stderr)), "\n", " ")
	} else if err != nil {
		in.RowsErr = err.Error()
	}
	if b, e := exec.Command("uname", "-s").Output(); e == nil && strings.TrimSpace(string(b)) == "Darwin" {
		in.Darwin = true
	}
	return in
}

// runCIStep runs one CI step. The ENVIRONMENT is part of the check: CI's root job has no go.work, so the module-boundary
// guard sees the root module graph in isolation, and a developer box with a committed go.work would report a false red.
// The override comes from whether the job sets up a workspace ("-" means none), not from here.
func runCIStep(r ciRow) ciStepResult {
	cmd := exec.Command("bash", "-c", unescapeCI(r.Cmd))
	cmd.Env = withGoBin(os.Environ())
	if r.Env != "-" && r.Env != "" {
		cmd.Env = append(cmd.Env, r.Env)
	}
	res := ciStepResult{Job: r.Job, Name: r.Name, OK: true}
	if out, err := cmd.CombinedOutput(); err != nil {
		res.OK, res.Out = false, string(out)
	}
	return res
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
// verdict prints the verdict block, writes the run's record (gpu_record.go) and returns the exit code.
func (g *gpuGate) verdict() int {
	code, label := g.verdictLabel()
	g.writeRecord(label)
	return code
}

// verdictLabel is the verdict block: the exit code and the one-word outcome the record carries (PASS, FAIL, INCONCLUSIVE for a green run on a dirty
// tree, NO_GATE, VACUOUS or EMPTY for a run that vouches for nothing).
func (g *gpuGate) verdictLabel() (int, string) {
	in := g.runIn.Verdict
	if in == nil {
		in = g.verdictInputs()
	}
	return g.gradeVerdict(*in)
}

// verdictInputs gathers what the verdict block prints beyond the tally: the host, the dirty tree's first lines, the time.
func (g *gpuGate) verdictInputs() *gpuVerdictInputs {
	in := gpuVerdictInputs{Backend: g.backend, Commit: g.commit, Host: "?", Dirty: g.dirty,
		Now: time.Now().UTC().Format("2006-01-02T15:04:05Z")}
	if b, err := exec.Command("uname", "-s").Output(); err == nil {
		in.Host = strings.TrimSpace(string(b))
	}
	if g.dirty {
		if b, err := exec.Command("git", "status", "--porcelain").Output(); err == nil {
			for i, ln := range strings.Split(strings.TrimRight(string(b), "\n"), "\n") {
				if i >= 10 {
					break
				}
				in.DirtyLines = append(in.DirtyLines, ln)
			}
		}
	}
	return &in
}

// gradeVerdict is gradeGPU on the tally so far; reconciliation failures belong to no group.
func (g *gpuGate) gradeVerdict(in gpuVerdictInputs) (int, string) {
	g.cur = ""
	t := g.tally()
	code, label := gradeGPU(g.w, &t, in)
	g.fail = t.Fail
	return code, label
}

// tally is the check groups' result so far, as gradeGPU and the record read it.
func (g *gpuGate) tally() gpuTally {
	return gpuTally{Expect: g.expect, Emitted: g.emitted, Pass: g.pass, Skip: g.skipped, Fail: g.fail, Ran: g.ran,
		Vacuous: g.vacuousCells, Empty: g.emptyCells, Notes: g.notes}
}
