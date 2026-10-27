package decoder

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/mem/probe"
)

// The admission and routing census, layer D (SB2, docs/tasks/task-subsystems-2026-10.md, "SB2 — the admission and routing
// census"): every admission and routing decision the decoder makes, recorded so the pure function that replaces them can be held to
// the same answers, line for line. Backend-side decisions (BuildResident's own declines) are later layers in the backend modules:
// here a fake resident backend named cuda, metal or webgpu builds whatever it is asked to, so a line records the decoder's decision.

var sb2CensusUpdate = flag.Bool("sb2census.update", false, "rewrite testdata/sb2_admission_census.txt.gz")

const sb2CensusGolden = "../testdata/sb2_admission_census.txt.gz"

// sb2Cell is one option set a fixture is loaded with.
type sb2Cell struct {
	name     string
	quant    string
	actGroup int
	ssm      string // the GOINFER_SSM_RESIDENT knob
}

var sb2Cells = []sb2Cell{
	{"int4", "int4", 0, ""}, {"int4/g32", "int4", 32, ""},
	{"int8int8", "int8int8", 0, ""}, {"int8int8/g32", "int8int8", 32, ""},
	{"f32", "f32", 0, ""}, {"f32/g32", "f32", 32, ""},
	{"int4/ssm", "int4", 0, "1"},
}

// sb2Fakes registers one fake resident backend per GPU backend name, under a name of its own, and returns the registry name for each.
var sb2Fakes = sync.OnceValue(func() map[string]string {
	out := map[string]string{}
	for _, name := range []string{"cuda", "metal", "webgpu"} {
		reg := "sb2-fake-" + name
		RegisterBackend(reg, func() (Backend, error) {
			cpu, err := NewBackend("cpu")
			if err != nil {
				return nil, err
			}
			return &namedResidencyBackend{Backend: cpu, name: name}, nil
		})
		out[name] = reg
	}
	return out
})

// sb2Clean makes a line independent of where the repository is checked out.
func sb2Clean(s string) string {
	if wd, err := os.Getwd(); err == nil {
		s = strings.ReplaceAll(s, filepath.Dir(wd), "<repo>")
	}
	return strings.ReplaceAll(s, "\n", " / ")
}

func buildSB2Census(t *testing.T) []string {
	t.Helper()
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "")
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	// Every machine reading the load path takes comes from this fake, so no reading of this machine enters a line.
	useProbe(t, &probe.Fake{PhysicalBytes: 64 << 30, AvailableBytes: 48 << 30})
	var lines []string

	// Model-free, per registered family: the runner-shape decline and every resident gate.
	keys := make([]string, 0, len(registry))
	for k := range registry {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, mt := range keys {
		cfg := representativeConfig(mt)
		if cfg == nil {
			t.Fatalf("model_type %q has no representativeConfig", mt)
		}
		arch, _, err := resolveArchitecture(cfg)
		if err != nil {
			t.Fatalf("model_type %q: %v", mt, err)
		}
		for _, be := range []string{"webgpu", "cuda", "metal"} {
			gates := func(actSafe bool) string {
				var gs []string
				for _, g := range residentGateReasons(arch, be, actSafe) {
					gs = append(gs, g.Kind+": "+g.Reason)
				}
				return strings.Join(gs, " | ")
			}
			lines = append(lines, fmt.Sprintf("F %s %s eligible=%v shape=%q gates=%q gates-per32=%q",
				mt, be, ResidentEligible(arch, be), arch.runnerShapeDecline(), gates(false), gates(true)))
		}
	}

	// Backend choice, per machine profile and linked set.
	for _, m := range sb2Machines(t) {
		probeFor := func(b string) (int64, bool) {
			switch {
			case b == "cuda" && m.CUDAFreeBytes != nil:
				return *m.CUDAFreeBytes, true
			case b == "metal" && m.MetalBudget != nil:
				return *m.MetalBudget, true
			}
			return 0, false
		}
		for _, linked := range [][]string{{"cpu"}, {"cpu", "cuda"}, {"cpu", "metal"}, {"cpu", "cuda", "metal"}, {"cpu", "webgpu"}} {
			c := autoBackend(linked, probeFor, m.GOARCH)
			lines = append(lines, fmt.Sprintf("A %s %s backend=%s skipped=%v reason=%q", m.Name, strings.Join(linked, "+"), c.Backend, c.Skipped, c.Reason))
		}
	}

	// Per tiny fixture, option cell and backend: the decoder's decision on a real Load.
	fakes := sb2Fakes()
	// Only committed fixtures: a checkout with local-only weights (the .safetensors ignore keeps many out of git) would otherwise
	// census more than CI does.
	out, err := exec.Command("git", "ls-files", "--", "..").Output()
	if err != nil {
		t.Skipf("the census needs git to tell committed fixtures from local ones: %v", err)
	}
	committed := map[string]bool{}
	for _, l := range strings.Split(string(out), "\n") {
		committed[filepath.Clean(l)] = true
	}
	var fixtures []censusFamily
	for _, f := range censusFamilies(t) {
		w := f.path
		if f.dir {
			ws, _ := filepath.Glob(filepath.Join(f.path, "*.safetensors"))
			sort.Strings(ws)
			w = ws[0]
		}
		if !committed[filepath.Clean(w)] { // ls-files prints paths relative to the package directory, as w is
			continue
		}
		if _, err := censusConfig(f.path, f.dir); err == nil {
			fixtures = append(fixtures, f)
		}
	}
	if len(fixtures) < 20 {
		t.Fatalf("only %d fixtures resolved a config: a census that read little covers little", len(fixtures))
	}
	for _, f := range fixtures {
		// A fixture that is not a decoder checkpoint (a tower, an encoder) is recorded once, without the list of supported
		// model types an unknown one's error carries (it grows with every family).
		if m, err := Load(f.path, Options{Backend: "cpu", Quant: "int4"}); err != nil {
			msg, _, _ := strings.Cut(sb2Clean(err.Error()), " (have:")
			lines = append(lines, fmt.Sprintf("X %s not-a-decoder=%q", f.name, msg))
			continue
		} else {
			m.Close()
		}
		for _, cell := range sb2Cells {
			for _, be := range []string{"cpu", "cuda", "metal", "webgpu"} {
				reg := be
				if be != "cpu" {
					reg = fakes[be]
				}
				o := Options{Backend: reg, Quant: cell.quant, ActQuantGroup: cell.actGroup}
				if cell.ssm != "" {
					o.Knobs = &Knobs{knobSSMResident: cell.ssm}
				}
				lines = append(lines, sb2LoadLine(f.name, cell.name, be, f.path, o))
			}
		}
	}
	return lines
}

// sb2Machine is a testdata/machines profile's fields backend choice reads.
type sb2Machine struct {
	Name          string `json:"name"`
	GOARCH        string `json:"goarch"`
	CUDAFreeBytes *int64 `json:"cuda_free_bytes"`
	MetalBudget   *int64 `json:"metal_budget_bytes"`
}

func sb2Machines(t *testing.T) []sb2Machine {
	t.Helper()
	paths, _ := filepath.Glob("../testdata/machines/*.json")
	sort.Strings(paths)
	var ms []sb2Machine
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var m sb2Machine
		if err := json.Unmarshal(b, &m); err != nil || m.Name == "" {
			t.Fatalf("%s: a profile needs a name (%v)", p, err)
		}
		ms = append(ms, m)
	}
	if len(ms) == 0 {
		t.Fatal("no machine profiles")
	}
	return ms
}

// sb2LoadLine loads one cell and records the decoder's decisions on it.
func sb2LoadLine(fixture, cell, be, path string, o Options) string {
	head := fmt.Sprintf("L %s %s %s", fixture, cell, be)
	m, err := Load(path, o)
	if err != nil {
		return head + " load-error=" + fmt.Sprintf("%q", sb2Clean(err.Error()))
	}
	defer m.Close()
	resident := m.resident != nil
	line := fmt.Sprintf("%s resident=%v decline=%q path=%q", head, resident, sb2Clean(m.resDecline), sb2Clean(m.DecodePath()))
	if be == "cpu" {
		batched, why := m.PrefillPath()
		elig := "ok"
		if e := m.cpuBatchModelEligible(); e != nil {
			elig = sb2Clean(e.Error())
		}
		var kv []string
		for _, r := range m.KVPrecisionReports() {
			kv = append(kv, fmt.Sprintf("%s:%s->%s(%s)", r.Cache, r.Requested, r.Delivered, sb2Clean(r.Reason)))
		}
		line += fmt.Sprintf(" prefill=%v/%q batchN2=%v cpubatch=%q concurrent=%v specrollback=%v kv=%q",
			batched, sb2Clean(why), m.canBatchN(2), elig, m.CPUConcurrentSafe(), m.specRollbackSafe(), strings.Join(kv, " | "))
	}
	return line
}

// TestAdmissionCensus holds layer D of the admission and routing census to testdata/sb2_admission_census.txt.gz, line for line.
// It is built twice in one process and must agree with itself. -sb2census.update rewrites the golden.
func TestAdmissionCensus(t *testing.T) {
	slowTest(t)
	start := time.Now()
	a := buildSB2Census(t)
	b := buildSB2Census(t)
	if strings.Join(a, "\n") != strings.Join(b, "\n") {
		for i := range min(len(a), len(b)) {
			if a[i] != b[i] {
				t.Fatalf("two builds in one process differ at line %d:\n  first:  %s\n  second: %s", i, a[i], b[i])
			}
		}
		t.Fatalf("two builds in one process differ in length: %d and %d", len(a), len(b))
	}
	text := strings.Join(a, "\n") + "\n"
	t.Logf("%d census lines in %s", len(a), time.Since(start).Round(time.Millisecond))
	if *sb2CensusUpdate {
		var buf bytes.Buffer
		zw, _ := gzip.NewWriterLevel(&buf, gzip.BestCompression)
		zw.ModTime = time.Time{}
		if _, err := zw.Write([]byte(text)); err != nil {
			t.Fatal(err)
		}
		if err := zw.Close(); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(sb2CensusGolden, buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", sb2CensusGolden)
		return
	}
	want, err := readSB2Golden()
	if err != nil {
		t.Fatalf("%v (write it with -sb2census.update)", err)
	}
	if want == text {
		return
	}
	wl, gl := strings.Split(want, "\n"), strings.Split(text, "\n")
	ws, gs := map[string]bool{}, map[string]bool{}
	for _, l := range wl {
		ws[l] = true
	}
	for _, l := range gl {
		gs[l] = true
	}
	var diffs []string
	for _, l := range wl {
		if !gs[l] {
			diffs = append(diffs, "- "+l)
		}
	}
	for _, l := range gl {
		if !ws[l] {
			diffs = append(diffs, "+ "+l)
		}
	}
	t.Errorf("the admission census differs from the golden in %d line(s):\n%s", len(diffs), strings.Join(diffs[:min(len(diffs), 40)], "\n"))
}

func readSB2Golden() (string, error) {
	f, err := os.Open(sb2CensusGolden)
	if err != nil {
		return "", err
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return "", err
	}
	b, err := io.ReadAll(zr)
	if err != nil && !errors.Is(err, io.EOF) {
		return "", err
	}
	return string(b), nil
}
