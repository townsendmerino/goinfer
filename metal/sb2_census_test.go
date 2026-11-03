//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem"
	"github.com/townsendmerino/goinfer/internal/memgate"
	"github.com/townsendmerino/goinfer/internal/sb2census"
)

// Layer B-metal of the admission and routing census (docs/tasks/task-subsystems-2026-10.md, "SB2 step 2"): Metal's decisions
// before it allocates (residentPreBuild: the feature taxonomy, an explicit context cap, the fit choice and the memory guard with
// its context shrink), on layer D's fixtures and option cells, once per recorded machine with a Metal budget. A fake backend
// named metal takes the decoder's real admission; where that admits, it runs residentPreBuild against the profile's readings,
// records the answer and declines, so no device is opened. A cell the decoder declines is layer D's line and is not repeated.

var metalCensusUpdate = flag.Bool("sb2census.update", false, "rewrite this architecture's testdata/sb2_metal_census.<goarch>.txt.gz")

// censusBackend is the fake: named metal, it records residentPreBuild's answer for what the decoder admitted, and declines.
type censusBackend struct {
	decoder.Backend
	mu      sync.Mutex
	reached bool
	answer  string
}

func (b *censusBackend) Name() string { return "metal" }

// SelfTestEligible keeps the decoder's startup self-test (Metal kernels on the device) out of the census: that is a device decision.
func (b *censusBackend) SelfTestEligible() (bool, string) { return false, "admission census" }

func (b *censusBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	fit, from, err := residentPreBuild(m)
	ans := ""
	if err != nil {
		ans = "decline " + err.Error()
	} else {
		ctx, _ := resolveMetalCtxCap(m)
		ans = fmt.Sprintf("ok ctx=%d fit=%d from=%d why=%s", ctx, fit.Ctx, from, fit.Why)
	}
	metalFitChoice.Delete(m)
	metalCtxCeiling.Delete(m)
	b.mu.Lock()
	b.reached, b.answer = true, ans
	b.mu.Unlock()
	return nil, false, decoder.DeclineResident("admission census: pre-build decisions only")
}

func (b *censusBackend) take() (bool, string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	r, a := b.reached, b.answer
	b.reached, b.answer = false, ""
	return r, a
}

// censusProfile is a recorded machine (testdata/machines/*.json) with a Metal budget.
type censusProfile struct {
	name                       string
	physical, available, metal int64
}

func readCensusProfiles(t *testing.T) []censusProfile {
	t.Helper()
	paths, err := filepath.Glob("../testdata/machines/*.json")
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(paths)
	var out []censusProfile
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var j struct {
			Physical  int64  `json:"physical_bytes"`
			Available int64  `json:"available_bytes"`
			Metal     *int64 `json:"metal_budget_bytes"`
		}
		if err := json.Unmarshal(b, &j); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		if j.Metal == nil {
			continue
		}
		out = append(out, censusProfile{strings.TrimSuffix(filepath.Base(p), ".json"), j.Physical, j.Available, *j.Metal})
	}
	if len(out) == 0 {
		t.Fatal("no recorded machine has a Metal budget: the census would price nothing")
	}
	return out
}

// usingProfile sets every machine reading residentPreBuild takes from p, and returns the restore.
func usingProfile(p censusProfile) (restore func()) {
	live, ws, ram := metalLiveAvailable, metalWorkingSetCap, physicalRAM
	metalLiveAvailable = func() int64 { return p.available }
	metalWorkingSetCap = func() int64 { return p.metal }
	physicalRAM = func() (uint64, error) {
		if p.physical <= 0 {
			return 0, errRAMUnknown
		}
		return uint64(p.physical), nil
	}
	r1 := decoder.SetMemoryProbeForTest("metal", p.metal, true)
	r2 := decoder.SetHostRAMAvailableForTest(p.available)
	return func() {
		r2()
		r1()
		metalLiveAvailable, metalWorkingSetCap, physicalRAM = live, ws, ram
	}
}

// censusReg is the name the fake is registered under, once per process (registerCensusBackend).
const censusReg = "sb2-census-metal"

var (
	censusBE       = &censusBackend{}
	censusRegister sync.Once
)

func registerCensusBackend() {
	censusRegister.Do(func() {
		decoder.RegisterBackend(censusReg, func() (decoder.Backend, error) {
			cpu, err := decoder.NewBackend("cpu")
			if err != nil {
				return nil, err
			}
			censusBE.Backend = cpu
			return censusBE, nil
		})
	})
}

// buildMetalCensus returns one line per (cell, profile) that reached Metal's build. Every cell layer D built resident on metal
// must reach it under every profile.
func buildMetalCensus(t *testing.T) []string {
	t.Helper()
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	registerCensusBackend()
	be, reg := censusBE, censusReg
	be.take() // nothing a previous build left behind decides a line here
	d, err := sb2census.ReadLayerD("..", "metal")
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Fixtures) < 20 {
		t.Fatalf("only %d fixtures in layer D's golden: a census that read little covers little", len(d.Fixtures))
	}
	var lines []string
	for _, prof := range readCensusProfiles(t) {
		lines = append(lines, censusProfileLines(t, be, reg, d, prof)...)
	}
	return lines
}

// censusProfileLines is one profile's lines. The profile's readings are restored on every way out, a panic or a fatal
// check included, so they never reach another test in the package.
func censusProfileLines(t *testing.T, be *censusBackend, reg string, d sb2census.LayerD, prof censusProfile) []string {
	t.Helper()
	restore := usingProfile(prof)
	defer restore()
	var lines []string
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("..", f)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range sb2census.Cells {
			o := decoder.Options{Backend: reg, Quant: c.Quant, ActQuantGroup: c.ActGroup, KVPrecision: c.KVPrec, KVQuant: c.KVQuant}
			if c.SSM != "" {
				o.Knobs = &decoder.Knobs{"GOINFER_SSM_RESIDENT": c.SSM}
			}
			m, err := decoder.Load(path, o)
			reached, ans := be.take()
			if err == nil {
				m.Close()
			}
			if reached {
				lines = append(lines, fmt.Sprintf("B %s %s metal %s %q", f, c.Name, prof.name, sb2census.Clean(ans)))
			} else if d.Resident[f+" "+c.Name] {
				t.Errorf("%s %s on %s: layer D builds it resident on metal, but Metal's build was not reached", f, c.Name, prof.name)
			}
		}
	}
	return lines
}

// TestBackendAdmissionCensus_metal holds layer B-metal to testdata/sb2_metal_census.<goarch>.txt.gz, line for line, built twice.
func TestBackendAdmissionCensus_metal(t *testing.T) {
	start := time.Now()
	a, b := buildMetalCensus(t), buildMetalCensus(t)
	if strings.Join(a, "\n") != strings.Join(b, "\n") {
		t.Fatalf("two builds in one process differ")
	}
	t.Logf("%d census lines in %s", len(a), time.Since(start).Round(time.Millisecond))
	golden := sb2census.Golden("testdata", "sb2_metal_census")
	if *metalCensusUpdate {
		if err := sb2census.Write(golden, a); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", golden)
		return
	}
	diffs, err := sb2census.Diff(golden, a, 5)
	if err != nil {
		for _, l := range a {
			fmt.Println("SB2METALCENSUS " + l)
		}
		t.Fatalf("%v: the lines above (prefixed SB2METALCENSUS) are this architecture's census", err)
	}
	if len(diffs) > 0 {
		t.Errorf("the metal admission census differs from %s in %d line(s):\n%s", golden, len(diffs), strings.Join(diffs[:min(len(diffs), 30)], "\n"))
	}
}

// TestBackendAdmissionCensus_metal_plant shows the census can go red. No line takes one of residentPreBuild's declines (the
// decoder declines a missing feature first, no cell pins -ctx, and no tiny fixture reaches the memory guard), so the plant is in
// the decision that varies: with the Metal budget planted unknown, every line whose context the fit choice raised must differ
// from the golden, and no other line may.
func TestBackendAdmissionCensus_metal_plant(t *testing.T) {
	golden := sb2census.Golden("testdata", "sb2_metal_census")
	clean := buildMetalCensus(t)
	const untouched = "why=the default"
	wantRed := 0
	for _, l := range clean {
		if !strings.Contains(l, untouched) {
			wantRed++
		}
	}
	if wantRed == 0 {
		t.Fatal("no line's context was raised by the fit choice: the plant has nothing to change")
	}
	defer func(v bool) { metalFitFreeUnknownForTest = v }(metalFitFreeUnknownForTest)
	metalFitFreeUnknownForTest = true
	diffs, err := sb2census.Diff(golden, buildMetalCensus(t), 5)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range diffs {
		if strings.Contains(d, untouched) && !strings.Contains(d, "fit:") {
			t.Errorf("planted unknown budget: a line the fit choice did not raise changed: %s", d)
		}
	}
	if len(diffs) != wantRed {
		t.Fatalf("planted unknown budget: %d line(s) differ from the golden, want %d", len(diffs), wantRed)
	}
	t.Logf("planted unknown budget: %d of %d lines red, as expected", len(diffs), len(clean))
}

// TestCensusProfile_restores holds usingProfile's contract: the profile's readings are in force until restore, restore puts
// back every machine reading the census replaced, even on the way out of a panic, and a profile with no physical RAM figure
// reads as unknown, as the machine's own reading would.
func TestCensusProfile_restores(t *testing.T) {
	ram0, _ := physicalRAM()
	live0, ws0 := metalLiveAvailable(), metalWorkingSetCap()
	func() {
		defer func() { _ = recover() }()
		restore := usingProfile(censusProfile{name: "x", physical: 0, available: 5, metal: 7})
		defer restore()
		if _, err := physicalRAM(); err != errRAMUnknown {
			t.Errorf("a profile with no RAM figure: physicalRAM err %v, want errRAMUnknown", err)
		}
		if metalLiveAvailable() != 5 || metalWorkingSetCap() != 7 {
			t.Errorf("the profile's readings are not in force")
		}
		panic("a load that panics")
	}()
	if ram, _ := physicalRAM(); ram != ram0 || metalLiveAvailable() != live0 || metalWorkingSetCap() != ws0 {
		t.Errorf("after a panic, the machine's readings were not restored")
	}
}

// metalRounding is Metal's allocator: a buffer's allocatedSize rounds its length up to a 16 KiB page at most.
var metalRounding = memgate.PerLease("Metal's 16 KiB sub-allocation", 16<<10)

// metalMemGaps is Metal's table of known reds (MM7). It may only shrink: the fix that removes a row lowers metalMemGapsPin. Each
// row is bounded, so a second gap on the same line stays red.
var metalMemGaps = []memgate.Gap{
	{Row: "*", Line: mem.LineState, What: "the recurrent checkpoint leases on the state line, and the plan prices one slot's state and no checkpoint",
		Item: "MM8 Metal, recurrent checkpoint", Bytes: func(r memgate.Reading) int64 { return r.Facts["checkpoint"] }},
	{Row: "*", Line: mem.LineDense, What: "the plan's dense figure sums weight matrices: norms, biases, sinks, rotation tables and the greedy chain's embedding table are leased on dense and priced nowhere",
		Item: "MM8 Metal, dense vectors and the chain table", Bytes: func(r memgate.Reading) int64 { return r.Facts["tables"] + r.Facts["chain"] }},
	{Row: "testdata/gemma4-moe-kv-tiny32 *", Line: mem.LineDense, What: "Gemma 4's MoE leases 21,128 bytes on dense past the row above: not diagnosed",
		Item: "MM8 Metal, Gemma 4 MoE dense", Bytes: func(r memgate.Reading) int64 { return r.Facts["tables"] + r.Facts["chain"] + 21128 }},
	{Row: "testdata/gptoss_tiny.gguf *", Line: mem.LineExperts, What: "gpt-oss's experts lease 8,192 bytes over the plan's expert figure: not diagnosed",
		Item: "MM8 Metal, gpt-oss experts", Bytes: func(memgate.Reading) int64 { return 8192 }},
}

const metalMemGapsPin = 4

// metalGateRow loads fixture path at options o, runs it, and returns the row: the build's ledger after the build and after a short
// generation, the plan for the same load, and the ledger after Close. ok is false when Metal's own build declines, which has no
// allocation to hold (the census's rows record admission, not the build).
func metalGateRow(t *testing.T, row, path string, o decoder.Options, kvI8 bool) (memgate.Reading, bool) {
	t.Helper()
	m, err := decoder.Load(path, o)
	if err != nil {
		return memgate.Reading{}, false
	}
	closed := false
	defer func() {
		if !closed {
			m.Close()
		}
	}()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		return memgate.Reading{}, false
	}
	r := rf.(*metalResident).r
	l := metalLedgerOf(r.d)
	if l == nil {
		t.Errorf("%s: the build's device has no ledger", row)
		return memgate.Reading{}, false
	}
	read := memgate.Reading{Row: row, KVSlots: r.kvSlotCount(), Facts: map[string]int64{"checkpoint": r.ckptBytes, "tables": metalDenseTableBytes(r)}}
	read.Plan = m.Plan("metal", math.MaxInt64/4, decoder.PlanRequest{Ctx: r.ctxCap, CtxPinned: true, KVI8: kvI8})
	read.Open = append(read.Open, memgate.Snapshot{Label: "built", Usage: l.Usage()})
	ch, g := m.Generate(context.Background(), []int{1, 2, 3, 4, 5}, 4, decoder.SamplingParams{})
	for range ch {
	}
	if err := g.Err(); err != nil {
		t.Logf("%s: generation: %v; the row holds the build alone", row, err)
	} else {
		u := l.Usage()
		read.Open = append(read.Open, memgate.Snapshot{Label: "generated", Usage: u})
		read.Facts["chain"] = u.ByLine[mem.LineDense] - read.Open[0].Usage.ByLine[mem.LineDense] // the greedy chain's table, built on first use
	}
	// Scratch has no plan line: it is the margin the Metal fit leaves under its budget, and what the build charges for the step.
	scratch := int64(mem.MetalFitMargin)
	if r.batch != nil {
		scratch += r.stepScratchBytes()
	}
	read.Named = map[mem.LineID]int64{mem.LineScratch: scratch}
	closed = true
	m.Close()
	read.Closed = l.Usage()
	return read, true
}

// TestMemGate_metalPlantGoesRed: the gate reads the real ledger. llama-tiny's dense line is under plan after the build; a mebibyte
// charged over its request on one dense buffer lands on the ledger and is red, past what the allocator's rounding would forgive.
func TestMemGate_metalPlantGoesRed(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const fx, plant = "../testdata/llama-tiny", int64(1 << 20)
	o := decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256}
	builtRed := func(r memgate.Reading) bool {
		for _, f := range memgate.Check(r, metalRounding) {
			if f.Line == mem.LineDense && strings.HasPrefix(f.What, "built:") {
				return true
			}
		}
		return false
	}
	clean, ok := metalGateRow(t, "clean", fx, o, false)
	if !ok {
		t.Fatal("llama-tiny did not build resident")
	}
	if builtRed(clean) {
		t.Fatalf("the unplanted build is already over plan on dense: %v", memgate.Check(clean, metalRounding))
	}
	ledgerChargeDeltaForTest, ledgerChargeLineForTest = plant, mem.LineDense
	defer func() { ledgerChargeDeltaForTest, ledgerChargeLineForTest = 0, "" }()
	planted, ok := metalGateRow(t, "planted", fx, o, false)
	if !ok {
		t.Fatal("llama-tiny did not build resident")
	}
	if got, want := planted.Open[0].Usage.ByLine[mem.LineDense], clean.Open[0].Usage.ByLine[mem.LineDense]+plant; got != want {
		t.Fatalf("the plant did not land: dense leased %d, want %d", got, want)
	}
	if !builtRed(planted) {
		t.Error("a mebibyte over plan on the dense line is not red")
	}
}

// metalDenseTableBytes is what the build leases on the dense line that the plan's dense figure, a sum of weight matrices, does not
// count: every norm's weights, the biases, the attention sinks, the layer scale, the rotation tables and the m-RoPE axis map.
func metalDenseTableBytes(r *resident) int64 {
	var n int64
	add := func(bufs ...Buffer) {
		for _, b := range bufs {
			if b != (Buffer{}) {
				n += int64(b.Len()) * 4
			}
		}
	}
	for l := range r.layers {
		L := &r.layers[l]
		add(L.preNorm, L.postNorm, L.preNormBias, L.postNormBias, L.postAttnNorm, L.postMLPNorm, L.postPLENorm, L.qNorm, L.kNorm,
			L.oBias, L.upBias, L.downBias, L.qkvBias, L.attnSinks, L.uLayerScalar, L.invf, L.mscale)
	}
	add(r.finalNorm, r.finalNormBias, r.invf, r.vNormUnit, r.mropeAxis)
	return n
}

// TestMemGate_metal is MM7's gate on Metal (docs/tasks/task-subsystems-2026-10.md): every census fixture at every option cell whose
// build goes resident, and each at int4 with two KV slots and the step's blocks on, holds what it leased on each line to the plan
// for the same load. It runs every row in one test, since a filtered run would see known gaps go quiet.
func TestMemGate_metal(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	d, err := sb2census.ReadLayerD("..", "metal")
	if err != nil {
		t.Fatal(err)
	}
	g := memgate.Gate{Allowance: metalRounding, Gaps: metalMemGaps, GapsPin: metalMemGapsPin}
	built, declined := 0, 0
	run := func(row, path string, o decoder.Options, kvI8 bool) {
		o.Backend, o.ResidentContext = "metal", 256
		read, ok := metalGateRow(t, row, path, o, kvI8)
		if !ok {
			declined++
			return
		}
		built++
		g.Check(t, read)
	}
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("..", f)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range sb2census.Cells {
			if !d.Resident[f+" "+c.Name] {
				continue
			}
			o := decoder.Options{Quant: c.Quant, ActQuantGroup: c.ActGroup, KVPrecision: c.KVPrec, KVQuant: c.KVQuant}
			if c.SSM != "" {
				o.Knobs = &decoder.Knobs{"GOINFER_SSM_RESIDENT": c.SSM}
			}
			run(f+" "+c.Name, path, o, c.KVPrec == "i8")
		}
		if d.Resident[f+" int4"] {
			run(f+" int4/slots2", path, decoder.Options{Quant: "int4", ResidentKVSlots: 2}, false)
			run(f+" int4/slots2/blocks", path, decoder.Options{Quant: "int4", ResidentKVSlots: 2, ResidentMoEStep: true, ResidentRecurrentStep: true, ResidentGemma4Step: true}, false)
		}
	}
	t.Logf("%d builds held, %d options declined by Metal's own build", built, declined)
	if built < 50 {
		t.Errorf("only %d builds held: a gate that read little covers little", built)
	}
	g.Finish(t)
}
