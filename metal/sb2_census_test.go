//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem"
	"github.com/townsendmerino/goinfer/internal/memgate"
	"github.com/townsendmerino/goinfer/internal/program"
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
// row is bounded by the build's own shape, so a second gap on the same line stays red.
var metalMemGaps = []memgate.Gap{
	{Row: "*", Line: mem.LineScratch, What: "decode's own scratch and uniforms lease on the scratch line, and nothing prices them",
		Item: "MM5, decode's buffer list", Bytes: func(f map[string]int64) int64 { return f["decode"] }},
	{Row: "*", Line: mem.LineState, What: "the recurrent checkpoint leases on the state line, and the plan prices one slot's state and no checkpoint",
		Item: "MM8 Metal, recurrent checkpoint", Bytes: func(f map[string]int64) int64 { return f["checkpoint"] }},
	{Row: "*", Line: mem.LineDense, What: "the plan's dense figure sums weight matrices: norms, biases, sinks and rotation tables are leased on dense and priced nowhere",
		Item: "MM8 Metal, dense vectors", Bytes: func(f map[string]int64) int64 { return f["tables"] }},
	{Row: "testdata/gemma4-moe-kv-tiny32 *", Line: mem.LineDense, What: "Gemma 4's MoE leases 21,128 bytes on dense past the row above: not diagnosed",
		Item: "MM8 Metal, Gemma 4 MoE dense", Bytes: func(map[string]int64) int64 { return 21128 }},
	{Row: "testdata/gptoss_tiny.gguf *", Line: mem.LineExperts, What: "gpt-oss's experts lease 8,192 bytes over the plan's expert figure: not diagnosed",
		Item: "MM8 Metal, gpt-oss experts", Bytes: func(map[string]int64) int64 { return 8192 }},
}

const metalMemGapsPin = 5

// metalGatePlantForTest, when set, runs on a row's resident after its generation and before the ledger is read; what it returns
// runs after that read, before Close.
var metalGatePlantForTest func(r *resident) (free func())

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
	read := memgate.Reading{Row: row, KVSlots: r.kvSlotCount()}
	read.Plan = m.Plan("metal", math.MaxInt64/4, decoder.PlanRequest{Ctx: r.ctxCap, CtxPinned: true, KVI8: kvI8})
	facts := func() map[string]int64 {
		f := map[string]int64{"tables": metalDenseVectorBytes(m, r), "decode": metalDecodeScratchBytes(m, r)}
		if r.ckptBytes > 0 { // whether the load kept one is the build's; its size is one slot's list
			f["checkpoint"] = metalRecurrentSlotBytes(m)
		}
		return f
	}
	read.Open = append(read.Open, memgate.Snap("built", l, facts()))
	ch, g := m.Generate(context.Background(), []int{1, 2, 3, 4, 5}, 4, decoder.SamplingParams{})
	for range ch {
	}
	free := func() {}
	if metalGatePlantForTest != nil {
		free = metalGatePlantForTest(r)
	}
	if err := g.Err(); err != nil {
		t.Errorf("%s: generation: %v", row, err)
	}
	read.Open = append(read.Open, memgate.Snap("generated", l, facts()))
	free()
	// Scratch has no plan line: the one thing anything prices on it is the step, whose list the gate holds it to, before the price's
	// page rounding, which would hide what else leases there.
	var scratch int64
	if r.batch != nil {
		scratch = r.stepShape().listBytes()
	}
	// The reserve is the build's own answer read before the build: every input to it agrees, so it is the step's price where
	// batchIneligible admits it and 0 where not, on shapes no fixture has as well as on these.
	built, before := r.stepInputs(), metalStepInputs(m)
	if r.mla != nil { // MLA's slots are one allocation only above one slot, which a reading before the build cannot know; the step's tables refuse it
		before.oneKVAllocation = built.oneKVAllocation
	}
	if !reflect.DeepEqual(built, before) {
		t.Errorf("%s: what the step reads differs between the build and the model:\nbuilt  %+v\nbefore %+v", row, *built, *before)
	}
	var price int64
	why := r.batchIneligible()
	if why == "" {
		price = r.stepScratchBytes()
	}
	if reserve := metalStepScratchReserve(m); reserve != price {
		t.Errorf("%s: the reserve before the build is %d bytes, and the build prices the step at %d (batchIneligible %q)", row, reserve, price, why)
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

// TestMemGate_metalPlantsThroughTheTable: a buffer really allocated over plan during a row stays red through Metal's allowance and
// its table of known gaps. It is 4 MiB and 4 bytes, so Metal rounds it and the allowance has something to forgive; the fixtures are
// a dense model and a recurrent one, whose state gap is live. The buffer is freed before Close, so what is red is the plan.
func TestMemGate_metalPlantsThroughTheTable(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const floats, requested = 1<<20 + 1, 4 * (1<<20 + 1)
	g := memgate.Gate{Allowance: metalRounding, Gaps: metalMemGaps, GapsPin: metalMemGapsPin}
	for _, c := range []struct {
		fx   string
		line mem.LineID
	}{{"llama-tiny", mem.LineDense}, {"qwen35-tiny", mem.LineDense}, {"qwen35-tiny", mem.LineState}, {"llama-tiny", mem.LineScratch}} {
		t.Run(c.fx+"/"+string(c.line), func(t *testing.T) {
			planted := false
			metalGatePlantForTest = func(r *resident) func() {
				planted = true
				b := r.d.NewBufferLenOn(string(c.line), floats)
				return func() { r.d.ReleaseBuf(b) }
			}
			defer func() { metalGatePlantForTest = nil }()
			row := "testdata/" + c.fx + " int4"
			read, ok := metalGateRow(t, row, "../testdata/"+c.fx, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256}, false)
			if !ok || !planted {
				t.Fatalf("resident %v, plant ran %v", ok, planted)
			}
			rounded := false
			for _, l := range read.Open[1].Leases {
				rounded = rounded || l.Line == c.line && l.Requested == requested && l.Charged > l.Requested
			}
			if !rounded {
				t.Fatalf("the planted lease was not rounded: the allowance had nothing to forgive")
			}
			if read.Closed.Leases != 0 {
				t.Errorf("%d leases open after Close: the plant should have been freed", read.Closed.Leases)
			}
			red := false
			for _, f := range g.Unknown(read) {
				red = red || f.Line == c.line
			}
			if !red {
				t.Errorf("4 MiB leased on %q during a row is not red through the table", c.line)
			}
		})
	}
}

// TestStepBuffers_listIsWhatBuilds: the step's scratch is one list. Across buildBatch, the ledger gains exactly the list's buffers,
// to the byte and the lease, on each model shape the step takes, so a buffer allocated outside the list is red, and the price, which
// sums the same list, cannot drift from it. The q4k lane has no tiny fixture, so its row reads the 0.5B GGUF in ~/models and skips
// without it. Recurrent state is held the same way by TestRecurrentSlots_eachSlotPricesItsState.
func TestStepBuffers_listIsWhatBuilds(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	t.Cleanup(func() { metalMemoryCeilingForTest = 0 })
	home, _ := os.UserHomeDir()
	for _, c := range []struct {
		dir   string
		quant string
		o     decoder.Options
	}{
		{dir: "../testdata/llama-tiny", quant: "int4"},
		{dir: "../testdata/llama-tiny", quant: "int8int8"},
		{dir: "../testdata/mixtral-tiny", quant: "int4", o: decoder.Options{ResidentMoEStep: true}},
		{dir: "../testdata/qwen35-tiny", quant: "int4", o: decoder.Options{ResidentMoEStep: true, ResidentRecurrentStep: true}},
		{dir: "../testdata/nemotron-tiny", quant: "int4", o: decoder.Options{ResidentRecurrentStep: true}},
		{dir: "../testdata/gemma4-emodel-tiny", quant: "int4", o: decoder.Options{ResidentGemma4Step: true}},
		{dir: filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"), quant: "q4k"},
	} {
		t.Run(filepath.Base(c.dir)+"/"+c.quant, func(t *testing.T) {
			if _, err := os.Stat(c.dir); err != nil {
				t.Skipf("no model at %s", c.dir)
			}
			o := c.o
			o.Quant, o.ResidentContext, o.ResidentKVSlots = c.quant, 256, 2
			m, err := decoder.Load(c.dir, o)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { m.Close() })
			metalMemoryCeilingForTest = residentNeedBytes(m) + metalSlotBytes(m) // two slots and no room for the step
			rf, ok, err := (&metalBackend{}).BuildResident(m)
			metalMemoryCeilingForTest = 0
			if err != nil || !ok {
				t.Fatalf("the load was refused: %v", err)
			}
			r := rf.(*metalResident).r
			t.Cleanup(func() { r.Close() })
			if r.batch != nil || r.stepDecline == "" {
				t.Fatalf("step built %v, decline %q; want the step declined so buildBatch runs here", r.batch != nil, r.stepDecline)
			}
			requested, list := r.stepShape().listBytes(), r.stepShape().buffers(&batchState{})
			l := metalLedgerOf(r.d)
			before := l.Usage()
			r.stepHeadroom, r.stepDecline = 1<<40, ""
			r.buildBatch()
			if r.batch == nil {
				t.Fatalf("no step: %s", r.stepDecline)
			}
			after := l.Usage()
			if got := after.Requested - before.Requested; got != requested {
				t.Errorf("buildBatch leased %d bytes, and the list holds %d", got, requested)
			}
			if got := after.Leases - before.Leases; got != len(list) {
				t.Errorf("buildBatch took %d leases, and the list holds %d buffers", got, len(list))
			}
			if r.stepScratchBytes() < requested {
				t.Errorf("the price %d is under the list's %d bytes", r.stepScratchBytes(), requested)
			}
			if got, want := metalStepScratchReserve(m), r.stepScratchBytes(); got != want {
				t.Errorf("the reserve before the build is %d bytes, and the build prices the step at %d", got, want)
			}
		})
	}
}

// TestStepBuffers_q4kWidths holds the q4k lane's part of the list on a synthetic shape, since no tiny fixture builds that lane: one
// uniform per distinct width, each reachable at its width.
func TestStepBuffers_q4kWidths(t *testing.T) {
	distinct := stepShape{H: 64, V: 256, I: 32, nHhd: 48, kvDim: 48, ffnWidth: 32, q4k: true}
	collapsed := distinct
	collapsed.nHhd, collapsed.kvDim = 64, 64
	plain := func(s stepShape) stepShape { s.q4k = false; return s }
	for _, c := range []struct {
		s    stepShape
		want []int
	}{{distinct, []int{32, 48, 64, 144}}, {collapsed, []int{32, 64, 192}}} {
		b := &batchState{}
		list := c.s.buffers(b)
		if got, want := len(list)-len(plain(c.s).buffers(&batchState{})), 2+len(c.want); got != want {
			t.Errorf("%+v: the lane adds %d buffers, want axB, swB and %d uniforms", c.s, got, len(c.want))
		}
		for _, sb := range list {
			sb.set(Buffer{})
		}
		if got := slices.Sorted(maps.Keys(b.uLd)); !slices.Equal(got, c.want) {
			t.Errorf("%+v: uniforms at %v, want %v", c.s, got, c.want)
		}
	}
}

// TestStepInputs_reserveOnShapesNoFixtureHas: the census's fixtures have uniform geometry, so these shapes are built by hand. A model
// the step declines reserves nothing, and one with layers of two widths reserves at the geometry layer's, as stepShape prices it, not
// at its widest.
func TestStepInputs_reserveOnShapesNoFixtureHas(t *testing.T) {
	attn := func(hd int) stepLayerInfo {
		return stepLayerInfo{steps: []layerStep{stepAttention, stepDenseFFN}, hasGeom: true, hd: hd, nKV: 4, ffnI: 128}
	}
	base := func(layers ...stepLayerInfo) *stepInputs {
		return &stepInputs{prog: &program.Program{}, H: 64, V: 256, I: 128, nH: 4, oneKVAllocation: true, layers: layers}
	}
	if base(attn(16), attn(16)).reserve() == 0 {
		t.Fatal("a plain uniform model reserves nothing: the cases below would pass for no reason")
	}
	noAttn := base(stepLayerInfo{steps: []layerStep{stepDeltaNet, stepNonGatedFFN}, ffnI: 128})
	noAttn.recurrentBlocks = true
	notReproduced := base(attn(16), attn(32))
	kEqV := attn(16)
	kEqV.kEqV = true
	for name, in := range map[string]*stepInputs{
		"no attention layer":                      noAttn,
		"a layer of another width":                notReproduced,
		"a layer the batched kernels do not hold": base(attn(16), kEqV),
		"an attention sink":                       func() *stepInputs { in := base(attn(16)); in.attnSink = true; return in }(),
		"a paged MoE":                             func() *stepInputs { in := base(attn(16)); in.pagedMoE = true; return in }(),
		"the w4f16 lane":                          func() *stepInputs { in := base(attn(16)); in.w4f16Lane = true; return in }(),
		"an untiled width":                        func() *stepInputs { in := base(attn(16)); in.V = 255; return in }(),
	} {
		if why, got := in.ineligible(), in.reserve(); why == "" || got != 0 {
			t.Errorf("%s: declined for %q and reserves %d; want a decline and 0", name, why, got)
		}
	}
	wholeWide := attn(32)
	wholeWide.whole = true
	mixed := base(attn(16), wholeWide)
	if why := mixed.ineligible(); why != "" {
		t.Fatalf("a whole wide layer beside a narrow batched one declines: %s", why)
	}
	narrow := base(attn(16), attn(16))
	narrow.layers[1].whole = true
	if got, want := mixed.reserve(), narrow.reserve(); got != want {
		t.Errorf("a whole wide layer reserves %d bytes and the narrow geometry layer's price is %d", got, want)
	}
}

// TestMetalKVSlots_defaultLeavesRoomForTheStep: an unpinned slot count gives up a slot to keep the batched step's scratch in the
// budget where that makes the step fit, and a count the caller asked for keeps its slots and loses the step.
func TestMetalKVSlots_defaultLeavesRoomForTheStep(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	t.Cleanup(func() { metalMemoryCeilingForTest = 0 })
	build := func(unpinned bool, ceiling func(m *decoder.Model) int64) *metalResident {
		m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Quant: "int4", ResidentContext: 256, ResidentKVSlots: 2, ResidentKVSlotsDefault: unpinned})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		metalMemoryCeilingForTest = ceiling(m)
		rf, ok, err := (&metalBackend{}).BuildResident(m)
		if err != nil || !ok {
			t.Fatalf("the load was refused: %v", err)
		}
		a := rf.(*metalResident)
		t.Cleanup(func() { a.r.Close() })
		return a
	}
	twoSlots := func(m *decoder.Model) int64 { return residentNeedBytes(m) + metalSlotBytes(m) }
	if a := build(true, func(m *decoder.Model) int64 { return twoSlots(m) + metalStepScratchReserve(m) - 1 }); len(a.r.kvSlotBufs) != 1 || a.r.batch == nil {
		t.Errorf("unpinned, a byte short of two slots and the reserve: %d slots, step built %v (%s); want one slot and the step", len(a.r.kvSlotBufs), a.r.batch != nil, a.r.stepDecline)
	}
	if a := build(true, func(m *decoder.Model) int64 { return twoSlots(m) + metalStepScratchReserve(m) }); len(a.r.kvSlotBufs) != 2 || a.r.batch == nil {
		t.Errorf("unpinned, room for two slots and the reserve: %d slots, step built %v (%s); want both", len(a.r.kvSlotBufs), a.r.batch != nil, a.r.stepDecline)
	}
	if a := build(false, twoSlots); len(a.r.kvSlotBufs) != 2 || a.r.batch != nil {
		t.Errorf("pinned at two, no room for the scratch: %d slots, step built %v; want two slots and no step", len(a.r.kvSlotBufs), a.r.batch != nil)
	}
	// One slot's KV is smaller than the scratch here, so giving it up would still leave no room for the step, and the load would lose both.
	if a := build(true, func(m *decoder.Model) int64 {
		if metalSlotBytes(m) >= metalStepScratchReserve(m) {
			t.Fatalf("a slot of %d bytes is not under the reserve of %d: this case needs a smaller slot", metalSlotBytes(m), metalStepScratchReserve(m))
		}
		return twoSlots(m)
	}); len(a.r.kvSlotBufs) != 2 || a.r.batch != nil {
		t.Errorf("unpinned, giving up a slot would not make room for the step: %d slots, step built %v; want two slots and no step", len(a.r.kvSlotBufs), a.r.batch != nil)
	}
}

// metalDecodeScratchBytes is, from the model's widths and the blocks this build has on, what a build leases on the scratch line for
// decode itself. Nothing prices these. Its buffers are sized as the build sizes them, and its one allowance is a fixed count of small
// uniforms, so it holds a leak in any width-sized buffer to well under a buffer. MM5's list for decode replaces it.
func metalDecodeScratchBytes(m *decoder.Model, r *resident) int64 {
	H, nL, nH, _, _, I, V := m.Dims()
	var nHhd, kvDim, faPartial int
	for l := range nL {
		hd := m.HeadDimAtResident(l)
		nHhd, kvDim = max(nHhd, nH*hd), max(kvDim, m.KVHeadsAtResident(l)*hd)
		if r.layers[l].geom != nil {
			faPartial = max(faPartial, attnFAPartialFloats(hd, nH, m.KVHeadsAtResident(l)))
		}
	}
	gu := I
	for _, lw := range m.Weights().Layers {
		gu = max(gu, lw.GateProj.Rows())
	}
	floats, bytes := 0, 0
	if mo := r.moe; mo != nil {
		gu = max(gu, mo.inter, mo.sharedInter)
		floats += mo.nE + 2*mo.k + H // the router's logits, indices and weights, and the shared expert's down
		if mo.guK != (Buffer{}) {    // allocated before an int8 build turns those kernels off
			floats += mo.k*2*mo.inter + mo.k + mo.k*H
			bytes += mo.k * mo.inter
		}
	}
	if g := r.g4moe; g != nil {
		gu = max(gu, g.denseInter, g.moeInter)
		floats += 2*(g.nE+2*g.topK) + 4*H
	}
	if dp := r.dnet; dp != nil {
		floats += 2*dp.convDim + 4*dp.nv + 2*dp.keyDim + 3*dp.valueDim + 3*nHhd // its mixer, and gated attention's rows
		bytes += dp.valueDim
	}
	if p := r.mamba; p != nil {
		floats += p.projDim + p.convDim + 2*p.dInner
		bytes += p.dInner
	}
	if p := r.mla; p != nil {
		floats += max(p.qLoRA, 1) + p.nH*p.qkHead + p.latDim + p.nH*p.latDim + p.nH*p.rank + p.nH*p.vHead
		bytes += max(p.qLoRA, 4) + p.nH*p.vHead
	}
	if P := r.pleP; P > 0 {
		floats += nL*P + P
		bytes += P
	}
	floats += 3*H + nHhd + 2*kvDim + 2*gu + nHhd + V + 2*((V+7)/8) + faPartial // x, oO, dO; qkv; gu; ctx; logits; part; the fused partial
	floats += r.batchCap * (H + V)                                             // ensureBatchCap's rows, once a call has asked for them
	floats += 2 * r.gumbelNB
	bytes += 6*H + 3*nHhd + gu // aq, mq, the f16 twins of a and m; cq and its twin; dq
	const uniforms = 450       // each a few bytes: the model's, the chain's and the families'
	return 4*int64(floats) + int64(bytes) + 4*int64(uniforms+8*nL)
}

// metalDenseVectorBytes is what a build leases on the dense line that the plan's dense figure, a sum of weight matrices, does not
// count: the norms' weights, the biases, the attention sinks, the layer scale, the rotation tables and the m-RoPE axis map. Which of
// them this build holds is read from its buffers, and each one's size from the model's weights and geometry, so a buffer that grows
// does not raise its own allowance. Where the build uploads zeros in place of a family's missing bias, the zeros are counted.
func metalDenseVectorBytes(m *decoder.Model, r *resident) int64 {
	w := m.Weights()
	_, _, nH, _, _, _, _ := m.Dims()
	has := func(b Buffer) bool { return b != (Buffer{}) }
	var n, maxHd int
	add := func(b Buffer, floats int) {
		if has(b) {
			n += floats
		}
	}
	for l := range w.Layers {
		lw, L := &w.Layers[l], &r.layers[l]
		hd, kvDim := m.HeadDimAtResident(l), m.KVHeadsAtResident(l)*m.HeadDimAtResident(l)
		maxHd = max(maxHd, hd)
		add(L.preNorm, len(lw.PreAttnNorm))
		if L.postNorm != L.preNorm {
			add(L.postNorm, len(lw.PreMLPNorm))
		}
		add(L.preNormBias, len(lw.PreAttnNormBias))
		add(L.postNormBias, len(lw.PreMLPNormBias))
		add(L.postAttnNorm, len(lw.PostAttnNorm))
		add(L.postMLPNorm, len(lw.PostMLPNorm))
		add(L.postPLENorm, len(lw.PostPLENorm))
		qN, kN := lw.QNorm, lw.KNorm
		if _, _, _, _, gq, gk := m.Qwen35AttnWeights(l); L.qGate {
			qN, kN = gq, gk
		}
		add(L.qNorm, len(qN))
		if L.kNorm != L.qNorm {
			add(L.kNorm, len(kN))
		}
		add(L.oBias, len(lw.OBias))
		if r.attnSink {
			add(L.attnSinks, len(lw.AttnSinks))
		} else {
			add(L.attnSinks, 1)
		}
		add(L.upBias, max(len(lw.UpBias), lw.UpProj.Rows()))
		add(L.downBias, max(len(lw.DownBias), lw.DownProj.Rows()))
		add(L.qkvBias, max(len(lw.QBias)+len(lw.KBias)+len(lw.VBias), nH*hd+2*kvDim))
		add(L.invf, len(m.RopeInvFreqLayerResident(l)))
		add(L.mscale, 1)
		add(L.uLayerScalar, 1)
		if L.moe != nil {
			add(L.moe.routerBias, max(len(lw.RouterBias), r.moe.nE))
		}
	}
	add(r.finalNorm, len(w.FinalNorm))
	add(r.finalNormBias, len(w.FinalNormBias))
	add(r.invf, len(m.RopeInvFreq()))
	add(r.vNormUnit, maxHd)
	if g := r.prefillGeom(); g != nil {
		add(r.mropeAxis, len(m.MRopeAxisResident(g.half)))
	}
	return 4 * int64(n)
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
