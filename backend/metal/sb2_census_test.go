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
	"github.com/townsendmerino/goinfer/decoder/fit"
	"github.com/townsendmerino/goinfer/internal/diag/sb2census"
	"github.com/townsendmerino/goinfer/internal/guard/memgate"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/prequant"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
	"github.com/townsendmerino/goinfer/internal/wrappedupload"
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
	paths, err := filepath.Glob("../../testdata/machines/*.json")
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
	d, err := sb2census.ReadLayerD("../..", "metal")
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
		path, err := sb2census.Path("../..", f)
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
var metalMemGaps []memgate.Gap

const metalMemGapsPin = 0

// metalGatePlantForTest, when set, runs on a row's resident after its generation and before the ledger is read, by G-MS2 and the
// snapshot; what it returns runs after that read, before Close.
var metalGatePlantForTest func(r *resident) (free func())

// metalGateRow loads fixture path at options o, runs it, and returns the row: the build's ledger after the build and after a short
// generation, the plan for the same load, and the ledger after Close; exactRed is G-MS2's findings, for the caller to fail on (or, under
// a plant, to expect). ok is false when Metal's own build declines, which has no allocation to hold (the census's rows record
// admission, not the build).
func metalGateRow(t *testing.T, row, path string, o decoder.Options, kvI8 bool) (_ memgate.Reading, exactRed []string, ok bool) {
	t.Helper()
	m, err := decoder.Load(path, o)
	if err != nil {
		return memgate.Reading{}, nil, false
	}
	closed := false
	defer func() {
		if !closed {
			m.Close()
		}
	}()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		return memgate.Reading{}, nil, false
	}
	r := rf.(*metalResident).r
	l := metalLedgerOf(r.d)
	if l == nil {
		t.Errorf("%s: the build's device has no ledger", row)
		return memgate.Reading{}, nil, false
	}
	read := memgate.Reading{Row: row, KVSlots: r.kvSlotCount()}
	read.Plan = m.Plan("metal", math.MaxInt64/4, fit.PlanRequest{Ctx: r.ctxCap, CtxPinned: true, KVI8: kvI8})
	// G-MS2 (docs/tasks/task-forward-architecture-2026-10.md §4.9c): the scratch line holds decode's list and the step's to the byte,
	// both ways. The ledger's requests are read, not its charges, so the allocator's rounding hides nothing.
	exact := func(at string) {
		if held, want := scratchRequested(l), scratchLists(r); held != want {
			exactRed = append(exactRed, fmt.Sprintf("%s: %s: the scratch line holds %d bytes requested, and the decode and step lists %d",
				row, at, held, want))
		}
	}
	read.Open = append(read.Open, memgate.Snap("built", l, nil))
	exact("built")
	// MM5b: the decoder's price of decode's scratch is the build's list, with the chain's buffers, which the build holds once a chain
	// opens (none has yet).
	var chain int64
	for _, sb := range r.chainBuffers(true, true) {
		chain += int64(sb.bytes())
	}
	if got, want := m.MetalDecodeScratchBytes(metalMoESlotsFromEnv(m)), r.decode.listBytes(r)+chain; got != want {
		t.Errorf("%s: the decoder prices decode's scratch at %d bytes; the build's list with the chain's is %d (%+d)", row, got, want, got-want)
	}
	ch, g := m.Generate(context.Background(), []int{1, 2, 3, 4, 5}, 4, decoder.SamplingParams{})
	// Draining the channel joins the generation: its goroutine closes it once the body has returned.
	for range ch {
	}
	if err := g.Err(); err != nil {
		t.Errorf("%s: generation: %v", row, err)
	}
	free := func() {}
	if metalGatePlantForTest != nil {
		free = metalGatePlantForTest(r)
	}
	exact("generated")
	read.Open = append(read.Open, memgate.Snap("generated", l, nil))
	free()
	// Scratch has no plan line: it holds decode's list and the step's, which the gate holds it to, before the step price's page
	// rounding, which would hide what else leases there.
	scratch := scratchLists(r)
	// The reserve is the build's own answer read before the build: every input to it agrees, so it is the step's price where
	// batchIneligible admits it and 0 where not, on shapes no fixture has as well as on these.
	built, before := r.stepInputs(), metalStepInputs(m)
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
	return read, exactRed, true
}

// TestMemGate_metalPlantGoesRed: the gate reads the real ledger. llama-tiny's dense line is under plan after the build; a mebibyte
// charged over its request on one dense buffer lands on the ledger and is red, past what the allocator's rounding would forgive.
func TestMemGate_metalPlantGoesRed(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const fx, plant = "../../testdata/llama-tiny", int64(1 << 20)
	o := decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256}
	builtRed := func(r memgate.Reading) bool {
		for _, f := range memgate.Check(r, metalRounding) {
			if f.Line == mem.LineDense && strings.HasPrefix(f.What, "built:") {
				return true
			}
		}
		return false
	}
	clean, exactRed, ok := metalGateRow(t, "clean", fx, o, false)
	if !ok {
		t.Fatal("llama-tiny did not build resident")
	}
	for _, e := range exactRed {
		t.Error(e)
	}
	if builtRed(clean) {
		t.Fatalf("the unplanted build is already over plan on dense: %v", memgate.Check(clean, metalRounding))
	}
	ledgerChargeDeltaForTest, ledgerChargeLineForTest = plant, mem.LineDense
	defer func() { ledgerChargeDeltaForTest, ledgerChargeLineForTest = 0, "" }()
	planted, _, ok := metalGateRow(t, "planted", fx, o, false)
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
// a dense model and a recurrent one. The buffer is freed before Close, so what is red is the plan. On scratch it is red in G-MS2's
// exact read as well, and on any other line it is not.
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
			read, exactRed, ok := metalGateRow(t, row, "../../testdata/"+c.fx, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256}, false)
			if !ok || !planted {
				t.Fatalf("resident %v, plant ran %v", ok, planted)
			}
			switch {
			case c.line == mem.LineScratch && (len(exactRed) != 1 || !strings.Contains(exactRed[0], ": generated: ")):
				t.Errorf("G-MS2 on a plant on scratch: %q; want one red, after the generation", exactRed)
			case c.line != mem.LineScratch && len(exactRed) != 0:
				t.Errorf("G-MS2 on a plant on %q, a line it does not read: %q; want none", c.line, exactRed)
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

// TestDecodeScratch_handTable is G-MS3 (docs/tasks/task-forward-architecture-2026-10.md §4.9c): a build's decode list, as it stands
// after the build, equals arithmetic written here on the config's numbers. G-MS2 holds the ledger to the list; this holds the list to
// the model, which a declaration one element short fails. Its rows reach every branch of the list a registered row builds; the chain
// and the paged fence, which a build holds later or under a switch, have their own tests below. No registered row builds the q4k lane
// (a GGUF of K-quants) or the per-layer embedding (Gemma 4's PLE), so neither has a row.
func TestDecodeScratch_handTable(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const batchCap = 16 // the build's batch rows (ensureBatchCap(16))
	// core is what every build holds, for hidden H, vocab V, gate|up half gu, query width q and K/V width kv at the widest attention
	// layer: its f32 rows, its int8 and f16 twins, and its uniforms (the eleven core ones; uAct, uAddOne, uWindow, uLNHasBias; the
	// sampler's seven).
	core := func(H, V, gu, q, kv int) (f32Rows, twins, uniforms int) {
		f32Rows = H + 1 + // x, aSc
			q + 2*kv + // fused q | k | v
			2*gu + // fused gate | up
			q + 1 + // ctx, cSc
			H + 1 + // oO, mSc
			1 + H + // dSc, dO
			V + 2*((V+7)/8) + 1 + // logits, the argmax partials, tok
			2*((V+1023)/1024) + 1 + // the sampler's block keys and indices, its draw
			batchCap*(H+V) // batchX, batchLogits
		twins = H + q + H + // aq, cq, mq
			2*H + 2*H + 2*q + // the W4F16 lane's f16 twins of aq, mq and cq
			gu // dq
		return f32Rows, twins, 11 + 4 + 7 + 3*batchCap
	}
	for _, c := range []struct {
		name, quant string
		want        func() int64
	}{
		// llama-tiny: hidden 64, vocab 256, FFN 128, 4 query heads and 2 K/V heads of 16, 4 layers, all attention; head width 16,
		// which the fused attention does not run, so no partial buffer.
		{"llama-tiny", "int8int8", func() int64 {
			const H, V, I, nH, nKV, hd, nL = 64, 256, 128, 4, 2, 16, 4
			f, b, u := core(H, V, I, nH*hd, nKV*hd)
			u += 7 + 2*nL // one attention geometry; each layer's sink flag and window
			return int64(4*f + b + 4*u)
		}},
		// qwen35-tiny: hidden 64, vocab 256, 4 layers of which the last is gated attention (4 query heads and 2 K/V heads of 16)
		// and three Gated DeltaNet (2 key heads and 4 value heads of 16, conv kernel 4); every FFN a MoE of 4 experts, top 2,
		// expert and shared width 32, whose k-slot kernels run (2 x 32 and 64 are multiples of 8).
		{"qwen35-tiny", "int4", func() int64 {
			const H, V, nH, nKV, hd = 64, 256, 4, 2, 16
			const nk, hk, nv, hv = 2, 16, 4, 16
			const nE, k, inter = 4, 2, 32
			f, b, u := core(H, V, inter, nH*hd, nKV*hd)
			u += 7 + 2 // the one attention geometry; the attention layer's sink flag and window
			keyDim, valueDim := nk*hk, nv*hv
			convDim := 2*keyDim + valueDim
			f += 2*convDim + 2*nv + 2*nv + 2*keyDim + 3*valueDim + 1 // the mixer's rows, its head parameters, dnGSc
			b += valueDim                                            // dnGq
			f += 2*nH*hd + nH*hd                                     // gated attention's [query ‖ gate] and its gate
			u += 12                                                  // the mixer's uniforms
			f += nE + k + 1 + H                                      // the router's logits and weights, the shared gate and down
			b += 4 * k                                               // the router's indices
			f += k*2*inter + k + k*H                                 // the k slots' gate|up, scales and accumulators
			b += k * inter                                           // their int8 rows
			u += 10 + k                                              // the MoE's uniforms and one a slot
			return int64(4*f + b + 4*u)
		}},
		// gemma2-tiny: hidden 64, vocab 512, FFN 128, 4 query heads and 2 K/V heads of 32, 4 layers alternating a 4-position window
		// with full attention; the window changes each layer's uniform's value, not its size, and head width 32 has no partial.
		{"gemma2-tiny", "int8int8", func() int64 {
			const H, V, I, nH, nKV, hd, nL = 64, 512, 128, 4, 2, 32, 4
			f, b, u := core(H, V, I, nH*hd, nKV*hd)
			u += 7 + 2*nL
			return int64(4*f + b + 4*u)
		}},
		// deepseek-tiny: hidden 64, vocab 128, 4 layers, every one Multi-head Latent Attention (4 heads, q LoRA 24, KV LoRA 16, rope 8,
		// nope 16, v 16), so no attention layer: the attention rows take the build's width of 4. Layer 0's FFN is dense (128), the rest
		// a MoE of 8 experts, top 2, width 32, one shared expert of 32, whose k-slot kernels run.
		{"deepseek-tiny", "int4", func() int64 {
			const H, V, I, nH = 64, 128, 128, 4
			const qLoRA, rank, rope, nope, vHead = 24, 16, 8, 16, 16
			const nE, k, inter = 8, 2, 32
			f, b, u := core(H, V, I, 4, 4)
			qkHead, latDim := nope+rope, rank+rope
			f += qLoRA + 1 + nH*qkHead + latDim + nH*latDim + nH*rank + nH*vHead + 1 // qa, qaSc, q, kv, qAbs, wsum, ctx, ctxSc
			b += qLoRA + nH*vHead                                                    // qaQ, ctxQ
			b += 4 * (3 + 5 + 5 + 7 + 3)                                             // MLA's packed shapes
			u += 5                                                                   // and its five single uniforms
			f += nE + k + 1 + H + k*2*inter + k + k*H                                // the router and the k slots, as above
			b += 4*k + k*inter
			u += 10 + k
			return int64(4*f + b + 4*u)
		}},
		// nemotron-tiny: hidden 64, vocab 256, FFN 128, five layers (Mamba-2, attention, MLP, Mamba-2, attention); attention of 4 query
		// heads and 2 K/V heads of 16; Mamba-2 of 8 heads of 8 (inner 64), state 16, one group, conv kernel 4. Metal runs it at int4.
		{"nemotron-tiny", "int4", func() int64 {
			const H, V, I, nH, nKV, hd = 64, 256, 128, 4, 2, 16
			const nh, mhd, ds, ng = 8, 8, 16, 1
			f, b, u := core(H, V, I, nH*hd, nKV*hd)
			u += 7 + 2*2 // the one attention geometry; the two attention layers' sink flags and windows
			dInner, gSize := nh*mhd, ng*ds
			convDim, projDim := dInner+2*gSize, 2*dInner+2*gSize+nh
			f += projDim + convDim + 2*dInner + 1 // in_proj's output, the conv's, the scan's and the gated norm's, mbGSc
			b += dInner                           // mbGq
			u += 13                               // the mixer's uniforms
			return int64(4*f + b + 4*u)
		}},
		// llama-attnfa-tiny: hidden 512, vocab 256, FFN 256, 4 query heads and 2 K/V heads of 128, 2 layers, all attention. Head width
		// 128 runs the fused attention, whose partial holds each query head's accumulators, max and sum for each of its 32 splits; its
		// group and split-count uniforms come with it.
		{"llama-attnfa-tiny", "int4", func() int64 {
			const H, V, I, nH, nKV, hd, nL, splits = 512, 256, 256, 4, 2, 128, 2, 32
			f, b, u := core(H, V, I, nH*hd, nKV*hd)
			f += nH * (hd + 2) * splits
			u += 2 + 7 + 2*nL
			return int64(4*f + b + 4*u)
		}},
		// olmo3-tiny: hidden 64, vocab 512, FFN 128, 8 query heads and 8 K/V heads of 8, 4 layers (three with an 8-position window, then
		// full attention). Its QK norm spans the whole q and k vectors, which adds that norm's two uniforms.
		{"olmo3-tiny", "int4", func() int64 {
			const H, V, I, nH, nKV, hd, nL = 64, 512, 128, 8, 8, 8, 4
			f, b, u := core(H, V, I, nH*hd, nKV*hd)
			u += 2 + 7 + 2*nL
			return int64(4*f + b + 4*u)
		}},
		// gemma4-moe-kv-tiny32: hidden 64, vocab 256, 2 layers: one with a 4-position window (4 query heads, 2 K/V heads of 16) and one
		// with full attention (4 query heads, 1 K/V head of 512). That is two attention geometries, and the rows take the widest. Each
		// layer runs a dense FFN of 64 beside a MoE of 8 experts, top 2. The MoE's list has two sets, the second for the async route.
		// Each set holds the router's logits, indices and weights and a normed hidden row; two more hidden rows and the MoE's uniforms
		// are held once.
		{"gemma4-moe-kv-tiny32", "int4", func() int64 {
			const H, V, I, nH, nL = 64, 256, 64, 4, 2
			const nE, k = 8, 2
			f, b, u := core(H, V, I, nH*512, 1*512) // K/V width: the wider of 2 x 16 and 1 x 512
			u += 2*7 + 2*nL                         // the two attention geometries; each layer's sink flag and window
			f += 2*(nE+k) + 4*H                     // rLogits, rWgt and their twins; g4x1, g4x2, g4rn, g4rnP
			b += 2 * 4 * k                          // rIdx, rIdxP
			u += 10 + k                             // the MoE's uniforms and one a slot
			return int64(4*f + b + 4*u)
		}},
		// gptoss_tiny.gguf (scripts/gptoss_tiny_golden.py): hidden 64, vocab 128, 8 query heads and 2 K/V heads of 8, 4 layers
		// alternating a 4-position window with full attention, each with a sink. Every FFN is a MoE of 4 experts, top 2, width 32, with
		// no shared expert. Its clamped SwiGLU and biased down run their own kernels, so it has no k-slot rows; its alpha, limit and
		// bias flag add three uniforms.
		{"gptoss_tiny.gguf", "int4", func() int64 {
			const H, V, nH, nKV, hd, nL = 64, 128, 8, 2, 8, 4
			const nE, k, inter = 4, 2, 32
			f, b, u := core(H, V, inter, nH*hd, nKV*hd)
			u += 7 + 2*nL
			f += nE + k + 1 + H // the router's logits and weights, the shared gate and down
			b += 4 * k          // the router's indices
			u += 3 + 10 + k
			return int64(4*f + b + 4*u)
		}},
	} {
		path, err := sb2census.Path("../..", "testdata/"+c.name)
		if err != nil {
			t.Fatal(err)
		}
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: c.quant, ResidentContext: 256})
		if err != nil {
			t.Fatal(err)
		}
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			m.Close()
			t.Fatalf("%s: not Metal-resident: %s", c.name, m.ResidentDecline())
		}
		if got, want := a.r.decode.listBytes(a.r), c.want(); got != want {
			t.Errorf("%s: the decode list is %d bytes; the config's arithmetic gives %d", c.name, got, want)
		}
		m.Close()
	}
}

// scratchRequested is what l's open leases on the scratch line asked for.
func scratchRequested(l *mem.Ledger) int64 {
	var n int64
	for _, o := range l.OpenLeases() {
		if o.Line == mem.LineScratch {
			n += o.Requested
		}
	}
	return n
}

// scratchLists is what G-MS2 holds r's scratch line to: decode's list, and the step's where the build holds one.
func scratchLists(r *resident) int64 {
	n := r.decode.listBytes(r)
	if r.batch != nil {
		n += r.stepShape().listBytes()
	}
	return n
}

// TestDecodeScratch_handTableChain is G-MS3 for the chain, which the list counts once it is held. On llama-tiny (vocab 256), opening the
// greedy chain adds six uniforms to each of its two buffers: the token, position, key count, rope position, temperature scale and split
// count. A sampled chain then adds the draw's temperature and two keys, plus each buffer's logits row and two draw words. The scratch
// line holds the list throughout (G-MS2), and the decoder's price, which counts both chains from the start, is the list once both are
// open.
func TestDecodeScratch_handTableChain(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	m, err := decoder.Load("../../testdata/llama-tiny", decoder.Options{Backend: "metal", Quant: "int8int8", ResidentContext: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	r, l := a.r, metalLedgerOf(a.r.d)
	const V, sets = 256, 2
	for _, c := range []struct {
		name string
		open func() error
		want int64
	}{
		{"greedy", func() error { return r.chainStart(1, 0) }, 4 * 6 * sets},
		{"sampled", func() error { return r.chainStartSampled(1, 0, 1, 7, 0) }, 4*3 + sets*(4*V+4*2)},
	} {
		before := r.decode.listBytes(r)
		if err := c.open(); err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		r.stopChain()
		after := r.decode.listBytes(r)
		if got := after - before; got != c.want {
			t.Errorf("%s: opening the chain grew the decode list by %d bytes; the config's arithmetic gives %d", c.name, got, c.want)
		}
		if held, want := scratchRequested(l), scratchLists(r); held != want {
			t.Errorf("%s: the scratch line holds %d bytes requested, and the decode and step lists %d", c.name, held, want)
		}
	}
	if got, want := m.MetalDecodeScratchBytes(metalMoESlotsFromEnv(m)), r.decode.listBytes(r); got != want {
		t.Errorf("with both chains open the decoder prices decode's scratch at %d bytes, and the list is %d", got, want)
	}
}

// TestDecodeScratch_handTablePagedFence is G-MS3 for a paged Gemma 4 MoE, which no registered row builds. The fixture is the tiny Gemma 4
// MoE (4 experts, top 2, 2 layers) as a metal .giw, paged at 3 experts. Paging adds the staged top-k's slot indices. The route fence,
// switched on, adds the mirrored route (two words a pick), its word, its top-k uniform, one sequence number a layer and its two flags.
// The scratch line holds the list on each build (G-MS2). With the fence off, as it ships, the decoder prices the paged build's scratch at
// the list plus the chain.
func TestDecodeScratch_handTablePagedFence(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const ckpt = "../../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	giw := filepath.Join(t.TempDir(), "g4.metal.giw")
	if err := prequant.Transcode(context.Background(), ckpt, giw, "int4", false, decoder.GIWTargetMetal); err != nil {
		t.Fatal(err)
	}
	prev := pagedFenceOn
	defer func() { pagedFenceOn = prev }()
	const k, nL = 2, 2
	list := func(slots string, fence bool) int64 {
		t.Helper()
		pagedFenceOn = fence
		o := decoder.Options{Backend: "metal", Quant: "int4"}
		if slots != "" {
			o.Knobs = &decoder.Knobs{"GOINFER_METAL_MOE_SLOTS": slots}
		}
		m, err := decoder.Load(giw, o)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("slots %q: not Metal-resident: %s", slots, m.ResidentDecline())
		}
		r := a.r
		if paged := r.g4moe != nil && r.g4moe.paged; paged != (slots != "") {
			t.Fatalf("slots %q: the Gemma 4 MoE is paged %v", slots, paged)
		}
		if fence && r.g4moe.fence == nil {
			t.Skip("no fence was built: this machine's compiler rejects MSL 3.2")
		}
		n := r.decode.listBytes(r)
		if held, want := scratchRequested(metalLedgerOf(r.d)), scratchLists(r); held != want {
			t.Errorf("slots %q, fence %v: the scratch line holds %d bytes requested, and the decode and step lists %d", slots, fence, held, want)
		}
		if !fence {
			var chain int64
			for _, sb := range r.chainBuffers(true, true) {
				chain += int64(sb.bytes())
			}
			if got, want := m.MetalDecodeScratchBytes(metalMoESlotsFromEnv(m)), n+chain; got != want {
				t.Errorf("slots %q: the decoder prices decode's scratch at %d bytes; the list with the chain's is %d", slots, got, want)
			}
		}
		return n
	}
	resident, paged, fenced := list("", false), list("3", false), list("3", true)
	if got, want := paged-resident, int64(4*k); got != want {
		t.Errorf("paging grew the decode list by %d bytes; the config's arithmetic gives %d", got, want)
	}
	if got, want := fenced-paged, int64(4*(2*k+1+1+nL+2)); got != want {
		t.Errorf("the fence grew the decode list by %d bytes; the config's arithmetic gives %d", got, want)
	}
}

// TestMetalScratch_batchGrowthGivesBackTheOld: ensureBatchCap gives the rows it replaces back to the device, so the scratch line holds
// one capacity's batch rows, not every capacity the resident has had.
func TestMetalScratch_batchGrowthGivesBackTheOld(t *testing.T) {
	_, r := kv2Resident(t, false)
	l := metalLedgerOf(r.d)
	if l == nil {
		t.Fatal("the build's device has no ledger")
	}
	rowsOf := func(c int) (logits, x int) {
		for _, o := range l.OpenLeases() {
			switch {
			case o.Line != mem.LineScratch:
			case o.Requested == int64(4*c*r.V):
				logits++
			case o.Requested == int64(4*c*r.H):
				x++
			}
		}
		return logits, x
	}
	c0 := r.batchCap
	if lg, x := rowsOf(c0); lg != 1 || x != 1 {
		t.Fatalf("at the build's capacity of %d rows the scratch line holds %d logits and %d hidden leases of that size; want 1 each", c0, lg, x)
	}
	r.ensureBatchCap(4*c0 + 1) // past two doublings, so no old size equals a new one when V is a multiple of H
	if lg, x := rowsOf(c0); lg != 0 || x != 0 {
		t.Errorf("after growing from %d rows to %d the scratch line still holds %d logits and %d hidden leases of the old size; want none",
			c0, r.batchCap, lg, x)
	}
	if lg, x := rowsOf(r.batchCap); lg != 1 || x != 1 {
		t.Errorf("after growing to %d rows the scratch line holds %d logits and %d hidden leases of that size; want 1 each", r.batchCap, lg, x)
	}
}

// TestMetalScratch_prefillGivesItsScratchBack: a batched prefill gives back every buffer it allocates, its uniforms included, so a
// second prefill of the same rows leaves the scratch line where the first did.
func TestMetalScratch_prefillGivesItsScratchBack(t *testing.T) {
	a, r := kv2Resident(t, false)
	l := metalLedgerOf(r.d)
	if l == nil {
		t.Fatal("the build's device has no ledger")
	}
	if err := r.useKVSlot(0); err != nil {
		t.Fatal(err)
	}
	embs := kv2Embs(r, 70, 1)
	run := func() int64 {
		t.Helper()
		if _, err := a.PrefillLast(context.Background(), embs, 0); err != nil {
			t.Fatal(err)
		}
		return l.Usage().ByLine[mem.LineScratch]
	}
	if first, second := run(), run(); second != first {
		t.Errorf("the scratch line holds %d bytes after one prefill and %d after a second of the same rows; want the same", first, second)
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
		{dir: "../../testdata/llama-tiny", quant: "int4"},
		{dir: "../../testdata/llama-tiny", quant: "int8int8"},
		{dir: "../../testdata/mixtral-tiny", quant: "int4", o: decoder.Options{ResidentMoEStep: true}},
		{dir: "../../testdata/qwen35-tiny", quant: "int4", o: decoder.Options{ResidentMoEStep: true, ResidentRecurrentStep: true}},
		{dir: "../../testdata/nemotron-tiny", quant: "int4", o: decoder.Options{ResidentRecurrentStep: true}},
		{dir: "../../testdata/gemma4-emodel-tiny", quant: "int4", o: decoder.Options{ResidentGemma4Step: true}},
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

// TestStepInputs_layerWithNoStepsOnBothPaths: a layer whose operations have no steps is the same answer read from the build and from
// the model, a decline and no reserve, on a plain layer and on a Gemma 4 layer the step runs whole.
func TestStepInputs_layerWithNoStepsOnBothPaths(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	for _, c := range []struct {
		name, dir string
		o         decoder.Options
		whole     bool
	}{
		{"plain", "../../testdata/llama-tiny", decoder.Options{}, false},
		{"whole Gemma 4 layer", "../../testdata/gemma4-emodel-tiny", decoder.Options{ResidentGemma4Step: true}, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			o := c.o
			o.Backend, o.Quant, o.ResidentContext, o.ResidentKVSlots = "metal", "int4", 256, 2
			m, err := decoder.Load(c.dir, o)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { m.Close() })
			rf := m.ResidentForwardForTest()
			if rf == nil {
				t.Fatalf("not resident: %s", m.ResidentDecline())
			}
			r := rf.(*metalResident).r
			if why := r.batchIneligible(); why != "" {
				t.Fatalf("the unmodified model builds no step: %s", why)
			}
			if got := r.stepInputs().layers[0].whole; got != c.whole {
				t.Fatalf("layer 0 whole %v, want %v", got, c.whole)
			}
			p := m.Program()
			saved := p.Layers[0]
			p.Layers[0] = append(slices.Clone(saved), program.Op{Kind: program.OpNorm}) // ends with no residual add, so it has no steps
			defer func() { p.Layers[0] = saved }()
			built, before := r.stepInputs(), metalStepInputs(m)
			if before.stepsErr == "" || built.stepsErr != before.stepsErr {
				t.Errorf("stepsErr: the build reads %q and the model %q; want one non-empty answer", built.stepsErr, before.stepsErr)
			}
			if why := r.batchIneligible(); why == "" || why != before.ineligible() {
				t.Errorf("declined: the build for %q and the model for %q; want one decline", why, before.ineligible())
			}
			if len(before.layers) != len(p.Layers) {
				t.Fatalf("the model's reading stops at %d of %d layers", len(before.layers), len(p.Layers))
			}
			// Only the program moved, so the build's layer 0 keeps its old steps. Every other layer's are the same on both.
			for l := 1; l < len(p.Layers); l++ {
				if !slices.Equal(built.layers[l].steps, before.layers[l].steps) {
					t.Errorf("layer %d's steps: the build reads %v and the model %v", l, built.layers[l].steps, before.layers[l].steps)
				}
			}
			if got := metalStepScratchReserve(m); got != 0 {
				t.Errorf("the reserve is %d bytes for a step that cannot build", got)
			}
		})
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
		m, err := decoder.Load("../../testdata/llama-tiny", decoder.Options{Quant: "int4", ResidentContext: 256, ResidentKVSlots: 2, ResidentKVSlotsDefault: unpinned})
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

// TestMetalDensePrice_isTheBuildsLeases holds the plan's dense line to what a Metal build requests on it, to the byte, on builds whose
// only other dense term is the token-embedding table an untied model reads from host memory: plain attention, Q/K/V biases, a sliding
// window, Gemma 2's sandwich norms, and Gemma 4's MoE block with K=V layers. MM7's gate holds the plan only from below, so an
// over-price passes there and is red here. The last row drops the Q bias from a biased checkpoint, which the build then replaces
// with zeros for all three projections.
func TestMetalDensePrice_isTheBuildsLeases(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	for _, c := range []struct {
		fx    string
		plant func(w *decoder.Weights)
	}{
		{fx: "llama-tiny"}, {fx: "glm-tiny-bias"}, {fx: "mistral-tiny-window"}, {fx: "gemma2-tiny"}, {fx: "gemma4-moe-kv-tiny32"},
		{fx: "glm-tiny-bias", plant: func(w *decoder.Weights) {
			for l := range w.Layers {
				w.Layers[l].QBias = nil
			}
		}},
	} {
		name := c.fx
		if c.plant != nil {
			name += "/no-q-bias"
		}
		t.Run(name, func(t *testing.T) {
			// Laid out as Metal's load lays it out, and not built yet, so a plant lands before the build.
			m, err := decoder.Load("../../testdata/"+c.fx, decoder.Options{Quant: "int4", SidecarFor: "metal", ResidentContext: 256})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			w := m.Weights()
			if c.plant != nil {
				if len(w.Layers[0].QBias) == 0 || len(w.Layers[0].KBias) == 0 {
					t.Fatalf("VOID: %s holds no Q and K biases to plant on", c.fx)
				}
				c.plant(w)
			}
			r, err := buildResident(m)
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			defer r.Close()
			var requested int64
			for _, ls := range metalLedgerOf(r.d).OpenLeases() {
				if ls.Line == mem.LineDense {
					requested += ls.Requested
				}
			}
			var hostEmbed int64
			if w.LMHead.Rows() != 0 {
				if q, sc, _, ok := w.Embed.Int8(); ok {
					hostEmbed = int64(len(q)) + 4*int64(len(sc))
				} else if f, ok := w.Embed.F32(); ok {
					hostEmbed = 4 * int64(len(f))
				} else {
					t.Fatalf("VOID: the untied embedding is %s, which this test does not size", w.Embed.Kind())
				}
			}
			plan := m.Plan("metal", math.MaxInt64/4, fit.PlanRequest{Ctx: r.ctxCap, CtxPinned: true})
			if got, want := memgate.Planned(plan, r.kvSlotCount())[mem.LineDense]-hostEmbed, requested; got != want {
				t.Errorf("the plan prices %d dense bytes beside the host-read embedding (%d), and the build requests %d (%+d)", got, hostEmbed, want, got-want)
			}
		})
	}
}

// TestMixerWalk_issuesTheBlocksDispatches is OP4m-pod's P-1 and P-2 and OP4m-pod2's (docs/tasks/task-forward-and-memory-2026-11.md,
// "OP4m-pod: registration", "OP4m-pod2: registration"): on every census fixture and option cell whose build goes resident, eight
// decode tokens issue the same pipelines, dispatch for dispatch, and the same logits, bit for bit, under the walk as under the
// blocks. mixerWalkBlocksForTest switches the mixer steps and the FFN and tail steps together, so one comparison covers both. The
// cells that walk a mixer step and the cells that walk an FFN or tail step are counted separately, so a walk nothing reaches cannot
// pass.
func TestMixerWalk_issuesTheBlocksDispatches(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	d, err := sb2census.ReadLayerD("../..", "metal")
	if err != nil {
		t.Fatal(err)
	}
	checked, walked, ffnWalked := 0, 0, 0
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("../..", f)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range sb2census.Cells {
			if !d.Resident[f+" "+c.Name] {
				continue
			}
			row := f + " " + c.Name
			o := decoder.Options{Backend: "metal", ResidentContext: 256, Quant: c.Quant, ActQuantGroup: c.ActGroup, KVPrecision: c.KVPrec, KVQuant: c.KVQuant}
			if c.SSM != "" {
				o.Knobs = &decoder.Knobs{"GOINFER_SSM_RESIDENT": c.SSM}
			}
			m, err := decoder.Load(path, o)
			if err != nil {
				continue
			}
			a, ok := m.ResidentForwardForTest().(*metalResident)
			if !ok {
				m.Close()
				continue
			}
			if slices.ContainsFunc(a.r.layers, func(L residLayer) bool { return slices.ContainsFunc(L.steps, mixerWalks) }) {
				walked++
			}
			if slices.ContainsFunc(a.r.layers, func(L residLayer) bool { return slices.ContainsFunc(L.steps, ffnWalks) }) {
				ffnWalked++
			}
			diffs := mixerWalkDiffs(t, m, a)
			m.Close()
			for _, d := range diffs {
				t.Errorf("%s: %s", row, d)
			}
			checked++
		}
	}
	t.Logf("%d resident cells checked, %d of them walk a mixer step, %d an FFN or tail step", checked, walked, ffnWalked)
	if walked < 6 {
		t.Errorf("only %d cells walk a mixer step: a gate the walk barely reaches covers little", walked)
	}
	if ffnWalked < 6 {
		t.Errorf("only %d cells walk an FFN or tail step: a gate the walk barely reaches covers little", ffnWalked)
	}
}

// TestMixerWalk_plantsGoRed: each of P-1's planted defects in the walk differs from the blocks on the fixtures it can show on. A
// residual added again after an output projection that adds it (DeltaNet's out_proj, a fused attention o-proj) is an extra
// dispatch; the input norm ignoring (1 + w) moves the logits wherever a walked step's RMSNorm has it, Gemma's among them, and
// stays quiet elsewhere.
func TestMixerWalk_plantsGoRed(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	reds := map[int]int{}
	for _, f := range []string{"testdata/qwen35-tiny", "decoder/testdata/qwen3_5-tiny-normw", "testdata/qwen3next-tiny", "testdata/olmo_hybrid-tiny",
		"testdata/nemotron-tiny", "testdata/deepseek-tiny", "testdata/llama-tiny", "testdata/gemma2-tiny", "testdata/gemma1-tiny"} {
		m, err := decoder.Load("../../"+f, decoder.Options{Backend: "metal", ResidentContext: 256, Quant: "int4"})
		if err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			m.Close()
			t.Fatalf("%s did not build resident", f)
		}
		fused, addOne := false, false
		for l := range a.r.layers {
			L := &a.r.layers[l]
			for i, st := range L.steps {
				if !mixerWalks(st) {
					continue
				}
				ops := L.stepOps[i]
				fused = fused || st == stepDeltaNet || (st == stepAttention && ops[len(ops)-1].Kind == program.OpResidual &&
					ops[len(ops)-2].Kind == program.OpProjO)
				addOne = addOne || (ops[0].Norm.AddOne && ops[0].Norm.Form == program.NormRMS && !a.r.q4kLane && !(st == stepAttention && a.r.canUseF16Lane(l)))
			}
		}
		for plant, wantRed := range map[int]bool{mixerWalkPlantResidual: fused, mixerWalkPlantAddOne: addOne} {
			mixerWalkPlantForTest = plant
			diffs := mixerWalkDiffs(t, m, a)
			mixerWalkPlantForTest = 0
			if red := len(diffs) > 0; red != wantRed {
				t.Errorf("%s, plant %d: red %v, want %v (%v)", f, plant, red, wantRed, diffs)
			} else if red {
				reds[plant]++
			}
		}
		m.Close()
	}
	t.Logf("fixtures red: the extra residual on %d, (1 + w) ignored on %d", reds[mixerWalkPlantResidual], reds[mixerWalkPlantAddOne])
	if reds[mixerWalkPlantResidual] == 0 {
		t.Error("the extra-residual plant is red on no fixture")
	}
}

// TestFFNWalk_plantsGoRed verifies that each planted defect in the FFN walk differs from the FFN blocks on every fixture whose
// FFN sublayer exercises that path, and stays quiet on fixtures that do not.
func TestFFNWalk_plantsGoRed(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	reds := map[int]int{}
	for _, f := range []string{"testdata/llama-tiny", "testdata/gpt2", "testdata/deepseek-tiny", "testdata/gemma2-tiny",
		"testdata/olmo_hybrid-tiny", "testdata/cohere-tiny", "testdata/nemotron-tiny"} {
		m, err := decoder.Load("../../"+f, decoder.Options{Backend: "metal", ResidentContext: 256, Quant: "int4"})
		if err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			m.Close()
			t.Fatalf("%s did not build resident", f)
		}
		fused, postNorm, parallel := false, false, false
		for l := range a.r.layers {
			L := &a.r.layers[l]
			if !L.walk.HasFFN {
				continue
			}
			switch L.walk.FFN.OutMode {
			case program.OutFusedResidual:
				fused = true
			case program.OutPostNormResidual:
				postNorm = postNorm || L.walk.FFN.Kind == stepDenseFFN
			case program.OutTwoResidual:
				parallel = true
			}
		}
		for plant, wantRed := range map[int]bool{
			ffnWalkPlantResidual:          fused,
			ffnWalkPlantDropPostNorm:      postNorm,
			ffnWalkPlantFreshParallelNorm: parallel,
		} {
			ffnWalkPlantForTest = plant
			diffs := mixerWalkDiffs(t, m, a)
			ffnWalkPlantForTest = 0
			if red := len(diffs) > 0; red != wantRed {
				t.Errorf("%s, plant %d: red %v, want %v (%v)", f, plant, red, wantRed, diffs)
			} else if red {
				reds[plant]++
			}
		}
		m.Close()
	}
	t.Logf("FFN fixtures red: extra residual %d, dropped post-MLP norm %d, fresh parallel norm %d",
		reds[ffnWalkPlantResidual], reds[ffnWalkPlantDropPostNorm], reds[ffnWalkPlantFreshParallelNorm])
	for _, plant := range []int{ffnWalkPlantResidual, ffnWalkPlantDropPostNorm, ffnWalkPlantFreshParallelNorm} {
		if reds[plant] == 0 {
			t.Errorf("FFN walk plant %d is red on no fixture", plant)
		}
	}
}

// mixerWalkDiffs runs eight decode tokens on a under the mixer steps' blocks and then under their walk, and names each token whose
// pipeline sequence or logits differ.
func mixerWalkDiffs(t *testing.T, m *decoder.Model, a *metalResident) []string {
	t.Helper()
	names := a.r.pipelineNames()
	run := func(blocks bool) (seqs [][]string, logits [][]float32) {
		mixerWalkBlocksForTest = blocks
		defer func() { mixerWalkBlocksForTest = false }()
		a.r.stopExec()
		a.Reset()
		var cur []string
		a.r.d.SetDispatchHook(func(pl Pipeline) { cur = append(cur, names[pl]) })
		defer a.r.d.SetDispatchHook(nil)
		for pos := range 8 {
			cur = nil
			lg, err := a.Forward(m.EmbedResidentForTest((pos*7+3)%a.r.V), pos)
			if err != nil {
				t.Fatalf("forward at %d: %v", pos, err)
			}
			seqs, logits = append(seqs, cur), append(logits, slices.Clone(lg))
		}
		a.r.stopExec()
		return seqs, logits
	}
	sb, lb := run(true)
	sw, lw := run(false)
	var out []string
	for pos := range 8 {
		if !slices.Equal(sb[pos], sw[pos]) {
			out = append(out, fmt.Sprintf("token %d: the walk issues %d dispatches and the blocks %d, first apart at %d", pos, len(sw[pos]), len(sb[pos]), firstApart(sb[pos], sw[pos])))
		}
		if !slices.EqualFunc(lb[pos], lw[pos], func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) }) {
			out = append(out, fmt.Sprintf("token %d: the logits differ", pos))
		}
	}
	return out
}

// firstApart is the first index where a and b differ, or the shorter length.
func firstApart(a, b []string) int {
	for i := range min(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}
	return min(len(a), len(b))
}

// TestMemGate_metal is MM7's gate on Metal (docs/tasks/task-subsystems-2026-10.md): every census fixture at every option cell whose
// build goes resident, and each at int4 with two KV slots and the step's blocks on, holds what it leased on each line to the plan
// for the same load. It runs every row in one test, since a filtered run would see known gaps go quiet.
func TestMemGate_metal(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	d, err := sb2census.ReadLayerD("../..", "metal")
	if err != nil {
		t.Fatal(err)
	}
	g := memgate.Gate{Allowance: metalRounding, Gaps: metalMemGaps, GapsPin: metalMemGapsPin}
	built, declined := 0, 0
	run := func(row, path string, o decoder.Options, kvI8 bool) {
		o.Backend, o.ResidentContext = "metal", 256
		read, exactRed, ok := metalGateRow(t, row, path, o, kvI8)
		if !ok {
			declined++
			return
		}
		for _, e := range exactRed {
			t.Error(e)
		}
		built++
		g.Check(t, read)
	}
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("../..", f)
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

// TestMM5_buildRefusesADeclarationItsKernelsDoNotAddress: Metal's build checks each layer's declared rows against what its kernels address
// before it sizes a pool from them, and refuses a mismatch: on the MC3 fixture's layer 0, V declared with one head too many, K at f32, a
// row no pool has, and K declared twice; on qwen35-tiny, a DeltaNet layer, which holds no pool, declaring K and V. The declarations as
// built pass.
func TestMM5_buildRefusesADeclarationItsKernelsDoNotAddress(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	m, err := decoder.Load(mc3FixtureWriter(t, 4096), decoder.Options{Quant: mc3TestQuant, ResidentContext: 2048})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	all := slices.Clone(m.LayerBuffers(0))
	var orig, rest []program.Buf // the per-position rows the plants edit, and the rest of the declaration, kept as it is
	for _, b := range all {
		if b.Unit == program.PerPosition {
			orig = append(orig, b)
		} else {
			rest = append(rest, b)
		}
	}
	if len(orig) != 2 || orig[0].Name != "k" || orig[1].Name != "v" {
		t.Fatalf("layer 0 declares %+v per position; the test needs its K and V", orig)
	}
	edit := func(f func([]program.Buf) []program.Buf) []program.Buf { return append(f(slices.Clone(orig)), rest...) }
	for _, c := range []struct {
		name string
		bufs []program.Buf
	}{
		{"V with one head too many", edit(func(b []program.Buf) []program.Buf { b[1].Heads++; return b })},
		{"K at f32", edit(func(b []program.Buf) []program.Buf { b[0].Prec = program.PrecF32; return b })},
		{"a row no pool has", edit(func(b []program.Buf) []program.Buf {
			return append(b, program.Buf{Name: "x", Unit: program.PerPosition, Prec: program.PrecKV, Elems: b[0].Elems})
		})},
		{"K twice", edit(func(b []program.Buf) []program.Buf { return append(b, b[0]) })},
	} {
		m.SetLayerBuffersForTest(0, c.bufs)
		if r, err := buildResident(m); err == nil {
			r.Close()
			t.Errorf("%s: the build accepted it", c.name)
		}
	}
	m.SetLayerBuffersForTest(0, all)
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("the declaration as built: %v", err)
	}
	r.Close()
	// A layer that holds no pool (qwen35-tiny's DeltaNet layer 0) declaring K and V rows: priced, never allocated.
	q, err := decoder.Load("../../decoder/selftestdata/qwen35-tiny", decoder.Options{Quant: "int4", ResidentContext: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if r, err := buildResident(q); err != nil {
		t.Fatalf("qwen35-tiny as built: %v", err)
	} else {
		r.Close()
	}
	q.SetLayerBuffersForTest(0, append(slices.Clone(q.LayerBuffers(0)), orig...))
	if r, err := buildResident(q); err == nil {
		r.Close()
		t.Error("a DeltaNet layer declaring K and V rows: the build accepted it")
	}
}

// TestMM5b_buildRefusesDecodeRowsItsKernelsDoNotAddress: Metal's build holds the program's declared per-row widths to what its decode
// kernels address (decodeRowsCheck), so a declaration one element short, a row decode holds no buffer for, a missing row, a mixer's
// rows on a model that runs no such mixer, a gate wider than the FFN kernels, and a parallel block's row where none runs or missing
// where one does are each refused, on a dense model, a parallel-block one and Qwen3.5's DeltaNet and gated attention; the
// declaration as built passes.
func TestMM5b_buildRefusesDecodeRowsItsKernelsDoNotAddress(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer func() { decodeRowsPlantForTest = nil }()
	for _, c := range []struct {
		fx     string
		plants map[string]func(map[string]int)
	}{
		{"../../testdata/llama-tiny", map[string]func(map[string]int){
			"ctx a column short":                       func(r map[string]int) { r["ctx"]-- },
			"a row decode holds no buffer for":         func(r map[string]int) { r["kda-q"] = 8 },
			"no logits row":                            func(r map[string]int) { delete(r, "logits") },
			"a DeltaNet row on a model with none":      func(r map[string]int) { r["dn-mixed"] = 8 },
			"a gate wider than the FFN kernels":        func(r map[string]int) { r["gate"] += 8 },
			"no gate row":                              func(r map[string]int) { delete(r, "gate") },
			"gate a column short":                      func(r map[string]int) { r["gate"]-- },
			"v a column short":                         func(r map[string]int) { r["v"]-- },
			"no k or v rows":                           func(r map[string]int) { delete(r, "k"); delete(r, "v") },
			"a shared expert on a dense model":         func(r map[string]int) { r["shared-gate"], r["shared-up"] = 8, 8 },
			"a parallel block's row on a serial model": func(r map[string]int) { r["b"] = r["h"] },
		}},
		{"../../testdata/cohere-tiny", map[string]func(map[string]int){
			"no parallel block's row": func(r map[string]int) { delete(r, "b") },
		}},
		{"../../decoder/selftestdata/qwen35-tiny", map[string]func(map[string]int){
			"the gated row one short": func(r map[string]int) { r["dn-gated"]-- },
			"no [query ‖ gate] row":   func(r map[string]int) { delete(r, "q-gate") },
		}},
	} {
		m, err := decoder.Load(c.fx, decoder.Options{Quant: "int4", ResidentContext: 256})
		if err != nil {
			t.Fatal(err)
		}
		decodeRowsPlantForTest = nil
		if r, err := buildResident(m); err != nil {
			t.Errorf("%s as declared: %v", c.fx, err)
		} else {
			r.Close()
		}
		for name, plant := range c.plants {
			decodeRowsPlantForTest = plant
			if r, err := buildResident(m); err == nil {
				r.Close()
				t.Errorf("%s, %s: the build accepted it", c.fx, name)
			} else if !strings.Contains(err.Error(), "row") {
				t.Errorf("%s, %s: refused for another reason: %v", c.fx, name, err)
			}
		}
		decodeRowsPlantForTest = nil
		m.Close()
	}
}

// TestMM5_kvCheckRows: the check the build runs on each layer's declared rows (kvCheckRows), on every kind of layer, a KV-shared one and a
// layer with no pool included, which no committed fixture builds.
func TestMM5_kvCheckRows(t *testing.T) {
	k := program.Buf{Name: "k", Unit: program.PerPosition, Prec: program.PrecKV, Elems: 32, Heads: 2}
	v := k
	v.Name = "v"
	lat := program.Buf{Name: "latent", Unit: program.PerPosition, Prec: program.PrecF32, Elems: 24}
	st := program.Buf{Name: "state", Unit: program.PerSlot, Prec: program.PrecF32, Elems: 99}
	owner, mla, none := kvLayerWant{kvDim: 32, nKV: 2}, kvLayerWant{latent: 24}, kvLayerWant{}
	with := func(b program.Buf, f func(*program.Buf)) program.Buf { f(&b); return b }
	for _, c := range []struct {
		name string
		bufs []program.Buf
		want kvLayerWant
		ok   bool
	}{
		{"an owner's K and V", []program.Buf{k, v}, owner, true},
		{"an owner's K and V beside state", []program.Buf{k, st, v}, owner, true},
		{"an owner missing V", []program.Buf{k}, owner, false},
		{"V a column short", []program.Buf{k, with(v, func(b *program.Buf) { b.Elems-- })}, owner, false},
		{"V with a head too many", []program.Buf{k, with(v, func(b *program.Buf) { b.Heads++ })}, owner, false},
		{"K at f32", []program.Buf{with(k, func(b *program.Buf) { b.Prec = program.PrecF32 }), v}, owner, false},
		{"K twice", []program.Buf{k, v, k}, owner, false},
		{"an MLA latent", []program.Buf{lat}, mla, true},
		{"an MLA latent at the KV precision", []program.Buf{with(lat, func(b *program.Buf) { b.Prec = program.PrecKV })}, mla, false},
		{"an MLA layer with K and V", []program.Buf{k, v}, mla, false},
		{"a KV-shared layer with none", nil, kvLayerWant{shared: true}, true},
		{"a KV-shared layer declaring K and V", []program.Buf{k, v}, kvLayerWant{shared: true}, false},
		{"a layer with no pool and only state", []program.Buf{st}, none, true},
		{"a layer with no pool declaring K and V", []program.Buf{k, v}, none, false},
	} {
		if _, err := kvCheckRows(0, c.bufs, c.want); (err == nil) != c.ok {
			t.Errorf("%s: kvCheckRows says %v; want ok %v", c.name, err, c.ok)
		}
	}
}

// TestMM5_poolsAreTheDeclaredRows is G-MP2 on Metal (docs/tasks/task-forward-architecture-2026-10.md §4.9b): every census fixture's int4
// build at two slots, with an f16 and an int8 cache, sizes each layer's KV pool from the row its operations declare
// (decoder.Model.LayerBuffers), the null block and every slot's blocks at the cache's element size (f32 for MLA's latent), and a layer
// that declares no row holds no pool of its own. The build's whole KV, every pool, int8's scales and the tables, is what MetalKVBytes
// prices, to the byte: a term priced twice or not at all is red either way.
func TestMM5_poolsAreTheDeclaredRows(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	d, err := sb2census.ReadLayerD("../..", "metal")
	if err != nil {
		t.Fatal(err)
	}
	pools, models, int8s := 0, 0, 0
	for _, f := range d.Fixtures {
		if !d.Resident[f+" int4"] {
			continue
		}
		path, err := sb2census.Path("../..", f)
		if err != nil {
			t.Fatal(err)
		}
		for _, kv := range []string{"", "i8"} {
			m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", KVPrecision: kv, ResidentContext: 256, ResidentKVSlots: 2})
			if err != nil && kv != "" && strings.Contains(err.Error(), "an MLA model keeps its latent cache in f32") {
				continue // the one family Metal does not give an int8 cache
			} else if err != nil {
				t.Fatalf("%s kv=%s: %v", f, kv, err)
			}
			if a, ok := m.ResidentForwardForTest().(*metalResident); ok {
				models++
				if a.r.kvI8 {
					int8s++
				}
				pools += mm5CheckPools(t, f+" kv="+kv, m, a.r)
			}
			m.Close()
		}
	}
	t.Logf("%d pools on %d builds (%d with an int8 cache) held to their declared rows and to MetalKVBytes", pools, models, int8s)
	if pools < 80 || int8s < 10 {
		t.Fatalf("only %d pools and %d int8 builds checked: the test reads too little", pools, int8s)
	}
}

// mm5CheckPools holds r's KV pools to m's declared rows and its whole KV allocation to MetalKVBytes, and returns the pools it checked.
func mm5CheckPools(t *testing.T, f string, m *decoder.Model, r *resident) int {
	t.Helper()
	pools, total := 0, int64(0)
	for l := range r.layers {
		if L := &r.layers[l]; L.kvShared { // it declares no row and reads its source's pool, which is counted there
			for _, b := range m.LayerBuffers(l) {
				if b.Unit == program.PerPosition {
					t.Errorf("%s layer %d reads layer %d's KV but declares a %q row", f, l, L.kvSrc, b.Name)
				}
			}
			if r.kc[l] != r.kc[L.kvSrc] || r.vc[l] != r.vc[L.kvSrc] {
				t.Errorf("%s layer %d reads layer %d's KV but holds pools of its own", f, l, L.kvSrc)
			}
			continue
		}
		row, elem := 0, r.kvElemBytes()
		for _, b := range m.LayerBuffers(l) {
			if b.Unit == program.PerPosition && (b.Name == "k" || b.Name == "latent") {
				row = b.Elems
				if b.Prec == program.PrecF32 {
					elem = 4
				}
			}
		}
		if row == 0 {
			if r.kc[l] != (Buffer{}) {
				t.Errorf("%s layer %d declares no KV row but holds a %d-byte pool", f, l, r.kc[l].Len())
			}
			continue
		}
		want := (1 + kvBlocksFor(r.ctxCap)*r.kvSlotCount()) * decoder.KVBlockRows * row * elem
		if got := r.kc[l].Len(); got != want {
			t.Errorf("%s layer %d: its pool holds %d bytes; the declared %d-element row at %d B over %d slots is %d", f, l, got, row, elem, r.kvSlotCount(), want)
		}
		total += int64(r.kc[l].Len())
		if r.vc[l] != r.kc[l] {
			total += int64(r.vc[l].Len())
		}
		if r.kvI8 && r.ks[l] != (Buffer{}) {
			total += 4 * int64(r.ks[l].Len()+r.vs[l].Len()) // one f32 scale per position and head
		}
		pools++
	}
	if pools > 0 {
		total += 4 * int64(r.kvSlotCount()*r.kvTabBlocks) // the tables, one entry a block
		if want := m.MetalKVBytes(r.kvSlotCount(), r.ctxCap); total != want {
			t.Errorf("%s: the build holds %d bytes of KV, MetalKVBytes(%d, %d) prices %d", f, total, r.kvSlotCount(), r.ctxCap, want)
		}
	}
	return pools
}

// A wrapped upload on the shared bridge: the same tests every resident backend runs (internal/wrappedupload).
func TestWrappedUpload_rows(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	wrappedupload.Rows(t, "metal")
}

func TestWrappedUpload_reuse(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	wrappedupload.Reuse(t, "metal")
}

func TestWrappedUpload_turns(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	wrappedupload.Turns(t, "metal")
}
