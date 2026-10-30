//go:build gpu && goinfer_testhooks

package gpu

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/sb2census"
)

// Layer B-webgpu of the admission and routing census (docs/tasks/task-subsystems-2026-10.md, "SB2 step 2"): WebGPU's own static
// admission, staticResidentDecline, on layer D's fixtures and option cells. A fake backend named webgpu takes the decoder's real
// admission; where that admits, it records staticResidentDecline's answer and declines, so no device is opened. A cell the decoder
// declines is layer D's line (decoder's TestAdmissionCensus) and is not repeated here.

var webgpuCensusUpdate = flag.Bool("sb2census.update", false, "rewrite this architecture's testdata/sb2_webgpu_census.<goarch>.txt.gz")

// censusBackend is the fake: named webgpu, it records the static admission of what the decoder admitted and declines.
type censusBackend struct {
	decoder.Backend
	mu      sync.Mutex
	reached bool
	answer  string
}

func (b *censusBackend) Name() string { return "webgpu" }

// SelfTestEligible keeps the decoder's startup self-test (WebGPU kernels on the device) out of the census: that is a device decision.
func (b *censusBackend) SelfTestEligible() (bool, string) { return false, "admission census" }

func (b *censusBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	ans := "static ok"
	if err := staticResidentDecline(m); err != nil {
		ans = "decline " + err.Error()
	}
	b.mu.Lock()
	b.reached, b.answer = true, ans
	b.mu.Unlock()
	return nil, false, decoder.DeclineResident("admission census: static gates only")
}

func (b *censusBackend) take() (bool, string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	r, a := b.reached, b.answer
	b.reached, b.answer = false, ""
	return r, a
}

// buildWebGPUCensus returns one line per cell that reached WebGPU's build. Every cell layer D built resident must be among them.
func buildWebGPUCensus(t *testing.T) []string {
	t.Helper()
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "")
	// The fit guard's host reading is fixed, so no reading of this machine decides a line; the fake's registered name has no
	// memory probe or build demand of its own, as in layer D.
	t.Cleanup(decoder.SetHostRAMAvailableForTest(48 << 30))
	be := &censusBackend{}
	const reg = "sb2-census-webgpu"
	decoder.RegisterBackend(reg, func() (decoder.Backend, error) {
		cpu, err := decoder.NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	t.Cleanup(func() { decoder.UnregisterBackendForTest(reg) })
	d, err := sb2census.ReadLayerD("..", "webgpu")
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Fixtures) < 20 {
		t.Fatalf("only %d fixtures in layer D's golden: a census that read little covers little", len(d.Fixtures))
	}
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
			// A reached cell is recorded whatever the load did next: the fake's decline moves the KV to the CPU cache, so a KV
			// refusal after the build is the fake's, not the model's.
			if reached {
				lines = append(lines, fmt.Sprintf("B %s %s webgpu %q", f, c.Name, sb2census.Clean(ans)))
			} else if d.Resident[f+" "+c.Name] {
				t.Errorf("%s %s: layer D builds it resident on webgpu, but WebGPU's build was not reached", f, c.Name)
			}
		}
	}
	return lines
}

// TestBackendAdmissionCensus_webgpu holds layer B-webgpu to testdata/sb2_webgpu_census.<goarch>.txt.gz, line for line, built twice.
func TestBackendAdmissionCensus_webgpu(t *testing.T) {
	start := time.Now()
	a, b := buildWebGPUCensus(t), buildWebGPUCensus(t)
	if strings.Join(a, "\n") != strings.Join(b, "\n") {
		t.Fatalf("two builds in one process differ")
	}
	t.Logf("%d census lines in %s", len(a), time.Since(start).Round(time.Millisecond))
	golden := sb2census.Golden("testdata", "sb2_webgpu_census")
	if *webgpuCensusUpdate {
		if err := sb2census.Write(golden, a); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", golden)
		return
	}
	diffs, err := sb2census.Diff(golden, a, 4)
	if err != nil {
		for _, l := range a {
			fmt.Println("SB2WEBGPUCENSUS " + l)
		}
		t.Fatalf("%v: the lines above (prefixed SB2WEBGPUCENSUS) are this architecture's census", err)
	}
	if len(diffs) > 0 {
		t.Errorf("the webgpu admission census differs from %s in %d cell(s):\n%s", golden, len(diffs), strings.Join(diffs[:min(len(diffs), 30)], "\n"))
	}
}

// TestBackendAdmissionCensus_webgpu_plant shows the census can go red: with the quantized-KV layout decline planted off, every
// cell that takes it must differ from the golden, and no other cell may.
func TestBackendAdmissionCensus_webgpu_plant(t *testing.T) {
	golden := sb2census.Golden("testdata", "sb2_webgpu_census")
	clean := buildWebGPUCensus(t)
	const reason = "KV on webgpu (only the generic GQA path does)"
	wantRed := 0
	for _, l := range clean {
		if strings.Contains(l, reason) {
			wantRed++
		}
	}
	if wantRed == 0 {
		t.Fatal("no cell takes the planted decline: the plant has nothing to change")
	}
	defer func(off bool) { quantKVDeclineOffForTest = off }(quantKVDeclineOffForTest)
	quantKVDeclineOffForTest = true
	diffs, err := sb2census.Diff(golden, buildWebGPUCensus(t), 4)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range diffs {
		if !strings.Contains(d, reason) {
			t.Errorf("planted quantized-KV decline off: a cell outside the plant changed: %s", d)
		}
	}
	if len(diffs) != wantRed {
		t.Fatalf("planted quantized-KV decline off: %d cell(s) differ from the golden, want %d", len(diffs), wantRed)
	}
	t.Logf("planted quantized-KV decline off: %d of %d cells red, as expected", len(diffs), len(clean))
}

// planCells is layer D's option cells plus the serving flag --kv=i8, which sets both KV options; layer D's kv-i8 cell sets
// KVQuant alone, which webgpu does not deliver.
var planCells = append(append([]sb2census.Cell(nil), sb2census.Cells...), sb2census.Cell{Name: "int4/kv=i8", Quant: "int4", KVPrec: "i8", KVQuant: "i8"})

// planVsLoadWebGPU compares Plan's webgpu verdict, with the request req builds for each cell, to what the load does on webgpu:
// the decoder's admission, then staticResidentDecline. Plan reads a CPU load of the cell without its KV options, priced against
// far more memory than any fixture needs, so only its eligibility can decline. A cell with no CPU model must not reach the build.
func planVsLoadWebGPU(t *testing.T, req func(c sb2census.Cell) decoder.PlanRequest) (diffs []string, cells, noModel int) {
	t.Helper()
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "")
	t.Cleanup(decoder.SetHostRAMAvailableForTest(48 << 30))
	be := &censusBackend{}
	const reg = "sb2-plan-webgpu"
	decoder.RegisterBackend(reg, func() (decoder.Backend, error) {
		cpu, err := decoder.NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	t.Cleanup(func() { decoder.UnregisterBackendForTest(reg) })
	d, err := sb2census.ReadLayerD("..", "webgpu")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("..", f)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range planCells {
			cells++
			o := decoder.Options{Backend: reg, Quant: c.Quant, ActQuantGroup: c.ActGroup, KVPrecision: c.KVPrec, KVQuant: c.KVQuant}
			if c.SSM != "" {
				o.Knobs = &decoder.Knobs{"GOINFER_SSM_RESIDENT": c.SSM}
			}
			if m, err := decoder.Load(path, o); err == nil {
				m.Close()
			}
			reached, ans := be.take()
			loads := reached && ans == "static ok"
			o.Backend, o.KVPrecision, o.KVQuant = "cpu", "", ""
			cpuM, err := decoder.Load(path, o)
			if err != nil {
				noModel++
				if reached {
					diffs = append(diffs, fmt.Sprintf("%s %s: no CPU model for Plan (%v), but the load reached the build", f, c.Name, err))
				}
				continue
			}
			p := cpuM.Plan("webgpu", 64<<30, req(c))
			cpuM.Close()
			if planned := p.Placement != decoder.PlacementDecline; planned != loads {
				diffs = append(diffs, fmt.Sprintf("%s %s: Plan places=%v (%s), the load goes resident=%v (%s)", f, c.Name, planned, p.Reason, loads, ans))
			}
		}
	}
	return diffs, cells, noModel
}

// TestPlanAgreesWithLoad_webgpu holds Plan's webgpu verdict, with each cell's KV request, to what the load does on webgpu
// (docs/tasks/task-subsystems-2026-10.md, "SB2 — Plan asks Admit", gate 2), and shows the comparison goes red on a Plan that
// ignores the KV request.
func TestPlanAgreesWithLoad_webgpu(t *testing.T) {
	// Plan's KV request is the serving flag's precision (KVPrecision).
	diffs, cells, noModel := planVsLoadWebGPU(t, func(c sb2census.Cell) decoder.PlanRequest {
		return decoder.PlanRequest{Ctx: 4096, KVF16: c.KVPrec == "f16", KVI8: c.KVPrec == "i8"}
	})
	if len(diffs) > 0 {
		t.Errorf("Plan disagrees with the webgpu load in %d of %d cells:\n%s", len(diffs), cells, strings.Join(diffs[:min(len(diffs), 30)], "\n"))
	}
	t.Logf("%d cells: Plan and the webgpu load agree (%d with no CPU model)", cells, noModel)

	// The plant: a Plan asked without the KV request must disagree on the lossy-KV cells of the families webgpu declines, and only there.
	planted, _, _ := planVsLoadWebGPU(t, func(sb2census.Cell) decoder.PlanRequest { return decoder.PlanRequest{Ctx: 4096} })
	if len(planted) == 0 {
		t.Fatal("a Plan that ignores the KV request agrees everywhere: the comparison cannot go red")
	}
	for _, d := range planted {
		if !strings.Contains(d, " int4/kv-f16: ") && !strings.Contains(d, " int4/kv=i8: ") {
			t.Errorf("planted KV-blind Plan: a cell outside the lossy-KV cells disagrees: %s", d)
		}
	}
	t.Logf("planted KV-blind Plan: %d cells red", len(planted))
}

// The slot census (docs/tasks/task-subsystems-2026-10.md, "SB2 memory reads, step 4"): WebGPU's KV-slot decisions (slotsGranted,
// slotsBeforeContextOn) on every cell layer B-webgpu passes, asked for 4 slots, against each recorded machine. A unified-memory
// profile feeds its RAM and available memory; a discrete one feeds its free-VRAM figure less the build's weights and slot 0's KV,
// a derived figure that G-WS3 checks against what Vulkan leaves.

var webgpuSlotCensusUpdate = flag.Bool("sb2slotcensus.update", false, "rewrite this architecture's testdata/sb2_webgpu_slot_census.<goarch>.txt.gz")

// slotCensusSlots is the slot request every census cell makes, and slotCensusBound the figure's error bound the marks take.
const (
	slotCensusSlots = 4
	slotCensusBound = 64 << 20
)

// slotProfile is a recorded machine with a figure the slot decision reads: unified memory (metal_budget_bytes set) or a discrete
// GPU's free VRAM (cuda_free_bytes).
type slotProfile struct {
	Name        string `json:"name"`
	Physical    int64  `json:"physical_bytes"`
	Available   int64  `json:"available_bytes"`
	CUDAFree    *int64 `json:"cuda_free_bytes"`
	MetalBudget *int64 `json:"metal_budget_bytes"`
}

func slotProfiles(t *testing.T) []slotProfile {
	t.Helper()
	paths, err := filepath.Glob("../testdata/machines/*.json")
	if err != nil {
		t.Fatal(err)
	}
	var ps []slotProfile
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var m slotProfile
		if err := json.Unmarshal(b, &m); err != nil || m.Name == "" {
			t.Fatalf("%s: not a machine profile: %v", p, err)
		}
		if m.CUDAFree != nil || m.MetalBudget != nil {
			ps = append(ps, m)
		}
	}
	if len(ps) == 0 {
		t.Fatal("no machine profile carries a VRAM or unified-memory figure")
	}
	return ps
}

// slotCensusLine is m's slot decision against pr: the context and the slots granted, each marked ~ where the profile's figure moved
// by slotCensusBound either way moves it.
func slotCensusLine(m *decoder.Model, pr slotProfile) string {
	kvDelivered := m.KVDeliveredFor("webgpu")
	kvF16, kvI8 := kvDelivered == "f16", kvDelivered == "i8"
	want := m.ResidentKVSlotsRequest()
	before := webgpuCtxBeforeSlots(m, kvF16, kvI8)
	perPos := kvBytesPerPosition(m, kvF16, kvI8)
	mark := func(v int, others ...int) string {
		for _, o := range others {
			if o != v {
				return "~"
			}
		}
		return ""
	}
	if pr.MetalBudget != nil {
		weights, hostCopy := m.ResidentWeightBytes(), m.ResidentHostCopyBytes(0)
		decide := func(avail int64) (int, int) {
			ctx := slotsBeforeContextOn(true, func() int64 { return pr.Physical }, m, before, kvF16, kvI8, avail)
			return ctx, slotsGranted(want, true, pr.Physical, avail, weights, hostCopy, perPos*int64(ctx), 0, false)
		}
		ctx, n := decide(pr.Available)
		ctxLo, nLo := decide(pr.Available - slotCensusBound)
		ctxHi, nHi := decide(pr.Available + slotCensusBound)
		return fmt.Sprintf("%s unified ram=%d(profile) avail0=%d(profile) weights=%d hostCopy=%d ctx=%d%s perSlot=%d slots=%d%s/%d",
			pr.Name, pr.Physical, pr.Available, weights, hostCopy, ctx, mark(ctx, ctxLo, ctxHi), perPos*int64(ctx), n, mark(n, nLo, nHi), want)
	}
	// A discrete GPU: the context is not priced (slotsBeforeContextOn), and the figure is the profile's less what the build holds
	// once slot 0 is up (slotCensusBuildBytes).
	weights := slotCensusWeightBytes(m)
	ctx := slotsBeforeContextOn(false, func() int64 { return pr.Physical }, m, before, kvF16, kvI8, 0)
	perSlot := perPos * int64(ctx)
	free := *pr.CUDAFree - weights - perSlot
	n := slotsGranted(want, false, 0, 0, 0, 0, perSlot, free, true)
	nLo := slotsGranted(want, false, 0, 0, 0, 0, perSlot, free-slotCensusBound, true)
	nHi := slotsGranted(want, false, 0, 0, 0, 0, perSlot, free+slotCensusBound, true)
	return fmt.Sprintf("%s discrete free=%d(profile) weights=%d ctx=%d perSlot=%d after-slot0=%d(derived; G-WS3 band -77/+35 MiB) slots=%d%s/%d",
		pr.Name, *pr.CUDAFree, weights, ctx, perSlot, free, n, mark(n, nLo, nHi), want)
}

// slotCensusWeightBytes is the device weights the slot census takes off a discrete figure, as the planner prices them for webgpu: the
// dense weights, the routed experts and the allocation slack. With slot 0's KV it is what G-WS3 holds against the build's live bytes.
func slotCensusWeightBytes(m *decoder.Model) int64 {
	return m.ResidentDenseWeightBytesFor("webgpu") + max(0, m.ResidentWeightBytesPaged(0)-m.ResidentDenseWeightBytes()) + m.PlanSlack("webgpu")
}

// slotCensusBackend is the fake for the slot census: where WebGPU's static gates pass, it records one line per profile and declines.
type slotCensusBackend struct {
	decoder.Backend
	profiles []slotProfile
	mu       sync.Mutex
	lines    []string
	reached  bool
}

func (b *slotCensusBackend) Name() string                     { return "webgpu" }
func (b *slotCensusBackend) SelfTestEligible() (bool, string) { return false, "slot census" }

func (b *slotCensusBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	var lines []string
	if staticResidentDecline(m) == nil {
		for _, pr := range b.profiles {
			lines = append(lines, slotCensusLine(m, pr))
		}
	}
	b.mu.Lock()
	b.reached, b.lines = true, lines
	b.mu.Unlock()
	return nil, false, decoder.DeclineResident("slot census")
}

// buildWebGPUSlotCensus returns one line per cell layer B-webgpu passes and profile, each cell asking for slotCensusSlots slots.
func buildWebGPUSlotCensus(t *testing.T) (lines []string, passed int) {
	t.Helper()
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "")
	t.Cleanup(decoder.SetHostRAMAvailableForTest(48 << 30))
	be := &slotCensusBackend{profiles: slotProfiles(t)}
	const reg = "sb2-slotcensus-webgpu"
	decoder.RegisterBackend(reg, func() (decoder.Backend, error) {
		cpu, err := decoder.NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	t.Cleanup(func() { decoder.UnregisterBackendForTest(reg) })
	d, err := sb2census.ReadLayerD("..", "webgpu")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("..", f)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range sb2census.Cells {
			o := decoder.Options{Backend: reg, Quant: c.Quant, ActQuantGroup: c.ActGroup, KVPrecision: c.KVPrec, KVQuant: c.KVQuant, ResidentKVSlots: slotCensusSlots}
			if c.SSM != "" {
				o.Knobs = &decoder.Knobs{"GOINFER_SSM_RESIDENT": c.SSM}
			}
			m, err := decoder.Load(path, o)
			if err == nil {
				m.Close()
			}
			be.mu.Lock()
			reached, ls := be.reached, be.lines
			be.reached, be.lines = false, nil
			be.mu.Unlock()
			if !reached || len(ls) == 0 {
				continue
			}
			passed++
			for _, l := range ls {
				lines = append(lines, fmt.Sprintf("S %s %s webgpu %s", f, c.Name, sb2census.Clean(l)))
			}
		}
	}
	return lines, passed
}

// TestBackendSlotCensus_webgpu is G-WS2: the slot census, built twice and equal, held to testdata/sb2_webgpu_slot_census.<goarch>.txt.gz,
// one line per passing cell and profile.
func TestBackendSlotCensus_webgpu(t *testing.T) {
	a, passed := buildWebGPUSlotCensus(t)
	b, _ := buildWebGPUSlotCensus(t)
	if strings.Join(a, "\n") != strings.Join(b, "\n") {
		t.Fatal("two builds in one process differ")
	}
	if want := passed * len(slotProfiles(t)); passed == 0 || len(a) != want {
		t.Fatalf("%d slot census lines, want %d (%d passing cells x %d profiles)", len(a), want, passed, len(slotProfiles(t)))
	}
	marked, granted := 0, 0
	for _, l := range a {
		if strings.Contains(l, "~") {
			marked++
		}
		if !strings.Contains(l, " slots=1/") && !strings.Contains(l, " slots=1~/") {
			granted++
		}
	}
	t.Logf("%d slot census lines (%d passing cells); %d grant more than one slot; %d carry a decision the 64 MiB bound cannot settle (~)", len(a), passed, granted, marked)
	golden := sb2census.Golden("testdata", "sb2_webgpu_slot_census")
	if *webgpuSlotCensusUpdate {
		if err := sb2census.Write(golden, a); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", golden)
		return
	}
	diffs, err := sb2census.Diff(golden, a, 4)
	if err != nil {
		for _, l := range a {
			fmt.Println("SB2WEBGPUSLOTCENSUS " + l)
		}
		t.Fatalf("%v: the lines above (prefixed SB2WEBGPUSLOTCENSUS) are this architecture's slot census", err)
	}
	if len(diffs) > 0 {
		t.Errorf("the webgpu slot census differs from %s in %d line(s):\n%s", golden, len(diffs), strings.Join(diffs[:min(len(diffs), 30)], "\n"))
	}
}

// TestBackendSlotCensus_webgpu_plant is G-WS2's plant: with the headroom raised past any profile, every discrete line that granted
// more than one slot grants one, and no unified line changes.
func TestBackendSlotCensus_webgpu_plant(t *testing.T) {
	clean, _ := buildWebGPUSlotCensus(t)
	defer func(v int64) { slotHeadroomPlantForTest = v }(slotHeadroomPlantForTest)
	slotHeadroomPlantForTest = 1 << 50
	planted, _ := buildWebGPUSlotCensus(t)
	if len(planted) != len(clean) {
		t.Fatalf("planted headroom: %d lines, want %d", len(planted), len(clean))
	}
	red := 0
	for i := range clean {
		discrete := strings.Contains(clean[i], " discrete ")
		more := !strings.Contains(clean[i], " slots=1/") && !strings.Contains(clean[i], " slots=1~/")
		switch {
		case discrete && more:
			red++
			if planted[i] == clean[i] || !strings.Contains(planted[i], " slots=1/") {
				t.Errorf("planted headroom: a discrete line that granted more than one does not grant one: %s", planted[i])
			}
		case !discrete && planted[i] != clean[i]:
			t.Errorf("planted headroom: a unified line changed:\n%s\n%s", clean[i], planted[i])
		}
	}
	if red == 0 {
		t.Fatal("planted headroom: no discrete line granted more than one slot: the plant has nothing to change")
	}
	t.Logf("planted headroom: all %d discrete lines that granted more than one now grant one; unified lines unchanged", red)
}
