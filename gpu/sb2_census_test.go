//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
	"github.com/townsendmerino/goinfer/internal/mem"
	"github.com/townsendmerino/goinfer/internal/memgate"
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
	// A discrete GPU: the context is not priced (slotsBeforeContextOn). Where the profile's CUDA context is measured, newKVSlot's
	// outcome is predicted with M1′ (vkSlots) against the Vulkan blocks the profile's card holds; elsewhere the figure is the profile's
	// less what the build holds once slot 0 is up, within G-WS3's band.
	weights := slotCensusWeightBytes(m)
	ctx := slotsBeforeContextOn(false, func() int64 { return pr.Physical }, m, before, kvF16, kvI8, 0)
	perSlot := perPos * int64(ctx)
	if cudaCtx, ok := vkCUDAContext[pr.Name]; ok {
		slot0, slotKV := vkPlanBuffers(m, ctx, kvF16, kvI8)
		capacity := *pr.CUDAFree + cudaCtx - vkDeviceConst - vkHostKept - vkCapacityPlantForTest
		if vkCapacityTightForTest {
			_, use := vkSlots(1, slot0, slotKV, 0)
			capacity = use - vkHostKept - vkDeviceConst + vkAligned(webgpuSlotHeadroom) - 1
		}
		n, use := vkSlots(want, slot0, slotKV, capacity)
		nLo, _ := vkSlots(want, slot0, slotKV, capacity-slotCensusBound)
		nHi, _ := vkSlots(want, slot0, slotKV, capacity+slotCensusBound)
		return fmt.Sprintf("%s discrete free=%d(profile) cudaCtx=%d capacity=%d(M1′) weights=%d ctx=%d perSlot=%d slot0=%d(M1′) slots=%d%s/%d",
			pr.Name, *pr.CUDAFree, cudaCtx, capacity, weights, ctx, perSlot, use, n, mark(n, nLo, nHi), want)
	}
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

// M1′, the model of Vulkan's device memory that holds on nobara (docs/measurements/sb2-vk-2026-10-31/README.md, "M1′'s held-out
// pass"): gpu-allocator 0.28.0's placement under wgpu-native's Performance hints, one 64 MiB host-visible block, and 144 MiB for
// the device itself. score_vk1.py's m1p_predict is the same model; TestVkM1p_matchesTheScorer holds the two to the byte.
const (
	vkMiB         = int64(1) << 20
	vkInternal    = 512<<10 + 256 + 256 // wgpu-core's own per-device buffers: the zero buffer, external-texture params, indirect validation
	vkAlign       = 256                 // the driver's buffer alignment, assumed (M1's source reading)
	vkDevFirst    = 128 * vkMiB
	vkDevMax      = 256 * vkMiB
	vkHostFirst   = 64 * vkMiB
	vkHostMax     = 128 * vkMiB
	vkHostKept    = 64 * vkMiB  // gpu-allocator keeps a memory type's last block: the upload staging's
	vkDeviceConst = 144 * vkMiB // fitted on G-VK1's six loads, judged on five held-out
)

// vkCUDAContext is the CUDA context a profile's cuda_free_bytes was read inside: the probe's figure excludes it and a Vulkan load does
// not pay it. Measured on nobara's card (G-VK1); a discrete profile not listed keeps the census's previous line.
var vkCUDAContext = map[string]int64{"nobara-desktop": 105 * vkMiB, "nobara-no-desktop": 105 * vkMiB}

// vkCapacityPlantForTest is subtracted from every restated discrete capacity, and vkCapacityTightForTest sets each capacity one byte
// short of slot 0's blocks and the probe: G-VR2's plants.
var (
	vkCapacityPlantForTest int64
	vkCapacityTightForTest bool
)

// vkAligned is a buffer's size as gpu-allocator is asked for it: wgpu-core's 4-byte padding, then the driver's alignment.
func vkAligned(n int64) int64 {
	n = (max(n, 4) + 3) / 4 * 4
	return (n + vkAlign - 1) / vkAlign * vkAlign
}

// vkPacker is gpu-allocator's placement on one memory type: a request larger than the current block size gets a block of its own,
// of exactly its size; any other goes to the first open block it fits, newest first, or opens a block. total is the bytes allocated.
type vkPacker struct {
	first, max int64
	free       []int64
	total      int64
}

func (p *vkPacker) place(r int64) {
	mb := min(p.first<<min(len(p.free), 7), p.max)
	if r > mb {
		p.total += r
		return
	}
	for i := len(p.free) - 1; i >= 0; i-- {
		if p.free[i] >= r {
			p.free[i] -= r
			return
		}
	}
	p.free = append(p.free, mb-r)
	p.total += mb
}

func (p vkPacker) clone() vkPacker {
	p.free = append([]int64(nil), p.free...)
	return p
}

// vkM1p is M1′'s device use for buffers placed in order, on a device with a host-visible device-local memory type (nobara's BAR
// heap). mapRead buffers live in host memory and are not counted; mapWrite buffers go to the host-visible type.
func vkM1p(sizes []int64, mapRead, mapWrite []bool) int64 {
	dev := vkPacker{first: vkDevFirst, max: vkDevMax}
	host := vkPacker{first: vkHostFirst, max: vkHostMax}
	dev.place(vkInternal)
	for i, n := range sizes {
		switch {
		case mapRead != nil && mapRead[i]:
		case mapWrite != nil && mapWrite[i]:
			host.place(vkAligned(n))
		default:
			dev.place(vkAligned(n))
		}
	}
	return dev.total + max(host.total, vkHostKept) + vkDeviceConst
}

// vkPlanBuffers is the buffer list a WebGPU build of m makes, from the model alone: the planner's own list
// (decoder.Model.WebGPUDeviceBuffersForTest), so the census and the plan price the same buffers. slotKV is one further slot's K and V.
func vkPlanBuffers(m *decoder.Model, ctx int, kvF16, kvI8 bool) (slot0, slotKV []int64) {
	return m.WebGPUDeviceBuffersForTest(ctx, kvF16, kvI8)
}

// vkSlots is newKVSlot's outcome as M1′ predicts it on a device that can hold capacity bytes of Vulkan blocks: slot 0 is placed,
// then each further slot's K and V buffers and the headroom probe, which takes a block of its own and is freed; a slot is granted
// while the blocks and the probe fit. use is M1′'s device use with slot 0 alone.
func vkSlots(want int, slot0, slotKV []int64, capacity int64) (granted int, use int64) {
	dev := vkPacker{first: vkDevFirst, max: vkDevMax}
	dev.place(vkInternal)
	for _, n := range slot0 {
		dev.place(vkAligned(n))
	}
	use = dev.total + vkHostKept + vkDeviceConst
	headroom := int64(webgpuSlotHeadroom)
	if slotHeadroomPlantForTest > 0 {
		headroom = slotHeadroomPlantForTest
	}
	granted = 1
	for s := 1; s < want; s++ {
		next := dev.clone()
		for _, n := range slotKV {
			next.place(vkAligned(n))
		}
		if next.total+vkAligned(headroom) > capacity {
			break
		}
		dev = next
		granted++
	}
	return granted, use
}

// TestVkM1p_matchesTheScorer is G-VR0: on G-VK1's six recorded buffer lists, the census's packer and the planner's both give
// score_vk1.py's m1p_predict to the byte.
func TestVkM1p_matchesTheScorer(t *testing.T) {
	want := map[string]int64{ // python3 score_vk1.py's m1p_predict(bufs, True) on each pass1-<tag>-probe-buffers.tsv
		"llama1b": 2231369728, "mistral7b": 6794772480, "qwen0.5b": 1157627904,
		"qwen1.5b-ctx4096": 1694498816, "qwen1.5b": 2231369728, "qwen7b": 6802898944,
	}
	dir := "../docs/measurements/sb2-vk-2026-10-31"
	for tag, w := range want {
		b, err := os.ReadFile(filepath.Join(dir, "pass1-"+tag+"-probe-buffers.tsv"))
		if err != nil {
			t.Fatal(err)
		}
		var sizes []int64
		var mr, mw []bool
		for _, row := range strings.Split(strings.TrimSpace(string(b)), "\n")[1:] {
			f := strings.Split(row, "\t")
			n, err1 := strconv.ParseInt(f[1], 10, 64)
			u, err2 := strconv.ParseUint(strings.TrimPrefix(f[2], "0x"), 16, 64)
			if err1 != nil || err2 != nil {
				t.Fatalf("%s: %q: %v %v", tag, row, err1, err2)
			}
			sizes, mr, mw = append(sizes, n), append(mr, u&1 != 0), append(mw, u&2 != 0)
		}
		if got := vkM1p(sizes, mr, mw); got != w {
			t.Errorf("%s: Go M1′ %d, the scorer %d (%+d)", tag, got, w, got-w)
		}
		var dev []int64 // the planner's packer takes the device-local buffers; these lists hold no host-visible upload buffer
		for i, n := range sizes {
			if mw[i] {
				t.Fatalf("%s: a host-visible upload buffer in the list: the planner's packer does not take one", tag)
			}
			if !mr[i] {
				dev = append(dev, n)
			}
		}
		if got := decoder.WebGPUVkBlocksForTest(dev) + decoder.WebGPUVkFixed; got != w {
			t.Errorf("%s: the planner's M1′ %d, the scorer %d (%+d)", tag, got, w, got-w)
		}
	}
}

// TestBackendSlotCensus_webgpu_m1p is G-VR2's restatement and plants: every discrete line on a profile with a measured CUDA context
// carries M1′'s figures; a capacity 1 GiB lower changes discrete lines only; a capacity one byte short of slot 0 and the probe turns
// every discrete line that granted more than one slot to one, and changes no unified line.
func TestBackendSlotCensus_webgpu_m1p(t *testing.T) {
	clean, _ := buildWebGPUSlotCensus(t)
	restated, multi := 0, 0
	for _, l := range clean {
		f := strings.Fields(l)
		if len(f) < 6 || f[5] != "discrete" {
			continue
		}
		if _, ok := vkCUDAContext[f[4]]; ok {
			restated++
			if !strings.Contains(l, " capacity=") || !strings.Contains(l, "(M1′)") {
				t.Errorf("a discrete line on a profile with a measured CUDA context is not restated: %s", l)
			}
			if !strings.Contains(l, " slots=1/") && !strings.Contains(l, " slots=1~/") {
				multi++
			}
		}
	}
	if restated == 0 || multi == 0 {
		t.Fatalf("%d restated discrete lines, %d granting more than one slot: nothing for the plants to change", restated, multi)
	}
	plant := func(name string, set func(), undo func()) []string {
		set()
		defer undo()
		p, _ := buildWebGPUSlotCensus(t)
		if len(p) != len(clean) {
			t.Fatalf("%s: %d lines, want %d", name, len(p), len(clean))
		}
		for i := range clean {
			if !strings.Contains(clean[i], " discrete ") && p[i] != clean[i] {
				t.Errorf("%s: a unified line changed:\n%s\n%s", name, clean[i], p[i])
			}
		}
		return p
	}
	lower := plant("capacity-1GiB", func() { vkCapacityPlantForTest = 1 << 30 }, func() { vkCapacityPlantForTest = 0 })
	changed := 0
	for i := range clean {
		if lower[i] != clean[i] {
			changed++
		}
	}
	if changed == 0 {
		t.Errorf("capacity-1GiB: no discrete line changed")
	}
	tight := plant("tight", func() { vkCapacityTightForTest = true }, func() { vkCapacityTightForTest = false })
	turned := 0
	for i := range clean {
		if !strings.Contains(clean[i], " capacity=") || strings.Contains(clean[i], " slots=1/") || strings.Contains(clean[i], " slots=1~/") {
			continue
		}
		if !strings.Contains(tight[i], " slots=1/") && !strings.Contains(tight[i], " slots=1~/") {
			t.Errorf("tight: a restated line that granted more than one does not grant one: %s", tight[i])
			continue
		}
		turned++
	}
	t.Logf("%d discrete lines restated (%d grant more than one slot); capacity-1GiB changed %d; tight turned %d to one slot", restated, multi, changed, turned)
}

// webgpuRounding names Vulkan's block allocator. The ledger charges each buffer its requested size, so today it forgives nothing: the
// blocks are outside what a row reads (webgpuVkBlocks prices them in the slot census).
var webgpuRounding = memgate.PowerOfTwo("Vulkan's power-of-two blocks")

// webgpuDenseVectorFixtures are the fixtures whose dense line reads over plan: a tied head uploads the table the plan counts, so the
// host-kept table that hides the vectors on other fixtures is not there to hide them.
var webgpuDenseVectorFixtures = []struct {
	row      string
	perLayer int
}{
	{"selftestdata/gemma3-vl-tiny", 4},
	{"testdata/gemma1-tiny", 2},
	{"testdata/gemma1-tiny.gguf", 2},
	{"testdata/gemma3-vl-tiny", 4},
	{"testdata/gptoss_tiny.gguf", 1},
	{"testdata/qwen3asr-tiny", 0},
	{"testdata/smollm3-tiny", 2},
}

// webgpuMemGaps is WebGPU's table of known reds. It may only shrink: the fix that removes a row lowers webgpuMemGapsPin.
var webgpuMemGaps = func() []memgate.Gap {
	var out []memgate.Gap
	for _, v := range webgpuDenseVectorFixtures {
		out = append(out, memgate.Gap{Row: v.row + " *", Line: mem.LineDense, What: "the plan's dense figure sums weight matrices: the norms, biases, rotation tables and small constant buffers are leased on dense and priced nowhere",
			Item: "MM8, dense vectors", Bytes: func(f map[string]int64) int64 {
				return int64(v.perLayer)*f["vectorUnit"] + f["vectorBase"] + f["tables"]
			}})
	}
	moeDense := func(row string, over int64) {
		out = append(out, memgate.Gap{Row: row, Line: mem.LineDense, What: "Gemma 4's MoE block leases router and expert tables on dense beside the vectors, priced nowhere: not diagnosed, so it names this row's own bytes",
			Item: "MM8, Gemma 4 MoE dense", Bytes: func(map[string]int64) int64 { return over }})
	}
	moeDense("testdata/gemma4-moe-kv-tiny int8int8", 9768)
	moeDense("testdata/gemma4-moe-kv-tiny32 int8int8", 9768)
	out = append(out, memgate.Gap{Row: "testdata/deepseek-tiny *", Line: mem.LineKV, OverPriced: true, What: "an MLA layer holds one latent row, and the plan prices K and V",
		Item: "MM8, MLA KV counted twice", Bytes: memgate.KVPricedOver})
	for _, f := range []string{
		"testdata/olmo3-tiny", "testdata/olmo_hybrid-tiny", "testdata/smollm3-tiny",
		"internal/vision/testdata/qwen25vl-vision-tiny", "selftestdata/gemma3-vl-tiny", "selftestdata/llama-tiny",
		"testdata/gemma3-vl-tiny", "testdata/gguf-split/glm-tiny-00001-of-00004.gguf", "testdata/glm-tiny",
		"testdata/glm-tiny-bias", "testdata/glm-tiny.gguf", "testdata/gptoss_tiny.gguf",
		"testdata/granite-dense-tiny", "testdata/internlm2-tiny", "testdata/internlm3-tiny",
		"testdata/llama-tiny", "testdata/mixtral-tiny", "testdata/qwen25vl-tiny",
		"testdata/qwen3asr-tiny", "testdata/qwen3moe-tiny", "testdata/qwen3moe-tiny-k3",
		"testdata/qwen3vlmoe-tiny", "testdata/voxtral-tiny",
	} {
		out = append(out, memgate.Gap{Row: f + " int4/kvprecision-i8", Line: mem.LineKV, What: "int8 K/V holds one scale a head, and the plan prices 1/8 of a byte an element, which under-prices a narrow head",
			Item: "MM8, int8 KV's scale", Bytes: memgate.KVPricedUnder})
	}
	for _, f := range []string{"testdata/gemma4-moe-kv-tiny", "testdata/gemma4-moe-kv-tiny32", "testdata/llama-attnfa-tiny"} {
		out = append(out, memgate.Gap{Row: f + " int4/kvprecision-i8", Line: mem.LineKV, OverPriced: true, What: "int8 K/V holds one scale a head, and the plan prices 1/8 of a byte an element, which over-prices a wide head",
			Item: "MM8, int8 KV's scale", Bytes: memgate.KVPricedOver})
	}
	stacked := func(row string, over int64) {
		out = append(out, memgate.Gap{Row: row, Line: mem.LineExperts, What: "the stacked experts lease more than the plan's expert figure: not diagnosed, so it names this row's own bytes",
			Item: "MM8 WebGPU, stacked experts", Bytes: func(map[string]int64) int64 { return over }})
	}
	stacked("testdata/qwen3vlmoe-tiny int8int8", 16384)
	if runtime.GOARCH != "arm64" { // gpu-darwin's run fires none of these int4 rows: not diagnosed, and not shown to be the architecture rather than the adapter
		moeDense("testdata/gemma4-moe-kv-tiny int4*", 10792)
		moeDense("testdata/gemma4-moe-kv-tiny32 int4*", 9768)
		stacked("testdata/gemma4-moe-kv-tiny int4*", 8192)
		stacked("testdata/qwen3vlmoe-tiny int4*", 8192)
	}
	return out
}()

// webgpuGateRowsPath names every row that must build resident, and webgpuGateRowsUpdate has TestMemGate_webgpu rewrite it.
const webgpuGateRowsPath = "testdata/memgate_rows.txt"

var webgpuGateRowsUpdate = flag.Bool("memgate.rows.update", false, "have TestMemGate_webgpu rewrite testdata/memgate_rows.txt")

// webgpuMemGapsPin is len(webgpuMemGaps) on this architecture.
var webgpuMemGapsPin = map[string]int{"arm64": 37}[runtime.GOARCH] + map[bool]int{true: 41}[runtime.GOARCH != "arm64"]

// webgpuGatePlantForTest, when set, runs on a row's resident after its generation and before the ledger is read; the function it returns
// runs after that read, so the plant is released before Close and the package's own ledger check stays clean.
var webgpuGatePlantForTest func(rd *residentDecoder) (release func())

// webgpuGateRow loads fixture path at options o, runs it, and returns the row: the build's ledger after the build and after a short
// generation, the plan for the same load, and the ledger after Close. ok is false when WebGPU's own build declines, or, on a row outside the census
// (census false), when the load refuses the option.
func webgpuGateRow(t *testing.T, row, path string, o decoder.Options, census bool) (memgate.Reading, bool) {
	t.Helper()
	m, err := decoder.Load(path, o)
	if err != nil {
		if census {
			t.Errorf("%s: layer D builds it resident, and the load failed: %v", row, err)
		}
		return memgate.Reading{}, false
	}
	closed := false
	defer func() {
		if !closed {
			m.Close()
		}
	}()
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		return memgate.Reading{}, false
	}
	l := rd.c.Ledger()
	read := memgate.Reading{Row: row, KVSlots: rd.KVSlots()}
	kv := m.KVDeliveredFor("webgpu") // what the build allocated: Options.KVQuant is the CPU's, and a family can refuse what was asked
	read.Plan = m.Plan("webgpu", math.MaxInt64/4, decoder.PlanRequest{Ctx: rd.ctxCap, CtxPinned: true, KVF16: kv == mem.KVF16, KVI8: kv == mem.KVI8})
	read.Exact = []mem.LineID{mem.LineKV, mem.LineState}
	facts := memgate.GeometryOf(m, rd.ctxCap, rd.KVSlots(), kv, false).Facts()
	read.Open = append(read.Open, memgate.Snap("built", l, facts))
	ch, g := m.Generate(context.Background(), []int{1, 2, 3, 4, 5}, 4, decoder.SamplingParams{})
	for range ch {
	}
	if err := g.Err(); err != nil {
		t.Errorf("%s: generation: %v", row, err)
	}
	release := func() {}
	if webgpuGatePlantForTest != nil {
		release = webgpuGatePlantForTest(rd)
	}
	read.Open = append(read.Open, memgate.Snap("generated", l, facts))
	release()
	// Scratch has no plan line: it is the headroom WebGPU's slot rule leaves on the device for the per-call buffers (webgpuSlotHeadroom).
	read.Named = map[mem.LineID]int64{mem.LineScratch: webgpuSlotHeadroom}
	closed = true
	m.Close()
	read.Closed = l.Usage()
	return read, true
}

// TestMemGate_webgpu is MM7's gate on WebGPU (docs/tasks/task-subsystems-2026-10.md): every census fixture at every option cell whose
// build goes resident, and each at int4 with two KV slots. It runs every row in one test, since a filtered run would see known gaps go quiet.
func TestMemGate_webgpu(t *testing.T) {
	newOrSkip(t).Close()
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	d, err := sb2census.ReadLayerD("..", "webgpu")
	if err != nil {
		t.Fatal(err)
	}
	g := memgate.Gate{Allowance: webgpuRounding, Gaps: webgpuMemGaps, GapsPin: webgpuMemGapsPin}
	var built []string
	declined := 0
	run := func(row, path string, o decoder.Options, census bool) {
		o.Backend, o.ResidentContext = "webgpu", 256
		read, ok := webgpuGateRow(t, row, path, o, census)
		if !ok {
			declined++
			return
		}
		built = append(built, row)
		g.Check(t, read)
	}
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("..", f)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range sb2census.Cells {
			if !d.Resident[f+" "+c.Name] || c.KVQuant != "" { // KVQuant is the CPU's: the row would be int4's again
				continue
			}
			o := decoder.Options{Quant: c.Quant, ActQuantGroup: c.ActGroup, KVPrecision: c.KVPrec}
			if c.SSM != "" {
				o.Knobs = &decoder.Knobs{"GOINFER_SSM_RESIDENT": c.SSM}
			}
			run(f+" "+c.Name, path, o, true)
		}
		if d.Resident[f+" int4"] {
			run(f+" int4/slots2", path, decoder.Options{Quant: "int4", ResidentKVSlots: 2}, false)
			// The census's cells ask the CPU's KVQuant for int8: this is the GPU's own.
			run(f+" int4/kvprecision-i8", path, decoder.Options{Quant: "int4", KVPrecision: "i8"}, false)
		}
	}
	t.Logf("%d builds held, %d options declined by WebGPU's own build", len(built), declined)
	memgate.HoldRows(t, webgpuGateRowsPath, built, *webgpuGateRowsUpdate)
	g.Finish(t)
}

// TestMemGate_webgpuPlantsThroughTheTable: a buffer really allocated over plan after the generation stays red through the table, on a row whose
// line a live gap forgives part of (each is checked to fire unplanted), and on a line nothing forgives. A lease left open after Close is held by
// memgate's own test and the CPU's plant: here it would fail this package's ledger check at exit.
func TestMemGate_webgpuPlantsThroughTheTable(t *testing.T) {
	newOrSkip(t).Close()
	g := memgate.Gate{Allowance: webgpuRounding, Gaps: webgpuMemGaps, GapsPin: webgpuMemGapsPin}
	for _, c := range []struct {
		name, fx, row string
		line          mem.LineID
		o             decoder.Options
		live          bool
	}{
		{"dense", "../testdata/gemma1-tiny", "testdata/gemma1-tiny int8int8", mem.LineDense, decoder.Options{Quant: "int8int8"}, true},
		{"experts", "../testdata/qwen3vlmoe-tiny", "testdata/qwen3vlmoe-tiny int8int8", mem.LineExperts, decoder.Options{Quant: "int8int8"}, true},
		{"kv", "../testdata/llama-tiny", "testdata/llama-tiny int4/kvprecision-i8", mem.LineKV, decoder.Options{Quant: "int4", KVPrecision: "i8"}, true},
		{"state", "../testdata/llama-tiny", "testdata/llama-tiny int4", mem.LineState, decoder.Options{Quant: "int4"}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			c.o.Backend, c.o.ResidentContext = "webgpu", 256
			clean, ok := webgpuGateRow(t, c.row, c.fx, c.o, true)
			if !ok {
				t.Fatalf("%s did not build resident", c.fx)
			}
			if u := g.Unknown(clean); len(u) != 0 {
				t.Fatalf("the unplanted row is red through the table: %v", u)
			}
			fired := false
			for _, f := range memgate.Check(clean, webgpuRounding) {
				fired = fired || (f.Line == c.line && f.Kind == memgate.OverPlan)
			}
			if fired != c.live {
				t.Fatalf("unplanted, a gap on %q fires %v, want %v", c.line, fired, c.live)
			}
			ran := false
			webgpuGatePlantForTest = func(rd *residentDecoder) func() {
				extra, err := kernels.CreateBufferOn(rd.c, string(c.line), &wgpu.BufferDescriptor{Label: "gate-plant", Size: 64 << 20, Usage: wgpu.BufferUsageStorage})
				if err != nil {
					t.Fatal(err)
				}
				ran = true
				return func() { kernels.ReleaseBuffer(extra) }
			}
			defer func() { webgpuGatePlantForTest = nil }()
			read, ok := webgpuGateRow(t, c.row, c.fx, c.o, true)
			if !ok || !ran {
				t.Fatalf("the plant did not run (built %v)", ok)
			}
			red := false
			for _, f := range g.Unknown(read) {
				red = red || (f.Line == c.line && f.Kind == memgate.OverPlan)
			}
			if !red {
				t.Errorf("64 MiB on %q is not red through the table: %v", c.line, g.Unknown(read))
			}
		})
	}
}
