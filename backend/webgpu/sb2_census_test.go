//go:build gpu && goinfer_testhooks

package webgpu

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/decoder/fit"
	"github.com/townsendmerino/goinfer/internal/diag/sb2census"
	"github.com/townsendmerino/goinfer/internal/guard/memgate"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// Layer B-webgpu of the admission and routing census: WebGPU's own static
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
	d, err := sb2census.ReadLayerD("../..", "webgpu")
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Fixtures) < 20 {
		t.Fatalf("only %d fixtures in layer D's golden: a census that read little covers little", len(d.Fixtures))
	}
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
var planCells = append(slices.Clone(sb2census.Cells), sb2census.Cell{Name: "int4/kv=i8", Quant: "int4", KVPrec: "i8", KVQuant: "i8"})

// planVsLoadWebGPU compares Plan's webgpu verdict, with the request req builds for each cell, to what the load does on webgpu:
// the decoder's admission, then staticResidentDecline. Plan reads a CPU load of the cell without its KV options, priced against
// far more memory than any fixture needs, so only its eligibility can decline. A cell with no CPU model must not reach the build.
func planVsLoadWebGPU(t *testing.T, req func(c sb2census.Cell) fit.PlanRequest) (diffs []string, cells, noModel int) {
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
	d, err := sb2census.ReadLayerD("../..", "webgpu")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("../..", f)
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
			if planned := p.Placement != fit.PlacementDecline; planned != loads {
				diffs = append(diffs, fmt.Sprintf("%s %s: Plan places=%v (%s), the load goes resident=%v (%s)", f, c.Name, planned, p.Reason, loads, ans))
			}
		}
	}
	return diffs, cells, noModel
}

// TestPlanAgreesWithLoad_webgpu holds Plan's webgpu verdict, with each cell's KV request, to what the load does on webgpu,
// and shows the comparison goes red on a Plan that
// ignores the KV request.
func TestPlanAgreesWithLoad_webgpu(t *testing.T) {
	// Plan's KV request is the serving flag's precision (KVPrecision).
	diffs, cells, noModel := planVsLoadWebGPU(t, func(c sb2census.Cell) fit.PlanRequest {
		return fit.PlanRequest{Ctx: 4096, KVF16: c.KVPrec == "f16", KVI8: c.KVPrec == "i8"}
	})
	if len(diffs) > 0 {
		t.Errorf("Plan disagrees with the webgpu load in %d of %d cells:\n%s", len(diffs), cells, strings.Join(diffs[:min(len(diffs), 30)], "\n"))
	}
	t.Logf("%d cells: Plan and the webgpu load agree (%d with no CPU model)", cells, noModel)

	// The plant: a Plan asked without the KV request must disagree on the lossy-KV cells of the families webgpu declines, and only there.
	planted, _, _ := planVsLoadWebGPU(t, func(sb2census.Cell) fit.PlanRequest { return fit.PlanRequest{Ctx: 4096} })
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

// The slot census: WebGPU's KV-slot decisions (slotsGranted,
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
	paths, err := filepath.Glob("../../testdata/machines/*.json")
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
	d, err := sb2census.ReadLayerD("../..", "webgpu")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range d.Fixtures {
		path, err := sb2census.Path("../..", f)
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
	p.free = slices.Clone(p.free)
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
	dir := "../../docs/measurements/sb2-vk-2026-10-31"
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

// webgpuMemGaps is WebGPU's table of known reds. It may only shrink: the fix that removes a row lowers webgpuMemGapsPin.
var webgpuMemGaps = func() []memgate.Gap {
	var out []memgate.Gap
	stacked := func(row string, over int64) {
		out = append(out, memgate.Gap{Row: row, Line: mem.LineExperts, What: "the stacked experts lease more than the plan's expert figure: not diagnosed, so it names this row's own bytes",
			Item: "MM8 WebGPU, stacked experts", Bytes: func(map[string]int64) int64 { return over }})
	}
	stacked("testdata/qwen3vlmoe-tiny int8int8", 16384)
	if runtime.GOARCH != "arm64" { // gpu-darwin's run fires none of these int4 rows: not diagnosed, and not shown to be the architecture rather than the adapter
		stacked("testdata/gemma4-moe-kv-tiny int4*", 8192)
		stacked("testdata/qwen3vlmoe-tiny int4*", 8192)
	}
	return out
}()

// webgpuGateRowsPath names every row that must build resident, and webgpuGateRowsUpdate has TestMemGate_webgpu rewrite it.
const webgpuGateRowsPath = "testdata/memgate_rows.txt"

var webgpuGateRowsUpdate = flag.Bool("memgate.rows.update", false, "have TestMemGate_webgpu rewrite testdata/memgate_rows.txt")

// webgpuMemGapsPin is len(webgpuMemGaps) on this platform: 1 row everywhere and two more off arm64.
var webgpuMemGapsPin = 1 + map[bool]int{true: 2}[runtime.GOARCH != "arm64"]

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
	read.Plan = m.Plan("webgpu", math.MaxInt64/4, fit.PlanRequest{Ctx: rd.ctxCap, CtxPinned: true, KVF16: kv == mem.KVF16, KVI8: kv == mem.KVI8})
	read.Exact = []mem.LineID{mem.LineKV, mem.LineState}
	facts := memgate.GeometryOf(m, rd.ctxCap, rd.KVSlots(), kv, false).Facts()
	read.Open = append(read.Open, memgate.Snap("built", l, facts))
	if err := webgpuScratchListHolds(rd, l); err != nil {
		t.Errorf("%s: %v", row, err)
	}
	if covered, err := webgpuScratchPlanHolds(m, rd); err != nil {
		t.Errorf("%s: %v", row, err)
	} else if covered {
		webgpuScratchPlanCovered++
	} else {
		why := webgpuDecodeScratchWhy(m)
		if i := strings.Index(why, ": "); strings.HasPrefix(why, "layer ") && i > 0 {
			why = why[i+2:] // the layer's number varies, the reason is the row's
		}
		webgpuScratchPlanWhy[why]++
	}
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

// TestMemGate_webgpu is MM7's gate on WebGPU: every census fixture at every option cell whose
// build goes resident, and each at int4 with two KV slots. It runs every row in one test, since a filtered run would see known gaps go quiet.
func TestMemGate_webgpu(t *testing.T) {
	newOrSkip(t).Close()
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	d, err := sb2census.ReadLayerD("../..", "webgpu")
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
		path, err := sb2census.Path("../..", f)
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
	t.Logf("%d builds held, %d options declined by WebGPU's own build; the plan's decode scratch covered %d", len(built), declined, webgpuScratchPlanCovered)
	for why, n := range webgpuScratchPlanWhy {
		t.Logf("no plan figure for %d rows: %s", n, why)
	}
	if webgpuScratchPlanCovered == 0 {
		t.Error("VOID: webgpuDecodeScratchTerms gave a figure for no row, so the plan-side gate compared nothing")
	}
	memgate.HoldRows(t, webgpuGateRowsPath, built, *webgpuGateRowsUpdate)
	g.Finish(t)
}

// TestMemGate_webgpuPlantsThroughTheTable: a buffer really allocated over plan after the generation stays red through the table, on a row whose
// line a live gap forgives part of (each is checked to fire unplanted), and on a line nothing
// forgives. A lease left open after Close is held by
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
		{"dense", "../../testdata/gemma4-moe-kv-tiny", "testdata/gemma4-moe-kv-tiny int8int8", mem.LineDense, decoder.Options{Quant: "int8int8"}, false},
		{"experts", "../../testdata/qwen3vlmoe-tiny", "testdata/qwen3vlmoe-tiny int8int8", mem.LineExperts, decoder.Options{Quant: "int8int8"}, true},
		{"kv", "../../testdata/llama-tiny", "testdata/llama-tiny int4/kvprecision-i8", mem.LineKV, decoder.Options{Quant: "int4", KVPrecision: "i8"}, false},
		{"state", "../../testdata/llama-tiny", "testdata/llama-tiny int4", mem.LineState, decoder.Options{Quant: "int4"}, false},
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

// TestWebGPUDeviceBuffers_inTheBuildsOrder: on a real WebGPU build of each fixture, the planner's buffer list (vkPlanBuffers, its
// trailing scratch line left out) appears in the build's ledger of weight, KV, state and expert buffers, buffer for buffer and in
// order, with only vectors between (Storage|CopySrc, as up32 makes them: the norms and gains the plan prices beside the blocks).
// M1′ places first fit, so the order decides the blocks (docs/measurements/sb2-vk-2026-10-31/README.md, "The hybrid's miss").
func TestWebGPUDeviceBuffers_inTheBuildsOrder(t *testing.T) {
	for _, dir := range []string{"olmo_hybrid-tiny", "qwen35-tiny", "qwen3next-tiny", "nemotron-tiny", "nemotron3nano-tiny", "deepseek-tiny",
		"gemma4-moe-kv-tiny", "llama-tiny", "mistral-tiny-window", "mixtral-tiny", "qwen3moe-tiny", "glm-tiny", "glm-tiny-bias",
		"granite-dense-tiny", "olmo3-tiny", "smollm3-tiny", "gemma1-tiny", "qwen25vl-tiny"} {
		t.Run(dir, func(t *testing.T) {
			path := filepath.Join("../../testdata", dir)
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s", path)
			}
			o := decoder.Options{Quant: "int4", ResidentContext: 64}
			if dir == "nemotron-tiny" {
				o.Knobs = &decoder.Knobs{"GOINFER_SSM_RESIDENT": "1"}
			}
			m, rd := loadWebGPUResident(t, path, o, 1)
			defer m.Close()
			kv := m.KVDeliveredFor("webgpu")
			plan, _ := vkPlanBuffers(m, rd.ctxCap, kv == "f16", kv == "i8")
			plan = plan[:len(plan)-1]
			const vector = uint64(wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc)
			var led []kernels.LeasedBufferForTest
			for _, b := range rd.c.LeasedBuffersForTest() {
				if b.Line == "dense" || b.Line == "kv" || b.Line == "state" || b.Line == "experts" {
					led = append(led, b)
				}
			}
			// ok[i][j]: the ledger from i on holds the plan from j on, in order, skipping vectors only.
			ok := make([][]bool, len(led)+1)
			for i := range ok {
				ok[i] = make([]bool, len(plan)+1)
			}
			ok[len(led)][len(plan)] = true
			for i, l := range slices.Backward(led) {
				for j := len(plan); j >= 0; j-- {
					ok[i][j] = l.Usage == vector && l.Line == "dense" && ok[i+1][j] ||
						j < len(plan) && l.Bytes == plan[j] && ok[i+1][j+1]
				}
			}
			if !ok[0][0] {
				t.Fatalf("the planner's %d buffers are not the build's %d weight, KV, state and expert buffers in order, with only vectors between", len(plan), len(led))
			}
		})
	}
}

// webgpuScratchListHolds: the
// scratch line the ledger shows after a build is the sum of the build's named scratch list, summed over every runner the build made (one a
// KV slot), and no lease on it is unnamed. A lease made beside the list, or one no builder names, is red here.
func webgpuScratchListHolds(rd *residentDecoder, l *mem.Ledger) error {
	runners := []*DecodeRunner{rd.runner}
	if len(rd.slots) > 0 {
		runners = runners[:0]
		for _, s := range rd.slots {
			runners = append(runners, s.runner)
		}
	}
	var listed, unnamed int64
	roles := map[string]int64{}
	for _, r := range runners {
		for _, e := range r.ScratchList() {
			listed += e.Bytes
			roles[e.Role] += e.Bytes
			if e.Role == "" {
				unnamed += e.Bytes
			}
		}
	}
	held := l.Usage().ByLine[mem.LineScratch]
	var parts []string
	for role, n := range roles {
		parts = append(parts, fmt.Sprintf("%q=%d", role, n))
	}
	switch {
	case unnamed != 0:
		return fmt.Errorf("%d B of the build's scratch leases name no builder (list %d B of the ledger's %d B: %s)", unnamed, listed, held, strings.Join(parts, " "))
	case listed != held:
		return fmt.Errorf("the build's scratch list sums to %d B and the ledger's scratch line holds %d B (%+d B beside the list)", listed, held, held-listed)
	}
	return nil
}

// webgpuScratchPlanCovered counts the rows webgpuScratchPlanHolds found the plan's figure for, so a gate that covers none is VOID.
var webgpuScratchPlanCovered int

// webgpuScratchPlanWhy counts the rows the plan gave no figure for, by the reason it gave (logged, so the next layer shape to derive is read, not guessed).
var webgpuScratchPlanWhy = map[string]int{}

// webgpuScratchPlanHolds is G-MS2w's second half (MM5b): where the plan gives a figure for m (webgpuDecodeScratchTerms), the build's
// scratch list equals it role by role, once for each runner the build made. covered is false where the plan gives none.
func webgpuScratchPlanHolds(m *decoder.Model, rd *residentDecoder) (covered bool, err error) {
	terms, ok := webgpuDecodeScratchTerms(m)
	if !ok {
		return false, nil
	}
	return true, webgpuScratchTermsHold(terms, rd)
}

// webgpuScratchTermsHold compares terms with the build's scratch list, role by role, for each runner the build made.
func webgpuScratchTermsHold(terms []webgpuScratchTerm, rd *residentDecoder) error {
	n := int64(max(1, len(rd.slots)))
	want := map[string]int64{}
	for _, tm := range terms {
		want[tm.Role] += n * tm.Bytes
	}
	got := map[string]int64{}
	runners := []*DecodeRunner{rd.runner}
	if len(rd.slots) > 0 {
		runners = runners[:0]
		for _, s := range rd.slots {
			runners = append(runners, s.runner)
		}
	}
	for _, r := range runners {
		for _, e := range r.ScratchList() {
			got[e.Role] += e.Bytes
		}
	}
	var diffs []string
	for role := range want {
		if got[role] != want[role] {
			diffs = append(diffs, fmt.Sprintf("%s: plan %d B, build %d B", role, want[role], got[role]))
		}
	}
	for role, b := range got {
		if _, ok := want[role]; !ok {
			diffs = append(diffs, fmt.Sprintf("%s: the build leases %d B the plan names no term for", role, b))
		}
	}
	if len(diffs) > 0 {
		slices.Sort(diffs)
		return fmt.Errorf("the plan's decode scratch and the build's list differ: %s", strings.Join(diffs, "; "))
	}
	return nil
}

// TestScratchList_plantsGoRed holds G-MS2w to its plants (MM5b): on a built resident, the list and the ledger agree; a scratch buffer leased
// beside the list makes them disagree by exactly its bytes; and the list names every lease with a builder (no empty role) on a family that
// builds the plain attention path.
func TestScratchList_plantsGoRed(t *testing.T) {
	m, err := decoder.Load("../../testdata/llama-tiny", decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: 256})
	if err != nil {
		t.Skipf("no WebGPU device: %v", err)
	}
	defer m.Close()
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Skip("llama-tiny did not build resident on this adapter")
	}
	l := rd.c.Ledger()
	if err := webgpuScratchListHolds(rd, l); err != nil {
		t.Fatalf("unplanted: %v", err)
	}
	var listed int64
	for _, e := range rd.runner.ScratchList() {
		if e.Role == "" {
			t.Errorf("an unnamed role holds %d B in %d buffers", e.Bytes, e.Count)
		}
		listed += e.Bytes
	}
	if listed == 0 || listed != rd.runner.ScratchBytes() {
		t.Fatalf("VOID: the list sums to %d B and ScratchBytes says %d B", listed, rd.runner.ScratchBytes())
	}
	extra, err := kernels.CreateBufferOn(rd.c, string(mem.LineScratch), &wgpu.BufferDescriptor{Label: "plant", Size: 4096, Usage: wgpu.BufferUsageStorage})
	if err != nil {
		t.Fatal(err)
	}
	defer kernels.ReleaseBuffer(extra)
	err = webgpuScratchListHolds(rd, l)
	if err == nil || !strings.Contains(err.Error(), "+4096 B beside the list") {
		t.Fatalf("a 4096 B scratch buffer leased beside the list: err = %v, want one naming +4096 B beside the list", err)
	}
}

// TestScratchPlan_plantsGoRed holds the plan-side gate to its plants (MM5b): on llama-tiny the plan equals the build's list; a term one
// uniform short (a dispatch the plan does not count) is red naming its role and the 16 B; and a build role the plan has no term for is red.
func TestScratchPlan_plantsGoRed(t *testing.T) {
	m, err := decoder.Load("../../testdata/llama-tiny", decoder.Options{Backend: "webgpu", Quant: "int4", ResidentContext: 256})
	if err != nil {
		t.Skipf("no WebGPU device: %v", err)
	}
	defer m.Close()
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Skip("llama-tiny did not build resident on this adapter")
	}
	terms, ok := webgpuDecodeScratchTerms(m)
	if !ok {
		t.Fatal("VOID: no plan figure for llama-tiny")
	}
	if err := webgpuScratchTermsHold(terms, rd); err != nil {
		t.Fatalf("unplanted: %v", err)
	}
	short := slices.Clone(terms)
	for i := range short {
		if short[i].Role == "gemvAdd" {
			short[i].Bytes -= 16
		}
	}
	if err := webgpuScratchTermsHold(short, rd); err == nil || !strings.Contains(err.Error(), "gemvAdd: plan 112 B, build 128 B") {
		t.Errorf("a gemvAdd term 16 B short: err = %v, want one naming gemvAdd, plan 112 B, build 128 B", err)
	}
	if err := webgpuScratchTermsHold(terms[1:], rd); err == nil || !strings.Contains(err.Error(), "build: the build leases 1028 B the plan names no term for") {
		t.Errorf("the build term dropped: err = %v, want one naming the build role the plan has no term for", err)
	}
}

// webgpuLoRAScratchRows is the f32 rows a WebGPU build always leases for compute-time LoRA (kernels' loraRMax). The memory gate holds
// webgpuDecodeScratchTerms to the build's own scratch list, so a drift here is red there.
const webgpuLoRAScratchRows = 256

// webgpuScratchTerm is one named part of WebGPU's decode scratch: the bytes the runner's op builders of that role lease at build.
type webgpuScratchTerm struct {
	Role  string
	Bytes int64
}

// webgpuDecodeScratchTerms is what a WebGPU resident build of m leases on the scratch line for decode (the sum of the build's scratch list,
// kernels.DecodeRunner.ScratchList), as terms named by the op builder that leases them: each from the program's declared per-row
// activations, plus the per-dispatch parameter buffers the builders create. ok is false for a model whose layers are not all a
// plain attention sublayer and a gated dense FFN with one geometry (the runner's other mixers are not yet derived): no figure is
// given for it, rather than a guess. Reported, not charged: no admission decision reads it.
func webgpuDecodeScratchTerms(m *decoder.Model) (terms []webgpuScratchTerm, ok bool) {
	terms, why := webgpuDecodeScratch(m)
	return terms, why == ""
}

// webgpuDecodeScratchWhy is the first reason webgpuDecodeScratchTerms gives no figure for m, or "" where it gives one.
func webgpuDecodeScratchWhy(m *decoder.Model) string {
	_, why := webgpuDecodeScratch(m)
	return why
}

func webgpuDecodeScratch(m *decoder.Model) (terms []webgpuScratchTerm, why string) {
	H, _, _, _, _, _, V := m.Dims()
	view := m.BackendView()
	wts := view.Weights()
	if wts == nil || H <= 0 || V <= 0 {
		return nil, "the model has no hidden or vocabulary width"
	}
	declared := m.DecodeRowsForTest()
	kv := m.KVDeliveredFor("webgpu")
	win := view.SlidingWindow() > 0
	type geom struct {
		hd, nKV, half int
		kEqV          bool
	}
	geoms := map[geom]map[float32]bool{} // each distinct attention geometry, and the rope scales its uniforms are built for
	var gemv, gemvAdd, actQuant, attnLayer, quantB, rmsQuantB, qkNormB, rmsnormB, biasAddB, gemvBiasB, moeFFN, moeRoute, moeExpert, swiglu, sharedCombine int64
	var moeNE int
	var deltaLayer, qSplit, attnGate, mlaLayer, mlaAbsorb, mlaLift, nRmsExtra int64
	anyDelta, anyLatent, anySSM := false, false, false
	var moeBlock, vNormB, scaleVecB, routedRows, mambaLayer, relu2, clampedRoute, clampedAct, clampedDown int64
	pk := func(k int) int64 { return int64((k + 31) &^ 31) }
	// qrow is one dense quantize's int8 row, its scale buffer (one float, or one per 32 under per-32 activations) and its uniform.
	qrow := func(k int) int64 {
		if m.ActQuantGroup() == 32 {
			return pk(k) + pk(k)/8 + 16
		}
		return pk(k) + 20
	}
	rmsQuantB = qrow(H) // one call per sublayer norm and the final norm: the int8 row, its scale, its uniform
	nRms, nQuant := int64(1), int64(0)
	ffnTerms := func(l int, shape denseShape, rows map[string]int, nE, topK, shared int, hasShared bool) string {
		if shape.postMLP { // the output projection splits as the attention's does
			if shape.moe {
				return fmt.Sprintf("layer %d: a routed FFN with an output norm", l)
			}
			gemvAdd -= 16
			gemv += 4*int64(H) + 16
			rmsnormB += 4*int64(H) + 16
			biasAddB += 16
		}
		inter := int64(rows["gate"])
		if !shape.moe {
			gemv += 4*(inter+int64(rows["up"])) + 2*16
			gemvAdd += 16
			actQuant += qrow(rows["gate"])
		} else {
			// The router's logits row; the selection rows and the experts' gate and up rows (reused over the top-k slots); the router's
			// uniform; per slot the expert projections' uniforms and the fused activation's int8 row, scale and uniform.
			gemv += 4*int64(nE) + 16
			moeFFN += 4*int64(2*topK) + 4*2*inter
			if shape.clamped { // the router's four words, the gate and up uniforms, the clamped activation's eight, the down's four
				clampedRoute += 16
				moeExpert += 2 * int64(topK) * 16
				clampedAct += int64(topK) * (pk(rows["gate"]) + 4 + 32)
				clampedDown += int64(topK) * 16
			} else {
				moeRoute += 32
				moeExpert += 3 * int64(topK) * 16
				swiglu += int64(topK) * (pk(rows["gate"]) + 20)
			}
			moeNE = nE
			if hasShared { // the shared expert: a gated SwiGLU MLP, sigmoid-gated by a one-row projection or added ungated
				gemv += 4*2*int64(shared) + 2*16
				swiglu += pk(shared) + 20
				if wts.Layers[l].SharedGate.Rows() > 0 {
					gemv += 4*int64(H) + 16 + 4 + 16
					sharedCombine += 16
				} else {
					gemvAdd += 16
				}
			}
		}
		return ""
	}
	// g4Terms is a parallel dense and routed block on the post-attention residual: the router's norm and projection, a dense branch and an
	// expert branch each behind its own pre-norm, the post-norms, the join and the layer's output scalar.
	g4Terms := func(l int) string {
		g, ok := parallelBlockOf(view.Layer(l))
		if !ok {
			return fmt.Sprintf("layer %d: no parallel routed block", l)
		}
		rows := declared[l]
		if rows["g4moe-router"] != g.nE || rows["g4moe-experts"] != g.topK || rows["gate"] != g.denseInter {
			return fmt.Sprintf("layer %d: the block's declared rows disagree with its weights", l)
		}
		nRms += 2 // the dense branch's and the expert branch's pre-norms
		moeNE = g.nE
		rmsnormB += 4 * (4*int64(H) + 16)
		moeBlock += 4*int64(g.nE) + 4*int64(2*g.topK) + 4*int64(H) + 4*3*int64(g.moeInter) + 16 + 32 + 16 + int64(g.topK)*16
		gemv += 4*int64(2*g.denseInter) + 2*16 + 4*int64(H) + 16
		actQuant += pk(g.denseInter) + 20 + int64(g.topK)*(pk(g.moeInter)+20)
		moeExpert += 2 * int64(g.topK) * 16
		scaleVecB += 16
		biasAddB += 2 * 16
		if sc := g.layerScalar; sc != 0 && sc != 1 {
			scaleVecB += 16
		}
		return ""
	}
	for l, ops := range m.Program().Layers {
		shape, isDense := denseAttnLayer(ops)
		if !isDense {
			return nil, fmt.Sprintf("layer %d: operations %v (FFN form %d) are not a plain attention and dense FFN sublayer", l, opKinds(ops), shape.otherForm)
		}
		if routesExperts(declared[l]) != (shape.moe || shape.block == blockRouted || shape.g4) {
			return nil, fmt.Sprintf("layer %d: a routed-experts FFN of a form not derived", l)
		}
		rows := declared[l]
		mixer := []string{"q", "k", "v", "ctx"}
		switch {
		case shape.gated:
			mixer = append(mixer, "q-gate", "attn-gate")
		case shape.g4:
			mixer = []string{"q", "k", "v", "ctx", "g4moe-dense-out", "g4moe-experts", "g4moe-moe-out", "g4moe-router", "g4moe-router-in", "g4moe-weights"}
		case shape.block == blockSSM:
			mixer = []string{"mb-conv", "mb-gated", "mb-proj", "mb-y"}
		case shape.block == blockMLP:
			mixer = []string{"up"}
		case shape.block == blockRouted:
			p := shape.routedPrefix
			mixer = []string{"up", "shared-up", p + "-router", p + "-experts", p + "-weights"}
		case shape.latent:
			mixer = []string{"mla-ctx", "mla-kv", "mla-q", "mla-q-abs", "mla-wsum"}
			if rows["mla-qa"] > 0 { // a low-rank query projection
				mixer = append(mixer, "mla-qa")
			}
		case shape.delta:
			mixer = []string{"dn-ba", "dn-beta", "dn-conv", "dn-core", "dn-decay", "dn-gated", "dn-k", "dn-mixed", "dn-q", "dn-z"}
		}
		want := len(mixer) + 2 // and the FFN's gate and up
		if shape.block != blockPair {
			want = len(mixer) // a layer of one operation has no FFN rows beside it
		}
		var nE, topK, shared int
		hasShared := shape.moe && wts.Layers[l].SharedExpert.Gate.Rows() > 0
		if shape.moe {
			pre := "moe" // the router's declared rows are named by the form
			if shape.clamped {
				pre = "gptoss"
			}
			nE, topK, shared = rows[pre+"-router"], rows[pre+"-experts"], rows["shared-gate"]
			want += 3
			if hasShared {
				want += 2
			}
			if nE == 0 || topK == 0 || rows[pre+"-weights"] != topK || hasShared != (shared > 0) || rows["shared-up"] != shared {
				return nil, fmt.Sprintf("layer %d: the routed FFN's declared rows are not the generic router's", l)
			}
		}
		need := append(mixer, "gate", "up")
		if shape.block != blockPair {
			need = mixer
		}
		for _, k := range need {
			if rows[k] == 0 {
				return nil, fmt.Sprintf("layer %d: the declared row %q is missing", l, k)
			}
		}
		if len(rows) != want {
			return nil, fmt.Sprintf("layer %d: declared rows beyond the derived ones", l)
		}
		if shape.block == blockRouted {
			p := shape.routedPrefix
			nE, topK, inter, sh := rows[p+"-router"], rows[p+"-experts"], rows["up"], rows["shared-up"]
			if nE == 0 || topK == 0 || rows[p+"-weights"] != topK || len(wts.Layers[l].PreAttnNorm) == 0 {
				return nil, fmt.Sprintf("layer %d: the routed block's declared rows or its norm are not the derived ones", l)
			}
			nRms++
			gemv += 4*int64(nE) + 16
			routedRows += 4*int64(2*topK) + 4*int64(topK)*int64(inter)
			moeRoute += 32
			moeExpert += 2 * int64(topK) * 16
			relu2 += int64(topK)*(pk(inter)+20) + pk(sh) + 20
			gemv += 4*int64(sh) + 16
			gemvAdd += 16
			moeNE = nE
			continue
		}
		if shape.block == blockSSM || shape.block == blockMLP {
			if len(wts.Layers[l].PreAttnNorm) == 0 {
				nQuant++
			} else {
				nRms++
			}
			if shape.block == blockSSM {
				anySSM = true
				gemv += 4*int64(rows["mb-proj"]) + 16
				mambaLayer += 4 * int64(rows["mb-conv"]+rows["mb-y"]+rows["mb-gated"])
				quantB += pk(rows["mb-gated"]) + 20
			} else {
				gemv += 4*int64(rows["up"]) + 16
				relu2 += pk(rows["up"]) + 20
			}
			gemvAdd += 16
			continue
		}
		if shape.latent {
			anyLatent = true
			for _, pre := range [][]float32{wts.Layers[l].PreAttnNorm, wts.Layers[l].PreMLPNorm} {
				if len(pre) == 0 {
					nQuant++
				} else {
					nRms++
				}
			}
			// The query projection (through a low-rank bottleneck and its norm where there is one), the latent projection; the absorbed
			// query, the rank-space context sum and the lifted context rows; the absorb and lift dispatches' uniforms.
			gemv += 4*int64(rows["mla-q"]+rows["mla-kv"]) + 2*16
			if qa := rows["mla-qa"]; qa > 0 {
				gemv += 4*int64(qa) + 16
				nRmsExtra += pk(qa) + 20
			}
			mlaLayer += 4 * int64(rows["mla-q-abs"]+rows["mla-wsum"]+rows["mla-ctx"])
			mlaAbsorb += 32
			mlaLift += 32
			quantB += pk(rows["mla-ctx"]) + 20
			gemvAdd += 16
			if err := ffnTerms(l, shape, rows, nE, topK, shared, hasShared); err != "" {
				return nil, err
			}
			continue
		}
		if shape.delta {
			anyDelta = true
			for _, pre := range [][]float32{wts.Layers[l].PreAttnNorm, wts.Layers[l].PreMLPNorm} {
				if len(pre) == 0 {
					nQuant++
				} else {
					nRms++
				}
			}
			dn := int64(rows["dn-k"])
			gemv += 4*int64(rows["dn-mixed"]+rows["dn-beta"]+rows["dn-decay"]+rows["dn-z"]) + 4*16
			deltaLayer += 4 * (int64(rows["dn-conv"]) + 2*dn + int64(rows["dn-ba"]) + int64(rows["dn-core"]) + int64(rows["dn-gated"]))
			quantB += pk(rows["dn-gated"]) + 20
			gemvAdd += 16
			if err := ffnTerms(l, shape, rows, nE, topK, shared, hasShared); err != "" {
				return nil, err
			}
			continue
		}
		lv := view.Layer(l)
		w := &wts.Layers[l]
		rs := float32(lv.RopeMscale())
		switch {
		case rs == 0:
			return nil, fmt.Sprintf("layer %d: rope scale 0", l)
		case lv.VFromK() && !shape.g4:
			return nil, fmt.Sprintf("layer %d: V from K", l)
		}
		// Each optional operation is held by the weights it runs on: the build creates its dispatches from the weights, not the program.
		if shape.gated && (len(w.QBias) > 0 || len(w.OBias) > 0 || len(w.PostAttnNorm) > 0 || len(w.PostMLPNorm) > 0) {
			return nil, fmt.Sprintf("layer %d: a bias or output norm on a gated attention layer", l)
		}
		hasQNorm := len(w.QNorm) > 0
		if shape.gated { // the gated attention's QK-norm is part of its one operation, and its weights live apart from the layer's own
			hasQNorm = gatedAttnQKNorm(lv)
			shape.qkNorm = hasQNorm
		}
		if shape.g4 { // the block's own post-norms are its weights', not a sublayer output norm the layer's weights carry
			shape.qkNorm, shape.postMLP = hasQNorm, len(w.PostMLPNorm) > 0
		}
		if shape.qkNorm != hasQNorm || shape.postAttn != (len(w.PostAttnNorm) > 0) || shape.postMLP != (len(w.PostMLPNorm) > 0) {
			return nil, fmt.Sprintf("layer %d: a norm or QK-norm the weights and the operations disagree on", l)
		}
		g := geom{lv.HeadDim(), lv.KVHeads(), len(lv.RopeInvFreq()), lv.VFromK()}
		if geoms[g] == nil {
			geoms[g] = map[float32]bool{}
		}
		geoms[g][rs] = true
		if shape.g4 && lv.KVSrc() != l {
			return nil, fmt.Sprintf("layer %d: it shares another layer's K and V", l)
		}
		pres := [][]float32{w.PreAttnNorm, w.PreMLPNorm}
		if shape.block == blockAttn || shape.g4 { // one pre-norm: a parallel block's own are counted with it
			pres = pres[:1]
		}
		for _, pre := range pres {
			if len(pre) == 0 { // a family with no pre-norm (norms after the sublayer only) quantizes the raw residual
				nQuant++
			} else {
				nRms++
			}
		}
		gemv += 4*int64(rows["q"]+rows["k"]+rows["v"]) + 3*16
		if shape.g4 { // the scale-less V norm writes its own row; a layer whose V is its K has no V projection
			attnLayer += 4 * int64(rows["k"])
			vNormB += 16
			if lv.VFromK() {
				gemv -= 4*int64(rows["v"]) + 16
			}
		}
		gemvAdd += 16    // the output projection, fused with its residual add
		if shape.gated { // q_proj emits [query | gate] per head: its output row is the wide one, then a split and a gate applied to the context
			gemv += 4 * int64(rows["q-gate"]-rows["q"])
			qSplit += 4*2*int64(rows["q"]) + 16
			attnGate += 16
		}
		if shape.qkNorm {
			qkNormB += 2 * 16 // q's and k's
		}
		if len(w.QBias) > 0 { // q, k and v's bias: folded into an int8 weight's projection, else a separate add each
			if w.QProj.Kind() == "int4" {
				biasAddB += 3 * 16
			} else {
				q3 := 4*int64(rows["q"]+rows["k"]+rows["v"]) + 3*16
				gemv -= q3
				gemvBiasB += q3
			}
		}
		// A post-sublayer norm splits the fused output projection into a bare gemv (its own output row), a norm (its own row and
		// uniform) and a separate residual add (one uniform).
		if shape.postAttn || len(w.OBias) > 0 { // an output bias splits it the same way
			gemvAdd -= 16
			gemv += 4*int64(H) + 16
			biasAddB += 16
		}
		if shape.postAttn {
			rmsnormB += 4*int64(H) + 16
		}
		if len(w.OBias) > 0 {
			biasAddB += 16 // the bias add on the bare output
		}
		if shape.block == blockPair && !shape.g4 {
			if err := ffnTerms(l, shape, rows, nE, topK, shared, hasShared); err != "" {
				return nil, err
			}
		}
		if shape.g4 {
			if err := g4Terms(l); err != "" {
				return nil, err
			}
		}
		attnLayer += 4*int64(rows["ctx"]) + 16
		quantB += qrow(rows["ctx"])
	}
	if len(geoms) == 0 && !anyDelta && !anyLatent && !anySSM {
		return nil, "the model has no layers"
	}
	gemv += 4*int64(V) + 16 // the head
	geomB := int64(0)
	nScale := int64(0) // each rope scale's uniforms: one fused (f32 KV), or q's and k's (f16 and int8 KV)
	for _, scales := range geoms {
		one := int64(16 + 32) // the v-store and attention uniforms
		if win {
			one += 32
		}
		if kv == mem.KVI8 {
			one += 32
		}
		geomB += one
		nScale += int64(len(scales))
	}
	mambaSetupB := int64(0)
	if anySSM {
		mambaSetupB = 16 + 32 + 32 // the conv's, the scan's and the gated norm's uniforms, shared across layers
	}
	mlaSetupB := int64(0)
	if anyLatent {
		mlaSetupB = 32 + 48 + 32 // the latent store's, the query rope's (twelve words) and the attention's uniforms, shared across layers
	}
	deltaSetupB := int64(0)
	if anyDelta {
		deltaSetupB = 16 + 16 + 32 + 32 + 32 // the five dispatch parameter buffers the delta mixer shares across layers
	}
	terms = []webgpuScratchTerm{
		{"build", 4 + 4*webgpuLoRAScratchRows},
		{"residual", 4 * int64(H)},
		{"logitsStaging", 4 * int64(V)},
		{"rmsQuant", nRms*rmsQuantB + nRmsExtra},
		{"mlaLayer", mlaLayer},
		{"mlaAbsorb", mlaAbsorb},
		{"mlaLift", mlaLift},
		{"mlaSetup", mlaSetupB},
		{"mambaLayer", mambaLayer},
		{"mambaSetup", mambaSetupB},
		{"relu2Quant", relu2},
		{"vNorm", vNormB},
		{"gemma4MoE", moeBlock},
		{"scaleVec", scaleVecB},
		{"nemoMoE", routedRows},
		{clampedRoles[0], clampedRoute},
		{clampedRoles[1], clampedAct},
		{clampedRoles[2], clampedDown},
		{"quant", quantB + nQuant*qrow(H)},
		{"gemv", gemv},
		{"gemvBias", gemvBiasB},
		{"gemvAdd", gemvAdd},
		{"moeFFN", moeFFN},
		{"moeRoute", moeRoute},
		{"moeExpert", moeExpert},
		{"swigluQuant", swiglu},
		{"sharedGatedCombine", sharedCombine},
		{"moeZeroBias", 4 * int64(moeNE)},
		{"deltaLayer", deltaLayer},
		{"qSplit", qSplit},
		{"attnGate", attnGate},
		{"deltaSetup", deltaSetupB},
		{"actQuantPad", actQuant},
		{"attnLayer", attnLayer},
		{"qkNorm", qkNormB},
		{"rmsnormF32", rmsnormB},
		{"biasAdd", biasAddB},
		{"geomFor", geomB},
	}
	if kv == mem.KVF16 || kv == mem.KVI8 {
		terms = append(terms, webgpuScratchTerm{"ropeQUniFor", 32 * nScale}, webgpuScratchTerm{"ropeKUniFor", 32 * nScale})
	} else {
		terms = append(terms, webgpuScratchTerm{"qkvFinUniFor", 32 * nScale})
	}
	kept := terms[:0]
	for _, t := range terms {
		if t.Bytes != 0 { // a role the model's layers never run
			kept = append(kept, t)
		}
	}
	return kept, ""
}

// routesExperts reports whether a layer's declared rows include a router's.
func routesExperts(rows map[string]int) bool {
	for name := range rows {
		if strings.HasSuffix(name, "-router") {
			return true
		}
	}
	return false
}

// denseShape is the optional operations a plain attention layer carries: a QK-norm after the projections, and a norm on each
// sublayer's output before its residual add.
type denseShape struct {
	qkNorm, postAttn, postMLP, moe, gated, delta, latent, clamped, g4 bool
	routedPrefix                                                      string          // the declared rows' name prefix of a one-operation routed layer
	otherForm                                                         program.FFNForm // the routed form a layer has that is not derived
	block                                                             int             // blockPair, or a layer of one operation behind its one norm
}

const (
	blockPair = iota
	blockSSM
	blockAttn
	blockMLP
	blockRouted
)

// routedPrefixOf names the declared router rows of the routed forms that stand alone in a layer.
var routedPrefixOf = map[program.FFNForm]string{program.FfnNemotronMoE: "nemotronmoe"}

// denseAttnLayer reports whether a layer's operations are a plain attention sublayer and a dense FFN, each behind its own pre-norm,
// with only the optional operations of denseShape and nothing else (no output gate, temperature or layer scale).
func denseAttnLayer(ops []program.Op) (s denseShape, ok bool) {
	i := 0
	next := func(k program.OpKind) bool {
		if i < len(ops) && ops[i].Kind == k {
			i++
			return true
		}
		return false
	}
	opt := func(k program.OpKind) bool { return next(k) }
	if !next(program.OpNorm) {
		return s, false
	}
	// A layer of one operation behind its one norm: a state-space mixer, an attention sublayer or a non-gated FFN.
	switch {
	case next(program.OpMamba2):
		s.block = blockSSM
		return s, next(program.OpResidual) && i == len(ops)
	case next(program.OpFFN):
		switch ops[i-1].FFN.Form {
		case program.FfnNonGated:
			s.block = blockMLP
		case program.FfnNemotronMoE:
			s.block, s.routedPrefix = blockRouted, routedPrefixOf[ops[i-1].FFN.Form]
		default:
			s.otherForm = ops[i-1].FFN.Form
			return s, false
		}
		return s, next(program.OpResidual) && i == len(ops)
	}
	switch {
	case next(program.OpDeltaNet):
		s.delta = true
	case next(program.OpGatedAttn):
		s.gated = true
	case next(program.OpMLA):
		s.latent = true
	case next(program.OpGemma4Attn):
		s.g4 = true
		s.postAttn = opt(program.OpNorm)
		return s, next(program.OpResidual) && next(program.OpGemma4MoE) && i == len(ops)
	default:
		if !next(program.OpProjQKV) {
			return s, false
		}
		s.qkNorm = opt(program.OpQKNorm)
		// A layer with no rotary embedding still runs the rope dispatch (over a zeroed frequency table, an identity), and a temperature
		// operation scales the query in a dispatch of its own that the runner folds into the attention's uniform.
		opt(program.OpRope)
		opt(program.OpAttnTemp)
		opt(program.OpRope)
		if !next(program.OpAttention) || !next(program.OpProjO) {
			return s, false
		}
		if i == len(ops)-1 && next(program.OpResidual) { // an attention block: no FFN follows in the layer
			s.block = blockAttn
			return s, true
		}
		s.postAttn = opt(program.OpNorm)
	}
	if !next(program.OpResidual) || !next(program.OpNorm) || !next(program.OpFFN) {
		return s, false
	}
	switch ops[i-1].FFN.Form {
	case program.FfnMoE:
		s.moe = true
	case program.FfnGptOss: // a router with a logit bias and clamped experts: its own builders, but the same dispatch shape
		s.moe, s.clamped = true, true
	case program.FfnGated, program.FfnNonGated:
	default: // gpt-oss, Llama 4, Nemotron and Gemma 4's own routed forms are other builders
		s.otherForm = ops[i-1].FFN.Form
		return s, false
	}
	s.postMLP = opt(program.OpNorm)
	return s, next(program.OpResidual) && i == len(ops)
}

// opKinds is a layer's operation kinds in order, for the reason a layer has no figure.
func opKinds(ops []program.Op) []program.OpKind {
	ks := make([]program.OpKind, len(ops))
	for i, o := range ops {
		ks[i] = o.Kind
	}
	return ks
}

// gatedAttnQKNorm reports whether a gated attention layer carries a QK-norm weight, which it keeps with its projections, not in the
// layer's own norm fields. Like denseAttnLayer it reads the layer shape's own accessor, so the pricing body names no family.
func gatedAttnQKNorm(lv decoder.ResidentLayerView) bool {
	_, _, _, _, qn, _ := lv.Qwen35AttnWeights()
	return len(qn) > 0
}

// clampedRoles are the runner's role names for the clamped router's three builders: the route, the clamped activation and the biased
// down projection. They are the runner's names for builders it names by family, so they live beside the matcher, not in the pricing.
var clampedRoles = [3]string{"routeGptOss", "gptossActQuant", "moeExpertGptOssDown"}

// parallelBlock is the dimensions of a layer's parallel dense and routed block.
type parallelBlock struct {
	nE, topK, denseInter, moeInter int
	layerScalar                    float32
}

// parallelBlockOf reads a layer's parallel block from the bundle the runner builds it from, as gatedAttnQKNorm reads the gated
// attention's: the pricing body names no family.
func parallelBlockOf(lv decoder.ResidentLayerView) (parallelBlock, bool) {
	g, ok := lv.Gemma4MoEBundle()
	if !ok {
		return parallelBlock{}, false
	}
	return parallelBlock{g.NE, g.TopK, g.DenseInter, g.MoeInter, g.LayerScalar}, true
}

// TestMM5b_webgpuDecodeScratchNamesNoFamily is G-MS4w: the figure is priced from the operations' per-row declarations, so its pricing names no family, mixer or
// FFN form. denseAttnLayer, which names the operation kinds of the one layer shape the figure is derived for, is the coverage
// test and not the pricing; it is not scanned.
func TestMM5b_webgpuDecodeScratchNamesNoFamily(t *testing.T) {
	banned := []string{"ResidentLayerOps", "Mixer", "KVSrcAtResident", "HeadDimAtResident", "KVHeadsAtResident",
		"Qwen35", "Nemotron", "MLA", "Gemma4", "GptOss", "DeltaNet", "Mamba", "KDA", "ShortConv", "PLE", "MoEResident", "QKNormWhole",
		"NonGatedMLP", "OpAttention", "OpGatedAttn", "Ffn", ".qwen35", ".lfm2", ".kda", ".granite", ".nemotron", ".gemma4", ".mla"}
	scanPricingBodies(t, map[string][]string{"sb2_census_test.go": {"webgpuDecodeScratchTerms", "webgpuDecodeScratch", "webgpuDecodeScratchWhy"}}, banned,
		"WebGPU's decode scratch is priced from the operations' per-row declarations, not the family")
}

// webgpuScratchHand is the figure for a plain-attention model written as arithmetic on its config: hidden h, l layers, vocabulary v, the
// query, key and FFN widths, a QK-norm per layer, a norm on each sublayer's output, at a KV precision, and the q/k/v bias an int4 or int8 weight carries.
func webgpuScratchHand(h, l, v, qW, kvW, ffn int, qkNorm, post bool, kv, bias string, window, noPre bool) map[string]int64 {
	pk := func(k int) int64 { return int64((k + 31) &^ 31) }
	H, L, V := int64(h), int64(l), int64(v)
	want := map[string]int64{
		"build": 4 + 4*256, "residual": 4 * H, "logitsStaging": 4 * V,
		"rmsQuant":    (2*L + 1) * (pk(h) + 4 + 16),
		"quant":       L * (pk(qW) + 4 + 16),
		"gemv":        L*(4*int64(qW+kvW+kvW+ffn+ffn)+5*16) + 4*V + 16,
		"gemvAdd":     L * 2 * 16,
		"actQuantPad": L * (pk(ffn) + 4 + 16),
		"attnLayer":   L * (4*int64(qW) + 16),
		"geomFor":     16 + 32,
	}
	if kv == "i8" || window { // the int8 V store's uniform; a sliding window's second attention uniform
		want["geomFor"] += 32
	}
	if noPre { // norms after the sublayer only: the raw residual is quantized in place of each pre-norm's fused norm
		want["rmsQuant"] = pk(h) + 20
		want["quant"] += 2 * L * (pk(h) + 4 + 16)
	}
	if kv == "f16" || kv == "i8" {
		want["ropeQUniFor"], want["ropeKUniFor"] = 32, 32
	} else {
		want["qkvFinUniFor"] = 32
	}
	switch bias { // q, k and v's bias: a separate add each on an int4 weight, folded into the projection on an int8 one
	case "int4":
		want["biasAdd"] = L * 3 * 16
	case "int8":
		q3 := L * (4*int64(qW+2*kvW) + 3*16)
		want["gemv"] -= q3
		want["gemvBias"] = q3
	}
	if qkNorm {
		want["qkNorm"] = L * 2 * 16
	}
	if post { // both sublayers: a bare gemv, a norm and a separate residual add in place of the fused projection
		delete(want, "gemvAdd") // no sublayer's output projection is fused
		want["gemv"] += L * 2 * (4*H + 16)
		want["rmsnormF32"] = L * 2 * (4*H + 16)
		want["biasAdd"] += L * 2 * 16
	}
	return want
}

// webgpuScratchHandMoE is the figure for an f32-KV model with no bias or post norms, layer by layer: each layer's attention, then its FFN, dense
// (ffn[l] > 0, that width) or routed (ffn[l] == 0, with nE experts of width inter, top k, and a shared expert of width shared, 0 for none,
// sigmoid-gated when gated).
func webgpuScratchHandMoE(h, v, qW, kvW int, ffn []int, nE, k, inter, shared int, gated, qkNorm bool) map[string]int64 {
	pk := func(n int) int64 { return int64((n + 31) &^ 31) }
	H, V := int64(h), int64(v)
	want := map[string]int64{"build": 4 + 4*256, "residual": 4 * H, "logitsStaging": 4 * V, "geomFor": 16 + 32, "qkvFinUniFor": 32,
		"rmsQuant": pk(h) + 20, "gemv": 4*V + 16}
	add := func(role string, n int64) { want[role] += n }
	anyMoE := false
	for _, f := range ffn {
		add("rmsQuant", 2*(pk(h)+20))
		add("gemv", 4*int64(qW+2*kvW)+3*16)
		add("quant", pk(qW)+20)
		add("attnLayer", 4*int64(qW)+16)
		add("gemvAdd", 16) // the output projection
		if qkNorm {
			add("qkNorm", 2*16)
		}
		if f > 0 {
			add("gemv", 4*int64(2*f)+2*16)
			add("actQuantPad", pk(f)+20)
			add("gemvAdd", 16)
			continue
		}
		anyMoE = true
		add("gemv", 4*int64(nE)+16)                  // the router's logits
		add("moeFFN", 4*int64(2*k)+4*int64(2*inter)) // the selection rows, the experts' gate and up rows
		add("moeRoute", 32)                          // the router's uniform
		add("moeExpert", 3*int64(k)*16)              // gate, up and down, a uniform each, per slot
		add("swigluQuant", int64(k)*(pk(inter)+20))  // per slot: the int8 row, its scale, its uniform
		if shared > 0 {
			add("gemv", 4*int64(2*shared)+2*16)
			add("swigluQuant", pk(shared)+20)
			if gated {
				add("gemv", 4*H+16+4+16)
				add("sharedGatedCombine", 16)
			} else {
				add("gemvAdd", 16)
			}
		}
	}
	if anyMoE {
		want["moeZeroBias"] = 4 * int64(nE)
	}
	return want
}

// webgpuScratchHandQwen35Tiny is qwen35-tiny's figure from its config (layer_types: three linear-attention layers then a full-attention one;
// hidden 64, vocabulary 256; 2 key heads of 16 and 4 value heads of 16; 4 query heads of 16 over 2 KV heads, with an output gate and a
// QK-norm; every FFN routed: 4 experts, top 2, width 32, and a sigmoid-gated shared expert of width 32).
func webgpuScratchHandQwen35Tiny() map[string]int64 {
	const H, V, L, nE, K, inter, shared = 64, 256, 4, 4, 2, 32, 32
	pk := func(n int) int64 { return int64((n + 31) &^ 31) }
	keyDim, valDim := 2*16, 4*16
	conv := 2*keyDim + valDim // the mixed projection's rows
	qW, kvW := 4*16, 2*16
	want := map[string]int64{"build": 4 + 4*256, "residual": 4 * H, "logitsStaging": 4 * V, "geomFor": 16 + 32, "qkvFinUniFor": 32,
		"rmsQuant": pk(H) + 20, "gemv": 4*V + 16, "deltaSetup": 16 + 16 + 32 + 32 + 32, "moeZeroBias": 4 * nE}
	add := func(role string, n int64) { want[role] += n }
	for l := range L {
		add("rmsQuant", 2*(pk(H)+20))
		add("gemvAdd", 16)
		if l < 3 { // linear attention: the mixed, beta, alpha and z projections; the conv, q, k, head-parameter, core and gated rows
			add("gemv", 4*int64(conv+4+4+valDim)+4*16)
			add("deltaLayer", 4*int64(conv+2*keyDim+2*4+valDim+valDim))
			add("quant", pk(valDim)+20)
		} else { // full attention: the wide query projection, k, v, the split, the gate, the QK-norm, the context
			add("gemv", 4*int64(2*qW+2*kvW)+3*16)
			add("qSplit", 4*2*int64(qW)+16)
			add("attnGate", 16)
			add("qkNorm", 2*16)
			add("attnLayer", 4*int64(qW)+16)
			add("quant", pk(qW)+20)
		}
		add("gemv", 4*int64(nE)+16)
		add("moeFFN", 4*2*K+4*2*inter)
		add("moeRoute", 32)
		add("moeExpert", 3*K*16)
		add("swigluQuant", int64(K)*(pk(inter)+20)+pk(shared)+20)
		add("gemv", 4*2*shared+2*16+4*H+16+4+16)
		add("sharedGatedCombine", 16)
	}
	return want
}

// webgpuScratchHandDeepseekTiny is deepseek-tiny's figure from its config (hidden 64, vocabulary 128, 4 layers of latent attention: 4 heads,
// query rank 24, latent rank 16, qk nope 16 and rope 8, v head 16; layer 0 a dense FFN of 128, layers 1 to 3 routed: 8 experts, top 2,
// width 32, an ungated shared expert of 32). Latent attention builds no per-geometry attention uniforms.
func webgpuScratchHandDeepseekTiny() map[string]int64 {
	const H, V, L, nH, qRank, rank, nope, rope, vHead = 64, 128, 4, 4, 24, 16, 16, 8, 16
	const nE, K, inter, shared, dense = 8, 2, 32, 32, 128
	pk := func(n int) int64 { return int64((n + 31) &^ 31) }
	qkHead, latDim := nope+rope, rank+rope
	want := map[string]int64{"build": 4 + 4*256, "residual": 4 * H, "logitsStaging": 4 * V, "mlaSetup": 32 + 48 + 32,
		"rmsQuant": pk(H) + 20, "gemv": 4*V + 16, "moeZeroBias": 4 * nE}
	add := func(role string, n int64) { want[role] += n }
	for l := range L {
		add("rmsQuant", 2*(pk(H)+20)+pk(qRank)+20) // the two pre-norms, and the query bottleneck's norm
		add("gemv", 4*int64(qRank)+16+4*int64(nH*qkHead)+16+4*int64(latDim)+16)
		add("mlaLayer", 4*int64(nH*latDim+nH*rank+nH*vHead)) // the absorbed query, the rank-space sum, the lifted context
		add("mlaAbsorb", 32)
		add("mlaLift", 32)
		add("quant", pk(nH*vHead)+20)
		add("gemvAdd", 16)
		if l == 0 {
			add("gemv", 4*2*dense+2*16)
			add("actQuantPad", pk(dense)+20)
			add("gemvAdd", 16)
			continue
		}
		add("gemv", 4*nE+16)
		add("moeFFN", 4*2*K+4*2*inter)
		add("moeRoute", 32)
		add("moeExpert", 3*K*16)
		add("swigluQuant", int64(K)*(pk(inter)+20)+pk(shared)+20)
		add("gemv", 4*2*shared+2*16)
		add("gemvAdd", 16) // the shared expert's down projection, fused
	}
	return want
}

// webgpuScratchHandNemotronTiny is nemotron-tiny's figure from its config: hidden 64, vocabulary 256, five layers of one operation each behind
// one norm (a state-space mixer whose input projection has 168 rows over an inner width of 64 and a conv of 96, attention of 4 heads of 16 over
// 2 KV heads, a non-gated FFN of 128, then a mixer and attention again).
func webgpuScratchHandNemotronTiny() map[string]int64 {
	return webgpuScratchHandNemotron([]string{"ssm", "attn", "mlp", "ssm", "attn"})
}

// webgpuScratchHandNemotron is the same layout over any run of its layer kinds; a "moe" layer is a routed block of 4 experts, top 2, width 32, with
// a shared expert of 64, each expert non-gated.
func webgpuScratchHandNemotron(kinds []string) map[string]int64 {
	const H, V, inner, convDim, projRows, nH, hd, nKV, up = 64, 256, 64, 96, 168, 4, 16, 2, 128
	pk := func(n int) int64 { return int64((n + 31) &^ 31) }
	want := map[string]int64{"build": 4 + 4*256, "residual": 4 * H, "logitsStaging": 4 * V, "geomFor": 16 + 32, "qkvFinUniFor": 32,
		"rmsQuant": pk(H) + 20, "gemv": 4*V + 16, "mambaSetup": 16 + 32 + 32}
	add := func(role string, n int64) { want[role] += n }
	const nE, K, expInter, shared = 4, 2, 32, 64
	for _, kind := range kinds {
		add("rmsQuant", pk(H)+20) // the layer's one norm
		add("gemvAdd", 16)
		switch kind {
		case "ssm":
			add("gemv", 4*projRows+16)
			add("mambaLayer", 4*(convDim+inner+inner))
			add("quant", pk(inner)+20)
		case "attn":
			add("gemv", 4*int64(nH*hd+2*nKV*hd)+3*16)
			add("attnLayer", 4*int64(nH*hd)+16)
			add("quant", pk(nH*hd)+20)
		case "mlp":
			add("gemv", 4*up+16)
			add("relu2Quant", pk(up)+20)
		case "moe": // the router's logits; the selection rows and each slot's up row; the router's uniform; the experts' uniforms; the activations; the shared expert
			add("gemv", 4*nE+16)
			add("nemoMoE", 4*2*K+4*K*expInter)
			add("moeRoute", 32)
			add("moeExpert", 2*K*16)
			add("relu2Quant", K*(pk(expInter)+20)+pk(shared)+20)
			add("gemv", 4*shared+16)
			want["moeZeroBias"] = 4 * nE
		}
	}
	return want
}

// webgpuScratchHandGptoss is gptoss_tiny's figure from its config: hidden 64, vocabulary 128, 4 layers of attention (8 query heads of 8 over 2 KV
// heads of 8, q/k/v bias, an output bias, a sliding window) and a routed FFN of 4 experts, top 2, width 32 with the clamped router. bias is
// "int4" or "int8", the weight kind that decides where q/k/v's bias goes.
func webgpuScratchHandGptoss(bias string) map[string]int64 {
	const H, V, L, qW, kvW, nE, K, inter = 64, 128, 4, 64, 16, 4, 2, 32
	pk := func(n int) int64 { return int64((n + 31) &^ 31) }
	want := map[string]int64{"build": 4 + 4*256, "residual": 4 * H, "logitsStaging": 4 * V, "geomFor": 16 + 32 + 32, "qkvFinUniFor": 32,
		"rmsQuant": pk(H) + 20, "gemv": 4*V + 16, "moeZeroBias": 4 * nE}
	add := func(role string, n int64) { want[role] += n }
	for range L {
		add("rmsQuant", 2*(pk(H)+20))
		add("quant", pk(qW)+20)
		add("attnLayer", 4*qW+16)
		qkv := int64(4*(qW+2*kvW) + 3*16)
		if bias == "int4" {
			add("gemv", qkv)
			add("biasAdd", 3*16)
		} else {
			add("gemvBias", qkv)
		}
		add("gemv", 4*H+16) // the output projection, bare: its bias is added, then it is added to the residual
		add("biasAdd", 2*16)
		add("gemv", 4*nE+16)
		add("moeFFN", 4*2*K+4*2*inter)
		add("routeGptOss", 16)
		add("moeExpert", 2*K*16)
		add("gptossActQuant", K*(pk(inter)+4+32))
		add("moeExpertGptOssDown", K*16)
	}
	return want
}

// webgpuScratchHandGemma4Tiny is gemma4-moe-kv-tiny's figure from its config: hidden 64, vocabulary 256, two layers, each a parallel dense (48) and routed
// (8 experts, top 2, width 16) block after attention with a QK-norm, a sliding window of 4 and a norm on its output. Layer 0 is local (4 heads of 16 over 2 KV
// heads); layer 1 is global (4 heads of 512 over 1 KV head, V the same projection as K, so it has no V projection). Both carry an output scalar.
func webgpuScratchHandGemma4Tiny() map[string]int64 {
	const H, V, nE, K, dense, moeInter = 64, 256, 8, 2, 48, 16
	pk := func(n int) int64 { return int64((n + 31) &^ 31) }
	want := map[string]int64{"build": 4 + 4*256, "residual": 4 * H, "logitsStaging": 4 * V, "geomFor": 2 * (16 + 32 + 32), "qkvFinUniFor": 2 * 32,
		"rmsQuant": pk(H) + 20, "gemv": 4*V + 16, "moeZeroBias": 4 * nE}
	add := func(role string, n int64) { want[role] += n }
	for _, c := range []struct {
		qW, kvW int
		vFromK  bool
	}{{4 * 16, 2 * 16, false}, {4 * 512, 512, true}} {
		add("rmsQuant", 3*(pk(H)+20)) // the attention's pre-norm and the block's two
		if c.vFromK {
			add("gemv", 4*int64(c.qW+c.kvW)+2*16)
		} else {
			add("gemv", 4*int64(c.qW+2*c.kvW)+3*16)
		}
		add("attnLayer", 4*int64(c.qW)+16+4*int64(c.kvW)) // the context, the sink flag, and the scale-less V norm's output row
		add("vNorm", 16)
		add("qkNorm", 2*16)
		add("quant", pk(c.qW)+20)
		add("gemv", 4*H+16) // the output projection is bare: its norm and the residual add follow
		add("rmsnormF32", 4*H+16)
		add("biasAdd", 16)
		// The block: the router's norm and the three post-norms; the router's rows and uniforms; the dense branch; the experts.
		add("rmsnormF32", 4*(4*H+16))
		add("gemma4MoE", 4*nE+4*2*K+4*H+4*3*moeInter+16+32+16+K*16)
		add("gemv", 4*2*dense+2*16+4*H+16)
		add("actQuantPad", pk(dense)+20+K*(pk(moeInter)+20))
		add("moeExpert", 2*K*16)
		add("scaleVec", 16+16) // the expert sum's clear, and the layer's output scalar
		add("biasAdd", 2*16)
	}
	return want
}

func withTerm(m map[string]int64, role string, b int64) map[string]int64 {
	m[role] = b
	return m
}

// TestWebGPUDecodeScratch_handTable is G-MS3w: the figure for llama-tiny at each KV precision, qwen3asr-tiny (a QK-norm) and glm-ocr-tiny
// (a norm on each sublayer's output), term by term as arithmetic on the config; and no figure where a layer is not a plain shape.
func TestWebGPUDecodeScratch_handTable(t *testing.T) {
	for _, c := range []struct {
		name, fx string
		o        decoder.Options
		want     map[string]int64
	}{
		{"llama f32", "llama-tiny", decoder.Options{}, webgpuScratchHand(64, 4, 256, 64, 32, 128, false, false, "f32", "", false, false)},
		{"llama f16", "llama-tiny", decoder.Options{KVPrecision: "f16"}, webgpuScratchHand(64, 4, 256, 64, 32, 128, false, false, "f16", "", false, false)},
		{"llama i8", "llama-tiny", decoder.Options{KVPrecision: "i8"}, webgpuScratchHand(64, 4, 256, 64, 32, 128, false, false, "i8", "", false, false)},
		{"qwen3asr (qk-norm)", "qwen3asr-tiny", decoder.Options{}, webgpuScratchHand(24, 2, 160, 24, 12, 48, true, false, "f32", "", false, false)},
		{"qwen25vl int4 (bias)", "qwen25vl-tiny", decoder.Options{Quant: "int4"}, webgpuScratchHand(64, 2, 300, 64, 32, 128, false, false, "f32", "int4", false, false)},
		{"qwen25vl int8 (bias)", "qwen25vl-tiny", decoder.Options{Quant: "int8int8"}, webgpuScratchHand(64, 2, 300, 64, 32, 128, false, false, "f32", "int8", false, false)},
		{"olmo3 (window, qk-norm, post-only norms)", "olmo3-tiny", decoder.Options{}, withTerm(webgpuScratchHand(64, 4, 512, 64, 64, 128, true, true, "f32", "", true, true), "qkvFinUniFor", 2*32)}, // two rope scales: the sliding layers' and the full ones'
		{"mistral window", "mistral-tiny-window", decoder.Options{}, webgpuScratchHand(128, 3, 128, 128, 64, 256, false, false, "f32", "", true, false)},
		{"mixtral (routed FFN)", "mixtral-tiny", decoder.Options{}, webgpuScratchHandMoE(64, 512, 64, 64, []int{0, 0, 0, 0}, 8, 2, 128, 0, false, false)},
		{"qwen3moe (routed FFN, qk-norm)", "qwen3moe-tiny", decoder.Options{}, webgpuScratchHandMoE(64, 512, 64, 16, []int{0, 0, 0, 0}, 8, 2, 32, 0, false, true)},
		{"glm (dense prefix, ungated shared expert, qk-norm)", "glm-tiny", decoder.Options{}, webgpuScratchHandMoE(64, 256, 64, 32, []int{128, 0, 0}, 8, 2, 32, 32, false, true)},
		{"qwen35 (delta-net, gated attention, routed FFN)", "qwen35-tiny", decoder.Options{}, webgpuScratchHandQwen35Tiny()},
		{"smollm3 (a layer with no rope operation)", "smollm3-tiny", decoder.Options{}, webgpuScratchHand(64, 4, 512, 64, 16, 128, false, false, "f32", "", false, false)},
		{"deepseek (latent attention, dense then routed FFN)", "deepseek-tiny", decoder.Options{}, webgpuScratchHandDeepseekTiny()},
		{"nemotron (a layer of one operation: state-space, attention, non-gated FFN)", "nemotron-tiny", decoder.Options{}, webgpuScratchHandNemotronTiny()},
		{"gpt-oss int4 (bias, output bias, window, clamped router)", "gptoss_tiny.gguf", decoder.Options{Quant: "int4"}, webgpuScratchHandGptoss("int4")},
		{"gpt-oss int8 (bias folded into the projection)", "gptoss_tiny.gguf", decoder.Options{Quant: "int8int8"}, webgpuScratchHandGptoss("int8")},
		{"nemotron3nano (routed blocks)", "nemotron3nano-tiny", decoder.Options{}, webgpuScratchHandNemotron([]string{"ssm", "moe", "ssm", "attn", "ssm", "moe"})},
		{"gemma4 (parallel dense and routed block, two geometries, V from K)", "gemma4-moe-kv-tiny", decoder.Options{}, webgpuScratchHandGemma4Tiny()},
		{"glm-ocr (post norms)", "glm-ocr-tiny", decoder.Options{}, webgpuScratchHand(48, 3, 160, 128, 64, 144, false, true, "f32", "", false, false)},
	} {
		path := "../../testdata/" + c.fx
		if c.fx == "gptoss_tiny.gguf" {
			path = "../../decoder/testdata/" + c.fx
		}
		m, err := decoder.Load(path, c.o)
		if err != nil {
			t.Fatal(err)
		}
		terms, ok := webgpuDecodeScratchTerms(m)
		if !ok {
			t.Fatalf("%s: no figure", c.name)
		}
		got := map[string]int64{}
		for _, tm := range terms {
			got[tm.Role] += tm.Bytes
		}
		for k, v := range c.want {
			if got[k] != v {
				t.Errorf("%s: %s = %d B, hand arithmetic says %d B", c.name, k, got[k], v)
			}
		}
		if len(got) != len(c.want) {
			t.Errorf("%s: %d terms, the hand table has %d", c.name, len(got), len(c.want))
		}
		m.Close()
	}
	m, err := decoder.Load("../../testdata/cohere-tiny", decoder.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, ok := webgpuDecodeScratchTerms(m); ok {
		t.Error("cohere-tiny runs its attention and FFN in parallel, a layer shape whose scratch is not derived: the plan must give no figure")
	}
}

// scanPricingBodies reports every function or variable in funcs (file to names) whose body or value names one of banned, and every
// name it finds no declaration for.
func scanPricingBodies(t *testing.T, funcs map[string][]string, banned []string, why string) {
	t.Helper()
	for file, names := range funcs {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		f, err := parser.ParseFile(fs, file, src, 0)
		if err != nil {
			t.Fatal(err)
		}
		found := map[string]bool{}
		bodies := map[string]ast.Node{} // each declaration's body (a function's) or value (a variable's)
		for _, d := range f.Decls {
			switch d := d.(type) {
			case *ast.FuncDecl:
				if d.Body != nil {
					bodies[d.Name.Name] = d.Body
				}
			case *ast.GenDecl:
				for _, sp := range d.Specs {
					if vs, ok := sp.(*ast.ValueSpec); ok {
						for i, n := range vs.Names {
							if i < len(vs.Values) {
								bodies[n.Name] = vs.Values[i]
							}
						}
					}
				}
			}
		}
		for _, name := range names {
			node, ok := bodies[name]
			if !ok {
				continue
			}
			found[name] = true
			body := string(src[fs.Position(node.Pos()).Offset:fs.Position(node.End()).Offset])
			for _, b := range banned {
				if strings.Contains(body, b) {
					t.Errorf("%s: %s names %q: %s", file, name, b, why)
				}
			}
		}
		for _, name := range names {
			if !found[name] {
				t.Errorf("%s: no func %s to scan; the scan reads nothing for it", file, name)
			}
		}
	}
}
