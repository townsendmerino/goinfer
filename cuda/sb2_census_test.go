//go:build cuda && goinfer_testhooks

package cuda

import (
	"encoding/json"
	"errors"
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

// Layer B-cuda of the admission and routing census (docs/tasks/task-subsystems-2026-10.md, "SB2 step 2"): CUDA's own static
// admission, the declines buildResident makes from the model alone, on layer D's fixtures and option cells. A fake backend named
// cuda takes the decoder's real admission; where that admits, it runs buildResident in plan-only mode (every static decline, then
// a return before anything is allocated), records the answer and declines, so nothing is uploaded and no device is needed. A cell
// the decoder declines is layer D's line (decoder's TestAdmissionCensus) and is not repeated here.

var cudaCensusUpdate = flag.Bool("sb2census.update", false, "rewrite this architecture's testdata/sb2_cuda_census.<goarch>.txt.gz")

// censusBackend is the fake: named cuda, it records the plan-only build of what the decoder admitted and declines.
type censusBackend struct {
	decoder.Backend
	mu      sync.Mutex
	reached bool
	answer  string
}

func (b *censusBackend) Name() string { return "cuda" }

// SelfTestEligible keeps the decoder's startup self-test (CUDA kernels on the device) out of the census: that is a device decision.
func (b *censusBackend) SelfTestEligible() (bool, string) { return false, "admission census" }

func (b *censusBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	_, _, err := (&cudaBackend{}).buildResident(m, true)
	var s planOnlySlack
	ans := "plan returned no answer"
	switch {
	case errors.As(err, &s):
		ans = fmt.Sprintf("plan slack=%d", int64(s))
	case err != nil:
		ans = "decline " + err.Error()
	}
	b.mu.Lock()
	b.reached, b.answer = true, ans
	b.mu.Unlock()
	return nil, false, decoder.DeclineResident("admission census: plan only")
}

func (b *censusBackend) take() (bool, string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	r, a := b.reached, b.answer
	b.reached, b.answer = false, ""
	return r, a
}

// buildCUDACensus returns one line per cell that reached CUDA's build. Every cell layer D built resident must be among them.
func buildCUDACensus(t *testing.T) []string {
	t.Helper()
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "")
	// The fit guard's host reading is fixed, so no reading of this machine decides a line; the fake's registered name has no
	// memory probe or build demand of its own, as in layer D.
	t.Cleanup(decoder.SetHostRAMAvailableForTest(48 << 30))
	be := &censusBackend{}
	const reg = "sb2-census-cuda"
	decoder.RegisterBackend(reg, func() (decoder.Backend, error) {
		cpu, err := decoder.NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	t.Cleanup(func() { decoder.UnregisterBackendForTest(reg) })
	d, err := sb2census.ReadLayerD("..", "cuda")
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
				lines = append(lines, fmt.Sprintf("B %s %s cuda %q", f, c.Name, sb2census.Clean(ans)))
			} else if d.Resident[f+" "+c.Name] {
				t.Errorf("%s %s: layer D builds it resident on cuda, but CUDA's build was not reached", f, c.Name)
			}
		}
	}
	return lines
}

// TestBackendAdmissionCensus_cuda holds layer B-cuda to testdata/sb2_cuda_census.<goarch>.txt.gz, line for line, built twice.
func TestBackendAdmissionCensus_cuda(t *testing.T) {
	start := time.Now()
	a, b := buildCUDACensus(t), buildCUDACensus(t)
	if strings.Join(a, "\n") != strings.Join(b, "\n") {
		t.Fatalf("two builds in one process differ")
	}
	t.Logf("%d census lines in %s", len(a), time.Since(start).Round(time.Millisecond))
	golden := sb2census.Golden("testdata", "sb2_cuda_census")
	if *cudaCensusUpdate {
		if err := sb2census.Write(golden, a); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", golden)
		return
	}
	diffs, err := sb2census.Diff(golden, a, 4)
	if err != nil {
		for _, l := range a {
			fmt.Println("SB2CUDACENSUS " + l)
		}
		t.Fatalf("%v: the lines above (prefixed SB2CUDACENSUS) are this architecture's census", err)
	}
	if len(diffs) > 0 {
		t.Errorf("the cuda admission census differs from %s in %d cell(s):\n%s", golden, len(diffs), strings.Join(diffs[:min(len(diffs), 30)], "\n"))
	}
}

// TestBackendAdmissionCensus_cuda_plant shows the census can go red: with the per-32 MoE decline planted off, every MoE cell at
// per-32 activations that reaches CUDA's build must differ from the golden, and no other cell may.
func TestBackendAdmissionCensus_cuda_plant(t *testing.T) {
	golden := sb2census.Golden("testdata", "sb2_cuda_census")
	clean := buildCUDACensus(t)
	const reason = "per-32 activations: the MoE expert/router kernels"
	wantRed := 0
	for _, l := range clean {
		if strings.Contains(l, reason) {
			wantRed++
		}
	}
	if wantRed == 0 {
		t.Fatal("no cell takes the planted decline: the plant has nothing to change")
	}
	defer func(off bool) { moeActG32DeclineOffForTest = off }(moeActG32DeclineOffForTest)
	moeActG32DeclineOffForTest = true
	diffs, err := sb2census.Diff(golden, buildCUDACensus(t), 4)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range diffs {
		if !strings.Contains(d, reason) {
			t.Errorf("planted MoE per-32 decline off: a cell outside the plant changed: %s", d)
		}
	}
	if len(diffs) != wantRed {
		t.Fatalf("planted MoE per-32 decline off: %d cell(s) differ from the golden, want %d", len(diffs), wantRed)
	}
	t.Logf("planted MoE per-32 decline off: %d of %d cells red, as expected", len(diffs), len(clean))
}

// The memory census (docs/tasks/task-subsystems-2026-10.md, "SB2 memory reads, step 2"): cuda's memory decisions
// (cudaBackend.planMemory) on every cell layer B-cuda plans, fed each recorded machine's free-VRAM figure. The KV fit reads free
// VRAM after allocation and is not fed here.

var cudaMemCensusUpdate = flag.Bool("sb2memcensus.update", false, "rewrite this architecture's testdata/sb2_cuda_mem_census.<goarch>.txt.gz")

// memProfile is a recorded machine with a CUDA free-VRAM figure, and moe_route's pool when it was measured there.
type memProfile struct {
	Name     string `json:"name"`
	CUDAFree *int64 `json:"cuda_free_bytes"`
	MoEPool  *int64 `json:"cuda_moe_route_pool_bytes"`
}

func memProfiles(t *testing.T) []memProfile {
	t.Helper()
	paths, err := filepath.Glob("../testdata/machines/*.json")
	if err != nil {
		t.Fatal(err)
	}
	var ps []memProfile
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var m memProfile
		if err := json.Unmarshal(b, &m); err != nil || m.Name == "" {
			t.Fatalf("%s: not a machine profile: %v", p, err)
		}
		if m.CUDAFree != nil {
			ps = append(ps, m)
		}
	}
	if len(ps) == 0 {
		t.Fatal("no machine profile carries cuda_free_bytes")
	}
	return ps
}

// memCensusBackend is the fake for the memory census: where the decoder admits, it plans the build's memory once per profile
// and declines.
type memCensusBackend struct {
	decoder.Backend
	profiles []memProfile
	mu       sync.Mutex
	lines    []string
	reached  bool
}

func (b *memCensusBackend) Name() string                     { return "cuda" }
func (b *memCensusBackend) SelfTestEligible() (bool, string) { return false, "memory census" }

func (b *memCensusBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	var lines []string
	for _, pr := range b.profiles {
		lines = append(lines, memCensusLine(m, pr))
	}
	b.mu.Lock()
	b.reached, b.lines = true, lines
	b.mu.Unlock()
	return nil, false, decoder.DeclineResident("memory census")
}

// memCensusLine plans m against profile pr. Every reading is the profile's figure, except that a MoE build whose context is
// planned pays moe_route's pool before its plan, so its plan and setup readings are the figure less the profile's pool.
func memCensusLine(m *decoder.Model, pr memProfile) string {
	p := *pr.CUDAFree
	in := memReadings{WeightsOpen: p, Plan: p, Weights: p, WeightsOpenOK: true, PlanOK: true, WeightsOK: true}
	plan, err := (&cudaBackend{}).planMemory(m, in)
	if err != nil {
		return fmt.Sprintf("%s static %s", pr.Name, err)
	}
	figures := fmt.Sprintf("weightsOpen=%d(profile) plan=%d(profile) weights=%d(profile)", p, p, p)
	if plan.PoolPaid {
		if pr.MoEPool == nil {
			return fmt.Sprintf("%s moe: no cuda_moe_route_pool_bytes in the profile, not censused", pr.Name)
		}
		after := p - *pr.MoEPool
		in.Plan, in.Weights = after, after
		if plan, err = (&cudaBackend{}).planMemory(m, in); err != nil {
			return fmt.Sprintf("%s static %s", pr.Name, err)
		}
		figures = fmt.Sprintf("weightsOpen=%d(profile) plan=%d(profile-pool) weights=%d(profile-pool)", p, after, after)
	}
	verdict := "weights ok"
	if plan.Decline != "" {
		verdict = "decline@" + plan.DeclineAt + " " + plan.Decline
	}
	return fmt.Sprintf("%s %s ctxPlanned=%d margin=%d extra=%d %s kv: step 3", pr.Name, figures, plan.CtxPlanned, plan.Margin, plan.ExtraBytes, verdict)
}

// buildCUDAMemCensus returns one line per cell layer B-cuda plans and profile.
func buildCUDAMemCensus(t *testing.T) []string {
	t.Helper()
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "")
	t.Cleanup(decoder.SetHostRAMAvailableForTest(48 << 30))
	be := &memCensusBackend{profiles: memProfiles(t)}
	const reg = "sb2-memcensus-cuda"
	decoder.RegisterBackend(reg, func() (decoder.Backend, error) {
		cpu, err := decoder.NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	t.Cleanup(func() { decoder.UnregisterBackendForTest(reg) })
	d, err := sb2census.ReadLayerD("..", "cuda")
	if err != nil {
		t.Fatal(err)
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
			if err == nil {
				m.Close()
			}
			be.mu.Lock()
			reached, ls := be.reached, be.lines
			be.reached, be.lines = false, nil
			be.mu.Unlock()
			if !reached {
				continue
			}
			for _, l := range ls {
				if strings.Contains(l, " static ") {
					continue // a static decline is layer B-cuda's line
				}
				lines = append(lines, fmt.Sprintf("M %s %s cuda %s", f, c.Name, sb2census.Clean(l)))
			}
		}
	}
	return lines
}

// TestBackendMemoryCensus_cuda holds the memory census to testdata/sb2_cuda_mem_census.<goarch>.txt.gz, built twice, and checks
// it covers every cell layer B-cuda plans, once per profile.
func TestBackendMemoryCensus_cuda(t *testing.T) {
	a, b := buildCUDAMemCensus(t), buildCUDAMemCensus(t)
	if strings.Join(a, "\n") != strings.Join(b, "\n") {
		t.Fatal("two builds in one process differ")
	}
	planned := 0
	for _, l := range buildCUDACensus(t) {
		if strings.Contains(l, "plan slack=") {
			planned++
		}
	}
	if want := planned * len(memProfiles(t)); len(a) != want {
		t.Errorf("%d memory census lines, want %d (%d planned cells x %d profiles)", len(a), want, planned, len(memProfiles(t)))
	}
	t.Logf("%d memory census lines (%d planned cells)", len(a), planned)
	golden := sb2census.Golden("testdata", "sb2_cuda_mem_census")
	if *cudaMemCensusUpdate {
		if err := sb2census.Write(golden, a); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", golden)
		return
	}
	diffs, err := sb2census.Diff(golden, a, 4)
	if err != nil {
		for _, l := range a {
			fmt.Println("SB2CUDAMEMCENSUS " + l)
		}
		t.Fatalf("%v: the lines above (prefixed SB2CUDAMEMCENSUS) are this architecture's memory census", err)
	}
	if len(diffs) > 0 {
		t.Errorf("the cuda memory census differs from %s in %d line(s):\n%s", golden, len(diffs), strings.Join(diffs[:min(len(diffs), 30)], "\n"))
	}
}

// TestBackendMemoryCensus_cuda_plant shows the memory census reads the decision: with weightsFitDecision's margin planted a
// thousandfold, every line must differ from the golden as a weights refusal.
func TestBackendMemoryCensus_cuda_plant(t *testing.T) {
	golden := sb2census.Golden("testdata", "sb2_cuda_mem_census")
	clean := buildCUDAMemCensus(t)
	defer func(on bool) { weightsMarginPlantForTest = on }(weightsMarginPlantForTest)
	weightsMarginPlantForTest = true
	planted := buildCUDAMemCensus(t)
	diffs, err := sb2census.Diff(golden, planted, 4)
	if err != nil {
		t.Fatal(err)
	}
	red := 0
	for _, l := range planted {
		if strings.Contains(l, "decline@weights") {
			red++
		}
	}
	if red != len(clean) || len(diffs) == 0 {
		t.Fatalf("planted margin: %d of %d lines refuse at the weights, %d differ from the golden; want all %d", red, len(planted), len(diffs), len(clean))
	}
	t.Logf("planted margin: all %d lines refuse at the weights", red)
}
