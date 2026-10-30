//go:build cuda && goinfer_testhooks

package cuda

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
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
	profiles   []memProfile
	mu         sync.Mutex
	lines      []string
	mismatches []string
	reached    bool
}

func (b *memCensusBackend) Name() string                     { return "cuda" }
func (b *memCensusBackend) SelfTestEligible() (bool, string) { return false, "memory census" }

func (b *memCensusBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	var lines, mismatches []string
	for _, pr := range b.profiles {
		l, mm := memCensusLine(m, pr)
		lines = append(lines, l)
		if mm != "" {
			mismatches = append(mismatches, pr.Name+": "+mm)
		}
	}
	b.mu.Lock()
	b.reached, b.lines = true, lines
	b.mismatches = append(b.mismatches, mismatches...)
	b.mu.Unlock()
	return nil, false, decoder.DeclineResident("memory census")
}

// memCensusLine plans m against profile pr. Every reading is the profile's figure, except that a MoE build whose context is
// planned pays moe_route's pool before its plan, so its plan and setup readings are the figure less the profile's pool. A line that
// reaches the KV fit goes on with the post-allocation decisions derived from that figure (cudaBackend.deriveMemory), each derived
// again at the model's error bound either way and marked ~ where that changes it. mismatch is non-empty when the derive's
// pre-allocation decisions differ from planMemory's (G-DC1).
func memCensusLine(m *decoder.Model, pr memProfile) (line, mismatch string) {
	p := *pr.CUDAFree
	in := memReadings{WeightsOpen: p, Plan: p, Weights: p, WeightsOpenOK: true, PlanOK: true, WeightsOK: true}
	plan, err := (&cudaBackend{}).planMemory(m, in)
	if err != nil {
		return fmt.Sprintf("%s static %s", pr.Name, err), ""
	}
	figures := fmt.Sprintf("weightsOpen=%d(profile) plan=%d(profile) weights=%d(profile)", p, p, p)
	if plan.PoolPaid {
		if pr.MoEPool == nil {
			return fmt.Sprintf("%s moe: no cuda_moe_route_pool_bytes in the profile, not censused", pr.Name), ""
		}
		after := p - *pr.MoEPool
		in.Plan, in.Weights = after, after
		if plan, err = (&cudaBackend{}).planMemory(m, in); err != nil {
			return fmt.Sprintf("%s static %s", pr.Name, err), ""
		}
		figures = fmt.Sprintf("weightsOpen=%d(profile) plan=%d(profile-pool) weights=%d(profile-pool)", p, after, after)
	}
	verdict := "weights ok"
	if plan.Decline != "" {
		verdict = "decline@" + plan.DeclineAt + " " + plan.Decline
	}
	head := fmt.Sprintf("%s %s ctxPlanned=%d margin=%d extra=%d %s", pr.Name, figures, plan.CtxPlanned, plan.Margin, plan.ExtraBytes, verdict)
	if plan.Decline != "" {
		return head, ""
	}
	var pool int64
	if pr.MoEPool != nil {
		pool = *pr.MoEPool
	}
	b := &cudaBackend{}
	base, rd, err := b.deriveMemoryAt(m, p, pool, memCensusSkewForTest)
	if err != nil {
		return head, "derive: " + err.Error()
	}
	if base.CtxPlanned != plan.CtxPlanned || base.Margin != plan.Margin || base.ExtraBytes != plan.ExtraBytes || base.PoolPaid != plan.PoolPaid ||
		rd.Plan != in.Plan || rd.Weights != in.Weights || base.DeclineAt == "weightsOpen" || base.DeclineAt == "weights" {
		return head, fmt.Sprintf("derive ctxPlanned=%d margin=%d extra=%d pool=%v plan=%d weights=%d decline@%s against the plan's ctxPlanned=%d margin=%d extra=%d pool=%v plan=%d weights=%d",
			base.CtxPlanned, base.Margin, base.ExtraBytes, base.PoolPaid, rd.Plan, rd.Weights, base.DeclineAt, plan.CtxPlanned, plan.Margin, plan.ExtraBytes, plan.PoolPaid, in.Plan, in.Weights)
	}
	lo, _, errLo := b.deriveMemoryAt(m, p, pool, memCensusSkewForTest-memCensusBound)
	hi, _, errHi := b.deriveMemoryAt(m, p, pool, memCensusSkewForTest+memCensusBound)
	if errLo != nil || errHi != nil {
		return head, fmt.Sprintf("derive at the bound: %v / %v", errLo, errHi)
	}
	return head + " " + derivedTail(base, rd, nearMarks(base, lo, hi)), ""
}

// memCensusBound is G-DM2's bound: the derived readings are within it of the card's (docs/measurements/sb2-derived-2026-10-29/README.md).
const memCensusBound = 64 << 20

// memCensusSkewForTest is added to every derived reading (G-DC2's plant). Tests that set it do not run in parallel.
var memCensusSkewForTest int64

// nearMarks names the post-allocation decisions that differ in lo or hi (the derive at the bound either way) from base: "kv" (the
// refusal site), "ctx", "kvSlots", "ckpt" (kept or not), "slack", "cacheSlots", "prefill".
func nearMarks(base, lo, hi memPlan) map[string]bool {
	marks := map[string]bool{}
	for _, o := range []memPlan{lo, hi} {
		for k, differ := range map[string]bool{
			"kv":         o.DeclineAt != base.DeclineAt,
			"ctx":        o.CtxCap != base.CtxCap,
			"kvSlots":    o.KVSlots != base.KVSlots,
			"ckpt":       o.CkptBytes != base.CkptBytes,
			"slack":      o.KVSlack != base.KVSlack,
			"cacheSlots": o.CacheSlots != base.CacheSlots,
			"prefill":    o.PrefillWidth != base.PrefillWidth,
		} {
			if differ {
				marks[k] = true
			}
		}
	}
	return marks
}

// derivedTail is a line's post-allocation part: each reading the build takes, as derived, then the decisions (the expert-cache slots
// only where the cache is sized), a ~ on those nearMarks names.
func derivedTail(pl memPlan, rd memReadings, marks map[string]bool) string {
	mark := func(k string) string {
		if marks[k] {
			return "~"
		}
		return ""
	}
	parts := []string{"derived(G-DM2 band -62/+58 MiB):"}
	for _, f := range []struct {
		site string
		v    int64
		ok   bool
	}{{"kv", rd.KV, rd.KVOK}, {"ckpt", rd.Ckpt, rd.CkptOK}, {"grantPre", rd.GrantPre, rd.GrantPreOK}, {"slots", rd.Slots, rd.SlotsOK},
		{"grant", rd.Grant, rd.GrantOK}, {"prefill", rd.Prefill, rd.PrefillOK}} {
		if f.ok {
			parts = append(parts, fmt.Sprintf("%s@%d", f.site, f.v))
		}
	}
	if pl.DeclineAt == "kv" {
		return strings.Join(append(parts, "decline@kv"+mark("kv")+" "+pl.Decline), " ")
	}
	ckpt := "none"
	switch {
	case pl.CkptBytes > 0:
		ckpt = strconv.FormatInt(pl.CkptBytes, 10)
	case pl.CkptDeclined != "":
		ckpt = "declined"
	}
	parts = append(parts, fmt.Sprintf("ctx=%d%s kvSlots=%d%s ckpt=%s%s slack=%d%s", pl.CtxCap, mark("ctx"), pl.KVSlots, mark("kvSlots"),
		ckpt, mark("ckpt"), pl.KVSlack, mark("slack")))
	if pl.DeclineAt == "slots" {
		return strings.Join(append(parts, "decline@slots"+mark("kv")+" "+pl.Decline), " ")
	}
	if rd.SlotsOK {
		parts = append(parts, fmt.Sprintf("cacheSlots=%d%s", pl.CacheSlots, mark("cacheSlots")))
	}
	return strings.Join(append(parts, fmt.Sprintf("prefill=%d%s", pl.PrefillWidth, mark("prefill"))), " ")
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
			reached, ls, mms := be.reached, be.lines, be.mismatches
			be.reached, be.lines, be.mismatches = false, nil, nil
			be.mu.Unlock()
			if !reached {
				continue
			}
			for _, mm := range mms {
				t.Errorf("G-DC1: %s %s: %s", f, c.Name, mm)
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
	marked := 0
	for _, l := range a {
		if strings.Contains(l, "~") {
			marked++
		}
	}
	t.Logf("%d memory census lines (%d planned cells); %d carry a decision the 64 MiB bound cannot settle (~)", len(a), planned, marked)
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

// TestBackendMemoryCensus_cuda_plant shows the memory census reads the decision and its MoE figure: with the setup job's weights
// check planted a thousandfold, every line must refuse there, and a MoE line's refusal must name the post-pool figure it was fed.
func TestBackendMemoryCensus_cuda_plant(t *testing.T) {
	golden := sb2census.Golden("testdata", "sb2_cuda_mem_census")
	clean := buildCUDAMemCensus(t)
	defer func(site string) { weightsMarginPlantSiteForTest = site }(weightsMarginPlantSiteForTest)
	weightsMarginPlantSiteForTest = "weights"
	planted := buildCUDAMemCensus(t)
	diffs, err := sb2census.Diff(golden, planted, 4)
	if err != nil {
		t.Fatal(err)
	}
	// Every line must refuse at the setup job's weights check, not at a MoE build's earlier one, so a MoE line's refusal is judged
	// against the post-pool figure: its text must name that figure, which is what shows the census feeds it.
	red, post := 0, 0
	for _, l := range planted {
		if !strings.Contains(l, " decline@weights ") {
			continue
		}
		red++
		if m := regexp.MustCompile(`weights=(\d+)\(profile-pool\)`).FindStringSubmatch(l); m != nil {
			f, _ := strconv.ParseInt(m[1], 10, 64)
			if want := fmt.Sprintf("only %.2f GB is free", float64(f)/1e9); !strings.Contains(l, want) {
				t.Errorf("planted margin: a MoE line's refusal does not name the post-pool figure (%q): %s", want, l)
			}
			post++
		}
	}
	if red != len(clean) || len(diffs) == 0 {
		t.Fatalf("planted margin: %d of %d lines refuse at the setup job's weights check, %d differ from the golden; want all %d", red, len(planted), len(diffs), len(clean))
	}
	if post == 0 {
		t.Fatal("planted margin: no line was judged against the post-pool figure: the census's MoE figure is unexercised")
	}
	t.Logf("planted margin: all %d lines refuse at the weights, %d of them against the post-pool figure", red, post)
}

// TestBackendMemoryCensus_cuda_derivedPlant is G-DC2: with every derived reading skewed by -8 GiB, more than any profile holds,
// every line that reaches the KV fit must refuse there, and no line refused before it may change.
func TestBackendMemoryCensus_cuda_derivedPlant(t *testing.T) {
	clean := buildCUDAMemCensus(t)
	defer func(v int64) { memCensusSkewForTest = v }(memCensusSkewForTest)
	memCensusSkewForTest = -8 << 30
	planted := buildCUDAMemCensus(t)
	if len(planted) != len(clean) {
		t.Fatalf("planted skew: %d lines, want %d", len(planted), len(clean))
	}
	reached, before := 0, 0
	for i := range clean {
		if !strings.Contains(clean[i], " derived(") {
			before++
			if planted[i] != clean[i] {
				t.Errorf("planted skew: a line refused before the KV fit changed:\n%s\n%s", clean[i], planted[i])
			}
			continue
		}
		reached++
		if planted[i] == clean[i] || !strings.Contains(planted[i], " decline@kv") {
			t.Errorf("planted skew: a line that reaches the KV fit does not refuse there: %s", planted[i])
		}
	}
	if reached == 0 {
		t.Fatal("planted skew: no line reaches the KV fit: the derived tail is unexercised")
	}
	t.Logf("planted skew: all %d lines that reach the KV fit refuse there; %d refused before it are unchanged", reached, before)
}

// TestNearMarks is G-DC3: a decision is marked when it differs at either end of the bound, and only then.
func TestNearMarks(t *testing.T) {
	base := memPlan{CtxCap: 4096, KVSlots: 1, CkptBytes: 10, KVSlack: 512, CacheSlots: 28, PrefillWidth: 2048}
	for _, tc := range []struct {
		name string
		edit func(*memPlan)
		want string
	}{
		{"ctx", func(p *memPlan) { p.CtxCap = 2048 }, "ctx"},
		{"kvSlots", func(p *memPlan) { p.KVSlots = 2 }, "kvSlots"},
		{"ckpt", func(p *memPlan) { p.CkptBytes, p.CkptDeclined = 0, "too small" }, "ckpt"},
		{"slack", func(p *memPlan) { p.KVSlack = 1088 }, "slack"},
		{"cacheSlots", func(p *memPlan) { p.CacheSlots = 27 }, "cacheSlots"},
		{"prefill", func(p *memPlan) { p.PrefillWidth = 1024 }, "prefill"},
		{"refusal", func(p *memPlan) { p.Decline, p.DeclineAt = "no room", "kv" }, "kv"},
	} {
		changed := base
		tc.edit(&changed)
		for _, side := range []string{"lo", "hi"} {
			lo, hi := base, base
			if side == "lo" {
				lo = changed
			} else {
				hi = changed
			}
			got := nearMarks(base, lo, hi)
			if len(got) != 1 || !got[tc.want] {
				t.Errorf("%s changed at %s: marks %v, want only %s", tc.name, side, got, tc.want)
			}
		}
	}
	if got := nearMarks(base, base, base); len(got) != 0 {
		t.Errorf("nothing changed: marks %v, want none", got)
	}
}
