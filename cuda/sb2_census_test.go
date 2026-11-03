//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/program"
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
	var lines, mismatches, tight []string
	for _, pr := range b.profiles {
		l, mm := memCensusLine(m, pr)
		lines = append(lines, l)
		if mm != "" {
			mismatches = append(mismatches, pr.Name+": "+mm)
		}
		if hl, hmm, ok := memCensusHalfCacheLine(m, pr); ok {
			tight = append(tight, hl)
			if hmm != "" {
				mismatches = append(mismatches, pr.Name+memCensusHalfCache+": "+hmm)
			}
		}
	}
	// After every profile's line, so a cell's lines stay aligned with its int4 line's by index (buildCUDAMemCensusCells).
	lines = append(lines, tight...)
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
// pre-allocation decisions differ from planMemory's, a refusal before the KV fit included (G-DC1).
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
	var pool int64
	if pr.MoEPool != nil {
		pool = *pr.MoEPool
	}
	b := &cudaBackend{}
	base, rd, err := b.deriveMemoryAt(m, p, pool, memCensusSkewForTest)
	if err != nil {
		return head, "derive: " + err.Error()
	}
	if plan.Decline != "" {
		if base.DeclineAt != plan.DeclineAt || base.Decline != plan.Decline {
			return head, fmt.Sprintf("derive refuses at %q (%s), the plan at %q (%s)", base.DeclineAt, base.Decline, plan.DeclineAt, plan.Decline)
		}
		return head, ""
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

// memCensusHalfCacheLine is pr's line at a tight figure: the largest, to the byte, at which m's derived expert cache holds at most half
// the slots it holds at pr's own figure and the derive does not refuse. ok is false where the plan sizes no cache of two slots or
// more, or no figure caps it partway: a cache whose slot buffers all fit one allocation quantum costs the same at any slot count.
func memCensusHalfCacheLine(m *decoder.Model, pr memProfile) (line, mismatch string, ok bool) {
	var pool int64
	if pr.MoEPool != nil {
		pool = *pr.MoEPool
	}
	b := &cudaBackend{}
	// The tight figure belongs to the clean plan: it is found with the plan's plants off, and the line is then derived at it under
	// whatever plant is set, so the slots plant (G-DC8) is judged on the line at the same figure.
	planted := memPlanPlantForTest
	memPlanPlantForTest = ""
	defer func() { memPlanPlantForTest = planted }()
	full, _, err := b.deriveMemoryAt(m, *pr.CUDAFree, pool, memCensusSkewForTest)
	if err != nil || full.Decline != "" || full.CacheSlots < 2 {
		return "", "", false
	}
	half := full.CacheSlots / 2
	tooTight := func(p int64) bool {
		pl, _, err := b.deriveMemoryAt(m, p, pool, memCensusSkewForTest)
		return err != nil || pl.Decline != "" || pl.CacheSlots <= half
	}
	lo, hi := int64(0), *pr.CUDAFree // tooTight(lo) holds (nothing fits at 0); tooTight(hi) does not
	for hi-lo > 1 {
		if mid := lo + (hi-lo)/2; tooTight(mid) {
			lo = mid
		} else {
			hi = mid
		}
	}
	at, _, err := b.deriveMemoryAt(m, lo, pool, memCensusSkewForTest)
	if err != nil || at.Decline != "" || at.CacheSlots == 0 {
		return "", "", false
	}
	memPlanPlantForTest = planted
	l, mm := memCensusLine(m, memProfile{Name: pr.Name + memCensusHalfCache, CUDAFree: &lo, MoEPool: pr.MoEPool})
	return strings.ReplaceAll(l, "(profile", "(tight"), mm, true
}

// memCensusBound is G-DM2's bound: the derived readings are within it of the card's (docs/measurements/sb2-derived-2026-10-29/README.md).
const memCensusBound = 64 << 20

// memCensusSkewForTest is added to every derived reading (G-DC2's plant). Tests that set it do not run in parallel.
var memCensusSkewForTest int64

// nearMarks names the post-allocation decisions that differ in lo or hi (the derive at the bound either way) from base: "ctx",
// "kvSlots", "ckpt" (kept or not), "slack", "cacheSlots", "prefill", each compared only where both derives reached it, and
// "refusal@<site>" where one refuses and the other does not, or at another site.
func nearMarks(base, lo, hi memPlan) map[string]bool {
	// reached is how far a derive's decisions go: none after a KV refusal, the slack and before after a slots refusal.
	reached := func(p memPlan) int {
		switch p.DeclineAt {
		case "kv":
			return 0
		case "slots":
			return 1
		}
		return 2
	}
	marks := map[string]bool{}
	for _, o := range []memPlan{lo, hi} {
		if o.DeclineAt != base.DeclineAt {
			site := o.DeclineAt
			if site == "" {
				site = base.DeclineAt
			}
			marks["refusal@"+site] = true
		}
		n := min(reached(base), reached(o))
		for k, differ := range map[string]bool{
			"ctx":        n >= 1 && o.CtxCap != base.CtxCap,
			"kvSlots":    n >= 1 && o.KVSlots != base.KVSlots,
			"ckpt":       n >= 1 && o.CkptBytes != base.CkptBytes,
			"slack":      n >= 1 && o.KVSlack != base.KVSlack,
			"cacheSlots": n >= 2 && o.CacheSlots != base.CacheSlots,
			"prefill":    n >= 2 && o.PrefillWidth != base.PrefillWidth,
		} {
			if differ {
				marks[k] = true
			}
		}
	}
	return marks
}

// derivedTail is a line's post-allocation part: each reading the build takes, as derived, then the decisions (the expert-cache slots
// only where the cache is sized), a ~ on those nearMarks names. A refusal at either end of the bound is written where the line
// ends: decline@<site>~ on a line that refuses, refuses~@<site> on one that builds.
func derivedTail(pl memPlan, rd memReadings, marks map[string]bool) string {
	mark := func(k string) string {
		if marks[k] {
			return "~"
		}
		return ""
	}
	var refusals []string
	for k := range marks {
		if site, ok := strings.CutPrefix(k, "refusal@"); ok {
			refusals = append(refusals, site)
		}
	}
	slices.Sort(refusals)
	end := func(parts []string) string {
		if pl.DeclineAt != "" {
			if len(refusals) > 0 {
				parts = append(parts, "decline@"+pl.DeclineAt+"~ "+pl.Decline)
			} else {
				parts = append(parts, "decline@"+pl.DeclineAt+" "+pl.Decline)
			}
			return strings.Join(parts, " ")
		}
		for _, site := range refusals {
			parts = append(parts, "refuses~@"+site)
		}
		return strings.Join(parts, " ")
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
		return end(parts)
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
		return end(parts)
	}
	if rd.SlotsOK {
		parts = append(parts, fmt.Sprintf("cacheSlots=%d%s", pl.CacheSlots, mark("cacheSlots")))
	}
	return end(append(parts, fmt.Sprintf("prefill=%d%s", pl.PrefillWidth, mark("prefill"))))
}

// buildCUDAMemCensus returns one line per cell layer B-cuda plans and profile.
func buildCUDAMemCensus(t *testing.T) []string { return buildCUDAMemCensusCells(t, true) }

// buildCUDAMemCensusCells is buildCUDAMemCensus, with the census's own cells (memCensusCells) only when own is set.
func buildCUDAMemCensusCells(t *testing.T, own bool) []string {
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
	fixtures := d.Fixtures
	if own {
		fixtures = append(slices.Clone(fixtures), memCensusWindowVariant, memCensusExpertVariant)
	}
	for _, f := range fixtures {
		variant := f == memCensusWindowVariant || f == memCensusExpertVariant
		var path string
		switch {
		case f == memCensusWindowVariant:
			path = windowVariantFixture(t, "../testdata/llama-tiny", 32768)
		case f == memCensusExpertVariant:
			path = expertVariantFixture(t, "../testdata/glm-tiny", 16384)
		default:
			if path, err = sb2census.Path("..", f); err != nil {
				t.Fatal(err)
			}
		}
		// plan loads one cell and returns its lines, one per profile, without the static declines (layer B-cuda's lines).
		plan := func(name string, o decoder.Options) []string {
			m, err := decoder.Load(path, o)
			if err == nil {
				m.Close()
			}
			be.mu.Lock()
			reached, ls, mms := be.reached, be.lines, be.mismatches
			be.reached, be.lines, be.mismatches = false, nil, nil
			be.mu.Unlock()
			if !reached {
				return nil
			}
			for _, mm := range mms {
				t.Errorf("G-DC1: %s %s: %s", f, name, mm)
			}
			var out []string
			for _, l := range ls {
				if !strings.Contains(l, " static ") {
					out = append(out, sb2census.Clean(l))
				}
			}
			return out
		}
		var int4 []string
		for _, c := range sb2census.Cells {
			o := decoder.Options{Backend: reg, Quant: c.Quant, ActQuantGroup: c.ActGroup, KVPrecision: c.KVPrec, KVQuant: c.KVQuant}
			if c.SSM != "" {
				o.Knobs = &decoder.Knobs{"GOINFER_SSM_RESIDENT": c.SSM}
			}
			if variant && c.Name != "int4" {
				continue // the variant carries only the census's own cells, compared with its int4 line
			}
			ls := plan(c.Name, o)
			if c.Name == "int4" {
				int4 = ls
			}
			if variant {
				continue
			}
			for _, l := range ls {
				lines = append(lines, fmt.Sprintf("M %s %s cuda %s", f, c.Name, l))
			}
		}
		// The census's own cells: an int4 load with one more option, kept where it changes the fixture's int4 line.
		for _, c := range memCensusCells {
			if !own {
				break
			}
			o := c.o
			o.Backend, o.Quant = reg, "int4"
			ls := plan(c.name, o)
			for i, l := range ls {
				if i < len(int4) && l == int4[i] {
					continue
				}
				lines = append(lines, fmt.Sprintf("X %s %s cuda %s", f, c.name, l))
			}
		}
	}
	return lines
}

// memCensusCells are the memory census's own cells, beyond layer B-cuda's: the options that reach the grant, the expert-cache slots
// and the raise of one conversation's cap, which no layer B-cuda cell sets.
var memCensusCells = []struct {
	name string
	o    decoder.Options
}{
	{"int4/windowed", decoder.Options{ResidentWindowedKV: true}},
	{"int4/expert-cache", decoder.Options{MoECacheExperts: true}},
	{"int4/slots4", decoder.Options{ResidentKVSlots: 4}},
	{"int4/windowed+expert-cache", decoder.Options{ResidentWindowedKV: true, MoECacheExperts: true}},
}

// memCensusWindowVariant names the census's own fixture: llama-tiny with a 32768-position window (windowVariantFixture), so that with
// several slots one conversation's cap can grow past the per-slot context, which no committed non-recurrent fixture allows.
const memCensusWindowVariant = "testdata/llama-tiny@window32768"

// memCensusExpertVariant names the census's wide-expert fixture: glm-tiny with 16384-wide experts and one MoE layer
// (expertVariantFixture), whose slot buffers cross the allocation quantum, so a figure can cap its expert cache partway.
const memCensusExpertVariant = "testdata/glm-tiny@experts16384"

// expertVariantLayers is the wide-expert variant's layer count: glm-tiny's dense layer 0 and its first MoE layer.
const expertVariantLayers = 2

// expertVariantFixture writes, under t's temporary directory, the glm4_moe fixture src with moe_intermediate_size set to inter and
// num_hidden_layers to expertVariantLayers: the later layers' tensors are dropped, the routed and shared experts' tensors are
// regenerated at the new width from a fixed stream in ±0.02, and every other tensor is copied. Other files are links to src's.
func expertVariantFixture(t *testing.T, src string, inter int) string {
	t.Helper()
	abs, err := filepath.Abs(src)
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), filepath.Base(src)+"-experts")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	ents, err := os.ReadDir(abs)
	if err != nil {
		t.Fatal(err)
	}
	var old int
	for _, e := range ents {
		from, to := filepath.Join(abs, e.Name()), filepath.Join(dir, e.Name())
		switch e.Name() {
		case "config.json":
			b, err := os.ReadFile(from)
			if err != nil {
				t.Fatal(err)
			}
			var cfg map[string]any
			if err := json.Unmarshal(b, &cfg); err != nil {
				t.Fatal(err)
			}
			old = int(cfg["moe_intermediate_size"].(float64))
			cfg["moe_intermediate_size"], cfg["num_hidden_layers"] = inter, expertVariantLayers
			out, _ := json.MarshalIndent(cfg, "", "  ")
			if err := os.WriteFile(to, out, 0o644); err != nil {
				t.Fatal(err)
			}
		case "model.safetensors":
		default:
			if err := os.Symlink(from, to); err != nil {
				t.Fatal(err)
			}
		}
	}
	if old == 0 {
		t.Fatalf("%s: no moe_intermediate_size", src)
	}
	raw, err := os.ReadFile(filepath.Join(abs, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	type tensor struct {
		Dtype   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	hlen := int(binary.LittleEndian.Uint64(raw[:8]))
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+hlen], &hdr); err != nil {
		t.Fatal(err)
	}
	data := raw[8+hlen:]
	names := make([]string, 0, len(hdr))
	layerOf := regexp.MustCompile(`^model\.layers\.(\d+)\.`)
	for k := range hdr {
		if k == "__metadata__" {
			continue
		}
		if m := layerOf.FindStringSubmatch(k); m != nil {
			if i, _ := strconv.Atoi(m[1]); i >= expertVariantLayers {
				continue
			}
		}
		names = append(names, k)
	}
	sort.Strings(names)
	out := map[string]tensor{}
	var body []byte
	seed := uint32(1)
	for _, k := range names {
		var tn tensor
		if err := json.Unmarshal(hdr[k], &tn); err != nil {
			t.Fatal(err)
		}
		if tn.Dtype != "F32" {
			t.Fatalf("%s: %s is %s; the variant writes f32 only", src, k, tn.Dtype)
		}
		start := len(body)
		if strings.Contains(k, ".mlp.experts.") || strings.Contains(k, ".mlp.shared_experts.") {
			// An expert's gate and up projections are [inter, hidden]; its down projection is [hidden, inter].
			dim := 0
			if strings.HasSuffix(k, ".down_proj.weight") {
				dim = 1
			} else if !strings.HasSuffix(k, ".gate_proj.weight") && !strings.HasSuffix(k, ".up_proj.weight") {
				t.Fatalf("%s: %s is an expert tensor the variant does not know how to widen", src, k)
			}
			if len(tn.Shape) != 2 || tn.Shape[dim] != old {
				t.Fatalf("%s: %s has shape %v, want dimension %d to be moe_intermediate_size %d", src, k, tn.Shape, dim, old)
			}
			tn.Shape[dim] = inter
			n := tn.Shape[0] * tn.Shape[1]
			buf := make([]byte, 4*n)
			for i := range n {
				seed = seed*1664525 + 1013904223
				binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(0.02*float32(int32(seed>>8)%2000-1000)/1000))
			}
			body = append(body, buf...)
		} else {
			body = append(body, data[tn.Offsets[0]:tn.Offsets[1]]...)
		}
		tn.Offsets = [2]int{start, len(body)}
		out[k] = tn
	}
	h, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	for len(h)%8 != 0 {
		h = append(h, ' ')
	}
	f := make([]byte, 8, 8+len(h)+len(body))
	binary.LittleEndian.PutUint64(f, uint64(len(h)))
	f = append(append(f, h...), body...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), f, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// memCensusHalfCache suffixes the profile of a line derived at a tight figure: the largest at which the cache holds at most half of
// the profile line's slots (memCensusHalfCacheLine).
const memCensusHalfCache = "@halfcache"

// windowVariantFixture writes, under t's temporary directory, fixture src with max_position_embeddings set to maxPos; every other file
// is a link to src's.
func windowVariantFixture(t *testing.T, src string, maxPos int) string {
	t.Helper()
	abs, err := filepath.Abs(src)
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), filepath.Base(src))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	ents, err := os.ReadDir(abs)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		from, to := filepath.Join(abs, e.Name()), filepath.Join(dir, e.Name())
		if e.Name() != "config.json" {
			if err := os.Symlink(from, to); err != nil {
				t.Fatal(err)
			}
			continue
		}
		b, err := os.ReadFile(from)
		if err != nil {
			t.Fatal(err)
		}
		var cfg map[string]any
		if err := json.Unmarshal(b, &cfg); err != nil {
			t.Fatal(err)
		}
		cfg["max_position_embeddings"] = maxPos
		out, _ := json.MarshalIndent(cfg, "", "  ")
		if err := os.WriteFile(to, out, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
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
	base, own := 0, map[string]int{}
	for _, l := range a {
		if strings.HasPrefix(l, "M ") {
			base++
			continue
		}
		for _, site := range []string{"grant", "grantPre"} {
			if strings.Contains(l, " "+site+"@") && memCensusSlack(l) > kvWindowSlack {
				own[site]++ // a grant made at this site, past the window's floor
			}
		}
		if strings.Contains(l, " cacheSlots=") {
			own["slots"]++
		}
	}
	if want := planned * len(memProfiles(t)); base != want {
		t.Errorf("%d memory census lines from layer B-cuda's cells, want %d (%d planned cells x %d profiles)", base, want, planned, len(memProfiles(t)))
	}
	for _, site := range []string{"grant", "grantPre", "slots"} {
		if own[site] == 0 {
			t.Errorf("G-DC5: no line of the census's own cells reaches the %s site", site)
		}
	}
	// The window variant (memCensusWindowVariant) is what reaches the raise, and the wide-expert variant (memCensusExpertVariant) the
	// slot cap: a committed tiny fixture's slot buffers are far under one allocation quantum, so one slot costs what all of them do.
	if own["raise"] = memCensusRaises(t, a); own["raise"] == 0 {
		t.Errorf("G-DC7: no int4/slots4 line raises one conversation's cap above the plan's context")
	}
	if own["halfcache"] = memCensusHalfCaps(t, a); own["halfcache"] == 0 {
		t.Errorf("G-DC7: no @halfcache line: no figure caps any expert cache partway")
	}
	t.Logf("the census's own cells: %d lines; %d reach a grant, %d a grant before the cache, %d the expert-cache slots, %d raise one conversation's cap, %d cap the cache",
		len(a)-base, own["grant"], own["grantPre"], own["slots"], own["raise"], own["halfcache"])
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
	clean := buildCUDAMemCensusCells(t, false)
	defer func(site string) { weightsMarginPlantSiteForTest = site }(weightsMarginPlantSiteForTest)
	weightsMarginPlantSiteForTest = "weights"
	planted := buildCUDAMemCensusCells(t, false)
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
// every line that reaches the KV fit must refuse there.
func TestBackendMemoryCensus_cuda_derivedPlant(t *testing.T) {
	clean := buildCUDAMemCensusCells(t, false)
	defer func(v int64) { memCensusSkewForTest = v }(memCensusSkewForTest)
	memCensusSkewForTest = -8 << 30
	planted := buildCUDAMemCensusCells(t, false)
	if len(planted) != len(clean) {
		t.Fatalf("planted skew: %d lines, want %d", len(planted), len(clean))
	}
	reached := 0
	for i := range clean {
		if !strings.Contains(clean[i], " derived(") {
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
	t.Logf("planted skew: all %d lines that reach the KV fit refuse there", reached)
}

// TestNearMarks is G-DC3: a decision is marked when it differs at either end of the bound, and only then; a refusal at either end is
// its own mark, and a refused derive's empty fields mark nothing (the shapes deriveMemoryAt returns: a KV refusal has no later
// decisions, a slots refusal none after the slack).
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
		{"kv refusal", func(p *memPlan) { *p = memPlan{Decline: "no room", DeclineAt: "kv"} }, "refusal@kv"},
		{"slots refusal", func(p *memPlan) { p.Decline, p.DeclineAt, p.CacheSlots, p.PrefillWidth = "no room", "slots", 0, 0 }, "refusal@slots"},
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
	refused := memPlan{Decline: "no room", DeclineAt: "kv"}
	if got := nearMarks(refused, refused, base); len(got) != 1 || !got["refusal@kv"] {
		t.Errorf("a refusal that builds 64 MiB higher: marks %v, want only refusal@kv", got)
	}
}

// TestDerivedTail_nearRefusal runs the census's marks on a real derive at the edge of the KV fit: llama-tiny skewed to the smallest
// reading at which it still builds, so the derive 64 MiB lower refuses. The line must say so, and mark no decision for differing
// from the refused derive's empty fields; a decision is marked only where the derive 64 MiB higher differs.
func TestDerivedTail_nearRefusal(t *testing.T) {
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const p = int64(7 << 30)
	b := &cudaBackend{}
	derive := func(skew int64) (memPlan, memReadings) {
		t.Helper()
		pl, rd, err := b.deriveMemoryAt(m, p, 0, skew)
		if err != nil {
			t.Fatal(err)
		}
		return pl, rd
	}
	refuses := func(skew int64) bool { pl, _ := derive(skew); return pl.DeclineAt == "kv" }
	lo, hi := -p, int64(0) // refuses at lo, builds at hi
	if !refuses(lo) || refuses(hi) {
		t.Fatalf("llama-tiny: refuses at skew %d %v, at 0 %v; want a refusal at the bottom and a build at 0", lo, refuses(lo), refuses(hi))
	}
	for hi-lo > 1<<20 {
		if mid := lo + (hi-lo)/2; refuses(mid) {
			lo = mid
		} else {
			hi = mid
		}
	}
	base, rd := derive(hi)
	down, _ := derive(hi - memCensusBound)
	up, _ := derive(hi + memCensusBound)
	if base.DeclineAt != "" || down.DeclineAt != "kv" {
		t.Fatalf("at skew %d: base refuses at %q, 64 MiB lower at %q; want a build over a KV refusal", hi, base.DeclineAt, down.DeclineAt)
	}
	marks := nearMarks(base, down, up)
	for k, differ := range map[string]bool{"ctx": up.CtxCap != base.CtxCap, "kvSlots": up.KVSlots != base.KVSlots, "ckpt": up.CkptBytes != base.CkptBytes,
		"slack": up.KVSlack != base.KVSlack, "prefill": up.PrefillWidth != base.PrefillWidth} {
		if marks[k] != differ {
			t.Errorf("%s marked %v, but the derive 64 MiB higher differs: %v (a refused derive marks nothing)", k, marks[k], differ)
		}
	}
	if tail := derivedTail(base, rd, marks); !marks["refusal@kv"] || !strings.Contains(tail, " refuses~@kv") {
		t.Errorf("a line 64 MiB from refusing at the KV fit: marks %v, tail %q; want refuses~@kv", marks, tail)
	}
}

// memCensusSlack is a census line's windowed slack, or -1 when it has none.
func memCensusSlack(l string) int {
	m := regexp.MustCompile(` slack=(\d+)`).FindStringSubmatch(l)
	if m == nil {
		return -1
	}
	n, _ := strconv.Atoi(m[1])
	return n
}

// memCensusKey is a census line's identity: its layer, fixture, cell, backend and profile.
func memCensusKey(l string) string {
	w := strings.Fields(l)
	return strings.Join(w[:min(len(w), 5)], " ")
}

// memCensusRaises counts the int4/slots4 lines of the census's own cells whose derived cap is above the context the plan chose for a
// slot: one conversation's cap raised to the pool (kvCapFor).
// memCensusHalfCaps counts the @halfcache lines (memCensusHalfCache) and checks each one: its derived cache holds at most half the slots
// of the line of the same fixture, cell and profile at the profile's own figure.
func memCensusHalfCaps(t *testing.T, lines []string) int {
	t.Helper()
	slots := func(l string) (int, bool) {
		m := regexp.MustCompile(` cacheSlots=(\d+)`).FindStringSubmatch(l)
		if m == nil {
			return 0, false
		}
		n, _ := strconv.Atoi(m[1])
		return n, true
	}
	byKey := map[string]string{}
	for _, l := range lines {
		byKey[memCensusKey(l)] = l
	}
	n := 0
	for _, l := range lines {
		k := memCensusKey(l)
		if !strings.HasSuffix(k, memCensusHalfCache) {
			continue
		}
		n++
		prof, ok := byKey[strings.TrimSuffix(k, memCensusHalfCache)]
		if !ok {
			t.Errorf("a @halfcache line has no profile line: %s", l)
			continue
		}
		h, okH := slots(l)
		f, okF := slots(prof)
		if !okH || !okF || h > f/2 {
			t.Errorf("a @halfcache line does not hold at most half its profile line's slots (%d of %d):\n%s\n%s", h, f, l, prof)
		}
	}
	return n
}

func memCensusRaises(t *testing.T, lines []string) int {
	t.Helper()
	field := func(l, name string) int {
		m := regexp.MustCompile(` ` + name + `=(\d+)`).FindStringSubmatch(l)
		if m == nil {
			return 0
		}
		n, _ := strconv.Atoi(m[1])
		return n
	}
	n := 0
	for _, l := range lines {
		if strings.HasPrefix(l, "X ") && strings.Fields(l)[2] == "int4/slots4" && field(l, "ctx") > field(l, "ctxPlanned") && field(l, "ctxPlanned") > 0 {
			n++
		}
	}
	return n
}

// TestBackendMemoryCensus_cuda_sitePlants is G-DC6: each grant site's plant changes exactly the lines of the census's own cells that make
// a grant there, and no line of layer B-cuda's cells; the slots plant (the cap ignored) changes exactly the @halfcache lines (G-DC8).
// Lines are compared by identity (memCensusKey): a line of the census's own cells is kept only where it differs from its int4 line, so
// a plant can remove one.
func TestBackendMemoryCensus_cuda_sitePlants(t *testing.T) {
	index := func(lines []string) map[string]string {
		m := map[string]string{}
		for _, l := range lines {
			m[memCensusKey(l)] = l
		}
		return m
	}
	clean := index(buildCUDAMemCensus(t))
	defer func(p string) { memPlanPlantForTest = p }(memPlanPlantForTest)
	for _, site := range []string{"grant", "grantPre", "slots"} {
		memPlanPlantForTest = site
		planted := index(buildCUDAMemCensus(t))
		reads := " " + site + "@"
		for k := range planted {
			if _, ok := clean[k]; !ok {
				t.Errorf("plant %s added a line: %s", site, planted[k])
			}
		}
		changed, granted := 0, 0
		for k, c := range clean {
			grants := strings.HasPrefix(c, "X ") && strings.Contains(c, reads) && memCensusSlack(c) > kvWindowSlack
			if site == "slots" {
				grants = strings.HasSuffix(k, memCensusHalfCache) // the cap decides only where the figure is too tight for the whole cache
			}
			if grants {
				granted++
			}
			if _, ok := planted[k]; !ok && site == "slots" && grants {
				t.Errorf("plant slots dropped a @halfcache line, so the line at its figure was never derived: %s", c)
			}
			if p, ok := planted[k]; ok && p == c {
				if grants {
					t.Errorf("plant %s left a line that makes a grant there unchanged: %s", site, c)
				}
				continue
			}
			changed++
			if !grants {
				t.Errorf("plant %s changed a line that makes no grant there: %s", site, c)
			}
		}
		if granted == 0 {
			t.Errorf("plant %s: no line makes a grant there: the site is not censused", site)
		}
		t.Logf("plant %s: %d lines changed, the %d that make a grant there", site, changed, granted)
	}
}

// TestResident_holdsTheModelsProgram is OP4cu-a's A-1 (docs/tasks/task-forward-architecture-2026-10.md §4.9a): a CUDA resident
// holds the model's own program, the pointer decoder.Model.Program returns, with one operation list per layer. Both fixtures
// are tracked, so a missing one fails. It lives in this file, which already imports decoder, so the decoder-import ratchet does
// not move.
func TestResident_holdsTheModelsProgram(t *testing.T) {
	for _, c := range []struct {
		dir         string
		layers, ops int // the fixture's program, read at OP4cu-a (docs/measurements/op4-cuda-2026-11/op4cu-a-a1-green.log)
	}{{"../testdata/llama-tiny", 4, 36}, {"../testdata/qwen35-tiny", 4, 24}} {
		dir := c.dir
		t.Run(filepath.Base(dir), func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
				t.Fatalf("no fixture (it is tracked): %v", err)
			}
			requireDeviceAndFixture(t, dir)
			m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatalf("not a CUDA resident: %s", m.ResidentDecline())
			}
			want := m.Program()
			if want == nil {
				t.Fatal("the model has no program")
			}
			if r.prog != want {
				t.Fatalf("the resident holds %p, not the model's program %p", r.prog, want)
			}
			ops := 0
			for l, ls := range r.prog.Layers {
				if len(ls) == 0 {
					t.Errorf("layer %d of the resident's program has no operations", l)
				}
				ops += len(ls)
			}
			if len(r.prog.Layers) != len(m.Weights().Layers) || len(r.prog.Layers) != c.layers || ops != c.ops {
				t.Fatalf("the resident's program has %d layers and %d operations, for a %d-layer model; want %d and %d",
					len(r.prog.Layers), ops, len(m.Weights().Layers), c.layers, c.ops)
			}
			t.Logf("%d layers, %d operations", len(r.prog.Layers), ops)
		})
	}
}

// walkFixtures are the tracked tiny checkpoints B2-1 builds on CUDA: every directory under the two testdata roots with a
// config.json and safetensors weights under 1 GB.
func walkFixtures(t *testing.T) []string {
	t.Helper()
	var dirs []string
	for _, root := range []string{"../testdata", "../decoder/testdata"} {
		ents, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, e := range ents {
			d := filepath.Join(root, e.Name())
			ws, _ := filepath.Glob(filepath.Join(d, "*.safetensors"))
			var n int64
			for _, w := range ws {
				if fi, err := os.Stat(w); err == nil {
					n += fi.Size()
				}
			}
			if _, err := os.Stat(filepath.Join(d, "config.json")); e.IsDir() && err == nil && len(ws) > 0 && n < 1<<30 {
				dirs = append(dirs, d)
			}
		}
	}
	return dirs
}

// walkDecode decodes steps greedy tokens from token 3 on a fresh CUDA resident of dir at int4 and returns every position's
// logits, or nil when the backend declines the fixture.
func walkDecode(t *testing.T, dir string, steps int) [][]float32 {
	t.Helper()
	m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		return nil
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		return nil
	}
	if walkPlant != walkPlantNone && rf.graphs {
		// The walk plants act in runStep's live path; a captured segment replays without passing them (captureGraphs does not
		// run through runStep), so the plants run with graphs off, as this box builds them (admitGraphs declines here).
		t.Fatalf("%s: a walk plant needs CUDA graphs off, and this build captured them", dir)
	}
	_, _, _, _, _, _, vocab := m.Dims()
	tok, out := 3%vocab, make([][]float32, 0, steps)
	for pos := range steps {
		l, err := rf.Forward(m.EmbedResidentForTest(tok), pos)
		if err != nil {
			t.Fatalf("%s: position %d: %v", dir, pos, err)
		}
		out = append(out, slices.Clone(l))
		best := 0
		for i, v := range l {
			if v > l[best] {
				best = i
			}
		}
		tok = best
	}
	return out
}

// walkNamedDeclines are the fixtures the walk's build declines by a registered, named outcome change, which the pre-walk build
// accepted. It is empty: Olmo Hybrid, declined by OP4cu-c1 until program.CUDA claimed its DeltaNet's two settings, builds again
// (docs/tasks/task-olmo-hybrid-cuda-2026-11.md).
var walkNamedDeclines = map[string]bool{}

// walkEqualsLegacy compares the walk with the pre-walk loop on every fixture and returns how many fixtures ran and how many
// differed, naming each difference.
func walkEqualsLegacy(t *testing.T, report func(format string, args ...any)) (ran, differ int) {
	const steps = 24
	for _, dir := range walkFixtures(t) {
		walk := walkDecode(t, dir, steps)
		var old [][]float32
		func() { // the pre-walk loop, on a build that skips the walk's build checks, as the pre-walk build did
			legacyLayerForTest = legacyLayer
			defer func() { legacyLayerForTest = nil }()
			old = walkDecode(t, dir, steps)
		}()
		switch {
		case walk == nil && old != nil && walkNamedDeclines[filepath.Base(dir)]:
			t.Logf("NOT RUN: %s: declined by the walk's build, a named outcome change (OP4cu-c1: program.CUDA)", filepath.Base(dir))
			continue
		case walk == nil && old != nil:
			report("%s: the walk's build declined a fixture the pre-walk build accepts", filepath.Base(dir))
			differ++
			continue
		case walk == nil:
			t.Logf("NOT RUN: %s: not a CUDA resident at int4", filepath.Base(dir))
			continue
		}
		ran++
		for pos := range walk {
			if !slices.Equal(walk[pos], old[pos]) {
				report("%s: position %d: the walk's logits differ from the pre-walk loop's", filepath.Base(dir), pos)
				differ++
				break
			}
		}
	}
	return ran, differ
}

// TestWalk_equalsLegacy is OP4cu-b2's B2-1 (docs/tasks/task-forward-architecture-2026-10.md §4.9a): on every tracked tiny
// fixture that builds a CUDA resident at int4, 24 greedy decode positions on the walk equal the pre-walk loop's bit for bit.
func TestWalk_equalsLegacy(t *testing.T) {
	requireDeviceAndFixture(t, "../testdata/llama-tiny")
	ran, differ := walkEqualsLegacy(t, t.Errorf)
	if ran < 10 {
		t.Fatalf("only %d fixtures ran on CUDA: a comparison over few fixtures proves little", ran)
	}
	t.Logf("%d fixtures compared, %d differ", ran, differ)
}

// walkSelfTestUnplanted makes the process's first CUDA load before any plant is set. The decoder's startup self-test runs once,
// on that load, and a planted walk fails it and declines the backend for the whole process, so nothing would build resident
// and a plant test would compare no fixture.
func walkSelfTestUnplanted(t *testing.T) {
	t.Helper()
	if walkDecode(t, "../testdata/llama-tiny", 1) == nil {
		t.Fatal("llama-tiny is not a CUDA resident unplanted")
	}
}

// TestWalk_plantsGoRed is OP4cu-b2's B2-2: a dropped step, two swapped steps and a parameter off by one each make the walk
// differ from the pre-walk loop on at least one fixture.
func TestWalk_plantsGoRed(t *testing.T) {
	requireDeviceAndFixture(t, "../testdata/llama-tiny")
	walkSelfTestUnplanted(t)
	for _, c := range []struct {
		name  string
		plant walkPlantKind
	}{{"attnGap dropped", walkPlantDropGap}, {"outFFN before attnGap", walkPlantSwap}, {"outFFN reads qDim-1", walkPlantQDimMinus1}} {
		t.Run(c.name, func(t *testing.T) {
			setWalkPlantForTest(t, c.plant)
			ran, differ := walkEqualsLegacy(t, t.Logf)
			t.Logf("plant %q: %d of %d fixtures differ", c.name, differ, ran)
			if differ == 0 {
				t.Errorf("plant %q: no fixture differs", c.name)
			}
		})
	}
}

// TestCudaStepsOf_patterns is OP4cu-b2's table check: each sublayer pattern a CUDA resident runs maps to its segment steps, a
// pattern with no entry and a sublayer that claims one operation too many (B2-2's fourth plant) are errors.
func TestCudaStepsOf_patterns(t *testing.T) {
	norm := func(slot program.NormSlot) program.Op {
		return program.Op{Kind: program.OpNorm, Norm: program.NormParams{Slot: slot, Form: program.NormRMS}}
	}
	ffn := func(f program.FFNForm) program.Op {
		return program.Op{Kind: program.OpFFN, FFN: program.FFNParams{Form: f}}
	}
	res := program.Op{Kind: program.OpResidual}
	attn := []program.Op{norm(program.SlotPreAttn), {Kind: program.OpProjQKV}, {Kind: program.OpRope}, {Kind: program.OpAttention}, {Kind: program.OpProjO}, res}
	mixer := func(k program.OpKind) []program.Op { return []program.Op{norm(program.SlotPreAttn), {Kind: k}, res} }
	ffnSub := func(f program.FFNForm) []program.Op { return []program.Op{norm(program.SlotPreMLP), ffn(f), res} }
	for _, c := range []struct {
		name string
		ops  []program.Op
		want []cudaStep
	}{
		{"attention and a dense FFN", slices.Concat(attn, ffnSub(program.FfnGated)), attnSteps},
		{"attention and a MoE FFN", slices.Concat(attn, ffnSub(program.FfnMoE)), attnSteps},
		{"Nemotron-H's attention block", attn, attnSteps},
		{"the E-model: a dense FFN, its PLE and its layer scale", slices.Concat(attn, ffnSub(program.FfnGemma4Dense), []program.Op{{Kind: program.OpPLE}, {Kind: program.OpLayerScale}}), attnSteps},
		{"MLA and a MoE FFN", slices.Concat(mixer(program.OpMLA), ffnSub(program.FfnMoE)), mlaSteps},
		{"DeltaNet and a dense FFN", slices.Concat(mixer(program.OpDeltaNet), ffnSub(program.FfnGated)), deltaNetSteps},
		{"Nemotron-H's Mamba-2 block", mixer(program.OpMamba2), mambaSteps},
		{"Nemotron-H's MLP block", []program.Op{norm(program.SlotPreAttn), ffn(program.FfnNonGated), res}, ffnOnlySteps},
		{"attention and Gemma 4's MoE block", slices.Concat(attn, []program.Op{{Kind: program.OpGemma4MoE}}), attnSteps},
		{"Gemma 4's dense layer and its layer scale", slices.Concat(attn, ffnSub(program.FfnGemma4Dense), []program.Op{{Kind: program.OpLayerScale}}), attnSteps},
		{"MLA and a dense FFN", slices.Concat(mixer(program.OpMLA), ffnSub(program.FfnGated)), mlaSteps},
		{"DeltaNet and a MoE FFN", slices.Concat(mixer(program.OpDeltaNet), ffnSub(program.FfnMoE)), deltaNetSteps},
		{"a parallel block", slices.Concat(attn[:5], []program.Op{ffn(program.FfnGated), {Kind: program.OpResidual, Res: program.ResidualParams{Two: true}}}), attnSteps},
	} {
		if got, err := cudaStepsOf(c.ops); err != nil || !slices.Equal(got, c.want) {
			t.Errorf("%s: %v, %v; want %v", c.name, got, err, c.want)
		}
	}
	// CUDA runs PLE and the layer scale only at the end of a dense FFN (segBFFN): any other place, or a second one, is refused.
	ple, ls := program.Op{Kind: program.OpPLE}, program.Op{Kind: program.OpLayerScale}
	for _, c := range []struct {
		name string
		ops  []program.Op
	}{
		{"a Mamba-2 mixer and a dense FFN (Granite-4.0-H's layer)", slices.Concat(mixer(program.OpMamba2), ffnSub(program.FfnGated))},
		{"a Mamba-2 block and a PLE", slices.Concat(mixer(program.OpMamba2), []program.Op{ple})},
		{"attention and a PLE, no FFN", slices.Concat(attn, []program.Op{ple})},
		{"a PLE before the attention", slices.Concat([]program.Op{ple}, attn, ffnSub(program.FfnGemma4Dense))},
		{"two PLEs", slices.Concat(attn, ffnSub(program.FfnGemma4Dense), []program.Op{ple, ple, ls})},
		{"a layer scale after a MoE FFN", slices.Concat(attn, ffnSub(program.FfnMoE), []program.Op{ls})},
		{"a PLE after the layer scale", slices.Concat(attn, ffnSub(program.FfnGemma4Dense), []program.Op{ls, ple})},
		{"a layer scale after Gemma 4's MoE block", slices.Concat(attn, []program.Op{{Kind: program.OpGemma4MoE}, ls})},
		// #588's second review, point 2: a parallel block followed by a layer scale or a PLE is not Gemma 4's sequential layer.
		{"a parallel block and a layer scale", slices.Concat(attn[:5], []program.Op{ffn(program.FfnGemma4Dense), {Kind: program.OpResidual, Res: program.ResidualParams{Two: true}}, ls})},
		{"a parallel block, a PLE and a layer scale", slices.Concat(attn[:5], []program.Op{ffn(program.FfnGemma4Dense), {Kind: program.OpResidual, Res: program.ResidualParams{Two: true}}, ple, ls})},
	} {
		if got, err := cudaStepsOf(c.ops); err == nil {
			t.Errorf("%s: %v accepted, want a refusal", c.name, got)
		}
	}
	t.Run("claim one more", func(t *testing.T) {
		program.SetShapePlantForTest(t, program.ShapePlantClaimOneMore)
		if got, err := cudaStepsOf(slices.Concat(attn, ffnSub(program.FfnGated))); err == nil {
			t.Errorf("the attention sublayer claiming the FFN's input norm: %v accepted, want a decline", got)
		}
	})
}

// TestWalk_claimPlantDeclinesTheBuild is B2-2's fourth plant on the card: with the attention sublayer claiming one operation too
// many, the CUDA build declines llama-tiny, naming the layer.
func TestWalk_claimPlantDeclinesTheBuild(t *testing.T) {
	requireDeviceAndFixture(t, "../testdata/llama-tiny")
	walkSelfTestUnplanted(t)
	program.SetShapePlantForTest(t, program.ShapePlantClaimOneMore)
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, ok := m.ResidentForwardForTest().(*cudaResident); ok {
		t.Fatal("the planted build went resident")
	}
	if d := m.ResidentDecline(); !strings.Contains(d, "layer 0") {
		t.Fatalf("the decline does not name the layer: %q", d)
	} else {
		t.Logf("declined: %s", d)
	}
}

// b26Cell is one B2-6 cell's reading: the pairs' walker-over-old rate ratios, and the box's state at the cell's start.
type b26Cell struct {
	name       string
	ratios     []float64
	load       float64
	availMB    int64
	void       bool
	voidReason string
}

// b26Load and b26AvailMB read the box's 1-minute load and MemAvailable, the void rule's two terms.
func b26Load() float64 {
	b, _ := os.ReadFile("/proc/loadavg")
	f, _ := strconv.ParseFloat(strings.Fields(string(b) + " 0")[0], 64)
	return f
}

func b26AvailMB() int64 {
	b, _ := os.ReadFile("/proc/meminfo")
	for _, ln := range strings.Split(string(b), "\n") {
		if f := strings.Fields(ln); len(f) >= 2 && f[0] == "MemAvailable:" {
			kb, _ := strconv.ParseInt(f[1], 10, 64)
			return kb >> 10
		}
	}
	return 0
}

// TestWalkSpeed_B26 is OP4cu-b2's B2-6 (docs/tasks/task-forward-architecture-2026-10.md §4.9a): a whole-token in-process A/B of the
// walk against the pre-walk loop, interleaved ABBA on one loaded model, 6 pairs per cell: decode at depth 128 (64 tokens timed
// from position 128) and prefill at 3,900 tokens, on the 0.5B and the 1.5B. The quantity is walk / old, as rates. It runs under the
// exclusive timing lock, set OP4CU_B26=1.
func TestWalkSpeed_B26(t *testing.T) {
	if os.Getenv("OP4CU_B26") == "" {
		t.Skip("set OP4CU_B26=1 under the exclusive timing lock")
	}
	home, _ := os.UserHomeDir()
	const pairs, depth, decodeN, prefillN = 6, 128, 64, 3900
	var cells []b26Cell
	for _, mc := range []struct{ file, label string }{
		{"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "0.5B"}, {"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "1.5B"},
	} {
		mname := mc.file
		path := filepath.Join(home, "models", mname)
		requireDeviceAndFixtureFile(t, path)
		fi, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		needMB := fi.Size()>>20 + 2048 // the fit need, taken as the checkpoint's size, plus the void rule's 2 GB
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda"})
		if err != nil {
			t.Fatal(err)
		}
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			m.Close()
			t.Fatalf("%s: not a CUDA resident: %s", mname, m.ResidentDecline())
		}
		embs := make([][]float32, prefillN)
		for i := range embs {
			embs[i] = m.EmbedResidentForTest(100 + i%500)
		}
		decodeRate := func() float64 { // tokens per second over decodeN tokens from position depth
			start := time.Now()
			for i := range decodeN {
				if _, err := r.Forward(embs[depth+i], depth+i); err != nil {
					t.Fatal(err)
				}
			}
			return float64(decodeN) / time.Since(start).Seconds()
		}
		prefillRate := func() float64 { // prompt tokens per second
			start := time.Now()
			if _, err := r.PrefillLast(context.Background(), embs, 0); err != nil {
				t.Fatal(err)
			}
			return float64(prefillN) / time.Since(start).Seconds()
		}
		if _, err := r.PrefillLast(context.Background(), embs[:depth], 0); err != nil { // the decode cells' context
			t.Fatal(err)
		}
		_, _ = decodeRate(), decodeRate() // warm both arms' kernels
		legacyLayerForTest = legacyLayer
		_ = decodeRate()
		legacyLayerForTest = nil
		for _, c := range []struct {
			name string
			rate func() float64
		}{{"decode@128", decodeRate}, {"prefill@3900", prefillRate}} {
			cell := b26Cell{name: mc.label + " " + c.name, load: b26Load(), availMB: b26AvailMB()}
			// Heavy load voids the cell: the 1-minute load over half of nobara's 16 cores, or less available memory than the
			// checkpoint's size plus 2 GB.
			if cell.load > 8.0 || cell.availMB < needMB {
				cell.void, cell.voidReason = true, fmt.Sprintf("load %.2f, %d MB available", cell.load, cell.availMB)
			}
			for p := range pairs {
				var walk, old float64
				arm := func(legacy bool) float64 {
					if legacy {
						legacyLayerForTest = legacyLayer
						defer func() { legacyLayerForTest = nil }()
					}
					if c.name != "decode@128" {
						return c.rate()
					}
					// each decode measurement starts from the same context: re-prefill the first depth positions
					if _, err := r.PrefillLast(context.Background(), embs[:depth], 0); err != nil {
						t.Fatal(err)
					}
					return c.rate()
				}
				if p%2 == 0 { // A B
					walk, old = arm(false), arm(true)
				} else { // B A
					old, walk = arm(true), arm(false)
				}
				cell.ratios = append(cell.ratios, walk/old)
				fmt.Fprintf(os.Stderr, "B2-6 %s pair %d/%d: walk %.1f, old %.1f tok/s, ratio %.4f (load %.2f)\n", cell.name, p+1, pairs, walk, old, walk/old, b26Load())
			}
			cells = append(cells, cell)
		}
		m.Close()
	}
	for _, c := range cells {
		verdict, med, lo, hi := b26Verdict(c.ratios, c.void, c.voidReason)
		t.Logf("B2-6 %-26s load %.2f, %d MB available at start; ratios %v; median %.4f, min %.4f, max %.4f: %s",
			c.name, c.load, c.availMB, fmtRatios(c.ratios), med, lo, hi, verdict)
		if verdict != "PASS" { // the gate passes only on a median of at least 0.99 in every cell; VOID and PARKED are not passes
			t.Errorf("B2-6 %s: %s", c.name, verdict)
		}
	}
}

// b26Verdict grades one B2-6 cell by the registered bands: VOID when it started under heavy load; PASS on a median of at least
// 0.99; FAIL when it resolves slower (every pair under 1); PARKED otherwise.
func b26Verdict(ratios []float64, void bool, voidReason string) (verdict string, median, lo, hi float64) {
	s := slices.Clone(ratios)
	slices.Sort(s)
	median, lo, hi = (s[len(s)/2-1]+s[len(s)/2])/2, s[0], s[len(s)-1]
	switch {
	case void:
		return "VOID (" + voidReason + ")", median, lo, hi
	case median >= 0.99:
		return "PASS", median, lo, hi
	case hi < 1:
		return "FAIL", median, lo, hi
	}
	return "PARKED", median, lo, hi
}

// TestB26Verdict_bands: B2-6's grading is the registered bands, and only PASS is a pass.
func TestB26Verdict_bands(t *testing.T) {
	for _, c := range []struct {
		ratios []float64
		void   bool
		want   string
	}{
		{[]float64{1.0, 0.999, 1.001, 0.998, 1.002, 1.0}, false, "PASS"},
		{[]float64{0.98, 0.97, 0.985, 0.975, 0.98, 0.99}, false, "FAIL"},
		{[]float64{0.98, 1.01, 0.97, 0.985, 0.975, 0.98}, false, "PARKED"},
		{[]float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, true, "VOID (load 9.00)"},
	} {
		if got, _, _, _ := b26Verdict(c.ratios, c.void, "load 9.00"); got != c.want {
			t.Errorf("%v void %v: %s, want %s", c.ratios, c.void, got, c.want)
		}
	}
}

// loadedGitignored are TestLoadedSublayers_matchTheProgram's fixtures that are gitignored, so a clean tree lacks them; the rest
// are tracked and must be present.
var loadedGitignored = map[string]bool{"gemma4-emodel-tiny": true, "gemma4-moe-tiny": true, "gemma4-dense-twogeom-tiny": true}

// loadedPlantNeeds names, for each plant only gitignored fixtures can reach, those fixtures: the plant is NOT RUN, not red, when
// none of them ran.
var loadedPlantNeeds = map[string][]string{
	"PLE gate dropped":               {"gemma4-emodel-tiny"},
	"layer scalar dropped":           {"gemma4-dense-twogeom-tiny", "gemma4-emodel-tiny"},
	"Gemma 4 MoE norm dropped":       {"gemma4-moe-tiny"},
	"Gemma 4 MoE dense gate dropped": {"gemma4-moe-tiny"},
}

// TestLoadedSublayers_matchTheProgram is OP4cu-b2's check of a layer's program sublayers against what its upload holds
// (#588's reviews, points 2 and 3): on fixtures covering every table row's mixer and FFN, each built layer's loadedSublayers
// equals its program's pattern, and a planted mismatch in a copy of a built layer (its PLE gate, router, layer scalar, q or
// Gemma 4 MoE norm dropped, or the parallel placement flipped) is refused.
func TestLoadedSublayers_matchTheProgram(t *testing.T) {
	requireDeviceAndFixture(t, "../testdata/llama-tiny")
	walkSelfTestUnplanted(t)
	plants, ran := map[string]int{}, map[string]bool{}
	for _, dir := range []string{"../testdata/llama-tiny", "../testdata/qwen35-tiny", "../testdata/gemma4-emodel-tiny", "../testdata/gemma4-moe-tiny",
		"../testdata/gemma4-dense-twogeom-tiny", "../testdata/nemotron-tiny", "../testdata/deepseek-tiny", "../testdata/cohere-tiny"} {
		if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
			if loadedGitignored[filepath.Base(dir)] && os.Getenv("OP4CU_REQUIRE_FIXTURES") == "" {
				t.Logf("NOT RUN: %s: a gitignored fixture, not placed (scripts/worktree_fixtures.sh places it; OP4CU_REQUIRE_FIXTURES=1 makes it required)", dir)
				continue
			}
			if loadedGitignored[filepath.Base(dir)] {
				t.Fatalf("no fixture %s (gitignored, required by OP4CU_REQUIRE_FIXTURES; scripts/worktree_fixtures.sh places it): %v", dir, err)
			}
			t.Fatalf("no fixture %s (it is tracked): %v", dir, err)
		}
		ran[filepath.Base(dir)] = true
		m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatal(err)
		}
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			m.Close()
			t.Fatalf("%s: not a CUDA resident: %s", dir, m.ResidentDecline())
		}
		for l := range r.layers {
			prog, par, err := cudaSublayersOf(m.Program().Layers[l])
			if err != nil {
				t.Fatalf("%s layer %d: %v", dir, l, err)
			}
			L := r.layers[l]
			if err := checkLoadedSublayers(prog, par, loadedSublayers(&L), r.parallelBlock); err != nil {
				t.Errorf("%s layer %d: %v", dir, l, err)
			}
			for name, plant := range map[string]func(L *cudaLayer){
				"PLE gate dropped":            func(L *cudaLayer) { L.pleGate.N = 0 },
				"router dropped":              func(L *cudaLayer) { L.routerW = Buffer{} },
				"layer scalar dropped":        func(L *cudaLayer) { L.layerScalar = 0 },
				"q dropped":                   func(L *cudaLayer) { L.q.N = 0 },
				"Gemma 4 MoE norm dropped":    func(L *cudaLayer) { L.g4preFFN2 = Buffer{} },
				"gate dropped (dense FFN)":    func(L *cudaLayer) { L.g.N = 0 },
				"DeltaNet projection dropped": func(L *cudaLayer) { L.dnQKV.N = 0 },
				// #588's third review: a PLE buffer where no dense FFN runs it, and Gemma 4's MoE block without its dense branch.
				"PLE gate on a non-dense layer": func(L *cudaLayer) {
					if !(L.g.N > 0 && L.routerW == (Buffer{})) { // MoE, Gemma 4 MoE, non-gated and FFN-less layers
						L.pleGate.N = 1
					}
				},
				"layer scalar on a non-dense layer": func(L *cudaLayer) {
					if !(L.g.N > 0 && L.routerW == (Buffer{})) && L.g4preFFN2 == (Buffer{}) { // Gemma 4's MoE block scales inside segC
						L.layerScalar = 1.5
					}
				},
				"Gemma 4 MoE dense gate dropped": func(L *cudaLayer) {
					if L.g4preFFN2 != (Buffer{}) {
						L.g.N = 0
					}
				},
			} {
				P := r.layers[l]
				plant(&P)
				if !slices.Equal(loadedSublayers(&P), loadedSublayers(&L)) &&
					checkLoadedSublayers(prog, par, loadedSublayers(&P), r.parallelBlock) != nil {
					plants[name]++
				}
			}
			if checkLoadedSublayers(prog, par, loadedSublayers(&L), !r.parallelBlock) != nil {
				plants["parallel placement flipped"]++
			}
		}
		m.Close()
	}
	for _, name := range []string{"PLE gate dropped", "router dropped", "layer scalar dropped", "q dropped", "Gemma 4 MoE norm dropped",
		"gate dropped (dense FFN)", "DeltaNet projection dropped", "parallel placement flipped", "PLE gate on a non-dense layer",
		"Gemma 4 MoE dense gate dropped", "layer scalar on a non-dense layer"} {
		t.Logf("plant %q: refused on %d layers", name, plants[name])
		need := loadedPlantNeeds[name]
		if plants[name] == 0 && len(need) > 0 && !slices.ContainsFunc(need, func(f string) bool { return ran[f] }) {
			t.Logf("NOT RUN: plant %q: only %v carry what it plants, and none ran", name, need)
		} else if plants[name] == 0 {
			t.Errorf("plant %q: no layer refuses it", name)
		}
	}
}

func fmtRatios(v []float64) string {
	var b strings.Builder
	for i, x := range v {
		if i > 0 {
			b.WriteString(" ")
		}
		fmt.Fprintf(&b, "%.4f", x)
	}
	return b.String()
}

// cudaTableHasSteps reports whether s has a step for every operation of ops (program.Support.HasStep: the kinds and FFN forms,
// not the parameters), naming the first it lacks.
func cudaTableHasSteps(s *program.Support, ops []program.Op) error {
	for i := range ops {
		if !s.HasStep(&ops[i]) {
			return fmt.Errorf("operation %d (%v, FFN form %v)", i, ops[i].Kind, ops[i].FFN.Form)
		}
	}
	return nil
}

// cudaOpLayer is a hand-written layer holding o: a mixer's sublayer, an FFN-only sublayer, a step of one operation, or, for a
// part of the attention sublayer, that sublayer with o inside it.
func cudaOpLayer(o program.Op) []program.Op {
	norm := program.Op{Kind: program.OpNorm, Norm: program.NormParams{Slot: program.SlotPreAttn, Form: program.NormRMS}}
	res := program.Op{Kind: program.OpResidual}
	switch o.Kind {
	case program.OpGemma4MoE, program.OpPLE, program.OpLayerScale:
		return []program.Op{o}
	case program.OpProjQKV, program.OpProjO, program.OpAttention:
		o = program.Op{Kind: program.OpAttention}
		fallthrough
	case program.OpQKNorm, program.OpRope, program.OpAttnTemp, program.OpOutGate:
		ops := []program.Op{norm, {Kind: program.OpProjQKV}}
		if o.Kind != program.OpAttention {
			ops = append(ops, o)
		}
		return append(ops, program.Op{Kind: program.OpAttention}, program.Op{Kind: program.OpProjO}, res)
	}
	return []program.Op{norm, o, res}
}

// TestCUDATable_equalsTheWalker is OP4cu-c1's C1-1 (docs/tasks/task-forward-architecture-2026-10.md §4.9a): on every layer of
// every tracked fixture program the decoder loads on the CPU, and on a synthetic layer of each operation kind and FFN form (one
// past the last of each included), program.CUDA has a step for every operation exactly when cudaSublayersOf accepts the layer.
// The plant (the table lists Kimi Delta Attention) must turn it red on Bailing Hybrid.
func TestCUDATable_equalsTheWalker(t *testing.T) {
	planted := *program.CUDA
	planted.Kinds = append(slices.Clone(program.CUDA.Kinds), program.OpKDA)
	check := func(name string, ops []program.Op, table *program.Support) bool {
		_, _, walk := cudaSublayersOf(ops)
		tab := cudaTableHasSteps(table, ops)
		if (walk == nil) != (tab == nil) {
			if table == program.CUDA {
				t.Errorf("%s: the walker gives %v and the table %v; want both to accept or both to decline", name, walk, tab)
			}
			return false
		}
		return true
	}
	layers, fixtures, red := 0, 0, 0
	for _, dir := range walkFixtures(t) {
		m, err := decoder.Load(dir, decoder.Options{})
		if err != nil {
			continue // a tower or an encoder: no decoder program
		}
		fixtures++
		for l, ops := range m.Program().Layers {
			name := fmt.Sprintf("%s layer %d", filepath.Base(dir), l)
			check(name, ops, program.CUDA)
			if !check(name, ops, &planted) && strings.HasPrefix(filepath.Base(dir), "bailing_hybrid") {
				red++
			}
			layers++
		}
		m.Close()
	}
	var cases []program.Op
	for k := program.OpNorm; k <= program.OpLayerScale+1; k++ {
		if k != program.OpFFN && k != program.OpNorm && k != program.OpResidual {
			cases = append(cases, program.Op{Kind: k})
		}
	}
	for f := program.FfnGated; f <= program.FfnGemma4Dense+1; f++ {
		cases = append(cases, program.Op{Kind: program.OpFFN, FFN: program.FFNParams{Form: f}})
	}
	accepted := 0
	for _, o := range cases {
		if check(fmt.Sprintf("a synthetic %v layer (FFN form %d)", o.Kind, o.FFN.Form), cudaOpLayer(o), program.CUDA) {
			if _, _, err := cudaSublayersOf(cudaOpLayer(o)); err == nil {
				accepted++
			}
		}
	}
	t.Logf("%d layers of %d fixtures and %d synthetic layers (%d accepted) checked; the KDA plant is red on %d Bailing Hybrid layers",
		layers, fixtures, len(cases), accepted, red)
	if fixtures < 30 || accepted == 0 || accepted == len(cases) {
		t.Fatalf("%d fixtures, %d of %d synthetic layers accepted: the sweep must hold many fixtures and both outcomes", fixtures, accepted, len(cases))
	}
	if red == 0 {
		t.Errorf("plant (the table lists kda): no Bailing Hybrid layer goes red")
	}
}

// cudaTableGaps is what m needs that CUDA does not implement (its program outside program.CUDA, or a feature the program does not
// express), or "" when CUDA's admission takes it: what a test that skips on a declined model checks, since CUDA's admission reads
// the table (OP4cu-c2).
func cudaTableGaps(m *decoder.Model) string {
	if err := program.CUDA.Check(m.Program()); err != nil {
		return err.Error()
	}
	if missing := m.FeaturesOutsideProgram(); len(missing) > 0 {
		return fmt.Sprint(missing)
	}
	return ""
}

// TestCUDAStep_tableEqualsTheFlagList is OP4cu-d2's D2-1 (docs/tasks/task-forward-architecture-2026-10.md §4.9a): on every tracked
// tiny fixture that builds a CUDA resident at int4, program.CUDAStep admits the program exactly when the step's former flag list
// (MoE, Gemma 4's MoE, MLA, a recurrent mixer) did not exclude the resident. The walk adds decoder/testdata/gptoss_tiny.gguf, which
// walkFixtures (directories with safetensors) does not reach. Each plant (the step's table lists the MoE form, DeltaNet, MLA,
// gpt-oss's FFN form, Mamba-2, or Gemma 4's MoE block, the last required with the gitignored fixtures) must turn it red on a fixture.
// At two KV slots, BatchStepRange must serve exactly what the former flag list and the step's runtime limits allow. Every MLA fixture also has MoE layers, so the MLA plant is scored on each MLA
// program rewritten with a dense FFN in place of its MoE one: the table must still refuse it, for MLA alone, and the plant admits it.
func TestCUDAStep_tableEqualsTheFlagList(t *testing.T) {
	requireDeviceAndFixture(t, "../testdata/llama-tiny")
	plant := func(edit func(s *program.Support)) *program.Support {
		p := *program.CUDAStep
		p.Kinds = slices.Clone(program.CUDAStep.Kinds)
		p.FFN = map[program.FFNForm][]program.ActKind{}
		for f, acts := range program.CUDAStep.FFN {
			p.FFN[f] = acts
		}
		edit(&p)
		return &p
	}
	plants := []struct {
		name     string
		table    *program.Support
		required bool
	}{
		{"the MoE form", plant(func(s *program.Support) { s.FFN[program.FfnMoE] = []program.ActKind{program.ActSiLU} }), true},
		{"DeltaNet", plant(func(s *program.Support) { s.Kinds = append(s.Kinds, program.OpDeltaNet) }), true},
		{"MLA", plant(func(s *program.Support) { s.Kinds = append(s.Kinds, program.OpMLA) }), true},
		{"gpt-oss's FFN form", plant(func(s *program.Support) { s.FFN[program.FfnGptOss] = []program.ActKind{program.ActSiLU} }), true},
		{"Mamba-2", plant(func(s *program.Support) { s.Kinds = append(s.Kinds, program.OpMamba2) }), true},
		// Its fixtures (gemma4-moe-*) are gitignored: required when they are (OP4CU_REQUIRE_FIXTURES=1, the push record's run).
		{"Gemma 4's MoE block", plant(func(s *program.Support) { s.Kinds = append(s.Kinds, program.OpGemma4MoE) }), os.Getenv("OP4CU_REQUIRE_FIXTURES") != ""},
	}
	// denseFFN is p with each MoE or gpt-oss FFN replaced by a gated SiLU one.
	denseFFN := func(p *program.Program) *program.Program {
		q := *p
		q.Layers = make([][]program.Op, len(p.Layers))
		for l, ops := range p.Layers {
			q.Layers[l] = slices.Clone(ops)
			for i := range q.Layers[l] {
				if o := &q.Layers[l][i]; o.Kind == program.OpFFN && (o.FFN.Form == program.FfnMoE || o.FFN.Form == program.FfnGptOss) {
					o.FFN = program.FFNParams{Form: program.FfnGated, Act: program.ActSiLU}
				}
			}
		}
		return &q
	}
	mlaRewrites, ranged, stepServes := 0, 0, 0
	red := make([]int, len(plants))
	built, admitted := 0, 0
	const gptOssTiny = "../decoder/testdata/gptoss_tiny.gguf"
	gptOss := false
	for _, dir := range append(walkFixtures(t), gptOssTiny) {
		m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			continue
		}
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			m.Close()
			continue
		}
		built++
		gptOss = gptOss || dir == gptOssTiny
		excluded := r.moe || r.gemma4Moe || r.isMLA || r.recurrent()
		table := program.CUDAStep.Check(r.prog) == nil
		if table == excluded {
			t.Errorf("%s: the step's table admits %v, the former flag list excluded %v", filepath.Base(dir), table, excluded)
		}
		if table {
			admitted++
		}
		for i, p := range plants {
			if p.name == "MLA" {
				continue
			}
			if (p.table.Check(r.prog) == nil) == excluded {
				red[i]++
			}
		}
		// The step's own gate: BatchStepRange serves the model, at two KV slots, exactly when the table admits it and its other
		// limits hold (#602's review, point 2).
		if m2, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 2}); err == nil {
			if r2, ok := m2.ResidentForwardForTest().(*cudaResident); ok && r2.KVSlots() >= 2 {
				ranged++
				_, hi := r2.BatchStepRange()
				// The oracle is the step's former flag list and its runtime limits, not the table BatchStepRange reads.
				ex2 := r2.moe || r2.gemma4Moe || r2.isMLA || r2.recurrent()
				want := !ex2 && r2.prefillReady && !r2.cacheExperts && r2.prefillStaticDecline() == nil
				if (hi > 0) != want {
					t.Errorf("%s at two slots: BatchStepRange serves %v, the former flag list and the limits %v", filepath.Base(dir), hi > 0, want)
				}
				if hi > 0 {
					stepServes++
				}
			}
			m2.Close()
		}
		if r.isMLA {
			mlaRewrites++
			d := denseFFN(r.prog)
			if program.CUDAStep.Check(d) == nil {
				t.Errorf("%s with a dense FFN: the step's table admits an MLA program", filepath.Base(dir))
			}
			for i, p := range plants {
				if p.name == "MLA" && p.table.Check(d) == nil {
					red[i]++
				}
			}
		}
		m.Close()
	}
	t.Logf("gptoss_tiny built resident: %v; MLA programs rewritten with a dense FFN: %d; at two slots %d built, %d served by BatchStepRange",
		gptOss, mlaRewrites, ranged, stepServes)
	if ranged < 20 || stepServes == 0 || stepServes == ranged {
		t.Errorf("at two slots %d built, %d served by BatchStepRange: the step's gate needs both outcomes", ranged, stepServes)
	}
	if os.Getenv("OP4CU_REQUIRE_FIXTURES") != "" && built < 46 {
		t.Errorf("%d fixtures built resident, want the 46 of #602's review reading with the fixtures required", built)
	}
	if !gptOss || mlaRewrites == 0 {
		t.Errorf("gptoss_tiny resident %v, %d MLA programs: the gpt-oss and MLA plants need both", gptOss, mlaRewrites)
	}
	t.Logf("%d fixtures built resident, %d admitted to the step", built, admitted)
	if built < 30 || admitted == 0 || admitted == built {
		t.Fatalf("%d built, %d admitted: the comparison must hold many fixtures and both outcomes", built, admitted)
	}
	for i, p := range plants {
		t.Logf("plant (the step's table lists %s): red on %d fixtures", p.name, red[i])
		if p.required && red[i] == 0 {
			t.Errorf("plant (the step's table lists %s): no fixture goes red", p.name)
		}
	}
}

// TestPrefillSteps_equalTheFlags is OP4cu-d1's D1-1 (docs/tasks/task-forward-architecture-2026-10.md §4.9a): on every tracked tiny
// fixture that builds a CUDA resident at int4, each layer's batched prefill steps, where it has any, are the blocks the former
// branches chose, a model the flags' guards batch (prefillFlagsDecline) has steps on every layer, and the batched pass serves
// exactly the models those guards batch. On a model they decline, a layer with no batched block (MLA, Mamba-2, an attention-only or
// FFN-only block) carries no steps.
func TestPrefillSteps_equalTheFlags(t *testing.T) {
	requireDeviceAndFixture(t, "../testdata/llama-tiny")
	// The build's own step checks are off for the comparison, so a regression meets these branches instead of a declined build
	// (#603's second review); each fixture the flags' guards batch is then loaded again with them on, and must stay resident.
	prefillStepsBuildChecksOffForTest = true
	t.Cleanup(func() { prefillStepsBuildChecksOffForTest = false })
	built, byFlags, batched, layers, resident := 0, 0, 0, 0, 0
	declinedStepless := 0 // fixtures the flags' guards decline whose block-less layers carry no steps
	emptied := false
	for _, dir := range walkFixtures(t) {
		m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			continue
		}
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			m.Close()
			continue
		}
		built++
		flagsServe := r.prefillFlagsDecline() == nil
		serves := r.prefillStaticDecline() == nil
		if flagsServe {
			byFlags++
		}
		if serves {
			batched++
		}
		if serves != flagsServe {
			t.Errorf("%s: the flags' guards batch it %v, the batched pass with its steps %v", filepath.Base(dir), flagsServe, serves)
		}
		for l := range r.layers {
			L := &r.layers[l]
			switch {
			case len(L.prefillSteps) > 0 && !slices.Equal(L.prefillSteps, prefillStepsFromFlags(L)):
				t.Errorf("%s layer %d: batched steps %v, the former branches chose %v", filepath.Base(dir), l, L.prefillSteps, prefillStepsFromFlags(L))
			case len(L.prefillSteps) == 0 && flagsServe:
				t.Errorf("%s layer %d: no batched steps on a model the flags' guards batch", filepath.Base(dir), l)
			case len(L.prefillSteps) > 0:
				layers++
			}
		}
		// The branch above is reachable: a layer emptied of its steps leaves the flags' guards batching the model, and the pass
		// declines it.
		if flagsServe && !emptied {
			saved := r.layers[0].prefillSteps
			r.layers[0].prefillSteps = nil
			if r.prefillFlagsDecline() != nil || r.prefillStaticDecline() == nil {
				t.Errorf("%s: with layer 0's steps emptied, the flags' guards decline %v and the pass declines %v, want false and true",
					filepath.Base(dir), r.prefillFlagsDecline() != nil, r.prefillStaticDecline() != nil)
			}
			r.layers[0].prefillSteps = saved
			emptied = true
		}
		// Read before Close, which frees the layers.
		if !flagsServe {
			// A layer the batched pass has no block for (MLA, Mamba-2, an attention-only or FFN-only block) carries no steps.
			stepless := 0
			for l := range r.layers {
				L := &r.layers[l]
				if L.mixer == decoder.MixerMLA || L.mixer == decoder.MixerMamba || L.mixer == decoder.MixerNone || !L.ffn {
					if len(L.prefillSteps) != 0 {
						t.Errorf("%s layer %d: no batched block for it, and steps %v", filepath.Base(dir), l, L.prefillSteps)
					}
					stepless++
				}
			}
			if stepless > 0 {
				declinedStepless++
				t.Logf("%s: declined by the flags' guards, %d block-less layers with no steps", filepath.Base(dir), stepless)
			}
			m.Close()
			continue
		}
		m.Close()
		prefillStepsBuildChecksOffForTest = false
		m2, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
		prefillStepsBuildChecksOffForTest = true
		if err != nil {
			t.Errorf("%s: with the build's step checks on, load: %v", filepath.Base(dir), err)
			continue
		}
		if _, ok := m2.ResidentForwardForTest().(*cudaResident); ok {
			resident++
		} else {
			t.Errorf("%s: the flags' guards batch it, and with the build's step checks on it is not resident: %s", filepath.Base(dir), m2.ResidentDecline())
		}
		m2.Close()
	}
	t.Logf("%d fixtures built resident, %d batched by the flags' guards (%d of them resident with the build's step checks on), "+
		"%d with a batched prefill, %d layers with batched steps checked", built, byFlags, resident, batched, layers)
	// 42 is D1-1's reading with the gitignored fixtures placed (docs/tasks/task-forward-architecture-2026-10.md §4.9a, OP4cu-d1);
	// a clean tree has fewer, and OP4CU_REQUIRE_FIXTURES=1 (the push record's run) asks for all of them.
	floor, declinedFloor := 1, 0
	if os.Getenv("OP4CU_REQUIRE_FIXTURES") != "" {
		floor, declinedFloor = 42, 3 // deepseek-tiny, kimi-tiny, nemotron-tiny
	}
	if declinedStepless < declinedFloor {
		t.Errorf("%d declined fixtures with block-less layers checked, want at least %d", declinedStepless, declinedFloor)
	}
	if built < 30 || byFlags < floor || batched != byFlags || resident != byFlags || !emptied {
		t.Fatalf("%d built, %d batched by the flags, %d batched, %d resident with the checks on (want all of them, and at least %d)",
			built, byFlags, batched, resident, floor)
	}
}

// TestPrefillSteps_buildDeclinesAStepLessLayer is the build-time half of D1-1 (#603's second review): a layer built with no
// batched steps on a model the batched guards admit declines the resident build, naming the layer.
func TestPrefillSteps_buildDeclinesAStepLessLayer(t *testing.T) {
	requireDeviceAndFixture(t, "../testdata/llama-tiny")
	prefillStepsEmptyLayerForTest = 1
	t.Cleanup(func() { prefillStepsEmptyLayerForTest = -1 })
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if m.ResidentForwardForTest() != nil {
		t.Fatalf("llama-tiny with layer 1 built step-less is resident; the build must decline it")
	}
	if d := m.ResidentDecline(); !strings.Contains(d, "layer 1: the batched prefill's guards admit the model and the program gives the layer no batched steps") {
		t.Fatalf("decline %q does not name the step-less layer", d)
	}
	t.Logf("declined: %s", m.ResidentDecline())
}

// TestPrefillWalk_plantsGoRed is OP4cu-d1's D1-2: on every fixture with a batched prefill, each plant (the dense FFN, MoE or DeltaNet
// step dropped, or the FFN step run before the mixer's) changes a 40-token PrefillLast's logits from the unplanted run's wherever
// the fixture runs that step (#603's review, point 3).
func TestPrefillWalk_plantsGoRed(t *testing.T) {
	requireDeviceAndFixture(t, "../testdata/llama-tiny")
	walkSelfTestUnplanted(t)
	t.Cleanup(func() { prefillPlant = prefillPlantNone })
	plants := []struct {
		kind prefillPlantKind
		name string
		step prefillStep // the step a fixture must run for the plant to apply; 0 for every fixture
	}{
		{prefillPlantDropDense, "the dense FFN step dropped", pstepDenseFFN},
		{prefillPlantDropMoE, "the MoE step dropped", pstepMoE},
		{prefillPlantDropDeltaNet, "the DeltaNet step dropped", pstepDeltaNet},
		{prefillPlantSwap, "the FFN step before the mixer's", 0},
	}
	red, applies := map[prefillPlantKind]int{}, map[prefillPlantKind]int{}
	ran := 0
	for _, dir := range walkFixtures(t) {
		m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			continue
		}
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok || r.prefillStaticDecline() != nil {
			m.Close()
			continue
		}
		_, _, _, _, _, _, vocab := m.Dims()
		embs := make([][]float32, 40)
		for i := range embs {
			embs[i] = m.EmbedResidentForTest((i*37 + 5) % vocab)
		}
		last := func() []float32 {
			r.Reset()
			l, err := r.PrefillLast(context.Background(), embs, 0)
			if err != nil {
				t.Fatalf("%s: %v", dir, err)
			}
			return slices.Clone(l)
		}
		base := last()
		for _, p := range plants {
			if p.step != 0 && !slices.ContainsFunc(r.layers, func(L cudaLayer) bool { return slices.Contains(L.prefillSteps, p.step) }) {
				continue
			}
			applies[p.kind]++
			prefillPlant = p.kind
			if !slices.Equal(last(), base) {
				red[p.kind]++
			} else {
				t.Errorf("plant %q: %s runs that step and its logits do not change", p.name, filepath.Base(dir))
			}
			prefillPlant = prefillPlantNone
		}
		ran++
		m.Close()
	}
	if os.Getenv("OP4CU_REQUIRE_FIXTURES") != "" && ran < 42 {
		t.Errorf("%d fixtures with a batched prefill, want D1-1's 42 with the fixtures required", ran)
	}
	for _, p := range plants {
		t.Logf("plant %q: red on %d of the %d fixtures it applies to (%d with a batched prefill)", p.name, red[p.kind], applies[p.kind], ran)
		if applies[p.kind] == 0 {
			t.Errorf("plant %q: no fixture runs its step", p.name)
		}
	}
}
