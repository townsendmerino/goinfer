//go:build cuda && goinfer_testhooks

package cuda

import (
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

// expertVariantFixture writes, under t's temporary directory, the glm4_moe fixture src with moe_intermediate_size set to inter and
// num_hidden_layers to 2 (layer 0 dense, layer 1 MoE): layer 2's tensors are dropped, the routed and shared experts' tensors are
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
			cfg["moe_intermediate_size"], cfg["num_hidden_layers"] = inter, 2
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
	for k := range hdr {
		if k != "__metadata__" && !strings.HasPrefix(k, "model.layers.2.") {
			names = append(names, k)
		}
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
			n := 1
			for i, d := range tn.Shape {
				if d == old {
					tn.Shape[i] = inter
				}
				n *= tn.Shape[i]
			}
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
		t.Errorf("G-DC7: no @halfcache line holds fewer expert-cache slots than its profile line")
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
// memCensusHalfCaps counts the @halfcache lines (memCensusHalfCache) whose derived cache holds fewer slots than the line of the same
// fixture, cell and profile at the profile's own figure.
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
		prof, ok := byKey[strings.TrimSuffix(k, memCensusHalfCache)]
		if !ok {
			t.Errorf("a @halfcache line has no profile line: %s", l)
			continue
		}
		h, okH := slots(l)
		f, okF := slots(prof)
		if okH && okF && h < f {
			n++
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
