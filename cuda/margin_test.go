//go:build cuda

package cuda

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSH6C_chargedKVArithmetic is H6C-G1's KV half (docs/tasks/task-kernel-survey-2026-10.md, "KS-H6 build"): kvChargedBytes against a
// hand count on synthetic layers (pool buffers under, exactly at and over the 2 MiB quantum; f16 and f32; an MLA latent pool; a
// KV-shared layer and a layer with no KV), the plan's rounding bound at least the real rounding, and the planted defect (the pools
// priced as requested, kvBytesForCap's linear figure) below the charged figure.
func TestKSH6C_chargedKVArithmetic(t *testing.T) {
	const q = allocQuantumBytes
	attn := func(kvDim int, half bool) cudaLayer {
		return cudaLayer{mixer: decoder.MixerAttention, kvDim: kvDim, kvHalf: half}
	}
	rowsAt := func(cap, slots int) int64 { return int64(slots*kvBlocksFor(cap)+1) * kvBlock }
	// At cap 1,984 a pool holds 31 blocks plus the null block, 2,048 rows, so an f32 buffer is 8,192*kvDim bytes: kvDim 128, 256 and
	// 384 give 1, 2 and 3 MiB, under, exactly at and over the quantum (and 384 at f16 is 1.5 MiB).
	cap := 1984
	full := rowsAt(cap, 1)
	if full != 2048 {
		t.Fatalf("a pool at cap %d holds %d rows, want 2,048 (kvBlock %d)", cap, full, kvBlock)
	}
	under, exact, over := 128, 256, 384
	mla := cudaLayer{mixer: decoder.MixerMLA, kvDim: 512}
	shared := attn(1024, false)
	shared.kvShared = true
	none := cudaLayer{mixer: decoder.MixerMamba}
	layers := []cudaLayer{attn(under, false), attn(exact, false), attn(over, false), attn(over, true), mla, shared, none}
	hand := func(n int64) int64 {
		if n < q {
			return n
		}
		return (n + q - 1) / q * q
	}
	var want, linear int64
	for _, Ly := range layers[:4] {
		elem := int64(4)
		if Ly.kvHalf {
			elem = 2
		}
		b := full * int64(Ly.kvDim) * elem
		want += 2 * hand(b)
		linear += 2 * b
	}
	mlaB := full * 512 * 4
	want += hand(mlaB)
	linear += mlaB
	tabs := int64(kvGroups*kvBlocksFor(cap)) * 4
	want += hand(tabs)
	linear += tabs
	if got := kvChargedBytes(cap, 1, layers); got != want {
		t.Errorf("kvChargedBytes %d, the hand count %d", got, want)
	}
	if bound := kvRoundingBound(4, 1); want-linear > bound {
		t.Errorf("the rounding %d exceeds the plan's bound %d", want-linear, bound)
	}
	// The plant: pricing the pools as requested (kvBytesForCap plus the null block) must fall below the charged figure here, or the
	// table cannot see the rounding.
	requested := kvBytesForCap(cap, layers) + kvPoolNullBytes(layers)
	if requested >= want {
		t.Errorf("the pools priced as requested (%d) are not below the charged figure (%d): the plant would pass", requested, want)
	}
	// Two slots share each pool: one buffer of 2*blocks+1 rows, not two of blocks+1.
	full2 := rowsAt(cap, 2)
	var want2 int64
	for _, Ly := range layers[:4] {
		elem := int64(4)
		if Ly.kvHalf {
			elem = 2
		}
		want2 += 2 * hand(full2*int64(Ly.kvDim)*elem)
	}
	want2 += hand(full2*512*4) + hand(int64(2*kvGroups*kvBlocksFor(cap))*4)
	if got := kvChargedBytes(cap, 2, layers); got != want2 {
		t.Errorf("two slots: kvChargedBytes %d, the hand count %d", got, want2)
	}
	// The search and the slot count invert kvChargedBytes.
	if c, ok := kvChargedCapForBytes(want, 1, 64, cap, layers); !ok || c != cap {
		t.Errorf("kvChargedCapForBytes(at the full figure) = %d, %v; want %d", c, ok, cap)
	}
	if c, ok := kvChargedCapForBytes(want-1, 1, 64, cap, layers); !ok || c >= cap {
		t.Errorf("kvChargedCapForBytes(one byte short) = %d, %v; want below %d", c, ok, cap)
	}
	if n := kvChargedSlotsFit(4, want2+1, cap, layers, 1); n != 2 {
		t.Errorf("kvChargedSlotsFit = %d with room for exactly two slots, want 2", n)
	}
	if n := kvChargedSlotsFit(0, 1<<40, cap, layers, 0); n != 1 {
		t.Errorf("kvChargedSlotsFit of a request for 0 slots = %d, want 1", n)
	}
	if n := kvChargedSlotsFit(4, 1, cap, layers, 0); n != 1 {
		t.Errorf("kvChargedSlotsFit with no room = %d, want 1 (the refusal is checkKVFits')", n)
	}
}

// testCtxMargin is the context margin the plan and fit-check tests that construct a resident by hand price against (the figure the
// flat margin was), and fixedMargin hands it to resolveCtxCapFit as a margin that does not vary with the context.
const testCtxMargin = int64(384 << 20)

func fixedMargin(b int64) func(int) int64 { return func(int) int64 { return b } }

// TestKSH6C_marginParts is H6C-G1's margin half: on a struct-only resident, postPoolBytes against a hand count, and the margin as
// exactly its four parts. Two layers of qDim 8, hidden 64, a dense FFN of 128: every post-pool buffer is under the quantum except
// the logits at a 600,000-token vocabulary (2,400,000 B, charged 4 MiB), so part 1 is one 2 MiB page plus 4 MiB, or one page at a
// 1,000-token vocabulary. Plant: a margin with the floor left out does not equal the margin.
func TestKSH6C_marginParts(t *testing.T) {
	r := declineFixture(2, "int4")
	r.hidden, r.inter, r.vocab = 64, 128, 1000
	if got := r.postPoolBytes(); got != 2<<20 {
		t.Errorf("postPoolBytes at a 1,000-token vocabulary %d B, the hand count one 2 MiB page", got)
	}
	r.vocab = 600000
	if got := r.postPoolBytes(); got != 6<<20 {
		t.Errorf("postPoolBytes at a 600,000-token vocabulary %d B, the hand count 6 MiB (one page, and the logits at 4 MiB)", got)
	}
	const ctx = 4096
	parts := r.postPoolBytes() + r.PrefillScratchBytes(prefillDefaultChunk) + r.PrefillSpanScratchBytes(prefillDefaultChunk, ctx)
	m := r.ctxMarginFor(ctx)
	if m != parts+allocFloorBytes+decodeAllowanceBytes {
		t.Errorf("ctxMarginFor %d B, its parts %d + floor %d + allowance %d", m, parts, int64(allocFloorBytes), int64(decodeAllowanceBytes))
	}
	if planted := parts + decodeAllowanceBytes; planted == m {
		t.Error("plant: a margin without the allocation floor equals the margin; the check cannot see it")
	}
}

// TestKSH6C_roundingBoundHolds is H6C-G1b: over a sweep of synthetic layer sets, the pools as charged (kvChargedBytes) never exceed
// Plan's linear figure (decoder's cudaKVBytes: the slots' blocks and tables plus the null blocks, which kvBytesForCap and
// kvPoolNullBytes mirror) by more than kvRoundingBound. Gemini's case, kvDim 768 at f32 and context 640, a buffer whose null
// block carries it past a 2 MiB boundary, is in the sweep.
func TestKSH6C_roundingBoundHolds(t *testing.T) {
	worst, checked := int64(-1<<62), 0
	for kvDim := 64; kvDim <= 1024; kvDim += 64 {
		for _, half := range []bool{false, true} {
			for _, mla := range []bool{false, true} {
				layers := make([]cudaLayer, 4)
				for i := range layers {
					layers[i] = cudaLayer{mixer: decoder.MixerAttention, kvDim: kvDim, kvHalf: half}
					if mla {
						layers[i].mixer = decoder.MixerMLA
					}
				}
				owners, nMLA := kvLayerCounts(layers)
				bound := kvRoundingBound(owners, nMLA)
				for cap := kvBlock; cap <= 20000; cap += kvBlock {
					for slots := 1; slots <= 4; slots++ {
						linear := int64(slots)*kvBytesForCap(cap, layers) + kvPoolNullBytes(layers)
						over := kvChargedBytes(cap, slots, layers) - linear
						worst = max(worst, over-bound)
						checked++
						if over > bound {
							t.Fatalf("kvDim %d half %v mla %v cap %d slots %d: the charged pools exceed Plan's figure by %d B, past the bound %d B",
								kvDim, half, mla, cap, slots, over, bound)
						}
					}
				}
			}
		}
	}
	t.Logf("%d configurations; the closest came %d B under the bound", checked, -worst)
}

// TestPlanMemory runs the memory plan on a tiny fixture without a device. With room everywhere it plans one slot at the
// default context; a short weights reading or KV reading refuses at that site; an unknown KV reading grants one slot.
func TestPlanMemory(t *testing.T) {
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("fixture: %v", err)
	}
	defer m.Close()
	const room = int64(8 << 30)
	full := memReadings{WeightsOpen: room, Plan: room, Weights: room, KV: room, WeightsOpenOK: true, PlanOK: true, WeightsOK: true, KVOK: true}
	b := &cudaBackend{}
	p, err := b.planMemory(m, full)
	if err != nil {
		t.Fatal(err)
	}
	if p.Decline != "" || p.CtxCap < cudaCtxCapDefault || p.KVSlots < 1 || p.Margin <= 0 {
		t.Fatalf("room everywhere: %+v", p)
	}
	short := full
	short.Weights = 1
	if p, err := b.planMemory(m, short); err != nil || p.DeclineAt != "weights" {
		t.Errorf("short weights reading: %+v %v, want a refusal at weights", p, err)
	}
	short = full
	short.KV = 1
	if p, err := b.planMemory(m, short); err != nil || p.DeclineAt != "kv" {
		t.Errorf("short KV reading: %+v %v, want a refusal at kv", p, err)
	}
	unknown := full
	unknown.KVOK = false
	if p, err := b.planMemory(m, unknown); err != nil || p.Decline != "" || p.KVSlots != 1 {
		t.Errorf("unknown KV reading: %+v %v, want one slot and no refusal", p, err)
	}
}

// TestWeightsFitDecision holds the weights check at its boundaries: an unknown reading passes, a fit exactly at the budget
// passes, one byte over refuses, and a companion attach is named in the refusal.
func TestWeightsFitDecision(t *testing.T) {
	const need, margin = int64(1000), int64(100)
	for _, c := range []struct {
		name        string
		extra, free int64
		ok          bool
		want        string // "" fits; else a substring of the refusal
	}{
		{"no reading", 0, 0, false, ""},
		{"exactly at the budget", 0, need + margin, true, ""},
		{"one byte over", 0, need + margin - 1, true, "dense weights need"},
		{"companion at the budget", 50, need + 50 + margin, true, ""},
		{"companion one byte over", 50, need + 50 + margin - 1, true, "reserved for a companion"},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := weightsFitDecision(need, c.extra, c.free, c.ok, margin)
			switch {
			case c.want == "" && err != nil:
				t.Errorf("refused: %v", err)
			case c.want != "" && (err == nil || !strings.Contains(err.Error(), c.want)):
				t.Errorf("got %v, want a refusal containing %q", err, c.want)
			}
		})
	}
	if weightsFitDecision(0, 0, 1, true, margin) != nil {
		t.Error("an unknown need (0) must pass")
	}
}

// kvTestLayers is two full-attention layers with f32 KV, kvDim 256: what kvChargedBytes prices.
func kvTestLayers() []cudaLayer {
	return []cudaLayer{{idx: 0, kvDim: 256, nKV: 2, hd: 128}, {idx: 1, kvDim: 256, nKV: 2, hd: 128}}
}

// TestKVFitDecision holds checkKVFits' decision at its boundaries: an unknown reading grants one slot at the planned context;
// an explicit context fits exactly at the budget and refuses one byte under it with kvWontFitError; an unpinned context over
// the budget is trimmed, never below cudaCtxCapDefault, and refused when even that does not fit; extra slots are clamped to
// what fits.
func TestKVFitDecision(t *testing.T) {
	ls := kvTestLayers()
	const margin = int64(1 << 20)
	one := func(ctx int) int64 { return kvChargedBytes(ctx, 1, ls) }

	if d := kvFitDecision(kvFitIn{ctxCap: 8192, slotsReq: 4, layers: ls, margin: margin}, 0, false); d.ctxCap != 8192 || d.slots != 1 || d.err != nil || len(d.logs) != 0 {
		t.Errorf("no reading: %+v, want ctx 8192, 1 slot, no refusal", d)
	}

	at := uint64(one(8192) + margin)
	if d := kvFitDecision(kvFitIn{ctxCap: 8192, explicit: true, slotsReq: 1, layers: ls, margin: margin}, at, true); d.err != nil || d.ctxCap != 8192 {
		t.Errorf("explicit, exactly at the budget: %+v", d)
	}
	d := kvFitDecision(kvFitIn{ctxCap: 8192, explicit: true, slotsReq: 1, layers: ls, margin: margin}, at-1, true)
	var wf *kvWontFitError
	if !errors.As(d.err, &wf) || d.ctxCap != 8192 {
		t.Errorf("explicit, one byte under: %+v, want kvWontFitError at ctx 8192", d)
	}

	d = kvFitDecision(kvFitIn{ctxCap: 8192, slotsReq: 1, layers: ls, margin: margin}, at-1, true)
	if d.err != nil || d.ctxCap >= 8192 || d.ctxCap < cudaCtxCapDefault || len(d.logs) != 1 || !strings.Contains(d.logs[0], "trimmed from 8192") {
		t.Errorf("unpinned, one byte under: %+v, want a trim below 8192, at or above %d, logged once", d, cudaCtxCapDefault)
	}
	if one(d.ctxCap)+margin > int64(at-1) {
		t.Errorf("the trimmed context %d does not fit the budget it was trimmed to", d.ctxCap)
	}

	floor := uint64(one(cudaCtxCapDefault) + margin)
	d = kvFitDecision(kvFitIn{ctxCap: 8192, slotsReq: 1, layers: ls, margin: margin}, floor-1, true)
	if d.err == nil || !strings.Contains(d.err.Error(), "default resident context") || errors.As(d.err, &wf) {
		t.Errorf("unpinned, under the floor: %+v, want the default-context decline (not kvWontFitError)", d)
	}

	// Slots before context: an unpinned 8192 where one slot fits but two do not gives up positions until both fit.
	twoAt := uint64(kvChargedBytes(8192, 1, ls) + margin + (kvChargedBytes(8192, 2, ls)-kvChargedBytes(8192, 1, ls))/2)
	d = kvFitDecision(kvFitIn{ctxCap: 8192, slotsReq: 2, layers: ls, margin: margin}, twoAt, true)
	if d.err != nil || d.slots != 2 || d.ctxCap >= 8192 || d.ctxCap < cudaCtxCapDefault || len(d.logs) != 1 || !strings.Contains(d.logs[0], "so the 2 requested KV slots") {
		t.Errorf("multi-slot trim: %+v, want 2 slots at a context trimmed below 8192, logged once", d)
	}
	if kvChargedBytes(d.ctxCap, 2, ls)+margin > int64(twoAt) {
		t.Errorf("multi-slot trim: two slots at %d do not fit the budget", d.ctxCap)
	}

	two := uint64(kvChargedBytes(cudaCtxCapDefault, 2, ls) + margin)
	d = kvFitDecision(kvFitIn{ctxCap: cudaCtxCapDefault, slotsReq: 4, layers: ls, margin: margin}, two, true)
	if d.err != nil || d.slots != 2 || len(d.logs) != 1 || !strings.Contains(d.logs[0], "2 resident KV slots of 4") {
		t.Errorf("slot clamp: %+v, want 2 slots of 4, logged once", d)
	}
}

// TestKVFitDecision_trimPlant shows the one-slot trim is what keeps an unpinned context over the budget loading: with the trim
// planted off (kvTrimOffForTest, the plant G-MP3b used on the card), the same reading refuses.
func TestKVFitDecision_trimPlant(t *testing.T) {
	ls := kvTestLayers()
	const margin = int64(1 << 20)
	under := uint64(kvChargedBytes(8192, 1, ls)+margin) - 1
	in := kvFitIn{ctxCap: 8192, slotsReq: 1, layers: ls, margin: margin}
	if d := kvFitDecision(in, under, true); d.err != nil {
		t.Fatalf("unplanted: %v", d.err)
	}
	defer func(off bool) { kvTrimOffForTest = off }(kvTrimOffForTest)
	kvTrimOffForTest = true
	if d := kvFitDecision(in, under, true); d.err == nil {
		t.Errorf("trim planted off: %+v, want a refusal", d)
	}
}

// TestPlanMemory_moePlannedContext plans qwen3next-tiny, a MoE fixture whose window (32768) puts its context through the plan:
// the plan reads the supplied figure (PoolPaid, a context above the floor), a short pre-pool reading refuses at weightsOpen with
// the margin and slot request it was judged against recorded, and the plan fires no reading hook (they record the real build's
// readings only).
func TestPlanMemory_moePlannedContext(t *testing.T) {
	m, err := decoder.Load("../testdata/qwen3next-tiny", decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("fixture: %v", err)
	}
	defer m.Close()
	const room = int64(8 << 30)
	full := memReadings{WeightsOpen: room, Plan: room, Weights: room, KV: room, WeightsOpenOK: true, PlanOK: true, WeightsOK: true, KVOK: true}
	var hooks []string
	defer func(a func(string, int64, bool), b func(int64)) { memReadSeenForTest, planFreeSeenForTest = a, b }(memReadSeenForTest, planFreeSeenForTest)
	memReadSeenForTest = func(site string, _ int64, _ bool) { hooks = append(hooks, site) }
	planFreeSeenForTest = func(int64) { hooks = append(hooks, "planFree") }
	b := &cudaBackend{}
	p, err := b.planMemory(m, full)
	if err != nil {
		t.Fatal(err)
	}
	if !p.PoolPaid || p.CtxPlanned <= cudaCtxCapDefault || p.Decline != "" {
		t.Errorf("room everywhere: %+v, want the pool paid and a context planned above %d", p, cudaCtxCapDefault)
	}
	short := full
	short.WeightsOpen = 1
	p, err = b.planMemory(m, short)
	if err != nil || p.DeclineAt != "weightsOpen" {
		t.Fatalf("short pre-pool reading: %+v %v, want a refusal at weightsOpen", p, err)
	}
	if p.Margin <= 0 || p.KVSlotsReq < 1 {
		t.Errorf("weightsOpen refusal: %+v, want the margin and slot request it was judged against", p)
	}
	if p.PoolPaid {
		t.Errorf("weightsOpen refusal: %+v reports the pool paid; the build pays it only after that check passes", p)
	}
	if len(hooks) != 0 {
		t.Errorf("the memory plan fired reading hooks %v: they record the real build's readings only", hooks)
	}
}

// TestCkptDeclineFor holds the recurrent checkpoint's decision at its boundary: kept exactly at bytes plus the reserve, refused one
// byte under with the figures named, and refused on an unreadable figure.
func TestCkptDeclineFor(t *testing.T) {
	const bytes, reserve = int64(64 << 20), int64(384 << 20)
	if why := ckptDeclineFor(bytes, reserve, uint64(bytes+reserve), nil); why != "" {
		t.Errorf("exactly at the reserve: refused (%s)", why)
	}
	if why := ckptDeclineFor(bytes, reserve, uint64(bytes+reserve-1), nil); !strings.Contains(why, "needed beside a 384 MiB reserve") {
		t.Errorf("one byte under: %q, want the figures named", why)
	}
	if why := ckptDeclineFor(bytes, reserve, 0, errors.New("x")); why != "free VRAM unreadable: x" {
		t.Errorf("unreadable: %q", why)
	}
}

// TestSlotsDecision holds the expert-cache decision's wrapper over capSlots: an unknown reading keeps the request; a reading that
// holds it changes nothing; one that holds fewer caps to capSlots' answer and logs it (naming the companion when one is reserved);
// one under top-k refuses naming the minimum.
func TestSlotsDecision(t *testing.T) {
	strides := []int64{1 << 20, 1 << 18, 1 << 20, 1 << 18}
	const nLayers, topK, want = 4, 2, 16
	if d := slotsDecision(0, false, 0, nLayers, strides, topK, want); d.slots != want || d.err != nil || len(d.logs) != 0 {
		t.Errorf("no reading: %+v", d)
	}
	plenty := uint64(1 << 34)
	if d := slotsDecision(plenty, true, 0, nLayers, strides, topK, want); d.slots != want || d.err != nil || len(d.logs) != 0 {
		t.Errorf("plenty: %+v", d)
	}
	mid := uint64(slotMarginBytes + slotRequirement(want, nLayers, strides)/2) // capSlots keeps slotMarginBytes back
	fit, _ := capSlots(int64(mid), nLayers, strides, topK, want)
	if d := slotsDecision(mid, true, 0, nLayers, strides, topK, want); d.slots != fit || fit >= want || len(d.logs) != 1 || d.err != nil {
		t.Errorf("half: %+v, want %d slots logged once", d, fit)
	}
	if d := slotsDecision(mid+1<<30, true, 1<<30, nLayers, strides, topK, want); d.slots != fit || len(d.logs) != 1 || !strings.Contains(d.logs[0], "companion") {
		t.Errorf("half beside a companion: %+v, want %d slots and the companion named", d, fit)
	}
	if d := slotsDecision(1, true, 0, nLayers, strides, topK, want); d.err == nil || !strings.Contains(d.err.Error(), "cannot fit its MINIMUM") {
		t.Errorf("under top-k: %+v, want the minimum's refusal", d)
	}
}

// TestPlanMemory_postAllocation plans each post-allocation site without a device: qwen3next-tiny's recurrent checkpoint, glm-tiny's
// expert cache, gemma2-tiny's windowed grant, and llama-tiny's prefill width; a short reading refuses or declines at that site.
func TestPlanMemory_postAllocation(t *testing.T) {
	const room = int64(8 << 30)
	full := memReadings{WeightsOpen: room, Plan: room, Weights: room, KV: room, Ckpt: room, GrantPre: room, Slots: room, Grant: room, Prefill: room,
		WeightsOpenOK: true, PlanOK: true, WeightsOK: true, KVOK: true, CkptOK: true, GrantPreOK: true, SlotsOK: true, GrantOK: true, PrefillOK: true}
	load := func(path string, o decoder.Options) *decoder.Model {
		t.Helper()
		o.Backend, o.Quant = "cpu", "int4"
		m, err := decoder.Load(path, o)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	b := &cudaBackend{}
	plan := func(m *decoder.Model, in memReadings) memPlan {
		t.Helper()
		p, err := b.planMemory(m, in)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}

	rec := load("../testdata/qwen3next-tiny", decoder.Options{})
	if p := plan(rec, full); p.CkptBytes <= 0 || p.CkptDeclined != "" {
		t.Errorf("qwen3next-tiny, room: %+v, want a checkpoint kept", p)
	}
	short := full
	short.Ckpt = 1
	if p := plan(rec, short); p.CkptBytes != 0 || !strings.Contains(p.CkptDeclined, "needed beside") {
		t.Errorf("qwen3next-tiny, short checkpoint reading: %+v, want none kept", p)
	}

	moe := load("../testdata/glm-tiny", decoder.Options{MoECacheExperts: true})
	if p := plan(moe, full); p.CacheSlots <= 0 || p.Decline != "" {
		t.Errorf("glm-tiny with the expert cache, room: %+v, want slots", p)
	}
	short = full
	short.Slots = 1
	if p := plan(moe, short); p.DeclineAt != "slots" {
		t.Errorf("glm-tiny with the expert cache, short slots reading: %+v, want a refusal at slots", p)
	}

	// Gemma 4's MoE block is cached through its own flag (g4moe), not isMoE: the plan must count its layers as the build does.
	g4 := load("../testdata/gemma4-moe-kv-tiny32", decoder.Options{MoECacheExperts: true})
	if p := plan(g4, short); p.DeclineAt != "slots" {
		t.Errorf("gemma4-moe-kv-tiny32 with the expert cache, short slots reading: %+v, want a refusal at slots", p)
	}

	win := load("../testdata/gemma2-tiny", decoder.Options{ResidentWindowedKV: true})
	if p := plan(win, full); p.KVSlack <= kvWindowSlack || p.PrefillWidth != prefillDefaultChunk {
		t.Errorf("gemma2-tiny windowed, room: %+v, want a grant past the slack %d and the default width", p, kvWindowSlack)
	}

	dense := load("../testdata/llama-tiny", decoder.Options{})
	if p := plan(dense, full); p.PrefillWidth <= prefillDefaultChunk {
		t.Errorf("llama-tiny, room: %+v, want a width above %d", p, prefillDefaultChunk)
	}
	short = full
	short.Prefill = 1
	if p := plan(dense, short); p.PrefillWidth != prefillDefaultChunk {
		t.Errorf("llama-tiny, short prefill reading: %+v, want the default width", p)
	}
}

// TestPlanMemory_postAllocationPlants shows each plan-side plant changes its site's decision: the checkpoint kept where the reserve
// refuses it, no grant, the slot cap ignored, the default width.
func TestPlanMemory_postAllocationPlants(t *testing.T) {
	defer func(p string) { memPlanPlantForTest = p }(memPlanPlantForTest)
	const room = int64(8 << 30)
	full := memReadings{WeightsOpen: room, Plan: room, Weights: room, KV: room, Ckpt: room, GrantPre: room, Slots: room, Grant: room, Prefill: room,
		WeightsOpenOK: true, PlanOK: true, WeightsOK: true, KVOK: true, CkptOK: true, GrantPreOK: true, SlotsOK: true, GrantOK: true, PrefillOK: true}
	b := &cudaBackend{}
	for _, c := range []struct {
		site, path string
		o          decoder.Options
		in         func(memReadings) memReadings
		differs    func(clean, planted memPlan) bool
	}{
		{"ckpt", "../testdata/qwen3next-tiny", decoder.Options{}, func(r memReadings) memReadings { r.Ckpt = 1 << 20; return r },
			func(c, p memPlan) bool { return c.CkptBytes == 0 && p.CkptBytes > 0 }},
		{"grant", "../testdata/gemma2-tiny", decoder.Options{ResidentWindowedKV: true}, func(r memReadings) memReadings { return r },
			func(c, p memPlan) bool { return p.KVSlack < c.KVSlack }},
		{"slots", "../testdata/glm-tiny", decoder.Options{MoECacheExperts: true}, func(r memReadings) memReadings { r.Slots = 1; return r },
			func(c, p memPlan) bool { return c.DeclineAt == "slots" && p.DeclineAt != "slots" }},
		{"prefill", "../testdata/llama-tiny", decoder.Options{}, func(r memReadings) memReadings { return r },
			func(c, p memPlan) bool { return p.PrefillWidth < c.PrefillWidth }},
	} {
		t.Run(c.site, func(t *testing.T) {
			o := c.o
			o.Backend, o.Quant = "cpu", "int4"
			m, err := decoder.Load(c.path, o)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			memPlanPlantForTest = ""
			clean, err := b.planMemory(m, c.in(full))
			if err != nil {
				t.Fatal(err)
			}
			memPlanPlantForTest = c.site
			planted, err := b.planMemory(m, c.in(full))
			if err != nil {
				t.Fatal(err)
			}
			if !c.differs(clean, planted) {
				t.Errorf("the %s plant did not change the plan: clean %+v, planted %+v", c.site, clean, planted)
			}
		})
	}
}

// TestDeriveMemory is G-DM1: derive mode derives each reading from one figure and the plan's own terms. llama-tiny's KV reading is
// the figure less the dense weights and the planned slack; qwen3next-tiny pays the pool before its plan, so its plan and weights
// readings are the figure less the pool; glm-tiny's expert cache is sized from what the KV pools and post-pool allocations leave, and
// the prefill width from what the slots leave after that.
func TestDeriveMemory(t *testing.T) {
	const p, pool = int64(7 << 30), int64(138412032)
	b := &cudaBackend{}
	load := func(path string, o decoder.Options) *decoder.Model {
		t.Helper()
		o.Backend, o.Quant = "cpu", "int4"
		m, err := decoder.Load(path, o)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}

	dense := load("../testdata/llama-tiny", decoder.Options{})
	plan, in, err := b.deriveMemory(dense, p, pool)
	if err != nil {
		t.Fatal(err)
	}
	if want := p - dense.ResidentDenseWeightBytesFor("cuda") - dense.PlanSlack("cuda"); in.KV != want || plan.PoolPaid || in.Plan != p {
		t.Errorf("llama-tiny: KV %d (want %d), plan %d (want %d), pool paid %v", in.KV, want, in.Plan, p, plan.PoolPaid)
	}
	if in.Ckpt >= in.KV || in.Prefill != in.Ckpt {
		t.Errorf("llama-tiny: after the KV pools %d (KV %d), prefill %d: want less after the pools, and nothing else granted before the width", in.Ckpt, in.KV, in.Prefill)
	}
	if in.WeightsOpenOK || !in.WeightsOK {
		t.Errorf("llama-tiny: read weightsOpen %v weights %v; want only the setup job's weights check (not a MoE)", in.WeightsOpenOK, in.WeightsOK)
	}
	if in.CkptOK || in.GrantOK || !in.PrefillOK {
		t.Errorf("llama-tiny: read ckpt %v grant %v prefill %v; want only the width read, as the build reads (no recurrent state, not windowed)", in.CkptOK, in.GrantOK, in.PrefillOK)
	}

	moe := load("../testdata/qwen3next-tiny", decoder.Options{})
	plan, in, err = b.deriveMemory(moe, p, pool)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.PoolPaid || in.Plan != p-pool || in.Weights != p-pool || in.WeightsOpen != p {
		t.Errorf("qwen3next-tiny: pool paid %v, readings open %d plan %d weights %d, want %d, %d, %d", plan.PoolPaid, in.WeightsOpen, in.Plan, in.Weights, p, p-pool, p-pool)
	}
	if !in.CkptOK || in.GrantOK {
		t.Errorf("qwen3next-tiny: read ckpt %v grant %v; want the checkpoint's site read (recurrent state) and no grant (not windowed)", in.CkptOK, in.GrantOK)
	}
	if !in.WeightsOpenOK || !in.PlanOK {
		t.Errorf("qwen3next-tiny: read weightsOpen %v plan %v; want both (a MoE build whose context is planned)", in.WeightsOpenOK, in.PlanOK)
	}
	if plan.CkptBytes == 0 {
		t.Fatalf("qwen3next-tiny: no checkpoint kept (%q); the checks on it below would pass at nothing", plan.CkptDeclined)
	}
	if in.Prefill >= in.Ckpt {
		t.Errorf("qwen3next-tiny: a kept checkpoint left prefill's reading %d, not under the checkpoint's %d", in.Prefill, in.Ckpt)
	}

	cache := load("../testdata/glm-tiny", decoder.Options{MoECacheExperts: true})
	plan, in, err = b.deriveMemory(cache, p, pool)
	if err != nil {
		t.Fatal(err)
	}
	if in.PlanOK || in.WeightsOpenOK || !in.SlotsOK {
		t.Errorf("glm-tiny with the expert cache: read plan %v weightsOpen %v slots %v; want only the slots (the cache's context is not planned)", in.PlanOK, in.WeightsOpenOK, in.SlotsOK)
	}
	if plan.CacheSlots <= 0 || in.Slots != in.Ckpt || in.Prefill >= in.Slots {
		t.Errorf("glm-tiny with the expert cache: slots %d, readings ckpt %d slots %d prefill %d: want the slots sized from what the pools leave, and the width from what the slots leave", plan.CacheSlots, in.Ckpt, in.Slots, in.Prefill)
	}
}

// TestKVGrantFor holds the windowed grant at its boundaries on synthetic layers: nothing at or under the reserve, one step for exactly
// one step's positions over it, a step's worth less one byte rounds down to nothing, and a large figure stops at the maximum.
func TestKVGrantFor(t *testing.T) {
	r := &cudaResident{ctxCap: 8192, kvWindow: 512, kvSlack: kvWindowSlack, kvSlotsN: 1,
		layers: []cudaLayer{{idx: 0, kvDim: 256, kvWin: true, window: 512}, {idx: 1, kvDim: 256}}}
	_, perPos := r.kvGrantFor(0, 0)
	if perPos != 2*256*4 {
		t.Fatalf("perPos %d, want one windowed layer's K and V at f32", perPos)
	}
	reserve := r.extraBytes + grantReserveBytes + 2*r.prefillPassBytes(prefillDefaultChunk)
	maxExtra := min(kvWindowRollbackMax-r.kvRollback(), r.ctxCap-1-r.kvWindow-r.kvSlack)
	step := int64(kvWindowStep) * perPos
	for _, c := range []struct {
		name string
		free int64
		want int
	}{
		{"at the reserve", reserve, 0},
		{"one step over", reserve + step, kvWindowStep},
		{"a byte short of one step", reserve + step - 1, 0},
		{"plenty", reserve + 1<<30, maxExtra / kvWindowStep * kvWindowStep},
	} {
		if got, _ := r.kvGrantFor(0, uint64(c.free)); got != c.want {
			t.Errorf("%s: %d positions, want %d", c.name, got, c.want)
		}
	}
	if got, _ := r.kvGrantFor(step, uint64(reserve+step)); got != 0 {
		t.Errorf("one step over, held back one step: %d positions, want 0", got)
	}
}

// TestPlanMemory_grantsAndWidth reaches the sites the room-everywhere plans do not: gemma2-tiny's grant at no room and capped,
// gemma4-moe-kv-tiny32's grant before its expert cache (windowed KV with the cache), each plant silencing its own site only, and
// llama-tiny's width down the ladder as its reading shrinks.
func TestPlanMemory_grantsAndWidth(t *testing.T) {
	defer func(p string) { memPlanPlantForTest = p }(memPlanPlantForTest)
	const room = int64(8 << 30)
	full := memReadings{WeightsOpen: room, Plan: room, Weights: room, KV: room, Ckpt: room, GrantPre: room, Slots: room, Grant: room, Prefill: room,
		WeightsOpenOK: true, PlanOK: true, WeightsOK: true, KVOK: true, CkptOK: true, GrantPreOK: true, SlotsOK: true, GrantOK: true, PrefillOK: true}
	b := &cudaBackend{}
	load := func(path string, o decoder.Options) *decoder.Model {
		t.Helper()
		o.Backend, o.Quant = "cpu", "int4"
		m, err := decoder.Load(path, o)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	plan := func(m *decoder.Model, in memReadings) memPlan {
		t.Helper()
		p, err := b.planMemory(m, in)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}

	win := load("../testdata/gemma2-tiny", decoder.Options{ResidentWindowedKV: true})
	most := plan(win, full).KVSlack
	none := full
	none.Grant = 1
	if p := plan(win, none); p.KVSlack != kvWindowSlack {
		t.Errorf("gemma2-tiny, no room: slack %d, want the floor %d", p.KVSlack, kvWindowSlack)
	}
	// The reserve is a few hundred MiB plus two passes; a figure 1 MiB over it grants less than the room-everywhere maximum.
	capped := full
	capped.Grant = grantReserveBytes + 4<<20
	if p := plan(win, capped); p.KVSlack <= kvWindowSlack || p.KVSlack >= most {
		t.Errorf("gemma2-tiny, capped: slack %d, want between %d and %d", p.KVSlack, kvWindowSlack, most)
	}

	both := load("../testdata/gemma4-moe-kv-tiny32", decoder.Options{MoECacheExperts: true, ResidentWindowedKV: true})
	if p := plan(both, full); p.KVSlack <= kvWindowSlack {
		t.Errorf("gemma4-moe-kv-tiny32 with the cache and windowed KV, room: slack %d, want a grant before the cache", p.KVSlack)
	}
	none = full
	none.GrantPre = 1
	if p := plan(both, none); p.KVSlack != kvWindowSlack {
		t.Errorf("gemma4-moe-kv-tiny32, no room before the cache: slack %d, want %d", p.KVSlack, kvWindowSlack)
	}
	memPlanPlantForTest = "grant"
	if p := plan(both, full); p.KVSlack <= kvWindowSlack {
		t.Errorf("the grant plant silenced the grant before the cache (slack %d): plants match their site exactly", p.KVSlack)
	}
	memPlanPlantForTest = "grantPre"
	if p := plan(both, full); p.KVSlack != kvWindowSlack {
		t.Errorf("the grantPre plant left the grant before the cache in place (slack %d)", p.KVSlack)
	}
	memPlanPlantForTest = ""

	dense := load("../testdata/llama-tiny", decoder.Options{})
	prev := plan(dense, full).PrefillWidth
	seen := map[int]bool{prev: true}
	for f := int64(1 << 30); f > 0; f -= 1 << 20 {
		in := full
		in.Prefill = f
		p := plan(dense, in)
		if p.PrefillWidth > prev || p.PrefillWidth < prefillDefaultChunk {
			t.Errorf("llama-tiny at %d B: width %d after %d, want no wider as the reading shrinks, and at least %d", f, p.PrefillWidth, prev, prefillDefaultChunk)
		}
		prev, seen[p.PrefillWidth] = p.PrefillWidth, true
	}
	in := full
	in.Prefill = 1
	prev = plan(dense, in).PrefillWidth
	for _, w := range []int{2048, 1024, prefillDefaultChunk} {
		if !seen[w] {
			t.Errorf("llama-tiny: the width never took the %d-row rung as the reading shrank 1 GiB to 1 MiB (took %v)", w, seen)
		}
	}
	if prev != prefillDefaultChunk {
		t.Errorf("llama-tiny at 1 B: width %d, want the default %d", prev, prefillDefaultChunk)
	}
}

// TestPlanSlotGeomFor_degenerate: with no expert count or no per-expert stride the plan caches nothing, as the build's expertSlotGeom
// returns no layers when the first routed layer's slot bytes are zero.
func TestPlanSlotGeomFor_degenerate(t *testing.T) {
	hls := []hlayer{{isMoE: true}, {isMoE: true}}
	if g, _ := planSlotGeomFor(hls, 0); g.n != 0 {
		t.Errorf("no expert count: %d layers, want none", g.n)
	}
	if g, _ := planSlotGeomFor(hls, 4); g.n != 0 {
		t.Errorf("zero-size stacks: %d layers, want none", g.n)
	}
}

// TestPlanMemory_kvRaise: with more than one KV slot and no context asked for, the build lets one conversation's cap grow to the whole
// pool, up to the model's window (kvCapFor); the plan reports the same cap, and the pool line the build logs.
func TestPlanMemory_kvRaise(t *testing.T) {
	r := &cudaResident{kvRaise: true, kvSlotsN: 2, ctxCap: 4096, modelCtx: 32768}
	mp := &memPlanRun{}
	r.postAllocPlan(nil, mp)
	want := 2 * kvBlocksFor(4096) * kvBlock
	if mp.out.CtxCap != want {
		t.Errorf("two slots of 4096: the plan's cap %d, want the pool's %d", mp.out.CtxCap, want)
	}
	if !strings.Contains(strings.Join(mp.out.Logs, ""), "one conversation can use up to") {
		t.Errorf("no pool line among the plan's logs: %q", mp.out.Logs)
	}
	r = &cudaResident{kvRaise: false, kvSlotsN: 2, ctxCap: 4096, modelCtx: 32768}
	mp = &memPlanRun{}
	r.postAllocPlan(nil, mp)
	if mp.out.CtxCap != 4096 {
		t.Errorf("a context asked for (no raise): the plan's cap %d, want 4096", mp.out.CtxCap)
	}
}

// TestDeriveMemory_raiseScoreScratch: when one conversation's cap is raised past the per-slot context, the build reallocates the
// split-KV score scratch at the raised cap (kvAfterRaise); derive mode takes off its growth, as charged, after the raise.
func TestDeriveMemory_raiseScoreScratch(t *testing.T) {
	defer func(p string) { memPlanPlantForTest = p }(memPlanPlantForTest)
	memPlanPlantForTest = "prefill" // the reading is derived; the width's choice needs a whole resident
	const free = int64(6 << 30)
	for _, slots := range []int{1, 2} {
		r := &cudaResident{kvRaise: true, kvSlotsN: slots, ctxCap: 10752, modelCtx: 32768, nH: 28, layers: []cudaLayer{{idx: 0, kvDim: 512}}}
		mp := &memPlanRun{derive: true, in: memReadings{KV: free, KVOK: true}}
		r.postAllocPlan(nil, mp)
		share, raised := r.kvShareCtx, mp.out.CtxCap
		if (slots > 1) != (raised > share) {
			t.Fatalf("%d slots: cap %d over a per-slot %d; want a raise exactly when there is more than one slot", slots, raised, share)
		}
		grow := chargedBytes(int64(28*raised)*4) - chargedBytes(int64(28*share)*4)
		if want := free - kvChargedBytes(share, slots, r.layers) - r.postPoolBytes() - grow; !mp.in.PrefillOK || mp.in.Prefill != want {
			t.Errorf("%d slots (%d -> %d): prefill reading %d, want %d (the pools, the post-pool bytes and the score scratch's growth, %d B)",
				slots, share, raised, mp.in.Prefill, want, grow)
		}
	}
}

// TestDeriveMemory_grantCharged: in derive mode the windowed grant before an expert cache takes off what the card charges for the
// window pools' growth (each K and V pool reallocated at window+slack+extra positions, as kvReallocWindowed sizes it), not extra
// positions at their raw cost; the expert cache's reading follows it.
func TestDeriveMemory_grantCharged(t *testing.T) {
	// Two windowed layers of a 26B-like width (kvDim 4096, f32), one sharing the first's K/V, and a routed layer for the cache.
	r := &cudaResident{ctxCap: 27019, kvWindow: 1024, kvSlack: kvWindowSlack, kvSlotsN: 1, kvWin: true, cacheExperts: true,
		nE: 8, topK: 2, cacheSlots: 4,
		layers: []cudaLayer{{idx: 0, kvDim: 4096, kvWin: true, window: 1024}, {idx: 1, kvDim: 4096, kvWin: true, window: 1024},
			{idx: 2, kvDim: 4096, kvWin: true, window: 1024, kvShared: true, kvSrc: 0}, {idx: 3, kvDim: 2048}}}
	hls := []hlayer{{}, {}, {}, {isMoE: true, expGU: hostW{N: 8 * 1024, K: 2048}, expDown: hostW{N: 8 * 2048, K: 1024}}}
	pool := func(slack int) int64 {
		old := r.kvSlack
		r.kvSlack = slack
		blocks := r.kvPoolBlocks(kvGroupWin, 1)
		r.kvSlack = old
		return 2 * 2 * chargedBytes(int64(blocks*kvBlock*4096)*4) // K and V of the two owning windowed layers, f32
	}
	const free = int64(6 << 30)
	mp := &memPlanRun{derive: true}
	mp.in = memReadings{KV: free, KVOK: true}
	r.postAllocPlan(hls, mp)
	extra := mp.out.KVSlack - kvWindowSlack
	if extra <= 0 || !mp.in.SlotsOK || !mp.in.GrantPreOK {
		t.Fatalf("no grant before the cache, or its readings not taken: slack %d, readings %+v", mp.out.KVSlack, mp.in)
	}
	if want := free - kvChargedBytes(r.kvShareCtx, 1, r.layers) - r.postPoolBytes(); mp.in.Ckpt != want || mp.in.GrantPre != want {
		t.Errorf("after the KV pools: checkpoint and grant readings %d and %d, want %d (the pools as charged and the post-pool bytes)", mp.in.Ckpt, mp.in.GrantPre, want)
	}
	if want := mp.in.GrantPre - (pool(kvWindowSlack+extra) - pool(kvWindowSlack)); mp.in.Slots != want {
		t.Errorf("after a %d-position grant: the cache's reading %d, want %d (the grant at its charged cost, %d B; taken off was %d B)",
			extra, mp.in.Slots, want, mp.in.GrantPre-want, mp.in.GrantPre-mp.in.Slots)
	}
}

// TestPlanMemory_degenerateExpertCache: a cached load whose routed layer has no per-expert bytes turns the cache off, logs it as
// allocSlots does, and keeps its slot request; the grant then runs as for a load without the cache.
func TestPlanMemory_degenerateExpertCache(t *testing.T) {
	r := &cudaResident{cacheExperts: true, nE: 4, cacheSlots: 8, kvSlotsN: 1, ctxCap: 4096}
	mp := &memPlanRun{}
	r.postAllocPlan([]hlayer{{}, {isMoE: true}, {isMoE: true}}, mp)
	if r.cacheExperts || mp.out.CacheSlots != 8 || mp.out.Decline != "" {
		t.Errorf("degenerate cache: cacheExperts %v, slots %d, decline %q; want the cache off, the request kept, no refusal", r.cacheExperts, mp.out.CacheSlots, mp.out.Decline)
	}
	if !slices.Contains(mp.out.Logs, zeroExpertBytesLine(1)) {
		t.Errorf("degenerate cache: logs %q, want %q", mp.out.Logs, zeroExpertBytesLine(1))
	}
	// Derived, the slots site is not read: allocSlots returns before its reading when no routed layer has per-expert bytes.
	defer func(p string) { memPlanPlantForTest = p }(memPlanPlantForTest)
	memPlanPlantForTest = "prefill"
	r = &cudaResident{cacheExperts: true, nE: 4, cacheSlots: 8, kvSlotsN: 1, ctxCap: 4096}
	mp = &memPlanRun{derive: true, in: memReadings{KV: 6 << 30, KVOK: true}}
	r.postAllocPlan([]hlayer{{}, {isMoE: true}, {isMoE: true}}, mp)
	if mp.in.SlotsOK {
		t.Errorf("degenerate cache, derived: the slots site marked read; allocSlots takes no reading there")
	}
}

// TestDeriveMemory_windowed derives gemma2-tiny with windowed KV: the grant is read and made; neither the width (a windowed load keeps
// the default, unread) nor an expert-cache site is read.
func TestDeriveMemory_windowed(t *testing.T) {
	m, err := decoder.Load("../testdata/gemma2-tiny", decoder.Options{Backend: "cpu", Quant: "int4", ResidentWindowedKV: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	plan, in, err := (&cudaBackend{}).deriveMemory(m, 7<<30, 138412032)
	if err != nil {
		t.Fatal(err)
	}
	if !in.GrantOK || in.PrefillOK || in.GrantPreOK || in.SlotsOK || plan.KVSlack <= kvWindowSlack || plan.PrefillWidth != prefillDefaultChunk {
		t.Errorf("gemma2-tiny windowed: %+v %+v; want the grant read and made, and no width or expert-cache site read (a windowed load keeps the default width)", plan, in)
	}
}

// TestDeriveMemory_poolInSetup: a MoE build whose context is not planned (the expert cache) pays moe_route's pool in the setup job,
// just before the KV fit (payRoutePool), so the derived KV reading is the figure less the dense weights, the slack and the pool.
func TestDeriveMemory_poolInSetup(t *testing.T) {
	const p, pool = int64(7 << 30), int64(138412032)
	m, err := decoder.Load("../testdata/glm-tiny", decoder.Options{Backend: "cpu", Quant: "int4", MoECacheExperts: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	plan, in, err := (&cudaBackend{}).deriveMemory(m, p, pool)
	if err != nil {
		t.Fatal(err)
	}
	if want := p - m.ResidentDenseWeightBytesFor("cuda") - m.PlanSlack("cuda") - pool; plan.PoolPaid || in.KV != want {
		t.Errorf("glm-tiny with the expert cache: pool paid before the plan %v, KV reading %d, want %d (the pool paid in the setup job)", plan.PoolPaid, in.KV, want)
	}
}
