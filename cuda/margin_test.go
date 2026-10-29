//go:build cuda

package cuda

import (
	"errors"
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
	if len(hooks) != 0 {
		t.Errorf("the memory plan fired reading hooks %v: they record the real build's readings only", hooks)
	}
}
