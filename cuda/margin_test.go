//go:build cuda

package cuda

import (
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
