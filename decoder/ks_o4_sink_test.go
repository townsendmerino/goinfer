package decoder

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// KS-O4's G-SINK (docs/measurements/ks-o4-2026-10/README.md): the batched acc64 attention's sink term. (a) With every sink at −∞ the
// path equals the same path with no sinks, word for word, so the term adds only to the max-shift and the denominator. (b) With finite
// sinks it agrees with attendSinksToken, today's gpt-oss attention, to float32 rounding: the two differ only in AV's accumulation and
// in the score's rounding before the scale.

type ksO4Shape struct{ nH, nKV, hd int }

// The shapes of KS-F2's G0: hd 64/80/128, GQA groups 1/2/8 (gpt-oss is 64 heads over 8), and key counts that are not multiples of 8.
var ksO4Shapes = []ksO4Shape{{8, 8, 64}, {8, 4, 80}, {16, 2, 128}, {64, 8, 64}}
var ksO4Keys = []int{1, 7, 8, 9, 15, 64, 129, 300}

func ksO4Floats(r *rand.Rand, n int, amp float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((r.Float64()*2 - 1) * amp)
	}
	return out
}

func ksO4SameBits(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d words, want %d", what, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: word %d is %v, want %v", what, i, got[i], want[i])
		}
	}
}

func ksO4Batched(t *testing.T, arch *Architecture, sh ksO4Shape, p attentionParams, cache *KVCache, q []float32, pos int) []float32 {
	t.Helper()
	ctx := make([]float32, sh.nH*sh.hd)
	nKeys := len(cache.Keys(0)) / (sh.nKV * sh.hd)
	scr := newDecodeScratch(arch)
	pool := scr.headWorkerPool(sh.nH, 1, nKeys, sh.hd, false, true)
	attendBatchedHeadsP(q, ctx, cache.Keys(0), cache.Vals(0), 0, cache, 0, pos, 1, p, arch, true, pool)
	return ctx
}

func TestKSO4_sinkTerm(t *testing.T) {
	cells, close := 0, 0
	for _, sh := range ksO4Shapes {
		for _, nKeys := range ksO4Keys {
			for _, window := range []int{0, 40} {
				name := fmt.Sprintf("%+v, %d keys, window %d", sh, nKeys, window)
				r := rand.New(rand.NewSource(int64(nKeys*31 + sh.hd + window)))
				arch := &Architecture{NumHeads: sh.nH, NumKVHeads: sh.nKV, HeadDim: sh.hd}
				kvDim := sh.nKV * sh.hd
				cache := NewKVCache(1, sh.nKV, sh.hd, window, nKeys, nil)
				ref := NewKVCache(1, sh.nKV, sh.hd, window, nKeys, nil)
				for range nKeys - 1 {
					k, v := ksO4Floats(r, kvDim, 2), ksO4Floats(r, kvDim, 2)
					cache.Append(0, k, v)
					ref.Append(0, k, v)
				}
				q, k, v := ksO4Floats(r, sh.nH*sh.hd, 2), ksO4Floats(r, kvDim, 2), ksO4Floats(r, kvDim, 2)
				cache.Append(0, k, v)
				p := attentionParams{Window: window, Scale: 1 / math.Sqrt(float64(sh.hd)), Order: attnBatchedAcc64}

				// (a) a sink of −∞ changes nothing.
				none := ksO4Batched(t, arch, sh, p, cache, q, nKeys-1)
				inf := make([]float32, sh.nH)
				for i := range inf {
					inf[i] = float32(math.Inf(-1))
				}
				pInf := p
				pInf.Sinks = inf
				ksO4SameBits(t, name+": a sink of -inf", ksO4Batched(t, arch, sh, pInf, cache, q, nKeys-1), none)

				// (b) finite sinks: today's path, to float32 rounding.
				sinks := ksO4Floats(r, sh.nH, 3)
				pS := p
				pS.Sinks = sinks
				got := ksO4Batched(t, arch, sh, pS, cache, q, nKeys-1)
				want := make([]float32, sh.nH*sh.hd)
				attendSinksToken(attentionParams{Window: window, Scale: p.Scale}, q, k, v, want, ref, arch, &LayerWeights{AttnSinks: sinks}, 0, nKeys-1)
				var maxAbs, maxDiff float64
				for i := range want {
					maxAbs = math.Max(maxAbs, math.Abs(float64(want[i])))
					maxDiff = math.Max(maxDiff, math.Abs(float64(got[i])-float64(want[i])))
				}
				if maxDiff > 1e-5*maxAbs {
					t.Fatalf("%s: max |diff| %g against today's path, over 1e-5 of the context's largest magnitude %g", name, maxDiff, maxAbs)
				}
				if maxDiff == 0 {
					close++
				}
				cells++
			}
		}
	}
	t.Logf("%d cells: (a) every word equal at a sink of -inf; (b) within 1e-5 of today's path (%d of them exactly equal)", cells, close)
}
