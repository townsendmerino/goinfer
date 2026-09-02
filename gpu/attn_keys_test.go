//go:build gpu

package gpu

import (
	"math"
	"testing"
)

// TestAttnKeys_parity gates the key-split decode attention kernel against the same f64 CPU
// reference TestAttention_parity uses, and against the dim-split kernel it replaces.
//
// The depths are chosen around the TILE boundary on purpose. attnKeys carries online-softmax
// state (m, l, acc) ACROSS tiles because WGSL has no dynamic workgroup storage, and that
// carry is the only genuinely new arithmetic in the kernel — a test that only ran nKeys < TILE
// would exercise the single-tile path, never the rescale, and pass on a kernel that is broken
// for every real long context. nKeys=1 is included for the opposite reason and is NOT
// sufficient on its own: softmax over one element is 1.0 at any scale, so it cannot see a
// wrong max or a wrong denominator (the repro-minimality trap in CLAUDE.md).
func TestAttnKeys_parity(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()
	if err := ctx.ensureAttn(); err != nil {
		t.Fatalf("ensureAttn: %v", err)
	}

	const nH, nKV, hd = 12, 2, 128
	kvDim := nKV * hd
	group := nH / nKV
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	if !attnKeysEligible(hd, kvDim, false, false) {
		t.Fatalf("attnKeysEligible(%d,%d) = false — this geometry is the shipped 1.5B's; the guard is wrong", hd, kvDim)
	}

	for _, nKeys := range []int{1, 2, 40, 127, 128, 129, attnKeysTile - 1, attnKeysTile, attnKeysTile + 1, 2*attnKeysTile + 37, 5000} {
		q := randMat(nH*hd, 1)
		keys := randMat(nKeys*kvDim, 2)
		vals := randMat(nKeys*kvDim, 3)

		// f64 CPU reference (mirrors attendQuery), the same one TestAttention_parity uses.
		ref := make([]float32, nH*hd)
		for qh := range nH {
			kvh := qh / group
			maxS := math.Inf(-1)
			sc := make([]float64, nKeys)
			for s := range nKeys {
				var dot float64
				for d := range hd {
					dot += float64(q[qh*hd+d]) * float64(keys[s*kvDim+kvh*hd+d])
				}
				sc[s] = dot * float64(scale)
				if sc[s] > maxS {
					maxS = sc[s]
				}
			}
			var sum float64
			for s := range nKeys {
				sc[s] = math.Exp(sc[s] - maxS)
				sum += sc[s]
			}
			for s := range nKeys {
				w := sc[s] / sum
				for d := range hd {
					ref[qh*hd+d] += float32(w * float64(vals[s*kvDim+kvh*hd+d]))
				}
			}
		}

		got, err := ctx.attentionOn(ctx.attnKeysPipeline, ctx.attnKeysLayout, q, keys, vals, nH, nKV, hd, nKeys, 0, scale)
		if err != nil {
			t.Fatalf("nKeys=%d: attnKeys: %v", nKeys, err)
		}
		for i, v := range got {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("nKeys=%d: non-finite at %d: %v", nKeys, i, v)
			}
		}
		cos, maxAbs := cosine(got, ref)
		tiles := (nKeys + attnKeysTile - 1) / attnKeysTile
		t.Logf("nKeys=%5d (%d tile(s)): cosine=%.8f maxAbs=%.3e", nKeys, tiles, cos, maxAbs)
		if cos < 0.999999 || maxAbs > 1e-4 {
			t.Errorf("nKeys=%d: key-split attention diverges from the f64 reference: cosine=%.8f maxAbs=%.3e", nKeys, cos, maxAbs)
		}

		// And against the kernel it replaces, on the same inputs — the two must agree far more
		// tightly than either agrees with f64, since they differ only in reduction order.
		old, err := ctx.Attention(q, keys, vals, nH, nKV, hd, nKeys, 0, scale)
		if err != nil {
			t.Fatalf("nKeys=%d: attn (dim-split): %v", nKeys, err)
		}
		cosO, maxAbsO := cosine(got, old)
		if cosO < 0.9999999 || maxAbsO > 1e-5 {
			t.Errorf("nKeys=%d: key-split disagrees with dim-split: cosine=%.9f maxAbs=%.3e", nKeys, cosO, maxAbsO)
		}
	}
}
