//go:build gpu

package gpu

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// attnRef64 is TestAttnKeys_parity's float64 reference with a window start: each query head's softmax over the keys
// from start, and the weighted sum of their values.
func attnRef64(q, keys, vals []float32, nH, nKV, hd, nKeys, start int, scale float32) []float32 {
	kvDim, group := nKV*hd, nH/nKV
	ref := make([]float32, nH*hd)
	for qh := range nH {
		kvh := qh / group
		maxS := math.Inf(-1)
		sc := make([]float64, nKeys)
		for s := start; s < nKeys; s++ {
			var dot float64
			for d := range hd {
				dot += float64(q[qh*hd+d]) * float64(keys[s*kvDim+kvh*hd+d])
			}
			sc[s] = dot * float64(scale)
			maxS = math.Max(maxS, sc[s])
		}
		var sum float64
		for s := start; s < nKeys; s++ {
			sc[s] = math.Exp(sc[s] - maxS)
			sum += sc[s]
		}
		for s := start; s < nKeys; s++ {
			w := sc[s] / sum
			for d := range hd {
				ref[qh*hd+d] += float32(w * float64(vals[s*kvDim+kvh*hd+d]))
			}
		}
	}
	return ref
}

// G-S12.4c-1 (docs/tasks/task-multimodal-support-2026-10.md): the wide key-split attention kernel against the float64
// reference and against the per-dimension wide kernel it replaces, at Gemma's geometry (8 heads of 256 over 4 KV
// heads) and at a head of 192, at key counts on both sides of the lane count and of a tile, from key 0 and from a
// window start. Then the planted defects, each of which must fail the same bars.
func TestAttnKeysWide_parity(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()
	if err := ctx.ensureAttn(); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ensureAttnWide(); err != nil {
		t.Fatal(err)
	}
	type shape struct{ nH, nKV, hd int }
	// grade reads one pipeline against the two references and returns "" when it holds both bars.
	grade := func(pl tiledPipes, g shape, nKeys, start int) string {
		kvDim := g.nKV * g.hd
		scale := float32(1.0 / math.Sqrt(float64(g.hd)))
		q, keys, vals := randMat(g.nH*g.hd, 1), randMat(nKeys*kvDim, 2), randMat(nKeys*kvDim, 3)
		got, err := ctx.attentionOn(pl.pl, pl.layout, q, keys, vals, g.nH, g.nKV, g.hd, nKeys, start, scale)
		if err != nil {
			t.Fatalf("%+v nKeys=%d start=%d: %v", g, nKeys, start, err)
		}
		for i, v := range got {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Sprintf("a non-finite output at %d", i)
			}
		}
		if cos, maxAbs := cosine(got, attnRef64(q, keys, vals, g.nH, g.nKV, g.hd, nKeys, start, scale)); !(cos >= 0.999999) || !(maxAbs <= 1e-4) {
			return "against the float64 reference: cosine and max |diff| outside 0.999999 and 1e-4"
		}
		old, err := ctx.attentionOn(ctx.attnWidePipeline, ctx.attnWideLayout, q, keys, vals, g.nH, g.nKV, g.hd, nKeys, start, scale)
		if err != nil {
			t.Fatal(err)
		}
		if cos, maxAbs := cosine(got, old); !(cos >= 0.9999999) || !(maxAbs <= 1e-5) {
			return "against the per-dimension wide kernel: cosine and max |diff| outside 0.9999999 and 1e-5"
		}
		return ""
	}
	shapes := []shape{{8, 4, 256}, {4, 2, 192}}
	type cut struct{ nKeys, start int }
	cuts := []cut{{1, 0}, {2, 0}, {40, 0}, {127, 0}, {128, 0}, {129, 0}, {300, 0}, {1500, 476}, {attnKeysTile - 1, 0}, {attnKeysTile, 0}, {attnKeysTile + 1, 0}, {2*attnKeysTile + 37, 0}, {2*attnKeysTile + 37, attnKeysTile + 5}}
	for _, g := range shapes {
		if !attnKeysWideEligible(g.hd, g.nKV*g.hd, false) {
			t.Fatalf("attnKeysWideEligible refuses %+v", g)
		}
		for _, c := range cuts {
			if why := grade(ctx.attnKeysWide, g, c.nKeys, c.start); why != "" {
				t.Errorf("%+v nKeys=%d start=%d: %s", g, c.nKeys, c.start, why)
			}
		}
	}
	src, err := attnKeysWideSource()
	if err != nil {
		t.Fatal(err)
	}
	for name, edit := range map[string][2]string{
		"a lane's second dimension dropped": {"for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n        let d = t + j * 128u;\n        if (d < hd) { ctx[", "for (var j: u32 = 0u; j < 1u; j = j + 1u) {\n        let d = t + j * 128u;\n        if (d < hd) { ctx["},
		"a lane's two dimensions swapped":   {"if (d < hd) { ctx[qh * hd + d] = acc[j] / l; }", "if (d < hd) { ctx[qh * hd + d] = acc[1u - j] / l; }"},
	} {
		if strings.Count(src, edit[0]) != 1 {
			t.Fatalf("%s: the text to edit appears %d times", name, strings.Count(src, edit[0]))
		}
		sh, pl, ly, err := ctx.mkPipeline("attn-keys-wide-defect", strings.Replace(src, edit[0], edit[1], 1))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if why := grade(tiledPipes{sh: sh, pl: pl, layout: ly}, shapes[0], 300, 0); why == "" {
			t.Errorf("%s: the gate stayed green", name)
		} else {
			t.Logf("%s: red (%s)", name, why)
		}
	}
}

// G-S12.4c-2: a head of 128 or under gets the pipeline it got, and a wide head on a float32 cache gets the wide
// key-split kernel; f16 caches keep theirs.
func TestAttnKernel_choice(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()
	if err := ctx.ensureAttn(); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ensureAttnWide(); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name      string
		hd, kvDim int
		f16       bool
		want      any
	}{
		{"a head of 128, f32", 128, 256, false, ctx.attnKeysPipeline},
		{"a head of 64, f32", 64, 128, false, ctx.attnKeysPipeline},
		{"a head of 48 over an odd cache width, f32", 48, 50, false, ctx.attnPipeline},
		{"a head of 128, f16", 128, 256, true, ctx.attnF16Pipeline},
		{"a head of 256, f32", 256, 1024, false, ctx.attnKeysWide.pl},
		{"a head of 192, f32", 192, 384, false, ctx.attnKeysWide.pl},
		{"a head of 256, f16", 256, 1024, true, ctx.attnF16WidePipeline},
		{"a head of 320, f32", 320, 640, false, ctx.attnWidePipeline},
	} {
		if got, _ := ctx.attnKernel(c.hd, c.kvDim, c.f16); any(got) != c.want {
			t.Errorf("%s: attnKernel picked another pipeline", c.name)
		}
	}
}
