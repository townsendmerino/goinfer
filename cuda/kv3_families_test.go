//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// The row-batched exact attention (prefill_splitkv.cu) on the families the first KV3 gates did not name: Gemma 4 dense (head dims 256 and 512, K = V
// on the global layers, a sliding window on the others), a Gated-DeltaNet hybrid (Qwen3.5: attention on one layer in four, the rest recurrent), and the
// image-block twin on Gemma 3 VL. Each gate holds the row-batched route to attn_batched (or attn_img_batched) bit for bit, and each names its plant.
//
//	go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run '^TestKV3Families_' -v ./cuda/

// kv3FamilyFixtures are the fixtures of the forced-route gate: the dense Gemma 4 geometries (head dim 16 and 256 with a global head dim of 512,
// K = V on the global layers, sliding-window layers between) and a Qwen3.5 hybrid. All are read with an f32 cache.
var kv3FamilyFixtures = []string{"gemma4-dense-twogeom-tiny", "gemma4-dense-scaled", "qwen35-tiny"}

// TestKV3Families_forcedRouteEqualsAttnBatched: a whole batched prefill below the shared-memory bound, with every exact-attention layer forced onto the
// row-batched kernels, leaves the logits and every K and V row attn_batched leaves, bit for bit, on each family. Two lever settings, as the dense gate
// has: both off, and the tensor-core GEMM alone. The prompts span several 512-row passes.
//
// Plant: the unforced arm launches no row-batched group and the forced arm launches some, so the two arms are different routes; and a case whose reference
// is mostly NaN or zero is refused, so a failed write cannot compare equal to a failed write.
func TestKV3Families_forcedRouteEqualsAttnBatched(t *testing.T) {
	defer func() { prefillSplitKVForceForTest = false }()
	for _, name := range kv3FamilyFixtures {
		base := "../testdata/" + name
		t.Run(name, func(t *testing.T) {
			requireDeviceAndFixture(t, base)
			g := kv0LoadCtxAny(t, kv3LongContextCopy(t, base, 32768), "f32", 4096, kv0RandomTokens) // a copy whose window admits the prompts
			if !g.r.prefillSplitKVReady() {
				t.Fatal("the row-batched kernels did not load")
			}
			for _, gemm := range []bool{false, true} {
				g.r.fastAttn, g.r.fastGemm = false, gemm
				for _, n := range []int{300, 1000, 2600} {
					label := fmt.Sprintf("gemm=%v n=%d", gemm, n)
					e := g.embs(g.tokens(n, n))
					arm := func(force bool) ([]float32, kv0KV, int) {
						prefillSplitKVForceForTest = force
						g.poison(n)
						p0 := g.r.psLaunches
						lg := g.prefill(e, 0)
						return lg, g.snapshot(n), g.r.psLaunches - p0
					}
					want, wantKV, wantPS := arm(false)
					got, gotKV, gotPS := arm(true)
					kv3NoNaN(t, label, want, wantKV)
					if wantPS != 0 || gotPS == 0 {
						t.Fatalf("%s: row-batched launch groups: %d unforced (want 0), %d forced (want some)", label, wantPS, gotPS)
					}
					ld, _ := kv0LogitsDiff(want, got)
					kd, total := kv0CompareRows(wantKV, gotKV, 0, n)
					fmt.Fprintf(os.Stderr, "[kv3-families] %-26s %-24s logits differing %d of %d, KV differing %d of %d (%d groups)\n", name, label, ld, len(want), kd, total, gotPS)
					if ld != 0 || kd != 0 {
						t.Errorf("%s: the row-batched route differs from attn_batched: %d logits, %d KV elements", label, ld, kd)
					}
				}
			}
		})
	}
}

// TestKV3Families_pastBoundBatchedEqualsPerToken: past the 12,160-key bound a prompt prefills in batched passes on the row-batched kernels (it declined before
// them) and leaves the logits and the K and V rows of the per-token path, bit for bit, on the Gemma 4 dense geometry and on the Qwen3.5 hybrid (whose
// recurrent layers run beside the attention layers the route serves). Negative arm: with the row-batched module absent, the same prompt declines by name.
func TestKV3Families_pastBoundBatchedEqualsPerToken(t *testing.T) {
	for _, name := range []string{"gemma4-dense-twogeom-tiny", "qwen35-tiny"} {
		base := "../testdata/" + name
		t.Run(name, func(t *testing.T) {
			requireDeviceAndFixture(t, base)
			dir := kv3LongContextCopy(t, base, 32768)
			g := kv0LoadCtxAny(t, dir, "f32", 13000, kv0RandomTokens)
			g.r.fastAttn, g.r.fastGemm = false, false
			const n = 12400
			e := g.embs(g.tokens(n, 3*n))
			g.poison(n)
			p0 := g.r.psLaunches
			t0 := time.Now()
			got := g.prefill(e, 0) // a decline is a failure here
			tb := time.Since(t0)
			groups := g.r.psLaunches - p0
			gotKV := g.snapshot(n)
			if groups == 0 {
				t.Fatalf("%s: no row-batched launch group past the bound: the route did not serve this prompt", name)
			}
			g.poison(n)
			t0, last := time.Now(), time.Now()
			for i := 0; i < n-1; i++ {
				if err := g.r.ForwardNoLogits(e[i], i); err != nil {
					t.Fatalf("per-token row %d: %v", i, err)
				}
				if time.Since(last) > 30*time.Second {
					fmt.Fprintf(os.Stderr, "[kv3-families] %s per-token reference: %d of %d rows, %.0f s\n", name, i+1, n, time.Since(t0).Seconds())
					last = time.Now()
				}
			}
			out, err := g.r.Forward(e[n-1], n-1)
			if err != nil {
				t.Fatalf("per-token last row: %v", err)
			}
			want := append([]float32(nil), out...)
			tp := time.Since(t0)
			wantKV := g.snapshot(n)
			kv3NoNaN(t, name, want, wantKV)
			ld, worst := kv0LogitsDiff(want, got)
			kd, total := kv0CompareRows(wantKV, gotKV, 0, n)
			fmt.Fprintf(os.Stderr, "[kv3-families] past the bound %-26s logits differing %d of %d (worst %.3g), KV differing %d of %d; batched %.2f s (%d groups), per-token %.2f s\n",
				name, ld, len(want), worst, kd, total, tb.Seconds(), groups, tp.Seconds())
			if ld != 0 || kd != 0 {
				t.Errorf("%s: batched differs from the per-token path: %d logits, %d KV elements", name, ld, kd)
			}

			// The negative arm: no row-batched module, the same prompt declines by name.
			sc, sm, vs := g.r.psScores, g.r.psSoftmax, g.r.psVsum
			g.r.psScores, g.r.psSoftmax, g.r.psVsum = Pipeline{}, Pipeline{}, Pipeline{}
			_, err = g.r.PrefillLast(g.ctx, e, 0)
			g.r.psScores, g.r.psSoftmax, g.r.psVsum = sc, sm, vs
			if err == nil || !errors.Is(err, errPrefillDeclined) || !strings.Contains(err.Error(), "shared memory") {
				t.Errorf("%s: with no row-batched kernels the prompt must decline by name, got %v", name, err)
			}
		})
	}
}

// kv3ImageBlocks are the image blocks of the image-block gates: two blocks, the second well over a sliding window into the sequence.
var kv3ImageBlocks = [][2]int{{170, 210}, {250, 290}}

// TestKV3Families_imageBlockForcedRouteEqualsAttnImgBatched: an image-block pass on Gemma 3 VL below the shared-memory bound, with every exact-attention layer
// forced onto the row-batched kernels (block table passed), leaves the logits and every K and V row attn_img_batched leaves, bit for bit. The block rows
// attend their whole block, the rest stay causal, and the sliding-window layer's start comes from the causal count.
//
// Plant: the arms differ by route (the forced one launches row-batched groups, the unforced none), and the unforced arm really ran the image kernel:
// without the block table the row kernels would differ, which TestPrefillSplitKV_rowsEqualAttnImgBatched pins at the kernel level.
func TestKV3Families_imageBlockForcedRouteEqualsAttnImgBatched(t *testing.T) {
	const base = "../testdata/gemma3-vl-tiny"
	requireDeviceAndFixture(t, base)
	defer func() { prefillSplitKVForceForTest = false }()
	g := kv0LoadCtxAny(t, kv3LongContextCopy(t, base, 32768), "f32", 4096, kv0RandomTokens)
	if !g.r.imgPrefillReady || !g.r.prefillSplitKVReady() {
		t.Fatalf("image kernel ready %v, row-batched kernels ready %v: both are needed", g.r.imgPrefillReady, g.r.prefillSplitKVReady())
	}
	g.r.fastAttn, g.r.fastGemm = false, false
	const lead, rows = 150, 200
	e := g.embs(g.tokens(lead+rows, 11))
	arm := func(force bool) ([]float32, kv0KV, int) {
		prefillSplitKVForceForTest = force
		g.poison(lead + rows)
		g.prefill(e[:lead], 0)
		p0 := g.r.psLaunches
		lg, err := g.r.PrefillImageBlocksLast(g.ctx, e[lead:], lead, kv3ImageBlocks)
		if err != nil {
			t.Fatalf("PrefillImageBlocksLast (force=%v): %v", force, err)
		}
		return append([]float32(nil), lg...), g.snapshot(lead + rows), g.r.psLaunches - p0
	}
	want, wantKV, wantPS := arm(false)
	got, gotKV, gotPS := arm(true)
	kv3NoNaN(t, "image block", want, wantKV)
	if wantPS != 0 || gotPS == 0 {
		t.Fatalf("row-batched launch groups: %d unforced (want 0), %d forced (want some)", wantPS, gotPS)
	}
	ld, _ := kv0LogitsDiff(want, got)
	kd, total := kv0CompareRows(wantKV, gotKV, 0, lead+rows)
	fmt.Fprintf(os.Stderr, "[kv3-families] image block forced route: logits differing %d of %d, KV differing %d of %d (%d groups)\n", ld, len(want), kd, total, gotPS)
	if ld != 0 || kd != 0 {
		t.Errorf("the row-batched route with image blocks differs from attn_img_batched: %d logits, %d KV elements", ld, kd)
	}
}

// TestKV3Families_imageBlockPastTheBound: an image-block pass that starts past 12,160 keys, which attn_img_batched cannot run (its row of scores does not
// fit shared memory), prefills on the row-batched kernels with no decline and finite logits. The reference for the numbers is the forced-route gate above
// and the kernel-level one (both below the bound, where attn_img_batched can run); there is no per-token oracle for a bidirectional block.
// Negative arm: with the row-batched module absent the same pass declines by name, as it did before the module could take an image block.
func TestKV3Families_imageBlockPastTheBound(t *testing.T) {
	const base = "../testdata/gemma3-vl-tiny"
	requireDeviceAndFixture(t, base)
	dir := kv3LongContextCopy(t, base, 32768)
	g := kv0LoadCtxAny(t, dir, "f32", 13000, kv0RandomTokens)
	g.r.fastAttn, g.r.fastGemm = false, false
	const lead, rows = 12300, 256
	blocks := [][2]int{{12320, 12380}}
	e := g.embs(g.tokens(lead+rows, 5))
	g.prefill(e[:lead], 0)

	p0 := g.r.psLaunches
	lg, err := g.r.PrefillImageBlocksLast(g.ctx, e[lead:], lead, blocks)
	if err != nil {
		t.Fatalf("an image pass past the bound must prefill on the row-batched kernels: %v", err)
	}
	if g.r.psLaunches == p0 {
		t.Fatal("no row-batched launch group: the pass did not take the route")
	}
	nz := 0
	for _, v := range lg {
		if v != v || v > 3e38 || v < -3e38 {
			t.Fatalf("a non-finite logit past the bound")
		}
		if v != 0 {
			nz++
		}
	}
	if nz == 0 {
		t.Fatal("all-zero logits: the pass wrote nothing")
	}

	sc, sm, vs := g.r.psScores, g.r.psSoftmax, g.r.psVsum
	g.r.psScores, g.r.psSoftmax, g.r.psVsum = Pipeline{}, Pipeline{}, Pipeline{}
	// the lead rows are in the cache already (a text prefill of this length would decline with no row-batched kernels too)
	_, err = g.r.PrefillImageBlocksLast(g.ctx, e[lead:], lead, blocks)
	g.r.psScores, g.r.psSoftmax, g.r.psVsum = sc, sm, vs
	if err == nil || !errors.Is(err, errPrefillDeclined) || !strings.Contains(err.Error(), "image-block attention") {
		t.Errorf("with no row-batched kernels the image pass past the bound must decline by name, got %v", err)
	}
}
