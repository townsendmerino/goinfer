//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/testgate"
)

// KV3's gates for the row-batched exact attention (docs/measurements/kv3-tiled-exact-2026-10-13.md §3 and §6): G2b, G3 and G7.
// G1 is TestPrefillSplitKV_rowsEqualAttnBatched.

// kv3NoNaN fails when a snapshot or a logits row holds a NaN: the rig poisons the cache with NaN before each arm, so two arms that
// both failed to write would otherwise compare equal.
func kv3NoNaN(t *testing.T, name string, logits []float32, kv kv0KV) {
	t.Helper()
	for _, v := range logits {
		if math.IsNaN(float64(v)) {
			t.Fatalf("%s: NaN in the logits", name)
		}
	}
	for i := range kv.t {
		for _, v := range kv.t[i] {
			if math.IsNaN(float64(v)) {
				t.Fatalf("%s: NaN in KV tensor %d: a row was not written", name, i)
			}
		}
	}
}

// TestKV3Tiled_forcedRouteEqualsAttnBatched_tiny (G2b): a whole batched prefill below the shared-memory bound, with every
// exact-attention layer forced onto the row-batched kernels, leaves the logits and every K and V row attn_batched leaves, bit for
// bit. Two lever settings: both off (the exact class) and the tensor-core GEMM alone (exact attention beside fast projections,
// the shape of a head-dim-256 model above the floor). The prompts span several 512-row passes, so later passes start past zero.
func TestKV3Tiled_forcedRouteEqualsAttnBatched_tiny(t *testing.T) {
	const base = "../../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	defer func() { prefillSplitKVForceForTest = false }()
	for _, kv := range []string{"f32", "f16"} {
		g := kv0LoadCtx(t, base, kv, 4096, kv0RandomTokens)
		if !g.r.prefillSplitKVReady() {
			t.Fatal("the row-batched kernels did not load")
		}
		for _, gemm := range []bool{false, true} {
			g.r.fastAttn, g.r.fastGemm = false, gemm
			for _, n := range []int{300, 1000, 3900} {
				name := fmt.Sprintf("kv=%s gemm=%v n=%d", kv, gemm, n)
				e := g.embs(g.tokens(n, n))
				arm := func(force bool) ([]float32, kv0KV, int, int) {
					prefillSplitKVForceForTest = force
					g.poison(n)
					p0, m0 := g.r.psLaunches, g.r.fastGemmLaunches
					lg := g.prefill(e, 0)
					return lg, g.snapshot(n), g.r.psLaunches - p0, g.r.fastGemmLaunches - m0
				}
				want, wantKV, wantPS, wantGemm := arm(false)
				got, gotKV, gotPS, _ := arm(true)
				kv3NoNaN(t, name, want, wantKV)
				if wantPS != 0 || gotPS == 0 {
					t.Fatalf("%s: row-batched launch groups: %d unforced (want 0), %d forced (want some)", name, wantPS, gotPS)
				}
				if gemm && n >= 512 && wantGemm == 0 {
					t.Fatalf("%s: the tensor-core GEMM did not run: this arm is not the mixed class it names", name)
				}
				ld, _ := kv0LogitsDiff(want, got)
				kd, total := kv0CompareRows(wantKV, gotKV, 0, n)
				fmt.Fprintf(os.Stderr, "[kv3-tiled] forced route %-26s logits differing %d of %d, KV differing %d of %d (%d row-batched groups)\n", name, ld, len(want), kd, total, gotPS)
				if ld != 0 || kd != 0 {
					t.Errorf("%s: the row-batched route differs from attn_batched: %d logits, %d KV elements", name, ld, kd)
				}
			}
		}
	}
}

// TestKV3Tiled_batchedEqualsPerToken_tiny (G3): past the 12,160-key bound, with the fast levers off, a prompt prefills in batched
// passes (it declined before) and leaves the logits and the K and V rows of the per-token path, bit for bit.
//
//	go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run '^TestKV3Tiled_batchedEqualsPerToken_tiny$' -v ./cuda/
func TestKV3Tiled_batchedEqualsPerToken_tiny(t *testing.T) {
	const base = "../../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	dir := kv3LongContextCopy(t, base, 32768)
	for _, kv := range []string{"f32", "f16"} {
		g := kv0LoadCtx(t, dir, kv, 17000, kv0RandomTokens)
		g.r.fastAttn, g.r.fastGemm = false, false
		for _, n := range []int{12400, 16384} {
			name := fmt.Sprintf("kv=%s n=%d", kv, n)
			e := g.embs(g.tokens(n, 3*n))
			g.poison(n)
			p0 := g.r.psLaunches
			t0 := time.Now()
			got := g.prefill(e, 0) // a decline is a failure here
			tb := time.Since(t0)
			groups := g.r.psLaunches - p0
			gotKV := g.snapshot(n)
			if groups == 0 || g.r.fastAttnLaunches != 0 || g.r.fastGemmLaunches != 0 {
				t.Fatalf("%s: %d row-batched groups, %d fused attention and %d tensor-core launches: not an exact pass on the row kernels", name, groups, g.r.fastAttnLaunches, g.r.fastGemmLaunches)
			}
			g.poison(n)
			t0, last := time.Now(), time.Now()
			for i := 0; i < n-1; i++ {
				if err := g.r.ForwardNoLogits(e[i], i); err != nil {
					t.Fatalf("per-token row %d: %v", i, err)
				}
				if time.Since(last) > 30*time.Second {
					fmt.Fprintf(os.Stderr, "[kv3-tiled] %s per-token reference: %d of %d rows, %.0f s\n", name, i+1, n, time.Since(t0).Seconds())
					last = time.Now()
				}
			}
			out, err := g.r.Forward(e[n-1], n-1)
			if err != nil {
				t.Fatalf("per-token last row: %v", err)
			}
			want := slices.Clone(out)
			tp := time.Since(t0)
			wantKV := g.snapshot(n)
			kv3NoNaN(t, name, want, wantKV)
			ld, worst := kv0LogitsDiff(want, got)
			kd, total := kv0CompareRows(wantKV, gotKV, 0, n)
			fmt.Fprintf(os.Stderr, "[kv3-tiled] past the bound %-16s logits differing %d of %d (worst %.3g), KV differing %d of %d; batched %.2f s (%d groups), per-token %.2f s\n",
				name, ld, len(want), worst, kd, total, tb.Seconds(), groups, tp.Seconds())
			if ld != 0 || kd != 0 {
				t.Errorf("%s: batched differs from the per-token path: %d logits, %d KV elements", name, ld, kd)
			}
		}
	}
}

// TestKV3Tiled_pastBoundPromptRunsExact_tiny: with the tensor-core GEMM on and no fused attention (the shape of a head-dim-256 model
// above the floor), a prompt whose attention takes the row-batched kernels past the bound runs no tensor-core launch and leaves the
// bits of the levers-off pass (the per-token bits, TestKV3Tiled_batchedEqualsPerToken_tiny); below the bound the GEMM still runs.
func TestKV3Tiled_pastBoundPromptRunsExact_tiny(t *testing.T) {
	const base = "../../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	dir := kv3LongContextCopy(t, base, 32768)
	g := kv0LoadCtx(t, dir, "f32", 13000, kv0RandomTokens)
	if !g.r.prefillSplitKVReady() {
		t.Fatal("the row-batched kernels did not load")
	}
	pass := func(n int, gemm bool) ([]float32, kv0KV, int, int) {
		g.r.fastAttn, g.r.fastGemm = false, gemm
		e := g.embs(g.tokens(n, 3*n))
		g.poison(n)
		p0, m0 := g.r.psLaunches, g.r.fastGemmLaunches
		lg := g.prefill(e, 0)
		return lg, g.snapshot(n), g.r.psLaunches - p0, g.r.fastGemmLaunches - m0
	}
	if _, _, ps, mma := pass(3900, true); ps != 0 || mma == 0 {
		t.Fatalf("below the bound: %d row-batched groups (want 0), %d tensor-core launches (want some)", ps, mma)
	}
	const n = 12400
	want, wantKV, _, _ := pass(n, false)
	got, gotKV, ps, mma := pass(n, true)
	kv3NoNaN(t, "past the bound", got, gotKV)
	ld, _ := kv0LogitsDiff(want, got)
	kd, total := kv0CompareRows(wantKV, gotKV, 0, n)
	fmt.Fprintf(os.Stderr, "[kv3-tiled] past the bound, GEMM lever on: %d row-batched groups, %d tensor-core launches; logits differing %d of %d, KV differing %d of %d\n",
		ps, mma, ld, len(want), kd, total)
	if ps == 0 || mma != 0 {
		t.Fatalf("past the bound: %d row-batched groups (want some), %d tensor-core launches (want 0)", ps, mma)
	}
	if ld != 0 || kd != 0 {
		t.Errorf("past the bound: differs from the levers-off pass: %d logits, %d KV elements", ld, kd)
	}
}

// TestKV3Tiled_withoutTheRowKernelsTheExactPassDeclines: the negative arm. With the row-batched module absent, an exact pass past
// the bound declines by name, as it did before the module existed; with it, the same prompt prefills.
func TestKV3Tiled_withoutTheRowKernelsTheExactPassDeclines(t *testing.T) {
	const base = "../../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	g := kv0LoadCtx(t, kv3LongContextCopy(t, base, 32768), "f16", 17000, kv0RandomTokens)
	e := g.embs(g.tokens(12400, 1))
	g.r.fastAttn, g.r.fastGemm = false, false
	sc, sm, vs := g.r.psScores, g.r.psSoftmax, g.r.psVsum
	g.r.psScores, g.r.psSoftmax, g.r.psVsum = Pipeline{}, Pipeline{}, Pipeline{}
	_, err := g.r.PrefillLast(g.ctx, e, 0)
	if err == nil {
		t.Fatal("a 12,400-token exact pass prefilled with no row-batched kernels: attn_batched's span bound is not being applied")
	}
	if !errors.Is(err, errPrefillDeclined) || !strings.Contains(err.Error(), "layer 0") {
		t.Errorf("the decline must wrap errPrefillDeclined and name the layer: %v", err)
	}
	g.r.psScores, g.r.psSoftmax, g.r.psVsum = sc, sm, vs
	if _, err := g.r.PrefillLast(g.ctx, e, 0); err != nil {
		t.Errorf("with the row-batched kernels the same prompt must prefill: %v", err)
	}
}

// TestKV3Tiled_spanScratchMatchesThePass (G7): the buffers a pass asks for are PrefillScratchBytes(M) plus
// prefillSpanScratchPass(M, keys), exactly, for passes below and past the bound, fused-served and not; PrefillSpanScratchBytes, the
// bound admission adds, is at least the pass's own figure.
func TestKV3Tiled_spanScratchMatchesThePass(t *testing.T) {
	const base = "../../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	g := kv0LoadCtx(t, kv3LongContextCopy(t, base, 32768), "f16", 17000, kv0RandomTokens)
	defer func() { prefillScratchSeenForTest = nil }()
	sawSpan := 0
	for _, levers := range []bool{true, false} {
		g.r.fastAttn, g.r.fastGemm = levers, levers
		for _, start := range []int{0, 12300} {
			for _, M := range []int{8, 16, 300, 512} {
				var seen int64
				prefillScratchSeenForTest = func(b int64) { seen += b + allocRoundSlack(b) }
				if _, err := g.r.PrefillLast(g.ctx, g.embs(g.tokens(M, 5)), start); err != nil {
					t.Fatalf("levers=%v start=%d M=%d: %v", levers, start, M, err)
				}
				prefillScratchSeenForTest = nil
				base, span := g.r.PrefillScratchBytes(M), g.r.prefillSpanScratchPass(M, start+M)
				if bound := g.r.PrefillSpanScratchBytes(M, start+M); bound < span || (bound > 0) != (span > 0) {
					t.Errorf("levers=%v start=%d M=%d: the span bound %d B against the pass's own %d B", levers, start, M, bound, span)
				}
				if span > 0 {
					sawSpan++
				}
				if want := base + span; want != seen {
					t.Errorf("levers=%v start=%d M=%d: PrefillScratchBytes %d + PrefillSpanScratchBytes %d = %d B, the pass allocated %d B", levers, start, M, base, span, want, seen)
				} else {
					fmt.Fprintf(os.Stderr, "[kv3-tiled] scratch levers=%v start=%d M=%d: %d B + span %d B = the pass's allocations\n", levers, start, M, base, span)
				}
				// the cases that must and must not carry the score scratch
				routes := start+M > 12160 && (!levers || M < attnFusedMinRows)
				if (span > 0) != routes {
					t.Errorf("levers=%v start=%d M=%d: span scratch %d B, want it present exactly when the pass is exact past the bound (%v)", levers, start, M, span, routes)
				}
			}
		}
	}
	if sawSpan == 0 {
		t.Fatal("no case carried the score scratch: the gate did not exercise it")
	}
}

// TestKV3Tiled_forcedRouteOnGemma3 is a measurement and a real-weights identity reading, not a gate (KV3_TILED_SPEED=1): on
// Gemma-3-4B (head dim 256, the family the row kernels exist for), one 512-row pass just under the shared-memory bound, with the
// global layers' attention on attn_batched and then forced onto the row-batched kernels, interleaved. It prints each pass's wall
// time and whether the logits and the pass's K and V rows agree bit for bit.
//
//	GOINFER_HEAVY_TESTS=1 KV3_TILED_SPEED=1 go test -count=1 -timeout 20m -tags 'cuda goinfer_testhooks' -run '^TestKV3Tiled_forcedRouteOnGemma3$' -v ./cuda/
func TestKV3Tiled_forcedRouteOnGemma3(t *testing.T) {
	testgate.RequireHeavy(t)
	if os.Getenv("KV3_TILED_SPEED") == "" {
		t.Skip("set KV3_TILED_SPEED=1")
	}
	path := modelPath("gemma-3-4b-it-Q4_K_M.gguf")
	requireDeviceAndFile(t, path)
	defer func() { prefillSplitKVForceForTest = false }()
	g := kv0LoadCtx(t, path, "f32", 13000, kv0TextTokens(t, path))
	const lead, M = 11520, 512 // the pass's last row attends 12,032 keys: under the bound, so both routes run
	ids := g.tokens(lead+M, 77)
	e := g.embs(ids)
	t0 := time.Now()
	g.prefill(e[:lead], 0)
	fmt.Fprintf(os.Stderr, "[kv3-tiled] gemma-3-4b: %d-token lead prefilled in %.1f s\n", lead, time.Since(t0).Seconds())
	var ref []float32
	var refKV kv0KV
	for pair := range 3 {
		for _, force := range []bool{false, true} {
			prefillSplitKVForceForTest = force
			p0 := g.r.psLaunches
			t0 := time.Now()
			lg := g.prefill(e[lead:], lead)
			dt := time.Since(t0)
			kv := g.snapshot(lead + M)
			groups := g.r.psLaunches - p0
			if force != (groups > 0) {
				t.Fatalf("force=%v but %d row-batched groups", force, groups)
			}
			if ref == nil {
				ref, refKV = lg, kv
				kv3NoNaN(t, "gemma-3-4b", ref, refKV)
			}
			ld, _ := kv0LogitsDiff(ref, lg)
			kd, total := kv0CompareRows(refKV, kv, lead, lead+M)
			fmt.Fprintf(os.Stderr, "[kv3-tiled] gemma-3-4b pair %d rows=%v: pass of %d rows at %d keys %.3f s (%d groups); logits differing %d of %d, KV rows differing %d of %d\n",
				pair, force, M, lead+M, dt.Seconds(), groups, ld, len(lg), kd, total)
			if ld != 0 || kd != 0 {
				t.Errorf("pair %d force=%v: differs from the first attn_batched pass", pair, force)
			}
		}
	}
}

// TestKV3Tiled_batchedEqualsPerToken_real (G4): on the real head-dim-256 and 512 models, past the bound, the batched prefill
// against the per-token path, bit for bit, logits and every K and V row: Gemma-4-E4B at default settings, and Gemma-3-4B with the
// fast levers off and at default settings (a prompt past the bound runs the exact kernels, prefillPastBoundExact). KV3_TILED_N
// sets the prompt length (default 12,400); KV3_TILED_FORCE=1 forces the row kernels at a shorter length, which is how the plumbing
// is checked by day: there Gemma-3's default arm runs the tensor-core GEMM and is read for fidelity, cosine 0.99 and equal argmax.
//
//	GOINFER_HEAVY_TESTS=1 KV3_TILED_REAL=1 go test -count=1 -timeout 60m -tags 'cuda goinfer_testhooks' -run '^TestKV3Tiled_batchedEqualsPerToken_real$' -v ./cuda/
func TestKV3Tiled_batchedEqualsPerToken_real(t *testing.T) {
	testgate.RequireHeavy(t)
	if os.Getenv("KV3_TILED_REAL") == "" {
		t.Skip("set KV3_TILED_REAL=1")
	}
	n := 12400
	if v := os.Getenv("KV3_TILED_N"); v != "" {
		if _, err := fmt.Sscanf(v, "%d", &n); err != nil {
			t.Fatalf("KV3_TILED_N=%q: %v", v, err)
		}
	}
	if os.Getenv("KV3_TILED_FORCE") != "" {
		prefillSplitKVForceForTest = true
		defer func() { prefillSplitKVForceForTest = false }()
	}
	for _, mdl := range []struct {
		name, file string
		eModel     bool
	}{
		{"gemma-3-4b", "gemma-3-4b-it-Q4_K_M.gguf", false},
		{"gemma-4-e4b", "gemma-4-e4b-gguf/gemma-4-E4B_q4_0-it.gguf", true},
	} {
		t.Run(mdl.name, func(t *testing.T) {
			path := modelPath(mdl.file)
			requireDeviceAndFile(t, path)
			g := kv0LoadCtx(t, path, "f32", n+600, kv0TextTokens(t, path))
			if g.r.eModel != mdl.eModel {
				t.Fatalf("%s: eModel=%v, the test expects %v", mdl.name, g.r.eModel, mdl.eModel)
			}
			e := g.embs(g.tokens(n, 4242))
			// the per-token path, once: it does not depend on the levers
			g.poison(n)
			t0, last := time.Now(), time.Now()
			for i := 0; i < n-1; i++ {
				if err := g.r.ForwardNoLogits(e[i], i); err != nil {
					t.Fatalf("per-token row %d: %v", i, err)
				}
				if time.Since(last) > 30*time.Second {
					fmt.Fprintf(os.Stderr, "[kv3-tiled] %s per-token reference: %d of %d rows, %.0f s\n", mdl.name, i+1, n, time.Since(t0).Seconds())
					last = time.Now()
				}
			}
			out, err := g.r.Forward(e[n-1], n-1)
			if err != nil {
				t.Fatalf("per-token last row: %v", err)
			}
			want := slices.Clone(out)
			tp := time.Since(t0)
			wantKV := g.snapshot(n)
			kv3NoNaN(t, mdl.name+" per-token", want, wantKV)
			fmt.Fprintf(os.Stderr, "[kv3-tiled] %s per-token reference: %d rows in %.1f s\n", mdl.name, n, tp.Seconds())

			type arm struct {
				name   string
				levers bool
				exact  bool // an exact pass: graded bit for bit
			}
			arms := []arm{{"default", true, mdl.eModel}}
			if !mdl.eModel {
				arms = []arm{{"levers off", false, true}, {"default", true, splitKVRequired(n)}}
			}
			attn0, gemm0 := g.r.fastAttn, g.r.fastGemm
			defer func() { g.r.fastAttn, g.r.fastGemm = attn0, gemm0 }()
			for _, a := range arms {
				g.r.fastAttn, g.r.fastGemm = attn0 && a.levers, gemm0 && a.levers
				g.poison(n)
				p0, m0 := g.r.psLaunches, g.r.fastGemmLaunches
				t0 := time.Now()
				got := g.prefill(e, 0) // a decline is a failure
				tb := time.Since(t0)
				groups, gemm := g.r.psLaunches-p0, g.r.fastGemmLaunches-m0
				gotKV := g.snapshot(n)
				if groups == 0 {
					t.Fatalf("%s %s: no row-batched launch groups: the pass did not take the row kernels", mdl.name, a.name)
				}
				if a.exact && gemm != 0 {
					t.Fatalf("%s %s: %d tensor-core launches in a pass graded as exact", mdl.name, a.name, gemm)
				}
				ld, worst := kv0LogitsDiff(want, got)
				kd, total := kv0CompareRows(wantKV, gotKV, 0, n)
				cos, same := kv0Cosine(got, want), kv0Argmax(got) == kv0Argmax(want)
				fmt.Fprintf(os.Stderr, "[kv3-tiled] G4 %s %-10s n=%d: batched %.2f s (%d row groups, %d tensor-core launches), per-token %.1f s; logits differing %d of %d (worst %.3g), KV differing %d of %d; cosine %.6f, argmax equal %v\n",
					mdl.name, a.name, n, tb.Seconds(), groups, gemm, tp.Seconds(), ld, len(want), worst, kd, total, cos, same)
				if a.exact {
					if ld != 0 || kd != 0 {
						t.Errorf("%s %s: an exact batched pass differs from the per-token path: %d logits, %d KV elements", mdl.name, a.name, ld, kd)
					}
				} else if cos < 0.99 || !same {
					t.Errorf("%s %s: fidelity against the per-token path: cosine %.6f (bar 0.99), argmax equal %v", mdl.name, a.name, cos, same)
				}
			}
		})
	}
}

// g46Depths are the prompt lengths G46 reads, from where the tensor-core GEMM's existing gate stops to the shared-memory bound
// (splitKVRequired is false at every one of them, so the pass runs attn_batched and not the row kernels).
var g46Depths = []int{4096, 6144, 8192, 10240, 12000, 12160}

// g46Seeds are two independent prompts from the corpus stream.
var g46Seeds = []int{4242, 9001}

// TestG46_gemma3GEMMFidelityLadder is G46's gate (docs/measurements/g46-gemma3-gemm-fidelity-2026-10-21.md): Gemma-3-4B at default
// settings, the batched prefill of each depth against the per-token path, last-row logits, cosine 0.99 and equal argmax. One per-token
// pass per prompt supplies the reference at every depth. A levers-off control at the shortest and longest depth must read 0 logits
// differing, so a nonzero reading elsewhere is the tensor-core GEMM's. The verdict line applies the registered rule.
//
//	GOINFER_HEAVY_TESTS=1 G46_REAL=1 go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run '^TestG46_gemma3GEMMFidelityLadder$' -v ./backend/cuda/
func TestG46_gemma3GEMMFidelityLadder(t *testing.T) {
	testgate.RequireHeavy(t)
	if os.Getenv("G46_REAL") == "" {
		t.Skip("set G46_REAL=1")
	}
	path := modelPath("gemma-3-4b-it-Q4_K_M.gguf")
	requireDeviceAndFile(t, path)
	nMax := g46Depths[len(g46Depths)-1]
	g := kv0LoadCtx(t, path, "f32", nMax+600, kv0TextTokens(t, path))
	attn0, gemm0 := g.r.fastAttn, g.r.fastGemm
	defer func() { g.r.fastAttn, g.r.fastGemm = attn0, gemm0 }()
	if !gemm0 {
		t.Fatalf("the tensor-core GEMM is off at load: the default arm would not read it")
	}
	checkpoint := make(map[int]bool, len(g46Depths))
	for _, d := range g46Depths {
		if splitKVRequired(d) {
			t.Fatalf("depth %d needs the row kernels: the ladder is below the bound", d)
		}
		checkpoint[d] = true
	}

	pass, ambiguous := true, false
	start := time.Now()
	for si, seed := range g46Seeds {
		e := g.embs(g.tokens(nMax, seed))
		g.poison(nMax)
		want := make(map[int][]float32, len(g46Depths))
		t0, last := time.Now(), time.Now()
		for i := range nMax {
			if err := g.ctx.Err(); err != nil {
				t.Fatalf("seed %d per-token row %d: %v", seed, i, err)
			}
			if checkpoint[i+1] {
				out, err := g.r.Forward(e[i], i)
				if err != nil {
					t.Fatalf("seed %d per-token row %d: %v", seed, i, err)
				}
				want[i+1] = slices.Clone(out)
			} else if err := g.r.ForwardNoLogits(e[i], i); err != nil {
				t.Fatalf("seed %d per-token row %d: %v", seed, i, err)
			}
			if time.Since(last) > 30*time.Second {
				fmt.Fprintf(os.Stderr, "[g46] seed %d (%d of %d) per-token reference: %d of %d rows, %.0f s, run %.0f s\n",
					seed, si+1, len(g46Seeds), i+1, nMax, time.Since(t0).Seconds(), time.Since(start).Seconds())
				last = time.Now()
			}
		}
		fmt.Fprintf(os.Stderr, "[g46] seed %d per-token reference: %d rows in %.1f s\n", seed, nMax, time.Since(t0).Seconds())

		for di, d := range g46Depths {
			if di == 0 || di == len(g46Depths)-1 {
				g.r.fastAttn, g.r.fastGemm = false, false
				g.poison(d)
				p0, m0 := g.r.psLaunches, g.r.fastGemmLaunches
				got := g.prefill(e[:d], 0)
				ld, worst := kv0LogitsDiff(want[d], got)
				fmt.Fprintf(os.Stderr, "[g46] control seed %d depth %d levers off: logits differing %d of %d (worst %.3g), %d row groups, %d tensor-core launches\n",
					seed, d, ld, len(got), worst, g.r.psLaunches-p0, g.r.fastGemmLaunches-m0)
				if ld != 0 || g.r.fastGemmLaunches != m0 {
					t.Errorf("control seed %d depth %d: levers off must equal the per-token path bit for bit and launch no tensor-core GEMM (%d logits differ, %d launches): the harness cannot attribute a difference to the GEMM",
						seed, d, ld, g.r.fastGemmLaunches-m0)
					pass = false
				}
				g.r.fastAttn, g.r.fastGemm = attn0, gemm0
			}
			g.poison(d)
			p0, m0 := g.r.psLaunches, g.r.fastGemmLaunches
			t1 := time.Now()
			got := g.prefill(e[:d], 0)
			groups, gemm := g.r.psLaunches-p0, g.r.fastGemmLaunches-m0
			if gemm == 0 {
				t.Fatalf("seed %d depth %d: no tensor-core GEMM launch: the default arm did not run the path G46 reads", seed, d)
			}
			if groups != 0 {
				t.Errorf("seed %d depth %d: %d row-kernel groups below the bound", seed, d, groups)
				pass = false
			}
			cos, same := kv0Cosine(got, want[d]), kv0Argmax(got) == kv0Argmax(want[d])
			ld, worst := kv0LogitsDiff(want[d], got)
			kv3NoNaN(t, fmt.Sprintf("seed %d depth %d", seed, d), got, kv0KV{})
			verdict := "PASS"
			switch {
			case cos < 0.99 || !same:
				verdict, pass = "FAIL", false
			case cos < 0.992:
				verdict, ambiguous = "AMBIGUOUS", true
			}
			fmt.Fprintf(os.Stderr, "[g46] seed %d depth %5d: batched %.2f s, %d tensor-core launches; logits differing %d of %d (worst %.3g); cosine %.6f, argmax equal %v: %s\n",
				seed, d, time.Since(t1).Seconds(), gemm, ld, len(got), worst, cos, same, verdict)
		}
	}
	switch {
	case !pass:
		t.Errorf("G46 FAIL: a cell read under cosine 0.99 or with a different argmax, or a control did not hold (the cell lines above); run %.0f s", time.Since(start).Seconds())
	case ambiguous:
		t.Logf("G46 AMBIGUOUS: no cell under 0.99, at least one under 0.992; parked, read cell by cell; run %.0f s", time.Since(start).Seconds())
	default:
		t.Logf("G46 PASS: every cell at cosine 0.992 or better with equal argmax; run %.0f s", time.Since(start).Seconds())
	}
}
