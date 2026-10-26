//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"
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
	const base = "../testdata/llama-attnfa-tiny"
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

// TestKV3Tiled_batchedEqualsPerToken_tiny (G3; B15-G2's per-token half): past the 12,160-key default bound, with the fast levers
// off, a prompt prefills in batched passes and leaves the logits and the K and V rows of the per-token path (decode on split-KV),
// bit for bit: at 12,400 on attn_batched under the opt-in (no row-batched group), at 16,384 past it on the row kernels.
//
//	go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run '^TestKV3Tiled_batchedEqualsPerToken_tiny$' -v ./cuda/
func TestKV3Tiled_batchedEqualsPerToken_tiny(t *testing.T) {
	const base = "../testdata/llama-attnfa-tiny"
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
			rows := !g.r.prefillAttnFits(n) // past the opt-in's bound the row kernels serve; inside it attn_batched
			if (groups > 0) != rows || g.r.fastAttnLaunches != 0 || g.r.fastGemmLaunches != 0 {
				t.Fatalf("%s: %d row-batched groups (want some: %v), %d fused attention and %d tensor-core launches: not the exact pass expected", name, groups, rows, g.r.fastAttnLaunches, g.r.fastGemmLaunches)
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
			want := append([]float32(nil), out...)
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
// above the floor), a prompt past the 12,160-key default bound runs no tensor-core launch and leaves the bits of the levers-off pass
// (the per-token bits, TestKV3Tiled_batchedEqualsPerToken_tiny): at 12,400, inside the opt-in's bound, on attn_batched, and past it
// on the row-batched kernels. Below the default bound the GEMM still runs.
func TestKV3Tiled_pastBoundPromptRunsExact_tiny(t *testing.T) {
	const base = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	dir := kv3LongContextCopy(t, base, 32768)
	g := kv0LoadCtx(t, dir, "f32", 17000, kv0RandomTokens)
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
	for _, n := range []int{12400, kv3PastBound(g.r)} {
		want, wantKV, _, _ := pass(n, false)
		got, gotKV, ps, mma := pass(n, true)
		kv3NoNaN(t, "past the bound", got, gotKV)
		ld, _ := kv0LogitsDiff(want, got)
		kd, total := kv0CompareRows(wantKV, gotKV, 0, n)
		fmt.Fprintf(os.Stderr, "[kv3-tiled] past the default bound at %d, GEMM lever on: %d row-batched groups, %d tensor-core launches; logits differing %d of %d, KV differing %d of %d\n",
			n, ps, mma, ld, len(want), kd, total)
		if rows := !g.r.prefillAttnFits(n); (ps > 0) != rows || mma != 0 {
			t.Fatalf("n=%d: %d row-batched groups (want some: %v), %d tensor-core launches (want 0)", n, ps, rows, mma)
		}
		if ld != 0 || kd != 0 {
			t.Errorf("n=%d: differs from the levers-off pass: %d logits, %d KV elements", n, ld, kd)
		}
	}
}

// TestKV3Tiled_withoutTheRowKernelsTheExactPassDeclines: the negative arm. With the row-batched module absent, an exact pass past
// the bound declines by name, as it did before the module existed; with it, the same prompt prefills.
func TestKV3Tiled_withoutTheRowKernelsTheExactPassDeclines(t *testing.T) {
	const base = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	g := kv0LoadCtx(t, kv3LongContextCopy(t, base, 32768), "f16", 17000, kv0RandomTokens)
	n := kv3PastBound(g.r)
	e := g.embs(g.tokens(n, 1))
	g.r.fastAttn, g.r.fastGemm = false, false
	sc, sm, vs := g.r.psScores, g.r.psSoftmax, g.r.psVsum
	g.r.psScores, g.r.psSoftmax, g.r.psVsum = Pipeline{}, Pipeline{}, Pipeline{}
	_, err := g.r.PrefillLast(g.ctx, e, 0)
	if err == nil {
		t.Fatalf("a %d-token exact pass prefilled with no row-batched kernels: attn_batched's span bound is not being applied", n)
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
// PrefillSpanScratchBytes(M, keys), exactly, for passes below and past the bound, fused-served and not. Admission adds the two.
func TestKV3Tiled_spanScratchMatchesThePass(t *testing.T) {
	const base = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, base)
	g := kv0LoadCtx(t, kv3LongContextCopy(t, base, 32768), "f16", 17000, kv0RandomTokens)
	defer func() { prefillScratchSeenForTest = nil }()
	sawSpan := 0
	for _, levers := range []bool{true, false} {
		g.r.fastAttn, g.r.fastGemm = levers, levers
		for _, start := range []int{0, 12300, kv3PastBound(g.r) - 512} {
			for _, M := range []int{8, 16, 300, 512} {
				var seen int64
				prefillScratchSeenForTest = func(b int64) { seen += b + allocRoundSlack(b) }
				if _, err := g.r.PrefillLast(g.ctx, g.embs(g.tokens(M, 5)), start); err != nil {
					t.Fatalf("levers=%v start=%d M=%d: %v", levers, start, M, err)
				}
				prefillScratchSeenForTest = nil
				base, span := g.r.PrefillScratchBytes(M), g.r.PrefillSpanScratchBytes(M, start+M)
				if span > 0 {
					sawSpan++
				}
				if want := base + span; want != seen {
					t.Errorf("levers=%v start=%d M=%d: PrefillScratchBytes %d + PrefillSpanScratchBytes %d = %d B, the pass allocated %d B", levers, start, M, base, span, want, seen)
				} else {
					fmt.Fprintf(os.Stderr, "[kv3-tiled] scratch levers=%v start=%d M=%d: %d B + span %d B = the pass's allocations\n", levers, start, M, base, span)
				}
				// the cases that must and must not carry the score scratch
				routes := !g.r.prefillAttnFits(start+M) && (!levers || M < attnFusedMinRows)
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
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("KV3_TILED_SPEED") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 and KV3_TILED_SPEED=1")
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
	for pair := 0; pair < 3; pair++ {
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
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("KV3_TILED_REAL") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 and KV3_TILED_REAL=1")
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
			want := append([]float32(nil), out...)
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
				if rows := !g.r.prefillAttnFits(n); (groups > 0) != rows {
					t.Fatalf("%s %s: %d row-batched launch groups, want some: %v (attn_batched serves inside the opt-in's bound)", mdl.name, a.name, groups, rows)
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
