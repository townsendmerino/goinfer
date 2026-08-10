//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestMLALatentReuse_prototype is micro-leg B of docs/task-mla-cuda-residency.md: the experiment
// that decides that task's 3-days-vs-3-weeks fork, and the skeleton of its eventual parity gate.
//
// THE HYPOTHESIS. MLA's decode attention looks like a new kernel shape (score width latDim, value
// width rank, one KV row shared by every query head) — but it is structurally nKV=1 with hd=latDim,
// which attn_batched ALREADY expresses:
//
//	kvDim = nKV*hd; group = nH/nKV; kvh = h/group   // nKV=1 ⇒ every head reads the same row
//	qDim  = nH*hd                                   // ⇒ q laid out nH×latDim == qAbs
//
// and MLA's value is the *prefix of its own key row*, so a latDim-wide weighted sum's first `rank`
// dims ARE the rank-space collapse the absorb path wants. If that holds, MLA needs no new attention
// kernel on CUDA and inherits the existing byte-identical gate (TestAttnBatched_bitIdentical).
//
// SCOPE. Test-only, no production wiring. Geometry is testdata/deepseek-tiny's real MLA shape
// (nH=4, kv_lora_rank=16, qk_rope_head_dim=8 ⇒ latDim=24) — the fixture cannot be loaded RESIDENT
// here (CUDA declines MLA on features, which is the whole point of the task), so this drives the
// kernel at the fixture's dims against an independent CPU reference rather than through a model.
//
// ALIASING. The production answer is a ~20-line attn_batched_shared_kv with __restrict__ dropped on
// the value pointer; passing one buffer as both kc and vc would be restrict-UB. The PROTOTYPE side-
// steps that by DUPLICATING the latent into two allocations — the aliasing fix is production work,
// not a prerequisite for answering the question.
//
// NOT the WebGPU approach. gpu/mla.go uses online (FlashAttention-style) softmax and is gated on
// tolerances; CUDA's decode attention is byte-identical by contract and Campaign A rejected online
// rescale explicitly. See finding (a) in docs/task-mla-cuda-residency.md.
func TestMLALatentReuse_prototype(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	defer cx.Close()
	bg := context.Background()
	mod, err := cx.LoadModule(prefillBatchedPTX)
	if err != nil {
		t.Fatalf("module: %v", err)
	}
	fn, err := mod.Function("attn_batched")
	if err != nil {
		t.Fatalf("fn(attn_batched): %v", err)
	}
	stream := mustStream(t, cx)

	// testdata/deepseek-tiny: num_attention_heads 4, kv_lora_rank 16, qk_rope_head_dim 8.
	// (qk_nope_head_dim 16 sets the SCALE, not the cache width — MLA scales by 1/sqrt(qkNope+qkRope).)
	const nH, rank, qkRope, qkNope = 4, 16, 8, 16
	const latDim = rank + qkRope
	const nKeys = 37
	const nKV, M, window = 1, 1, 0
	scale := float32(1.0 / math.Sqrt(float64(qkNope+qkRope)))

	var s uint32 = 20260809
	rnd := func() float32 { s = s*1664525 + 1013904223; return float32(int32(s>>8)%2000-1000) / 1000 }
	qAbs := make([]float32, nH*latDim) // qNopeAbs[rank] ‖ qRope[qkRope] per head
	for i := range qAbs {
		qAbs[i] = rnd()
	}
	lat := make([]float32, nKeys*latDim) // cn[rank] ‖ krj[qkRope] per key
	for i := range lat {
		lat[i] = rnd()
	}
	// DUPLICATE, do not alias (see the header note).
	latV := append([]float32(nil), lat...)

	dQ := mustAlloc[float32](t, cx, len(qAbs))
	dK := mustAlloc[float32](t, cx, len(lat))
	dV := mustAlloc[float32](t, cx, len(latV))
	dCtx := mustAlloc[float32](t, cx, nH*latDim)
	defer dQ.Close()
	defer dK.Close()
	defer dV.Close()
	defer dCtx.Close()
	for _, c := range []struct {
		b *gc.Buffer[float32]
		h []float32
	}{{dQ, qAbs}, {dK, lat}, {dV, latV}} {
		if e := gc.CopyHtoD(bg, c.b, c.h); e != nil {
			t.Fatalf("H2D: %v", e)
		}
	}

	launch := func(vbuf *gc.Buffer[float32]) []float32 {
		cfg := gc.LaunchConfig{GridX: nH, GridY: M, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1,
			SharedMemBytes: uint32((nKeys + 128) * 4)}
		if e := fn.LaunchOn(bg, stream, cfg,
			gc.Arg(dQ), gc.Arg(dK), gc.Arg(vbuf),
			gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(latDim)),
			gc.ArgValue(int32(nKeys-1)), gc.ArgValue(scale), gc.ArgValue(int32(window)),
			gc.ArgValue(int32(M)), gc.Arg(dCtx)); e != nil {
			t.Fatalf("launch: %v", e)
		}
		if e := stream.Synchronize(bg); e != nil {
			t.Fatalf("sync: %v", e)
		}
		out := make([]float32, nH*latDim)
		if e := gc.CopyDtoH(bg, out, dCtx); e != nil {
			t.Fatalf("D2H: %v", e)
		}
		return out
	}

	// ---- CPU reference: the absorb-path math, written independently of the kernel (f64).
	// score_j = scale·(qAbs[h]·lat[j]) over the FULL latDim; wsum[h] = Σ_j softmax_j · cn_j.
	refRank := make([]float64, nH*rank)
	for h := 0; h < nH; h++ {
		sc := make([]float64, nKeys)
		mx := math.Inf(-1)
		for j := 0; j < nKeys; j++ {
			var dot float64
			for d := 0; d < latDim; d++ {
				dot += float64(qAbs[h*latDim+d]) * float64(lat[j*latDim+d])
			}
			sc[j] = dot * float64(scale)
			mx = math.Max(mx, sc[j])
		}
		var sum float64
		for j := range sc {
			sc[j] = math.Exp(sc[j] - mx)
			sum += sc[j]
		}
		for j := range sc {
			sc[j] /= sum
		}
		for d := 0; d < rank; d++ { // value = the key row's own rank-prefix
			var acc float64
			for j := 0; j < nKeys; j++ {
				acc += sc[j] * float64(lat[j*latDim+d])
			}
			refRank[h*rank+d] = acc
		}
	}

	got := launch(dV)

	// ---- (1)+(2) the first `rank` dims of each head's row must BE the rank-space collapse.
	var maxAbs, maxRel float64
	for h := 0; h < nH; h++ {
		for d := 0; d < rank; d++ {
			g, r := float64(got[h*latDim+d]), refRank[h*rank+d]
			ad := math.Abs(g - r)
			if ad > maxAbs {
				maxAbs = ad
			}
			if den := math.Abs(r); den > 1e-6 && ad/den > maxRel {
				maxRel = ad / den
			}
		}
	}
	t.Logf("rank-prefix vs CPU absorb reference: max|Δ| = %.3e  max relΔ = %.3e", maxAbs, maxRel)
	if maxAbs > 1e-5 {
		t.Errorf("REUSE DOES NOT HOLD: attn_batched at nKV=1/hd=latDim does not reproduce the "+
			"absorb-path rank collapse (max|Δ| = %.3e) — the bespoke-kernel row in "+
			"docs/task-mla-cuda-residency.md activates", maxAbs)
	}

	// ---- (2b) per-dim independence VERIFIED, not assumed: the wasted rope-tail accumulate must not
	// perturb the rank dims at all. Zero the VALUE buffer's rope tail and require the first `rank`
	// output dims to come back BIT-IDENTICAL. (A shared reduction across dims would show up here.)
	latVzero := append([]float32(nil), latV...)
	for j := 0; j < nKeys; j++ {
		for d := rank; d < latDim; d++ {
			latVzero[j*latDim+d] = 0
		}
	}
	if e := gc.CopyHtoD(bg, dV, latVzero); e != nil {
		t.Fatalf("H2D zeroed: %v", e)
	}
	gotZ := launch(dV)
	for h := 0; h < nH; h++ {
		for d := 0; d < rank; d++ {
			if got[h*latDim+d] != gotZ[h*latDim+d] {
				t.Errorf("per-dim independence VIOLATED at head %d dim %d: %v vs %v — the rope-tail "+
					"accumulate is NOT ignorable waste and the reuse argument does not hold",
					h, d, got[h*latDim+d], gotZ[h*latDim+d])
			}
		}
	}
	t.Log("per-dim independence: rank dims bit-identical with the rope tail zeroed ✅ (waste is inert)")

	// ---- (3) measure the cost of the wasted dims: hd=latDim vs a hypothetical hd=rank.
	if e := gc.CopyHtoD(bg, dV, latV); e != nil {
		t.Fatalf("H2D restore: %v", e)
	}
	timeIt := func(hd int) float64 {
		cfg := gc.LaunchConfig{GridX: nH, GridY: M, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1,
			SharedMemBytes: uint32((nKeys + 128) * 4)}
		run := func() {
			_ = fn.LaunchOn(bg, stream, cfg, gc.Arg(dQ), gc.Arg(dK), gc.Arg(dV),
				gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)),
				gc.ArgValue(int32(nKeys-1)), gc.ArgValue(scale), gc.ArgValue(int32(window)),
				gc.ArgValue(int32(M)), gc.Arg(dCtx))
		}
		for i := 0; i < 200; i++ {
			run()
		}
		_ = stream.Synchronize(bg)
		best := time.Hour
		for rep := 0; rep < 5; rep++ {
			t0 := time.Now()
			for i := 0; i < 2000; i++ {
				run()
			}
			_ = stream.Synchronize(bg)
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return float64(best.Nanoseconds()) / 2000
	}
	// Order-alternated: a single-order A/B on this box measures GPU clock ramp, not the kernel
	// (learned the hard way in cuda/testdata/REGEN.md).
	full, narrow := math.Inf(1), math.Inf(1)
	for pass := 0; pass < 2; pass++ {
		if pass == 0 {
			full = math.Min(full, timeIt(latDim))
			narrow = math.Min(narrow, timeIt(rank))
		} else {
			narrow = math.Min(narrow, timeIt(rank))
			full = math.Min(full, timeIt(latDim))
		}
	}
	t.Logf("wasted-dim overhead: hd=latDim(%d) %.0f ns | hd=rank(%d) %.0f ns | %.3f× "+
		"(rope tail is %d/%d = %.1f%% of the value width)",
		latDim, full, rank, narrow, full/narrow, qkRope, latDim, 100*float64(qkRope)/latDim)

	t.Log("VERDICT: attn_batched reuse HOLDS at the latent geometry — " +
		"docs/task-mla-cuda-residency.md's ~3-day path is the live one")
}
