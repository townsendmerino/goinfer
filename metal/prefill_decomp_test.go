//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMetalPrefillDecomp is S0 of the Metal prefill GEMM scoping (2026-09-25): where does the fast
// prefill's GPU time go, by kernel category, on a real checkpoint at the prompt lengths the peer
// claim grades (K=512, 3900)? It answers one question before any kernel is designed: what share of
// TTFT the GEMM is, since a GEMM-only redesign can move TTFT by at most 1/((1-X) + X/s).
//
// METHOD. PrefillLast is one command buffer, so its GPU timestamps give only a total. Each category
// here gets its OWN command buffer holding that category's dispatch for EVERY layer — each layer's
// own weights, PrefillLast's exact grids and buffers — so the cache footprint matches production
// (repeating one layer's dispatch would keep its weights hot in the SLC and flatter the kernel).
// Three checks make the split trustworthy rather than assumed:
//   - a full in-order replay of all categories must produce logits BIT-IDENTICAL to PrefillLast,
//     which proves the replica issues production's dispatches;
//   - the categories' sum is compared with the full replay's GPU time (non-additivity);
//   - the full replay's GPU time is compared with PrefillLast's wall time (host share).
//
// Plain dense families only (no MoE / sandwich / postOnly / QK-norm), which is what it is run on.
//
//	GOINFER_METAL_DECOMP=1 go test -tags goinfer_testhooks -run TestMetalPrefillDecomp -v -timeout 40m ./metal/
//
// GOINFER_METAL_DECOMP_MODEL overrides the checkpoint, GOINFER_METAL_DECOMP_K the lengths
// (comma-separated), GOINFER_METAL_DECOMP_REPS the repetitions per measurement (default 5).
func TestMetalPrefillDecomp(t *testing.T) {
	if os.Getenv("GOINFER_METAL_DECOMP") != "1" {
		t.Skip("set GOINFER_METAL_DECOMP=1 (loads a real checkpoint; minutes of GPU time)")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_METAL_DECOMP_MODEL")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md): a timing from it measures the disk", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s: %v", path, err)
	}
	ks := []int{512, 3900}
	if v := os.Getenv("GOINFER_METAL_DECOMP_K"); v != "" {
		ks = nil
		for _, f := range strings.Split(v, ",") {
			k, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil {
				t.Fatalf("GOINFER_METAL_DECOMP_K: %v", err)
			}
			ks = append(ks, k)
		}
	}
	reps := 5
	if v := os.Getenv("GOINFER_METAL_DECOMP_REPS"); v != "" {
		reps, _ = strconv.Atoi(v)
	}
	t0 := time.Now()
	hb := func(format string, a ...any) { // streams under -v; t.Logf would arrive only at the end
		fmt.Fprintf(os.Stderr, "[decomp %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}

	m, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	defer r.Close()
	if r.moe != nil || r.sandwich || r.postOnly || r.qkNorm {
		t.Skip("decomposition covers the plain dense layer shape only")
	}
	hb("loaded %s: H=%d I=%d nL=%d nH=%d V=%d", filepath.Base(path), r.H, r.I, r.nL, r.nH, r.V)
	vocab := r.V

	for _, K := range ks {
		embs := make([][]float32, K)
		for i := range embs {
			embs[i] = m.EmbedResidentForTest((i*131 + 7) % vocab)
		}
		runDecompAt(t, r, embs, reps, hb)
	}
}

// decompCat is one timed category: a name, and an encoder of its dispatches for layer l (the head
// category ignores l and encodes once).
type decompCat struct {
	name  string
	flops func(Mpad int) float64 // per layer, 0 for non-GEMM categories
	enc   func(e *Encoder, l int)
	once  bool // encode once, not per layer (the LM head)
}

func runDecompAt(t *testing.T, r *resident, embs [][]float32, reps int, hb func(string, ...any)) {
	r.ensurePrefill()
	d, pf := r.d, r.pf
	M := len(embs)
	Mpad := (M + 7) / 8 * 8
	H, I := r.H, r.I
	g0 := r.layers[0].geom
	nHhd := r.nH * g0.hd
	kvDim := g0.kvDim
	qkvDim := nHhd + 2*kvDim
	qDim := nHhd

	// A. The production call, end to end (wall), after one warm-up.
	var ref []float32
	wall := make([]float64, 0, reps)
	for i := 0; i <= reps; i++ {
		s := time.Now()
		out := r.PrefillLast(embs, 0)
		if i == 0 {
			ref = out
			hb("K=%d warm-up PrefillLast %.1f ms", M, ms(time.Since(s)))
			continue
		}
		wall = append(wall, ms(time.Since(s)))
	}
	if err := r.takeExecErr(); err != nil {
		t.Fatalf("PrefillLast: %v", err)
	}
	hb("K=%d PrefillLast wall ms %v", M, fmtMs(wall))

	// B. The replica. Buffers and uniforms exactly as PrefillLast allocates them.
	xh := make([]uint16, Mpad*H)
	parallelEmbedsF32ToF16(xh, embs, H)
	var bufs []Buffer
	nb := func(b Buffer) Buffer { bufs = append(bufs, b); return b }
	defer func() {
		for _, b := range bufs {
			d.ReleaseBuf(b)
		}
	}()
	xF := nb(NewBufferU16s(d, append([]uint16(nil), xh...)))
	normF := nb(NewBufferU16s(d, make([]uint16, Mpad*H)))
	qkvF := nb(NewBufferU16s(d, make([]uint16, Mpad*qkvDim)))
	ctxF := nb(NewBufferU16s(d, make([]uint16, Mpad*qDim)))
	guF := nb(NewBufferU16s(d, make([]uint16, Mpad*2*I)))
	dqF := nb(NewBufferU16s(d, make([]uint16, Mpad*I)))
	posv := make([]uint32, Mpad)
	for i := range M {
		posv[i] = uint32(i)
	}
	posB := nb(NewBufferUint32s(d, posv))
	uM := nb(NewBufferU32(d, uint32(Mpad)))
	uI := nb(NewBufferU32(d, uint32(I)))
	u2I := nb(NewBufferU32(d, uint32(2*I)))
	uQkv := nb(NewBufferU32(d, uint32(qkvDim)))
	uQDim := nb(NewBufferU32(d, uint32(qDim)))
	uStride := nb(NewBufferU32(d, uint32(qkvDim)))
	uKOff := nb(NewBufferU32(d, uint32(nHhd)))
	uVOff := nb(NewBufferU32(d, uint32(nHhd+kvDim)))
	uStartPos := nb(NewBufferU32(d, 0))
	uTotalQ := nb(NewBufferU32(d, uint32(nHhd)))
	uTotalK := nb(NewBufferU32(d, uint32(kvDim)))
	uBase0 := nb(NewBufferU32(d, 0))
	uBaseK := nb(NewBufferU32(d, uint32(nHhd)))
	m0, m1, m2 := nb(NewBufferU32(d, 0)), nb(NewBufferU32(d, 1)), nb(NewBufferU32(d, 2))
	dummyBias := nb(NewBufferFloats(d, make([]float32, 1)))
	uMReal := nb(NewBufferU32(d, uint32(M)))

	gg := func(N int) (int, int) { // PrefillLast's ggM(Mpad, N)
		total := ((Mpad/8 + 3) / 4) * ((N + 31) / 32) * 32
		return (total + 255) / 256 * 256, 256
	}
	const attnFusedSGPT = 4
	attnFusedTotal := r.nH * ((M + 7) / 8)
	attnFusedTotal = (attnFusedTotal + attnFusedSGPT - 1) / attnFusedSGPT * attnFusedSGPT * 32
	useFusedAttn := metalFusedAttentionEnabled(r.knobValue("GOINFER_METAL_FUSED_ATTENTION")) && g0.hd%8 == 0 && g0.hd <= 128
	gemmFlops := func(N, Kd int) func(int) float64 {
		return func(mp int) float64 { return 2 * float64(mp) * float64(N) * float64(Kd) }
	}

	cats := []decompCat{
		{name: "rmsnorm (pre-attn)", enc: func(e *Encoder, l int) {
			L := &r.layers[l]
			e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, xF, L.preNorm, normF, r.uH, r.uEps, r.uAddOne)
		}},
		{name: "GEMM qkv", flops: gemmFlops(qkvDim, H), enc: func(e *Encoder, l int) {
			L := &r.layers[l]
			n, tg := gg(qkvDim)
			e.Dispatch(pf.pGemmStore, n, tg, normF, L.qkvW, L.qkvS, qkvF, uM, uQkv, r.uH, L.qkvBias, m1)
		}},
		{name: "rope q+k", enc: func(e *Encoder, l int) {
			L := &r.layers[l]
			e.Dispatch(pf.pRope, M*r.nH*g0.half, 128, qkvF, L.invf, g0.uHd, posB, uTotalQ, uStride, uBase0, g0.uHalf, L.mscale)
			e.Dispatch(pf.pRope, M*g0.nKV*g0.half, 128, qkvF, L.invf, g0.uHd, posB, uTotalK, uStride, uBaseK, g0.uHalf, L.mscale)
		}},
		{name: "kv store", enc: func(e *Encoder, l int) {
			e.Dispatch(pf.pKv, M*kvDim, 128, qkvF, r.kc[l], r.vc[l], posB, g0.uKvDim, uStride, uKOff, uVOff)
		}},
		{name: "attention", enc: func(e *Encoder, l int) {
			L := &r.layers[l]
			if useFusedAttn {
				e.Dispatch(pf.pAttnFused, attnFusedTotal, attnFusedSGPT*32, qkvF, r.kc[l], r.vc[l], ctxF, r.uNH, g0.uNKV, g0.uHd, uStartPos, r.uScale, uStride, L.uWindow, uMReal)
			} else {
				e.Dispatch(pf.pAttn, M*r.nH*tgReduceAttn, tgReduceAttn, qkvF, r.kc[l], r.vc[l], ctxF, r.uNH, g0.uNKV, g0.uHd, uStartPos, r.uScale, uStride, L.uWindow)
			}
		}},
		{name: "GEMM o (+residual)", flops: gemmFlops(H, qDim), enc: func(e *Encoder, l int) {
			L := &r.layers[l]
			n, tg := gg(H)
			e.Dispatch(pf.pGemmStore, n, tg, ctxF, L.oW, L.oS, xF, uM, r.uH, uQDim, dummyBias, m2)
		}},
		{name: "rmsnorm (pre-MLP)", enc: func(e *Encoder, l int) {
			L := &r.layers[l]
			e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, xF, L.postNorm, normF, r.uH, r.uEps, r.uAddOne)
		}},
		{name: "GEMM gate/up", flops: gemmFlops(2*I, H), enc: func(e *Encoder, l int) {
			L := &r.layers[l]
			n, tg := gg(2 * I)
			e.Dispatch(pf.pGemmStore, n, tg, normF, L.guW, L.guS, guF, uM, u2I, r.uH, dummyBias, m0)
		}},
		{name: "swiglu", enc: func(e *Encoder, l int) {
			e.Dispatch(pf.pSw, M*I, 256, guF, dqF, uI, r.uAct)
		}},
		{name: "GEMM down (+residual)", flops: gemmFlops(H, I), enc: func(e *Encoder, l int) {
			L := &r.layers[l]
			n, tg := gg(H)
			e.Dispatch(pf.pGemmStore, n, tg, dqF, L.dW, L.dS, xF, uM, r.uH, uI, dummyBias, m2)
		}},
		{name: "LM head (last row)", once: true, enc: func(e *Encoder, _ int) {
			e.Dispatch(pf.pRmsQ, tgReduceNorm, tgReduceNorm, xF.At((M-1)*H*2), r.finalNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
			e.Dispatch(r.pGemvW8, r.V*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
		}},
	}
	resetX := func() { copy(xF.U16s(), xh) }
	gpuMs := func(encode func(e *Encoder)) float64 {
		e := r.q.Begin()
		encode(e)
		e.End()
		if err := e.Err(); err != nil {
			t.Fatalf("command buffer aborted: %v", err)
		}
		return (e.GPUEnd() - e.GPUStart()) * 1e3
	}
	full := func(e *Encoder) { // production order: layer-major, then the head
		for l := 0; l < r.nL; l++ {
			for _, c := range cats {
				if !c.once {
					c.enc(e, l)
				}
			}
		}
		for _, c := range cats {
			if c.once {
				c.enc(e, 0)
			}
		}
	}

	// Replica fidelity: the full replay must reproduce PrefillLast's logits bit for bit.
	resetX()
	gpuMs(full)
	got := r.logits.Floats()[:r.V]
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(ref[i]) {
			t.Fatalf("K=%d: replica logits differ from PrefillLast at %d (%v vs %v) — the replica does not issue production's dispatches, so its split would not describe production", M, i, got[i], ref[i])
		}
	}
	hb("K=%d replica logits bit-identical to PrefillLast (%d values)", M, len(got))

	// Full replay GPU time, then each category, interleaved rep by rep so drift hits every arm alike.
	fullMs := []float64{}
	catMs := make([][]float64, len(cats))
	for rep := 0; rep < reps; rep++ {
		resetX()
		fullMs = append(fullMs, gpuMs(full))
		for ci, c := range cats {
			resetX()
			v := gpuMs(func(e *Encoder) {
				if c.once {
					c.enc(e, 0)
					return
				}
				for l := 0; l < r.nL; l++ {
					c.enc(e, l)
				}
			})
			catMs[ci] = append(catMs[ci], v)
		}
		hb("K=%d rep %d/%d: full replay %.1f ms GPU", M, rep+1, reps, fullMs[rep])
	}

	// Report.
	var sum float64
	var lines []string
	fullMed := median(fullMs)
	wallMed := median(wall)
	for ci, c := range cats {
		med := median(catMs[ci])
		sum += med
		tf := ""
		if c.flops != nil {
			tf = fmt.Sprintf("  %.2f TFLOPS", c.flops(Mpad)*float64(r.nL)/(med*1e-3)/1e12)
		}
		lines = append(lines, fmt.Sprintf("  %-22s %9.2f ms  %5.1f%% of full  spread %4.1f%%%s", c.name, med, 100*med/fullMed, 100*spreadOf(catMs[ci]), tf))
	}
	var gemm float64
	for ci, c := range cats {
		if c.flops != nil {
			gemm += median(catMs[ci])
		}
	}
	fmt.Fprintf(os.Stderr, "\n=== Metal prefill decomposition, K=%d (Mpad=%d), %d reps, medians ===\n", M, Mpad, reps)
	for _, l := range lines {
		fmt.Fprintln(os.Stderr, l)
	}
	fmt.Fprintf(os.Stderr, "  %-22s %9.2f ms  (%.1f%% of the full replay — non-additivity)\n", "sum of categories", sum, 100*sum/fullMed)
	fmt.Fprintf(os.Stderr, "  %-22s %9.2f ms  spread %4.1f%%\n", "full replay (GPU)", fullMed, 100*spreadOf(fullMs))
	fmt.Fprintf(os.Stderr, "  %-22s %9.2f ms  spread %4.1f%%  → host/other %.1f ms (%.1f%%)\n", "PrefillLast (wall)", wallMed, 100*spreadOf(wall), wallMed-fullMed, 100*(wallMed-fullMed)/wallMed)
	fmt.Fprintf(os.Stderr, "  GEMM share: %.1f%% of GPU, %.1f%% of PrefillLast wall\n\n", 100*gemm/fullMed, 100*gemm/wallMed)
	t.Logf("K=%d: GEMM %.1f ms of %.1f ms GPU (%.1f%%), PrefillLast wall %.1f ms", M, gemm, fullMed, 100*gemm/fullMed, wallMed)
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1e3 }

func spreadOf(xs []float64) float64 {
	lo, hi := xs[0], xs[0]
	for _, x := range xs {
		lo, hi = math.Min(lo, x), math.Max(hi, x)
	}
	return (hi - lo) / median(xs)
}

func fmtMs(xs []float64) string {
	var b strings.Builder
	for i, x := range xs {
		if i > 0 {
			b.WriteString(" ")
		}
		fmt.Fprintf(&b, "%.1f", x)
	}
	return b.String()
}
