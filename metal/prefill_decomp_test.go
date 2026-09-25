//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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
// (comma-separated), GOINFER_METAL_DECOMP_REPS the repetitions per measurement (default 5), and
// GOINFER_METAL_DECOMP_LOO=1 adds the leave-one-out in-sequence costs and GOINFER_METAL_DECOMP_STATE=1 the
// prior-state test, and GOINFER_METAL_S2=1 the R16 prototype comparison (all below).
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
	hb("loaded %s: H=%d I=%d nL=%d nH=%d V=%d, weights aliased from the file: %v", filepath.Base(path), r.H, r.I, r.nL, r.nH, r.V, r.alias != nil)
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

	gg := func(N int) (int, int) { // the retired kernel's 1-D grid (PrefillLast's ggM before R16)
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

	// R16 (S2): the GEMM categories dispatch through dispatchGemm(), which runs either production's gemm_w4f16_store (the
	// R16 kernel since 2026-09-25) or,
	// when protoOn, the test-only prototype (prefill_gemm_s2_test.go) with the same buffer list. protoOn stays
	// false unless GOINFER_METAL_S2=1 compiled the prototype, so every other phase is unchanged.
	var protoPipe Pipeline
	protoOn := false
	protoRows := 32 // tokens per threadgroup: 32 for prototypes 1-3, 64 for gemm_w4f16_tg4; 0 = the retired 1-D kernel
	dispatchGemm := func(e *Encoder, N int, bufs ...Buffer) {
		if protoOn {
			if protoRows == 0 { // gemm_w4f16_store_r15, the kernel R16 retired: its original 1-D grid
				n, tg := gg(N)
				e.Dispatch(protoPipe, n, tg, bufs...)
				return
			}
			e.Dispatch2D(protoPipe, (N+63)/64, (Mpad+protoRows-1)/protoRows, 128, 1, bufs...)
			return
		}
		// production: PrefillLast's gemm() grid since R16 (64×64 tiles, 128 threads)
		e.Dispatch2D(pf.pGemmStore, (N+63)/64, (Mpad+63)/64, 128, 1, bufs...)
	}

	cats := []decompCat{
		{name: "rmsnorm (pre-attn)", enc: func(e *Encoder, l int) {
			L := &r.layers[l]
			e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, xF, L.preNorm, normF, r.uH, r.uEps, r.uAddOne)
		}},
		{name: "GEMM qkv", flops: gemmFlops(qkvDim, H), enc: func(e *Encoder, l int) {
			L := &r.layers[l]
			dispatchGemm(e, qkvDim, normF, L.qkvW, L.qkvS, qkvF, uM, uQkv, r.uH, L.qkvBias, m1)
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
			dispatchGemm(e, H, ctxF, L.oW, L.oS, xF, uM, r.uH, uQDim, dummyBias, m2)
		}},
		{name: "rmsnorm (pre-MLP)", enc: func(e *Encoder, l int) {
			L := &r.layers[l]
			e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, xF, L.postNorm, normF, r.uH, r.uEps, r.uAddOne)
		}},
		{name: "GEMM gate/up", flops: gemmFlops(2*I, H), enc: func(e *Encoder, l int) {
			L := &r.layers[l]
			dispatchGemm(e, 2*I, normF, L.guW, L.guS, guF, uM, u2I, r.uH, dummyBias, m0)
		}},
		{name: "swiglu", enc: func(e *Encoder, l int) {
			e.Dispatch(pf.pSw, M*I, 256, guF, dqF, uI, r.uAct)
		}},
		{name: "GEMM down (+residual)", flops: gemmFlops(H, I), enc: func(e *Encoder, l int) {
			L := &r.layers[l]
			dispatchGemm(e, H, dqF, L.dW, L.dS, xF, uM, r.uH, uI, dummyBias, m2)
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
		pi0, si0 := vmPageCounts()
		fullMs = append(fullMs, gpuMs(full))
		pi1, si1 := vmPageCounts()
		hb("K=%d rep %d/%d: full replay %.1f ms GPU  pageins +%d swapins +%d  %s", M, rep+1, reps, fullMs[rep], pi1-pi0, si1-si0, thermalNote())
		for ci, c := range cats {
			resetX()
			pa, sa := vmPageCounts()
			v := gpuMs(func(e *Encoder) {
				if c.once {
					c.enc(e, 0)
					return
				}
				for l := 0; l < r.nL; l++ {
					c.enc(e, l)
				}
			})
			pb, sb := vmPageCounts()
			catMs[ci] = append(catMs[ci], v)
			// Every repeat of every category that matters, so a wide spread can be traced to its reps
			// (S0's first run printed medians only, and its two >50% spreads could not be read back).
			if v >= 100 {
				hb("K=%d rep %d/%d:   %-22s %9.1f ms GPU  pageins +%d swapins +%d", M, rep+1, reps, c.name, v, pb-pa, sb-sa)
			}
		}
	}

	// Leave-one-out (GOINFER_METAL_DECOMP_LOO=1): a category timed in its own command buffer ran in two
	// modes ~2x apart on the MLP GEMMs at K=3900 (S0's first run, 52-55% spreads), and in a stable
	// all-fast run the categories summed to only 64.6% of the full replay — so an isolated time is not
	// that kernel's cost inside production. Here each large category's IN-SEQUENCE cost is the full
	// replay's time minus the time of the same replay with that category's dispatches removed, rep by
	// rep interleaved. Kernel timing does not depend on the values it reads (no data-dependent
	// branches), so the stale inputs a removed category leaves behind change no downstream kernel's
	// work — only the cache and memory state it sees, which is the effect being measured.
	if os.Getenv("GOINFER_METAL_DECOMP_LOO") == "1" {
		fullWithout := func(skip int) func(e *Encoder) {
			return func(e *Encoder) {
				for l := 0; l < r.nL; l++ {
					for ci, c := range cats {
						if !c.once && ci != skip {
							c.enc(e, l)
						}
					}
				}
				for ci, c := range cats {
					if c.once && ci != skip {
						c.enc(e, 0)
					}
				}
			}
		}
		var big []int
		for ci := range cats {
			if median(catMs[ci]) >= 100 {
				big = append(big, ci)
			}
		}
		looFull := []float64{}
		looWithout := make(map[int][]float64)
		for rep := 0; rep < reps; rep++ {
			resetX()
			looFull = append(looFull, gpuMs(full))
			for _, ci := range big {
				resetX()
				v := gpuMs(fullWithout(ci))
				looWithout[ci] = append(looWithout[ci], v)
				hb("K=%d LOO rep %d/%d: full %.1f ms, without %-22s %.1f ms → in-sequence %.1f ms", M, rep+1, reps, looFull[rep], cats[ci].name, v, looFull[rep]-v)
			}
		}
		fm := median(looFull)
		fmt.Fprintf(os.Stderr, "\n=== leave-one-out, K=%d: in-sequence cost = full − full-without (medians of %d paired reps) ===\n", M, reps)
		var sumIn float64
		for _, ci := range big {
			var d []float64
			for i := range looFull {
				d = append(d, looFull[i]-looWithout[ci][i])
			}
			in := median(d)
			sumIn += in
			tf := ""
			if cats[ci].flops != nil {
				tf = fmt.Sprintf("  %.2f TFLOPS in sequence", cats[ci].flops(Mpad)*float64(r.nL)/(in*1e-3)/1e12)
			}
			fmt.Fprintf(os.Stderr, "  %-22s isolated %9.1f ms   in-sequence %9.1f ms (%5.1f%% of full)  ratio %.2fx  paired deltas %s%s\n",
				cats[ci].name, median(catMs[ci]), in, 100*in/fm, in/median(catMs[ci]), fmtMs(d), tf)
		}
		fmt.Fprintf(os.Stderr, "  in-sequence sum %.1f ms = %.1f%% of the full replay (%.1f ms, spread %.1f%%)\n\n", sumIn, 100*sumIn/fm, fm, 100*spreadOf(looFull))
	}

	// Prior-state test (GOINFER_METAL_DECOMP_STATE=1): the MLP GEMMs run ~1.9x slower inside the sequence
	// than alone, at both K, and alone they flip between two levels ~2x apart. If that tracks the GPU's
	// RECENT WORKLOAD (a clock/power state) rather than the kernel's inputs, gate/up timed alone should
	// be slow straight after heavy work and fast after an idle gap. Four conditions, interleaved per rep.
	if os.Getenv("GOINFER_METAL_DECOMP_STATE") == "1" {
		gu, att := -1, -1
		for ci, c := range cats {
			switch c.name {
			case "GEMM gate/up":
				gu = ci
			case "attention":
				att = ci
			}
		}
		only := func(ci int) func(e *Encoder) {
			return func(e *Encoder) {
				for l := 0; l < r.nL; l++ {
					cats[ci].enc(e, l)
				}
			}
		}
		conds := []string{"after 2 s idle", "right after a full replay", "right after attention-only", "back-to-back (2nd of 2)"}
		res := make([][]float64, len(conds))
		for rep := 0; rep < reps; rep++ {
			time.Sleep(2 * time.Second)
			resetX()
			res[0] = append(res[0], gpuMs(only(gu)))
			resetX()
			gpuMs(full)
			res[1] = append(res[1], gpuMs(only(gu)))
			gpuMs(only(att))
			res[2] = append(res[2], gpuMs(only(gu)))
			gpuMs(only(gu))
			res[3] = append(res[3], gpuMs(only(gu)))
			hb("K=%d STATE rep %d/%d: gate/up alone — idle %.1f · after full %.1f · after attention %.1f · back-to-back %.1f ms",
				M, rep+1, reps, res[0][rep], res[1][rep], res[2][rep], res[3][rep])
		}
		fmt.Fprintf(os.Stderr, "\n=== prior-state, K=%d: GEMM gate/up alone, by what ran just before (medians of %d) ===\n", M, reps)
		for i, c := range conds {
			fmt.Fprintf(os.Stderr, "  %-28s %9.1f ms  spread %5.1f%%  reps %s\n", c, median(res[i]), 100*spreadOf(res[i]), fmtMs(res[i]))
		}
		fmt.Fprintln(os.Stderr)
	}

	// R16 / S2 (GOINFER_METAL_S2=1): the prototype against the current kernel, per the pre-registration in
	// docs/tasks/red-october.md R16 — bitwise per GEMM on real inputs, full-replay logits, the registered metric
	// (sum of the four GEMM marginals by leave-one-out, both arms in the same session), and burst vs sustained.
	if os.Getenv("GOINFER_METAL_S2") == "1" {
		func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			pool := NewARPool()
			defer pool.Drain()
			lib, err := r.d.CompileLibrary(gemmS2Kernels, MSL3_1)
			if err != nil {
				t.Fatalf("compile S2 prototype: %v", err)
			}
			// GOINFER_METAL_S2_KERNEL picks the comparison arm: gemm_w4f16_tg (prototype 1, the default), _tg2, _tg3, _tg4,
			// or gemm_w4f16_store_r15 — the retired production kernel, for an old-vs-new A/B.
			kname := "gemm_w4f16_tg"
			if v := os.Getenv("GOINFER_METAL_S2_KERNEL"); v != "" {
				kname = v
			}
			switch kname {
			case "gemm_w4f16_tg4":
				protoRows = 64
			case "gemm_w4f16_store_r15":
				protoRows = 0
			}
			hb("K=%d S2 prototype kernel: %s (%d tokens per threadgroup)", M, kname, protoRows)
			if protoPipe, err = r.d.NewComputePipeline(lib, kname); err != nil {
				t.Fatalf("S2 pipeline: %v", err)
			}
		}()
		var gemmCats []int
		for ci, c := range cats {
			if c.flops != nil {
				gemmCats = append(gemmCats, ci)
			}
		}
		outOf := map[string]Buffer{"GEMM qkv": qkvF, "GEMM o (+residual)": xF, "GEMM gate/up": guF, "GEMM down (+residual)": xF}

		// (1) Bitwise, per GEMM, layer 0, on the activations a real prefill left in the buffers.
		protoOn = false
		resetX()
		gpuMs(full)
		allSame := true
		for _, ci := range gemmCats {
			c := cats[ci]
			out := outOf[c.name]
			before := append([]uint16(nil), out.U16s()...)
			runOne := func(proto bool) []uint16 {
				copy(out.U16s(), before)
				protoOn = proto
				gpuMs(func(e *Encoder) { c.enc(e, 0) })
				protoOn = false
				return append([]uint16(nil), out.U16s()...)
			}
			cur, pro := runOne(false), runOne(true)
			copy(out.U16s(), before)
			diff, maxAbs := 0, 0.0
			for i := range cur {
				if cur[i] != pro[i] {
					diff++
					maxAbs = math.Max(maxAbs, math.Abs(float64(f16ToF32(cur[i]))-float64(f16ToF32(pro[i]))))
				}
			}
			if diff != 0 {
				allSame = false
			}
			hb("K=%d S2 bitwise %-22s %d / %d elements differ (max |diff| %.3g)", M, c.name, diff, len(cur), maxAbs)
		}

		// (2) Full replay with the prototype as every GEMM: logits against PrefillLast's.
		protoOn = true
		resetX()
		gpuMs(full)
		protoOn = false
		pl := r.logits.Floats()[:r.V]
		nd, dot, na, nb2 := 0, 0.0, 0.0, 0.0
		am, bm := 0, 0
		for i := range pl {
			if math.Float32bits(pl[i]) != math.Float32bits(ref[i]) {
				nd++
			}
			dot += float64(pl[i]) * float64(ref[i])
			na += float64(pl[i]) * float64(pl[i])
			nb2 += float64(ref[i]) * float64(ref[i])
			if pl[i] > pl[am] {
				am = i
			}
			if ref[i] > ref[bm] {
				bm = i
			}
		}
		hb("K=%d S2 full-replay logits with the comparison kernel: %d / %d differ from PrefillLast (cosine %.9f, argmax %d vs %d); per-GEMM bitwise identical: %v",
			M, nd, len(pl), dot/math.Sqrt(na*nb2), am, bm, allSame)

		// (3) The registered metric: per arm, the sum of the four GEMM marginals (full − full-without-that-GEMM).
		without := func(skip map[int]bool) func(e *Encoder) {
			return func(e *Encoder) {
				for l := 0; l < r.nL; l++ {
					for ci, c := range cats {
						if !c.once && !skip[ci] {
							c.enc(e, l)
						}
					}
				}
				for ci, c := range cats {
					if c.once && !skip[ci] {
						c.enc(e, 0)
					}
				}
			}
		}
		allG := map[int]bool{}
		for _, ci := range gemmCats {
			allG[ci] = true
		}
		type armRes struct{ full, sum, allAtOnce []float64 }
		arms := map[bool]*armRes{false: {}, true: {}}
		perCat := map[bool]map[int][]float64{false: {}, true: {}}
		var ratios []float64
		for rep := 0; rep < reps; rep++ {
			order := []bool{false, true}
			if rep%2 == 1 {
				order = []bool{true, false}
			}
			for _, arm := range order {
				protoOn = arm
				resetX()
				f := gpuMs(full)
				sum := 0.0
				for _, ci := range gemmCats {
					resetX()
					w := gpuMs(without(map[int]bool{ci: true}))
					perCat[arm][ci] = append(perCat[arm][ci], f-w)
					sum += f - w
				}
				resetX()
				ng := gpuMs(without(allG))
				protoOn = false
				arms[arm].full = append(arms[arm].full, f)
				arms[arm].sum = append(arms[arm].sum, sum)
				arms[arm].allAtOnce = append(arms[arm].allAtOnce, f-ng)
			}
			ratios = append(ratios, arms[false].sum[rep]/arms[true].sum[rep])
			hb("K=%d S2 rep %d/%d: GEMM category in sequence — production %.1f ms, comparison %.1f ms → %.2fx (full replay %.1f vs %.1f ms)",
				M, rep+1, reps, arms[false].sum[rep], arms[true].sum[rep], ratios[rep], arms[false].full[rep], arms[true].full[rep])
		}

		// (4) Burst vs sustained, gate/up alone, both kernels (R16 precondition 2 grades the sustained number).
		gu := -1
		for ci, c := range cats {
			if c.name == "GEMM gate/up" {
				gu = ci
			}
		}
		only := func(ci int) func(e *Encoder) {
			return func(e *Encoder) {
				for l := 0; l < r.nL; l++ {
					cats[ci].enc(e, l)
				}
			}
		}
		idle := map[bool][]float64{}
		sus := map[bool][]float64{}
		for rep := 0; rep < reps; rep++ {
			for _, arm := range []bool{false, true} {
				protoOn = arm
				time.Sleep(2 * time.Second)
				resetX()
				idle[arm] = append(idle[arm], gpuMs(only(gu)))
				gpuMs(only(gu))
				sus[arm] = append(sus[arm], gpuMs(only(gu)))
				protoOn = false
			}
		}

		// Report and grade against R16's band (ship >= 2.85x, park 1.8-2.85x, kill < 1.8x on the sustained,
		// in-sequence GEMM category at K=512; only K=512 on the 1.5B decides — other runs are reported).
		// Arms are named by kernel: false = production's gemm_w4f16_store, true = the comparison kernel. Since R16
		// wired prototype 4 in, "production" is the new kernel and gemm_w4f16_store_r15 the retired one.
		name := func(a bool) string {
			if a {
				return "comparison"
			}
			return "production"
		}
		fmt.Fprintf(os.Stderr, "\n=== R16 / S2, K=%d: prototype vs current (medians of %d paired reps, arms alternating) ===\n", M, reps)
		for _, ci := range gemmCats {
			c := cats[ci]
			mc, mp := median(perCat[false][ci]), median(perCat[true][ci])
			fmt.Fprintf(os.Stderr, "  %-22s production %8.1f ms (%.2f TFLOPS)  comparison %8.1f ms (%.2f TFLOPS)  production/comparison %.2fx\n", c.name,
				mc, c.flops(Mpad)*float64(r.nL)/(mc*1e-3)/1e12, mp, c.flops(Mpad)*float64(r.nL)/(mp*1e-3)/1e12, mc/mp)
		}
		for _, a := range []bool{false, true} {
			fmt.Fprintf(os.Stderr, "  %-9s GEMM category (sum of marginals) %8.1f ms spread %4.1f%% · all-GEMMs-out cross-check %8.1f ms · full replay %8.1f ms\n",
				name(a), median(arms[a].sum), 100*spreadOf(arms[a].sum), median(arms[a].allAtOnce), median(arms[a].full))
		}
		for _, a := range []bool{false, true} {
			fmt.Fprintf(os.Stderr, "  %-9s gate/up alone: after 2 s idle %8.1f ms · sustained %8.1f ms  (idle/sustained %.2f)\n",
				name(a), median(idle[a]), median(sus[a]), median(idle[a])/median(sus[a]))
		}
		rm := median(ratios)
		band := "KILL (< 1.8x)"
		switch {
		case rm >= 2.85:
			band = "SHIP band (>= 2.85x)"
		case rm >= 1.8:
			band = "PARK band (1.8-2.85x)"
		}
		// The band reads "how much faster the comparison kernel is than production" — R16's grade when production was
		// the old kernel. Against the retired kernel (production is now the new one) a ratio BELOW 1 is the good
		// outcome, and a band would be meaningless, so none is printed.
		if protoRows == 0 {
			fmt.Fprintf(os.Stderr, "  production is %.2fx FASTER than the retired kernel (median; reps %s)", 1/rm, fmtMs(ratios))
		} else {
			fmt.Fprintf(os.Stderr, "  GEMM category speedup (comparison over production), paired: median %.2fx, reps %s → %s", rm, fmtMs(ratios), band)
		}
		if M != 512 {
			fmt.Fprintf(os.Stderr, "  [K=%d: reported, not deciding — R16 decides at K=512 on the 1.5B]", M)
		}
		fmt.Fprintf(os.Stderr, "\n  preconditions: fidelity — per-GEMM bitwise identical %v, full-replay logits differ in %d values; graded on the sustained timing\n\n", allSame, nd)
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

// vmPageCounts reads the system-wide page-in and swap-in counters from vm_stat. A delta across one
// command buffer that is nonzero while the GPU reads mapped weights means pages were faulted back in
// during the kernel — Metal binds .giw weights from the file mapping (S6 aliasing), so under memory
// pressure a kernel's timing can include re-reading evicted weight pages. System-wide, so another
// process's paging also counts; a zero delta is the informative reading.
func vmPageCounts() (pageins, swapins int64) {
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		return -1, -1
	}
	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Fields(strings.TrimSuffix(strings.TrimSpace(line), "."))
		if len(f) < 2 {
			continue
		}
		n, err := strconv.ParseInt(f[len(f)-1], 10, 64)
		if err != nil {
			continue
		}
		switch f[0] {
		case "Pageins:":
			pageins = n
		case "Swapins:":
			swapins = n
		}
	}
	return pageins, swapins
}

// thermalNote condenses `pmset -g therm` to "thermal: none" or the first warning it reports.
func thermalNote() string {
	out, err := exec.Command("pmset", "-g", "therm").Output()
	if err != nil {
		return "thermal: ?"
	}
	for _, line := range strings.Split(string(out), "\n") {
		l := strings.TrimSpace(line)
		if l == "" || strings.Contains(l, "No thermal warning") || strings.Contains(l, "No performance warning") || strings.Contains(l, "No CPU power status") {
			continue
		}
		return "thermal: " + l
	}
	return "thermal: none"
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
