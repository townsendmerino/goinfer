//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"os"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TOLERANCE — PRE-REGISTERED, THEN MEASURED TO BE MIS-DERIVED, AND CORRECTED IN THE OPEN.
//
// The first version of this file pre-registered "max |delta| <= 1e-3 of THE ROW'S OWN max |ctx|"
// plus "cosine >= 0.9999 per row", derived from f16 operand rounding. That bar failed widely --
// worst cosine 0.9642, worst row-relative delta 1.6 -- and the failure was NOT the kernel.
//
// What settled it (docs/measurements/prefill-l2l3-phase1-2026-09-05.md records the run): scoring
// attn_fused against exact f64 math on inputs FIRST ROUNDED TO f16 -- which is what the kernel
// actually receives -- gives cosine 1.00000000, worst 0.99999996 over 64 rows x 4 heads. The kernel
// reproduces its own inputs' arithmetic essentially exactly. Meanwhile attn_batched scores
// 1.00000000 against the f32 reference, so the exact path is sound too, and the gap between them is
// the f16 operand precision and nothing else.
//
// THE DERIVATION'S ERROR was the DENOMINATOR, not the numerator. It assumed |ctx| ~ |V|/sqrt(nKeys).
// With synthetic V of quasi-random sign the weighted average can cancel far more deeply than that:
// measured here, |ctx|/rms|V| ran from 1.02 down to 0.000211, and the cosine gap tracked it
// monotonically (1.02 -> 0.99999997; 0.0099 -> 0.99996; 0.00021 -> 0.9642). A row whose context
// cancels to one part in 4700 is ill-conditioned by construction: rounding the INPUTS alone rotates
// it 27%, so no kernel of any quality can meet a bar scaled by that row's own |ctx|.
//
// THE CORRECTED BARS, and why each is the right shape:
//
//	attnFusedMaxDeltaVsV -- max |delta| relative to max |V| over the head, NOT to |ctx|. |V| is the
//	scale the output is drawn from and does not collapse, so this is well-conditioned everywhere
//	while still catching any real defect: a kernel that attends the wrong keys, mismaps a fragment
//	or drops a seam is wrong by a fraction of |V|, not of |ctx|.
//
//	attnFusedMinCosine -- kept, but applied only to rows that are actually conditioned enough to
//	carry a direction (|ctx| >= attnFusedCondFloor * rms|V|). The conditioning of every row is
//	REPORTED either way, and the count of rows excluded is reported too, so this cannot silently
//	become a bar that tests nothing.
//
// TestAttnFused_vsF16Reference below is the logic gate that does not depend on any of this: it is
// immune to conditioning because it compares the kernel against its own inputs' exact arithmetic.
const (
	attnFusedMaxDeltaVsV = 1e-3
	attnFusedMinCosine   = 0.9999
	attnFusedCondFloor   = 0.01
)

// TestAttnFused_vsExact is the L2 correctness gate: every seam attn_batched handles, handled by
// attn_fused, checked against attn_batched itself over synthetic q/k/v.
//
// It launches BOTH KERNELS DIRECTLY rather than going through prefillCore, for two reasons. The
// production selector applies an M floor (small M does not fill a 64-row tile), so a test driven
// through the selector could not reach M ∈ {1, 7} at all — and those are exactly the shapes where
// the tile guards and the all-masked-row path are exercised. And a direct launch pins the KERNEL,
// not the selector's opinion of it; the selector gets its own test.
//
// The seam list is attn_batched's, from its own header: causal per row, per-row sliding window,
// GQA head grouping, and the gpt-oss sinks term. Each appears as a named case below and each is
// varied independently of the others, so a pass attributes to the seam rather than to the mix.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestAttnFused_vsExact -v
func TestAttnFused_vsExact(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident did not engage")
	}
	if r.bAttnFused64 == (Pipeline{}) || r.bAttnFused128 == (Pipeline{}) {
		t.Skip("attn_fused pipelines not loaded (set GOINFER_CUDA_FAST_PREFILL=1)")
	}

	type seam struct {
		name     string
		nH, nKV  int
		hd       int
		window   int32
		sinks    bool
		M        int
		startPos int
	}
	// M ∈ {1, 7, 64, 65, 512} crosses the 64-row tile boundary from both sides; startPos ∈ {0, 5,
	// large} covers the fresh pass and the chunked pass (prefillChunked calls with startPos > 0).
	var cases []seam
	for _, hd := range []int{64, 128} {
		for _, M := range []int{1, 7, 64, 65, 512} {
			for _, sp := range []int{0, 5, 1024} {
				cases = append(cases,
					seam{fmt.Sprintf("hd%d/M%d/sp%d/mha", hd, M, sp), 4, 4, hd, 0, false, M, sp},
					seam{fmt.Sprintf("hd%d/M%d/sp%d/gqa", hd, M, sp), 8, 2, hd, 0, false, M, sp},
					seam{fmt.Sprintf("hd%d/M%d/sp%d/window", hd, M, sp), 4, 2, hd, 96, false, M, sp},
					seam{fmt.Sprintf("hd%d/M%d/sp%d/sinks", hd, M, sp), 4, 2, hd, 0, true, M, sp},
					seam{fmt.Sprintf("hd%d/M%d/sp%d/window+sinks+gqa", hd, M, sp), 8, 2, hd, 96, true, M, sp},
				)
			}
		}
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			qDim, kvDim := c.nH*c.hd, c.nKV*c.hd
			nKeys := c.startPos + c.M
			q := make([]float32, c.M*qDim)
			kc := make([]float32, nKeys*kvDim)
			vc := make([]float32, nKeys*kvDim)
			for i := range q {
				q[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
			}
			for i := range kc {
				kc[i] = float32(math.Cos(float64(i) * 0.21))
				vc[i] = float32(math.Sin(float64(i)*0.13)) * 0.7
			}
			var sinkVals []float32
			if c.sinks {
				sinkVals = make([]float32, c.nH)
				for i := range sinkVals {
					sinkVals[i] = float32(math.Cos(float64(i)*1.7)) * 2
				}
			}
			const scale = 0.125

			var exact, fused []float32
			err := r.do(func() error {
				qb, kb, vb := r.af(len(q)), r.af(len(kc)), r.af(len(vc))
				outA, outB := r.af(c.M*qDim), r.af(c.M*qDim)
				sinkArg := ArgNull()
				if c.sinks {
					sb := r.af(len(sinkVals))
					if e := gpu.Upload(sb, sinkVals); e != nil {
						return e
					}
					sinkArg = Arg(sb)
				}
				for b, src := range map[Buffer][]float32{qb: q, kb: kc, vb: vc} {
					if e := gpu.Upload(b, src); e != nil {
						return e
					}
				}
				args := func(dst Buffer) []gpu.KernelArg {
					return []gpu.KernelArg{Arg(qb), Arg(kb), Arg(vb),
						gpu.ArgValue(int32(c.nH)), gpu.ArgValue(int32(c.nKV)), gpu.ArgValue(int32(c.hd)),
						gpu.ArgValue(int32(c.startPos)), gpu.ArgValue(float32(scale)),
						gpu.ArgValue(c.window), gpu.ArgValue(int32(c.M)), Arg(dst), sinkArg}
				}
				// exact: attn_batched, one block per (head, query row)
				maxNWin := nKeys
				if c.window > 0 && int(c.window) < maxNWin {
					maxNWin = int(c.window)
				}
				exCfg := LaunchConfig{GridX: uint32(c.nH), GridY: uint32(c.M), GridZ: 1,
					BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((maxNWin + 128) * 4)}
				if e := r.launch(r.bAttn, exCfg, args(outA)...); e != nil {
					return fmt.Errorf("attn_batched: %w", e)
				}
				if e := r.stream.Sync(); e != nil {
					return fmt.Errorf("attn_batched sync: %w", e)
				}
				// fused: one block per (head, 64-row query tile)
				pipe, shmem := r.attnFusedFor(c.hd)
				fuCfg := LaunchConfig{GridX: uint32(c.nH), GridY: uint32((c.M + attnFusedBM - 1) / attnFusedBM),
					GridZ: 1, BlockX: attnFusedThreads, BlockY: 1, BlockZ: 1, SharedMemBytes: shmem}
				if e := r.launch(pipe, fuCfg, args(outB)...); e != nil {
					return fmt.Errorf("attn_fused: %w", e)
				}
				if e := r.stream.Sync(); e != nil {
					return fmt.Errorf("attn_fused sync: %w", e)
				}
				exact = make([]float32, c.M*qDim)
				fused = make([]float32, c.M*qDim)
				if e := gpu.Download(outA, exact); e != nil {
					return e
				}
				return gpu.Download(outB, fused)
			})
			if err != nil {
				t.Fatalf("launch: %v", err)
			}

			// Per (row, head), never pooled -- an average would let one badly wrong row hide
			// behind hundreds of correct ones, which is the failure this test exists to catch.
			var vmax float64
			for _, x := range vc {
				if a := math.Abs(float64(x)); a > vmax {
					vmax = a
				}
			}
			var vsq float64
			for _, x := range vc {
				vsq += float64(x) * float64(x)
			}
			vrms := math.Sqrt(vsq / float64(len(vc)))

			worstDelta, worstCos, worstCond := 0.0, 1.0, math.Inf(1)
			var worstAt, worstCosAt string
			skippedCos := 0
			for m := 0; m < c.M; m++ {
				for h := 0; h < c.nH; h++ {
					off := m*qDim + h*c.hd
					var dot, na, nb, maxDiff, ctxSq float64
					for d := 0; d < c.hd; d++ {
						a, b := float64(exact[off+d]), float64(fused[off+d])
						dot += a * b
						na += a * a
						nb += b * b
						ctxSq += a * a
						maxDiff = math.Max(maxDiff, math.Abs(a-b))
					}
					rel := maxDiff / vmax
					if rel > worstDelta {
						worstDelta, worstAt = rel, fmt.Sprintf("m=%d h=%d", m, h)
					}
					cond := math.Sqrt(ctxSq/float64(c.hd)) / vrms
					if cond < worstCond {
						worstCond = cond
					}
					if cond < attnFusedCondFloor {
						skippedCos++
						continue // ill-conditioned by construction; see the header
					}
					if na > 0 && nb > 0 {
						if cos := dot / (math.Sqrt(na) * math.Sqrt(nb)); cos < worstCos {
							worstCos, worstCosAt = cos, fmt.Sprintf("m=%d h=%d", m, h)
						}
					}
				}
			}
			t.Logf("%s: worst delta/|V|=%.3e (%s) worst cosine=%.8f (%s) worst |ctx|/|V|=%.2e "+
				"cos-eligible=%d/%d", c.name, worstDelta, worstAt, worstCos, worstCosAt, worstCond,
				c.M*c.nH-skippedCos, c.M*c.nH)
			if worstDelta > attnFusedMaxDeltaVsV {
				t.Errorf("max-delta %.3e of |V| at %s exceeds the bar %.0e",
					worstDelta, worstAt, attnFusedMaxDeltaVsV)
			}
			if worstCos < attnFusedMinCosine {
				t.Errorf("worst well-conditioned per-row cosine %.8f at %s below %.4f",
					worstCos, worstCosAt, attnFusedMinCosine)
			}
		})
	}
}

// TestAttnFused_attendsKeysBeforeStartPos is the seam prefillChunked depends on and the one a
// "minimal" test would most easily drop: with startPos > 0 the rows of THIS pass must attend the
// keys earlier passes wrote, not just the ones in their own chunk. A kernel that reset its key
// origin to the chunk would still pass every same-chunk comparison and would silently truncate
// every prompt longer than one chunk.
//
// The proof does not compare against attn_batched (which could share the same bug). It changes
// ONLY the cache contents BEFORE startPos and requires the output to move.
func TestAttnFused_attendsKeysBeforeStartPos(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident did not engage")
	}
	if r.bAttnFused128 == (Pipeline{}) {
		t.Skip("attn_fused pipelines not loaded (set GOINFER_CUDA_FAST_PREFILL=1)")
	}

	const (
		nH, nKV, hd = 4, 2, 128
		startPos    = 300
		M           = 64
		scale       = 0.125
	)
	qDim, kvDim := nH*hd, nKV*hd
	nKeys := startPos + M
	q := make([]float32, M*qDim)
	base := make([]float32, nKeys*kvDim)
	vbase := make([]float32, nKeys*kvDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
	}
	for i := range base {
		base[i] = float32(math.Cos(float64(i) * 0.21))
		vbase[i] = float32(math.Sin(float64(i)*0.13)) * 0.7
	}
	// The perturbed copy differs ONLY in the prefix [0, startPos).
	pert := append([]float32(nil), vbase...)
	for i := 0; i < startPos*kvDim; i++ {
		pert[i] += 1.5
	}

	run := func(v []float32) []float32 {
		var out []float32
		err := r.do(func() error {
			qb, kb, vb := r.af(len(q)), r.af(len(base)), r.af(len(v))
			ob := r.af(M * qDim)
			for b, src := range map[Buffer][]float32{qb: q, kb: base, vb: v} {
				if e := gpu.Upload(b, src); e != nil {
					return e
				}
			}
			pipe, shmem := r.attnFusedFor(hd)
			cfg := LaunchConfig{GridX: nH, GridY: (M + attnFusedBM - 1) / attnFusedBM, GridZ: 1,
				BlockX: attnFusedThreads, BlockY: 1, BlockZ: 1, SharedMemBytes: shmem}
			if e := r.launch(pipe, cfg, Arg(qb), Arg(kb), Arg(vb),
				gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
				gpu.ArgValue(int32(startPos)), gpu.ArgValue(float32(scale)),
				gpu.ArgValue(int32(0)), gpu.ArgValue(int32(M)), Arg(ob), ArgNull()); e != nil {
				return e
			}
			if e := r.stream.Sync(); e != nil {
				return e
			}
			out = make([]float32, M*qDim)
			return gpu.Download(ob, out)
		})
		if err != nil {
			t.Fatalf("launch: %v", err)
		}
		return out
	}

	a, b := run(vbase), run(pert)
	diff := 0
	for i := range a {
		if a[i] != b[i] {
			diff++
		}
	}
	if diff == 0 {
		t.Fatalf("changing V at positions [0, %d) changed NOTHING in the output: the kernel is not "+
			"attending keys before startPos, so every chunked prompt would be silently truncated", startPos)
	}
	t.Logf("prefix-sensitivity: %d/%d output elements moved when V[0:%d) changed ✓", diff, len(a), startPos)
}

// f16rne rounds a float32 through f16 precision with round-to-nearest-EVEN, which is what the
// device's __float2half_rn does. Deliberately NOT cuda/kernels.go's f32tof16: that one is
// round-half-up, on purpose, because it encodes the int4 GROUP SCALES that must stay bit-identical
// with metal/aikit/CPU (audit C-15). Using it here would model the wrong rounding.
func f16rne(x float32) float32 {
	b := math.Float32bits(x)
	sign := b & 0x80000000
	exp := int32((b>>23)&0xFF) - 127
	if exp > 15 {
		return math.Float32frombits(sign | 0x7F800000)
	}
	if exp < -24 {
		return math.Float32frombits(sign)
	}
	shift := uint(13)
	if exp < -14 {
		shift = uint(13 + (-14 - exp))
	}
	m := b & 0x7FFFFF
	if exp < -14 {
		m |= 0x800000
	}
	half := m >> shift
	rem := m & ((1 << shift) - 1)
	mid := uint32(1) << (shift - 1)
	if rem > mid || (rem == mid && half&1 == 1) {
		half++
	}
	if exp < -14 {
		v := float32(half) * float32(math.Pow(2, -24))
		if sign != 0 {
			v = -v
		}
		return v
	}
	out := sign | uint32(exp+127)<<23 | (half << shift)
	if half<<shift >= 0x800000 {
		out = sign | uint32(exp+1+127)<<23
	}
	return math.Float32frombits(out)
}

// TestAttnFused_vsF16Reference is THE logic gate for attn_fused, and the one assertion here that
// no amount of input conditioning can distort.
//
// TestAttnFused_vsExact compares two GPU kernels that use different operand precision, so a
// disagreement there cannot by itself say which one is wrong — the same trap docs/task-prefill-
// gap.md §3.1 corrected at the model level, where a fast path was scored against an exact path that
// was itself a quantisation, and the distance was booked against the faster one. This test avoids
// it by scoring attn_fused against EXACT f64 arithmetic on ITS OWN INPUTS, rounded to f16 exactly
// as the kernel rounds them. Any error left is the kernel's logic: a mismapped mma fragment, a
// dropped seam, the wrong keys attended, a botched online rescale. Operand precision is factored
// out by construction rather than budgeted for.
//
// Small shapes on purpose: the reference is O(M · nH · nKeys · hd) in Go and this needs to stay a
// test, not a benchmark. Seam BREADTH is TestAttnFused_vsExact's job; DEPTH of correctness is this
// one's. Both are needed — neither substitutes for the other.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_FAST_PREFILL=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestAttnFused_vsF16Reference -v
func TestAttnFused_vsF16Reference(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident did not engage")
	}
	if r.bAttnFused64 == (Pipeline{}) || r.bAttnFused128 == (Pipeline{}) {
		t.Skip("attn_fused pipelines not loaded (set GOINFER_CUDA_FAST_PREFILL=1)")
	}

	cases := []struct {
		name        string
		nH, nKV, hd int
		M, startPos int
		window      int32
		sinks       bool
	}{
		{"hd64/mha/1tile", 4, 4, 64, 64, 0, 0, false},
		{"hd64/gqa/multitile", 8, 2, 64, 40, 200, 0, false},
		{"hd128/mha/multitile", 4, 4, 128, 40, 200, 0, false},
		{"hd128/gqa/window", 8, 2, 128, 40, 200, 96, false},
		{"hd128/gqa/sinks", 8, 2, 128, 40, 200, 0, true},
		{"hd64/window+sinks", 4, 2, 64, 70, 150, 96, true},
	}
	const scale = 0.125
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			qDim, kvDim := c.nH*c.hd, c.nKV*c.hd
			nKeys := c.startPos + c.M
			q := make([]float32, c.M*qDim)
			kc := make([]float32, nKeys*kvDim)
			vc := make([]float32, nKeys*kvDim)
			for i := range q {
				q[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
			}
			for i := range kc {
				kc[i] = float32(math.Cos(float64(i) * 0.21))
				vc[i] = float32(math.Sin(float64(i)*0.13)) * 0.7
			}
			var sinkVals []float32
			if c.sinks {
				sinkVals = make([]float32, c.nH)
				for i := range sinkVals {
					sinkVals[i] = float32(math.Cos(float64(i)*1.7)) * 2
				}
			}

			// Reference: exact f64 over f16-rounded operands, ROUNDED AND SEQUENCED THE WAY THE
			// KERNEL DOES IT.
			//
			// A one-pass softmax over the global max is mathematically equal to the online form but
			// NOT numerically equal, and modelling it that way is a real error rather than a nicety:
			// the kernel rounds each tile's PROVISIONAL weights exp(s - m_running) to f16 and then
			// rescales the f32 accumulator by exp(m_old - m_new), so the f16 rounding happens at a
			// different scale than a global-max reference would apply. Measured, before this was
			// fixed: a global-max reference put the multi-tile cases at cosine 0.9992-0.9994 while
			// single-tile cases sat at 0.99999996 — a gap that reads exactly like a rescale defect
			// and is in fact the reference not modelling the algorithm. So the reference walks the
			// same BN-key tiles in the same order and carries the same running state.
			ref := make([]float64, c.M*qDim)
			for m := 0; m < c.M; m++ {
				nk := c.startPos + m + 1
				win := 0
				if c.window > 0 && nk > int(c.window) {
					win = nk - int(c.window)
				}
				for h := 0; h < c.nH; h++ {
					kvh := h / (c.nH / c.nKV)
					// Block-uniform tile bounds, exactly as the kernel computes them.
					qTile := (m / attnFusedBM) * attnFusedBM
					lastRow := qTile + attnFusedBM - 1
					if lastRow > c.M-1 {
						lastRow = c.M - 1
					}
					blockMax := c.startPos + lastRow + 1
					blockMinWin := 0
					if c.window > 0 {
						if fk := c.startPos + qTile + 1; fk > int(c.window) {
							blockMinWin = fk - int(c.window)
						}
					}
					mRun := math.Inf(-1)
					lRun := 0.0
					if c.sinks {
						mRun, lRun = float64(sinkVals[h]), 1.0
					}
					acc := make([]float64, c.hd)
					for s0 := blockMinWin; s0 < blockMax; s0 += attnFusedBM {
						nkTile := attnFusedBM
						if blockMax-s0 < nkTile {
							nkTile = blockMax - s0
						}
						tileMax := math.Inf(-1)
						sc := make([]float64, nkTile)
						ok := make([]bool, nkTile)
						for j := 0; j < nkTile; j++ {
							key := s0 + j
							if key < win || key >= nk {
								continue
							}
							var dot float64
							for d := 0; d < c.hd; d++ {
								dot += float64(f16rne(q[m*qDim+h*c.hd+d])) * float64(f16rne(kc[key*kvDim+kvh*c.hd+d]))
							}
							sc[j] = dot * scale
							ok[j] = true
							tileMax = math.Max(tileMax, sc[j])
						}
						mNew := math.Max(mRun, tileMax)
						alpha := math.Exp(mRun - mNew)
						if math.IsInf(mRun, -1) && math.IsInf(mNew, -1) {
							alpha = 1
						}
						live := !math.IsInf(mNew, -1)
						var sum float64
						for j := 0; j < nkTile; j++ {
							if ok[j] && live {
								sc[j] = math.Exp(sc[j] - mNew)
								sum += sc[j]
							} else {
								sc[j] = 0
							}
						}
						for d := 0; d < c.hd; d++ {
							acc[d] *= alpha
						}
						for j := 0; j < nkTile; j++ {
							if sc[j] == 0 {
								continue
							}
							pj := float64(f16rne(float32(sc[j])))
							for d := 0; d < c.hd; d++ {
								acc[d] += pj * float64(f16rne(vc[(s0+j)*kvDim+kvh*c.hd+d]))
							}
						}
						lRun = alpha*lRun + sum
						mRun = mNew
					}
					for d := 0; d < c.hd; d++ {
						ref[m*qDim+h*c.hd+d] = acc[d] / lRun
					}
				}
			}

			var fused []float32
			err := r.do(func() error {
				qb, kb, vb := r.af(len(q)), r.af(len(kc)), r.af(len(vc))
				ob := r.af(c.M * qDim)
				sinkArg := ArgNull()
				if c.sinks {
					sb := r.af(len(sinkVals))
					if e := gpu.Upload(sb, sinkVals); e != nil {
						return e
					}
					sinkArg = Arg(sb)
				}
				for b, src := range map[Buffer][]float32{qb: q, kb: kc, vb: vc} {
					if e := gpu.Upload(b, src); e != nil {
						return e
					}
				}
				pipe, shmem := r.attnFusedFor(c.hd)
				cfg := LaunchConfig{GridX: uint32(c.nH), GridY: uint32((c.M + attnFusedBM - 1) / attnFusedBM),
					GridZ: 1, BlockX: attnFusedThreads, BlockY: 1, BlockZ: 1, SharedMemBytes: shmem}
				if e := r.launch(pipe, cfg, Arg(qb), Arg(kb), Arg(vb),
					gpu.ArgValue(int32(c.nH)), gpu.ArgValue(int32(c.nKV)), gpu.ArgValue(int32(c.hd)),
					gpu.ArgValue(int32(c.startPos)), gpu.ArgValue(float32(scale)),
					gpu.ArgValue(c.window), gpu.ArgValue(int32(c.M)), Arg(ob), sinkArg); e != nil {
					return e
				}
				if e := r.stream.Sync(); e != nil {
					return e
				}
				fused = make([]float32, c.M*qDim)
				return gpu.Download(ob, fused)
			})
			if err != nil {
				t.Fatalf("launch: %v", err)
			}

			worst, worstAt := 1.0, ""
			for m := 0; m < c.M; m++ {
				for h := 0; h < c.nH; h++ {
					off := m*qDim + h*c.hd
					var dot, na, nb float64
					for d := 0; d < c.hd; d++ {
						x, y := float64(fused[off+d]), ref[off+d]
						dot += x * y
						na += x * x
						nb += y * y
					}
					if na == 0 || nb == 0 {
						continue
					}
					if cos := dot / (math.Sqrt(na) * math.Sqrt(nb)); cos < worst {
						worst, worstAt = cos, fmt.Sprintf("m=%d h=%d", m, h)
					}
				}
			}
			t.Logf("%s: worst per-row cosine vs f16-input f64 reference = %.8f (%s)", c.name, worst, worstAt)
			// The kernel reproduces its own inputs' arithmetic; anything below this is LOGIC.
			if worst < 0.99999 {
				t.Errorf("%s: worst cosine %.8f at %s vs the f16-input reference — this is a kernel "+
					"logic defect, not operand precision, because the reference uses the same f16 "+
					"operands the kernel does", c.name, worst, worstAt)
			}
		})
	}
}
