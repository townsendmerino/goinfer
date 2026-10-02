//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestQGateKernels_cpuParity (D-G02, docs/audit-metal-2026-09-30.md) checks the two kernels of the gated softmax layer
// (Qwen3.5/3.6, Qwen3-Next) against the CPU code they mirror (decoder/forward_qwen35.go): delta_qsplit, which splits
// the double-width q_proj output into q and gate, and delta_attn_gate, ctx *= sigmoid(gate). The resident parity test
// covers them only inside a whole forward, which cannot say which of the two is wrong. Chained as encodeLayer chains
// them (the gate kernel reads the split's output), over three geometries, with gate values out to ±100 so both tails
// of the sigmoid are reached. The split must match exactly and write nothing past its n elements.
func TestQGateKernels_cpuParity(t *testing.T) {
	splitDiff, gateErr, skip := qgateRun(t, allKernels)
	if skip {
		return
	}
	t.Logf("split: %d elements differ from the CPU's; gate: worst |kernel - CPU| / |ctx| = %.3g", splitDiff, gateErr)
	if splitDiff != 0 || !(gateErr <= qgateBar) {
		t.Errorf("split differs in %d elements (want 0); gate error %.3g (want <= %g)", splitDiff, gateErr, qgateBar)
	}
}

// qgateBar bounds the gate factor's error: float32 rounding of sigmoid and of the product, with room for the kernel's
// fast-math exp.
const qgateBar = 1e-6

// TestQGateKernels_mutations: the plausible wrong versions of the two kernels fail the check above. Each applies one
// textual change to the shipped source (Fatal if the anchor has drifted, so a rewrite cannot make a mutation a no-op).
func TestQGateKernels_mutations(t *testing.T) {
	for _, c := range []struct{ name, from, to string }{
		{"q and gate swapped", "q[t] = qg[base];\n    gate[t] = qg[base+hd];", "q[t] = qg[base+hd];\n    gate[t] = qg[base];"},
		// Reading the projection as two concatenated blocks gave plausible logits from the wrong tensor on WebGPU
		// (cosine 0.90; the kernel's comment).
		{"two blocks, not interleaved per head", "uint base = h*2*hd + d;\n    q[t] = qg[base];\n    gate[t] = qg[base+hd];",
			"q[t] = qg[t] + 0.0f * float(h + d);\n    gate[t] = qg[n + t];"},
		{"gate without the sigmoid", "ctx[t] = ctx[t] / (1.0f + exp(-gate[t]));", "ctx[t] = ctx[t] * gate[t];"},
		{"sigmoid of the wrong sign", "ctx[t] = ctx[t] / (1.0f + exp(-gate[t]));", "ctx[t] = ctx[t] / (1.0f + exp(gate[t]));"},
	} {
		t.Run(c.name, func(t *testing.T) {
			if n := strings.Count(allKernels, c.from); n != 1 {
				t.Fatalf("mutation anchor matched %d times (want 1); the kernel source has drifted from this test: %q", n, c.from)
			}
			splitDiff, gateErr, skip := qgateRun(t, strings.Replace(allKernels, c.from, c.to, 1))
			if skip {
				return
			}
			t.Logf("split: %d elements differ; gate error %.3g", splitDiff, gateErr)
			if splitDiff == 0 && gateErr <= qgateBar {
				t.Errorf("mutation %q passed the check; it is weaker than it looks", c.name)
			}
		})
	}
}

// qgateRun runs delta_qsplit then delta_attn_gate, compiled from src and launched as encodeLayer launches them, and
// returns how many q/gate elements differ from decoder's split (counting any write past n) and the worst gate-factor
// error, |kernel - CPU| / |ctx|, against decoder's gate.
func qgateRun(t *testing.T, src string) (splitDiff int, gateErr float64, skip bool) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
		return 0, 0, true
	}
	defer d.ReleaseObjects()
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(name string) Pipeline {
		p, e := d.NewComputePipeline(lib, name)
		if e != nil {
			t.Fatalf("pipeline %s: %v", name, e)
		}
		return p
	}
	pSplit, pGate := pipe("delta_qsplit"), pipe("delta_attn_gate")
	cq := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(35))
	const tail, sentinel = 64, float32(12345)
	for _, g := range []struct{ nH, hd int }{{16, 256}, {4, 128}, {3, 40}} {
		n := g.nH * g.hd
		qg, ctx := make([]float32, 2*n), make([]float32, n)
		for i := range qg {
			qg[i] = float32(rng.NormFloat64() * 4)
		}
		for i := range ctx {
			ctx[i] = float32(rng.NormFloat64())
		}
		// Plant the sigmoid's tails at gate positions taken from the CPU's own split of an index ramp, so the test does
		// not restate the layout it is checking.
		ramp := make([]float32, 2*n)
		for i := range ramp {
			ramp[i] = float32(i)
		}
		_, gateAt := decoder.SplitQGateForTest(ramp, g.nH, g.hd)
		for j, v := range []float32{100, -100, 30, -30, 0, 17, -17} {
			qg[int(gateAt[j*n/7])] = v
		}
		padded := func(v []float32) []float32 {
			out := append(append([]float32(nil), v...), make([]float32, tail)...)
			for i := len(v); i < len(out); i++ {
				out[i] = sentinel
			}
			return out
		}
		qB, gB, ctxB := NewBufferFloats(d, padded(make([]float32, n))), NewBufferFloats(d, padded(make([]float32, n))), NewBufferFloats(d, ctx)
		uN, uHd := NewBufferU32(d, uint32(n)), NewBufferU32(d, uint32(g.hd))
		e := cq.Begin()
		e.Dispatch(pSplit, n, 256, NewBufferFloats(d, qg), qB, gB, uN, uHd)
		e.Dispatch(pGate, n, 256, ctxB, gB, uN)
		e.End()
		if err := e.Err(); err != nil {
			t.Fatalf("nH=%d hd=%d: dispatch: %v", g.nH, g.hd, err)
		}
		wantQ, wantG := decoder.SplitQGateForTest(qg, g.nH, g.hd)
		gotQ, gotG := qB.Floats(), gB.Floats()
		for i := range n {
			if gotQ[i] != wantQ[i] {
				splitDiff++
			}
			if gotG[i] != wantG[i] {
				splitDiff++
			}
		}
		for i := n; i < n+tail; i++ {
			if gotQ[i] != sentinel || gotG[i] != sentinel {
				splitDiff++
			}
		}
		wantCtx := append([]float32(nil), ctx...)
		decoder.QGateContextForTest(wantCtx, wantG)
		for i, got := range ctxB.Floats()[:n] {
			gateErr = math.Max(gateErr, math.Abs(float64(got)-float64(wantCtx[i]))/math.Abs(float64(ctx[i])))
		}
		d.ReleaseAll()
	}
	return splitDiff, gateErr, false
}
