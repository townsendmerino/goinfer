//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestMoERoute is the parity gate for the on-GPU router, against a CPU reference written
// independently from the kernel.
//
// The router is where MoE goes wrong QUIETLY. Its output steers a DISCRETE choice, so a
// disagreement is not a small numeric error — it runs a different expert, and the output is
// unrelated rather than slightly off. goinfer has already paid for that class once: the Granite
// SSM investigation traced a 66%-agreement wall to discrete expert flips, and proved no
// precision knob could recover it. So the bar here is EXACT on the selected indices, not a
// cosine.
//
// The cases cover every routing flavour the registry produces, because they compose and each
// combination is a separate way to be wrong:
//   - softmax (Mixtral/Qwen-MoE) vs per-expert sigmoid (DeepSeek/GLM)
//   - selection bias (DeepSeek noaux_tc): steers SELECTION but must NOT enter the weight
//   - norm_topk_prob: renormalize the k weights to sum 1
//   - routed_scaling_factor
//   - group-limited top-k (DeepSeek NGroup>1): top-2-sum per group, keep topkGroup groups
func TestMoERoute(t *testing.T) {
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

	mod, err := cx.LoadModule(moePTX)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	fn, err := mod.Function("moe_route")
	if err != nil {
		t.Fatalf("Function(moe_route): %v", err)
	}
	stream := mustStream(t, cx)

	cases := []struct {
		name              string
		nE, k             int
		sigmoid, norm     bool
		scale             float64
		nGroup, topkGroup int
		withBias          bool
	}{
		{"mixtral/softmax-norm", 8, 2, false, true, 0, 0, 0, false},
		{"qwen2_moe/softmax", 60, 4, false, false, 0, 0, 0, false},
		{"glm/sigmoid-bias-norm", 64, 8, true, true, 0, 0, 0, true},
		{"deepseek/sigmoid-bias-scale", 64, 6, true, true, 2.5, 0, 0, true},
		{"deepseek/group-limited", 64, 6, true, true, 2.5, 8, 4, true},
		{"kimi/group-limited-wide", 128, 8, true, true, 2.827, 8, 4, true},
		{"k=1/top1", 16, 1, false, false, 0, 0, 0, false},
		{"k==nE/all-experts", 8, 8, false, true, 0, 0, 0, false},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var seed uint32 = 20260717
			rnd := func() float32 {
				seed = seed*1664525 + 1013904223
				return float32(int32(seed>>8)%2000-1000) / 500 // ~[-2,2]
			}
			logits := make([]float32, c.nE)
			for i := range logits {
				logits[i] = rnd()
			}
			bias := make([]float32, c.nE)
			if c.withBias {
				for i := range bias {
					bias[i] = rnd() * 0.25
				}
			}

			wantIdx, wantWgt := cpuRoute(logits, bias, c.nE, c.k, c.sigmoid, c.norm, c.scale, c.nGroup, c.topkGroup)

			dLog := mustAlloc[float32](t, cx, c.nE)
			dBias := mustAlloc[float32](t, cx, c.nE)
			dIdx := mustAlloc[uint32](t, cx, c.k)
			dWgt := mustAlloc[float32](t, cx, c.k)
			defer dLog.Close()
			defer dBias.Close()
			defer dIdx.Close()
			defer dWgt.Close()
			if e := gc.CopyHtoD(bg, dLog, logits); e != nil {
				t.Fatalf("H2D logits: %v", e)
			}
			if e := gc.CopyHtoD(bg, dBias, bias); e != nil {
				t.Fatalf("H2D bias: %v", e)
			}
			sig, nrm := int32(0), int32(0)
			if c.sigmoid {
				sig = 1
			}
			if c.norm {
				nrm = 1
			}
			if e := fn.LaunchOn(bg, stream, gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1},
				gc.Arg(dLog), gc.Arg(dBias), gc.Arg(dIdx), gc.Arg(dWgt),
				gc.ArgValue(int32(c.nE)), gc.ArgValue(int32(c.k)), gc.ArgValue(sig), gc.ArgValue(nrm),
				gc.ArgValue(float32(c.scale)), gc.ArgValue(int32(c.nGroup)), gc.ArgValue(int32(c.topkGroup))); e != nil {
				t.Fatalf("launch: %v", e)
			}
			if e := stream.Synchronize(bg); e != nil {
				t.Fatalf("sync: %v", e)
			}
			gotIdx := make([]uint32, c.k)
			gotWgt := make([]float32, c.k)
			if e := gc.CopyDtoH(bg, gotIdx, dIdx); e != nil {
				t.Fatalf("D2H idx: %v", e)
			}
			if e := gc.CopyDtoH(bg, gotWgt, dWgt); e != nil {
				t.Fatalf("D2H wgt: %v", e)
			}

			// EXACT on indices: a different expert is a different computation entirely.
			for j := 0; j < c.k; j++ {
				if int(gotIdx[j]) != wantIdx[j] {
					t.Fatalf("slot %d: expert %d, want %d — the GPU router selected a DIFFERENT expert. "+
						"That is not a numeric error: the token is routed through unrelated weights.\n"+
						"  got  %v\n  want %v", j, gotIdx[j], wantIdx[j], gotIdx, wantIdx)
				}
			}
			// Weights: f32 order only.
			for j := 0; j < c.k; j++ {
				if d := math.Abs(float64(gotWgt[j] - wantWgt[j])); d > 1e-5 {
					t.Errorf("slot %d (expert %d): weight %v, want %v (|d|=%.3g)", j, gotIdx[j], gotWgt[j], wantWgt[j], d)
				}
			}
			if c.norm {
				var sum float64
				for _, w := range gotWgt {
					sum += float64(w)
				}
				want := 1.0
				if c.scale != 0 && c.scale != 1 {
					want = c.scale // norm-then-scale ⇒ the weights sum to the scaling factor
				}
				if math.Abs(sum-want) > 1e-4 {
					t.Errorf("norm_topk_prob set but weights sum to %v, want %v", sum, want)
				}
			}
			t.Logf("idx=%v wgt=%v", gotIdx, gotWgt)
		})
	}
}

// cpuRoute is an INDEPENDENT reference for the router — deliberately written from the
// semantics (decoder's CPU MoE + gpu/moe.go), not transliterated from the kernel, so a shared
// misreading does not cancel out.
func cpuRoute(logits, bias []float32, nE, k int, sigmoid, norm bool, scale float64, nGroup, topkGroup int) ([]int, []float32) {
	score := make([]float64, nE)
	if sigmoid {
		for i := range score {
			score[i] = 1 / (1 + math.Exp(-float64(logits[i])))
		}
	} else {
		mx := float64(logits[0])
		for _, v := range logits {
			mx = math.Max(mx, float64(v))
		}
		var sum float64
		for i, v := range logits {
			score[i] = math.Exp(float64(v) - mx)
			sum += score[i]
		}
		for i := range score {
			score[i] /= sum
		}
	}
	sel := make([]float64, nE)
	for i := range sel {
		sel[i] = score[i] + float64(bias[i])
	}
	negInf := math.Inf(-1)
	if nGroup > 1 {
		gsz := nE / nGroup
		type gs struct {
			g int
			v float64
		}
		gscore := make([]gs, nGroup)
		for g := 0; g < nGroup; g++ {
			t1, t2 := negInf, negInf
			for i := g * gsz; i < (g+1)*gsz; i++ {
				if sel[i] > t1 {
					t1, t2 = sel[i], t1
				} else if sel[i] > t2 {
					t2 = sel[i]
				}
			}
			gscore[g] = gs{g, t1 + t2}
		}
		keep := make([]bool, nGroup)
		for j := 0; j < topkGroup; j++ {
			bg, bv := -1, negInf
			for _, x := range gscore {
				if !keep[x.g] && x.v > bv {
					bv, bg = x.v, x.g
				}
			}
			if bg >= 0 {
				keep[bg] = true
			}
		}
		for g := 0; g < nGroup; g++ {
			if !keep[g] {
				for i := g * gsz; i < (g+1)*gsz; i++ {
					sel[i] = negInf
				}
			}
		}
	}
	idx := make([]int, k)
	wgt := make([]float32, k)
	var wsum float64
	for j := 0; j < k; j++ {
		best, bv := 0, negInf
		for i := 0; i < nE; i++ {
			if sel[i] > bv {
				bv, best = sel[i], i
			}
		}
		idx[j] = best
		wgt[j] = float32(score[best]) // the raw score — bias steers selection only
		wsum += score[best]
		sel[best] = negInf
	}
	if norm && wsum > 0 {
		for j := range wgt {
			wgt[j] = float32(float64(wgt[j]) / wsum)
		}
	}
	if scale != 0 && scale != 1 {
		for j := range wgt {
			wgt[j] = float32(float64(wgt[j]) * scale)
		}
	}
	return idx, wgt
}
