//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestMoERouteSG_bitIdentical (D-P04): moe_route_sg and route_gptoss_sg, one simdgroup, write the same idx and the
// same weight bits as moe_route and route_gptoss, one thread, across every routing option the kernel takes: softmax
// and sigmoid scoring, renorm, routed scale, a selection bias, DeepSeek group limiting (including a group size that
// does not divide nE and more picks than unmasked experts), and inputs built to tie (logits on a 0.25 grid, every
// logit equal, -INFINITY entries). Ties are where a parallel top-k goes wrong: the serial scan takes the lowest index.
func TestMoERouteSG_bitIdentical(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(name string) Pipeline {
		p, err := d.NewComputePipeline(lib, name)
		if err != nil {
			t.Fatalf("pipeline %s: %v", name, err)
		}
		return p
	}
	pSer, pSG := pipe("moe_route"), pipe("moe_route_sg")
	pGSer, pGSG := pipe("route_gptoss"), pipe("route_gptoss_sg")
	q := d.NewCommandQueue()

	type cfg struct {
		nE, k, sigmoid, norm int
		scale                float32
		nGroup, topkGroup    int
		bias, gptoss         bool
	}
	cfgs := []cfg{
		{nE: 8, k: 2, norm: 1},
		{nE: 60, k: 4},
		{nE: 64, k: 8},
		{nE: 128, k: 8, norm: 1},
		{nE: 160, k: 6, scale: 16},
		{nE: 256, k: 8, norm: 1, scale: 2.5},
		{nE: 256, k: 8, sigmoid: 1, norm: 1, scale: 2.5, nGroup: 8, topkGroup: 4, bias: true},
		{nE: 160, k: 6, sigmoid: 1, norm: 1, nGroup: 8, topkGroup: 3, bias: true},
		{nE: 60, k: 4, sigmoid: 1, norm: 1, nGroup: 8, topkGroup: 2, bias: true}, // gsz 7: the last 4 experts are in no group
		{nE: 16, k: 6, sigmoid: 1, norm: 1, nGroup: 8, topkGroup: 2},             // 4 unmasked experts, 6 picks
		{nE: 128, k: 8, sigmoid: 1, scale: 1.5, bias: true},
		{nE: 32, k: 4, gptoss: true, bias: true},
		{nE: 128, k: 4, gptoss: true, bias: true},
		{nE: 32, k: 4, gptoss: true}, // no bias: ties come straight from the logits
	}
	fills := []string{"normal", "grid", "equal", "neginf", "wide"}
	rng := rand.New(rand.NewSource(404))
	cases, tied := 0, 0
	for _, c := range cfgs {
		for _, fill := range fills {
			for rep := range 6 {
				name := fmt.Sprintf("nE%d_k%d_sig%d_g%d_gptoss%v/%s/%d", c.nE, c.k, c.sigmoid, c.nGroup, c.gptoss, fill, rep)
				logits, bias := make([]float32, c.nE), make([]float32, c.nE)
				for i := range logits {
					switch fill {
					case "normal":
						logits[i] = float32(rng.NormFloat64())
					case "grid": // a 0.25 grid over [-1, 1]: nine values for up to 256 experts, ties everywhere
						logits[i] = float32(rng.Intn(9)-4) * 0.25
					case "equal":
						logits[i] = 0.5
					case "neginf":
						logits[i] = float32(rng.NormFloat64())
						if rng.Intn(3) == 0 {
							logits[i] = float32(math.Inf(-1))
						}
					case "wide":
						logits[i] = float32(rng.NormFloat64() * 30)
					}
					if c.bias {
						bias[i] = float32(rng.Intn(5)-2) * 0.05 // a grid too, so the bias makes ties of its own
					}
				}
				if fill == "neginf" && c.sigmoid == 0 && !c.gptoss {
					logits[rng.Intn(c.nE)] = 1 // softmax over all -INFINITY is NaN in both; keep one finite
				}
				run := func(p Pipeline, threads int) ([]uint32, []uint32) {
					dIdx := NewBufferUint32s(d, make([]uint32, c.k))
					dW := d.NewBufferLen(c.k)
					dL, dB := NewBufferFloats(d, logits), NewBufferFloats(d, bias)
					uNE, uK := NewBufferU32(d, uint32(c.nE)), NewBufferU32(d, uint32(c.k))
					if c.gptoss {
						q.Run1D(p, threads, threads, dL, dB, dIdx, dW, uNE, uK)
					} else {
						q.Run1D(p, threads, threads, dL, dB, dIdx, dW, uNE, uK,
							NewBufferU32(d, uint32(c.sigmoid)), NewBufferU32(d, uint32(c.norm)),
							NewBufferFloats(d, []float32{c.scale}),
							NewBufferU32(d, uint32(c.nGroup)), NewBufferU32(d, uint32(c.topkGroup)))
					}
					w := dW.Floats()
					wb := make([]uint32, len(w))
					for i, x := range w {
						wb[i] = math.Float32bits(x)
					}
					return dIdx.U32s(), wb
				}
				ser, sg := pSer, pSG
				if c.gptoss {
					ser, sg = pGSer, pGSG
				}
				wantIdx, wantW := run(ser, 1)
				gotIdx, gotW := run(sg, 32)
				cases++
				if hasTie(logits, bias) {
					tied++
				}
				for j := range c.k {
					if gotIdx[j] != wantIdx[j] || gotW[j] != wantW[j] {
						t.Errorf("%s: slot %d: sg idx %d wgt %08x, serial idx %d wgt %08x (sg %v, serial %v)",
							name, j, gotIdx[j], gotW[j], wantIdx[j], wantW[j], gotIdx, wantIdx)
						break
					}
				}
			}
		}
	}
	if tied < cases/3 {
		t.Fatalf("only %d of %d cases have a tied score: the tie-break this exists for is barely exercised", tied, cases)
	}
	t.Logf("%d cases (%d with tied scores): idx and weight bits identical", cases, tied)
}

func hasTie(logits, bias []float32) bool {
	seen := map[float32]bool{}
	for i, x := range logits {
		v := x + bias[i]
		if seen[v] {
			return true
		}
		seen[v] = true
	}
	return false
}
