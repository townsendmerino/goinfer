//go:build gpu && goinfer_testhooks

package gpu_test

import (
	"math"
	"sort"
	"testing"

	"github.com/townsendmerino/goinfer/gpu"
)

// refRoute is the nGroup==1 reference for the GPU router top-k kernel — the same
// selection decoder.routeExperts performs (softmax/sigmoid score, +bias, top-k by
// selection score, weight = un-biased score, optional renorm + scale). Kept here so the
// gpu test stays self-contained (routeExperts is unexported in the decoder package).
func refRoute(logits, bias []float32, k int, sigmoid, norm bool, scale float32) (idx []int, wts []float32) {
	nE := len(logits)
	score := make([]float64, nE)
	if sigmoid {
		for i, l := range logits {
			score[i] = 1.0 / (1.0 + math.Exp(-float64(l)))
		}
	} else {
		mx := math.Inf(-1)
		for _, l := range logits {
			mx = math.Max(mx, float64(l))
		}
		var sum float64
		for i, l := range logits {
			score[i] = math.Exp(float64(l) - mx)
			sum += score[i]
		}
		for i := range score {
			score[i] /= sum
		}
	}
	sel := append([]float64(nil), score...)
	if bias != nil {
		for i := range sel {
			sel[i] += float64(bias[i])
		}
	}
	order := make([]int, nE)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return sel[order[a]] > sel[order[b]] })
	idx = order[:k]
	wts = make([]float32, k)
	var wsum float64
	for j, e := range idx {
		wts[j] = float32(score[e])
		wsum += score[e]
	}
	if norm && wsum > 0 {
		for j := range wts {
			wts[j] = float32(float64(wts[j]) / wsum)
		}
	}
	if scale != 0 && scale != 1 {
		for j := range wts {
			wts[j] *= scale
		}
	}
	return idx, wts
}

// TestMoERoute_parity gates the C3a GPU router top-k kernel against the reference for
// the routing flavors the MoE families use (nGroup==1): Mixtral (softmax + renorm),
// Qwen2-MoE (softmax, no renorm), and the DeepSeek/GLM sigmoid + selection-bias + scale
// path (group-limit deferred to C3d). Same chosen experts (as a set; the kernel emits
// them in descending selection order, matching the reference sort) and weights to f32 tol.
func TestMoERoute_parity(t *testing.T) {
	c, err := gpu.New()
	if err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	defer c.Close()

	// A fixed, non-degenerate logit set (8 experts) + a bias for the deepseek case.
	logits := []float32{0.4, -1.2, 2.1, 0.05, 1.7, -0.3, 0.9, -2.0}
	bias := []float32{0.1, 0.0, -0.5, 0.2, 0.3, -0.1, 0.0, 0.4}
	cases := []struct {
		name          string
		k             int
		sigmoid, norm bool
		scale         float32
		bias          []float32
	}{
		{"mixtral softmax+norm", 2, false, true, 1, nil},
		{"qwen2moe softmax", 4, false, false, 1, nil},
		{"deepseek sigmoid+bias+scale", 3, true, false, 2.5, bias},
	}
	for _, tc := range cases {
		gi, gw, err := c.RouteExpertsForTest(logits, tc.bias, tc.k, tc.sigmoid, tc.norm, tc.scale)
		if err != nil {
			t.Fatalf("%s: RouteExpertsForTest: %v", tc.name, err)
		}
		ri, rw := refRoute(logits, tc.bias, tc.k, tc.sigmoid, tc.norm, tc.scale)
		for j := 0; j < tc.k; j++ {
			if gi[j] != ri[j] {
				t.Errorf("%s: idx[%d] gpu=%d ref=%d (gpu=%v ref=%v)", tc.name, j, gi[j], ri[j], gi, ri)
			}
			if d := math.Abs(float64(gw[j] - rw[j])); d > 1e-5 {
				t.Errorf("%s: wgt[%d] gpu=%.6f ref=%.6f (Δ%.2g)", tc.name, j, gw[j], rw[j], d)
			}
		}
		t.Logf("%s: idx=%v wgt=%v", tc.name, gi, gw)
	}
}
