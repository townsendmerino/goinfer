//go:build gpu

package kernels

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/vision"

	"github.com/townsendmerino/goinfer/multimodal"
)

// Qwen2.5-VL's vision tower on WebGPU, on the grid engine (grid_vision.go), in float32 from aikit's GPUWeights. It
// plugs into aikit's resident seam, vision.RegisterQwenResident, so EnableResident routes the tower's ForwardViT here.
//
// aikit's BuildWindowPlan runs on the host (the window permutation, both segmentations, RoPE in window order). The
// pixel rows are permuted into window order before the patch embed (a per-row matmul, so this equals permuting its
// output), the blocks attend per window except the full-attention ones (per image frame), and the output is permuted
// back to the original order; aikit's merger (MergeHidden) stays on the host, where Forward calls it.

func init() {
	vision.RegisterQwenResident(func(enc *vision.QwenVisionEncoder) (vision.QwenResidentEncoder, error) {
		return newQwen25VResident(enc)
	})
	multimodal.MarkResidentTower(multimodal.TowerQwen25VL, "webgpu") // aikit's slot cannot be asked what is registered
}

// qwen25VResident is the Qwen2.5-VL tower on WebGPU (vision.QwenResidentEncoder).
type qwen25VResident struct {
	a     *gridVTower
	enc   *vision.QwenVisionEncoder
	merge int
	full  map[int]bool // the full-attention blocks
}

func newQwen25VResident(enc *vision.QwenVisionEncoder) (*qwen25VResident, error) {
	w := enc.GPUWeights()
	a, err := newGridVTower(gvkQwen25, w.Hidden, w.Inter, w.NumHeads, w.PatchDim, 1e-6) // aikit's rmsNorm: eps 1e-6
	if err != nil {
		return nil, err
	}
	f32 := func(name string, m vision.QwenGPUMat, bias []float32) vision.VisionProj {
		if a.uerr == nil && (m.Quantized || len(m.F32) != m.Rows*m.Cols) {
			a.uerr = fmt.Errorf("the tower runs float32 (load it with quant=false); %s is not", name)
		}
		return vision.VisionProj{W: m.F32, B: bias, Out: m.Rows, In: m.Cols}
	}
	if len(w.PatchW) != w.Hidden*w.PatchDim {
		a.uerr = fmt.Errorf("patch embed has %d weights, want [%d, %d]", len(w.PatchW), w.Hidden, w.PatchDim)
	}
	a.patchW = a.up(w.PatchW) // no patch bias
	r := &qwen25VResident{a: a, enc: enc, merge: w.SpatialMergeSize, full: map[int]bool{}}
	for _, li := range w.FullattBlockIndexes {
		r.full[li] = true
	}
	for i, b := range w.Blocks {
		n := func(s string) string { return fmt.Sprintf("block %d %s", i, s) }
		qkv := f32(n("qkv"), b.QKVw, b.QKVb)
		var q, k, v gvProj
		if a.uerr == nil {
			q, k, v = a.splitQKV(qkv)
		}
		a.blocks = append(a.blocks, gvBlock{norm1w: a.up(b.Norm1w), norm2w: a.up(b.Norm2w), q: q, k: k, v: v,
			proj: a.proj(n("proj"), f32(n("proj"), b.Projw, b.Projb)), gate: a.proj(n("gate"), f32(n("gate"), b.Gatew, b.Gateb)),
			up: a.proj(n("up"), f32(n("up"), b.Upw, b.Upb)), down: a.proj(n("down"), f32(n("down"), b.Downw, b.Downb))})
	}
	if _, err := a.built("Qwen2.5-VL"); err != nil {
		return nil, err
	}
	return r, nil
}

// segsOf turns per-row [start, end) segment bounds into cumulative offsets.
func segsOf(start, end []int32) ([]int, error) {
	segs := []int{0}
	for i := 0; i < len(end); {
		if int(start[i]) != i || int(end[i]) <= i {
			return nil, fmt.Errorf("gpu: segment bounds [%d, %d) at row %d are not contiguous", start[i], end[i], i)
		}
		i = int(end[i])
		segs = append(segs, i)
	}
	return segs, nil
}

// ForwardViT is the pre-merge hidden state [n_patches, hidden] in the original patch order (vision.QwenResidentEncoder).
func (r *qwen25VResident) ForwardViT(pixels []float32, gridTHW [][3]int) ([]float32, error) {
	a := r.a
	plan, err := r.enc.BuildWindowPlan(gridTHW)
	if err != nil {
		return nil, err
	}
	np, H, pd := plan.NPatches, a.hidden, a.patchDim
	if len(pixels) != np*pd {
		return nil, fmt.Errorf("gpu: %d pixel values for %d patches of %d", len(pixels), np, pd)
	}
	win, err := segsOf(plan.WinStart, plan.WinEnd)
	if err != nil {
		return nil, err
	}
	fullSegs, err := segsOf(plan.FullStart, plan.FullEnd)
	if err != nil {
		return nil, err
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	// order[i] is the original row that window-order row i holds.
	mu := r.merge * r.merge
	order := make([]int, np)
	for g, src := range plan.WinIdx {
		for u := range mu {
			order[g*mu+u] = src*mu + u
		}
	}
	if a.planted.noWindowOrder {
		for i := range order {
			order[i] = i
		}
	}
	pt := make([]float32, np*pd)
	for i, src := range order {
		copy(pt[i*pd:(i+1)*pd], pixels[src*pd:(src+1)*pd])
	}
	h, err := a.run(gvInput{pixels: pt, np: np, cos: plan.Cos, sin: plan.Sin, segs: func(bi int) []int {
		if r.full[bi] || a.planted.noWindows {
			return fullSegs
		}
		return win
	}}, nil)
	if err != nil {
		return nil, err
	}
	out := make([]float32, np*H)
	for i, dst := range order {
		copy(out[dst*H:(dst+1)*H], h[i*H:(i+1)*H])
	}
	return out, nil
}

func (r *qwen25VResident) Close() { r.a.Close() }
