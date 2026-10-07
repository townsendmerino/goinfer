//go:build darwin

package metal

import (
	"fmt"

	"github.com/townsendmerino/aikit/vision"
)

// Gemma 3's SigLIP tower and Qwen2.5-VL's tower on Metal (S3 of docs/tasks/task-multimodal-support-2026-10.md), on the
// same base as the Qwen3.5+ and GLM-OCR towers (grid_vision.go), in float32. They plug into aikit's own resident seams,
// vision.RegisterResident and vision.RegisterQwenResident, so serve's EnableResident routes the tower's Forward here
// under --backend metal. aikit's tagged gpu/visionmetal and gpu/qwenmetal register on the same seams and are not
// imported: G-S3a measured the first wrong at real size and the second no faster than the CPU.
//
// SigLIP: aikit's GridPatches (im2col) on the host, the patch embed plus the fixed position table, every layer over one
// segment of all the patches with no RoPE, then FinishHidden (the post-layernorm) on the host. Float32 weights
// (LoadEncoder quant=false), from Encoder.Weights.
//
// Qwen2.5-VL: aikit's BuildWindowPlan on the host (the window permutation, both segmentations, RoPE in window order). The
// pixel rows are permuted into window order before the patch embed (a per-row matmul, so this equals permuting its
// output), the blocks attend per window except the full-attention ones (per image frame), and the output is permuted
// back to the original order; aikit's merger (MergeHidden) stays on the host, where Forward calls it. Float32 weights
// from GPUWeights' float32 form.

func init() {
	vision.RegisterResident(func(enc *vision.Encoder) (vision.ResidentEncoder, error) {
		return newSiglipVResident(enc)
	})
	vision.RegisterQwenResident(func(enc *vision.QwenVisionEncoder) (vision.QwenResidentEncoder, error) {
		return newQwen25VResident(enc)
	})
}

// siglipVResident is the SigLIP tower on Metal (vision.ResidentEncoder).
type siglipVResident struct {
	a   *gridVAccel
	enc *vision.Encoder
	np  int
}

func newSiglipVResident(enc *vision.Encoder) (*siglipVResident, error) {
	w, err := enc.Weights()
	if err != nil {
		return nil, fmt.Errorf("metal: the SigLIP tower runs float32 (load it with quant=false): %w", err)
	}
	c := w.Cfg
	a, err := newGridVAccel(gvkSiglip, c.HiddenSize, c.IntermediateSize, c.NumAttentionHeads,
		c.NumChannels*c.PatchSize*c.PatchSize, c.LayerNormEps)
	if err != nil {
		return nil, err
	}
	if len(w.PosEmb) != w.NumPatches*c.HiddenSize {
		return nil, fmt.Errorf("metal: SigLIP position table %d values for %d patches of %d", len(w.PosEmb), w.NumPatches, c.HiddenSize)
	}
	a.patchW, a.patchB = a.up(w.PatchW), a.up(w.PatchB)
	pos := w.PosEmb
	a.posEmbeds = func([][3]int) []float32 { return pos }
	for _, b := range w.Blocks {
		a.blocks = append(a.blocks, gvBlock{norm1w: a.up(b.LN1W), norm1b: a.up(b.LN1B), norm2w: a.up(b.LN2W), norm2b: a.up(b.LN2B),
			q: a.proj(b.Q), k: a.proj(b.K), v: a.proj(b.V), proj: a.proj(b.O), fc1: a.proj(b.FC1), fc2: a.proj(b.FC2)})
	}
	return &siglipVResident{a: a, enc: enc, np: w.NumPatches}, nil
}

// hidden is the last block's output for GridPatches' rows (the stage FinishHidden takes).
func (r *siglipVResident) hidden(patches []float32) ([]float32, error) {
	a, np := r.a, r.np
	if len(patches) != np*a.patchDim {
		return nil, fmt.Errorf("metal: %d patch values for %d patches of %d", len(patches), np, a.patchDim)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.grow(np, np)
	copy(a.pt.Floats()[:np*a.patchDim], patches)
	if err := a.stagePos(a.posEmbeds(nil), np); err != nil {
		return nil, err
	}
	segs := []int{0, np}
	return a.run(np, func(int) []int { return segs })
}

// ForwardPatches is last_hidden_state for GridPatches' rows (vision.ResidentEncoder).
func (r *siglipVResident) ForwardPatches(patches []float32) ([]float32, error) {
	h, err := r.hidden(patches)
	if err != nil {
		return nil, err
	}
	return r.enc.FinishHidden(h)
}

func (r *siglipVResident) Close() {}

// qwen25VResident is the Qwen2.5-VL tower on Metal (vision.QwenResidentEncoder).
type qwen25VResident struct {
	a     *gridVAccel
	enc   *vision.QwenVisionEncoder
	merge int
}

func newQwen25VResident(enc *vision.QwenVisionEncoder) (*qwen25VResident, error) {
	w := enc.GPUWeights()
	f32 := func(name string, m vision.QwenGPUMat) ([]float32, error) {
		if m.Quantized || len(m.F32) != m.Rows*m.Cols {
			return nil, fmt.Errorf("metal: the Qwen2.5-VL tower runs float32 (load it with quant=false); %s is not", name)
		}
		return m.F32, nil
	}
	a, err := newGridVAccel(gvkQwen25, w.Hidden, w.Inter, w.NumHeads, w.PatchDim, 1e-6) // aikit's rmsNorm: eps 1e-6
	if err != nil {
		return nil, err
	}
	a.patchW = a.up(w.PatchW) // no patch bias
	full := map[int]bool{}
	for _, li := range w.FullattBlockIndexes {
		full[li] = true
	}
	for i, b := range w.Blocks {
		n := func(s string) string { return fmt.Sprintf("block %d %s", i, s) }
		var ws [5][]float32
		for j, m := range []struct {
			name string
			m    vision.QwenGPUMat
		}{{"qkv", b.QKVw}, {"proj", b.Projw}, {"gate", b.Gatew}, {"up", b.Upw}, {"down", b.Downw}} {
			if ws[j], err = f32(n(m.name), m.m); err != nil {
				return nil, err
			}
		}
		q, k, v, err := a.splitQKV(vision.VisionProj{W: ws[0], B: b.QKVb, Out: b.QKVw.Rows, In: b.QKVw.Cols})
		if err != nil {
			return nil, err
		}
		p := func(w []float32, bias []float32, m vision.QwenGPUMat) gvProj {
			return a.proj(vision.VisionProj{W: w, B: bias, Out: m.Rows, In: m.Cols})
		}
		a.blocks = append(a.blocks, gvBlock{norm1w: a.up(b.Norm1w), norm2w: a.up(b.Norm2w), q: q, k: k, v: v,
			proj: p(ws[1], b.Projb, b.Projw), gate: p(ws[2], b.Gateb, b.Gatew), up: p(ws[3], b.Upb, b.Upw),
			down: p(ws[4], b.Downb, b.Downw), full: full[i]})
	}
	return &qwen25VResident{a: a, enc: enc, merge: w.SpatialMergeSize}, nil
}

// segsOf turns per-row [start, end) segment bounds into cumulative offsets.
func segsOf(start, end []int32) ([]int, error) {
	segs := []int{0}
	for i := 0; i < len(end); {
		if int(start[i]) != i || int(end[i]) <= i {
			return nil, fmt.Errorf("metal: segment bounds [%d, %d) at row %d are not contiguous", start[i], end[i], i)
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
		return nil, fmt.Errorf("metal: %d pixel values for %d patches of %d", len(pixels), np, pd)
	}
	win, err := segsOf(plan.WinStart, plan.WinEnd)
	if err != nil {
		return nil, err
	}
	fullSegs, err := segsOf(plan.FullStart, plan.FullEnd)
	if err != nil {
		return nil, err
	}
	// order[i] is the original row that window-order row i holds; the identity under G-S3a's planted defect (4).
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
	a.mu.Lock()
	defer a.mu.Unlock()
	a.grow(np, max(maxSegment(win), maxSegment(fullSegs)))
	pt := a.pt.Floats()
	for i, src := range order {
		copy(pt[i*pd:(i+1)*pd], pixels[src*pd:(src+1)*pd])
	}
	if err := a.stageRope(plan.Cos, plan.Sin, np); err != nil {
		return nil, err
	}
	h, err := a.run(np, func(bi int) []int {
		if a.blocks[bi].full || a.planted.noWindows { // G-S3a (3): every block attends its whole frame
			return fullSegs
		}
		return win
	})
	if err != nil {
		return nil, err
	}
	out := make([]float32, np*H)
	for i, dst := range order {
		copy(out[dst*H:(dst+1)*H], h[i*H:(i+1)*H])
	}
	return out, nil
}

func (r *qwen25VResident) Close() {}
