//go:build cuda

package cuda

import (
	"fmt"
	"math"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// Qwen2.5-VL's vision tower in float32 on the CUDA tower base (docs/tasks/task-multimodal-support-2026-10.md): the port
// of metal/vl_towers.go's qwen25VResident, replacing aikit's gpu/qwencuda (unfused attention, and an allocation that
// failed beside the decoder at serve's defaults). Everything positional comes from aikit's own exports, not a
// reimplementation:
// BuildWindowPlan gives the window permutation, both segmentations and the RoPE tables in window order; the pixel rows are permuted into window order before the patch embed (a per-row matmul, so this equals permuting
// its output), each block attends per window except the full-attention blocks (per frame), and the result is permuted back to the original patch order; the merger (MergeHidden) stays aikit's, on the host, where Forward calls it.
// Block: RMSNorm (eps 1e-6, aikit's rmsNorm), biased q/k/v from the fused qkv split at upload, NeoX rotate-half RoPE, attention at scale 1/sqrt(head_dim) through the fused kernel (tower_base.cu), biased proj and residual,
// RMSNorm, the SiLU-gated biased MLP, residual. Float32 only: an int8-loaded encoder declines, as Metal's does.

func init() {
	vision.RegisterQwenResident(func(enc *vision.QwenVisionEncoder) (vision.QwenResidentEncoder, error) {
		return newQwen25Tower(enc)
	})
	multimodal.MarkResidentTower(multimodal.TowerQwen25VL, "cuda") // S8: aikit's slot cannot be asked what is registered
}

type qwen25Tower struct {
	g     *gridTower
	enc   *vision.QwenVisionEncoder
	merge int
	full  []bool // per block: attends per frame, not per window
}

func newQwen25Tower(enc *vision.QwenVisionEncoder) (*qwen25Tower, error) {
	w := enc.GPUWeights()
	f32 := func(name string, m vision.QwenGPUMat) ([]float32, error) {
		if m.Quantized || len(m.F32) != m.Rows*m.Cols {
			return nil, fmt.Errorf("cuda: the Qwen2.5-VL tower runs float32 (load it with quant=false); %s is not", name)
		}
		return m.F32, nil
	}
	// The intermediate width is padded up to a multiple of 64 with zeros: Qwen2.5-VL-3B's is 3420 (3420 % 16 = 12), which
	// would send the down projection (K = 3420) and the biased epilogue through the tiled GEMM, about half the speed of the
	// register-blocked kernel. The extra gate and up rows and bias entries are zero, so silu(0) * 0 = 0 exactly and the
	// padded columns of down add nothing: the arithmetic is the unpadded tower's.
	inter := (w.Inter + 63) / 64 * 64
	g, err := newGridTower(gridQwen25, w.Hidden, inter, w.NumHeads, w.PatchDim, 1e-6) // aikit's rmsNorm: eps 1e-6
	if err != nil {
		return nil, err
	}
	r := &qwen25Tower{g: g, enc: enc, merge: w.SpatialMergeSize, full: make([]bool, len(w.Blocks))}
	for _, li := range w.FullattBlockIndexes {
		if li >= 0 && li < len(r.full) {
			r.full[li] = true
		}
	}
	if _, err := g.build(func() error {
		g.patchW = g.ops.up(w.PatchW) // no patch bias
		for i, b := range w.Blocks {
			n := func(s string) string { return fmt.Sprintf("block %d %s", i, s) }
			var ws [5][]float32
			for j, m := range []struct {
				name string
				m    vision.QwenGPUMat
			}{{"qkv", b.QKVw}, {"proj", b.Projw}, {"gate", b.Gatew}, {"up", b.Upw}, {"down", b.Downw}} {
				var e error
				if ws[j], e = f32(n(m.name), m.m); e != nil {
					return e
				}
			}
			q, k, v, e := g.splitQKV(vision.VisionProj{W: ws[0], B: b.QKVb, Out: b.QKVw.Rows, In: b.QKVw.Cols})
			if e != nil {
				return e
			}
			p := func(wt, bias []float32, m vision.QwenGPUMat) gridProj {
				return g.proj(vision.VisionProj{W: wt, B: bias, Out: m.Rows, In: m.Cols})
			}
			// gate and up [Inter, hidden] get zero rows (and zero bias entries) up to the padded width; down [hidden, Inter] gets zero columns.
			padRows := func(wt, bias []float32, m vision.QwenGPUMat) gridProj {
				pw := make([]float32, inter*m.Cols)
				copy(pw, wt)
				var pb []float32
				if len(bias) > 0 {
					pb = make([]float32, inter)
					copy(pb, bias)
				}
				return g.proj(vision.VisionProj{W: pw, B: pb, Out: inter, In: m.Cols})
			}
			pd := make([]float32, b.Downw.Rows*inter)
			for r := 0; r < b.Downw.Rows; r++ {
				copy(pd[r*inter:r*inter+b.Downw.Cols], ws[4][r*b.Downw.Cols:(r+1)*b.Downw.Cols])
			}
			g.blocks = append(g.blocks, gridBlock{norm1w: g.ops.up(b.Norm1w), norm2w: g.ops.up(b.Norm2w), q: q, k: k, v: v,
				proj: p(ws[1], b.Projb, b.Projw), gate: padRows(ws[2], b.Gateb, b.Gatew), up: padRows(ws[3], b.Upb, b.Upw),
				down: g.proj(vision.VisionProj{W: pd, B: b.Downb, Out: b.Downw.Rows, In: inter})})
		}
		return nil
	}); err != nil {
		return nil, err
	}
	return r, nil
}

// qwenSegsOf turns per-row [start, end) segment bounds into cumulative offsets, refusing bounds that are not contiguous.
func qwenSegsOf(start, end []int32) ([]int, error) {
	segs := []int{0}
	for i := 0; i < len(end); {
		if int(start[i]) != i || int(end[i]) <= i {
			return nil, fmt.Errorf("cuda: segment bounds [%d, %d) at row %d are not contiguous", start[i], end[i], i)
		}
		i = int(end[i])
		segs = append(segs, i)
	}
	return segs, nil
}

// ForwardViT is the pre-merge hidden state [n_patches, hidden] in the original patch order (vision.QwenResidentEncoder).
func (r *qwen25Tower) ForwardViT(pixels []float32, gridTHW [][3]int) ([]float32, error) {
	a := r.g
	plan, err := r.enc.BuildWindowPlan(gridTHW)
	if err != nil {
		return nil, err
	}
	np, H, I, nH, hd, pd := plan.NPatches, a.hidden, a.inter, a.heads, a.hd, a.patchDim
	if len(pixels) != np*pd {
		return nil, fmt.Errorf("cuda: %d pixel values for %d patches of %d", len(pixels), np, pd)
	}
	if len(plan.Cos) != np*hd || len(plan.Sin) != np*hd {
		return nil, fmt.Errorf("cuda: rope tables %d/%d for %d patches of head_dim %d", len(plan.Cos), len(plan.Sin), np, hd)
	}
	win, err := qwenSegsOf(plan.WinStart, plan.WinEnd)
	if err != nil {
		return nil, err
	}
	fullSegs, err := qwenSegsOf(plan.FullStart, plan.FullEnd)
	if err != nil {
		return nil, err
	}
	// order[i] is the original row that window-order row i holds; the identity under the planted window-order defect.
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
	cos, sin := plan.Cos, plan.Sin
	if a.planted.swapRope { // the row and column quarters swapped within each half of the cat(f, f) row
		cos, sin = append([]float32(nil), cos...), append([]float32(nil), sin...)
		half, q := hd/2, hd/4
		for i := range np {
			for base := 0; base < hd; base += half {
				for _, tb := range [][]float32{cos, sin} {
					row := tb[i*hd+base : i*hd+base+half]
					for j := range q {
						row[j], row[j+q] = row[j+q], row[j]
					}
				}
			}
		}
	}
	px := make([]float32, np*pd)
	for i, src := range order {
		copy(px[i*pd:(i+1)*pd], pixels[src*pd:(src+1)*pd])
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	var h []float32
	err = a.ops.do(func() error {
		defer a.ops.releaseScratch()
		o := a.ops
		x, xn, pt := o.af(np*H), o.af(np*H), o.af(np*pd)
		q, k, v, ctx, tmp := o.af(np*H), o.af(np*H), o.af(np*H), o.af(np*H), o.af(np*H)
		f1, f2 := o.af(np*I), o.af(np*I)
		csB, snB := o.af(np*hd), o.af(np*hd)
		for _, u := range []struct {
			b Buffer
			v []float32
		}{{pt, px}, {csB, cos}, {snB, sin}} {
			if e := gpu.Upload(u.b, u.v); e != nil {
				return e
			}
		}
		a.linear(pt, &gridProj{w: a.patchW, out: H, in: pd}, x, np) // no patch bias
		scale := float32(1 / math.Sqrt(float64(hd)))
		if a.planted.noScale {
			scale = 1
		}
		for bi := range a.blocks {
			b := &a.blocks[bi]
			a.norm(x, b.norm1w, Buffer{}, xn, np)
			a.linear(xn, &b.q, q, np)
			a.linear(xn, &b.k, k, np)
			a.linear(xn, &b.v, v, np)
			o.ropeHalfTo(q, csB, snB, np, nH, hd)
			o.ropeHalfTo(k, csB, snB, np, nH, hd)
			segs := win
			if r.full[bi] || a.planted.noWindows { // the full-attention blocks, or the planted defect: every block attends its whole frame
				segs = fullSegs
			}
			for si := 1; si < len(segs); si++ {
				off, T := segs[si-1]*H*4, segs[si]-segs[si-1]
				if e := o.attention(q.At(off), k.At(off), v.At(off), ctx.At(off), T, nH, hd, scale); e != nil {
					return e
				}
			}
			a.linearAdd(ctx, &b.proj, x, tmp, np)
			a.norm(x, b.norm2w, Buffer{}, xn, np)
			a.linear(xn, &b.gate, f1, np)
			a.linear(xn, &b.up, f2, np)
			o.cls(clsElem)
			o.launch(o.vit.SiLUMul, gpu.Grid1D(np*I, 256), Arg(f1), Arg(f2), i32(int32(np*I)))
			a.linearAdd(f1, &b.down, x, tmp, np)
		}
		if e := o.finish(); e != nil {
			return e
		}
		h = make([]float32, np*H)
		return gpu.Download(x, h)
	})
	if err != nil {
		return nil, fmt.Errorf("cuda: Qwen2.5-VL tower: %w", err)
	}
	out := make([]float32, np*H)
	for i, dst := range order {
		copy(out[dst*H:(dst+1)*H], h[i*H:(i+1)*H])
	}
	return out, nil
}

func (r *qwen25Tower) Close() { _ = r.g.Close() }
