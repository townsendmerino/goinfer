//go:build cuda

package cuda

import (
	"fmt"
	"math"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/vision"
)

// Gemma 3's SigLIP tower in float32 on the CUDA tower base (docs/tasks/task-multimodal-support-2026-10.md): the port of
// metal's siglipVResident (metal/vl_towers.go). aikit's GridPatches (im2col) rows arrive on the host; the patch embed
// plus the fixed position table, every block over ONE segment of all the patches with no RoPE, then FinishHidden (the
// post-layernorm) on the host. Blocks: LayerNorm with bias, biased separate q, k and v, attention at scale
// 1/sqrt(head_dim) over every patch, biased o and residual, LayerNorm, biased fc1, GELU-tanh, biased fc2, residual.
// Float32 weights from Encoder.Weights (LoadEncoder quant=false). Registered through cuda/vision_register.go's
// dispatching factory, which keeps the int8 NewVisionEncoder for an int8 encoder.

// siglipTower is the SigLIP tower on CUDA (vision.ResidentEncoder).
type siglipTower struct {
	g   *gridTower
	enc *vision.Encoder
	np  int
	pos []float32 // aikit's fixed position table, [np, hidden]
}

func newSiglipTower(enc *vision.Encoder) (*siglipTower, error) {
	w, err := enc.Weights()
	if err != nil {
		return nil, fmt.Errorf("cuda: the SigLIP tower runs float32 (load it with quant=false): %w", err)
	}
	c := w.Cfg
	if len(w.PosEmb) != w.NumPatches*c.HiddenSize {
		return nil, fmt.Errorf("cuda: SigLIP position table %d values for %d patches of %d", len(w.PosEmb), w.NumPatches, c.HiddenSize)
	}
	g, err := newGridTower(gridSiglip, c.HiddenSize, c.IntermediateSize, c.NumAttentionHeads, c.NumChannels*c.PatchSize*c.PatchSize, float64(c.LayerNormEps))
	if err != nil {
		return nil, err
	}
	if _, err := g.build(func() error {
		g.patchW, g.patchB = g.ops.up(w.PatchW), g.ops.up(w.PatchB)
		for _, b := range w.Blocks {
			g.blocks = append(g.blocks, gridBlock{norm1w: g.ops.up(b.LN1W), norm1b: g.ops.up(b.LN1B), norm2w: g.ops.up(b.LN2W), norm2b: g.ops.up(b.LN2B),
				q: g.proj(b.Q), k: g.proj(b.K), v: g.proj(b.V), proj: g.proj(b.O), fc1: g.proj(b.FC1), fc2: g.proj(b.FC2)})
		}
		return nil
	}); err != nil {
		return nil, err
	}
	return &siglipTower{g: g, enc: enc, np: w.NumPatches, pos: w.PosEmb}, nil
}

// hidden is the last block's output for GridPatches' rows, [np, hidden] (the stage FinishHidden takes).
func (r *siglipTower) hidden(patches []float32) ([]float32, error) {
	a, np := r.g, r.np
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(patches) != np*a.patchDim {
		return nil, fmt.Errorf("cuda: %d patch values for %d patches of %d", len(patches), np, a.patchDim)
	}
	H, I, nH, hd := a.hidden, a.inter, a.heads, a.hd
	var out []float32
	err := a.ops.do(func() error {
		defer a.ops.releaseScratch()
		o := a.ops
		x, xn, pt := o.af(np*H), o.af(np*H), o.af(np*a.patchDim)
		q, k, v, ctx, tmp := o.af(np*H), o.af(np*H), o.af(np*H), o.af(np*H), o.af(np*H)
		f1 := o.af(np * I)
		posB := o.af(np * H)
		if e := gpu.Upload(pt, patches); e != nil {
			return e
		}
		if e := gpu.Upload(posB, r.pos); e != nil {
			return e
		}
		pb := a.patchB
		if a.planted.noPatchBias {
			pb = Buffer{}
		}
		a.linear(pt, &gridProj{w: a.patchW, b: pb, out: H, in: a.patchDim}, x, np)
		if !a.planted.noPosEmbed {
			o.addVec(x, posB, np*H)
		}
		scale := float32(1 / math.Sqrt(float64(hd)))
		if a.planted.noScale {
			scale = 1
		}
		for bi := range a.blocks {
			b := &a.blocks[bi]
			a.norm(x, b.norm1w, b.norm1b, xn, np)
			a.linear(xn, &b.q, q, np)
			a.linear(xn, &b.k, k, np)
			a.linear(xn, &b.v, v, np)
			if e := o.attention(q, k, v, ctx, np, nH, hd, scale); e != nil {
				return e
			}
			a.linearAdd(ctx, &b.proj, x, tmp, np)
			a.norm(x, b.norm2w, b.norm2b, xn, np)
			a.linear(xn, &b.fc1, f1, np)
			o.geluTanh(f1, np*I)
			a.linearAdd(f1, &b.fc2, x, tmp, np)
		}
		if e := o.finish(); e != nil {
			return e
		}
		out = make([]float32, np*H)
		return gpu.Download(x, out)
	})
	if err != nil {
		return nil, fmt.Errorf("cuda: SigLIP tower: %w", err)
	}
	return out, nil
}

// ForwardPatches is last_hidden_state for GridPatches' rows (vision.ResidentEncoder).
func (r *siglipTower) ForwardPatches(patches []float32) ([]float32, error) {
	h, err := r.hidden(patches)
	if err != nil {
		return nil, err
	}
	return r.enc.FinishHidden(h)
}

func (r *siglipTower) Close() { _ = r.g.Close() }
