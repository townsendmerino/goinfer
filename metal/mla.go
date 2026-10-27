//go:build darwin

package metal

import (
	"fmt"
	"github.com/townsendmerino/goinfer/internal/mem"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// MLA, DeepSeek and Kimi's multi-head latent attention, on the resident decode path (docs/tasks/task-forward-architecture-
// 2026-10.md §4.7, OP2a), in gpu/mla.go's absorbed form: kernels/mla_kernels_msl.go has the four kernels. A layer's KV cache is
// one latent row per position, [cn (kvLoRA) | krj (qkRope)] f32, held in r.kc[l] so the KV slot code gives each slot its own
// rows (r.vc[l] aliases it; MLA stores no separate V). The projections run on the int8 GEMV (W8A8), as WebGPU's do; W_UK and
// W_UV, sliced from kv_b on the host, stay f32 for the per-head matvecs.

// mlaParams is the model's MLA geometry.
type mlaParams struct {
	nH, qLoRA, rank, qkNope, qkRope, vHead int
	qkHead, latDim                         int
	interleave                             bool
	attnScale, ropeScale                   float32
}

// mlaLayer is one MLA layer's device weights.
type mlaLayer struct {
	qAW, qAS, qANorm, qBW, qBS Buffer // the q-LoRA path (qLoRA > 0)
	qW, qS                     Buffer // the direct q projection (qLoRA == 0, V2-Lite)
	kvAW, kvAS, kvANorm        Buffer // the KV down-projection [latDim, H] and the latent's norm
	wuk, wuv                   Buffer // f32 [nH, rank, qkNope] (W_UK transposed) and [nH, vHead, rank]
	oW, oS                     Buffer // the output projection [H, nH*vHead]
}

// buildMLA returns the model's MLA geometry, (nil, nil) for a model without MLA, or the reason this resident cannot run it.
func buildMLA(m *decoder.Model, nH, H int) (*mlaParams, error) {
	qLoRA, kvLoRA, qkNope, qkRope, vHead, interleave, attnScale, ropeScale, ok := m.MLAResidentParams()
	if !ok {
		return nil, nil
	}
	p := &mlaParams{nH: nH, qLoRA: qLoRA, rank: kvLoRA, qkNope: qkNope, qkRope: qkRope, vHead: vHead, interleave: interleave,
		attnScale: float32(attnScale), ropeScale: float32(ropeScale)}
	p.qkHead, p.latDim = qkNope+qkRope, kvLoRA+qkRope
	switch {
	case p.rank < 1 || p.rank > 1024:
		return nil, fmt.Errorf("metal: MLA kv_lora_rank %d is outside mla_attn's 1..1024", p.rank)
	case qkRope%2 != 0 || qkRope < 2:
		return nil, fmt.Errorf("metal: MLA qk_rope_head_dim %d is not a positive even width", qkRope)
	case H%4 != 0 || qLoRA%4 != 0 || (nH*vHead)%4 != 0:
		return nil, fmt.Errorf("metal: MLA widths (hidden %d, q_lora_rank %d, heads·v_head_dim %d) are not multiples of 4, which the int8 GEMV reads", H, qLoRA, nH*vHead)
	}
	if inv := m.RopeInvFreq(); len(inv) != qkRope/2 {
		return nil, fmt.Errorf("metal: MLA's rope table has %d frequencies, want qk_rope_head_dim/2 = %d", len(inv), qkRope/2)
	}
	return p, nil
}

// int8Proj uploads an f32 projection [rows, cols] as per-row int8 (linalg.QuantizeInt8, the codes WebGPU's projF32 makes).
func int8Proj(d *Device, f []float32, rows, cols int, name string, l int) (Buffer, Buffer, error) {
	if len(f) != rows*cols {
		return Buffer{}, Buffer{}, fmt.Errorf("metal: layer %d: MLA %s has %d values, want %d×%d", l, name, len(f), rows, cols)
	}
	wm := linalg.QuantizeInt8(f, rows, cols, false)
	return int8BufOn(mem.LineDense, d, &wm)
}

// buildMLALayer uploads layer l's MLA weights.
func buildMLALayer(d *Device, m *decoder.Model, l, H int, p *mlaParams) (*mlaLayer, error) {
	qA, qANorm, qB, qProj, kvA, kvANorm, kvB, oProj := m.MLALayerWeights(l)
	qAQ, qBQ, qQ, kvAQ, oQ := m.MLALayerInt8(l)
	// proj uploads a projection's int8 form when the load holds one (the codes int8Proj would make from the f32), else quantizes the f32.
	proj := func(q *linalg.WeightMat, f []float32, rows, cols int, name string) (Buffer, Buffer, error) {
		if q.Rows() > 0 {
			return int8BufOn(mem.LineDense, d, q)
		}
		return int8Proj(d, f, rows, cols, name, l)
	}
	L := &mlaLayer{}
	var err error
	if p.qLoRA > 0 {
		if L.qAW, L.qAS, err = proj(qAQ, qA, p.qLoRA, H, "q_a_proj"); err != nil {
			return nil, err
		}
		if L.qBW, L.qBS, err = proj(qBQ, qB, p.nH*p.qkHead, p.qLoRA, "q_b_proj"); err != nil {
			return nil, err
		}
		if len(qANorm) != p.qLoRA {
			return nil, fmt.Errorf("metal: layer %d: MLA q_a_layernorm has %d values, want %d", l, len(qANorm), p.qLoRA)
		}
		L.qANorm = NewBufferFloatsOn(mem.LineDense, d, qANorm)
	} else if L.qW, L.qS, err = proj(qQ, qProj, p.nH*p.qkHead, H, "q_proj"); err != nil {
		return nil, err
	}
	if L.kvAW, L.kvAS, err = proj(kvAQ, kvA, p.latDim, H, "kv_a_proj"); err != nil {
		return nil, err
	}
	if len(kvANorm) != p.rank {
		return nil, fmt.Errorf("metal: layer %d: MLA kv_a_layernorm has %d values, want %d", l, len(kvANorm), p.rank)
	}
	L.kvANorm = NewBufferFloatsOn(mem.LineDense, d, kvANorm)
	if L.oW, L.oS, err = proj(oQ, oProj, H, p.nH*p.vHead, "o_proj"); err != nil {
		return nil, err
	}
	// kv_b is [nH*(qkNope+vHead), rank], per head its k_nope rows then its v rows: W_UK transposed to [nH, rank, qkNope] for
	// the absorb, W_UV as [nH, vHead, rank] for the lift (gpu/residency.go's slicing).
	hRow := p.qkNope + p.vHead
	if len(kvB) != p.nH*hRow*p.rank {
		return nil, fmt.Errorf("metal: layer %d: MLA kv_b_proj has %d values, want %d", l, len(kvB), p.nH*hRow*p.rank)
	}
	wuk := make([]float32, p.nH*p.rank*p.qkNope)
	wuv := make([]float32, p.nH*p.vHead*p.rank)
	for h := range p.nH {
		for dd := range p.qkNope {
			src := kvB[(h*hRow+dd)*p.rank:]
			for c := range p.rank {
				wuk[(h*p.rank+c)*p.qkNope+dd] = src[c]
			}
		}
		for ev := range p.vHead {
			copy(wuv[(h*p.vHead+ev)*p.rank:(h*p.vHead+ev+1)*p.rank], kvB[(h*hRow+p.qkNope+ev)*p.rank:])
		}
	}
	L.wuk, L.wuv = NewBufferFloatsOn(mem.LineDense, d, wuk), NewBufferFloatsOn(mem.LineDense, d, wuv)
	return L, nil
}

// mlaScratch is the MLA mixer's per-token scratch and its uniforms, shared by every layer.
type mlaScratch struct {
	qa, qaQ, qaSc, q, kv, qAbs, wsum, ctx, ctxQ, ctxSc Buffer
	invf                                               Buffer
	uStore, uAbsorb, uLift, uQRope, uAttn              Buffer // packed shapes (kernels/mla_kernels_msl.go)
	uQLoRA, uOK, uZero, uRopeScale, uScale             Buffer
}

func buildMLAScratch(d *Device, m *decoder.Model, p *mlaParams) *mlaScratch {
	b := func(v bool) uint32 {
		if v {
			return 1
		}
		return 0
	}
	s := &mlaScratch{
		qa: d.NewBufferLenOn(mem.LineScratch, max(p.qLoRA, 1)), qaQ: byteBufOn(mem.LineScratch, d, max(p.qLoRA, 4)), qaSc: d.NewBufferLenOn(mem.LineScratch, 1),
		q: d.NewBufferLenOn(mem.LineScratch, p.nH*p.qkHead), kv: d.NewBufferLenOn(mem.LineScratch, p.latDim), qAbs: d.NewBufferLenOn(mem.LineScratch, p.nH*p.latDim),
		wsum: d.NewBufferLenOn(mem.LineScratch, p.nH*p.rank), ctx: d.NewBufferLenOn(mem.LineScratch, p.nH*p.vHead), ctxQ: byteBufOn(mem.LineScratch, d, p.nH*p.vHead), ctxSc: d.NewBufferLenOn(mem.LineScratch, 1),
		invf: NewBufferFloatsOn(mem.LineScratch, d, m.RopeInvFreq()),
	}
	s.uStore = NewBufferUint32sOn(mem.LineScratch, d, []uint32{uint32(p.rank), uint32(p.qkRope), b(p.interleave)})
	s.uAbsorb = NewBufferUint32sOn(mem.LineScratch, d, []uint32{uint32(p.nH), uint32(p.rank), uint32(p.qkNope), uint32(p.qkHead), uint32(p.latDim)})
	s.uLift = NewBufferUint32sOn(mem.LineScratch, d, []uint32{uint32(p.nH), uint32(p.vHead), uint32(p.rank), uint32(p.rank), uint32(p.vHead)})
	s.uQRope = NewBufferUint32sOn(mem.LineScratch, d, []uint32{uint32(p.nH), uint32(p.qkHead), uint32(p.qkNope), uint32(p.qkRope), uint32(p.rank), uint32(p.latDim), b(p.interleave)})
	s.uAttn = NewBufferUint32sOn(mem.LineScratch, d, []uint32{uint32(p.nH), uint32(p.latDim), uint32(p.rank)})
	s.uQLoRA, s.uOK, s.uZero = NewBufferU32On(mem.LineScratch, d, uint32(p.qLoRA)), NewBufferU32On(mem.LineScratch, d, uint32(p.nH*p.vHead)), NewBufferU32On(mem.LineScratch, d, 0)
	s.uRopeScale, s.uScale = NewBufferFloatsOn(mem.LineScratch, d, []float32{p.ropeScale}), NewBufferFloatsOn(mem.LineScratch, d, []float32{p.attnScale})
	return s
}

// mlaKVCopyDeclineOffForTest removes CopyKV's MLA decline: the guard test's planted defect.
var mlaKVCopyDeclineOffForTest bool

// encodeMLAMixer is an MLA layer's attention block: the input norm, q (q-LoRA or direct) and the KV down-projection, the
// latent stored at uPos, q absorbed through W_UK and its rope slice rotated, attention over the slot's latent rows, the lift
// through W_UV, the output projection, and the residual add into x.
func (r *resident) encodeMLAMixer(e *Encoder, l int, L *residLayer, x, uPos, uRopePos, uNKeys Buffer) {
	p, M, s := r.mla, L.mla, r.mlaS
	r.encodeNorm(e, x, L.preNorm, L.preNormBias, r.aq, r.aSc)
	if p.qLoRA > 0 {
		e.Dispatch(r.pGemvW8, p.qLoRA*32, 32, r.aq, r.aSc, M.qAW, M.qAS, s.qa, r.uH)
		e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, s.qa, M.qANorm, s.qaQ, s.qaSc, s.uQLoRA, r.uEps, s.uZero)
		e.Dispatch(r.pGemvW8, p.nH*p.qkHead*32, 32, s.qaQ, s.qaSc, M.qBW, M.qBS, s.q, s.uQLoRA)
	} else {
		e.Dispatch(r.pGemvW8, p.nH*p.qkHead*32, 32, r.aq, r.aSc, M.qW, M.qS, s.q, r.uH)
	}
	e.Dispatch(r.pGemvW8, p.latDim*32, 32, r.aq, r.aSc, M.kvAW, M.kvAS, s.kv, r.uH)
	e.Dispatch(r.pMlaStore, 64, 64, s.kv, M.kvANorm, s.invf, r.kc[l], s.uStore, uPos, uRopePos, r.uEps, s.uRopeScale)
	e.Dispatch(r.pMlaHeadMV, p.nH*p.rank*32, 256, s.q, M.wuk, s.qAbs, s.uAbsorb)
	e.Dispatch(r.pMlaQRope, p.nH*(p.qkRope/2), 64, s.q, s.invf, s.qAbs, s.uQRope, uRopePos, s.uRopeScale)
	e.Dispatch(r.pMlaAttn, p.nH*128, 128, s.qAbs, r.kc[l], s.wsum, s.uAttn, uNKeys, s.uScale)
	e.Dispatch(r.pMlaHeadMV, p.nH*p.vHead*32, 256, s.wsum, M.wuv, s.ctx, s.uLift)
	e.Dispatch(r.pQv, 256, 256, s.ctx, s.ctxQ, s.ctxSc, s.uOK)
	e.Dispatch(r.pGemvW8, r.H*32, 32, s.ctxQ, s.ctxSc, M.oW, M.oS, r.dO, s.uOK)
	e.Dispatch(r.pRes, r.H, 256, x, r.dO)
}
