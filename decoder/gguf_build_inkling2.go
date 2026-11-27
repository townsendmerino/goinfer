package decoder

import (
	"errors"
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/inkling2"
)

// loadInkling2 loads an Inkling 2 GGUF: each physical layer once (inkling2.LayerTensors names its tensors), then every
// visit (Architecture.NumLayers) takes a copy of its physical layer's LayerWeights, which shares the tensors. It has no
// streaming (.giw) path.
func (b *ggufBuild) loadInkling2(p *archparts.Inkling2Params) (*darch.Weights, error) {
	if b.sink != nil {
		return nil, fmt.Errorf("decoder(gguf-inkling2): no .giw build for this family yet; load the GGUF directly (-direct-load)")
	}
	head := &archparts.Inkling2HeadWeights{}
	var err error
	if head.EmbedNorm, err = b.vnorm("token_embd_norm.weight", b.hidden); err != nil {
		return nil, err
	}
	if head.OutputResQuery, err = b.vec("output_res_query.weight", b.hidden); err != nil {
		return nil, err
	}
	b.arch.Inkling2Head = head
	phys := make([]darch.LayerWeights, p.PhysLayers)
	if err := parallelLayers(p.PhysLayers, b.abort, func(i int) error { return b.inkling2Layer(&phys[i], i) }); err != nil {
		return nil, err
	}
	for v, vis := range p.Visits() {
		b.w.Layers[v] = phys[vis.Phys]
	}
	return b.w, nil
}

// inkling2Layer loads physical layer i into l: its attention and convolution tensors, the f32 router, and the experts.
func (b *ggufBuild) inkling2Layer(l *darch.LayerWeights, i int) error {
	mats, vecs := inkling2.LayerTensors(b.arch, l, i)
	var err error
	for _, t := range mats {
		if *t.Dst, err = b.mat(t.Name, t.Out, t.In); err != nil {
			return err
		}
	}
	for _, t := range vecs {
		_, data, err := b.g.Tensor(t.Name)
		if err == nil && len(data) != t.N {
			err = fmt.Errorf("decoder(gguf-inkling2): %q has %d elements, want %d", t.Name, len(data), t.N)
		}
		if err != nil {
			return err
		}
		*t.Dst = data
	}
	moe, hidden, pf := b.arch.MoE, b.hidden, fmt.Sprintf("blk.%d.", i)
	// The router stays f32: top-k selection is discrete, so quantizing it flips which experts win.
	if l.Router, err = b.streamMat(pf+"ffn_gate_inp.weight", moe.NumExperts, hidden, quantNone, func(r int) int { return r * hidden }); err != nil {
		return err
	}
	gate, up, ge := b.splitGateUpExperts(pf+"ffn_gate_up_exps.weight", moe.IntermediateDim, hidden, moe.NumExperts)
	down, de := b.stackedExperts(pf+"ffn_down_exps.weight", hidden, moe.IntermediateDim, moe.NumExperts)
	if ge != nil || de != nil {
		return fmt.Errorf("decoder(gguf-inkling2): experts of layer %d: %w", i, errors.Join(ge, de))
	}
	l.Experts = make([]archparts.ExpertWeights, moe.NumExperts)
	for e := range l.Experts {
		l.Experts[e] = archparts.ExpertWeights{Gate: gate[e], Up: up[e], Down: down[e]}
	}
	return nil
}

// splitGateUpExperts loads a fused [in, 2*inter, nExpert] expert tensor whose first inter rows of each expert are the gate
// and the next inter the up projection, as one Gate and one Up WeightMat per expert.
func (b *ggufBuild) splitGateUpExperts(name string, inter, in, nExpert int) (gate, up []linalg.WeightMat, err error) {
	dims, into, err := ggufScaledRowDequantizer(b.g, name)
	if err != nil {
		return nil, nil, err
	}
	if len(dims) != 3 || dims[0] != in || dims[1] != 2*inter || dims[2] != nExpert {
		return nil, nil, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d, experts=%d]", name, dims, in, 2*inter, nExpert)
	}
	gate, up = make([]linalg.WeightMat, nExpert), make([]linalg.WeightMat, nExpert)
	mode := matmulQuant(b.quant, name)
	one := func(e int) error {
		for half, dst := range []*linalg.WeightMat{&gate[e], &up[e]} {
			base := (e*2 + half) * inter
			m, err := streamQuantizedLayer(inter, in, mode, b.needCanonical, b.skipRow4, func(r int, row []float32) error {
				return ggufRowFiltered(name, into, (base+r)*in, row)
			})
			if err != nil {
				return err
			}
			*dst = m
		}
		return nil
	}
	return gate, up, forExperts(nExpert, b.sink, one)
}

// inkling2Host is the engine side of an Inkling 2 visit (inkling2.Host): the backend's matmul and the cache's KV rows.
type inkling2Host struct {
	be    Backend
	cache *KVCache
}

func (h inkling2Host) MatVec(w *linalg.WeightMat, x, dst []float32) { matmul(h.be, w, x, dst, 1) }

func (h inkling2Host) AppendKV(l int, k, v []float32) (keys, vals []float32, stride, n int) {
	h.cache.Append(l, k, v)
	keys, vals, stride = h.cache.layerRows(l, len(k))
	return keys, vals, stride, h.cache.storedRows(l, len(k))
}

// paddedVocabRows reports whether a table of stored rows is an Inkling 2 embedding or head padded past the vocabulary of out
// rows: the loader then reads its first out rows.
func paddedVocabRows(arch *darch.Architecture, stored, out int) bool {
	return arch.Inkling2 != nil && stored == arch.Inkling2.PaddedVocab && out < stored
}
