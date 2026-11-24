//go:build gpu

package kernels

import "github.com/oliverbestmann/webgpu/wgpu"

// GPT-J pairwise RoPE (decoder.FeatPairwiseRoPE: Cohere, Cohere2, GLM-OCR's text decoder) rotates the adjacent pair
// (2d, 2d+1) of each head by pos·invFreq[d], where the NeoX kernels rotate (d, half+d). The twins below are the decode
// rope kernels with only the pair's indices changed, so the angle, the scale and the arithmetic are the NeoX kernels'
// and the partial-rotary tail [2·half, headDim) passes through as before. Each is an exact-count edit of its NeoX
// source (editSource), refused when that source changes shape. The bindings are the NeoX kernels', so the runner swaps
// pipelines and nothing else (RunModel.Pairwise).

// pairwiseIndexEdits turns a kernel that names the pair as [off + d] and [off + p.half + d] into the pairwise one.
func pairwiseIndexEdits(n int) []srcEdit {
	return []srcEdit{
		{"off + d]", "off + 2u * d]", n},
		{"off + p.half + d]", "off + 2u * d + 1u]", n},
	}
}

// pairwiseElemRot is the f16 and int8 stores' per-element rotation in pairwise form: element dd < 2·half is the even or
// odd member of pair dd/2.
const pairwiseElemRot = `    if (dd < 2u * p.half) {
        let d = dd / 2u;
        let theta = f32(p.pos) * invFreq[d];
        let cs = ropeCosSin(theta); let c = cs.x * p.scale;
        let s = cs.y * p.scale;
        if ((dd & 1u) == 0u) { return src[off + dd] * c - src[off + dd + 1u] * s; }
        return src[off + dd] * c + src[off + dd - 1u] * s;
    }
`

const neoxF16ElemRot = `    if (dd < p.half) {
        let theta = f32(p.pos) * invFreq[dd];
        let cs = ropeCosSin(theta); let c = cs.x * p.scale;
        let s = cs.y * p.scale;
        return src[off + dd] * c - src[off + p.half + dd] * s;
    }
    if (dd < 2u * p.half) {
        let d = dd - p.half;
        let theta = f32(p.pos) * invFreq[d];
        let cs = ropeCosSin(theta); let c = cs.x * p.scale;
        let s = cs.y * p.scale;
        return src[off + p.half + d] * c + src[off + d] * s;
    }
`

const neoxI8ElemRot = `    if (dd < p.half) {
        let theta = f32(p.pos) * invFreq[dd];
        let cs = ropeCosSin(theta);
        return src[off + dd] * cs.x * p.scale - src[off + p.half + dd] * cs.y * p.scale;
    }
    if (dd < 2u * p.half) {
        let d = dd - p.half;
        let theta = f32(p.pos) * invFreq[d];
        let cs = ropeCosSin(theta);
        return src[off + p.half + d] * cs.x * p.scale + src[off + d] * cs.y * p.scale;
    }
`

// pairwiseI8ElemRot is pairwiseElemRot in the int8 store's arithmetic form (the scale applied per term).
const pairwiseI8ElemRot = `    if (dd < 2u * p.half) {
        let d = dd / 2u;
        let theta = f32(p.pos) * invFreq[d];
        let cs = ropeCosSin(theta);
        if ((dd & 1u) == 0u) { return src[off + dd] * cs.x * p.scale - src[off + dd + 1u] * cs.y * p.scale; }
        return src[off + dd] * cs.x * p.scale + src[off + dd - 1u] * cs.y * p.scale;
    }
`

// PairwiseNeoXForTest is the plant of the pairwise gates: a runner built while it is set binds the NeoX rope kernels on a
// pairwise model. Read at newDecodeRunner.
var PairwiseNeoXForTest bool

// PairwiseNeoXStoresForTest is the narrower plant: on a pairwise model the K-cache rope stores (f32, f16, int8) bind NeoX
// while the query rotation and qkvFinalize stay pairwise, so a gate on an f16 or int8 cache sees one miswired store. Read
// at newDecodeRunner.
var PairwiseNeoXStoresForTest bool

// pairwisePipe is one compiled pairwise twin and its bind-group layout.
type pairwisePipe struct {
	pl *wgpu.ComputePipeline
	ly *wgpu.BindGroupLayout
}

// pairwisePipes are the pairwise twins of the decode rope kernels.
type pairwisePipes struct {
	rope, ropeStore, qkvFin, ropeStoreF16, ropeStoreI8 pairwisePipe
}

// ensurePairwise compiles the pairwise twins once per Context.
func (c *Context) ensurePairwise() error {
	if c.pairwise != nil {
		return nil
	}
	var p pairwisePipes
	for _, k := range []struct {
		dst        *pairwisePipe
		label, src string
		edits      []srcEdit
	}{
		{&p.rope, "rope-pairwise", ropeShaderWGSL, pairwiseIndexEdits(2)},
		{&p.ropeStore, "rope-store-pairwise", ropeStoreShaderWGSL, pairwiseIndexEdits(2)},
		{&p.qkvFin, "qkvFinalize-pairwise", qkvFinalizeShaderWGSL, pairwiseIndexEdits(4)},
		{&p.ropeStoreF16, "rope-store-f16-pairwise", ropeStoreF16ShaderWGSL, []srcEdit{{neoxF16ElemRot, pairwiseElemRot, 1}}},
		{&p.ropeStoreI8, "rope-store-i8-pairwise", ropeStoreI8ShaderWGSL, []srcEdit{{neoxI8ElemRot, pairwiseI8ElemRot, 1}}},
	} {
		code, err := editSource(k.label, k.src, k.edits)
		if err != nil {
			return err
		}
		_, pl, ly, err := c.mkPipeline(k.label, code)
		if err != nil {
			return err
		}
		*k.dst = pairwisePipe{pl, ly}
	}
	c.pairwise = &p
	return nil
}
