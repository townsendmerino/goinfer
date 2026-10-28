package decoder

import (
	"time"

	"github.com/townsendmerino/goinfer/internal/cost"
)

// The CPU's cost counters (SB11, internal/cost). They are keyed by the program's operation kinds, the unit OP4's walker will emit
// for every backend: when it lands, the walker records the operation and a backend adds only the device counts beneath it
// (launches, syncs, leases). Bytes are what an operation reads from weights, the KV cache and its input registers, and writes to
// its output registers and the cache; its internal scratch is not counted. The CPU launches nothing and never syncs, so those
// are a true 0; allocations are not counted on the CPU (listed unmeasured).

// opCostNames are the operation kinds' names in a cost report.
var opCostNames = [...]string{
	opNorm: "norm", opProjQKV: "proj_qkv", opQKNorm: "qk_norm", opRope: "rope", opAttnTemp: "attn_temp",
	opAttention: "attention", opOutGate: "out_gate", opProjO: "proj_o", opFFN: "ffn", opResidual: "residual", opMLA: "mla",
	opKDA: "kda", opShortConv: "short_conv", opMamba2: "mamba2", opDeltaNet: "delta_net", opGatedAttn: "gated_attn",
	opGemma4Attn: "gemma4_attn", opGemma4MoE: "gemma4_moe", opPLE: "ple", opLayerScale: "layer_scale",
}

// The stages around the layers, which the program does not list as operations.
const (
	costEmbed     = "embed"      // one token's embedding row
	costFinalNorm = "final_norm" // the head's norm
	costLMHead    = "lm_head"    // the head's projection (the embedding table when tied)
)

var cpuUnmeasured = []string{cost.FieldAllocs}
var cpuNoBytes = []string{cost.FieldAllocs, cost.FieldBytes}

// costStart is the time an operation starts, when the cache counts (zero otherwise: the off path is this nil check).
func costStart(cache *KVCache) time.Time {
	if cache.cost == nil {
		return time.Time{}
	}
	return time.Now()
}

func costF32(n int) int64 { return 4 * int64(n) }

// kvRowBytes is one position's stored K (or V) row in layer l's cache: its elements at the cache's precision, and int8's
// per-head scales.
func (c *KVCache) kvRowBytes() int64 {
	switch c.quant {
	case kvI8:
		return int64(c.kvDim) + 4*int64(c.kvDim/max(c.headDim, 1))
	case kvqF16:
		return 2 * int64(c.kvDim)
	}
	return 4 * int64(c.kvDim)
}

// countOp records one per-token operation o of layer l: one call, its bytes (see the file comment) and its time since t0.
func (m *Model) countOp(cache *KVCache, o *op, lw *LayerWeights, l int, t0 time.Time) {
	m.countOpN(cache, o, lw, l, 1, int64(cache.storedRows(l, m.w.arch.NumKVHeads*m.w.arch.HeadDim)), t0)
}

// countOpN records one pass of operation o over K rows: one call; its weights read once and its activations K times; for
// attention, kvRows stored K and V rows read in all (the decode row's stored rows, or a prefill pass's causal triangle) and K rows
// written. kvRows < 0 leaves attention's bytes unmeasured.
func (m *Model) countOpN(cache *KVCache, o *op, lw *LayerWeights, l, K int, kvRows int64, t0 time.Time) {
	if seamPlantForTest == plantCostDoubleCount && o.kind == opProjO {
		m.countOpN1(cache, o, lw, l, K, kvRows, t0)
	}
	m.countOpN1(cache, o, lw, l, K, kvRows, t0)
}

func (m *Model) countOpN1(cache *KVCache, o *op, lw *LayerWeights, l, K int, kvRows int64, t0 time.Time) {
	arch := m.w.arch
	nH, hd, kvDim := arch.headsAt(l), arch.HeadDim, arch.NumKVHeads*arch.HeadDim
	H := arch.HiddenDim
	d := cost.Counts{Calls: 1, TimeNs: int64(time.Since(t0)), Unmeasured: cpuUnmeasured}
	k := int64(K)
	// rw sets the bytes from the weights read once (wb) and the per-row activation reads and writes.
	rw := func(wb, readPerRow, writtenPerRow int64) {
		d.BytesRead, d.BytesWritten = wb+k*readPerRow, k*writtenPerRow
	}
	switch o.kind {
	case opNorm:
		w, b := lw.normWeights(o.norm)
		rw(costF32(len(w)+len(b)), costF32(H), costF32(H))
	case opProjQKV:
		wb := wmBytes(&lw.QProj) + wmBytes(&lw.KProj) + wmBytes(&lw.VProj)
		if o.proj.Bias {
			wb += costF32(len(lw.QBias) + len(lw.KBias) + len(lw.VBias))
		}
		rw(wb, costF32(H), costF32(nH*hd+2*kvDim))
	case opQKNorm:
		qk := costF32(nH*hd + kvDim)
		rw(costF32(len(lw.QNorm)+len(lw.KNorm)), qk, qk)
	case opRope:
		qk := costF32(nH*hd + kvDim)
		rw(0, qk, qk)
	case opAttnTemp:
		rw(0, costF32(nH*hd), costF32(nH*hd))
	case opAttention:
		if kvRows < 0 {
			d.Unmeasured = cpuNoBytes
			break
		}
		row := cache.kvRowBytes()
		rw(2*kvRows*row, costF32(nH*hd), costF32(nH*hd)+2*row)
	case opOutGate:
		rw(wmBytes(&lw.GProj), costF32(H+nH*hd), costF32(nH*hd))
	case opProjO:
		wb := wmBytes(&lw.OProj)
		if o.proj.Bias {
			wb += costF32(len(lw.OBias))
		}
		rw(wb, costF32(nH*hd), costF32(H))
	case opFFN:
		if o.ffn.Form != ffnGated && o.ffn.Form != ffnNonGated {
			d.Unmeasured = cpuNoBytes // routed experts: what a token reads depends on its routing
			break
		}
		wb := wmBytes(&lw.UpProj) + wmBytes(&lw.DownProj)
		if o.ffn.Form == ffnGated {
			wb += wmBytes(&lw.GateProj)
		}
		rw(wb, costF32(H), costF32(H))
	case opResidual:
		in := 2
		if o.res.Two {
			in = 3
		}
		rw(0, costF32(in*H), costF32(H))
	default:
		d.Unmeasured = cpuNoBytes // the family-specific sublayers are not priced yet
	}
	cache.cost.Add(cache.costPhase, opCostNames[o.kind], d)
}

// countStage records one call of a stage outside the layers (the embedding row, the head) with its bytes and its time since t0.
func countStage(cache *KVCache, name string, read, written int64, t0 time.Time) {
	cache.cost.Add(cache.costPhase, name, cost.Counts{Calls: 1, BytesRead: read, BytesWritten: written,
		TimeNs: int64(time.Since(t0)), Unmeasured: cpuUnmeasured})
}

// causalRows is the stored K (or V) rows a prefill pass of K rows from startPos reads in layer l: Σ (startPos+i+1) for i < K. A
// sliding-window layer or a tree-masked pass reads fewer, which is not priced yet (-1: unmeasured).
func causalRows(cache *KVCache, l, startPos, K int) int64 {
	if cache.rings[l] != nil || cache.treeMask != nil {
		return -1
	}
	k, s := int64(K), int64(startPos)
	return k*s + k*(k+1)/2
}
