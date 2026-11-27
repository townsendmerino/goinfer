package cpuengine

import (
	"github.com/townsendmerino/goinfer/internal/runtime/wbuild"
	"time"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/cost"
)

// The CPU's cost counters (internal/cost). They are keyed by the program's operation kinds, so a program walker can record
// the operation and a backend add only the device counts beneath it (launches, syncs, leases). Bytes are what an operation reads from weights, the KV cache and its input registers, and writes to
// its output registers and the cache; its internal scratch is not counted. The CPU launches nothing and never syncs, so those
// are a true 0; allocations are not counted on the CPU (listed unmeasured).

// OpCostNames are the operation kinds' names in a cost report.
var OpCostNames = [...]string{
	opNorm: "norm", opProjQKV: "proj_qkv", opQKNorm: "qk_norm", opRope: "rope", opAttnTemp: "attn_temp",
	opAttention: "attention", opOutGate: "out_gate", opProjO: "proj_o", opFFN: "ffn", opResidual: "residual", opMLA: "mla",
	opKDA: "kda", opShortConv: "short_conv", opMamba2: "mamba2", opDeltaNet: "delta_net", opGatedAttn: "gated_attn",
	opGemma4Attn: "gemma4_attn", opGemma4MoE: "gemma4_moe", opPLE: "ple", opLayerScale: "layer_scale", opInkling2: "inkling2",
}

// The stages around the layers, which the program does not list as operations.
const (
	CostEmbed     = "embed"      // one token's embedding row
	CostFinalNorm = "final_norm" // the head's norm
	CostLMHead    = "lm_head"    // the head's projection (the embedding table when tied)
)

var cpuUnmeasured = []string{cost.FieldAllocBytes, cost.FieldAllocs}
var cpuNoBytes = []string{cost.FieldAllocBytes, cost.FieldAllocs, cost.FieldBytesRead, cost.FieldBytesWritten}

// CostStart is the time an operation starts, when the cache counts (zero otherwise: the off path is this nil check).
func CostStart(cache *Cache) time.Time {
	if cache.Cost == nil {
		return time.Time{}
	}
	return time.Now()
}

func costF32(n int) int64 { return 4 * int64(n) }

// kvRowBytes is one position's stored K (or V) row in layer l's cache: its elements at the cache's precision, and int8's
// per-head scales.
func (c *Cache) kvRowBytes() int64 {
	switch c.Quant {
	case KVI8:
		return int64(c.KVDim) + 4*int64(c.KVDim/max(c.HeadDim, 1))
	case KVQF16:
		return 2 * int64(c.KVDim)
	}
	return 4 * int64(c.KVDim)
}

// CountOp records one per-token operation o of layer l: one call, its bytes (see the file comment) and its time since t0. A
// tree-masked token reads only its ancestors' rows, which is not priced yet (attention's bytes unmeasured, as in causalRows).
func (m *Engine) CountOp(cache *Cache, o *op, lw *darch.LayerWeights, l int, t0 time.Time) {
	rows := int64(-1)
	if cache.TreeMask == nil {
		rows = int64(cache.StoredRows(l, m.W.Arch.NumKVHeads*m.W.Arch.HeadDim))
	}
	m.countOpN(cache, o, lw, l, 1, rows, t0)
}

// countOpN records one pass of operation o over K rows: one call; its weights read once and its activations K times; for
// attention, kvRows stored K and V rows read in all (the decode row's stored rows, or a prefill pass's causal triangle) and K rows
// written. kvRows < 0 leaves attention's bytes unmeasured.
func (m *Engine) countOpN(cache *Cache, o *op, lw *darch.LayerWeights, l, K int, kvRows int64, t0 time.Time) {
	if activeHooks().SeamPlant == PlantCostDoubleCount && o.Kind == opProjO {
		m.countOpN1(cache, o, lw, l, K, kvRows, t0)
	}
	m.countOpN1(cache, o, lw, l, K, kvRows, t0)
}

func (m *Engine) countOpN1(cache *Cache, o *op, lw *darch.LayerWeights, l, K int, kvRows int64, t0 time.Time) {
	arch := m.W.Arch
	nH, hd, kvDim := arch.HeadsAt(l), arch.HeadDim, arch.NumKVHeads*arch.HeadDim
	H := arch.HiddenDim
	d := cost.Counts{Calls: 1, TimeNs: int64(time.Since(t0)), Unmeasured: cpuUnmeasured}
	k := int64(K)
	// rw sets the bytes from the weights read once (wb) and the per-row activation reads and writes.
	rw := func(wb, readPerRow, writtenPerRow int64) {
		d.BytesRead, d.BytesWritten = wb+k*readPerRow, k*writtenPerRow
	}
	switch o.Kind {
	case opNorm:
		w, b := lw.NormWeights(o.Norm)
		rw(costF32(len(w)+len(b)), costF32(H), costF32(H))
	case opProjQKV:
		wb := wbuild.WMBytes(&lw.QProj) + wbuild.WMBytes(&lw.KProj) + wbuild.WMBytes(&lw.VProj)
		if o.Proj.Bias {
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
		rw(wbuild.WMBytes(&lw.GProj), costF32(H+nH*hd), costF32(nH*hd))
	case opProjO:
		wb := wbuild.WMBytes(&lw.OProj)
		if o.Proj.Bias {
			wb += costF32(len(lw.OBias))
		}
		rw(wb, costF32(nH*hd), costF32(H))
	case opFFN:
		if o.FFN.Form != ffnGated && o.FFN.Form != ffnNonGated {
			d.Unmeasured = cpuNoBytes // routed experts: what a token reads depends on its routing
			break
		}
		wb := wbuild.WMBytes(&lw.UpProj) + wbuild.WMBytes(&lw.DownProj)
		if o.FFN.Form == ffnGated {
			wb += wbuild.WMBytes(&lw.GateProj)
		}
		rw(wb, costF32(H), costF32(H))
	case opResidual:
		in := 2
		if o.Res.Two {
			in = 3
		}
		rw(0, costF32(in*H), costF32(H))
	default:
		d.Unmeasured = cpuNoBytes // the family-specific sublayers are not priced yet
	}
	cache.Cost.Add(cache.CostPhase, OpCostNames[o.Kind], d)
}

// countStage records one call of a stage outside the layers (the embedding row, the head) with its bytes and its time since t0.
func countStage(cache *Cache, name string, read, written int64, t0 time.Time) {
	cache.Cost.Add(cache.CostPhase, name, cost.Counts{Calls: 1, BytesRead: read, BytesWritten: written,
		TimeNs: int64(time.Since(t0)), Unmeasured: cpuUnmeasured})
}

// CountEmbed records the embedding of n tokens since t0: n rows of the table (and of the learned positions when pos is set) read,
// n hidden rows written.
func (m *Engine) CountEmbed(cache *Cache, n int, pos bool, t0 time.Time) {
	read := wbuild.WMBytes(&m.W.Embed) / int64(m.W.Embed.Rows())
	if pos {
		read += wbuild.WMBytes(&m.W.PosEmbed) / int64(m.W.PosEmbed.Rows())
	}
	countStage(cache, CostEmbed, int64(n)*read, int64(n)*costF32(m.W.Arch.HiddenDim), t0)
}

// CountFinalNorm records the head's norm over n rows since t0: its weights once, each row read and written.
func (m *Engine) CountFinalNorm(cache *Cache, n int, bias []float32, t0 time.Time) {
	rows := int64(n) * costF32(m.W.Arch.HiddenDim)
	countStage(cache, CostFinalNorm, costF32(len(m.W.FinalNorm)+len(bias))+rows, rows, t0)
}

// CountHead records the head's projection of n rows since t0: its matrix once (the embedding table when tied), each row read,
// n rows of logits written.
func (m *Engine) CountHead(cache *Cache, n int, t0 time.Time) {
	head := &m.W.LMHead
	if m.Program().Head.Tied {
		head = &m.W.Embed
	}
	countStage(cache, CostLMHead, wbuild.WMBytes(head)+int64(n)*costF32(m.W.Arch.HiddenDim), int64(n)*costF32(m.W.Arch.VocabSize), t0)
}

// causalRows is the stored K (or V) rows a prefill pass of K rows from startPos reads in layer l: Σ (startPos+i+1) for i < K. A
// sliding-window layer or a tree-masked pass reads fewer, which is not priced yet (-1: unmeasured).
func causalRows(cache *Cache, l, startPos, K int) int64 {
	if cache.Rings[l] != nil || cache.TreeMask != nil {
		return -1
	}
	k, s := int64(K), int64(startPos)
	return k*s + k*(k+1)/2
}
