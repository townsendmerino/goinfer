package decoder

import (
	"context"
	"time"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
)

type (
	headWorkerScratch  = cpuengine.HeadWorkerScratch
	prefillPassScratch = cpuengine.PrefillPassScratch
	moeBatchScratch    = cpuengine.MoEBatchScratch
	batchRows          = cpuengine.BatchRows
	w4a8Act            = cpuengine.W4A8Act
)

const (
	prefillAttnScratchBudget = cpuengine.PrefillAttnScratchBudget
	fastAttnMinPrompt        = cpuengine.FastAttnMinPrompt
	attnGroupedNEONSize      = cpuengine.AttnGroupedNEONSize
	attnGroupedMinKeys       = cpuengine.AttnGroupedMinKeys
	maxAttnWorkers           = cpuengine.MaxAttnWorkers
	attnScoreTileBytes       = cpuengine.AttnScoreTileBytes
	fusedKeyBlock            = cpuengine.FusedKeyBlock
	costEmbed                = cpuengine.CostEmbed
	costLMHead               = cpuengine.CostLMHead
	costFinalNorm            = cpuengine.CostFinalNorm
	w4a8BatchDefault         = cpuengine.W4A8BatchDefault
	routerCaptureMax         = cpuengine.RouterCaptureMax
)

var opCostNames = cpuengine.OpCostNames

func newDecodeScratch(a *darch.Architecture, k cpuengine.KnobSource) *decodeScratch {
	return cpuengine.NewDecodeScratch(a, k)
}

func newHeadWorkerPool(n, K, nKeys, hd int, wantFused bool) []headWorkerScratch {
	return cpuengine.NewHeadWorkerPool(n, K, nKeys, hd, wantFused)
}

func serialMMWorkspace() *linalg.Workspace { return cpuengine.SerialMMWorkspace() }

func attnRowTile(K, nKeys int) int { return cpuengine.AttnRowTile(K, nKeys) }
func attnRowTileK(k cpuengine.KnobSource, K, nKeys int) int {
	return cpuengine.AttnRowTileK(k, K, nKeys)
}
func prefillAttnWorkers(K, nKeys, hd, nH int) int {
	return cpuengine.PrefillAttnWorkers(K, nKeys, hd, nH)
}
func prefillAttnWorkersK(k cpuengine.KnobSource, K, nKeys, hd, nH int) int {
	return cpuengine.PrefillAttnWorkersK(k, K, nKeys, hd, nH)
}
func scalarHeadWorkers(nH, span, hd int) int { return cpuengine.ScalarHeadWorkers(nH, span, hd) }
func attendTileFor(ws *headWorkerScratch, K, nKeys, hd int) int {
	return cpuengine.AttendTileFor(ws, K, nKeys, hd)
}
func cpuFastAttention() bool { return cpuengine.CPUFastAttention() }

func (m *Model) cpuFastAttention() bool { return m.knobs.cpuFastAttention() }

func (m *Model) walkBatch(reqCtx context.Context, prog *layerProgram, h []float32, cache *KVCache, fastAttn bool, floorK int, ps *prefillPassScratch, rows *batchRows) ([]float32, error) {
	e := m.cpuEngine()
	return e.WalkBatch(reqCtx, prog, h, unwrapCache(cache), fastAttn, floorK, ps, rows)
}

func (m *Model) countOp(cache *KVCache, o *op, lw *darch.LayerWeights, l int, t0 time.Time) {
	e := m.cpuEngine()
	e.CountOp(unwrapCache(cache), o, lw, l, t0)
}

func (m *Model) mlaAttentionAbsorb(op *mlaOpParams, n []float32, lw *darch.LayerWeights, arch *darch.Architecture, cache *KVCache, layer, pos int) []float32 {
	e := m.cpuEngine()
	return e.MLAAttentionAbsorb(op, n, lw, arch, unwrapCache(cache), layer, pos)
}

func (m *Model) gptOssMoE(h []float32, lw *darch.LayerWeights, arch *darch.Architecture) ([]float32, error) {
	e := m.cpuEngine()
	return e.GptOssMoE(h, lw, arch)
}

func (m *Model) gptOssMoEWith(h []float32, lw *darch.LayerWeights, p *ffnParams, hidden int, scr *decodeScratch) ([]float32, error) {
	e := m.cpuEngine()
	return e.GptOssMoEWith(h, lw, p, hidden, scr)
}

func (m *Model) gptOssMoEBatch(rows []float32, n int, lw *darch.LayerWeights, p *ffnParams, hidden int, dst []float32, s *moeBatchScratch) (bool, error) {
	e := m.cpuEngine()
	return e.GptOssMoEBatch(rows, n, lw, p, hidden, dst, s)
}

func (m *Model) llama4MoE(h, out []float32, lw *darch.LayerWeights, arch *darch.Architecture) {
	e := m.cpuEngine()
	e.Llama4MoE(h, out, lw, arch)
}

func (m *Model) nemotronMoE(n []float32, lw *darch.LayerWeights, arch *darch.Architecture, hidden int) []float32 {
	e := m.cpuEngine()
	return e.NemotronMoE(n, lw, arch, hidden)
}

func attendBatchedHeads(q, ctx, keys, vals []float32, base int, cache *KVCache, layer, startPos, K int, global bool, arch *darch.Architecture, useAcc64 bool, pool []headWorkerScratch) {
	syncCPUHooks()
	cpuengine.AttendBatchedHeads(q, ctx, keys, vals, base, unwrapCache(cache), layer, startPos, K, global, arch, useAcc64, pool)
}

func attendBatchedHeadsP(q, ctx, keys, vals []float32, base int, cache *KVCache, layer, startPos, K int, ap attentionParams, arch *darch.Architecture, useAcc64 bool, pool []headWorkerScratch) {
	syncCPUHooks()
	cpuengine.AttendBatchedHeadsP(q, ctx, keys, vals, base, unwrapCache(cache), layer, startPos, K, ap, arch, useAcc64, pool)
}

func attendQuery(q, ctx, scores []float32, cache *KVCache, layer, pos int, global bool, arch *darch.Architecture) {
	syncCPUHooks()
	cpuengine.AttendQuery(q, ctx, scores, unwrapCache(cache), layer, pos, global, arch)
}

func attendQueryP(p attentionParams, q, ctx, scores []float32, cache *KVCache, layer, pos int, arch *darch.Architecture) {
	syncCPUHooks()
	cpuengine.AttendQueryP(p, q, ctx, scores, unwrapCache(cache), layer, pos, arch)
}

func attendSinksToken(p attentionParams, q, k, v, ctx []float32, cache *KVCache, arch *darch.Architecture, lw *darch.LayerWeights, l, pos int) {
	syncCPUHooks()
	cpuengine.AttendSinksToken(p, q, k, v, ctx, unwrapCache(cache), arch, lw, l, pos)
}

func softcapScore(s, c float64) float64 {
	syncCPUHooks()
	return cpuengine.SoftcapScore(s, c)
}

func matmul(be Backend, w *linalg.WeightMat, a, dst []float32, M int) {
	syncCPUHooks()
	cpuengine.Matmul(be, w, a, dst, M)
}

func matmulInto(ws *linalg.Workspace, be Backend, w *linalg.WeightMat, a, dst []float32, M int) {
	syncCPUHooks()
	cpuengine.MatmulInto(ws, be, w, a, dst, M)
}

func matmulW4A8Batch(be Backend, ws *linalg.Workspace, a []float32, M, K, group int, ops []linalg.BatchOp, actGroup int) {
	syncCPUHooks()
	cpuengine.MatmulW4A8Batch(be, ws, a, M, K, group, ops, actGroup)
}

func w4a8FusedOps(ops []linalg.BatchOp, ws []*linalg.WeightMat, dsts [][]float32) (int, int, bool) {
	return cpuengine.W4A8FusedOps(ops, ws, dsts)
}

func isW4A8(w *linalg.WeightMat) bool { return cpuengine.IsW4A8(w) }
func isW8A8(w *linalg.WeightMat) bool { return cpuengine.IsW8A8(w) }
func wmW4A8Op(w *linalg.WeightMat, dst []float32) (linalg.BatchOp, int) {
	return cpuengine.WMW4A8Op(w, dst)
}

func gatedMLPFusedGateUp(h []float32, lw *darch.LayerWeights, arch *darch.Architecture, scr *decodeScratch) bool {
	syncCPUHooks()
	return cpuengine.GatedMLPFusedGateUp(h, lw, arch, scr)
}

func gemma4MoEFFN(be Backend, arch *darch.Architecture, h []float32, w *archparts.Gemma4MoEWeights, pager *expertPager) []float32 {
	syncCPUHooks()
	var ep cpuengine.ExpertPager
	if pager != nil {
		ep = pager
	}
	return cpuengine.Gemma4MoEFFN(be, arch, h, w, ep)
}

func routeExperts(logits, bias []float32, k int, sigmoid, norm bool, scale float64, nGroup, topkGroup int) ([]int, []float32) {
	return cpuengine.RouteExperts(logits, bias, k, sigmoid, norm, scale, nGroup, topkGroup)
}

func swiglu(gate, up []float32)               { cpuengine.Swiglu(gate, up) }
func softmaxF32(x []float32) []float32        { return cpuengine.SoftmaxF32(x) }
func addScaled(dst, src []float32, s float32) { cpuengine.AddScaled(dst, src, s) }
func addResidual(h, sub []float32)            { cpuengine.AddResidual(h, sub) }
func addResidual2(h, sub, sub2 []float32)     { cpuengine.AddResidual2(h, sub, sub2) }
func addBias(x, b []float32)                  { cpuengine.AddBias(x, b) }
func addDeepstack(h []float32, ds *deepstackRows, l, startPos, K, hidden int) {
	cpuengine.AddDeepstack(h, ds, l, startPos, K, hidden)
}
func applyNorm(p normParams, dst, src, w, b []float32, dim int) {
	cpuengine.ApplyNorm(p, dst, src, w, b, dim)
}
func normalize(arch *darch.Architecture, x, w, b []float32, dim int) {
	cpuengine.Normalize(arch, x, w, b, dim)
}
func mlaRope(vec []float32, heads, ropeDim, pos int, invFreq []float64, scale float64, interleave bool) {
	cpuengine.MLARope(vec, heads, ropeDim, pos, invFreq, scale, interleave)
}
func applyLoRA(d *loraDelta, x, y []float32, scr *decodeScratch) {
	cpuengine.ApplyLoRA(d, x, y, scr)
}
func matvecWM(be Backend, w *linalg.WeightMat, x []float32) []float32 {
	syncCPUHooks()
	return cpuengine.MatvecWM(be, w, x)
}
func matvecWMPre(be Backend, w *linalg.WeightMat, p *w4a8Act, x []float32) []float32 {
	syncCPUHooks()
	return cpuengine.MatvecWMPre(be, w, p, x)
}
func newDeltaState(p archparts.Qwen35Params) *deltaState { return cpuengine.NewDeltaState(p) }
func deltaNetCore(core, mixed, bt, at, z []float32, w *archparts.DeltaNetWeights, p archparts.Qwen35Params, eps float64, st *deltaState) {
	syncCPUHooks()
	cpuengine.DeltaNetCore(core, mixed, bt, at, z, w, p, eps, st)
}
func deltaNetCoreN(core, mixed, bt, at, z []float32, K int, w *archparts.DeltaNetWeights, p archparts.Qwen35Params, eps float64, st *deltaState) {
	syncCPUHooks()
	cpuengine.DeltaNetCoreN(core, mixed, bt, at, z, K, w, p, eps, st)
}
func deltaNetRecurrence(core, conv, at, bt []float32, w *archparts.DeltaNetWeights, p archparts.Qwen35Params, st *deltaState) {
	cpuengine.DeltaNetRecurrence(core, conv, at, bt, w, p, st)
}
func gatedDeltaNet(be Backend, h [][]float32, w *archparts.DeltaNetWeights, p archparts.Qwen35Params, hidden int, eps float64) [][]float32 {
	syncCPUHooks()
	return cpuengine.GatedDeltaNet(be, h, w, p, hidden, eps)
}
func mamba2Seq(h [][]float32, w *archparts.Mamba2Weights, p archparts.Mamba2Params, eps float64) [][]float32 {
	syncCPUHooks()
	return cpuengine.Mamba2Seq(h, w, p, eps)
}
func kdaRecurrence(core, q, k, v, gDecayRaw, betaLogits []float32, w *archparts.KDAWeights, p archparts.KDAParams, st *kdaState) {
	cpuengine.KDARecurrence(core, q, k, v, gDecayRaw, betaLogits, w, p, st)
}
func kdaLowerBoundGate(rawGate, dtBias []float32, aLog, lowerBound float32) []float32 {
	return cpuengine.KDALowerBoundGate(rawGate, dtBias, aLog, lowerBound)
}
func kdaRecurrentStep(q, k, v, decay []float32, beta float32, S []float32) []float32 {
	return cpuengine.KDARecurrentStep(q, k, v, decay, beta, S)
}
func dequantHeads(q []int8, scales []float32, nKV, headDim int, dst []float32) {
	cpuengine.DequantHeads(q, scales, nKV, headDim, dst)
}
func dequantKVRows(dstK, dstV []float32, kq, vq []int8, ksc, vsc []float32, rows, cols int) {
	cpuengine.DequantKVRows(dstK, dstV, kq, vq, ksc, vsc, rows, cols)
}
func topK(scores []float32, k int) ([]int, []float32)   { return cpuengine.TopK(scores, k) }
func softcapParallel(logits []float32, softcap float32) { cpuengine.SoftcapParallel(logits, softcap) }
func costStart(cache *KVCache) time.Time                { return cpuengine.CostStart(unwrapCache(cache)) }
func routerCaptureReset()                               { cpuengine.RouterCaptureReset() }
func routerCaptureDo(fn func())                         { cpuengine.RouterCaptureDo(fn) }
func cpuBatchCacheEligible(c *KVCache) error            { return cpuengine.CPUBatchCacheEligible(unwrapCache(c)) }

func gatedDeltaNetStep(be Backend, h []float32, w *archparts.DeltaNetWeights, p archparts.Qwen35Params, hidden int, eps float64, st *deltaState) []float32 {
	syncCPUHooks()
	return cpuengine.GatedDeltaNetStep(be, h, w, p, hidden, eps, st)
}
