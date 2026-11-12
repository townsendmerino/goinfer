//go:build cuda

package cuda

import (
	"context"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/cost"
)

// Compile-time pins for the CUDA resident runner's four subsystem interfaces. Kept on wrapper types rather than on
// cudaResident directly so cudaResident's own method set reflects only the capabilities CUDA actually implements.
var (
	_ decoder.ResidentSubsystemProvider = (*cudaResident)(nil)
	_ decoder.ResidentDecodeSubsystem   = cudaDecodeSub{}
	_ decoder.ResidentPrefillSubsystem  = cudaPrefillSub{}
	_ decoder.ResidentKVSubsystem       = cudaKVSub{}
	_ decoder.ResidentRuntimeSubsystem  = cudaRuntimeSub{}
)

// ResidentSubsystems returns the CUDA resident runner's four subsystem views with zero heap allocations.
func (r *cudaResident) ResidentSubsystems() decoder.ResidentSubsystems {
	return decoder.ResidentSubsystems{
		Decode:  cudaDecodeSub{r: r},
		Prefill: cudaPrefillSub{r: r},
		KV:      cudaKVSub{r: r},
		Runtime: cudaRuntimeSub{r: r},
	}
}

type cudaDecodeSub struct {
	decoder.NoopResidentDecode
	r *cudaResident
}

func (s cudaDecodeSub) GreedyAvailable() bool { return true }
func (s cudaDecodeSub) ForwardArgmax(embedding []float32, pos int) (int, error) {
	return s.r.ForwardArgmax(embedding, pos)
}
func (s cudaDecodeSub) TopKAvailable() bool { return s.r.TopKAvailable() }
func (s cudaDecodeSub) ForwardTopK(embedding []float32, pos, k int, temperature float64, wantZ bool) (decoder.TopKRow, error) {
	return s.r.ForwardTopK(embedding, pos, k, temperature, wantZ)
}
func (s cudaDecodeSub) SampleAvailable() bool { return s.r.SampleAvailable() }
func (s cudaDecodeSub) ForwardSample(embedding []float32, pos int, temperature float64, seed, draw uint64) (int, error) {
	return s.r.ForwardSample(embedding, pos, temperature, seed, draw)
}
func (s cudaDecodeSub) MRoPEDecodeAvailable() bool { return true }
func (s cudaDecodeSub) ForwardMRoPE(embedding []float32, pos, ropePos int) ([]float32, error) {
	return s.r.ForwardMRoPE(embedding, pos, ropePos)
}
func (s cudaDecodeSub) NoLogitsAvailable() bool { return true }
func (s cudaDecodeSub) ForwardNoLogits(e []float32, pos int) error {
	return s.r.ForwardNoLogits(e, pos)
}
func (s cudaDecodeSub) BatchStepperAvailable() bool  { return true }
func (s cudaDecodeSub) BatchStepRange() (lo, hi int) { return s.r.BatchStepRange() }
func (s cudaDecodeSub) StepBatch(seqs []decoder.ResidentBatchSeq) ([]decoder.ResidentBatchOut, error) {
	return s.r.StepBatch(seqs)
}
func (s cudaDecodeSub) RaggedStepperAvailable() bool { return true }
func (s cudaDecodeSub) StepRagged(seqs []decoder.ResidentRaggedSeq) ([]decoder.ResidentRaggedOut, error) {
	return s.r.StepRagged(seqs)
}

type cudaPrefillSub struct {
	decoder.NoopResidentPrefill
	r *cudaResident
}

func (s cudaPrefillSub) PrefillAvailable() bool { return true }
func (s cudaPrefillSub) PrefillLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return s.r.PrefillLast(ctx, embeddings, startPos)
}
func (s cudaPrefillSub) PrefillRowsAvailable() bool { return true }
func (s cudaPrefillSub) PrefillLastRows(ctx context.Context, n, startPos int, rows func(lo, hi int) [][]float32) ([]float32, error) {
	return s.r.PrefillLastRows(ctx, n, startPos, rows)
}
func (s cudaPrefillSub) PrefillMarkedAvailable() bool { return true }
func (s cudaPrefillSub) PrefillMarkAlign() int        { return s.r.PrefillMarkAlign() }
func (s cudaPrefillSub) PrefillLastMarked(ctx context.Context, embeddings [][]float32, startPos, mark int, at func(pos int)) ([]float32, error) {
	return s.r.PrefillLastMarked(ctx, embeddings, startPos, mark, at)
}
func (s cudaPrefillSub) PrefillPathInfo() (bool, string, bool) {
	b, reason := s.r.PrefillPath()
	return b, reason, true
}
func (s cudaPrefillSub) PrefillChunksAvailable() bool       { return true }
func (s cudaPrefillSub) PrefillChunkRows() int              { return s.r.PrefillChunkRows() }
func (s cudaPrefillSub) PrefillScratchBytes(rows int) int64 { return s.r.PrefillScratchBytes(rows) }
func (s cudaPrefillSub) PrefillSpanScratchBytes(rows, k int) int64 {
	return s.r.PrefillSpanScratchBytes(rows, k)
}
func (s cudaPrefillSub) HiddenLastAvailable() bool { return true }
func (s cudaPrefillSub) HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return s.r.HiddenLast(ctx, embeddings, startPos)
}
func (s cudaPrefillSub) ResidualAllAvailable() bool { return true }
func (s cudaPrefillSub) ResidualAll(ctx context.Context, embeddings [][]float32, startPos int) ([][]float32, error) {
	return s.r.ResidualAll(ctx, embeddings, startPos)
}
func (s cudaPrefillSub) ImagePrefillAvailable() bool { return true }
func (s cudaPrefillSub) PrefillImageLast(ctx context.Context, embeddings [][]float32, startPos, imgStart, imgEnd int) ([]float32, error) {
	return s.r.PrefillImageLast(ctx, embeddings, startPos, imgStart, imgEnd)
}
func (s cudaPrefillSub) ImageBlocksPrefillAvailable() bool { return true }
func (s cudaPrefillSub) PrefillImageBlocksLast(ctx context.Context, embeddings [][]float32, startPos int, blocks [][2]int) ([]float32, error) {
	return s.r.PrefillImageBlocksLast(ctx, embeddings, startPos, blocks)
}
func (s cudaPrefillSub) SpanPrefillAvailable() bool { return true }
func (s cudaPrefillSub) PrefillSpansLast(ctx context.Context, embeddings [][]float32, startPos int, spans [][2]int) ([]float32, error) {
	return s.r.PrefillSpansLast(ctx, embeddings, startPos, spans)
}
func (s cudaPrefillSub) HybridMRoPEPrefill() bool    { return s.r.HybridMRoPEPrefill() }
func (s cudaPrefillSub) MRoPEPrefillAvailable() bool { return true }
func (s cudaPrefillSub) PrefillMRoPELast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int) ([]float32, error) {
	return s.r.PrefillMRoPELast(ctx, embeddings, startPos, mropePos)
}
func (s cudaPrefillSub) MRoPEDeepstackPrefillAvailable() bool { return true }
func (s cudaPrefillSub) PrefillMRoPEDeepstackLast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int, deep [][]float32, imgStart, imgLen int) ([]float32, error) {
	return s.r.PrefillMRoPEDeepstackLast(ctx, embeddings, startPos, mropePos, deep, imgStart, imgLen)
}

type cudaKVSub struct {
	decoder.NoopResidentKV
	r *cudaResident
}

func (s cudaKVSub) KVSlotterAvailable() bool            { return true }
func (s cudaKVSub) KVSlots() int                        { return s.r.KVSlots() }
func (s cudaKVSub) UseKVSlot(i int) error               { return s.r.UseKVSlot(i) }
func (s cudaKVSub) SetKVSlotBusy(slot int, busy bool)   { s.r.SetKVSlotBusy(slot, busy) }
func (s cudaKVSub) KVPool() decoder.KVPoolDetail        { return s.r.KVPool() }
func (s cudaKVSub) ReusableKV(prefix int) int           { return s.r.ReusableKV(prefix) }
func (s cudaKVSub) ReusableKVSlotAvailable() bool       { return true }
func (s cudaKVSub) ReusableKVSlot(slot, prefix int) int { return s.r.ReusableKVSlot(slot, prefix) }
func (s cudaKVSub) KVUploadPassAvailable() bool         { return true }
func (s cudaKVSub) BeginKVUpload() error                { return s.r.BeginKVUpload() }
func (s cudaKVSub) EndKVUpload(failed bool) error       { return s.r.EndKVUpload(failed) }
func (s cudaKVSub) KVCopierAvailable() bool             { return true }
func (s cudaKVSub) CopyKV(src, dst, start, end int) error {
	return s.r.CopyKV(src, dst, start, end)
}
func (s cudaKVSub) KVSnapshotterAvailable() bool { return true }
func (s cudaKVSub) SnapshotKV(slot, end int) (decoder.ResidentKVSnapshot, error) {
	return s.r.SnapshotKV(slot, end)
}
func (s cudaKVSub) RestoreKV(snap decoder.ResidentKVSnapshot, slot, end int) error {
	return s.r.RestoreKV(snap, slot, end)
}
func (s cudaKVSub) KVSnapshotBytes(end int) int64          { return s.r.KVSnapshotBytes(end) }
func (s cudaKVSub) KVSnapshotRollback() int                { return s.r.KVSnapshotRollback() }
func (s cudaKVSub) KVSnapshotRecurrent() bool              { return s.r.KVSnapshotRecurrent() }
func (s cudaKVSub) KVDemoterAvailable() bool               { return true }
func (s cudaKVSub) SetKVHostBudget(b decoder.KVHostBudget) { s.r.SetKVHostBudget(b) }
func (s cudaKVSub) PromoteKV(slot, rows int) (int, error) {
	return s.r.PromoteKV(slot, rows)
}
func (s cudaKVSub) CopyDemotedKV(src, dst, start, end int) (int, bool, error) {
	return s.r.CopyDemotedKV(src, dst, start, end)
}
func (s cudaKVSub) ReserveKVHost(n int64) bool        { return s.r.ReserveKVHost(n) }
func (s cudaKVSub) RecurrentCheckpointBytes() int64   { return s.r.RecurrentCheckpointBytes() }
func (s cudaKVSub) SaveRecurrentCheckpoint() error    { return s.r.SaveRecurrentCheckpoint() }
func (s cudaKVSub) RestoreRecurrentCheckpoint() error { return s.r.RestoreRecurrentCheckpoint() }
func (s cudaKVSub) RecurrentCheckpointHeld(slot int) bool {
	return s.r.RecurrentCheckpointHeld(slot)
}

type cudaRuntimeSub struct {
	decoder.NoopResidentRuntime
	r *cudaResident
}

func (s cudaRuntimeSub) ResidentCaps() decoder.ResidentCaps { return s.r.ResidentCaps() }
func (s cudaRuntimeSub) CostAvailable() bool                { return true }
func (s cudaRuntimeSub) CostSpan(c *cost.Counter, p cost.Phase, work func() (int, error)) error {
	return s.r.CostSpan(c, p, work)
}
func (s cudaRuntimeSub) AdapterAvailable() bool { return true }
func (s cudaRuntimeSub) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	return s.r.SetAdapter(layers)
}
func (s cudaRuntimeSub) ExpertCacheAvailable() bool { return true }
func (s cudaRuntimeSub) ExpertCacheStats() (decoder.ExpertCacheStats, bool) {
	return s.r.ExpertCacheStats()
}
func (s cudaRuntimeSub) ResizeExpertCache(reserve int64) (int, error) {
	return s.r.ResizeExpertCache(reserve)
}
func (s cudaRuntimeSub) VerifyPathInfo() (bool, string, bool) {
	b, reason := s.r.VerifyPath()
	return b, reason, true
}
func (s cudaRuntimeSub) DecodeVerifyDivergence() error { return s.r.DecodeVerifyDivergence() }
func (s cudaRuntimeSub) EnterExactAttention() func()   { return s.r.EnterExactAttention() }
func (s cudaRuntimeSub) SeedArgmaxAvailable() bool     { return true }
func (s cudaRuntimeSub) PrefillSeedArgmax(embeddings [][]float32, startPos int) (int, error) {
	return s.r.PrefillSeedArgmax(embeddings, startPos)
}
func (s cudaRuntimeSub) LastNArgmaxAvailable() bool { return true }
func (s cudaRuntimeSub) PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	return s.r.PrefillLastNArgmax(embeddings, startPos)
}
func (s cudaRuntimeSub) DrafterHostAvailable() bool { return true }
func (s cudaRuntimeSub) CanStreamDrafterRows() bool { return true }
func (s cudaRuntimeSub) AttachBlockDrafter(w decoder.BlockDrafterWeights) (decoder.ResidentBlockDrafter, error) {
	return s.r.AttachBlockDrafter(w)
}
func (s cudaRuntimeSub) SetBatchedCapture(taps []int) error { return s.r.SetBatchedCapture(taps) }
func (s cudaRuntimeSub) BatchedCapture() [][]float32        { return s.r.BatchedCapture() }
func (s cudaRuntimeSub) RecurrentSpecAvailable() bool       { return true }
func (s cudaRuntimeSub) PrepareRecurrentSpec(rows int) error {
	return s.r.PrepareRecurrentSpec(rows)
}
func (s cudaRuntimeSub) RecurrentSpecRows() int { return s.r.RecurrentSpecRows() }
func (s cudaRuntimeSub) VerifyLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	return s.r.VerifyLastNArgmax(embeddings, startPos)
}
func (s cudaRuntimeSub) FoldVerified(n int) error { return s.r.FoldVerified(n) }
func (s cudaRuntimeSub) FlushFold() error         { return s.r.FlushFold() }
func (s cudaRuntimeSub) DropFold()                { s.r.DropFold() }
