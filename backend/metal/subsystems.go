//go:build darwin

package metal

import (
	"context"

	"github.com/townsendmerino/goinfer/decoder"
)

// Compile-time pins for the Metal resident runner's four subsystem interfaces. Kept on wrapper types rather than on
// metalResident directly so metalResident's own method set reflects only the capabilities Metal actually implements.
var (
	_ decoder.ResidentSubsystemProvider = (*metalResident)(nil)
	_ decoder.ResidentDecodeSubsystem   = metalDecodeSub{}
	_ decoder.ResidentPrefillSubsystem  = metalPrefillSub{}
	_ decoder.ResidentKVSubsystem       = metalKVSub{}
	_ decoder.ResidentRuntimeSubsystem  = metalRuntimeSub{}
)

// ResidentSubsystems returns the Metal resident runner's four subsystem views with zero heap allocations.
func (a *metalResident) ResidentSubsystems() decoder.ResidentSubsystems {
	return decoder.ResidentSubsystems{
		Decode:  metalDecodeSub{a: a},
		Prefill: metalPrefillSub{a: a},
		KV:      metalKVSub{a: a},
		Runtime: metalRuntimeSub{a: a},
	}
}

type metalDecodeSub struct {
	decoder.NoopResidentDecode
	a *metalResident
}

func (s metalDecodeSub) SampleAvailable() bool { return s.a.SampleAvailable() }
func (s metalDecodeSub) ForwardSample(embedding []float32, pos int, temperature float64, seed, draw uint64) (int, error) {
	return s.a.ForwardSample(embedding, pos, temperature, seed, draw)
}
func (s metalDecodeSub) MRoPEDecodeAvailable() bool { return true }
func (s metalDecodeSub) ForwardMRoPE(embedding []float32, pos, ropePos int) ([]float32, error) {
	return s.a.ForwardMRoPE(embedding, pos, ropePos)
}
func (s metalDecodeSub) NoLogitsAvailable() bool                       { return true }
func (s metalDecodeSub) ForwardNoLogits(e []float32, pos int) error    { return s.a.ForwardNoLogits(e, pos) }
func (s metalDecodeSub) GreedyChainAvailable() bool                    { return s.a.GreedyChainAvailable() }
func (s metalDecodeSub) GreedyChainStart(id, pos int) error            { return s.a.GreedyChainStart(id, pos) }
func (s metalDecodeSub) GreedyChainNext() (int, error)                 { return s.a.GreedyChainNext() }
func (s metalDecodeSub) GreedyChainStop()                              { s.a.GreedyChainStop() }
func (s metalDecodeSub) SampleChainAvailable() bool                    { return s.a.SampleChainAvailable() }
func (s metalDecodeSub) SampleChainStart(id, pos int, temperature float64, seed, draw uint64) error {
	return s.a.SampleChainStart(id, pos, temperature, seed, draw)
}
func (s metalDecodeSub) SampleChainNext(seed, draw uint64) (int, error) {
	return s.a.SampleChainNext(seed, draw)
}
func (s metalDecodeSub) BatchStepperAvailable() bool  { return true }
func (s metalDecodeSub) BatchStepRange() (lo, hi int) { return s.a.BatchStepRange() }
func (s metalDecodeSub) StepBatch(seqs []decoder.ResidentBatchSeq) ([]decoder.ResidentBatchOut, error) {
	return s.a.StepBatch(seqs)
}

type metalPrefillSub struct {
	decoder.NoopResidentPrefill
	a *metalResident
}

func (s metalPrefillSub) PrefillAvailable() bool { return true }
func (s metalPrefillSub) PrefillLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return s.a.PrefillLast(ctx, embeddings, startPos)
}
func (s metalPrefillSub) PrefillRowsAvailable() bool { return true }
func (s metalPrefillSub) PrefillLastRows(ctx context.Context, n, startPos int, rows func(lo, hi int) [][]float32) ([]float32, error) {
	return s.a.PrefillLastRows(ctx, n, startPos, rows)
}
func (s metalPrefillSub) PrefillPathInfo() (bool, string, bool) {
	b, reason := s.a.PrefillPath()
	return b, reason, true
}
func (s metalPrefillSub) PrefillTailExact() bool              { return s.a.PrefillTailExact() }
func (s metalPrefillSub) PrefillBlocksAvailable() bool        { return true }
func (s metalPrefillSub) PrefillBlockRows() int               { return s.a.PrefillBlockRows() }
func (s metalPrefillSub) PrefillScratchBytes(rows int) int64  { return s.a.PrefillScratchBytes(rows) }
func (s metalPrefillSub) HiddenLastAvailable() bool           { return true }
func (s metalPrefillSub) HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return s.a.HiddenLast(ctx, embeddings, startPos)
}
func (s metalPrefillSub) ResidualAllAvailable() bool { return true }
func (s metalPrefillSub) ResidualAll(ctx context.Context, embeddings [][]float32, startPos int) ([][]float32, error) {
	return s.a.ResidualAll(ctx, embeddings, startPos)
}
func (s metalPrefillSub) ImagePrefillAvailable() bool { return true }
func (s metalPrefillSub) PrefillImageLast(ctx context.Context, embeddings [][]float32, startPos, imgStart, imgEnd int) ([]float32, error) {
	return s.a.PrefillImageLast(ctx, embeddings, startPos, imgStart, imgEnd)
}
func (s metalPrefillSub) ImageBlocksPrefillAvailable() bool { return true }
func (s metalPrefillSub) PrefillImageBlocksLast(ctx context.Context, embeddings [][]float32, startPos int, blocks [][2]int) ([]float32, error) {
	return s.a.PrefillImageBlocksLast(ctx, embeddings, startPos, blocks)
}
func (s metalPrefillSub) MRoPEPrefillAvailable() bool { return true }
func (s metalPrefillSub) PrefillMRoPELast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int) ([]float32, error) {
	return s.a.PrefillMRoPELast(ctx, embeddings, startPos, mropePos)
}
func (s metalPrefillSub) MRoPEDeepstackPrefillAvailable() bool { return true }
func (s metalPrefillSub) PrefillMRoPEDeepstackLast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int, deep [][]float32, imgStart, imgLen int) ([]float32, error) {
	return s.a.PrefillMRoPEDeepstackLast(ctx, embeddings, startPos, mropePos, deep, imgStart, imgLen)
}

type metalKVSub struct {
	decoder.NoopResidentKV
	a *metalResident
}

func (s metalKVSub) KVSlotterAvailable() bool             { return true }
func (s metalKVSub) KVSlots() int                         { return s.a.KVSlots() }
func (s metalKVSub) UseKVSlot(i int) error                { return s.a.UseKVSlot(i) }
func (s metalKVSub) SetKVSlotBusy(slot int, busy bool)    { s.a.SetKVSlotBusy(slot, busy) }
func (s metalKVSub) KVPool() decoder.KVPoolDetail         { return s.a.KVPool() }
func (s metalKVSub) ReusableKV(prefix int) int            { return s.a.ReusableKV(prefix) }
func (s metalKVSub) ReusableKVSlotAvailable() bool        { return true }
func (s metalKVSub) ReusableKVSlot(slot, prefix int) int  { return s.a.ReusableKVSlot(slot, prefix) }
func (s metalKVSub) KVUploadPassAvailable() bool          { return true }
func (s metalKVSub) BeginKVUpload() error                 { return s.a.BeginKVUpload() }
func (s metalKVSub) EndKVUpload(failed bool) error        { return s.a.EndKVUpload(failed) }
func (s metalKVSub) KVCopierAvailable() bool              { return true }
func (s metalKVSub) CopyKV(src, dst, start, end int) error {
	return s.a.CopyKV(src, dst, start, end)
}
func (s metalKVSub) RecurrentCheckpointBytes() int64     { return s.a.RecurrentCheckpointBytes() }
func (s metalKVSub) SaveRecurrentCheckpoint() error      { return s.a.SaveRecurrentCheckpoint() }
func (s metalKVSub) RestoreRecurrentCheckpoint() error   { return s.a.RestoreRecurrentCheckpoint() }
func (s metalKVSub) RecurrentCheckpointHeld(slot int) bool {
	return s.a.RecurrentCheckpointHeld(slot)
}

type metalRuntimeSub struct {
	decoder.NoopResidentRuntime
	a *metalResident
}

func (s metalRuntimeSub) ResidentCaps() decoder.ResidentCaps { return s.a.ResidentCaps() }
func (s metalRuntimeSub) AdapterAvailable() bool             { return true }
func (s metalRuntimeSub) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	return s.a.SetAdapter(layers)
}
func (s metalRuntimeSub) VerifyPathInfo() (bool, string, bool) {
	b, reason := s.a.VerifyPath()
	return b, reason, true
}
func (s metalRuntimeSub) VerifyCost() []float64      { return s.a.VerifyCost() }
func (s metalRuntimeSub) LastNArgmaxAvailable() bool { return true }
func (s metalRuntimeSub) PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	return s.a.PrefillLastNArgmax(embeddings, startPos)
}
