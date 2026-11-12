//go:build gpu

package webgpu

import (
	"context"

	"github.com/townsendmerino/goinfer/decoder"
)

// Compile-time pins for the WebGPU resident runner's four subsystem interfaces.
var (
	_ decoder.ResidentSubsystemProvider = (*residentDecoder)(nil)
	_ decoder.ResidentDecodeSubsystem   = webgpuDecodeSub{}
	_ decoder.ResidentPrefillSubsystem  = webgpuPrefillSub{}
	_ decoder.ResidentKVSubsystem       = webgpuKVSub{}
	_ decoder.ResidentRuntimeSubsystem  = webgpuRuntimeSub{}
)

// ResidentSubsystems returns the WebGPU resident runner's four subsystem views with zero heap allocations.
func (rd *residentDecoder) ResidentSubsystems() decoder.ResidentSubsystems {
	return decoder.ResidentSubsystems{
		Decode:  webgpuDecodeSub{d: rd},
		Prefill: webgpuPrefillSub{d: rd},
		KV:      webgpuKVSub{d: rd},
		Runtime: webgpuRuntimeSub{d: rd},
	}
}

type webgpuDecodeSub struct {
	decoder.NoopResidentDecode
	d *residentDecoder
}

func (s webgpuDecodeSub) SampleAvailable() bool { return s.d.SampleAvailable() }
func (s webgpuDecodeSub) ForwardSample(embedding []float32, pos int, temperature float64, seed, draw uint64) (int, error) {
	return s.d.ForwardSample(embedding, pos, temperature, seed, draw)
}
func (s webgpuDecodeSub) MRoPEDecodeAvailable() bool { return true }
func (s webgpuDecodeSub) ForwardMRoPE(embedding []float32, pos, ropePos int) ([]float32, error) {
	return s.d.ForwardMRoPE(embedding, pos, ropePos)
}
func (s webgpuDecodeSub) NoLogitsAvailable() bool                    { return true }
func (s webgpuDecodeSub) ForwardNoLogits(e []float32, pos int) error { return s.d.ForwardNoLogits(e, pos) }

type webgpuPrefillSub struct {
	decoder.NoopResidentPrefill
	d *residentDecoder
}

func (s webgpuPrefillSub) PrefillAvailable() bool { return true }
func (s webgpuPrefillSub) PrefillLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return s.d.PrefillLast(ctx, embeddings, startPos)
}
func (s webgpuPrefillSub) PrefillChunksAvailable() bool       { return true }
func (s webgpuPrefillSub) PrefillChunkRows() int              { return s.d.PrefillChunkRows() }
func (s webgpuPrefillSub) PrefillScratchBytes(rows int) int64 { return s.d.PrefillScratchBytes(rows) }
func (s webgpuPrefillSub) MRoPEPrefillAvailable() bool        { return true }
func (s webgpuPrefillSub) PrefillMRoPELast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int) ([]float32, error) {
	return s.d.PrefillMRoPELast(ctx, embeddings, startPos, mropePos)
}

type webgpuKVSub struct {
	decoder.NoopResidentKV
	d *residentDecoder
}

func (s webgpuKVSub) KVSlotterAvailable() bool { return true }
func (s webgpuKVSub) KVSlots() int             { return s.d.KVSlots() }
func (s webgpuKVSub) UseKVSlot(i int) error    { return s.d.UseKVSlot(i) }
func (s webgpuKVSub) KVCopierAvailable() bool  { return true }
func (s webgpuKVSub) CopyKV(src, dst, start, end int) error {
	return s.d.CopyKV(src, dst, start, end)
}

type webgpuRuntimeSub struct {
	decoder.NoopResidentRuntime
	d *residentDecoder
}

func (s webgpuRuntimeSub) ResidentCaps() decoder.ResidentCaps { return s.d.ResidentCaps() }
func (s webgpuRuntimeSub) AdapterAvailable() bool             { return true }
func (s webgpuRuntimeSub) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	return s.d.SetAdapter(layers)
}
func (s webgpuRuntimeSub) VerifyPathInfo() (bool, string, bool) {
	b, reason := s.d.VerifyPath()
	return b, reason, true
}
