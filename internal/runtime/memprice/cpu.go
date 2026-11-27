package memprice

import (
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// CPUKVQuant identifies the storage format of a CPU KV cache for byte pricing.
type CPUKVQuant uint8

const (
	CPUKVF32 CPUKVQuant = 0
	CPUKVI8  CPUKVQuant = 1
	CPUKVF16 CPUKVQuant = 2
)

// CPUKVBytesForCtx prices what a CPU cache (NewCache) holds for ctx positions of a.
func CPUKVBytesForCtx(a *darch.Architecture, ctx int, f16, i8 bool) int64 {
	if a == nil || ctx <= 0 || a.NumLayers <= 0 {
		return 0
	}
	quant := CPUKVF32
	if i8 && a.KVInt8OK() {
		quant = CPUKVI8
	} else if f16 && a.KVF16OK() {
		quant = CPUKVF16
	}
	return CPUKVBytesForRows(a, ctx, quant, LayerBuffersOf(a))
}

// CPUKVBytesForRows prices a CPU cache from the per-position rows each layer declares at precision quant.
func CPUKVBytesForRows(a *darch.Architecture, ctx int, quant CPUKVQuant, bufsOf func(l int) []program.Buf) int64 {
	var total int64
	for l := range a.NumLayers {
		if a.HasNoAttentionKVAt(l) {
			continue
		}
		for _, b := range bufsOf(l) {
			if b.Unit != program.PerPosition {
				continue
			}
			rows := int64(ctx)
			if b.Window > 0 && a.RingLayerAt(l) {
				rows = int64(b.Window)
				if quant == CPUKVF32 && ctx > b.Window {
					rows *= 2
				}
			}
			row := 4 * int64(b.Elems)
			switch {
			case b.Prec != program.PrecKV:
			case quant == CPUKVI8:
				row = int64(b.Elems) + 4*int64(b.Heads)
			case quant == CPUKVF16:
				row = 2*int64(b.Elems) + 2*int64(cpuKVRowPadPriced(a, b.Elems, 2))
			default:
				row += 4 * int64(cpuKVRowPadPriced(a, b.Elems, 4))
			}
			total += rows * row
		}
	}
	return total
}

func cpuKVRowPadPriced(a *darch.Architecture, elems, elemBytes int) int {
	if elems != a.NumKVHeads*a.HeadDim {
		return 0
	}
	return a.KVRowPad(elemBytes)
}

// CPUStructBytes returns the host struct overhead of w beside its tensor slices, split into dense and expert lines.
func CPUStructBytes(w *darch.Weights) (dense, experts int64) {
	if w == nil {
		return 0, 0
	}
	dense = int64(unsafe.Sizeof(*w)) + int64(cap(w.Layers))*int64(unsafe.Sizeof(darch.LayerWeights{})) +
		int64(len(w.KQuant))*int64(unsafe.Sizeof((*int8)(nil))+unsafe.Sizeof(archparts.KQuantSrc{}))
	for i := range w.Layers {
		if w.Arch != nil && w.Arch.WeightLayerOf(i) != i {
			continue // a later visit of an Inkling 2 loop layer shares the first visit's structs
		}
		l := &w.Layers[i]
		dense += ptrSize(l.Delta) + ptrSize(l.QAttn) + ptrSize(l.ShortConv) + ptrSize(l.Mamba) + ptrSize(l.MLA) + ptrSize(l.KDA) + ptrSize(l.Gemma4MoE) + ptrSize(l.Inkling)
		experts += int64(cap(l.Experts)) * int64(unsafe.Sizeof(archparts.ExpertWeights{}))
		if mo := l.Gemma4MoE; mo != nil {
			experts += int64(cap(mo.ExpertsGateUp)+cap(mo.ExpertsDown)) * int64(unsafe.Sizeof(linalg.WeightMat{}))
		}
	}
	return dense, experts
}

func ptrSize[T any](p *T) int64 {
	if p == nil {
		return 0
	}
	return int64(unsafe.Sizeof(*p))
}
