//go:build cuda

package cuda

import (
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// L-01 hybrid CPU/GPU MoE expert execution (docs/task-l01-hybrid-moe-cpu-gpu.md) — PROTOTYPE.
// This file only extracts and CPU-computes one expert from C′'s pinned host stack; nothing
// here is called from loadRoutedExperts yet. See docs/task-l01-hybrid-moe-cpu-gpu.md §9/§10
// for why correctness of this extraction is verified in isolation FIRST, before any decode-path
// wiring: a wrong nibble layout here would silently produce plausible-looking wrong logits.

// unpermuteFast is the exact inverse of permuteFast (cuda/kernels.go): converts the
// fast-nibble-permuted layout the coalesced forward GEMV expects back to the plain,
// natural-order int4 byte layout every OTHER backend's kernel (and WrapInt4) assumes. A fixed
// nibble-POSITION permutation, content-independent — verified round-trip against permuteFast
// for 2,000,000 sample words (2026-09-10) before use, which already proves it for every
// possible word since the map never looks at nibble VALUES, only position.
func unpermuteFast(w uint32) uint32 {
	var o uint32
	for p := 0; p < 8; p++ {
		nv := (w >> (4 * p)) & 0xf
		var i int
		if p%2 == 0 {
			i = p / 2
		} else {
			i = 4 + (p-1)/2
		}
		o |= nv << (4 * i)
	}
	return o
}

// bytesToU32 / bytesToU16 are u32bytes/u16bytes (cuda/resident.go) run in reverse — same
// unsafe.Slice reinterpretation this package already uses in the other direction, applied to a
// *gpu.MappedHostBuffer's byte view so it can be read back as the packed words it holds.
func bytesToU32(b []byte) []uint32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func bytesToU16(b []byte) []uint16 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*uint16)(unsafe.Pointer(&b[0])), len(b)/2)
}

// l01ExtractExpertGU returns expert e's Gate and Up as separate WeightMats, read from the
// FUSED expGU pinned host stack (cuda/resident.go's own comment: "stacked [nE * 2*moeInter,
// hidden]: expert e's gate at e*2*moeInter, up at +moeInter") — weight nibbles unpermuted back
// to natural order, scales decoded f16→f32 (WrapInt4 wants f32; CUDA's f16 bit encoding matches
// every other backend's per audit C-15, so this introduces no rounding drift beyond what every
// backend's own f16 scale already carries). Uses w.perExpertW/perExpertS (cacheWQ's own stored
// per-expert stride) rather than re-deriving rowsPerExpert, so this can never drift from the
// value the DMA-miss path already trusts.
func (r *cudaResident) l01ExtractExpertGU(L *cudaLayer, e int) (gate, up linalg.WeightMat) {
	w := &L.expGU
	hidden := r.hidden
	moeInter := r.moeInter

	srcW := bytesToU32(w.srcW.Bytes())
	wOff := e * w.perExpertW
	scratch := make([]uint32, w.perExpertW)
	for i := range scratch {
		scratch[i] = unpermuteFast(srcW[wOff+i])
	}
	scratchBytes := u32bytes(scratch)

	srcS := bytesToU16(w.srcS.Bytes())
	sOff := e * w.perExpertS
	scales := make([]float32, w.perExpertS)
	for i := range scales {
		scales[i] = decoder.F16BitsToF32(srcS[sOff+i])
	}

	halfWords := w.perExpertW / 2
	halfScales := w.perExpertS / 2
	gate = linalg.WrapInt4(scratchBytes[:halfWords*4], scales[:halfScales], moeInter, hidden, 32)
	up = linalg.WrapInt4(scratchBytes[halfWords*4:], scales[halfScales:], moeInter, hidden, 32)
	return gate, up
}

// l01ExtractExpertDown returns expert e's Down projection, read from the expDown pinned host
// stack ("stacked [nE * hidden, moeInter]" — cuda/resident.go's own comment). Same
// unpermute+decode as l01ExtractExpertGU; no gate/up split needed since Down is one block.
func (r *cudaResident) l01ExtractExpertDown(L *cudaLayer, e int) linalg.WeightMat {
	w := &L.expDown
	hidden := r.hidden
	moeInter := r.moeInter

	srcW := bytesToU32(w.srcW.Bytes())
	wOff := e * w.perExpertW
	scratch := make([]uint32, w.perExpertW)
	for i := range scratch {
		scratch[i] = unpermuteFast(srcW[wOff+i])
	}

	srcS := bytesToU16(w.srcS.Bytes())
	sOff := e * w.perExpertS
	scales := make([]float32, w.perExpertS)
	for i := range scales {
		scales[i] = decoder.F16BitsToF32(srcS[sOff+i])
	}
	return linalg.WrapInt4(u32bytes(scratch), scales, hidden, moeInter, 32)
}

// l01ComputeExpert runs expert e's SwiGLU MLP entirely from L-01's pinned-host extraction
// (l01ExtractExpertGU/Down) — no CUDA involved, a pure CPU compute path exercising the exact
// bytes the DMA-miss path would otherwise have fetched. Not yet called from the decode path.
func (r *cudaResident) l01ComputeExpert(L *cudaLayer, e int, h, dst []float32) {
	gate, up := r.l01ExtractExpertGU(L, e)
	down := r.l01ExtractExpertDown(L, e)
	inter := r.moeInter
	gateScr := make([]float32, inter)
	upScr := make([]float32, inter)
	decoder.ComputeExpertMLP(decoder.CPUExpertWeights{Gate: gate, Up: up, Down: down}, h, dst, inter, gateScr, upScr)
}
