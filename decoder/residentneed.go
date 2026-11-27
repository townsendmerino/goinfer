package decoder

import (
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/memprice"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// WeightsMemFraction is the share of a memory figure the weights (and the rest of a resident build)
// may occupy before a load is refused.
const WeightsMemFraction = mem.WeightsFraction

// metalKVPad is the position granularity Metal's KV takes: its pools hold 64-row blocks.
const metalKVPad = KVBlockRows

// ResidentKVBytes is the KV cache a resident build on backend allocates for ctx positions.
func (m *Model) ResidentKVBytes(backend string, ctx int, kvF16, kvI8 bool) int64 {
	if ctx <= 0 {
		return 0
	}
	switch backend {
	case "cuda":
		slot, null := m.cudaKVParts(ctx, kvF16, kvI8)
		return slot + null
	case "cpu":
		if m.w.Arch == nil {
			return m.kvBytesPerPositionAllLayers(kvF16, kvI8) * int64(ctx)
		}
		return memprice.CPUKVBytesForCtx(m.w.Arch, ctx, kvF16, kvI8)
	case "metal":
		slot, null := memprice.MetalKVPartsFor(m.w.Arch, ctx, m.KVDeliveredFor("metal"), m.LayerBuffers, m.MetalWindowBlocks())
		return slot + null
	default:
		return m.kvBytesPerPositionAllLayers(kvF16, kvI8) * int64(ctx)
	}
}

// MetalKVSlotBytes is what one more resident KV slot costs a metal load at ctx positions: ResidentKVBytes less the pools' null blocks.
func (m *Model) MetalKVSlotBytes(ctx int) int64 {
	slot, _ := memprice.MetalKVPartsFor(m.w.Arch, ctx, m.KVDeliveredFor("metal"), m.LayerBuffers, m.MetalWindowBlocks())
	return slot
}

// MetalKVBytes is the KV a metal load of slots resident slots holds at ctx positions.
func (m *Model) MetalKVBytes(slots, ctx int) int64 {
	return m.ResidentKVBytes("metal", ctx, false, false) + int64(slots-1)*m.MetalKVSlotBytes(ctx) + m.MetalKVTableRaiseBytes(slots, ctx)
}

// MetalKVCap is one conversation's cap on a metal load of slots KV slots at a planned per-slot context of share positions.
func (m *Model) MetalKVCap(slots, share int) int {
	return memprice.MetalKVCap(slots, share, m.Config().MaxPositions, m.ResidentContextRequest() > 0,
		m.ExtraResidentKVPerPosition(), m.KVDeliveredFor("metal"))
}

// MetalKVTableRaiseBytes is what slots' block tables hold past the per-slot price at ctx when one conversation's cap is raised.
func (m *Model) MetalKVTableRaiseBytes(slots, ctx int) int64 {
	return memprice.MetalKVTableRaiseBytes(m.w.Arch, slots, ctx, m.MetalKVCap(slots, ctx), ctx > 0 && m.MetalKVSlotBytes(ctx) > 0)
}

// MetalKVRowBytes is one position's row of b in Metal's KV pools when delivered at precision delivered.
func MetalKVRowBytes(b program.Buf, delivered string) int64 {
	return memprice.MetalRowBytesFor(b, delivered)
}

// ResidentNeedBytes is what a resident build on backend asks the device (or, on Metal, unified memory)
// to hold with slots expert slots per layer (0 = every expert resident) and a ctx-position KV cache.
func (m *Model) ResidentNeedBytes(backend string, slots, ctx int, kvF16, kvI8 bool) int64 {
	need := m.ResidentWeightBytesPaged(slots) - m.kquantSavings(backend) - memprice.MambaProjSavings(backend, m.w) +
		m.ResidentKVBytes(backend, ctx, kvF16, kvI8) + m.stateBytes(backend)
	switch backend {
	case "metal":
		need += m.ResidentHostCopyBytes(slots) + m.MetalDecodeScratchBytes(slots) + memprice.MetalDerivedDenseBytes(m.w, m.Program())
	case "cpu":
		dense, experts := memprice.CPUStructBytes(m.w)
		need += dense + experts
	}
	return need
}

// residentHostCopyFor is ResidentNeedBytes' host-copy term alone, for Plan's piecewise accounting.
func (m *Model) residentHostCopyFor(backend string, slots int) int64 {
	if backend != "metal" {
		return 0
	}
	return m.ResidentHostCopyBytes(slots)
}

// CUDAKVSlotBytes is what one more resident KV slot costs a cuda load at ctx positions: cudaKVBytes less the null blocks.
func (m *Model) CUDAKVSlotBytes(ctx int, kvF16, kvI8 bool) int64 {
	slot, _ := m.cudaKVParts(ctx, kvF16, kvI8)
	return slot
}

func (m *Model) cudaKVParts(ctx int, kvF16, kvI8 bool) (slot, null int64) {
	a := m.w.Arch
	bare := Options{Backend: "cuda"}
	prec := kvPrecF32
	if p, ok := kvPolicies["cuda"]; kvI8 && ok && p.DeclineI8 != nil && p.DeclineI8(a.KVFacts(), bare) == "" {
		prec = kvPrecI8
	} else if kvF16 && kvF16Decline("cuda", a, bare) == "" {
		prec = kvPrecF16
	}
	return memprice.CUDAKVParts(a, ctx, prec, m.opts.ResidentWindowedKV, m.LayerBuffers)
}
