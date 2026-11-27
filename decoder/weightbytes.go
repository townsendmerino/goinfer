package decoder

import "github.com/townsendmerino/goinfer/internal/runtime/memprice"

// ResidentWeightBytes is the total byte footprint of this model's weight matrices when every routed MoE expert is resident.
func (m *Model) ResidentWeightBytes() int64 { return m.ResidentWeightBytesPaged(0) }

// ResidentWeightBytesPaged is ResidentWeightBytes with routed experts capped at slots per layer (0 = unpaged).
func (m *Model) ResidentWeightBytesPaged(slots int) int64 {
	if m == nil {
		return 0
	}
	return memprice.WeightBytesPaged(m.w, slots)
}

// ResidentDenseWeightBytes is ResidentWeightBytesPaged's non-expert term alone.
func (m *Model) ResidentDenseWeightBytes() int64 {
	if m == nil {
		return 0
	}
	dense, _, _, _ := memprice.WeightBytesSplit(m.w, nil)
	return dense
}

// ResidentDenseWeightBytesFor is ResidentDenseWeightBytes adjusted for backend's host-side tables, K-quant/Mamba savings, and derived device tables.
func (m *Model) ResidentDenseWeightBytesFor(backend string) int64 {
	if m == nil {
		return 0
	}
	n := m.ResidentDenseWeightBytes() - memprice.HostSideBytes(backend, m.w) - m.kquantSavings(backend) - memprice.MambaProjSavings(backend, m.w)
	switch backend {
	case "cpu":
		dense, _ := memprice.CPUStructBytes(m.w)
		n += dense
	case "cuda", "webgpu":
		n += memprice.ExpertBiasBytes(m.w)
	case "metal":
		n += memprice.MetalDerivedDenseBytes(m.w, m.Program())
	}
	if backend == "cuda" {
		n += memprice.CUDADerivedTableBytes(m.w, m.Program())
	}
	return n
}

// ResidentHostCopyBytes is the portion of ResidentWeightBytesPaged's footprint that a unified-memory backend holds in both a host WeightMat and a repacked device buffer.
func (m *Model) ResidentHostCopyBytes(slots int) int64 {
	if m == nil {
		return 0
	}
	var in func([]byte) bool
	if len(m.mmap) > 0 {
		in = func(b []byte) bool { _, ok := m.MmapByteOffset(b); return ok }
	} else if m.opts.SidecarFor != "" {
		in = func([]byte) bool { return true }
	}
	return memprice.HostCopyBytes(m.w, slots, in)
}
