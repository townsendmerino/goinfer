package vision

import (
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// newQMat wraps an f32 [rows, cols] matmul weight as a linalg.WeightMat, holding
// it either as f32 or as per-row int8 (W8A8) when quant is set. Quantizing once at
// load runs the integer W8A8 kernel — ~2–4× the f32 throughput on the matmul-bound
// SigLIP tower, at the cost of cosine ~0.999 vs the f32 path's 1.0.
//
// This function is the tower's storage POLICY — which weights quantize, and that
// f32 is copied. The precision dispatch itself is WeightMat's job: vision used to
// open-code its own `qmat` wrapper (one of the three WeightMat was written to
// consolidate, alongside encoder.LayerWeightsQ8 and goinfer's decoder.weightMat),
// and this is that migration. Weights dispatch through WeightMat.MatmulBT into the
// same linalg kernels as before, so the numerics are unchanged.
//
// Attention/FFN projections quantize under -vision-quant; the patch-embed conv
// deliberately does not — quant error on the input embedding would propagate
// through every layer.
//
// The source slice is never retained: the f32 path copies (WrapF32 aliases, and
// the caller's tensor is an mmap released after load), and the int8 path consumes
// w into fresh codes + scales.
func newQMat(w []float32, rows, cols int, quant bool) linalg.WeightMat {
	if quant {
		// w8a8=true selects MatmulBTW8A8 — the kernel this tower already used.
		return linalg.QuantizeInt8(w, rows, cols, true)
	}
	return linalg.WrapF32(append([]float32(nil), w...), rows, cols)
}

// tensorViewer is a tensor source that hands out a checkpoint tensor's bytes in place: a mapped safetensors file, or one
// behind a name prefix. A caller's TensorSource is not one; its weights take newQMat's copying path.
type tensorViewer interface {
	Tensor(name string) (embed.Tensor, error)
}

// aliasQMat is newQMat over the checkpoint tensor name without copying it (MEM4(a)): a bf16 or f16 tensor is held as
// linalg.WrapBF16 / WrapF16 over the mapped bytes, an f32 one as WrapF32 over them (a copy only where the payload is
// misaligned in the file, as embed.Tensor.Float32s does). Every matmul reads the same values newQMat's f32 copy holds
// (widening is exact), so the outputs are the same bits. quant quantizes from a widened copy, as newQMat does. ok is false
// when src cannot hand out tensors in place, or the tensor's bytes cannot be viewed (a 16-bit payload on an odd offset); the
// caller then takes newQMat's path. The result aliases src's mapping: the tower keeps it open (towerMaps) while the weight
// lives.
func aliasQMat(src any, name string, rows, cols int, quant bool) (w linalg.WeightMat, ok bool, err error) {
	v, isView := src.(tensorViewer)
	if quant || !isView || aliasOffForTest {
		return linalg.WeightMat{}, false, nil
	}
	t, err := v.Tensor(name)
	if err != nil {
		return linalg.WeightMat{}, false, err
	}
	if len(t.Shape) != 2 || t.Shape[0] != rows || t.Shape[1] != cols {
		return linalg.WeightMat{}, false, fmt.Errorf("vision: %s has shape %v, want [%d %d]", name, t.Shape, rows, cols)
	}
	switch t.DType {
	case "BF16", "F16":
		u, err := t.Uint16s()
		if err != nil { // an odd payload offset: no view; newQMat's widened copy instead
			return linalg.WeightMat{}, false, nil
		}
		aliasedForTest.Add(1)
		if t.DType == "BF16" {
			return linalg.WrapBF16(u, rows, cols), true, nil
		}
		return linalg.WrapF16(u, rows, cols), true, nil
	case "F32":
		f, err := t.Float32s()
		if err != nil {
			return linalg.WeightMat{}, false, err
		}
		aliasedForTest.Add(1)
		return linalg.WrapF32(f, rows, cols), true, nil
	}
	return linalg.WeightMat{}, false, nil
}

// aliasedForTest counts the weights aliasQMat has wrapped in place, for a test to see the path ran.
var aliasedForTest atomic.Int64

// aliasOffForTest makes every loader copy its weights as before MEM4(a) (aliasQMat declines): the old arm of ALS-G1.
var aliasOffForTest bool

// errTowerClosed is what a tower's forward returns after Close.
var errTowerClosed = errors.New("vision: the tower is closed")

// towerMaps is the checkpoint mappings a tower's aliased weights read from (aliasQMat). The tower owns them and Close releases
// them, after which its weights are gone and its forwards return errTowerClosed. base is the load's own mappings; reloads
// holds a block reload's (blockRelease), released again with the blocks. The zero value holds none.
type towerMaps struct {
	mu      sync.Mutex
	base    []io.Closer
	reloads []io.Closer
	closed  bool
}

// keep holds st for the tower's life.
func (m *towerMaps) keep(st io.Closer) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.base = append(m.base, st)
}

// adopt takes a block reload's mappings (another load of the same tower, whose blocks this tower now holds) as reloads.
func (m *towerMaps) adopt(from *towerMaps) {
	from.mu.Lock()
	got := append(from.base, from.reloads...)
	from.base, from.reloads = nil, nil
	from.mu.Unlock()
	m.mu.Lock()
	defer m.mu.Unlock()
	m.reloads = append(m.reloads, got...)
}

// dropReloads releases the reload mappings, once the blocks that read them are dropped (ReleaseBlocks).
func (m *towerMaps) dropReloads() {
	m.mu.Lock()
	rs := m.reloads
	m.reloads = nil
	m.mu.Unlock()
	closeAll(rs)
}

// close releases every mapping and marks the tower closed. Idempotent.
func (m *towerMaps) close() {
	m.mu.Lock()
	all := append(m.base, m.reloads...)
	m.base, m.reloads, m.closed = nil, nil, true
	m.mu.Unlock()
	closeAll(all)
}

// check is a forward's guard: errTowerClosed once close has run.
func (m *towerMaps) check() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return errTowerClosed
	}
	return nil
}

// closeAll closes each mapping. An unmap failure leaves the region mapped with nothing reading it, so it is not reported: the
// callers (Close, ReleaseBlocks) have no error of their own to carry it.
func closeAll(cs []io.Closer) {
	for _, c := range cs {
		_ = c.Close()
	}
}

// widenHalf16 is a 16-bit-held weight's exact f32 widening, for an export whose consumer reads f32 (a device upload); nil for
// any other kind.
func widenHalf16(w linalg.WeightMat) []float32 {
	u, bf16, ok := w.Half16()
	if !ok {
		return nil
	}
	f := make([]float32, len(u))
	if bf16 {
		for i, v := range u {
			f[i] = linalg.BF16ToF32(v)
		}
	} else {
		linalg.F16ToF32Slice(f, u)
	}
	return f
}
