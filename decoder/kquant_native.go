package decoder

import (
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"

	"fmt"
	"slices"
	"sync/atomic"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

// kquantNativeOff empties every backend's kquantLane, so a load keeps the int8 copies on the device and in its pricing: the arm
// without the native K-quant lane. Set only by a test (SetKQuantNativeOffForTest).
var kquantNativeOff atomic.Bool

// kquantLane is the weights backend's q4k lane uploads as the file's Q5_K and Q6_K blocks (KQuantReader) in place of their
// int8 copies, on a q4k load: those with a recorded source among the projections the backend's build hands its lane.
//   - metal: each layer's q, k, v, o, gate, up and down (buildResident calls q4kLaneProj on exactly those).
//   - cuda: the projections BuildResident plans (a KV-shared layer has no k or v, a K=V layer no v) and the untied LM head, under
//     per-32 activations only (the native GEMVs read the group sums quantG32 writes; every q4k load has them).
//
// Any other backend, or a load that is not q4k, has no lane.
func (m *Model) kquantLane(backend string) []*linalg.WeightMat {
	if kquantNativeOff.Load() || m == nil || m.w == nil || len(m.w.KQuant) == 0 || m.Quant() != "q4k" || (backend != "metal" && backend != "cuda") {
		return nil
	}
	if backend == "cuda" && m.opts.ActQuantGroup != 32 {
		return nil
	}
	var ws []*linalg.WeightMat
	for l := range m.w.Layers {
		lw := &m.w.Layers[l]
		ps := []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj, &lw.OProj, &lw.GateProj, &lw.UpProj, &lw.DownProj}
		if backend == "cuda" {
			switch {
			case m.BackendView().Layer(l).KVSrc() != l:
				ps = []*linalg.WeightMat{&lw.QProj, &lw.OProj, &lw.GateProj, &lw.UpProj, &lw.DownProj}
			case m.BackendView().Layer(l).VFromK():
				ps = []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.OProj, &lw.GateProj, &lw.UpProj, &lw.DownProj}
			}
		}
		for _, wm := range ps {
			if _, ok := m.w.KQuantOf(wm); ok {
				ws = append(ws, wm)
			}
		}
	}
	if _, ok := m.w.KQuantOf(&m.w.LMHead); ok && backend == "cuda" && m.w.LMHead.Rows() > 0 {
		ws = append(ws, &m.w.LMHead)
	}
	return ws
}

// kquantSavings is the device bytes backend's q4k lane does not allocate because it uploads the file's blocks for the weights
// of kquantLane: their int8 bytes less their native bytes. The host copy keeps the int8 copies, so only the device term of a
// backend's need takes it (ResidentNeedBytes, ResidentDenseWeightBytesFor).
func (m *Model) kquantSavings(backend string) int64 {
	var n int64
	for _, wm := range m.kquantLane(backend) {
		s, _ := m.w.KQuantOf(wm)
		n += s.Int8Bytes - s.NativeBytes
	}
	return n
}

// KQuantNative is "q5k" or "q6k" when backend's q4k lane uploads wm as the file's blocks (kquantLane), else "". wm is
// compared by address, so it must be the model's own weight (a pointer into Weights()).
func (m *Model) KQuantNative(backend string, wm *linalg.WeightMat) string {
	if slices.Contains(m.kquantLane(backend), wm) {
		s, _ := m.w.KQuantOf(wm)
		return s.Kind
	}
	return ""
}

// KQuantReader serves the GGUF's own Q5_K and Q6_K blocks for the int8 matmuls of a q4k load that the file stores that way,
// so a GPU lane can upload them in place of the int8 copy. The CPU path keeps the int8 copy. The blocks it passes alias the
// file mapping, valid until Close.
type KQuantReader struct {
	g     *embed.GGUFFile
	w     *darch.Weights
	owned bool // g was opened for this reader (Close unmaps it), not the load's mapping
}

// OpenKQuant returns a reader over m's Q5_K and Q6_K blocks: the load's own mapping while the model has not yet released it
// (through its resident build), else the file mapped again, which parses its header anew. It returns nil and no error when
// the model holds no such weight (a load at another quant, a .giw, a GGUF with no such tensor).
func (m *Model) OpenKQuant() (*KQuantReader, error) {
	if m.w == nil || len(m.w.KQuant) == 0 || m.w.GGUFPath == "" {
		return nil, nil
	}
	if m.w.KQuantFile != nil {
		return &KQuantReader{g: m.w.KQuantFile, w: m.w}, nil
	}
	g, err := OpenGGUFMmap(m.w.GGUFPath)
	if err != nil {
		return nil, err
	}
	return &KQuantReader{g: g, w: m.w, owned: true}, nil
}

// releaseKQuantFile unmaps the load's GGUF once the resident build that reads it has run. A later OpenKQuant maps the
// file again.
func (m *Model) releaseKQuantFile() {
	if m.w != nil && m.w.KQuantFile != nil {
		_ = m.w.KQuantFile.Close()
		m.w.KQuantFile = nil
	}
}

// Kind is "q5k" or "q6k" when wm is an int8 copy of a tensor the file stores that way, else "". A nil reader reports "".
func (k *KQuantReader) Kind(wm *linalg.WeightMat) string {
	if k == nil {
		return ""
	}
	s, _ := k.w.KQuantOf(wm)
	return s.Kind
}

// Rows calls each with every destination row of wm and that row's raw blocks in the file (176 bytes per 256 columns for
// Q5_K, 210 for Q6_K), in row order.
func (k *KQuantReader) Rows(wm *linalg.WeightMat, each func(r int, raw []byte) error) error {
	row, err := k.RowSource(wm)
	if err != nil {
		return err
	}
	for r := range wm.Rows() {
		if err := each(r, row(r)); err != nil {
			return err
		}
	}
	return nil
}

// RowSource validates wm's source once and returns destination row r's raw blocks in the file, for r in [0, wm.Rows()).
// The slices alias the mapping (valid until Close) and the function is safe for concurrent use.
func (k *KQuantReader) RowSource(wm *linalg.WeightMat) (func(r int) []byte, error) {
	src, ok := k.w.KQuantOf(wm)
	if !ok {
		return nil, fmt.Errorf("decoder: KQuantReader: weight has no Q5_K/Q6_K source")
	}
	get, bb := k.g.Q5KRaw, 176
	if src.Kind == "q6k" {
		get, bb = k.g.Q6KRaw, 210
	}
	dims, raw, ok, err := get(src.Tensor)
	if err != nil {
		return nil, err
	}
	if !archparts.KQuantServes(dims, ok, wm.Cols(), src, wm.Rows()) {
		return nil, fmt.Errorf("decoder: KQuantReader: %q is not %s covering %d rows of %d columns (dims %v)", src.Tensor, src.Kind, wm.Rows(), wm.Cols(), dims)
	}
	rb := dims[0] / 256 * bb
	return func(r int) []byte { s := src.Src(r); return raw[s*rb : (s+1)*rb] }, nil
}

// Close unmaps the file when the reader mapped it; the load's own mapping stays for releaseKQuantFile.
func (k *KQuantReader) Close() error {
	if k == nil || !k.owned {
		return nil
	}
	return k.g.Close()
}
