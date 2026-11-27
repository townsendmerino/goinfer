package decoder

import (
	"fmt"
	"sync/atomic"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/memprice"
)

// kquantNativeOff empties every backend's kquantLane, so a load keeps the int8 copies on the device and in its pricing: the arm
// without the native K-quant lane. Set only by a test (SetKQuantNativeOffForTest).
var kquantNativeOff atomic.Bool

// kquantSavings is the device bytes backend's q4k lane does not allocate because it uploads the file's blocks in place of int8 copies.
func (m *Model) kquantSavings(backend string) int64 {
	if m == nil || m.w == nil {
		return 0
	}
	return memprice.KQuantSavings(backend, m.w, m.Quant(), m.opts.ActQuantGroup, kquantNativeOff.Load())
}

// KQuantNative is "q5k" or "q6k" when backend's q4k lane uploads wm as the file's blocks, else "".
func (m *Model) KQuantNative(backend string, wm *linalg.WeightMat) string {
	if m == nil || m.w == nil {
		return ""
	}
	return memprice.KQuantNativeKind(backend, m.w, m.Quant(), m.opts.ActQuantGroup, kquantNativeOff.Load(), wm)
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
