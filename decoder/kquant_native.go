package decoder

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// kquantSrc is the GGUF tensor an int8 matmul of a q4k load was re-quantized from: kind "q5k" or "q6k", and the source row
// of destination row r: start+r for a plain load or a fused split (perm nil), else perm[r] (a RoPE permutation).
type kquantSrc struct {
	tensor string
	kind   string
	start  int
	perm   []int32
	// int8Bytes and nativeBytes are the weight's int8 copy (rows and per-row scales) and the file's blocks for it: what a lane
	// that uploads the blocks saves on the device (kquantSavings).
	int8Bytes, nativeBytes int64
}

// src is destination row r's source row.
func (s kquantSrc) src(r int) int {
	if s.perm != nil {
		return int(s.perm[r])
	}
	return s.start + r
}

// kquantRows maps rows destination rows to their source rows (rowSrc(r) / in): start for a run start, start+1, ... (perm nil),
// else every row's source in perm.
func kquantRows(rows, in int, rowSrc func(r int) int) (start int, perm []int32) {
	start = rowSrc(0) / in
	for r := range rows {
		if rowSrc(r)/in != start+r {
			perm = make([]int32, rows)
			for i := range perm {
				perm[i] = int32(rowSrc(i) / in)
			}
			return start, perm
		}
	}
	return start, nil
}

// noteKQuant records wm's source when quant is q4k, the load retains its layers (no .giw sink) and tensor name is stored as
// Q5_K or Q6_K: wm is then an int8 copy whose file blocks a GPU lane can multiply natively (KQuantReader). rowSrc(r) is
// destination row r's element offset in the tensor, as q4kMat takes it. A source kquantServes rejects is not recorded. The
// record is metadata only; nothing is copied.
func (w *Weights) noteKQuant(g *embed.GGUFFile, name string, wm linalg.WeightMat, in int, rowSrc func(r int) int) {
	q8, sc, _, ok := wm.Int8()
	if !ok || len(q8) == 0 || wm.Cols() != in {
		return
	}
	kind, bb := "", int64(0)
	var dims []int
	for k, get := range map[string]func(string) ([]int, []byte, bool, error){"q5k": g.Q5KRaw, "q6k": g.Q6KRaw} {
		if d, _, ok, err := get(name); err == nil && ok {
			kind, dims, bb = k, d, map[string]int64{"q5k": 176, "q6k": 210}[k]
		}
	}
	if kind == "" {
		return // another type: no row map is built for it
	}
	start, perm := kquantRows(wm.Rows(), in, rowSrc)
	src := kquantSrc{tensor: name, kind: kind, start: start, perm: perm}
	if !kquantServes(dims, true, wm.Cols(), src, wm.Rows()) {
		return
	}
	src.int8Bytes, src.nativeBytes = int64(len(q8))+4*int64(len(sc)), int64(wm.Rows())*int64(in/256)*bb
	kquantMu.Lock()
	defer kquantMu.Unlock()
	if w.kquant == nil {
		w.kquant = map[*int8]kquantSrc{}
	}
	w.kquant[unsafe.SliceData(q8)] = src
}

// kquantServes reports whether a raw accessor's answer (dims, ok; its caller has already handled an error) for a weight of rows
// destination rows over in columns, mapped by src, is one KQuantReader.Rows will serve: stored as the kind asked, two dims whose
// columns are in (positive, a multiple of 256), and every source row inside the tensor (a run, or a permutation of exactly rows
// entries).
// noteKQuant records only such a source and Rows checks the same, both with the weight's own columns and rows, so a recorded
// weight is one the lane uploads natively.
func kquantServes(dims []int, ok bool, in int, src kquantSrc, rows int) bool {
	if !ok || len(dims) != 2 || dims[0] != in || in <= 0 || in%256 != 0 || rows <= 0 {
		return false
	}
	if src.perm == nil {
		return src.start >= 0 && src.start+rows <= dims[1]
	}
	if len(src.perm) != rows {
		return false
	}
	for _, r := range src.perm {
		if r < 0 || int(r) >= dims[1] {
			return false
		}
	}
	return true
}

// kquantMu guards every Weights.kquant while a load records into it: parallelLayers builds layers on several goroutines.
// Package-level because Weights is copied by value in places, and loads are few.
var kquantMu sync.Mutex

// kquantNativeOff empties every backend's kquantLane, so a load keeps the int8 copies on the device and in its pricing: KS-O1's
// old arm. Set only by a test (SetKQuantNativeOffForTest).
var kquantNativeOff atomic.Bool

// kquantLane is the weights backend's q4k lane uploads as the file's Q5_K and Q6_K blocks (KQuantReader) in place of their
// int8 copies, on a q4k load: those with a recorded source among the projections the backend's build hands its lane.
//   - metal: each layer's q, k, v, o, gate, up and down (buildResident calls q4kLaneProj on exactly those).
//   - cuda: the projections BuildResident plans (a KV-shared layer has no k or v, a K=V layer no v) and the untied LM head, under
//     per-32 activations only (the native GEMVs read the group sums quantG32 writes; every q4k load has them).
//
// Any other backend, or a load that is not q4k, has no lane.
func (m *Model) kquantLane(backend string) []*linalg.WeightMat {
	if kquantNativeOff.Load() || m == nil || m.w == nil || len(m.w.kquant) == 0 || m.Quant() != "q4k" || (backend != "metal" && backend != "cuda") {
		return nil
	}
	if backend == "cuda" && m.actGroup != 32 {
		return nil
	}
	var ws []*linalg.WeightMat
	for l := range m.w.Layers {
		lw := &m.w.Layers[l]
		ps := []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj, &lw.OProj, &lw.GateProj, &lw.UpProj, &lw.DownProj}
		if backend == "cuda" {
			switch {
			case m.KVSrcAtResident(l) != l:
				ps = []*linalg.WeightMat{&lw.QProj, &lw.OProj, &lw.GateProj, &lw.UpProj, &lw.DownProj}
			case m.VFromKResident(l):
				ps = []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.OProj, &lw.GateProj, &lw.UpProj, &lw.DownProj}
			}
		}
		for _, wm := range ps {
			if _, ok := m.w.kquantOf(wm); ok {
				ws = append(ws, wm)
			}
		}
	}
	if _, ok := m.w.kquantOf(&m.w.LMHead); ok && backend == "cuda" && m.w.LMHead.Rows() > 0 {
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
		s, _ := m.w.kquantOf(wm)
		n += s.int8Bytes - s.nativeBytes
	}
	return n
}

// KQuantNative is "q5k" or "q6k" when backend's q4k lane uploads wm as the file's blocks (kquantLane), else "". wm is
// compared by address, so it must be the model's own weight (a pointer into Weights()).
func (m *Model) KQuantNative(backend string, wm *linalg.WeightMat) string {
	for _, x := range m.kquantLane(backend) {
		if x == wm {
			s, _ := m.w.kquantOf(wm)
			return s.kind
		}
	}
	return ""
}

// kquantOf is wm's recorded source, when it has one.
func (w *Weights) kquantOf(wm *linalg.WeightMat) (kquantSrc, bool) {
	q8, _, _, ok := wm.Int8()
	if !ok || len(q8) == 0 {
		return kquantSrc{}, false
	}
	s, ok := w.kquant[unsafe.SliceData(q8)]
	return s, ok
}

// KQuantReader serves the GGUF's own Q5_K and Q6_K blocks for the int8 matmuls of a q4k load that the file stores that way,
// so a GPU lane can upload them in place of the int8 copy. The CPU path keeps the int8 copy. The blocks it passes alias the
// file mapping, valid until Close.
type KQuantReader struct {
	g     *embed.GGUFFile
	w     *Weights
	owned bool // g was opened for this reader (Close unmaps it), not the load's mapping
}

// OpenKQuant returns a reader over m's Q5_K and Q6_K blocks: the load's own mapping while the model has not yet released it
// (through its resident build), else the file mapped again, which parses its header anew. It returns nil and no error when
// the model holds no such weight (a load at another quant, a .giw, a GGUF with no such tensor).
func (m *Model) OpenKQuant() (*KQuantReader, error) {
	if m.w == nil || len(m.w.kquant) == 0 || m.w.ggufPath == "" {
		return nil, nil
	}
	if m.w.kquantFile != nil {
		return &KQuantReader{g: m.w.kquantFile, w: m.w}, nil
	}
	g, err := OpenGGUFMmap(m.w.ggufPath)
	if err != nil {
		return nil, err
	}
	return &KQuantReader{g: g, w: m.w, owned: true}, nil
}

// releaseKQuantFile unmaps the load's GGUF once the resident build that reads it has run. A later OpenKQuant maps the
// file again.
func (m *Model) releaseKQuantFile() {
	if m.w != nil && m.w.kquantFile != nil {
		_ = m.w.kquantFile.Close()
		m.w.kquantFile = nil
	}
}

// Kind is "q5k" or "q6k" when wm is an int8 copy of a tensor the file stores that way, else "". A nil reader reports "".
func (k *KQuantReader) Kind(wm *linalg.WeightMat) string {
	if k == nil {
		return ""
	}
	s, _ := k.w.kquantOf(wm)
	return s.kind
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
	src, ok := k.w.kquantOf(wm)
	if !ok {
		return nil, fmt.Errorf("decoder: KQuantReader: weight has no Q5_K/Q6_K source")
	}
	get, bb := k.g.Q5KRaw, 176
	if src.kind == "q6k" {
		get, bb = k.g.Q6KRaw, 210
	}
	dims, raw, ok, err := get(src.tensor)
	if err != nil {
		return nil, err
	}
	if !kquantServes(dims, ok, wm.Cols(), src, wm.Rows()) {
		return nil, fmt.Errorf("decoder: KQuantReader: %q is not %s covering %d rows of %d columns (dims %v)", src.tensor, src.kind, wm.Rows(), wm.Cols(), dims)
	}
	rb := dims[0] / 256 * bb
	return func(r int) []byte { s := src.src(r); return raw[s*rb : (s+1)*rb] }, nil
}

// Close unmaps the file when the reader mapped it; the load's own mapping stays for releaseKQuantFile.
func (k *KQuantReader) Close() error {
	if k == nil || !k.owned {
		return nil
	}
	return k.g.Close()
}
