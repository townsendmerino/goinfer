package arch

import (
	"unsafe"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
)

// NoteKQuant records wm's source when quant is q4k, the load retains its layers (no .giw sink) and tensor name is stored as
// Q5_K or Q6_K: wm is then an int8 copy whose file blocks a GPU lane can multiply natively (KQuantReader). rowSrc(r) is
// destination row r's element offset in the tensor, as q4kMat takes it. A source kquantServes rejects is not recorded. The
// record is metadata only; nothing is copied.
func (w *Weights) NoteKQuant(g *embed.GGUFFile, name string, wm linalg.WeightMat, in int, rowSrc func(r int) int) {
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
	start, perm := archparts.KQuantRows(wm.Rows(), in, rowSrc)
	src := archparts.KQuantSrc{Tensor: name, Kind: kind, Start: start, Perm: perm}
	if !archparts.KQuantServes(dims, true, wm.Cols(), src, wm.Rows()) {
		return
	}
	src.Int8Bytes, src.NativeBytes = int64(len(q8))+4*int64(len(sc)), int64(wm.Rows())*int64(in/256)*bb
	archparts.KQuantMu.Lock()
	defer archparts.KQuantMu.Unlock()
	if w.KQuant == nil {
		w.KQuant = map[*int8]archparts.KQuantSrc{}
	}
	w.KQuant[unsafe.SliceData(q8)] = src
}

// KQuantOf is wm's recorded source, when it has one.
func (w *Weights) KQuantOf(wm *linalg.WeightMat) (archparts.KQuantSrc, bool) {
	q8, _, _, ok := wm.Int8()
	if !ok || len(q8) == 0 {
		return archparts.KQuantSrc{}, false
	}
	s, ok := w.KQuant[unsafe.SliceData(q8)]
	return s, ok
}
