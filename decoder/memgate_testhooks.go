//go:build goinfer_testhooks

package decoder

import (
	"cmp"
	"reflect"
	"slices"
	"strings"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/mem"
)

// CPUHoldsForTest leases on l what the CPU backend holds for this model and cache c: it has no allocator to account, so a row reads the
// loaded weights and the cache by walking them (holdsWalker). An expert's weights lease on the experts line, the cache's K/V on kv, its recurrent
// state on state, the rest of the weights on dense, and scratch on scratch. A cache field on no list leases one byte on a line named for it,
// whatever it holds, which is an unknown line and red. The returned function releases every lease.
func (m *Model) CPUHoldsForTest(l *mem.Ledger, c *KVCache) (release func()) {
	h := &holdsWalker{l: l}
	w := reflect.ValueOf(m.w).Elem()
	weights := func(path string) mem.LineID {
		for _, f := range strings.Split(path, ".") {
			if f == "Experts" || f == "expertsGateUp" || f == "expertsDown" {
				return mem.LineExperts
			}
		}
		return mem.LineDense
	}
	for i := range w.NumField() {
		switch f := w.Type().Field(i); f.Name {
		case "Cfg", "arch", "backing", "st", "bakedQuant":
		default:
			h.root(weights, f.Name, w.Field(i))
		}
	}
	cv := reflect.ValueOf(c).Elem()
	for i := range cv.NumField() {
		name := cv.Type().Field(i).Name
		if cpuCacheIgnored[name] {
			continue
		}
		ln, ok := cpuCacheLines[name]
		if !ok { // whatever it holds on this path: a new field is red until a list names it
			h.unread("cache field " + name)
			continue
		}
		h.root(func(path string) mem.LineID {
			if strings.Contains(path, ".hq") || strings.HasSuffix(path, ".tq") || strings.HasSuffix(path, ".tsc") || strings.HasSuffix(path, ".th") {
				return mem.LineScratch // rewritten every step
			}
			return ln
		}, name, cv.Field(i))
	}
	return h.finish()
}

// holdsWalker is what CPUHoldsForTest reads a value with, as address ranges.
type holdsWalker struct {
	l      *mem.Ledger
	spans  []holdSpan
	leases []*mem.Lease
	seen   map[holdVisit]bool // per root, so a thing two lines name is seen by both and read as an overlap
}

type holdSpan struct {
	lo, hi uintptr
	line   mem.LineID
}

type holdVisit struct {
	addr uintptr
	typ  reflect.Type
}

func (h *holdsWalker) unread(what string) {
	h.leases = append(h.leases, h.l.Reserve(mem.HostAnon, mem.LineID("unread "+what), 1, 1))
}

// root walks one top-level field, whose lines line names by path.
func (h *holdsWalker) root(line func(path string) mem.LineID, path string, v reflect.Value) {
	h.seen = map[holdVisit]bool{}
	h.walk(line, path, v)
}

// walk records what v holds. A scalar, an inline array, a slice's backing array, a map's entries and a string's bytes lease on the line of the field
// they came from, down through pointers and inside slices' spare capacity. A word that only names other memory (a slice, pointer, map, string or
// interface header) leases on scratch, since no plan prices it. Padding is not visible, nor are a map's buckets. A func, chan or unsafe pointer
// the walk cannot read leases one byte on a line named for it, which is an unknown line and red.
func (h *holdsWalker) walk(line func(string) mem.LineID, path string, v reflect.Value) {
	inline := func(ln mem.LineID) { // v's own bytes, where v has an address: a map's values do not, and their entry is leased with the map
		if ln != "" && v.CanAddr() {
			lo := v.UnsafeAddr()
			h.spans = append(h.spans, holdSpan{lo, lo + v.Type().Size(), ln})
		}
	}
	switch v.Kind() {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32,
		reflect.Uint64, reflect.Uintptr, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
		inline(line(path))
	case reflect.Pointer:
		inline(mem.LineScratch)
		if key := (holdVisit{v.Pointer(), v.Type()}); !v.IsNil() && !h.seen[key] {
			h.seen[key] = true
			h.walk(line, path, v.Elem())
		}
	case reflect.Interface:
		inline(mem.LineScratch)
		if !v.IsNil() {
			h.walk(line, path, v.Elem())
		}
	case reflect.Map:
		inline(mem.LineScratch)
		if ln := line(path); v.Len() > 0 && ln != "" {
			n := int64(v.Len()) * int64(v.Type().Key().Size()+v.Type().Elem().Size())
			h.leases = append(h.leases, h.l.Reserve(mem.HostAnon, ln, n, n))
		}
		for it := v.MapRange(); it.Next(); {
			h.walk(line, path, it.Key())
			h.walk(line, path, it.Value())
		}
	case reflect.Array:
		if holdsNoWords(v.Type().Elem()) {
			inline(line(path))
			return
		}
		for i := range v.Len() {
			h.walk(line, path, v.Index(i))
		}
	case reflect.String:
		inline(mem.LineScratch)
		if ln := line(path); v.Len() > 0 && ln != "" {
			lo := uintptr(unsafe.Pointer(unsafe.StringData(v.String())))
			h.spans = append(h.spans, holdSpan{lo, lo + uintptr(v.Len()), ln})
		}
	case reflect.Slice:
		inline(mem.LineScratch)
		if v.IsNil() || v.Cap() == 0 {
			return
		}
		if holdsNoWords(v.Type().Elem()) {
			if ln := line(path); ln != "" {
				lo := v.Pointer()
				h.spans = append(h.spans, holdSpan{lo, lo + uintptr(v.Cap())*v.Type().Elem().Size(), ln})
			}
			return
		}
		all := v.Slice(0, v.Cap()) // spare capacity is held too
		for i := range all.Len() {
			h.walk(line, path, all.Index(i))
		}
	case reflect.Struct:
		t := v.Type()
		for i := range v.NumField() {
			h.walk(line, path+"."+t.Field(i).Name, v.Field(i))
		}
	case reflect.Func, reflect.Chan, reflect.UnsafePointer:
		inline(mem.LineScratch)
		if !v.IsNil() {
			h.unread(path)
		}
	}
}

// holdsNoWords reports whether a value of t is all scalars, so its whole size is one field's bytes.
func holdsNoWords(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32,
		reflect.Uint64, reflect.Uintptr, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
		return true
	case reflect.Array:
		return holdsNoWords(t.Elem())
	}
	return false
}

// finish merges the ranges into leases. Overlapping ranges count once: scratch yields to any other line, and two other lines read as an unknown
// line, red, rather than one line's bytes read as the other's. It returns the function that releases them.
func (h *holdsWalker) finish() (release func()) {
	spans := h.spans
	slices.SortFunc(spans, func(a, b holdSpan) int { return cmp.Compare(a.lo, b.lo) })
	for i := 0; i < len(spans); {
		s := spans[i]
		for i++; i < len(spans) && spans[i].lo < s.hi; i++ {
			s.hi = max(s.hi, spans[i].hi)
			switch next := spans[i].line; {
			case next == s.line || next == mem.LineScratch:
			case s.line == mem.LineScratch:
				s.line = next
			default:
				s.line = mem.LineID("overlap of " + string(s.line) + " and " + string(next))
			}
		}
		n := int64(s.hi - s.lo)
		h.leases = append(h.leases, h.l.Reserve(mem.HostAnon, s.line, n, n))
	}
	return func() {
		for _, ls := range h.leases {
			ls.Release()
		}
	}
}

// CPUScratchBudgetForTest is the per-stream scratch the CPU's fit allows: the figure a row holds the scratch line to.
const CPUScratchBudgetForTest = prefillAttnScratchBudget

// cpuCacheLines is the plan line of each KVCache field that holds bytes or bookkeeping.
var cpuCacheLines = map[string]mem.LineID{"keys": mem.LineKV, "vals": mem.LineKV, "keysQ": mem.LineKV, "valsQ": mem.LineKV, "keyScale": mem.LineKV, "valScale": mem.LineKV,
	"keysH": mem.LineKV, "valsH": mem.LineKV, "rings": mem.LineKV, "mlaLatent": mem.LineKV,
	"delta": mem.LineState, "conv": mem.LineState, "mamba": mem.LineState, "kda": mem.LineState,
	"scr": mem.LineScratch, "imgBlocks": mem.LineScratch, "mropePos": mem.LineScratch, "deepstack": mem.LineScratch, "stride": mem.LineScratch, "captureLayers": mem.LineScratch,
	"captured": mem.LineScratch, "subAttn": mem.LineScratch, "subMLP": mem.LineScratch, "subMLPpre": mem.LineScratch, "subCtx": mem.LineScratch, "treeRowPos": mem.LineScratch,
	"treeMask": mem.LineScratch, "numLayers": mem.LineScratch, "planRows": mem.LineScratch, "rowCeiling": mem.LineScratch, "regrownBytes": mem.LineScratch, "kvDim": mem.LineScratch,
	"window": mem.LineScratch, "mropeDelta": mem.LineScratch, "pos": mem.LineScratch, "localAny": mem.LineScratch, "quant": mem.LineScratch, "headDim": mem.LineScratch,
	"manualPos": mem.LineScratch, "recurrentResets": mem.LineScratch, "latentDim": mem.LineScratch, "subCapture": mem.LineScratch}

// cpuCacheIgnored is what KVCache points at and does not own: a request's counter, the adapter the model holds, and a predicate.
var cpuCacheIgnored = map[string]bool{"cost": true, "costPhase": true, "lora": true, "skipAppend": true}

// WeightMatScalarBytesForTest is what one weight matrix holds in its own scalar fields, beyond its arrays: read by the walker, on an empty matrix.
func WeightMatScalarBytesForTest() int64 {
	var (
		w linalg.WeightMat
		l mem.Ledger
	)
	h := &holdsWalker{l: &l}
	h.root(func(string) mem.LineID { return mem.LineDense }, "w", reflect.ValueOf(&w).Elem())
	defer h.finish()()
	return l.Usage().ByLine[mem.LineDense]
}
