//go:build goinfer_testhooks

package decoder

import (
	"cmp"
	"reflect"
	"slices"
	"strings"
	"unsafe"

	"github.com/townsendmerino/goinfer/internal/mem"
)

// CPUHoldsForTest leases on l what the CPU backend holds for this model and cache c: it has no allocator to account, so a row reads the
// loaded weights and the cache by walking them, as the address ranges their slices' capacities reach, overlaps merged so an array aliased or
// viewed twice counts once. A range is seen only from the first address a slice can name, so a view into a longer array counts its tail. An
// expert's weights lease on the experts line, the cache's K/V on kv, its recurrent state on state, the rest of the weights on dense; scratch,
// arrays of headers or structs, the structs pointers name and strings on scratch, which yields to any other line over the same bytes. A map
// leases its entries' size on its field's line (its buckets are not visible). A cache field on no list, or a func, chan or unsafe pointer the walk
// cannot read, leases one byte on a line named for it, which is an unknown line and red. The returned function releases every lease.
func (m *Model) CPUHoldsForTest(l *mem.Ledger, c *KVCache) (release func()) {
	type span struct {
		lo, hi uintptr
		line   mem.LineID
	}
	type visit struct {
		addr uintptr
		typ  reflect.Type
	}
	var spans []span
	var leases []*mem.Lease
	unread := func(what string) {
		leases = append(leases, l.Reserve(mem.HostAnon, mem.LineID("unread "+what), 1, 1))
	}
	var seen map[visit]bool // per top-level field, so a thing two lines name is seen by both and read as an overlap
	var walk func(line func(path string) mem.LineID, path string, v reflect.Value)
	walk = func(line func(string) mem.LineID, path string, v reflect.Value) {
		switch v.Kind() {
		case reflect.Pointer:
			if key := (visit{v.Pointer(), v.Type()}); !v.IsNil() && !seen[key] {
				seen[key] = true
				spans = append(spans, span{v.Pointer(), v.Pointer() + v.Type().Elem().Size(), mem.LineScratch})
				walk(line, path, v.Elem())
			}
		case reflect.Interface:
			if !v.IsNil() {
				walk(line, path, v.Elem())
			}
		case reflect.Map:
			if ln := line(path); v.Len() > 0 && ln != "" {
				n := int64(v.Len()) * int64(v.Type().Key().Size()+v.Type().Elem().Size())
				leases = append(leases, l.Reserve(mem.HostAnon, ln, n, n))
			}
			for it := v.MapRange(); it.Next(); {
				walk(line, path, it.Key())
				walk(line, path, it.Value())
			}
		case reflect.Array:
			for i := range v.Len() {
				walk(line, path, v.Index(i))
			}
		case reflect.String:
			if v.Len() > 0 {
				lo := uintptr(unsafe.Pointer(unsafe.StringData(v.String())))
				spans = append(spans, span{lo, lo + uintptr(v.Len()), mem.LineScratch})
			}
		case reflect.Slice:
			if v.IsNil() || v.Cap() == 0 {
				return
			}
			ln := line(path)
			switch v.Type().Elem().Kind() {
			case reflect.Pointer, reflect.Slice, reflect.Interface, reflect.Map, reflect.Struct, reflect.Array, reflect.String:
				ln = mem.LineScratch // the array of them: what they hold is walked below
				for i := range v.Len() {
					walk(line, path, v.Index(i))
				}
			}
			if ln != "" {
				lo := v.Pointer()
				spans = append(spans, span{lo, lo + uintptr(v.Cap())*v.Type().Elem().Size(), ln})
			}
		case reflect.Struct:
			t := v.Type()
			for i := range v.NumField() {
				walk(line, path+"."+t.Field(i).Name, v.Field(i))
			}
		case reflect.Func, reflect.Chan, reflect.UnsafePointer:
			if !v.IsNil() {
				unread(path)
			}
		}
	}
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
			seen = map[visit]bool{}
			walk(weights, f.Name, w.Field(i))
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
			unread("cache field " + name)
			continue
		}
		seen = map[visit]bool{}
		walk(func(path string) mem.LineID {
			if strings.Contains(path, ".hq") || strings.HasSuffix(path, ".tq") || strings.HasSuffix(path, ".tsc") || strings.HasSuffix(path, ".th") {
				return mem.LineScratch // rewritten every step
			}
			return ln
		}, name, cv.Field(i))
	}
	slices.SortFunc(spans, func(a, b span) int { return cmp.Compare(a.lo, b.lo) })
	for i := 0; i < len(spans); {
		s := spans[i]
		for i++; i < len(spans) && spans[i].lo < s.hi; i++ {
			s.hi = max(s.hi, spans[i].hi)
			switch next := spans[i].line; {
			case next == s.line || next == mem.LineScratch: // decode scratch can name a weight's array: the weight owns those bytes
			case s.line == mem.LineScratch:
				s.line = next
			default: // one array on two plan lines: red, rather than one line's bytes read as the other's
				s.line = mem.LineID("overlap of " + string(s.line) + " and " + string(next))
			}
		}
		n := int64(s.hi - s.lo)
		leases = append(leases, l.Reserve(mem.HostAnon, s.line, n, n))
	}
	return func() {
		for _, ls := range leases {
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
