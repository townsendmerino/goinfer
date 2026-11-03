//go:build goinfer_testhooks

package decoder

import (
	"cmp"
	"reflect"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/internal/mem"
)

// CPUHoldsForTest leases on l what the CPU backend holds for this model and cache c: it has no allocator to account, so a row reads the
// loaded weights and the cache by walking them, as the address ranges their slices' capacities reach, overlaps merged so an array aliased or
// viewed twice counts once. A range is seen only from the first address a slice can name, so a view into a longer array counts its tail. Experts
// lease on the experts line, the cache's K/V on kv, its recurrent state on state, the rest of the weights on dense. A cache field on neither
// list leases one byte on a line named for it, whatever it holds, which is an unknown line and red. The returned function releases every lease.
func (m *Model) CPUHoldsForTest(l *mem.Ledger, c *KVCache) (release func()) {
	type span struct {
		lo, hi uintptr
		line   mem.LineID
	}
	var spans []span
	var walk func(line func(path string) mem.LineID, path string, v reflect.Value)
	walk = func(line func(string) mem.LineID, path string, v reflect.Value) {
		switch v.Kind() {
		case reflect.Pointer:
			if !v.IsNil() {
				walk(line, path, v.Elem())
			}
		case reflect.Slice:
			switch v.Type().Elem().Kind() {
			case reflect.Struct, reflect.Pointer, reflect.Slice:
				for i := range v.Len() {
					walk(line, path, v.Index(i))
				}
			case reflect.Int8, reflect.Uint8, reflect.Uint16, reflect.Int16, reflect.Uint32, reflect.Int32, reflect.Float32, reflect.Float64, reflect.Uint64, reflect.Int64:
				if ln := line(path); ln != "" && !v.IsNil() && v.Cap() > 0 {
					lo := v.Pointer()
					spans = append(spans, span{lo, lo + uintptr(v.Cap())*v.Type().Elem().Size(), ln})
				}
			}
		case reflect.Struct:
			t := v.Type()
			for i := range v.NumField() {
				walk(line, path+"."+t.Field(i).Name, v.Field(i))
			}
		}
	}
	w := reflect.ValueOf(m.w).Elem()
	weights := func(path string) mem.LineID {
		if strings.Contains(strings.ToLower(path), ".experts") {
			return mem.LineExperts
		}
		return mem.LineDense
	}
	for i := range w.NumField() {
		switch f := w.Type().Field(i); f.Name {
		case "Cfg", "arch", "backing", "st", "bakedQuant":
		default:
			walk(weights, f.Name, w.Field(i))
		}
	}
	var leases []*mem.Lease
	cv := reflect.ValueOf(c).Elem()
	for i := range cv.NumField() {
		name := cv.Type().Field(i).Name
		if cpuCacheScratch[name] {
			continue
		}
		ln, ok := cpuCacheLines[name]
		if !ok { // whatever it holds on this path: a new field is red until a list names it
			leases = append(leases, l.Reserve(mem.HostAnon, mem.LineID("cache field "+name), 1, 1))
			continue
		}
		walk(func(path string) mem.LineID {
			if strings.Contains(path, ".hq") || strings.HasSuffix(path, ".tq") || strings.HasSuffix(path, ".tsc") || strings.HasSuffix(path, ".th") {
				return "" // per-step scratch, rewritten every step
			}
			return ln
		}, name, cv.Field(i))
	}
	slices.SortFunc(spans, func(a, b span) int { return cmp.Compare(a.lo, b.lo) })
	for i := 0; i < len(spans); {
		s := spans[i]
		for i++; i < len(spans) && spans[i].lo < s.hi; i++ {
			s.hi = max(s.hi, spans[i].hi)
			if spans[i].line != s.line { // one array on two plan lines: red, rather than one line's bytes read as the other's
				s.line = mem.LineID("overlap of " + string(s.line) + " and " + string(spans[i].line))
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

// cpuCacheLines is the plan line of each KVCache field that holds K/V or recurrent state.
var cpuCacheLines = map[string]mem.LineID{"keys": mem.LineKV, "vals": mem.LineKV, "keysQ": mem.LineKV, "valsQ": mem.LineKV, "keyScale": mem.LineKV, "valScale": mem.LineKV,
	"keysH": mem.LineKV, "valsH": mem.LineKV, "rings": mem.LineKV, "mlaLatent": mem.LineKV,
	"delta": mem.LineState, "conv": mem.LineState, "mamba": mem.LineState, "kda": mem.LineState}

// cpuCacheScratch is every other KVCache field: per-stream scratch, a seam's captures, and bookkeeping no plan could price.
var cpuCacheScratch = map[string]bool{"cost": true, "costPhase": true, "numLayers": true, "planRows": true, "rowCeiling": true, "regrownBytes": true, "kvDim": true, "window": true,
	"imgBlocks": true, "mropePos": true, "mropeDelta": true, "deepstack": true, "stride": true, "pos": true, "localAny": true, "skipAppend": true, "quant": true, "headDim": true,
	"manualPos": true, "recurrentResets": true, "latentDim": true, "scr": true, "captureLayers": true, "captured": true, "subCapture": true, "subAttn": true, "subMLP": true,
	"subMLPpre": true, "subCtx": true, "treeRowPos": true, "treeMask": true, "lora": true}
