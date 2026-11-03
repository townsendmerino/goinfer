//go:build goinfer_testhooks

package decoder

import (
	"reflect"
	"strings"
	"unsafe"

	"github.com/townsendmerino/goinfer/internal/mem"
)

// CPUHoldsForTest leases on l what the CPU backend holds for this model and cache c: it has no allocator to account, so the row reads the
// loaded weights and the cache by walking them, each array once however many fields alias it. Experts lease on the experts line, the cache's
// K/V on kv, its recurrent state on state, and everything else on dense. A hybrid's per-step scratch and the cache's bookkeeping are not held
// bytes the plan could price, so they are left out. The returned function releases every lease.
func (m *Model) CPUHoldsForTest(l *mem.Ledger, c *KVCache) (release func()) {
	seen := map[unsafe.Pointer]bool{}
	var leases []*mem.Lease
	hold := func(line mem.LineID, v reflect.Value) {
		if line == "" || v.IsNil() || v.Cap() == 0 || seen[v.UnsafePointer()] {
			return
		}
		seen[v.UnsafePointer()] = true
		// Only the K/V grows past what it holds, by append. A weight or a state row can be a view into a longer array, and its capacity
		// would count its neighbours.
		size := v.Len()
		if line == mem.LineKV {
			size = v.Cap()
		}
		n := int64(size) * int64(v.Type().Elem().Size())
		leases = append(leases, l.Reserve(mem.HostAnon, line, n, n))
	}
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
				hold(line(path), v)
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
		if p := strings.ToLower(path); strings.Contains(p, ".experts") {
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
	cv := reflect.ValueOf(c).Elem()
	for name, line := range map[string]mem.LineID{"keys": mem.LineKV, "vals": mem.LineKV, "keysQ": mem.LineKV, "valsQ": mem.LineKV, "keysH": mem.LineKV, "valsH": mem.LineKV,
		"rings": mem.LineKV, "mlaLatent": mem.LineKV, "delta": mem.LineState, "conv": mem.LineState, "mamba": mem.LineState, "kda": mem.LineState} {
		fixed := func(path string) mem.LineID {
			if strings.Contains(path, ".hq") {
				return "" // scratch: rewritten every step
			}
			return line
		}
		walk(fixed, name, cv.FieldByName(name))
	}
	return func() {
		for _, ls := range leases {
			ls.Release()
		}
	}
}
