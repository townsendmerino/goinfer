//go:build cuda

package cuda

import (
	"reflect"
	"unsafe"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// weightAlign is the byte alignment of each weight inside the arena: cuMemAlloc's own, so a kernel reads a sub-allocated weight
// at the alignment the buffer it replaces had.
const weightAlign = 256

func weightAligned(n int64) int64 { return (n + weightAlign - 1) / weightAlign * weightAlign }

// weightArena is the one device allocation uploadLayers sub-allocates the dense weights from: every buffer the plan's walk counts
// outside the expert stacks (walkWeightBuffers), so the driver rounds the weights once instead of once per buffer.
type weightArena struct {
	buf        Buffer
	used, size int64
	missed     int64 // bytes of dense-line weight buffers the arena could not hold, each allocated on its own
}

// weightArenaStats is what a build's arena held, kept on the resident after the upload: used equals size and missed is 0 when the
// walk that sized it counted exactly the buffers uploadLayers made.
type weightArenaStats struct{ used, size, missed int64 }

// openWeightArena allocates the arena for the planned weights v, or none when they need no bytes. The weights' own bytes lease on the
// dense line and the padding and rounding on the slack line, where the plan prices them (packedAllocSlack).
func (r *cudaResident) openWeightArena(v any) {
	n, inside := weightArenaBytes(v)
	if n == 0 {
		return
	}
	allocSeen(int(n))
	buf := r.dev.MustBufSplitOn(string(mem.LineDense), int(n), inside, string(mem.LineSlack), "weight arena")
	r.wArena = &weightArena{buf: buf, size: n}
}

// closeWeightArena ends sub-allocation and keeps the arena's figures; the arena's memory lives until the device is released.
func (r *cudaResident) closeWeightArena() {
	if a := r.wArena; a != nil {
		r.wArenaStats = weightArenaStats{used: a.used, size: a.size, missed: a.missed}
	}
	r.wArena = nil
}

// wbuf allocates a weight buffer of n elements of T: from the arena while one is open and the line is the dense line, else a device
// allocation of its own. A dense-line buffer the arena cannot hold is allocated on its own and counted in missed.
func wbuf[T gpu.Scalar](r *cudaResident, n int) Buffer {
	var z T
	nb := int64(n) * int64(unsafe.Sizeof(z))
	if a := r.wArena; a != nil && r.line == mem.LineDense && nb > 0 {
		if a.used+nb <= a.size {
			b := a.buf.Sub(int(a.used), int(nb), n)
			a.used += weightAligned(nb)
			return b
		}
		a.missed += nb
	}
	allocSeen(int(nb))
	return gpu.NewBufferLenOfOn[T](r.dev, r.line, n)
}

// walkWeightBuffers calls emit with the size of every device buffer the upload makes for v, in struct order: each hostW as the
// buffers upW allocates for it (wpk, then ws, ws16 and nBytes) and each plain numeric vector as the one buffer up32 makes of it.
// own is true under an hlayer's expert stacks, which upload on the experts line, outside the arena. Walking by reflection, not
// by field name, keeps a family's extra projections (DeltaNet, MLA, Mamba) counted without this function learning about them.
func walkWeightBuffers(v any, emit func(n int64, own bool)) {
	var walk func(rv reflect.Value, own bool)
	walk = func(rv reflect.Value, own bool) {
		switch rv.Kind() {
		case reflect.Pointer, reflect.Interface:
			if !rv.IsNil() {
				walk(rv.Elem(), own)
			}
		case reflect.Slice, reflect.Array:
			switch rv.Type().Elem().Kind() {
			case reflect.Struct, reflect.Pointer, reflect.Interface, reflect.Slice, reflect.Array:
				for i := 0; i < rv.Len(); i++ {
					walk(rv.Index(i), own)
				}
			case reflect.Float32, reflect.Int32, reflect.Uint32: // a vector of numbers (biases, norms) holds no hostW: it is one buffer
				emit(int64(rv.Len())*4, own)
			}
		case reflect.Struct:
			if rv.Type() == hostWType {
				elems := func(packed, planned string) int64 { // a planned hostW carries its counts, a packed one its slices
					return max(int64(rv.FieldByName(packed).Len()), rv.FieldByName(planned).Int())
				}
				emit(elems("wpk", "nWpk")*4, own)
				emit(elems("ws", "nWs")*4, own)
				emit(elems("ws16", "nWs16")*2, own)
				emit(rv.FieldByName("nBytes").Int(), own)
				return
			}
			isLayer := rv.Type() == hlayerType
			for sf, field := range rv.Fields() {
				walk(field, own || (isLayer && (sf.Name == "expGU" || sf.Name == "expDown")))
			}
		}
	}
	walk(reflect.ValueOf(v), false)
}

var hlayerType = reflect.TypeFor[hlayer]()

// weightArenaBytes is the arena the planned weights v take, every buffer outside the expert stacks at weightAlign, and the bytes
// those buffers request.
func weightArenaBytes(v any) (n, inside int64) {
	walkWeightBuffers(v, func(b int64, own bool) {
		if !own && b > 0 {
			n += weightAligned(b)
			inside += b
		}
	})
	return n, inside
}
