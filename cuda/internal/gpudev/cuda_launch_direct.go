//go:build linux

package gpudev

import (
	"fmt"
	"reflect"
	"sync"
	"unsafe"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// maxDirectArgs bounds the parameter list LaunchDirect packs without allocating; a
// longer list takes Launch's path.
const maxDirectArgs = 32

// directArgs is the parameter storage of one LaunchDirect: each slot holds one
// argument's bytes and ptrs[i] points at slot i, the array cuLaunchKernel reads.
type directArgs struct {
	vals [maxDirectArgs]uint64
	ptrs [maxDirectArgs]unsafe.Pointer
}

var directArgsPool = sync.Pool{New: func() any {
	a := new(directArgs)
	for i := range a.ptrs {
		a.ptrs[i] = unsafe.Pointer(&a.vals[i])
	}
	return a
}}

// rawFunction reads f's CUfunction. gocudrv exports no accessor for it, so this
// reads the unexported field by reflection, and returns 0 (LaunchDirect then takes
// Launch's path) if the field is not there with the driver's handle type.
func rawFunction(f *gc.Function) cudasys.CUfunction {
	if f == nil {
		return 0
	}
	v := reflect.ValueOf(f).Elem().FieldByName("raw")
	if !v.IsValid() || v.Type() != reflect.TypeFor[cudasys.CUfunction]() {
		return 0
	}
	return cudasys.CUfunction(v.Uint())
}

// BindCurrentThread makes the device's context current on the calling OS thread,
// which LaunchDirect requires. The caller holds runtime.LockOSThread for as long as
// it launches directly; the binding is the thread's, not the goroutine's.
func (d *Device) BindCurrentThread() error {
	if d == nil || d.cx == nil {
		return fmt.Errorf("cuda: BindCurrentThread on a released device")
	}
	return cudaresult.CtxSetCurrent(d.cx.Driver(), d.cx.Raw())
}

// LaunchDirect is Launch issued from the caller's own thread: one cuLaunchKernel,
// instead of a hand-off to gocudrv's executor thread and back. The calling thread
// must be locked and bound (BindCurrentThread). The driver enforces that only on the
// null stream, where an unbound launch is refused and the error says so; a queue
// with its own stream launches in that stream's context. Stream order is program
// order, as for Launch, since every gocudrv call returns only after its work is
// enqueued.
//
// It takes none of gocudrv's locks: the caller keeps the pipeline's module, the
// queue's stream and every buffer argument open until the launch has completed,
// as a caller of Launch must already for the kernel to read them. A nil buffer
// argument is an error here, as it is for Launch.
func (q Queue) LaunchDirect(p Pipeline, cfg LaunchConfig, args ...KernelArg) error {
	if p.raw == 0 || len(args) > maxDirectArgs {
		return q.Launch(p, cfg, args...)
	}
	if !cfg.valid() {
		return fmt.Errorf("cuda: invalid launch geometry (grid %dx%dx%d block %dx%dx%d)",
			cfg.GridX, cfg.GridY, cfg.GridZ, cfg.BlockX, cfg.BlockY, cfg.BlockZ)
	}
	if q.d != nil {
		q.d.launches.Add(1)
	}
	if q.d == nil || q.d.cx == nil {
		return fmt.Errorf("cuda: LaunchDirect on a released device")
	}
	a := directArgsPool.Get().(*directArgs)
	for i, k := range args {
		if k.n == 0 {
			directArgsPool.Put(a)
			return fmt.Errorf("cuda: kernel arg %d is uninitialized or a nil buffer", i)
		}
		a.vals[i] = k.raw
	}
	var stream cudasys.CUstream
	if q.s != nil {
		stream = q.s.Raw()
	}
	var params *unsafe.Pointer
	if len(args) > 0 {
		params = &a.ptrs[0]
	}
	err := cudaresult.LaunchKernel(q.d.cx.Driver(), p.raw, cfg.GridX, cfg.GridY, cfg.GridZ,
		cfg.BlockX, cfg.BlockY, cfg.BlockZ, cfg.SharedMemBytes, stream, params)
	directArgsPool.Put(a)
	if err != nil {
		return fmt.Errorf("cuda: direct launch (the calling thread must be locked and bound with BindCurrentThread): %w", err)
	}
	return nil
}
