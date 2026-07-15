//go:build cuda

package cuda

import "errors"

// errCUDANotWired is returned by the stub driver until the gocudrv-backed
// implementation + the compiled megakernel PTX are wired on the box (Phase 2).
var errCUDANotWired = errors.New("cuda: backend skeleton — driver not wired (see docs/cuda-megakernel-spec.md §8)")

// dim3 is a CUDA launch geometry (grid or block).
type dim3 struct{ X, Y, Z uint32 }

// Opaque device handles (mirror gocudrv's typed handles).
type (
	devPtr uintptr // a cuMemAlloc'd allocation
	module uintptr // a cuModuleLoadData'd PTX module
	fn     uintptr // a kernel handle within a module
)

// driver is the minimal cgo-free CUDA driver surface the resident decode path
// needs — Layer A of the spike, a 1:1 shim over github.com/eitamring/gocudrv
// (dlopen libcuda, CGO_ENABLED=0). gocudrv covers ALL of this except
// LaunchCooperative (cuLaunchCooperativeKernel is not in its 106 exported syms),
// which only the single-dispatch-per-layer megakernel needs; the 3-super-kernel
// path (spec §5.2) uses plain Launch. Timing is CUDA-event based per gocudrv's own
// 18× lesson (CPU clocks measure enqueue, not kernel work).
type driver interface {
	Init() error
	Alloc(bytes int) (devPtr, error)
	Free(devPtr) error
	MemcpyHtoD(dst devPtr, src []byte) error
	MemcpyDtoH(dst []byte, src devPtr) error
	LoadPTX(ptx []byte) (module, error)
	Func(m module, name string) (fn, error)
	Launch(f fn, grid, block dim3, sharedBytes int, args ...any) error
	LaunchCooperative(f fn, grid, block dim3, sharedBytes int, args ...any) error
	Synchronize() error
	Close() error
}

// stubDriver is the not-wired placeholder: every op reports errCUDANotWired so the
// skeleton compiles and links with no libcuda present, and BuildResident cleanly
// declines (ok=false). The box replaces this with a gocudrv-backed driver.
type stubDriver struct{}

func (stubDriver) Init() error                                        { return errCUDANotWired }
func (stubDriver) Alloc(int) (devPtr, error)                          { return 0, errCUDANotWired }
func (stubDriver) Free(devPtr) error                                  { return errCUDANotWired }
func (stubDriver) MemcpyHtoD(devPtr, []byte) error                    { return errCUDANotWired }
func (stubDriver) MemcpyDtoH([]byte, devPtr) error                    { return errCUDANotWired }
func (stubDriver) LoadPTX([]byte) (module, error)                     { return 0, errCUDANotWired }
func (stubDriver) Func(module, string) (fn, error)                    { return 0, errCUDANotWired }
func (stubDriver) Launch(fn, dim3, dim3, int, ...any) error           { return errCUDANotWired }
func (stubDriver) LaunchCooperative(fn, dim3, dim3, int, ...any) error { return errCUDANotWired }
func (stubDriver) Synchronize() error                                 { return errCUDANotWired }
func (stubDriver) Close() error                                       { return nil }
