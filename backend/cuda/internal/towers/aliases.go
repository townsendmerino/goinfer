//go:build cuda

package towers

import (
	"fmt"
	"runtime"
	"runtime/debug"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
)

// The device types and launch helpers the CUDA towers use, under the names the cuda package gives them.
type (
	Device       = gpu.Device
	Buffer       = gpu.Buffer
	Queue        = gpu.Queue
	Pipeline     = gpu.Pipeline
	Encoder      = gpu.Encoder
	LaunchConfig = gpu.LaunchConfig
	KernelArg    = gpu.KernelArg
)

var (
	Arg     = gpu.Arg
	ArgNull = gpu.ArgNull
)

// PTX holds the precompiled PTX modules the CUDA towers load. They stay embedded in backend/cuda/kernels.go so
// TestPTX_matchesSourcesAndBindings checks every .cu/.ptx pair in one place.
type PTX struct {
	TowerBase      []byte
	AttnFusedVit   []byte
	AttnImgPrefill []byte
	AudioEncoder   []byte
	GeluQuant      []byte
	GemvW8Batched  []byte
	LayernormQuant []byte
	PrefillBatched []byte
}

// visWQ is a device int8 projection weight for VisionEncoder.
type visWQ struct {
	kind string
	W    Buffer
	ws   Buffer
	N, K int
}

// runJob runs j on a pinned executor goroutine, turning a panic into an error (or *executorBug for a runtime.Error so
// ackJob re-panics it on the calling goroutine with the executor's stack).
func runJob(j func() error) (err error) {
	defer func() {
		if p := recover(); p != nil {
			stack := debug.Stack()
			switch v := p.(type) {
			case runtime.Error:
				err = &executorBug{bug: v, stack: stack}
			case error:
				err = fmt.Errorf("cuda: executor job panicked: %w\n%s", v, stack)
			default:
				err = fmt.Errorf("cuda: executor job panicked: %v\n%s", p, stack)
			}
		}
	}()
	return j()
}

type executorBug struct {
	bug   runtime.Error
	stack []byte
}

func (b *executorBug) Error() string {
	return fmt.Sprintf("cuda: executor job panicked: %s\nexecutor goroutine stack:\n%s", b.bug.Error(), b.stack)
}

func (b *executorBug) RuntimeError() {}

func (b *executorBug) Unwrap() error { return b.bug }

func ackJob(err error) error {
	if b, ok := err.(*executorBug); ok {
		panic(b)
	}
	return err
}
