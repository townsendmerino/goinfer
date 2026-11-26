//go:build cuda

package cuda

import (
	"fmt"
	"runtime"
	"runtime/debug"

	"github.com/townsendmerino/goinfer/decoder"
)

func (r *cudaResident) do(j func() error) error {
	if r.reqCh == nil {
		return decoder.ErrClosed
	}
	r.reqCh <- j
	return ackJob(<-r.ackCh)
}

// doOn is do with the allocation helpers on plan line line for the job: the line is set on the executor, where they read it.
func (r *cudaResident) doOn(line string, j func() error) error {
	return r.do(func() error {
		defer r.onLine(line)()
		return j()
	})
}

// runJob is the executor goroutine's PANIC BOUNDARY: it runs one job and converts a panic into an ordinary error,
// so the pinned thread survives and `do` returns to its caller. It has to live here and not at a call site: every
// job runs on the executor goroutine but `do` blocks on a different one, so a `defer recover()` in BuildResident
// or any caller cannot catch a panic raised inside j(). The case is reachable by design: the allocator PANICS on
// allocation failure per its contract, and prefillCore allocates M*(2*inter+2*hidden+...) floats, hundreds of MB on
// a long prompt; without this a long prompt against a nearly-full card killed the serve process at the one seam
// whose job is to decline to the sequential path.
//
// A runtime.Error (a bug, not a device failure) comes back as an *executorBug, which ackJob raises again on the
// calling goroutine: a bug must fail loudly there, not become a decline to a slower path. Any other panic value is
// wrapped with %w when it is an error, so mem.DeviceAllocFailed still sees the allocator's failure type.
//
// The recovered error is deliberately NOT wrapped in errPrefillDeclined here: `do` is shared by every job (setup,
// decode, prefill), and only the prefill caller knows a decline is the right response. prefillCore wraps it at its
// own boundary.
func runJob(j func() error) (err error) {
	defer func() {
		if p := recover(); p != nil {
			// The STACK, not just the value: a recovered executor panic becomes a resident DECLINE printed once to stderr, and
			// an allocation failure with no frame is unactionable. The stack is a few KB on a path that runs at most once per
			// model load.
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

// executorBug is a runtime.Error recovered on an executor goroutine, with that goroutine's stack. It is itself a runtime.Error,
// so every recover the re-raised panic meets on the calling side (mem.RepanicBug) lets it through too.
type executorBug struct {
	bug   runtime.Error
	stack []byte
}

func (b *executorBug) Error() string {
	return fmt.Sprintf("cuda: executor job panicked: %s\nexecutor goroutine stack:\n%s", b.bug.Error(), b.stack)
}

func (b *executorBug) RuntimeError() {}

func (b *executorBug) Unwrap() error { return b.bug }

// ackJob is a job's result as `do` returns it: an executorBug panics on the calling goroutine, anything else is returned.
func ackJob(err error) error {
	if b, ok := err.(*executorBug); ok {
		panic(b)
	}
	return err
}
