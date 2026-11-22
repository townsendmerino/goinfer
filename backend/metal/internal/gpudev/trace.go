//go:build darwin

package gpudev

// TraceOp is the kind of encoder event TraceForTest sees.
type TraceOp int

const (
	TraceDispatch TraceOp = iota // one dispatch: Pipe, Grid, TG, TGBytes, Bufs
	TraceBegin                   // a command buffer and encoder began (Begin, BeginNP)
	TraceCommit                  // a command buffer was committed (Commit, End)
	TraceWait                    // the host waited on one (WaitDone, End)
	TraceBoundary                // EventBoundary split the encoder around a shared-event signal and wait
)

// TraceEvent is one encoder event as TraceForTest sees it. A dispatch's Grid is in threads for Dispatch and DispatchTG
// (dispatchThreads) and in threadgroups for Dispatch2D (dispatchThreadgroups), which TwoD says; Bufs is a copy of the
// bindings, in index order.
type TraceEvent struct {
	Op       TraceOp
	Pipe     Pipeline
	TwoD     bool
	Grid, TG [2]int
	TGBytes  int
	Bufs     []Buffer
}

// TraceForTest sees, when set, every event of every encoder, in issue order, from the goroutine that issued it. Nil in
// production, where each site costs one nil check; the bindings are copied only inside it, so the variadic slices of the
// Dispatch calls do not escape.
var TraceForTest func(TraceEvent)

// TraceIdentity is the MTLBuffer b binds and its byte offset, for a trace to name the binding.
func (b Buffer) TraceIdentity() (id, off uintptr) { return uintptr(b.id), b.off }

func traceOp(op TraceOp) {
	if t := TraceForTest; t != nil {
		t(TraceEvent{Op: op})
	}
}

func traceDispatch(p Pipeline, twoD bool, gx, gy, tgx, tgy, tgBytes int, bufs []Buffer) {
	if t := TraceForTest; t != nil {
		t(TraceEvent{Op: TraceDispatch, Pipe: p, TwoD: twoD, Grid: [2]int{gx, gy}, TG: [2]int{tgx, tgy}, TGBytes: tgBytes,
			Bufs: append([]Buffer(nil), bufs...)})
	}
}
