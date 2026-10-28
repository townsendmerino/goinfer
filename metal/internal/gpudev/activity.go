//go:build darwin

package gpudev

import "sync/atomic"

// activity is a Device's own count of what its queues ran (SB11's cost counters, internal/cost): the dispatches in each
// command buffer committed, the command buffers and the host's waits on them. It is always on: an encoder counts its
// dispatches in a plain field, and a commit adds them with one atomic add. It is the count a backend's per-pipeline
// attribution is checked against. hook, when set, is told the pipeline of each dispatch in a buffer committed while it is
// set, for a buffer whose encoder began while a hook was set; the other buffers' dispatches are counted and not attributed.
type activity struct {
	dispatches, submits, waits atomic.Int64
	hook                       atomic.Pointer[func(Pipeline)]
}

// Activity is the dispatches in the command buffers committed, those buffers and the host's waits on this device's queues
// so far.
func (d *Device) Activity() (dispatches, submits, waits int64) {
	return d.act.dispatches.Load(), d.act.submits.Load(), d.act.waits.Load()
}

// SetDispatchHook sets the function told each committed dispatch's pipeline (see activity); nil clears it.
func (d *Device) SetDispatchHook(h func(Pipeline)) {
	if h == nil {
		d.act.hook.Store(nil)
		return
	}
	d.act.hook.Store(&h)
}

func (a *activity) dispatchHook() func(Pipeline) {
	if a == nil {
		return nil
	}
	if h := a.hook.Load(); h != nil {
		return *h
	}
	return nil
}

// committed counts e's command buffer and its dispatches, and reports the tracked ones to the hook set now.
func (a *activity) committed(e *Encoder) {
	if a == nil {
		return
	}
	a.submits.Add(1)
	a.dispatches.Add(int64(e.n))
	if h := a.dispatchHook(); h != nil {
		for _, p := range e.pipes {
			h(p)
		}
	}
}

func (a *activity) waited() {
	if a != nil {
		a.waits.Add(1)
	}
}

// ran counts n dispatches of p in one command buffer that was encoded, committed and waited on in one call (the Queue.Run*
// helpers).
func (a *activity) ran(p Pipeline, n int) {
	if a == nil {
		return
	}
	a.dispatches.Add(int64(n))
	a.submits.Add(1)
	a.waits.Add(1)
	if h := a.dispatchHook(); h != nil {
		for range n {
			h(p)
		}
	}
}

// counted counts one dispatch of p encoded into e, keeping its pipeline when e is tracked.
func (e *Encoder) counted(p Pipeline) {
	e.n++
	if e.tracked {
		e.pipes = append(e.pipes, p)
	}
}
