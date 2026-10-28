//go:build darwin

package gpudev

import "sync/atomic"

// activity is a Device's own count of what its queues ran (SB11's cost counters, internal/cost): the dispatches encoded, the
// command buffers committed and the host's waits on them. It is always on, one atomic add each, and it is the independent
// count a backend's per-pipeline attribution is checked against. hook, when set, is told each dispatch's pipeline.
type activity struct {
	dispatches, submits, waits atomic.Int64
	hook                       atomic.Pointer[func(Pipeline)]
}

// DispatchTwiceForTest plants SB11's unattributed dispatch: when set, the next 1-D dispatch on an encoder with a hook is
// encoded a second time, counted by the device and not reported to the hook. It clears itself.
var DispatchTwiceForTest atomic.Bool

// Activity is the dispatches encoded, the command buffers committed and the host's waits on this device's queues so far.
func (d *Device) Activity() (dispatches, submits, waits int64) {
	return d.act.dispatches.Load(), d.act.submits.Load(), d.act.waits.Load()
}

// SetDispatchHook sets the function told each dispatch's pipeline, for every encoder begun from now on and the Queue.Run*
// helpers; nil clears it. An encoder keeps the hook it was begun with, so a buffer encoded before the hook was set reports
// nothing, though the device counts its dispatches.
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

func (a *activity) submitted() {
	if a != nil {
		a.submits.Add(1)
	}
}

func (a *activity) waited() {
	if a != nil {
		a.waits.Add(1)
	}
}

// ran counts n dispatches of p in one command buffer that was committed and waited on (the Queue.Run* helpers).
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

// counted counts one dispatch of p encoded into e, and reports it to e's hook.
func (e *Encoder) counted(p Pipeline) {
	if e.act != nil {
		e.act.dispatches.Add(1)
	}
	if e.hook != nil {
		e.hook(p)
	}
}
