//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/townsendmerino/goinfer/internal/cost"
	"github.com/townsendmerino/goinfer/internal/mem"
)

// The resident's cost counters (SB11, internal/cost). gpudev's Device counts, always, the dispatches in each command buffer
// committed, the buffers and the host's waits on them (Activity). For one counted token the resident attaches a dispatch
// hook, and each dispatch in a buffer the token commits is attributed to its pipeline, named by the resident field that
// holds it, when the buffer was encoded while a hook was set. The executor re-encodes a buffer it encoded ahead before the
// counter was attached, so the first counted token is attributed in full; a buffer any other path commits that was encoded
// with no counter attached is counted and not attributed: dispatch_other. Until OP4's walker names the operation a dispatch
// belongs to, a Metal operation is a pipeline. A dispatch's bytes and time are not counted: it is asynchronous. Only
// goinfer_testhooks builds count, until a production caller reports the counts (SB11b).

// costNames caches pipelineNames per resident. Only goinfer_testhooks builds reach it, and its entries live as long as the
// process, which holds the few residents a test counts.
var costNames sync.Map // *resident -> map[Pipeline]string

// dispatchUnmeasured is what a dispatch's counts leave out.
var dispatchUnmeasured = []string{cost.FieldBytesRead, cost.FieldBytesWritten, cost.FieldTime}

// costToken runs one token's work with c counting into phase p, and adds the token's device-level counts: its host waits
// (sync), its command buffers (submit), the leases it took on the device's ledger, whether or not it released them (alloc),
// any committed dispatch the hook did not see (dispatch_other) and its wall time (the phase's WallNs).
func (r *resident) costToken(c *cost.Counter, p cost.Phase, token func() error) error {
	names := r.pipelineNames()
	var attributed atomic.Int64
	r.d.SetDispatchHook(func(pl Pipeline) {
		attributed.Add(1)
		name, ok := names[pl]
		if !ok {
			name = "pipeline"
		}
		c.Add(p, name, cost.Counts{Calls: 1, Launches: 1, Unmeasured: dispatchUnmeasured})
	})
	defer r.d.SetDispatchHook(nil)
	led, _ := r.d.Leaser().(*deviceLedger)
	var u0 mem.Usage
	if led != nil {
		u0 = led.Usage()
	}
	d0, s0, w0 := r.d.Activity()
	t0 := time.Now()
	err := token()
	if costPlantForTest == 1 {
		r.q.Begin().End()
	}
	wall := time.Since(t0)
	d1, s1, w1 := r.d.Activity()
	c.Tokens(p, 1)
	c.Wall(p, int64(wall))
	c.Add(p, "sync", cost.Counts{Syncs: w1 - w0, Unmeasured: dispatchUnmeasured})
	c.Add(p, "submit", cost.Counts{Submits: s1 - s0, Unmeasured: dispatchUnmeasured})
	if led != nil {
		u1 := led.Usage()
		c.Add(p, "alloc", cost.Counts{Allocs: int64(u1.Reserved - u0.Reserved), AllocBytes: u1.ReservedRequested - u0.ReservedRequested,
			Unmeasured: dispatchUnmeasured})
	} else {
		c.Add(p, "alloc", cost.Counts{Unmeasured: append([]string{cost.FieldAllocs, cost.FieldAllocBytes}, dispatchUnmeasured...)})
	}
	if other := (d1 - d0) - attributed.Load(); other != 0 {
		c.Add(p, "dispatch_other", cost.Counts{Calls: other, Launches: other, Unmeasured: dispatchUnmeasured})
	}
	return err
}

// pipelineNames maps each pipeline the resident holds in a field, or in an array field, to that field's name (the first such
// field, in declaration order). It reads the fields by reflection, so it runs once per resident, and only when a counter is
// attached.
func (r *resident) pipelineNames() map[Pipeline]string {
	if names, ok := costNames.Load(r); ok {
		return names.(map[Pipeline]string)
	}
	names := map[Pipeline]string{}
	pt := reflect.TypeFor[Pipeline]()
	read := func(f reflect.Value) Pipeline {
		return *(*Pipeline)(unsafe.Pointer(f.UnsafeAddr()))
	}
	add := func(pl Pipeline, name string) {
		if _, seen := names[pl]; !seen && pl != (Pipeline{}) {
			names[pl] = name
		}
	}
	v := reflect.ValueOf(r).Elem()
	for i := range v.NumField() {
		f, name := v.Field(i), v.Type().Field(i).Name
		switch {
		case f.Type() == pt:
			add(read(f), name)
		case f.Kind() == reflect.Array && f.Type().Elem() == pt:
			for j := range f.Len() {
				add(read(f.Index(j)), fmt.Sprintf("%s[%d]", name, j))
			}
		}
	}
	costNames.Store(r, names)
	return names
}
