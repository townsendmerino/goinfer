//go:build gpu

package kernels

import (
	"reflect"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/internal/cost"
)

// The decode runners' cost counters (SB11, internal/cost). The Context counts, always, the dispatches recorded into a decode
// runner's passes, the command buffers they submit and their blocking polls (Activity). A runner with Cost set attributes each
// submitted token's steps, one dispatch per step, to the step's pipeline, named by the Context field that holds it. Until OP4's
// walker names the operation a dispatch belongs to, a WebGPU operation is a pipeline. A dispatch's bytes and time are not
// counted: it is asynchronous.

// DispatchUnmeasured is what a dispatch's counts leave out.
var DispatchUnmeasured = []string{cost.FieldBytesRead, cost.FieldBytesWritten, cost.FieldTime}

// CostPlantForTest plants SB11's WebGPU defect: 1 records each buffer's first step twice.
var CostPlantForTest int

// Activity is the dispatches recorded, the command buffers submitted and the blocking polls made by this Context's decode
// runners so far.
func (c *Context) Activity() (dispatches, submits, waits int64) {
	return c.dispatches.Load(), c.submits.Load(), c.waits.Load()
}

// submit submits cmd and counts it.
func (c *Context) submit(cmd *wgpu.CommandBuffer) {
	c.queue.Submit(cmd)
	c.submits.Add(1)
}

// wait blocks until the queue's submitted work is done, and counts the wait.
func (c *Context) wait() {
	c.device.Poll(true, nil)
	c.waits.Add(1)
}

// Wait is wait, for a caller outside the package (SB11's extra-sync plant).
func (c *Context) Wait() { c.wait() }

// countSubmitted records the submitted token's dispatches, one per step, against each step's pipeline.
func (r *DecodeRunner) countSubmitted() {
	for _, s := range r.steps {
		name, ok := r.costNames[s.Pl]
		if !ok {
			name = r.c.pipelineName(s.Pl)
			if r.costNames == nil {
				r.costNames = map[*wgpu.ComputePipeline]string{}
			}
			r.costNames[s.Pl] = name
		}
		r.Cost.Add(r.CostPhase, name, cost.Counts{Calls: 1, Launches: 1, Unmeasured: DispatchUnmeasured})
	}
}

// pipelineName is the name of the Context field that holds p ("pipeline" when no field holds it). It reads every field by
// reflection, so it runs once per pipeline per runner, and only when a counter is attached.
func (c *Context) pipelineName(p *wgpu.ComputePipeline) string {
	v := reflect.ValueOf(c).Elem()
	for i := range v.NumField() {
		f := v.Field(i)
		if f.Kind() == reflect.Pointer && !f.IsNil() && f.Pointer() == reflect.ValueOf(p).Pointer() {
			return v.Type().Field(i).Name
		}
	}
	return "pipeline"
}
