//go:build darwin

package gpudev

import (
	"fmt"
	"strings"
	"testing"
)

// bindLimitSrc builds a kernel that binds exactly n buffers and writes the index-weighted sum
// 1*b0[0] + 2*b1[0] + ... + n*b(n-1)[0] into b0[0]. The weights make it order-sensitive, so a buffer bound at the wrong
// index, or not bound at all (the last one in particular), changes the answer. withTG adds a threadgroup-memory
// argument, for DispatchTG.
func bindLimitSrc(name string, n int, withTG bool) string {
	var args, sum []string
	for k := range n {
		if k == 0 {
			args = append(args, "device float* b0 [[buffer(0)]]")
		} else {
			args = append(args, fmt.Sprintf("device const float* b%d [[buffer(%d)]]", k, k))
		}
		sum = append(sum, fmt.Sprintf("%d.0f*b%d[0]", k+1, k))
	}
	body := "b0[0] = " + strings.Join(sum, " + ") + ";"
	if withTG {
		args = append(args, "threadgroup float* scratch [[threadgroup(0)]]")
		body = "scratch[0] = 0.0f; " + body
	}
	return fmt.Sprintf("kernel void %s(%s) { %s }\n", name, strings.Join(args, ", "), body)
}

// TestEncoderBindBuffers_limit pins the bind-count guard for goinfer audit C-N01: Encoder's scratch holds
// maxBindBuffers (31, Metal's per-kernel buffer-argument-table limit) entries. A dispatch binding exactly the limit
// must encode and run correctly through all three dispatch entry points, the last buffer included, and one more must
// panic with a message that gives the count, the limit and the remedy. (Before the guard the scratch was [16] and a
// 17th buffer panicked with Go's bare "index out of range [16] with length 16", which names neither; it was a bounds
// panic, not a memory overrun.)
//
// With the guard removed the test fails twice over: the scratch is [31], so a 32nd buffer still panics, but with Go's
// "index out of range [31] with length 31", which the message check below rejects. Device-only; not run in CI.
func TestEncoderBindBuffers_limit(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	t.Cleanup(func() { d.ReleaseObjects(); d.ReleaseAll() })

	const lim = maxBindBuffers
	if lim != 31 {
		t.Fatalf("maxBindBuffers = %d, want 31 (Apple's Metal feature set tables, buffer argument table entries per kernel function)", lim)
	}
	src := "#include <metal_stdlib>\nusing namespace metal;\n" +
		bindLimitSrc("sumw", lim, false) + bindLimitSrc("sumw_tg", lim, true)
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	plain, err := d.NewComputePipeline(lib, "sumw")
	if err != nil {
		t.Fatalf("pipeline sumw: %v", err)
	}
	withTG, err := d.NewComputePipeline(lib, "sumw_tg")
	if err != nil {
		t.Fatalf("pipeline sumw_tg: %v", err)
	}
	q := d.NewCommandQueue()

	// lim+1 one-float buffers holding 1, 2, ..., lim+1, so buffer k holds k+1.
	bufs := make([]Buffer, lim+1)
	for k := range bufs {
		bufs[k] = d.NewBufferLen(1)
	}
	reset := func() {
		for k := range bufs {
			bufs[k].Floats()[0] = float32(k + 1)
		}
	}
	// sum over k = 0..lim-1 of (k+1)*(k+1), exact in float32 (10416 for lim = 31).
	var want float32
	for k := range lim {
		want += float32((k + 1) * (k + 1))
	}

	entries := []struct {
		name     string
		pipe     Pipeline
		dispatch func(e *Encoder, p Pipeline, b []Buffer)
	}{
		{"Dispatch", plain, func(e *Encoder, p Pipeline, b []Buffer) { e.Dispatch(p, 1, 1, b...) }},
		{"Dispatch2D", plain, func(e *Encoder, p Pipeline, b []Buffer) { e.Dispatch2D(p, 1, 1, 1, 1, b...) }},
		{"DispatchTG", withTG, func(e *Encoder, p Pipeline, b []Buffer) { e.DispatchTG(p, 1, 1, 16, b...) }},
	}
	for _, en := range entries {
		t.Run(en.name+"/exactlyTheLimit", func(t *testing.T) {
			reset()
			e := q.Begin()
			en.dispatch(e, en.pipe, bufs[:lim])
			e.FinishEncoding()
			e.Commit()
			e.WaitDone()
			if err := e.Err(); err != nil {
				t.Fatalf("command buffer: %v", err)
			}
			e.DrainPool()
			if got := bufs[0].Floats()[0]; got != want {
				t.Fatalf("%d buffers bound: weighted sum = %v, want %v (a buffer, the last one included, was not bound at its index)", lim, got, want)
			}
		})
		t.Run(en.name+"/oneOverPanicsWithTheMessage", func(t *testing.T) {
			reset()
			e := q.Begin()
			var msg string
			func() {
				defer func() { msg = fmt.Sprint(recover()) }()
				en.dispatch(e, en.pipe, bufs[:lim+1])
				msg = "<no panic>"
			}()
			// The panic left the encoder open (the pipeline was set, nothing dispatched): close it and drain the pool, so
			// this goroutine's OS-thread lock is released whatever the assertions below say.
			e.FinishEncoding()
			e.Commit()
			e.WaitDone()
			e.DrainPool()
			for _, must := range []string{
				fmt.Sprintf("binds %d buffers", lim+1),
				fmt.Sprintf("limit of %d", lim),
				"must bind fewer buffers",
			} {
				if !strings.Contains(msg, must) {
					t.Fatalf("%d buffers: panic message %q does not contain %q", lim+1, msg, must)
				}
			}
			t.Logf("panic message: %s", msg)
		})
	}
}
