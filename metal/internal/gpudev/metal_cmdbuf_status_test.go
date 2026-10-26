//go:build darwin

package gpudev

import (
	"strings"
	"testing"

	"github.com/ebitengine/purego/objc"
)

const statusProbeSrc = `#include <metal_stdlib>
using namespace metal;
kernel void fill(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = float(i); }`

// TestCmdBufStatus_C09 is the device-level proof for goinfer audit C-09: waitUntilCompleted returns
// cleanly regardless of how a command buffer ends, so the host must read status/error explicitly or
// it trusts stale results after a fault. It proves the two objc paths Err depends on:
//
//  1. On a genuinely-completed buffer, status reads Completed (4) via objc.Send[uintptr] (the arm64
//     integer-return path) and Err() returns nil — the happy path must not false-positive.
//  2. cmdBufError formats a real NSError's localizedDescription into the Go error string — the abort
//     branch that surfaces the failure.
//
// Why (2) uses a synthetic NSError instead of a real GPU abort: this machine's GPU silently tolerates
// every abort trigger tried (OOB and 32-GiB-unmapped stores, 64-MiB threadgroup memory vs a 32-KiB
// limit, 4096 threads/threadgroup vs a 1024 max) — all report status Completed. That permissiveness
// is the very hazard C-09 guards against; it also means status Error can't be provoked here, so the
// formatting path is validated with an NSError built directly. Device-only; not run in CI.
func TestCmdBufStatus_C09(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	t.Cleanup(func() { d.ReleaseObjects(); d.ReleaseAll() })

	// (1) Happy path on a real command buffer.
	lib, err := d.CompileLibrary(statusProbeSrc, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	fill, err := d.NewComputePipeline(lib, "fill")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	q := d.NewCommandQueue()
	out := d.NewBufferLen(256)
	e := q.Begin()
	e.Dispatch(fill, 256, 64, out)
	e.FinishEncoding()
	e.Commit()
	e.WaitDone()
	if got := int(objc.Send[uintptr](e.cb, selStatus)); got != 4 {
		t.Fatalf("completed buffer: status = %d, want 4 (Completed) — objc integer-return path wrong", got)
	}
	if err := e.Err(); err != nil {
		t.Fatalf("completed buffer: Err() = %v, want nil", err)
	}
	e.DrainPool()

	// (2) Abort-branch formatting with a real NSError (NSError errorWithDomain:code:userInfo:).
	dom := nsString("MTLCommandBufferErrorDomain")
	nsErr := objc.ID(objc.GetClass("NSError")).Send(
		objc.RegisterName("errorWithDomain:code:userInfo:"), dom, uintptr(3), objc.ID(0))
	if nsErr == 0 {
		t.Fatal("could not build a synthetic NSError")
	}
	got := cmdBufError(nsErr)
	if got == nil || !strings.Contains(got.Error(), "aborted") {
		t.Fatalf("cmdBufError(nsErr) = %v, want a non-nil 'aborted' error", got)
	}
	// The message must include the NSError's localizedDescription (goString of an NSString), proving
	// the selError → selLocalizedDesc → goString chain works on a real objc error object.
	if desc := goString(nsErr.Send(selLocalizedDesc)); desc == "" || !strings.Contains(got.Error(), desc) {
		t.Fatalf("cmdBufError message %q missing localizedDescription %q", got, desc)
	}
	if nilErr := cmdBufError(0); nilErr == nil || !strings.Contains(nilErr.Error(), "aborted") {
		t.Fatalf("cmdBufError(0) = %v, want the generic 'aborted' error", nilErr)
	}
	t.Logf("C-09 status path OK: completed=4/nil; abort formats %q", got)
}
