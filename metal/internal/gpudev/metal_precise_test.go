//go:build darwin

package gpudev

import (
	"strings"
	"testing"

	"github.com/ebitengine/purego/objc"
)

const trivialMSL = `#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = float(i) / 127.0; }`

// TestCompileLibraryPrecise_mathVerified is the read-back guard for the precise-math
// demand: setPreciseMath must actually disable fast-math (both the modern MTLMathMode
// and the derived fastMathEnabled reflect it), CompileLibraryPrecise must compile with
// the verification passing on this OS, and CompileLibrary (non-precise) must stay on the
// fast-math default. On a future OS where the setter silently no-ops, setPreciseMath
// returns an error instead of a wrong-numerics library.
func TestCompileLibraryPrecise_mathVerified(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	t.Cleanup(func() { d.ReleaseObjects(); d.ReleaseAll() })

	opts := objc.ID(objc.GetClass("MTLCompileOptions")).Send(selAlloc).Send(selInit)
	defer opts.Send(selRelease)
	if err := setPreciseMath(opts); err != nil {
		t.Fatalf("setPreciseMath: %v", err)
	}
	if mm := objc.Send[uintptr](opts, selMathMode); mm != mtlMathModeSafe {
		t.Errorf("mathMode = %d after setPreciseMath, want %d (Safe)", mm, mtlMathModeSafe)
	}
	if fm := objc.Send[uintptr](opts, selFastMath) & 0xff; fm != 0 {
		t.Errorf("fastMathEnabled = %d after setPreciseMath, want 0 (NO)", fm)
	}

	if lib, err := d.CompileLibraryPrecise(trivialMSL, MSL3_1); err != nil || lib == 0 {
		t.Fatalf("CompileLibraryPrecise: lib=%d err=%v", lib, err)
	}

	if lib, err := d.CompileLibrary(trivialMSL, MSL3_1); err != nil || lib == 0 {
		t.Fatalf("CompileLibrary: lib=%d err=%v", lib, err)
	}
	def := objc.ID(objc.GetClass("MTLCompileOptions")).Send(selAlloc).Send(selInit)
	defer def.Send(selRelease)
	if fm := objc.Send[uintptr](def, selFastMath) & 0xff; fm == 0 {
		t.Errorf("default MTLCompileOptions fastMathEnabled = 0, expected fast-math ON — CompileLibrary's default may have changed")
	}
}

// TestSetPreciseMath_unverifiableIsError forces the future-OS case where NEITHER math
// API responds. setPreciseMath must return an error naming both selectors it tried —
// "I could not verify" is the exact silent-pass this guard exists to prevent. It uses
// the respondsToFn seam, so it needs no device and no Metal load (the default branch
// never sends to the id).
func TestSetPreciseMath_unverifiableIsError(t *testing.T) {
	orig := respondsToFn
	respondsToFn = func(objc.ID, objc.SEL) bool { return false } // no math API responds
	defer func() { respondsToFn = orig }()

	err := setPreciseMath(objc.ID(0))
	if err == nil {
		t.Fatal("setPreciseMath returned nil when no math API responds — an unverified 'precise' library was reported as success")
	}
	for _, want := range []string{"setMathMode:", "setFastMathEnabled:"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q must name the selector %q it tried", err, want)
		}
	}
}

// TestSetPreciseMath_fallbackPath exercises the deprecated setFastMathEnabled: fallback
// on real hardware by hiding the modern API from the selector check. It must still set
// AND verify fast-math off.
func TestSetPreciseMath_fallbackPath(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	t.Cleanup(func() { d.ReleaseObjects(); d.ReleaseAll() })

	orig := respondsToFn
	respondsToFn = func(_ objc.ID, sel objc.SEL) bool { return sel == selSetFastMath || sel == selFastMath }
	defer func() { respondsToFn = orig }()

	opts := objc.ID(objc.GetClass("MTLCompileOptions")).Send(selAlloc).Send(selInit)
	defer opts.Send(selRelease)
	if err := setPreciseMath(opts); err != nil {
		t.Fatalf("fallback setFastMathEnabled path returned error: %v", err)
	}
	if fm := objc.Send[uintptr](opts, selFastMath) & 0xff; fm != 0 {
		t.Errorf("fallback did not disable fast-math: fastMathEnabled = %d, want 0", fm)
	}
}
