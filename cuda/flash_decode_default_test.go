//go:build cuda

package cuda

import (
	"io"
	"os"
	"strings"
	"testing"
)

// unsetenv makes key genuinely absent for the rest of the test (t.Setenv only ever sets a value). Call it after a t.Setenv on the
// same key so the original value is restored at cleanup.
func unsetenv(t *testing.T, key string) {
	t.Helper()
	if err := os.Unsetenv(key); err != nil {
		t.Fatalf("unsetenv %s: %v", key, err)
	}
}

// TestFlashDecodeSplit_default pins how GOINFER_CUDA_FLASH_DECODE resolves now that the lane is default ON (R6, 2026-09-23):
// unset or empty is the registered S, an explicit 0/off/false is the exact path, a positive integer picks S, and anything that
// is not a positive integer is OFF — a typo must never enable a non-exact attention path at some other S.
func TestFlashDecodeSplit_default(t *testing.T) {
	cases := []struct {
		name  string
		set   bool
		value string
		want  int
	}{
		{"unset is the registered S", false, "", flashDecodeDefaultSplit},
		{"empty is the registered S", true, "", flashDecodeDefaultSplit},
		{"0 is the exact path", true, "0", 0},
		{"off is the exact path", true, "off", 0},
		{"OFF is the exact path", true, "OFF", 0},
		{"false is the exact path", true, "false", 0},
		{"a positive integer picks S", true, "8", 8},
		{"1 is a real S (the S=1 identity check)", true, "1", 1},
		{"a negative number is off", true, "-4", 0},
		{"a typo is off, not the default", true, "sixteen", 0},
		{"a float is off", true, "16.5", 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if c.set {
				t.Setenv("GOINFER_CUDA_FLASH_DECODE", c.value)
			} else {
				// t.Setenv registers the restore; Unsetenv then makes the variable genuinely absent.
				t.Setenv("GOINFER_CUDA_FLASH_DECODE", "x")
				unsetenv(t, "GOINFER_CUDA_FLASH_DECODE")
			}
			if got := flashDecodeSplit(); got != c.want {
				t.Fatalf("flashDecodeSplit() with %q (set=%v) = %d, want %d", c.value, c.set, got, c.want)
			}
		})
	}
	if flashDecodeDefaultSplit != 16 {
		t.Fatalf("flashDecodeDefaultSplit = %d; the fidelity gate was pre-registered and passed at S=16 — changing the default S needs its own gate", flashDecodeDefaultSplit)
	}
}

// captureStderr runs f with os.Stderr redirected and returns what it wrote.
func captureStderr(t *testing.T, f func()) string {
	t.Helper()
	old := os.Stderr
	rd, wr, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stderr = wr
	done := make(chan string)
	go func() {
		b, _ := io.ReadAll(rd)
		done <- string(b)
	}()
	func() {
		defer func() { os.Stderr = old }()
		f()
	}()
	wr.Close()
	return <-done
}

// TestLoadFlashDecode_defaultSkipsExpertCache pins the one scoping the default carries: a C′ expert-cache model is sized to the byte
// from free VRAM, so the lane's partial buffer must not come out of that budget unless the variable was set explicitly.
//
// It needs neither a model nor a device, and that is what makes it easy to write VACUOUSLY: with the guard deleted the function still
// returns early further down (nLayers=0, "no positive head dim"), leaving faSplit at 0 either way. The two returns differ in one
// observable way — the guard is silent, the later one prints — so the assertion is on stderr, and the explicit-variable case below pins
// that the override really does get past the guard.
func TestLoadFlashDecode_defaultSkipsExpertCache(t *testing.T) {
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "x")
	unsetenv(t, "GOINFER_CUDA_FLASH_DECODE")
	r := &cudaResident{cacheExperts: true}
	out := captureStderr(t, func() { r.loadFlashDecode(nil, 0) })
	if r.faSplit != 0 {
		t.Fatalf("default-on loaded the lane on a C′ expert-cache model (faSplit=%d); its buffer would come out of the cache's VRAM budget", r.faSplit)
	}
	if out != "" {
		t.Fatalf("the default on a C′ model must return at the guard, silently; got past it (stderr %q) — the guard is not what stopped it", out)
	}

	// The explicit variable must still turn it on for that model: it gets past the guard, and here (no layers) stops at the next
	// check, which is what prints.
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "16")
	r = &cudaResident{cacheExperts: true}
	out = captureStderr(t, func() { r.loadFlashDecode(nil, 0) })
	if !strings.Contains(out, "no positive head dim") {
		t.Fatalf("an explicit GOINFER_CUDA_FLASH_DECODE=16 on a C′ model did not get past the default's guard (stderr %q)", out)
	}
}
