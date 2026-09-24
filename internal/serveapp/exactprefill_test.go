package serveapp

import (
	"os"
	"testing"
)

// unsetenvT unsets name for the duration of the test, restoring whatever value (or absence) it
// had beforehand once the test finishes.
func unsetenvT(t *testing.T, name string) {
	t.Helper()
	if v, ok := os.LookupEnv(name); ok {
		t.Cleanup(func() { os.Setenv(name, v) })
	} else {
		t.Cleanup(func() { os.Unsetenv(name) })
	}
	os.Unsetenv(name)
}

// TestApplyExactPrefillEnv_coversAllThreeBackends gates M-48: --exact-prefill's help promised
// "ALL backends" but only ever set the CPU and Metal env vars — CUDA's own fast-prefill knob
// (GOINFER_CUDA_FAST_PREFILL, cuda/prefill.go) was never touched, so a user who asked for
// bit-exact prompt ingestion on a CUDA server silently kept the fast, non-exact path. This is
// also the first test of applyExactPrefillEnv at all — the whole block (including the
// pre-existing Metal/CPU wiring) had zero coverage before this, since Main() itself isn't
// unit-testable (os.Args, os.Exit, starts a server).
func TestApplyExactPrefillEnv_coversAllThreeBackends(t *testing.T) {
	unsetenvT(t, "GOINFER_METAL_FAST_PREFILL")
	unsetenvT(t, "GOINFER_CUDA_FAST_PREFILL")
	unsetenvT(t, "GOINFER_CPU_FAST_ATTENTION")

	applyExactPrefillEnv(config{exactPrefill: true, cpuFastAttention: true})

	for _, want := range []struct{ name, value string }{
		{"GOINFER_METAL_FAST_PREFILL", "0"},
		{"GOINFER_CUDA_FAST_PREFILL", "0"},
		{"GOINFER_CPU_FAST_ATTENTION", "0"},
	} {
		if got := os.Getenv(want.name); got != want.value {
			t.Errorf("--exact-prefill: %s = %q, want %q", want.name, got, want.value)
		}
	}
}

// TestApplyExactPrefillEnv_defaultLeavesMetalAndCUDAUnset confirms the negative: without
// --exact-prefill, Metal and CUDA's own knobs are left alone (unset means "default on" to both
// backends' own env-reading code) — only CPU's var is set explicitly either way, per its own
// comment ("the server's own flags must win over whatever the shell happened to export").
func TestApplyExactPrefillEnv_defaultLeavesMetalAndCUDAUnset(t *testing.T) {
	unsetenvT(t, "GOINFER_METAL_FAST_PREFILL")
	unsetenvT(t, "GOINFER_CUDA_FAST_PREFILL")
	unsetenvT(t, "GOINFER_CPU_FAST_ATTENTION")

	applyExactPrefillEnv(config{cpuFastAttention: true})

	if v, ok := os.LookupEnv("GOINFER_METAL_FAST_PREFILL"); ok {
		t.Errorf("GOINFER_METAL_FAST_PREFILL = %q, want unset", v)
	}
	if v, ok := os.LookupEnv("GOINFER_CUDA_FAST_PREFILL"); ok {
		t.Errorf("GOINFER_CUDA_FAST_PREFILL = %q, want unset", v)
	}
	if got := os.Getenv("GOINFER_CPU_FAST_ATTENTION"); got != "1" {
		t.Errorf("GOINFER_CPU_FAST_ATTENTION = %q, want %q", got, "1")
	}
}

// TestApplyMoEPagerEnv_setsExplicitlyEitherWay gates task-never-swap-2026-09.md S5's Build item
// 1 — the --moe-pager flag exposes decoder/moepaging.go's existing GOINFER_MOE_PREAD_CPU switch,
// set explicitly either way (not left unset on "mmap") for the same reason applyExactPrefillEnv
// does: the flag must win over whatever the shell happened to export.
func TestApplyMoEPagerEnv_setsExplicitlyEitherWay(t *testing.T) {
	unsetenvT(t, "GOINFER_MOE_PREAD_CPU")

	applyMoEPagerEnv(config{moePager: "pool"})
	if got := os.Getenv("GOINFER_MOE_PREAD_CPU"); got != "1" {
		t.Errorf("--moe-pager=pool: GOINFER_MOE_PREAD_CPU = %q, want %q", got, "1")
	}

	applyMoEPagerEnv(config{moePager: "mmap"})
	if got := os.Getenv("GOINFER_MOE_PREAD_CPU"); got != "0" {
		t.Errorf("--moe-pager=mmap: GOINFER_MOE_PREAD_CPU = %q, want %q", got, "0")
	}
}

// TestMoEPagerDefault gates S5's registered default: pool on darwin (where MADV_DONTNEED is a
// no-op, so mmap mode cannot enforce its budget), mmap everywhere else.
func TestMoEPagerDefault(t *testing.T) {
	for goos, want := range map[string]string{"darwin": "pool", "linux": "mmap", "windows": "mmap", "freebsd": "mmap"} {
		if got := moePagerDefault(goos); got != want {
			t.Errorf("moePagerDefault(%q) = %q, want %q", goos, got, want)
		}
	}
}
