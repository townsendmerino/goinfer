//go:build gpu && goinfer_testhooks

package gpu

// GPUEverAvailable reports whether any Context was successfully created in this process.
//
// Exists for tests in the EXTERNAL gpu_test package, which cannot see package state but need
// the same distinction the internal helper makes: a device request that fails because the
// machine has no GPU is a skip, while one that fails after a device already worked means the
// process exhausted it — and skipping there silently turns a correctness gate into a no-op.
// Behind goinfer_testhooks so it is not part of the shipped surface.
func GPUEverAvailable() bool { return gpuEverAvailable.Load() }
