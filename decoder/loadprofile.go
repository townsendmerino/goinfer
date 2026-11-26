package decoder

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

// Load-time instrumentation: how long a model load takes and where the time goes.
//
// The split is the point, not the total. A slow load caused by storage and one caused by repacking call for different
// responses (a faster disk versus a different quant), and one number cannot tell them apart. The phases:
//
// 	map       the file becoming addressable: open + mmap + header parse. Not the file read.
// 	build     tensors becoming resident weights: dequantize, quantize, repack. CPU-bound.
// 	resident  weights becoming a device-side runner, where a backend builds one. PCIe/GPU-bound.
//
// `map` does not isolate storage. The loader mmaps, so pages fault in lazily during `build`, not `map`, and `map` tracks
// metadata count rather than file size. The storage cost lands inside `build`, visible as the cold-minus-warm delta
// rather than as its own phase; isolating it would need a non-mmap read path or per-phase fault accounting.
//
// The profile cannot tell whether the page cache was cold: a warm-cache `map` measures memcpy, a cold one the disk, and
// re-running a benchmark gives a warm one by accident. It reports bytes and a rate so the regime is visible in the
// number; the cold/warm label belongs in docs/benchmarks.md next to the storage class. The measurements that showed
// this: docs/code-notes/decoder.md#loadprofile.

// LoadProfile returns how long each phase of this model's load took, or nil if the path that
// produced it was not instrumented (safetensors and .giw are not, today). Every method on the
// result is nil-safe, so a caller can print Summary() unconditionally.
func (m *Model) LoadProfile() *darch.LoadProfile {
	if m == nil {
		return nil
	}
	return m.prof
}
