package hostmem

import "encoding/binary"

// vmStats64Size is sizeof(struct vm_statistics64) in mach/vm_statistics.h, HOST_VM_INFO64_COUNT (38) integer_t words.
const vmStats64Size = 152

// vmStats64 is the part of struct vm_statistics64 the formula reads, in pages.
type vmStats64 struct {
	free, active, inactive, wire, purgeable, speculative, compressor, external, internal uint32
}

// decodeVMStats64 decodes the struct host_statistics64(HOST_VM_INFO64) fills, little-endian as on every Mac: four natural_t
// counts, nine uint64s, purgeable and speculative, four uint64s, then compressor, throttled, external and internal page counts.
func decodeVMStats64(b []byte) (vmStats64, bool) {
	if len(b) != vmStats64Size {
		return vmStats64{}, false
	}
	u := func(off int) uint32 { return binary.LittleEndian.Uint32(b[off:]) }
	return vmStats64{free: u(0), active: u(4), inactive: u(8), wire: u(12), purgeable: u(88), speculative: u(92),
		compressor: u(128), external: u(136), internal: u(140)}, true
}

// fromVMStats64 is Activity Monitor's Memory split: Memory Used = App Memory + Wired Memory + Compressed, where App Memory is the
// internal (anonymous) pages less the purgeable ones, and what remains of physical memory (cached files and free pages) is
// available. Cached files count as available because the kernel reclaims them before it compresses or swaps anything.
func fromVMStats64(s vmStats64, pageSize, physical int64) Reading {
	app := (int64(s.internal) - int64(s.purgeable)) * pageSize
	if app < 0 {
		app = 0
	}
	wired, compressed := int64(s.wire)*pageSize, int64(s.compressor)*pageSize
	avail := physical - (app + wired + compressed)
	if avail < 0 {
		avail = 0
	}
	return Reading{Physical: physical, App: app, Wired: wired, Compressed: compressed, Available: avail,
		Source: "macOS host_statistics64: physical - (app + wired + compressed), Activity Monitor's split"}
}
