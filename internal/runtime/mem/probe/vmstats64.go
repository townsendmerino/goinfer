package probe

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

// fromVMStats64 adds up, as Reclaimable, what the kernel can reclaim without compressing or swapping: free, file-backed (external) and
// purgeable pages
// (B41, docs/measurements/hostmem-available-2026-10/README.md, amendment A5). host_statistics64's free count includes the speculative
// (read-ahead) pages, and so does its external count, so Free is the free count less speculative (vm_stat's "Pages free") and each
// speculative page is counted once, as file-backed. Adding these, rather than subtracting what is used from physical, leaves out the
// memory no page counter covers. App memory (internal less purgeable), wired and compressed pages are Activity Monitor's Memory Used
// terms, reported as context.
func fromVMStats64(s vmStats64, pageSize, physical int64) Reading {
	pg := func(n uint32) int64 { return int64(n) * pageSize }
	app := pg(s.internal) - pg(s.purgeable)
	if app < 0 {
		app = 0
	}
	free := pg(s.free) - pg(s.speculative)
	if free < 0 {
		free = 0
	}
	return Reading{Physical: physical, Reclaimable: free + pg(s.external) + pg(s.purgeable),
		Free: free, Speculative: pg(s.speculative), FileBacked: pg(s.external), Purgeable: pg(s.purgeable),
		App: app, Wired: pg(s.wire), Compressed: pg(s.compressor)}
}

// fromLevel is the macOS Available: level percent of physical, less PressureBuffer, and never below 0.
func fromLevel(level int, physical int64) int64 {
	return max(physical*int64(level)/100-PressureBuffer, 0)
}

// decodeTaskVM decodes the leading fields of struct task_vm_info (mach/task_info.h) that TaskVM reports.
func decodeTaskVM(b []byte) TaskVM {
	u := func(off int) int64 { return int64(binary.LittleEndian.Uint64(b[off:])) }
	return TaskVM{Resident: u(16), Internal: u(48), Compressed: u(120), CompressedPeak: u(128), PhysFootprint: u(144)}
}
