package decoder

import (
	"fmt"
	"os"
	"path/filepath"
)

// What a .giw load does about the page cache before it walks the bundle's matrix headers. The walk touches two or
// three pages of every matrix through the mapping, and the kernel reads around each fault, so on a bundle of thousands
// of small matrices (an MoE's experts) the walk is a scattered read of nearly the whole file. The measurements and the
// gates are in docs/tasks/task-giw-cold-load-2026-10.md.
type coldLoadAction int

const (
	// coldLoadNothing leaves the walk as it is: a dense bundle, a file already in the page cache, or a case no rule
	// below covers.
	coldLoadNothing coldLoadAction = iota
	// coldLoadReadFirst reads the mapping front to back before the walk. The weights are read either way; this reads
	// them at the disk's sequential rate.
	coldLoadReadFirst
	// coldLoadNoReadAround turns the kernel's read-around off for the walk and back on after it, so the walk reads
	// the headers and not the file: for a bundle that cannot stay in memory.
	coldLoadNoReadAround
)

// coldLoadFacts is what planColdLoad decides from.
type coldLoadFacts struct {
	smallMatrices bool    // the bundle is mostly small matrices, so its walk would read most of it (expertMatricesAreSmall)
	residentFrac  float64 // the sampled share of the file already in the page cache, 0..1
	fileBytes     int64
	availBytes    int64 // memory the kernel reports available; 0 = unknown
	ramBytes      int64 // installed memory; 0 = unknown
	paged         bool  // the load pages the weights against a budget (Options.StreamWeights)
}

// coldResidentWarm is the sampled share of a file in the page cache at and above which the load treats it as warm.
const coldResidentWarm = 0.5

// planColdLoad picks the action. A bundle is read first only when it fits with room to spare (three quarters of the
// available memory), and its read-around is turned off only when it certainly cannot stay (paged against a budget, or
// larger than the installed memory). Between the two, and whenever a size is unknown, the load is left as it is: with
// the read-around off, a bundle that does fit reaches its first reply later than with nothing done.
func planColdLoad(f coldLoadFacts) coldLoadAction {
	if !f.smallMatrices || f.residentFrac >= coldResidentWarm || f.fileBytes <= 0 {
		return coldLoadNothing
	}
	if f.paged || (f.ramBytes > 0 && f.fileBytes > f.ramBytes) {
		return coldLoadNoReadAround
	}
	if f.availBytes > 0 && f.fileBytes <= f.availBytes/4*3 {
		return coldLoadReadFirst
	}
	return coldLoadNothing
}

// coldSmallMatrixBytes is the average expert-matrix size at and below which the walk's read-around covers the whole
// matrix, so the walk reads the file.
const coldSmallMatrixBytes = 4 << 20

// expertMatricesAreSmall reports whether a bundle of fileBytes for arch stores its experts as separate matrices that
// average at most coldSmallMatrixBytes. A dense model, and an MoE with a few large experts, walk cheaply and are left
// alone.
func expertMatricesAreSmall(arch *Architecture, fileBytes int64) bool {
	if arch == nil || arch.MoE == nil || arch.MoE.NumExperts <= 0 || arch.NumLayers <= 0 {
		return false
	}
	matrices := int64(arch.NumLayers) * int64(arch.MoE.NumExperts) * 3 // gate, up, down
	return fileBytes/matrices <= coldSmallMatrixBytes
}

// The probes and actions prepareColdLoad uses, as variables so a test can drive Load through each plan.
var (
	coldResidentFraction = giwResidentFraction
	coldAvailBytes       = HostRAMAvailableBytes
	coldRAMBytes         = HostRAMBytes
	coldReadThrough      = giwReadThrough
	coldAdviseReadAround = giwAdviseReadAround
)

// prepareColdLoad runs the plan for the .giw at path, mapped as mapping, whose header resolved to arch. It is called
// once the header is read and before any matrix is. The returned function undoes what must be undone after the walk.
func prepareColdLoad(path string, mapping []byte, arch *Architecture, opts Options) (after func()) {
	after = func() {}
	if !coldLoadSupported || !expertMatricesAreSmall(arch, int64(len(mapping))) {
		return after
	}
	switch planColdLoad(coldLoadFacts{
		smallMatrices: true,
		residentFrac:  coldResidentFraction(mapping),
		fileBytes:     int64(len(mapping)),
		availBytes:    coldAvailBytes(),
		ramBytes:      coldRAMBytes(),
		paged:         opts.StreamWeights,
	}) {
	case coldLoadReadFirst:
		fmt.Fprintf(os.Stderr, "decoder: %s is not in memory: reading its %.1f GB once before the load\n",
			filepath.Base(path), float64(len(mapping))/1e9)
		coldReadThrough(mapping)
	case coldLoadNoReadAround:
		coldAdviseReadAround(mapping, false)
		after = func() { coldAdviseReadAround(mapping, true) }
	}
	return after
}
