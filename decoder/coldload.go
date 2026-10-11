package decoder

import (
	"fmt"
	"os"
	"path/filepath"
)

// What a .giw load does about the page cache before it walks the bundle's matrix headers. Two systems, two reasons
// (docs/tasks/task-giw-cold-load-2026-10.md has the measurements and the gates):
//
//   - Linux reads megabytes around every page fault, so on a bundle of thousands of small matrices (an MoE's experts)
//     the walk is a scattered read of nearly the whole file.
//   - macOS reads little around a fault, so the walk is cheap, and the first decode then faults the weights in a page
//     at a time.
//
// In both, a bundle that will be read anyway is better read once, in order, before the walk.
type coldLoadAction int

const (
	// coldLoadNothing leaves the walk as it is: a dense bundle, a file already in the page cache, or a case no rule
	// below covers.
	coldLoadNothing coldLoadAction = iota
	// coldLoadReadFirst brings the whole mapping into the page cache before the walk (giwReadThrough). The weights
	// are read either way; this reads them at the disk's sequential rate.
	coldLoadReadFirst
	// coldLoadNoReadAround turns the kernel's read-around off for the walk and back on after it, so the walk reads
	// the headers and not the file: for a bundle that cannot stay in memory, on a system whose read-around is large.
	coldLoadNoReadAround
)

// coldLoadPolicy is one system's rule.
type coldLoadPolicy struct {
	supported      bool // false: the load does nothing about the page cache
	anyBundle      bool // every bundle qualifies; false: only a small-matrix MoE (expertMatricesAreSmall)
	dropReadAround bool // the walk reads around its faults enough for coldLoadNoReadAround to matter
}

var (
	// coldPolicyLinux: only the bundle whose walk would read it anyway, and the read-around is worth turning off for
	// one that cannot stay.
	coldPolicyLinux = coldLoadPolicy{supported: true, anyBundle: false, dropReadAround: true}
	// coldPolicyDarwin: any bundle, because it is the decode's page-at-a-time faults that a read-first saves; the
	// walk's read-around is small and is left alone.
	coldPolicyDarwin = coldLoadPolicy{supported: true, anyBundle: true, dropReadAround: false}
)

// qualifies reports whether the policy reads a bundle of fileBytes for arch first when it is cold.
func (p coldLoadPolicy) qualifies(arch *Architecture, fileBytes int64) bool {
	return p.supported && (p.anyBundle || expertMatricesAreSmall(arch, fileBytes))
}

// coldLoadFacts is what planColdLoad decides from.
type coldLoadFacts struct {
	qualifies      bool    // the system's policy reads this bundle first when it is cold (coldLoadPolicy.qualifies)
	dropReadAround bool    // the system's policy turns the read-around off for a bundle that cannot stay
	residentFrac   float64 // the sampled share of the file already in the page cache, 0..1
	fileBytes      int64
	availBytes     int64 // memory the kernel reports available; 0 = unknown
	ramBytes       int64 // installed memory; 0 = unknown
	paged          bool  // the load pages the weights against a budget (Options.StreamWeights)
}

// coldResidentWarm is the sampled share of a file in the page cache at and above which the load treats it as warm.
const coldResidentWarm = 0.5

// planColdLoad picks the action. A bundle is read first only when it fits with room to spare (three quarters of the
// available memory). One that certainly cannot stay (paged against a budget, or larger than the installed memory) is
// never read first, and has its read-around turned off where the policy says that matters. Between the two, and
// whenever a size is unknown, the load is left as it is: with the read-around off, a bundle that does fit reaches its
// first reply later than with nothing done.
func planColdLoad(f coldLoadFacts) coldLoadAction {
	if !f.qualifies || f.residentFrac >= coldResidentWarm || f.fileBytes <= 0 {
		return coldLoadNothing
	}
	if f.paged || (f.ramBytes > 0 && f.fileBytes > f.ramBytes) {
		if f.dropReadAround {
			return coldLoadNoReadAround
		}
		return coldLoadNothing
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

// The policy, probes and actions prepareColdLoad uses, as variables so a test can drive Load through each plan under
// either system's policy.
var (
	coldPolicy           = coldPolicyHere
	coldResidentFraction = giwResidentFraction
	coldAvailBytes       = HostRAMAvailableBytes
	coldRAMBytes         = HostRAMBytes
	coldReadThrough      = giwReadThrough
	coldAdviseReadAround = giwAdviseReadAround
)

// prepareColdLoad runs the plan for the .giw at path, mapped as mapping, whose header resolved to arch. It is called
// once the header is read and before any matrix is. The returned function undoes what must be undone after the walk.
// A bundle the policy does not cover, and one already in the page cache, cost a residency sample at most: the memory
// probes run only for a cold bundle.
func prepareColdLoad(path string, mapping []byte, arch *Architecture, opts Options) (after func()) {
	after = func() {}
	if !coldPolicy.qualifies(arch, int64(len(mapping))) {
		return after
	}
	resident := coldResidentFraction(mapping)
	if resident >= coldResidentWarm {
		return after
	}
	switch planColdLoad(coldLoadFacts{
		qualifies:      true,
		dropReadAround: coldPolicy.dropReadAround,
		residentFrac:   resident,
		fileBytes:      int64(len(mapping)),
		availBytes:     coldAvailBytes(),
		ramBytes:       coldRAMBytes(),
		paged:          opts.StreamWeights,
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
