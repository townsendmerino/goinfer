// Command hostmem prints this machine's memory as internal/mem/probe reads it, so every reader uses one number: the available
// figure the fit guard decides loads by, or with -reclaimable the figure the timed runs' start checks (scripts/night.py,
// scripts/bench_peer.py) and the void rule decide validity by.
//
//	hostmem          available bytes
//	hostmem -mb      available MiB, rounded down
//	hostmem -reclaimable [-mb]  reclaimable bytes (MiB), what can be taken without compressing or swapping
//	hostmem -v       the breakdown and the formula
//	hostmem -pid N   process N's resident, internal, compressed and footprint bytes (task_info TASK_VM_INFO), one line
//	hostmem -oracle  the reading bracketed by the kernel's jetsam figure read before and after it, one line (B41 gate 1a)
//
// It exits 1 when this platform has no reading.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/internal/mem/probe"
)

func main() {
	mb := flag.Bool("mb", false, "print MiB instead of bytes")
	v := flag.Bool("v", false, "print the breakdown")
	recl := flag.Bool("reclaimable", false, "print the reclaimable figure (timed runs' validity) instead of the available one")
	pid := flag.Int("pid", 0, "print this process's memory instead")
	oracle := flag.Bool("oracle", false, "print the reading with the kernel's jetsam figure before and after it")
	flag.Parse()
	if *oracle {
		o, ok := probe.ReadWithOracle()
		if !ok {
			fmt.Fprintln(os.Stderr, "hostmem: no oracle on this platform")
			os.Exit(1)
		}
		r := o.Ours
		fresh := 0
		if o.Fresh {
			fresh = 1
		}
		fmt.Printf("ours %d jetsam_before %d jetsam_after %d free %d speculative %d file_backed %d purgeable %d physical %d fresh %d available %d level %d\n",
			r.Reclaimable, o.JetsamBefore, o.JetsamAfter, r.Free, r.Speculative, r.FileBacked, r.Purgeable, r.Physical, fresh, r.Available, r.Level)
		return
	}
	if *pid > 0 {
		t, ok := probe.ReadTask(*pid)
		if !ok {
			fmt.Fprintf(os.Stderr, "hostmem: no task_info for pid %d\n", *pid)
			os.Exit(1)
		}
		fmt.Printf("resident %d internal %d compressed %d compressed_peak %d phys_footprint %d\n", t.Resident, t.Internal, t.Compressed,
			t.CompressedPeak, t.PhysFootprint)
		return
	}
	r, ok := probe.Read()
	if !ok {
		fmt.Fprintln(os.Stderr, "hostmem: no reading on this platform")
		os.Exit(1)
	}
	switch {
	case *v:
		g := func(b int64) float64 { return float64(b) / (1 << 30) }
		fmt.Printf("physical %.2f GiB, available %.2f (level %d%%); reclaimable %.2f (free %.2f, speculative %.2f, file-backed %.2f, "+
			"purgeable %.2f); app %.2f, wired %.2f, compressed %.2f GiB (%s)\n", g(r.Physical), g(r.Available), r.Level, g(r.Reclaimable),
			g(r.Free), g(r.Speculative), g(r.FileBacked), g(r.Purgeable), g(r.App), g(r.Wired), g(r.Compressed), r.Source)
	default:
		fmt.Println(figure(r, *recl, *mb))
	}
}

// figure is the one number hostmem prints without -v: Reclaimable with -reclaimable, else Available; in MiB, rounded down, with -mb.
func figure(r probe.Reading, reclaimable, mb bool) int64 {
	v := r.Available
	if reclaimable {
		v = r.Reclaimable
	}
	if mb {
		return v >> 20
	}
	return v
}
