// Command hostmem prints this machine's available memory as goinfer's fit guard reads it (internal/hostmem), for the start checks
// of scripts/night.py, scripts/bench_peer.py and the job runners, so every reader uses one number.
//
//	hostmem          available bytes
//	hostmem -mb      available MiB, rounded down
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

	"github.com/townsendmerino/goinfer/internal/hostmem"
)

func main() {
	mb := flag.Bool("mb", false, "print MiB instead of bytes")
	v := flag.Bool("v", false, "print the breakdown")
	pid := flag.Int("pid", 0, "print this process's memory instead")
	oracle := flag.Bool("oracle", false, "print the reading with the kernel's jetsam figure before and after it")
	flag.Parse()
	if *oracle {
		o, ok := hostmem.ReadWithOracle()
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
		t, ok := hostmem.ReadTask(*pid)
		if !ok {
			fmt.Fprintf(os.Stderr, "hostmem: no task_info for pid %d\n", *pid)
			os.Exit(1)
		}
		fmt.Printf("resident %d internal %d compressed %d compressed_peak %d phys_footprint %d\n", t.Resident, t.Internal, t.Compressed,
			t.CompressedPeak, t.PhysFootprint)
		return
	}
	r, ok := hostmem.Read()
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
	case *mb:
		fmt.Println(r.Available >> 20)
	default:
		fmt.Println(r.Available)
	}
}
