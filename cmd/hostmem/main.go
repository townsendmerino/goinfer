// Command hostmem prints this machine's available memory as goinfer's fit guard reads it (internal/hostmem), for the start checks
// of scripts/night.py, scripts/bench_peer.py and the job runners, so every reader uses one number.
//
//	hostmem       available bytes
//	hostmem -mb   available MiB, rounded down
//	hostmem -v    the breakdown and the formula
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
	flag.Parse()
	r, ok := hostmem.Read()
	if !ok {
		fmt.Fprintln(os.Stderr, "hostmem: no reading on this platform")
		os.Exit(1)
	}
	switch {
	case *v:
		g := func(b int64) float64 { return float64(b) / (1 << 30) }
		fmt.Printf("physical %.2f GiB, app %.2f, wired %.2f, compressed %.2f, available %.2f GiB (%s)\n",
			g(r.Physical), g(r.App), g(r.Wired), g(r.Compressed), g(r.Available), r.Source)
	case *mb:
		fmt.Println(r.Available >> 20)
	default:
		fmt.Println(r.Available)
	}
}
