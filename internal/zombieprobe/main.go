// Command zombieprobe prints the x86 features this host's CPUID and XCR0 report, or with "segv" takes recovered nil-pointer
// faults while the heap churns and the GC runs (temporary, zombie-probe branch only).
package main

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	"golang.org/x/sys/cpu"
)

type node struct {
	next *node
	v    int
}

type big struct {
	pad [64]byte
	x   int
}

//go:noinline
func deref(p *big) int { return p.x }

func fault() (n int) {
	defer func() {
		if recover() != nil {
			n = -1
		}
	}()
	return deref(nil)
}

func main() {
	if len(os.Args) < 2 || os.Args[1] != "segv" {
		fmt.Printf("%+v\n", cpu.X86)
		return
	}
	var wg sync.WaitGroup
	for g := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var keep []*node
			for i := range 300000 {
				if g == 0 && fault() != -1 {
					panic("no fault")
				}
				n := &node{v: i}
				if len(keep) > 0 {
					n.next = keep[len(keep)-1]
				}
				if i%3 == 0 {
					keep = append(keep, n)
				}
				if len(keep) > 2000 {
					keep = keep[:0]
				}
				if g == 1 && i%20000 == 0 {
					runtime.GC()
				}
			}
		}()
	}
	wg.Wait()
	fmt.Println("segv probe: done, no fault")
}
