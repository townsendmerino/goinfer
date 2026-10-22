// cpuload.go: D8A-G4 (docs/tasks/task-memory-campaign-2026-10.md, §4 MEM13, D8a-build gates). One direct heap load with
// Backend "cpu" at int4, then the Go heap in use after runtime.GC + debug.FreeOSMemory, and 16 greedy tokens.
//
//	go run docs/measurements/d8a-2026-10/testdata/cpuload.go <gguf>
package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"

	"github.com/townsendmerino/goinfer/decoder"
)

func main() {
	m, err := decoder.Load(os.Args[1], decoder.Options{Backend: "cpu", Quant: "int4", DirectLoad: true})
	if err != nil {
		fmt.Println("load error:", err)
		os.Exit(1)
	}
	runtime.GC()
	debug.FreeOSMemory()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	fmt.Printf("heap_inuse=%.1f MiB heap_alloc=%.1f MiB\n", float64(ms.HeapInuse)/(1<<20), float64(ms.HeapAlloc)/(1<<20))
	out, gen := m.Generate(context.Background(), []int{151644, 872, 198, 7985, 264, 729, 304, 13027, 13, 151645, 198, 151644, 77091, 198}, 16, decoder.SamplingParams{})
	var ids []int
	for id := range out {
		ids = append(ids, id)
	}
	if err := gen.Err(); err != nil {
		fmt.Println("generate error:", err)
		os.Exit(1)
	}
	fmt.Println("tokens", ids)
	runtime.KeepAlive(m)
}
