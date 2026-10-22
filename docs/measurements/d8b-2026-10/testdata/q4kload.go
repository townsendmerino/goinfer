// q4kload.go: D8B-G4 (docs/tasks/task-memory-campaign-2026-10.md, §4 MEM13, D8b-build gates). One load at --quant q4k on
// the CPU, direct (arg "direct") or through the sidecar (arg "sidecar", Load's default), then the Go heap in use after
// runtime.GC + debug.FreeOSMemory and 16 greedy tokens.
//
//	go run docs/measurements/d8b-2026-10/testdata/q4kload.go <gguf> direct|sidecar
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
	opts := decoder.Options{Backend: "cpu", Quant: "q4k", DirectLoad: os.Args[2] == "direct"}
	m, err := decoder.Load(os.Args[1], opts)
	if err != nil {
		fmt.Println("load error:", err)
		os.Exit(1)
	}
	runtime.GC()
	debug.FreeOSMemory()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	fmt.Printf("%s: quant=%s heap_inuse=%.1f MiB heap_alloc=%.1f MiB\n", os.Args[2], m.Quant(), float64(ms.HeapInuse)/(1<<20), float64(ms.HeapAlloc)/(1<<20))
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
