// libload.go: MEM0's "library path before" (docs/tasks/task-memory-campaign-2026-10.md, MEM1): what a caller in another module gets from
// decoder.Load on a .gguf, with no sidecar. One load per process, then the process's own memory: RssAnon / RssFile / VmHWM from
// /proc/self/status and the Go heap (runtime.MemStats) after the load, and after runtime.GC + debug.FreeOSMemory.
//
//	go run docs/measurements/mem0-2026-10/testdata/libload.go <gguf> [quant]      (quant: "" for Options{}, or e.g. int4)
package main

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
)

func status() map[string]int64 {
	m := map[string]int64{}
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return m
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		k, v, _ := strings.Cut(sc.Text(), ":")
		switch k {
		case "VmRSS", "RssAnon", "RssFile", "VmHWM", "VmSwap":
			n, _ := strconv.ParseInt(strings.Fields(v)[0], 10, 64)
			m[k] = n >> 10 // MiB
		}
	}
	return m
}

func report(tag string) {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	fmt.Printf("%-18s %v  heap_inuse=%d MiB heap_alloc=%d MiB sys=%d MiB\n", tag, status(), ms.HeapInuse>>20, ms.HeapAlloc>>20, ms.Sys>>20)
}

func main() {
	path := os.Args[1]
	opts := decoder.Options{}
	if len(os.Args) > 2 && os.Args[2] != "" {
		opts.Quant = os.Args[2]
	}
	fmt.Printf("# %s, Options{Quant:%q}\n", path, opts.Quant)
	report("before load")
	m, err := decoder.Load(path, opts)
	if err != nil {
		fmt.Println("load error:", err)
		os.Exit(1)
	}
	report("after load")
	runtime.GC()
	debug.FreeOSMemory()
	report("after GC+free")
	runtime.KeepAlive(m)
}
