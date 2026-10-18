// fitdescribe.go: what the host fit guard prices for a load (decoder.FitDescribe), for the cases cuda-build-heap-2026-10-18 measures.
//
//	go run docs/measurements/cuda-build-heap-2026-10-18/testdata/fitdescribe.go <gguf> <quant> <ctx>
package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/townsendmerino/goinfer/decoder"
)

func main() {
	ctx, _ := strconv.Atoi(os.Args[3])
	d, err := decoder.FitDescribe(os.Args[1], decoder.Options{Backend: "cuda", Quant: os.Args[2], ResidentContext: ctx})
	fmt.Printf("%s %s ctx %d:\n  %s\n  err=%v\n", os.Args[1], os.Args[2], ctx, d, err)
}
