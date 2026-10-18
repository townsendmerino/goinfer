// fitdescribe.go: what the host fit guard prices for a CUDA-resident load, with the CUDA backend compiled in (cuda-stream-pack-2026-10-18).
//
//	cd cuda && CGO_ENABLED=0 go run -tags cuda ../docs/measurements/cuda-stream-pack-2026-10-18/testdata/fitdescribe.go <gguf> <quant> <ctx>
package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/townsendmerino/goinfer/decoder"
	_ "github.com/townsendmerino/goinfer/cuda"
)

func main() {
	ctx, _ := strconv.Atoi(os.Args[3])
	d, err := decoder.FitDescribe(os.Args[1], decoder.Options{Backend: "cuda", Quant: os.Args[2], ResidentContext: ctx})
	fmt.Printf("%s %s ctx %d:\n  %s\n  err=%v\n", os.Args[1], os.Args[2], ctx, d, err)
}
