// Command prequant builds a goinfer prequant bundle (.giw) from a GGUF model.
//
// It loads the model at a fixed quant (default int8int8), serializes the
// already-quantized resident weights, and pairs them with a metadata-only GGUF
// (the source truncated at the tensor-data boundary) that carries the tokenizer.
// The demo's -tags prequant build embeds the bundle and loads it with NO
// dequant/requant — the int8 weights are aliased straight from the binary image,
// so a 4B model no longer needs a multi-GB heap copy on every launch. serve also
// builds these on the fly for --stream-weights (see internal/prequant.EnsureCachedGIW).
//
// Usage:
//
//	go run ./cmd/prequant -o demo/chat/model.giw ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/internal/prequant"
)

func main() {
	out := flag.String("o", "", "output .giw bundle path (required)")
	quant := flag.String("quant", "int8int8", "weight quant baked into the bundle: int8int8 | int8 | int4")
	flag.Parse()
	in := flag.Arg(0)
	if in == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "usage: prequant -o <out.giw> <model.gguf | safetensors-dir>")
		flag.Usage()
		os.Exit(2)
	}
	if err := prequant.Transcode(in, *out, *quant); err != nil {
		fmt.Fprintf(os.Stderr, "prequant: %v\n", err)
		os.Exit(1)
	}
	if fi, err := os.Stat(*out); err == nil {
		fmt.Fprintf(os.Stderr, "wrote %s: %.0f MB (%s)\n", *out, float64(fi.Size())/1048576, *quant)
	}
}
