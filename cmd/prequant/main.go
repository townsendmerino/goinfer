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
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/townsendmerino/goinfer/internal/prequant"
)

func main() {
	out := flag.String("o", "", "output .giw bundle path (required)")
	quant := flag.String("quant", "int8int8", "weight quant baked into the bundle: int8int8 | int8 | int4")
	embedInt4 := flag.Bool("embed-int4", false, "in int4 mode, quantize the token-embedding/LM-head to int4 too (else int8-pinned); ~½ the head's per-token traffic, coherence-safe on big-vocab models (verified on gemma4-26b: trigram 0.911, Paris survives)")
	row4 := flag.Bool("row4", false, "in int4 mode, ALSO bake the arm64 split-half + 4-row-interleaved layout onto disk (weightMat kind 4) for every eligible tensor, so the paged-MoE path can use the faster W4A8 kernel without an in-RAM repack (docs/task-w4a8-neon-bandwidth.md's \"Format follow-on\"). Requires building/running on an arm64 box; a shape the repack rejects (or a non-arm64 build) falls back to kind 3 for that tensor automatically. Bit-identical dispatch either way — this changes the bundle's size and load characteristics, never decode output. Roughly doubles the bytes of each eligible int4 tensor on disk")
	flag.Parse()
	in := flag.Arg(0)
	if in == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "usage: prequant -o <out.giw> <model.gguf | safetensors-dir>")
		flag.Usage()
		os.Exit(2)
	}
	// Ctrl-C / SIGTERM aborts the (minutes-long, tens-of-GB) transcode and removes the
	// partial .giw, instead of running to completion after the user gives up (audit M-21).
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := prequant.Transcode(ctx, in, *out, *quant, *embedInt4, *row4); err != nil {
		fmt.Fprintf(os.Stderr, "prequant: %v\n", err)
		os.Exit(1)
	}
	if fi, err := os.Stat(*out); err == nil {
		fmt.Fprintf(os.Stderr, "wrote %s: %.0f MB (%s)\n", *out, float64(fi.Size())/1048576, *quant)
	}
}
